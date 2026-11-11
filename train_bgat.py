"""
mulsim/train_bgat.py
Training script for BGAT benchmark.
Usage:
    cd mulsim && python train_bgat.py --epochs 500 --batch_size 64
Output:
    ../models/bgat_best.pt
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim

from env import (PASSMultiUserEnv, optimize_alpha_multiuser, V, PMOTOR, PF,
                 delta2, SIGMA2, R_MIN)
from bgat_model import BGAT, compute_bgat_loss


def build_alpha(model, data, device='cpu', batch=256):
    """(n, M, N) radiation ratios for a whole dataset, computed in numpy.

    optimize_alpha_multiuser() is a per-sample subset/eigen search (~3 ms), so
    during training it is refreshed only every few epochs and reused in
    between: by the envelope theorem the omitted dA/dX term vanishes at the
    alpha optimum, so a stale A costs a second-order gradient error only.
    """
    M, N, K = data[0]['M'], data[0]['N'], data[0]['K']
    out = np.zeros((len(data), M, N), dtype=np.float32)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for b in range(0, len(data), batch):
            chunk = data[b:b + batch]
            X0 = np.stack([s['X0'] for s in chunk])
            up = np.stack([s['user_pos'] for s in chunk])
            YM = np.stack([s['YM'] for s in chunk])
            X, _ = model.forward_batch(
                X0, up, YM,
                np.array([float(s['Pmax']) for s in chunk]),
                np.array([float(s['D']) for s in chunk]),
                np.array([float(s['Dmax']) for s in chunk]),
                np.array([float(s['Delta_min']) for s in chunk]),
                M, N)
            Xn = X.detach().cpu().numpy()
            for i in range(len(chunk)):
                out[b + i] = optimize_alpha_multiuser(Xn[i], up[i], M, N, K,
                                                      YM[i])
    if was_training:
        model.train()
    return out


def generate_dataset(n_samples=10000, seed=42, fixed_shape=None,
                     report_range=False):
    """
    Generate training samples with randomized parameters.

    fixed_shape=(M, N, K) keeps the graph size constant, which is what lets the
    loss run on whole batches (an order of magnitude faster).  None keeps the
    original per-sample random M/N/K, in which case the loss must be called
    with batch_size = 1.

    report_range=True draws D, Pmax and Tframe from the range the reported
    sweep actually uses (D = 40 m, Pmax = 25 dBm, Tframe in [0.8, 1.3]) instead
    of the wide training range: the sweep is a narrow slice of it, and a
    validation set spread over D in [20, 60] selects a different (less moving)
    policy than the one that is best on the sweep.
    """
    rng = np.random.RandomState(seed)
    samples = []
    for i in range(n_samples):
        if fixed_shape is None:
            M = rng.randint(2, 5)
            N = rng.randint(3, 8)
            K = rng.randint(2, 8)
        else:
            M, N, K = fixed_shape
        if report_range:
            D = 40.0
            Pmax_dbm = 25.0
            Tframe = rng.uniform(0.8, 1.3)
        else:
            D = rng.uniform(20.0, 60.0)
            Pmax_dbm = rng.uniform(15.0, 35.0)
            Tframe = rng.uniform(0.5, 2.0)
        Pmax = 10 ** (Pmax_dbm / 10) * 1e-3

        env = PASSMultiUserEnv(seed=seed + i, randomize_params=False,
                               M=M, N=N, K=K, D=D,
                               Pmax_dbm=Pmax_dbm, Tframe=Tframe)
        env.reset()

        samples.append({
            'X0': env.X0.copy(),
            'user_pos': env.user_pos.copy(),
            'M': M, 'N': N, 'K': K,
            'YM': env.YM.copy(),
            'D': D,
            'Dmax': env.Dmax,
            'Pmax': Pmax,
            'Tframe': Tframe,
            'Rmin': R_MIN,
            'Delta_min': env.Delta_min,
        })
    return samples


def train_bgat(epochs=500, batch_size=64, lr=5e-4, device='cpu',
               model_dir='../models', n_train=20000, n_val=2000,
               fixed_shape=(3, 4, 3), report_range=False, seed=0,
               alpha_every=20, pos_scale=0.05):
    os.makedirs(model_dir, exist_ok=True)
    torch.manual_seed(seed)              # reproducible init + batches

    print("device: %s%s" % (device, '  (' + torch.cuda.get_device_name(0) + ')'
                            if str(device).startswith('cuda')
                            and torch.cuda.is_available() else ''), flush=True)
    print("Generating training dataset (shape %s, report_range=%s)..."
          % (fixed_shape, report_range), flush=True)
    train_data = generate_dataset(n_samples=n_train, seed=42,
                                  fixed_shape=fixed_shape,
                                  report_range=report_range)
    val_data = generate_dataset(n_samples=n_val, seed=12345,
                                fixed_shape=fixed_shape,
                                report_range=report_range)

    model = BGAT(n_blocks=5, feat_dim=16, n_heads=4, mlp_hidden=32).to(device)
    # Initial displacement magnitude (m).  The metric scale of the placement
    # matters: the learned 0.036 m stays in the lambda = 1 cm regime, while the
    # searches that actually gain EE (DSD) move 4-11 cm.
    with torch.no_grad():
        model.pos_scale.fill_(pos_scale)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # One decay point inside the run (the original 200 assumed >=400 epochs).
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=max(20, epochs // 3),
                                          gamma=0.5)

    best_val_loss = float('inf')
    best_epoch = 0
    A_train = A_val = None

    for epoch in range(1, epochs + 1):
        if A_train is None or (epoch - 1) % alpha_every == 0:
            A_train = build_alpha(model, train_data, device)
            A_val = build_alpha(model, val_data, device)
        model.train()
        # Shuffle
        rng = np.random.RandomState(epoch)
        indices = rng.permutation(len(train_data))

        train_losses = []
        train_ees = []
        for b in range(0, len(train_data), batch_size):
            batch_idx = indices[b:b + batch_size]
            batch = [train_data[i] for i in batch_idx]

            loss, avg_ee = compute_bgat_loss(model, batch, device=device,
                                             alpha=A_train[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_ees.append(float(avg_ee))

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        val_ees = []
        with torch.no_grad():
            for b in range(0, len(val_data), batch_size):
                batch = val_data[b:b + batch_size]
                loss, avg_ee = compute_bgat_loss(model, batch, device=device,
                                                 alpha=A_val[b:b + batch_size])
                val_losses.append(loss.item())
                val_ees.append(float(avg_ee))

        mean_train_loss = np.mean(train_losses)
        mean_train_ee = np.mean(train_ees)
        mean_val_loss = np.mean(val_losses)
        mean_val_ee = np.mean(val_ees)

        if epoch % max(1, epochs // 20) == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {mean_train_loss:.6f} "
                  f"Train EE: {mean_train_ee:.6f} | Val Loss: {mean_val_loss:.6f} "
                  f"Val EE: {mean_val_ee:.6f} | pos_scale: "
                  f"{float(model.pos_scale):.4f} m", flush=True)

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_dir, 'bgat_best.pt'))

    print(f"Training complete. Best validation loss: {best_val_loss:.6f} "
          f"(epoch {best_epoch})")
    print(f"Model saved to {os.path.join(model_dir, 'bgat_best.pt')}")
    return model


def load_bgat(model_path='../models/bgat_best.pt', device='cpu', model_dir=None):
    """Load pre-trained BGAT model (model_dir=... is what bgat_benchmark passes)."""
    if model_dir is not None:
        model_path = os.path.join(model_dir, 'bgat_best.pt')
    model = BGAT(n_blocks=5, feat_dim=16, n_heads=4, mlp_hidden=32).to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print(f"Loaded BGAT model from {model_path}")
        except RuntimeError as e:
            # e.g. an older checkpoint whose edge/readout shapes differ
            print(f"Warning: {model_path} is incompatible ({e}); random init.")
    else:
        print(f"Warning: BGAT model not found at {model_path}, using random init.")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model_dir', type=str, default='../models')
    parser.add_argument('--n_train', type=int, default=20000)
    parser.add_argument('--n_val', type=int, default=2000)
    parser.add_argument('--report-range', action='store_true',
                        help='draw D/Pmax/Tframe from the reported sweep range')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pos-scale', type=float, default=0.05,
                        help='initial displacement magnitude in metres')
    args = parser.parse_args()

    train_bgat(epochs=args.epochs, batch_size=args.batch_size,
               lr=args.lr, device=args.device, model_dir=args.model_dir,
               n_train=args.n_train, n_val=args.n_val,
               report_range=args.report_range, seed=args.seed,
               pos_scale=args.pos_scale)
