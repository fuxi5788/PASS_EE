"""
compare_drl_benchmark.py
Fair comparison between the trained GNN-SAC policy and traditional benchmarks.

All methods are evaluated on the SAME user realization, with the SAME
regularization (compute_ee_complete, reg = 1e-6) and the SAME movement
constraint (clip_feasible with Dmax).
"""
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env import (
    PASSMultiUserEnv,
    compute_ee_complete,
    optimize_alpha_multiuser,
    clip_feasible,
)
from benchmark_traditional import (
    admm_benchmark,
    dsd_benchmark,
    bgat_benchmark,
    fpa_benchmark,
)
from train_convergence_mu import GNNSACAgent, STATE_DIM, MAX_ACTION_DIM, device

METHODS = ['GNN-SAC', 'ADMM', 'DSD', 'BGAT', 'FPA']


def eval_gnn_sac(agent, env):
    """Run the trained GNN-SAC policy deterministically and return the best
    PA deployment evaluated with the unified EE function."""
    state = env.reset()
    best_env_ee = -np.inf
    best_X = env.x.copy()
    done = False
    while not done:
        action = agent.select_action(state, evaluate=True)
        next_state, _, done, info = env.step(action)
        state = next_state
        if info['ee'] > best_env_ee:
            best_env_ee = info['ee']
            best_X = env.x.copy()

    # Unified evaluation: optimal radiation ratios + optimal power
    Xc = clip_feasible(best_X, env.X0, env.Dmax, env.D, env.M, env.N)
    A = optimize_alpha_multiuser(Xc, env.user_pos, env.M, env.N, env.K, env.YM)
    ee, _, _, _ = compute_ee_complete(
        Xc, A, env.user_pos, env.M, env.N, env.K, env.YM,
        env.X0, env.Tframe, env.Pmax, env.Rmin)
    return ee if np.isfinite(ee) else -np.inf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='GNN-SAC_model_20260910_113046.pt')
    parser.add_argument('--n_mc', type=int, default=1000)
    parser.add_argument('--M', type=int, default=3)
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--D', type=float, default=40.0)
    parser.add_argument('--Pmax_dbm', type=float, default=25.0)
    parser.add_argument('--Tframe', type=float, default=1.0)
    parser.add_argument('--seed_start', type=int, default=0)
    args = parser.parse_args()

    print(f"Loading GNN-SAC model: {args.model}")
    agent = GNNSACAgent(STATE_DIM, MAX_ACTION_DIM)
    ckpt = torch.load(args.model, map_location=device)
    agent.actor_encoder.load_state_dict(ckpt['actor_encoder'])
    agent.critic_encoder.load_state_dict(ckpt['critic_encoder'])
    agent.actor.load_state_dict(ckpt['actor'])
    agent.critic1.load_state_dict(ckpt['critic1'])
    agent.critic2.load_state_dict(ckpt['critic2'])
    agent.actor.eval()
    print("Model loaded.")

    results = {m: [] for m in METHODS}
    t0 = time.time()

    for i in range(args.n_mc):
        seed = args.seed_start + i
        env = PASSMultiUserEnv(
            seed=seed, randomize_params=False,
            M=args.M, N=args.N, K=args.K, D=args.D,
            Pmax_dbm=args.Pmax_dbm, Tframe=args.Tframe)
        # eval_gnn_sac calls env.reset() and leaves the same user realization
        # in env, so all benchmarks afterwards see the identical users/X0.
        results['GNN-SAC'].append(eval_gnn_sac(agent, env))
        results['ADMM'].append(admm_benchmark(env))
        results['DSD'].append(dsd_benchmark(env))
        results['BGAT'].append(bgat_benchmark(env))
        results['FPA'].append(fpa_benchmark(env))

        if (i + 1) % 10 == 0 or i == 0:
            msg = f"[{i+1:4d}/{args.n_mc}] "
            for m in METHODS:
                vals = [v for v in results[m] if np.isfinite(v)]
                msg += f"{m}={np.mean(vals):.3f} " if vals else f"{m}=nan "
            msg += f"({time.time()-t0:.0f}s)"
            print(msg)

    # Summary
    print("\n===== Summary (mean +/- std) =====")
    summary = {}
    for m in METHODS:
        vals = np.array([v for v in results[m] if np.isfinite(v)])
        summary[m] = (vals.mean() if len(vals) else np.nan,
                      vals.std() if len(vals) else np.nan,
                      len(vals))
        print(f"{m:8s}: {summary[m][0]:.4f} +/- {summary[m][1]:.4f}  (n={summary[m][2]})")

    # Save
    np.savez('compare_drl_benchmark_result.npz',
             **{m.replace('-', '_'): np.array(results[m]) for m in METHODS},
             n_mc=args.n_mc,
             M=args.M, N=args.N, K=args.K, D=args.D,
             Pmax_dbm=args.Pmax_dbm, Tframe=args.Tframe)
    print("Saved: compare_drl_benchmark_result.npz")

    # Bar plot
    means = [summary[m][0] for m in METHODS]
    stds = [summary[m][1] for m in METHODS]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(METHODS, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('EE (bps/Hz/J)', fontsize=12)
    ax.set_title(f'Fair comparison ({args.n_mc} Monte-Carlo users)', fontsize=13)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig('compare_drl_benchmark.png', dpi=300)
    print("Saved: compare_drl_benchmark.png")


if __name__ == '__main__':
    main()
