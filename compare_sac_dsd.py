"""
compare_sac_dsd.py
20-run Monte-Carlo comparison between the trained GNN-SAC policy and DSD.
Both are evaluated on the same user realization with the same EE function.
"""
import numpy as np
import torch

from env import (
    PASSMultiUserEnv,
    compute_ee_complete,
    optimize_alpha_multiuser,
    clip_feasible,
)
from benchmark_traditional import dsd_benchmark
from train_convergence_mu import GNNSACAgent, STATE_DIM, MAX_ACTION_DIM, device

N_MC = 1000
MODEL = 'GNN-SAC_model_20260910_113046.pt'


def eval_gnn_sac(agent, env):
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
    Xc = clip_feasible(best_X, env.X0, env.Dmax, env.D, env.M, env.N)
    A = optimize_alpha_multiuser(Xc, env.user_pos, env.M, env.N, env.K, env.YM)
    ee, _, _, _ = compute_ee_complete(
        Xc, A, env.user_pos, env.M, env.N, env.K, env.YM,
        env.X0, env.Tframe, env.Pmax, env.Rmin)
    return ee if np.isfinite(ee) else -np.inf


def main():
    agent = GNNSACAgent(STATE_DIM, MAX_ACTION_DIM)
    ckpt = torch.load(MODEL, map_location=device)
    agent.actor_encoder.load_state_dict(ckpt['actor_encoder'])
    agent.critic_encoder.load_state_dict(ckpt['critic_encoder'])
    agent.actor.load_state_dict(ckpt['actor'])
    agent.critic1.load_state_dict(ckpt['critic1'])
    agent.critic2.load_state_dict(ckpt['critic2'])
    agent.actor.eval()

    sac_vals, dsd_vals = [], []
    for seed in range(N_MC):
        env = PASSMultiUserEnv(seed=seed, randomize_params=False,
                               M=3, N=4, K=3, D=40.0,
                               Pmax_dbm=25.0, Tframe=1.0)
        sac = eval_gnn_sac(agent, env)
        dsd = dsd_benchmark(env)
        sac_vals.append(sac)
        dsd_vals.append(dsd)
        print(f"[{seed+1:2d}/{N_MC}] GNN-SAC={sac:.4f}  DSD={dsd:.4f}  diff={sac-dsd:+.4f}")

    sac_vals = np.array(sac_vals)
    dsd_vals = np.array(dsd_vals)
    print("\n===== 20-run summary =====")
    print(f"GNN-SAC: {sac_vals.mean():.4f} +/- {sac_vals.std():.4f}")
    print(f"DSD    : {dsd_vals.mean():.4f} +/- {dsd_vals.std():.4f}")
    print(f"mean diff (SAC - DSD): {(sac_vals - dsd_vals).mean():+.4f}")
    print(f"wins (SAC > DSD): {int((sac_vals > dsd_vals).sum())}/{N_MC}")


if __name__ == '__main__':
    main()
