"""
gnn_sac_mc100.py
Single-step inference of the trained GNN-SAC policy over n_mc Monte-Carlo user
realizations (environment seed fixed). One forward pass from the initial state
yields one deployment; its EE is computed and averaged, then APPENDED to frame.txt.
"""
import sys
import argparse
import numpy as np
import torch

from env import PASSMultiUserEnv, optimize_alpha_multiuser, clip_feasible
from benchmark_traditional import _dsd_ee
from train_convergence_mu import GNNSACAgent, STATE_DIM, MAX_ACTION_DIM, device

MODEL = 'GNN-SAC_model_20260910_113046.pt'


class Suppressor:
    def write(self, x):
        pass

    def flush(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=MODEL)
    parser.add_argument('--n_mc', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--M', type=int, default=3)
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--D', type=float, default=40.0)
    parser.add_argument('--Pmax_dbm', type=float, default=25.0)
    parser.add_argument('--Tframe', type=float, default=1.0)
    parser.add_argument('--out', type=str, default='frame.txt')
    args = parser.parse_args()

    agent = GNNSACAgent(STATE_DIM, MAX_ACTION_DIM)
    ckpt = torch.load(args.model, map_location=device)
    agent.actor_encoder.load_state_dict(ckpt['actor_encoder'])
    agent.critic_encoder.load_state_dict(ckpt['critic_encoder'])
    agent.actor.load_state_dict(ckpt['actor'])
    agent.critic1.load_state_dict(ckpt['critic1'])
    agent.critic2.load_state_dict(ckpt['critic2'])
    agent.actor.eval()
    print(f"Loaded GNN-SAC model: {args.model}")

    env = PASSMultiUserEnv(seed=args.seed, randomize_params=False,
                           M=args.M, N=args.N, K=args.K, D=args.D,
                           Pmax_dbm=args.Pmax_dbm, Tframe=args.Tframe)

    env_ees, a_ees = [], []
    old_stdout = sys.stdout
    for i in range(args.n_mc):
        sys.stdout = Suppressor()
        state = env.reset()
        # One forward pass -> one deployment
        action = agent.select_action(state, evaluate=True)
        _, _, _, info = env.step(action)
        X = env.x.copy()
        sys.stdout = old_stdout

        env_ees.append(info['ee'])
        Xc = clip_feasible(X, env.X0, env.Dmax, env.D, env.M, env.N)
        A = optimize_alpha_multiuser(Xc, env.user_pos, env.M, env.N, env.K, env.YM)
        a_ees.append(_dsd_ee(Xc, A, env))

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1:3d}/{args.n_mc}] env_ee={env_ees[-1]:.4f}  "
                  f"a_aware_ee={a_ees[-1]:.4f}")

    env_ees = np.array([v for v in env_ees if np.isfinite(v)])
    a_ees = np.array([v for v in a_ees if np.isfinite(v)])

    block = [
        "",
        "[GNN-SAC]",
        "inference = single_step",
        f"n_mc = {args.n_mc}",
        f"seed = {args.seed}",
        (f"M = {args.M}, N = {args.N}, K = {args.K}, D = {args.D}, "
         f"Pmax_dbm = {args.Pmax_dbm}, Tframe = {args.Tframe}"),
        f"env_ee_mean    = {env_ees.mean():.6f}",
        f"env_ee_std     = {env_ees.std():.6f}",
        f"a_aware_ee_mean = {a_ees.mean():.6f}",
        f"a_aware_ee_std  = {a_ees.std():.6f}",
        f"ee_n           = {len(a_ees)}",
    ]
    with open(args.out, 'a', encoding='utf-8') as f:
        f.write('\n'.join(block) + '\n')

    print('\n'.join(block))
    print(f"Appended to {args.out}")


if __name__ == '__main__':
    main()
