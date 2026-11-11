"""
mulsim/evaluate_mu.py
Load trained GDRL weights and evaluate parameter sweeps.
No retraining needed.
Usage:
    cd mulsim
    python evaluate_mu.py --model ../results_convergence/SAC_model_*.pt --agent SAC
"""
import os
import sys
import argparse
import time
import numpy as np
import torch

from env import PASSMultiUserEnv, MAX_ACTION_DIM, STATE_DIM, STEPS_PER_EPISODE, device
from train_convergence_mu import SACAgent, DDPGAgent, PPOAgent, A2CAgent


def load_agent(agent_name, model_path):
    if agent_name == 'SAC':
        agent = SACAgent(STATE_DIM, MAX_ACTION_DIM)
        ckpt = torch.load(model_path, map_location=device)
        agent.actor.load_state_dict(ckpt['actor'])
        agent.critic1.load_state_dict(ckpt['critic1'])
        agent.critic2.load_state_dict(ckpt['critic2'])
    elif agent_name == 'DDPG':
        agent = DDPGAgent(STATE_DIM, MAX_ACTION_DIM)
        ckpt = torch.load(model_path, map_location=device)
        agent.actor.load_state_dict(ckpt['actor'])
        agent.critic.load_state_dict(ckpt['critic'])
    elif agent_name in ['PPO', 'A2C']:
        if agent_name == 'PPO':
            agent = PPOAgent(STATE_DIM, MAX_ACTION_DIM)
        else:
            agent = A2CAgent(STATE_DIM, MAX_ACTION_DIM)
        ckpt = torch.load(model_path, map_location=device)
        agent.actor.load_state_dict(ckpt['actor'])
        agent.critic.load_state_dict(ckpt['critic'])
    else:
        raise ValueError(f"Unknown agent {agent_name}")
    return agent


def evaluate_one(agent, agent_name, env, n_steps=STEPS_PER_EPISODE):
    state = env.reset()
    ep_ee = 0.0
    step = 0
    while True:
        if agent_name in ['PPO', 'A2C']:
            action_vec, _ = agent.select_action(state, evaluate=True)
        else:
            action_vec = agent.select_action(state, evaluate=True)
        next_state, reward, done, info = env.step(action_vec)
        ep_ee += info['ee']
        step += 1
        if done:
            break
    return ep_ee / step if step > 0 else 0.0


def sweep_param(agent, agent_name, param_name, values,
                fixed_M=3, fixed_N=4, fixed_K=3,
                fixed_D=40.0, fixed_Pmax_dbm=25.0, fixed_Tframe=1.0,
                n_users=1000):
    results = []
    for val in values:
        kwargs = {
            'M': fixed_M, 'N': fixed_N, 'K': fixed_K,
            'D': fixed_D, 'Pmax_dbm': fixed_Pmax_dbm, 'Tframe': fixed_Tframe,
        }
        kwargs[param_name] = val
        ees = []
        for u in range(n_users):
            env = PASSMultiUserEnv(seed=u, randomize_params=False, **kwargs)
            ee = evaluate_one(agent, agent_name, env)
            ees.append(ee)
        results.append(np.mean(ees))
        print(f"  {param_name}={val}: mean EE={results[-1]:.4f}")
    return np.array(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--agent', type=str, default='SAC', choices=['SAC', 'DDPG', 'PPO', 'A2C'])
    parser.add_argument('--n_users', type=int, default=1000)
    args = parser.parse_args()

    print(f"Loading {args.agent} from {args.model}")
    agent = load_agent(args.agent, args.model)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'results_eval')
    os.makedirs(save_dir, exist_ok=True)

    all_data = {'agent': args.agent, 'n_users': args.n_users}

    print("\n=== Sweep Pmax ===")
    pmax_dbm = np.array([15.0, 20.0, 25.0, 30.0, 35.0])
    all_data['ee_vs_pmax'] = sweep_param(agent, args.agent, 'Pmax_dbm', pmax_dbm, n_users=args.n_users)
    all_data['pmax_dbm'] = pmax_dbm

    print("\n=== Sweep K ===")
    ks = np.array([2, 3, 4, 5, 6, 7, 8])
    all_data['ee_vs_k'] = sweep_param(agent, args.agent, 'K', ks, n_users=args.n_users)
    all_data['k_values'] = ks

    print("\n=== Sweep N ===")
    ns = np.array([2, 3, 4, 5, 6, 7, 8])
    all_data['ee_vs_n'] = sweep_param(agent, args.agent, 'N', ns, n_users=args.n_users)
    all_data['n_values'] = ns

    print("\n=== Sweep M ===")
    ms = np.array([2, 3, 4, 5])
    all_data['ee_vs_m'] = sweep_param(agent, args.agent, 'M', ms, n_users=args.n_users)
    all_data['m_values'] = ms

    print("\n=== Sweep Tframe ===")
    tframes = np.array([0.5, 0.7, 1.0, 1.3, 1.5, 2.0])
    all_data['ee_vs_tframe'] = sweep_param(agent, args.agent, 'Tframe', tframes, n_users=args.n_users)
    all_data['tframe_values'] = tframes

    print("\n=== Sweep D ===")
    ds = np.array([20.0, 30.0, 40.0, 50.0, 60.0])
    all_data['ee_vs_d'] = sweep_param(agent, args.agent, 'D', ds, n_users=args.n_users)
    all_data['d_values'] = ds

    save_path = os.path.join(save_dir, f"eval_{args.agent}_{timestamp}.npz")
    np.savez(save_path, **all_data)
    print(f"\nSaved to: {save_path}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        sweeps = [
            ('pmax_dbm', 'ee_vs_pmax', r'$P_{\max}$ (dBm)'),
            ('k_values', 'ee_vs_k', 'Users $K$'),
            ('n_values', 'ee_vs_n', 'PAs $N$'),
            ('m_values', 'ee_vs_m', 'Waveguides $M$'),
            ('tframe_values', 'ee_vs_tframe', r'$T_{\mathrm{frame}}$ (s)'),
            ('d_values', 'ee_vs_d', 'Area $D$ (m)'),
        ]
        for ax, (xk, yk, xlab) in zip(axes.flat, sweeps):
            ax.plot(all_data[xk], all_data[yk], 'o-', linewidth=1.5)
            ax.set_xlabel(xlab)
            ax.set_ylabel('EE (bps/Hz/J)')
            ax.grid(True)
        plt.suptitle(f'{args.agent} Evaluation')
        plt.tight_layout()
        preview = os.path.join(save_dir, f"eval_{args.agent}_{timestamp}.png")
        plt.savefig(preview, dpi=300)
        print(f"Preview: {preview}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == '__main__':
    main()
