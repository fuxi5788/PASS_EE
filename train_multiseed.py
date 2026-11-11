# -*- coding: utf-8 -*-
"""
Train GNN-SAC and DDPG over multiple random seeds for Fig.7 confidence intervals.

The paper's "28% higher than DDPG" claim compares the final average reward
(actually the per-episode average EE, the *_ees field) of GNN-SAC against DDPG.
Seed 42 is already available (GNN-SAC final EE 9.993, DDPG 7.828, from
plot_eposide_reward1110.mat).  This script trains the remaining seeds so that the
mean/std across 10 seeds and a bootstrap 95% confidence interval can be computed.

For each (seed, algorithm) pair it trains MAX_EPISODES episodes and records
    final_ee_last100 = mean of the per-episode average EE over the last 100 eps.
Each finished pair is written to its own npz so the run is resumable.  After all
pairs finish, a summary (mean/std of the relative gain and its bootstrap 95% CI)
is written to summary_multiseed.txt.

Usage (run from this directory):
    python train_multiseed.py                       # 9 seeds x {GNN-SAC, DDPG}
    python train_multiseed.py --algos GNN-SAC DDPG --seeds 0 1 2 3 4 5 6 7 8
    python train_multiseed.py --seeds 0 1 --algos DDPG --episodes 10   # smoke
"""
import os
import re
import glob
import argparse
import numpy as np
import torch

from train_convergence_mu import (
    GNNSACAgent,
    DDPGAgent,
    STATE_DIM,
    MAX_ACTION_DIM,
    MAX_EPISODES,
    train_agent,
)
from env import PASSMultiUserEnv

ALGO_CLS = {
    'GNN-SAC': GNNSACAgent,
    'DDPG': DDPGAgent,
}

# Seed 42 already exists in the paper's Fig.7 data (plot_eposide_reward1110.mat).
# final_ee_last100 = mean of the *_ees field over the last 100 episodes.
SEED42_BASELINE = {'GNN-SAC': 9.993, 'DDPG': 7.828}

N_BOOT = 20000


def build_env(seed, Tframe=1.0):
    return PASSMultiUserEnv(
        seed=seed,
        randomize_params=False,
        M=3, N=4, K=3,
        D=40.0,
        Tframe=Tframe,
        v_fixed=1.0,
        Pmax_dbm=25.0,
    )


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize(outdir):
    """Collect every finished seed, fold in the seed-42 baseline, and compute
    the relative-gain bootstrap 95% CI.  Writes summary_multiseed.txt."""
    gnn = {42: SEED42_BASELINE['GNN-SAC']}
    ddpg = {42: SEED42_BASELINE['DDPG']}

    for f in glob.glob(os.path.join(outdir, 'GNN-SAC_seed*.npz')):
        m = re.search(r'seed(\d+)\.npz$', f)
        if not m:
            continue
        z = np.load(f)
        gnn[int(m.group(1))] = float(z['final_ee_last100'])
    for f in glob.glob(os.path.join(outdir, 'DDPG_seed*.npz')):
        m = re.search(r'seed(\d+)\.npz$', f)
        if not m:
            continue
        z = np.load(f)
        ddpg[int(m.group(1))] = float(z['final_ee_last100'])

    seeds = sorted(set(gnn.keys()) & set(ddpg.keys()))
    lines = []
    lines.append('=' * 72)
    lines.append('MULTISEED SUMMARY (GNN-SAC vs DDPG, final EE over last 100 eps)')
    lines.append('=' * 72)
    lines.append('%-6s %12s %12s %10s' % ('seed', 'GNN-SAC', 'DDPG', 'gain%'))
    gvals = []
    for s in seeds:
        gain = (gnn[s] - ddpg[s]) / ddpg[s] * 100.0
        gvals.append(gain)
        lines.append('%-6d %12.4f %12.4f %9.2f' % (s, gnn[s], ddpg[s], gain))

    gvals = np.array(gvals)
    rng = np.random.RandomState(0)
    boot = np.array([
        gvals[rng.randint(0, len(gvals), len(gvals))].mean()
        for _ in range(N_BOOT)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    lines.append('-' * 72)
    lines.append('n seeds            : %d' % len(seeds))
    lines.append('mean gain          : %.2f %%' % gvals.mean())
    lines.append('std of gain        : %.2f %%' % gvals.std(ddof=1))
    lines.append('bootstrap 95%% CI   : [%.2f %%, %.2f %%]' % (lo, hi))
    lines.append('=' * 72)

    text = '\n'.join(lines) + '\n'
    print(text)
    with open(os.path.join(outdir, 'summary_multiseed.txt'), 'w') as f:
        f.write(text)
    print('summary written to %s' % os.path.join(outdir, 'summary_multiseed.txt'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algos', nargs='+', type=str,
                        default=['GNN-SAC', 'DDPG'],
                        choices=['GNN-SAC', 'DDPG'])
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=list(range(9)),
                        help='Seeds to train (seed 42 is the existing baseline).')
    parser.add_argument('--episodes', type=int, default=MAX_EPISODES)
    parser.add_argument('--outdir', type=str, default='results_multiseed')
    parser.add_argument('--force', action='store_true',
                        help='Re-train even if the output npz already exists.')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.outdir)
    os.makedirs(outdir, exist_ok=True)

    print('=' * 70)
    print('MULTISEED TRAINING  algos=%s seeds=%s episodes=%d'
          % (args.algos, args.seeds, args.episodes))
    print('device =', 'cuda' if torch.cuda.is_available() else 'cpu')
    print('outdir =', outdir)
    print('=' * 70)

    for seed in args.seeds:
        for algo in args.algos:
            out_path = os.path.join(outdir, '%s_seed%d.npz' % (algo, seed))
            if os.path.exists(out_path) and not args.force:
                print('[skip] %s already exists' % out_path)
                continue

            print('\n===== seed=%d  algo=%s =====' % (seed, algo), flush=True)
            seed_everything(seed)
            env = build_env(seed)
            agent = ALGO_CLS[algo](STATE_DIM, MAX_ACTION_DIM)
            res = train_agent(env, agent, algo, args.episodes)

            ees = np.asarray(res['ees'], dtype=float)
            avg_rewards = np.asarray(res['avg_rewards'], dtype=float)
            final_ee_last100 = float(np.mean(ees[-100:]))
            final_ee_last50 = float(np.mean(ees[-50:]))
            print('[seed=%d %s] final_ee_last100=%.4f  final_ee_last50=%.4f'
                  % (seed, algo, final_ee_last100, final_ee_last50), flush=True)

            np.savez(out_path,
                     seed=seed,
                     algo=algo,
                     final_ee_last100=final_ee_last100,
                     final_ee_last50=final_ee_last50,
                     ees=ees,
                     avg_rewards=avg_rewards)
            print('[saved] %s' % out_path, flush=True)

    summarize(outdir)
    print('\nDone. Outputs in %s' % outdir)


if __name__ == '__main__':
    main()
