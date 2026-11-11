"""
bench_mc_append.py
Run a selected algorithm over n_mc Monte-Carlo user realizations with a fixed
environment seed, compute the average EE, and APPEND the result to frame.txt.

Usage:
    python bench_mc_append.py --algo DSD --n_mc 100 --seed 32 --Tframe 1.0
"""
import sys
import argparse
import numpy as np

from env import PASSMultiUserEnv
from benchmark_traditional import (
    admm_benchmark, dsd_benchmark, bgat_benchmark, fpa_benchmark,
)

ALGOS = {
    'ADMM': admm_benchmark,
    'DSD': dsd_benchmark,
    'BGAT': bgat_benchmark,
    'FPA': fpa_benchmark,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, choices=list(ALGOS))
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

    bench = ALGOS[args.algo]
    env = PASSMultiUserEnv(seed=args.seed, randomize_params=False,
                           M=args.M, N=args.N, K=args.K, D=args.D,
                           Pmax_dbm=args.Pmax_dbm, Tframe=args.Tframe)

    ees = []
    for i in range(args.n_mc):
        env.reset()
        ees.append(bench(env))
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{args.algo}] {i+1:3d}/{args.n_mc}  "
                  f"ee={ees[-1]:.4f}  running_mean={np.mean(ees):.4f}")

    ees = np.array([v for v in ees if np.isfinite(v)])
    block = [
        "",
        f"[{args.algo}]",
        f"n_mc = {args.n_mc}",
        f"seed = {args.seed}",
        (f"M = {args.M}, N = {args.N}, K = {args.K}, D = {args.D}, "
         f"Pmax_dbm = {args.Pmax_dbm}, Tframe = {args.Tframe}"),
        f"ee_mean = {ees.mean():.6f}",
        f"ee_std  = {ees.std():.6f}",
        f"ee_n    = {len(ees)}",
    ]
    with open(args.out, 'a', encoding='utf-8') as f:
        f.write('\n'.join(block) + '\n')

    print('\n'.join(block))
    print(f"Appended to {args.out}")


if __name__ == '__main__':
    main()
