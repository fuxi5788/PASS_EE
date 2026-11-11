"""EE versus N = number of pinch points (PAs) per waveguide.

Grid: N = 4, 5, 6, 7, 8, 9, 10 at Tframe = 1.0, everything else at the reported
setting (M = 3 waveguides, K = 3 users, D = 40 m, Pmax = 25 dBm).

PROTOCOL -- MATCHED TO THE OTHER TWO SWEEPS (2026-09-15)
-------------------------------------------------------
This sweep shares run_tf_planned.py's / run_area_sweep.py's protocol and
reporting pipeline exactly:

  * scenario stream : env seed 32, n_mc 100, M = 3, K = 3, D = 40 m,
                      Tframe = 1.0 s, Pmax = 25 dBm, PMOTOR = 1.0 W;
  * budgets         : ADMM WSUB budget 400, DSD budget 200 (for EVERY
                      algorithm -- the old frozen ADMM seed-77 / n_mc-200
                      column is retired);
  * models          : BGAT = ../models/bgat_best.pt (inference + refinement,
                      abort if the checkpoint is missing -- load_bgat would
                      otherwise SILENTLY fall back to a random network),
                      SAC = GNN-SAC_model_20260910_113046.pt (one step);
  * reporting       : ADMM / DSD / BGAT search their deployment with the usual
                      benchmark and the resulting X* is RE-EVALUATED through
                      the shared pipeline

                          A    = optimize_alpha_multiuser(X*)
                          Heff = _dsd_heff(X*, A, env)
                          Deff = _deff(X*, env, X_ref = env.X0)
                          EE   = _solve_wsub(Heff, env, Deff)

                      which is exactly how run_tf_planned.py evaluates every
                      Tframe.  FPA and SAC are single-stage.

CONSISTENCY (mechanical CROSS-CHECK in the report)
--------------------------------------------------
N = 4 is the same configuration as the EE-vs-Tframe figure at Tframe = 1.0 and
the EE-vs-area figure at s = 1.0, so its column reproduces those bit for bit;
the report checks it against tf_sweep_data.csv (Tframe = 1.0 rows), with FPA
checked against the previous n_sweep_data.csv (N = 4), since FPA is not in the
Tframe figure.

SPEED
-----
The five algorithms run per N in ONE pass over ONE scenario stream (one model
load per column) and the seven N columns run in a multiprocessing Pool.  The
old script ran SAC and BGAT serially over all N and ADMM at n_mc = 200, which
was the bottleneck; this one finishes in about the wall time of its slowest
column (N = 10).

Notes
-----
* BGAT_REFINE's coordinate descent is O(M*N) evaluations per sweep, so N = 10
  costs roughly twice N = 4 per scenario -- the Pool hides this behind the
  cheaper columns.
* SAC is native for N <= 13: state layout MAX_M*MAX_N = 40 pinch points, actor
  head MAX_M*MAX_N + MAX_K = 48 wide (3N + K = 33 at N = 10), which is why the
  sweep stops at N = 10.
* Radiation ratios A: exact 2^N enumeration for N <= env.ALPHA_EXACT_MAX_N (8),
  projected-ascent solver above it; the report prints a check of the two.

Usage:
    python run_n_sweep.py --n_mc 100 --write-csv
    python run_n_sweep.py --n_mc 2 --ns 4 10            # quick smoke test
"""
import argparse
import os
import time
from datetime import datetime

import numpy as np

TF = 1.0
NS = [4, 5, 6, 7, 8, 9, 10]
ALGOS = ['FPA', 'ADMM', 'DSD', 'SAC', 'BGAT']
CSV = 'n_sweep_data.csv'
REPORT = 'n_sweep_report.txt'
MODEL = 'GNN-SAC_model_20260910_113046.pt'
MODEL_DIR = 'models' if os.path.isdir('models') else '../models'
TF_CSV = 'tf_sweep_data.csv'          # Tframe = 1.0 reference (ADMM/DSD/SAC/BGAT)
NSWEEP_CSV = 'n_sweep_data.csv'       # previous N = 4 reference (FPA)

ENV_SEED = 32
ADMM_BUDGET = 400
DSD_BUDGET = 200
PMOTOR = 1.0


def run_n(job):
    """Run a subset of algorithms over n_mc scenarios at one N.

    One pass per column; the env stream (seed 32) is the same for every
    algorithm and, at N = 4, identical to run_tf_planned.py's stream.  The
    algorithms run in the fixed order ADMM, DSD, BGAT, FPA, SAC, so the first
    100 scenarios of a longer run reproduce the n_mc = 100 columns bit for bit
    (extra scenarios merely extend the same stream for a lower-SEM mean).
    """
    N, n_mc, pmotor, algos = job
    algos = tuple(algos)
    import env as E
    import benchmark_traditional as BT
    from env import PASSMultiUserEnv, optimize_alpha_multiuser
    E.PMOTOR = pmotor
    BT.PMOTOR = pmotor
    BT.WSUB_BUDGET = ADMM_BUDGET
    BT.DSD_WSUB_BUDGET = DSD_BUDGET

    model = None
    if 'BGAT' in algos:
        from train_bgat import load_bgat
        _ckpt = os.path.join(MODEL_DIR, 'bgat_best.pt')
        if not os.path.exists(_ckpt):
            raise FileNotFoundError('BGAT checkpoint not found: %s' % _ckpt)
        model = load_bgat(model_dir=MODEL_DIR, device='cpu')
    agent = None
    if 'SAC' in algos:
        import torch
        from train_convergence_mu import (GNNSACAgent, STATE_DIM,
                                          MAX_ACTION_DIM, device)
        agent = GNNSACAgent(STATE_DIM, MAX_ACTION_DIM)
        ck = torch.load(MODEL, map_location=device)
        for k in ('actor_encoder', 'critic_encoder', 'actor', 'critic1',
                  'critic2'):
            getattr(agent, k).load_state_dict(ck[k])
        agent.actor.eval()

    env = PASSMultiUserEnv(seed=ENV_SEED, randomize_params=False, M=3, N=N,
                           K=3, D=40.0, Pmax_dbm=25.0, Tframe=TF)
    out = {a: [] for a in algos}
    mds = {a: [] for a in algos}
    for _ in range(n_mc):
        s0 = env.reset()
        for name, run_fn in (
                ('ADMM', lambda: BT.admm_benchmark(env, return_X=True)),
                ('DSD', lambda: BT.dsd_benchmark(env, return_X=True)),
                ('BGAT', lambda: BT.bgat_benchmark(env, model=model,
                                                   device='cpu',
                                                   return_X=True))):
            if name not in algos:
                continue
            _, X = run_fn()
            if X is None:
                continue
            X = np.asarray(X, dtype=float)
            mds[name].append(float(np.max(np.abs(X - env.X0))))
            # shared reporting pipeline (the run_tf_planned.py evaluator)
            A = optimize_alpha_multiuser(X, env.user_pos, env.M, env.N,
                                         env.K, env.YM)
            Heff = BT._dsd_heff(X, A, env)
            Deff = BT._deff(X, env, X_ref=env.X0)
            ee = -np.inf
            if Deff is not None:
                ee, _ = BT._solve_wsub(Heff, env, Deff)
            if np.isfinite(ee):
                out[name].append(float(ee))
        if 'FPA' in algos:
            v = BT.fpa_benchmark(env)
            if np.isfinite(v):
                out['FPA'].append(float(v))
        if 'SAC' in algos:
            _, _, _, info = env.step(agent.select_action(s0, evaluate=True))
            if np.isfinite(info['ee']):
                out['SAC'].append(float(info['ee']))
                mds['SAC'].append(float(info['Md']))
    return dict(N=N,
                res={a: (float(np.mean(v)), float(np.std(v)), len(v))
                     for a, v in out.items() if v},
                md={a: float(np.mean(v)) for a, v in mds.items() if v})


def read_tf_reference():
    """{algo: (mean, std, n)} at Tframe = 1.0 from tf_sweep_data.csv."""
    ref = {}
    if os.path.exists(TF_CSV):
        with open(TF_CSV) as f:
            next(f)
            for ln in f:
                p = ln.strip().split(',')
                if (len(p) >= 5 and p[0] != 'FPA'
                        and abs(float(p[1]) - 1.0) < 1e-9):
                    ref[p[0]] = (float(p[2]), float(p[3]), int(p[4]))
    return ref


def read_fpa_reference():
    """FPA at N = 4 from the previous n_sweep_data.csv (pre-overwrite)."""
    if not os.path.exists(NSWEEP_CSV):
        return {}
    with open(NSWEEP_CSV) as f:
        next(f)
        for ln in f:
            p = ln.strip().split(',')
            if len(p) >= 5 and p[0] == 'FPA' and int(float(p[1])) == 4:
                return {'FPA': (float(p[2]), float(p[3]), int(p[4]))}
    return {}


def alpha_check(ns):
    """Solver vs exact enumeration on random radiation-ratio problems."""
    import env as E
    out = []
    for Nn in ns:
        rng = np.random.RandomState(Nn)
        C = rng.randn(Nn, 3) + 1j * rng.randn(Nn, 3)
        Q = np.real(C @ C.conj().T)
        _, v_ex = E.solve_ratio(Q)
        _, v_lg = E.solve_ratio_large(Q)
        out.append((Nn, v_ex, v_lg))
    return out


def main():
    ap = argparse.ArgumentParser(
        description='EE vs N per waveguide, protocol matched to '
                    'run_tf_planned.py (the EE-vs-Tframe rerun)')
    ap.add_argument('--n_mc', type=int, default=1000)
    ap.add_argument('--pmotor', type=float, default=PMOTOR)
    ap.add_argument('--ns', nargs='*', type=int, default=NS)
    ap.add_argument('--algos', nargs='*', default=ALGOS,
                    help='subset of FPA/ADMM/DSD/SAC/BGAT to run')
    ap.add_argument('--serial', action='store_true',
                    help='no multiprocessing (debugging; the Pool is what '
                         'makes the sweep fast)')
    ap.add_argument('--write-csv', action='store_true')
    ap.add_argument('--report', default=REPORT)
    ap.add_argument('--out', default=CSV)
    args = ap.parse_args()

    ns = [int(n) for n in args.ns]
    algos = tuple(args.algos)
    jobs = [(n, args.n_mc, args.pmotor, algos) for n in ns]
    print('PMOTOR %.1f W   Tframe %.1f   n_mc %d   model_dir %s'
          % (args.pmotor, TF, args.n_mc, MODEL_DIR), flush=True)
    print('N = %s' % ns, flush=True)

    t0 = time.perf_counter()
    results = []
    if args.serial:
        results = [run_n(j) for j in jobs]
        for r in results:
            m = ', '.join('%s %.4f' % (a, r['res'][a][0])
                          for a in ALGOS if a in r['res'])
            print('  N=%-3d done (%.0fs): %s'
                  % (r['N'], time.perf_counter() - t0, m), flush=True)
    else:
        try:
            from multiprocessing import Pool
            with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
                for r in pool.imap_unordered(run_n, jobs):
                    results.append(r)
                    m = ', '.join('%s %.4f' % (a, r['res'][a][0])
                                  for a in ALGOS if a in r['res'])
                    print('  N=%-3d done (%.0fs): %s'
                          % (r['N'], time.perf_counter() - t0, m), flush=True)
        except Exception as exc:
            print('parallel failed (%s); serial fallback' % type(exc).__name__,
                  flush=True)
            results = [run_n(j) for j in jobs]
    results.sort(key=lambda r: r['N'])

    # references are read BEFORE --write-csv overwrites n_sweep_data.csv
    ref = read_tf_reference()
    ref.update(read_fpa_reference())
    n4 = next((r for r in results if r['N'] == 4), None)

    L = []

    def A(s):
        L.append(s)

    A('=' * 78)
    A('EE vs NUMBER OF PINCH POINTS PER WAVEGUIDE   --   PMOTOR = %.1f W'
      % args.pmotor)
    A('=' * 78)
    A('generated : %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    A('command   : python run_n_sweep.py --n_mc %d%s'
      % (args.n_mc, ' --write-csv' if args.write_csv else ''))
    A('')
    A('settings  : Tframe = %.1f s (fixed), M = 3 waveguides, K = 3 users,' % TF)
    A('            D = 40 m, Pmax = 25 dBm')
    A('            X0: N pinch points per waveguide, 40*lambda_g apart,')
    A('            centred on the user (env._initial_positions)')
    A('')
    A('convention : protocol and reporting pipeline MATCHED to the')
    A('             EE-vs-Tframe rerun (run_tf_planned.py): all schemes on the')
    A('             env seed %d / n_mc %d stream; ADMM budget %d, DSD budget %d;'
      % (ENV_SEED, args.n_mc, ADMM_BUDGET, DSD_BUDGET))
    A('             BGAT = %s inference + refinement; SAC = %s'
      % (os.path.join(MODEL_DIR, 'bgat_best.pt'), MODEL))
    A('             (one step).  ADMM / DSD / BGAT search the deployment with')
    A('             the usual benchmark and the resulting X* is re-evaluated')
    A('             through the shared pipeline (optimize_alpha_multiuser +')
    A('             _dsd_heff + _deff(X_ref = X0) + _solve_wsub) -- the same')
    A('             evaluator the Tframe / area figures use.  FPA / SAC are')
    A('             single-stage.  The old frozen ADMM seed-77 / n_mc-200')
    A('             column is retired.')
    A('')
    A('--- RESULTS (mean EE, Mbit/J) -------------------------------------')
    A('%-6s %s' % ('N', ''.join('%-12s' % a for a in ALGOS)))
    for r in results:
        A('%-6d %s' % (r['N'], ''.join(
            '%-12.4f' % r['res'][a][0] if a in r['res'] else '%-12s' % 'n/a'
            for a in ALGOS)))
    A('')
    A('std / n / M_d (m):')
    for a in ALGOS:
        for r in results:
            if a in r['res']:
                m, sd, n = r['res'][a]
                A('  %-5s N=%-3d std %-9.4f n %-4d M_d %s'
                  % (a, r['N'], sd, n,
                     '%.4f' % r['md'][a] if a in r['md'] else '-.----'))
    A('')
    A('--- CROSS-CHECK: N = 4 must equal the EE-vs-Tframe figure at')
    A('    Tframe = 1.0 and the EE-vs-area figure at s = 1.0 (same scenario')
    A('    stream, same reporting pipeline); FPA is checked against the')
    A('    previous n_sweep_data.csv (N = 4), because FPA is not in the')
    A('    Tframe figure.')
    if n4:
        worst = 0.0
        for a in ALGOS:
            if a in n4['res'] and a in ref:
                d = abs(n4['res'][a][0] - ref[a][0])
                worst = max(worst, d)
                A('  %-5s  %.6f  vs  %.6f  (|dev| = %.1e)  %s'
                  % (a, n4['res'][a][0], ref[a][0], d,
                     'OK' if d < 5e-4 else 'MISMATCH'))
        A('  worst |dev| = %.1e Mbit/J' % worst)
    A('')
    A('radiation-ratio check (exact 2^N enumeration vs projected ascent):')
    for Nn, v_ex, v_lg in alpha_check(ns):
        A('  N=%-3d exact %.8f   solver %.8f   rel gap %+.1e'
          % (Nn, v_ex, v_lg, (v_lg - v_ex) / abs(v_ex)))
    A('=' * 78)
    with open(args.report, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))

    if args.write_csv:
        with open(args.out, 'w') as f:
            f.write('algo,N,ee_mean,ee_std,n\n')
            for r in results:
                for a in ALGOS:
                    if a in r['res']:
                        m, sd, n = r['res'][a]
                        f.write('%s,%d,%.6f,%.6f,%d\n'
                                % (a, r['N'], m, sd, n))
        print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
