"""EE versus the SCALE of the service area (multi-user simulation).

Service area = the square of side D (the paper: "a square service area of side
length D = 40 m", and the waveguides are uniformly arranged over it).  The sweep
is therefore a sweep of D, with the AREA multipliers

    area = (40 m) x (40 m) x s,   s = 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8

so the side length used for scale s is

    D(s) = 40 * sqrt(s)  =  30.98, 35.78, 40.00, 43.82, 47.33, 50.60, 53.67 m

(area = D^2 = 1600 * s).  All computed D values stay inside the training range
D ~ U(20, 60) m, which is what makes the reuse of the saved GNN-SAC / BGAT
models for inference legitimate.

EVALUATION CONVENTION -- MATCHED TO THE EE-vs-Tframe RERUN
----------------------------------------------------------
This sweep shares run_tf_planned.py's protocol and reporting pipeline exactly:

  * scenario stream : env seed 32, n_mc 100, M = 3, N = 4, K = 3,
                      Tframe = 1.0 s, Pmax = 25 dBm, PMOTOR = 1.0 W;
  * budgets         : ADMM WSUB budget 400, DSD budget 200;
  * models          : BGAT = models/bgat_best.pt (inference + refinement),
                      SAC  = GNN-SAC_model_20260910_113046.pt (one step);
  * reporting       : ADMM / DSD / BGAT search their deployment with the usual
                      benchmark, and the resulting X* is RE-EVALUATED through
                      the shared pipeline

                          A    = optimize_alpha_multiuser(X*)
                          Heff = _dsd_heff(X*, A, env)
                          Deff = _deff(X*, env, X_ref = env.X0)
                          EE   = _solve_wsub(Heff, env, Deff)

                      which is exactly how run_tf_planned.py evaluates every
                      Tframe (the deployment there is searched once at the
                      nominal point).  FPA and SAC are single-stage and need
                      no re-evaluation.

EVERY scale including s = 1.0 is computed here (no reused row).  Because
s = 1.0 (D = 40 m) is the same configuration as the EE-vs-Tframe figure at
Tframe = 1.0, its column reproduces that column bit for bit; the report
cross-checks this mechanically against tf_sweep_data.csv (Tframe = 1.0 rows),
with FPA checked against n_sweep_data.csv (N = 4), since FPA is not in the
Tframe figure.

Output: area_sweep_data.csv (algo,scale,ee_mean,ee_std,n,src) and a text report.

Usage:
    python run_area_sweep.py --n_mc 100 --write-csv
    python run_area_sweep.py --n_mc 2 --scales 0.6        # quick smoke test
"""
import argparse
import os
import time
from datetime import datetime

import numpy as np

D_REF = 40.0
TF = 1.0
N_PA = 4
SCALES = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
ALGOS = ['FPA', 'ADMM', 'DSD', 'SAC', 'BGAT']
CSV = 'area_sweep_data.csv'
REPORT = 'area_sweep_report.txt'
MODEL = 'GNN-SAC_model_20260910_113046.pt'
MODEL_DIR = 'models' if os.path.isdir('models') else '../models'
TF_CSV = 'tf_sweep_data.csv'          # Tframe = 1.0 reference (ADMM/DSD/SAC/BGAT)
NSWEEP_CSV = 'n_sweep_data.csv'       # N = 4 reference (FPA)

ENV_SEED = 32
ADMM_BUDGET = 400
DSD_BUDGET = 200
PMOTOR = 1.0


def D_of(scale):
    """Side length of the square service area whose area is 1600 * scale."""
    return D_REF * float(np.sqrt(scale))


def run_scale(job):
    """All five algorithms over n_mc scenarios at one area scale.

    One pass per scale; the env stream (seed 32) is the same for every
    algorithm and, at s = 1.0, identical to run_tf_planned.py's stream.
    """
    scale, n_mc, pmotor = job
    import torch
    import env as E
    import benchmark_traditional as BT
    from env import PASSMultiUserEnv, optimize_alpha_multiuser
    from train_bgat import load_bgat
    from train_convergence_mu import (GNNSACAgent, STATE_DIM, MAX_ACTION_DIM,
                                      device)
    E.PMOTOR = pmotor
    BT.PMOTOR = pmotor
    BT.WSUB_BUDGET = ADMM_BUDGET
    BT.DSD_WSUB_BUDGET = DSD_BUDGET

    _ckpt = os.path.join(MODEL_DIR, 'bgat_best.pt')
    if not os.path.exists(_ckpt):
        raise FileNotFoundError('BGAT checkpoint not found: %s' % _ckpt)
    model = load_bgat(model_dir=MODEL_DIR, device='cpu')
    agent = GNNSACAgent(STATE_DIM, MAX_ACTION_DIM)
    ck = torch.load(MODEL, map_location=device)
    for k in ('actor_encoder', 'critic_encoder', 'actor', 'critic1', 'critic2'):
        getattr(agent, k).load_state_dict(ck[k])
    agent.actor.eval()

    env = PASSMultiUserEnv(seed=ENV_SEED, randomize_params=False, M=3, N=N_PA,
                           K=3, D=D_of(scale), Pmax_dbm=25.0, Tframe=TF)
    out = {a: [] for a in ALGOS}
    mds = {a: [] for a in ALGOS}
    for _ in range(n_mc):
        s0 = env.reset()
        for name, run_fn in (
                ('ADMM', lambda: BT.admm_benchmark(env, return_X=True)),
                ('DSD', lambda: BT.dsd_benchmark(env, return_X=True)),
                ('BGAT', lambda: BT.bgat_benchmark(env, model=model,
                                                   device='cpu',
                                                   return_X=True))):
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
        v = BT.fpa_benchmark(env)
        if np.isfinite(v):
            out['FPA'].append(float(v))
        _, _, _, info = env.step(agent.select_action(s0, evaluate=True))
        if np.isfinite(info['ee']):
            out['SAC'].append(float(info['ee']))
            mds['SAC'].append(float(info['Md']))
    return dict(scale=scale,
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
    """FPA at N = 4 from n_sweep_data.csv (FPA is not in the Tframe figure)."""
    if not os.path.exists(NSWEEP_CSV):
        return {}
    with open(NSWEEP_CSV) as f:
        next(f)
        for ln in f:
            p = ln.strip().split(',')
            if len(p) >= 5 and p[0] == 'FPA' and int(float(p[1])) == N_PA:
                return {'FPA': (float(p[2]), float(p[3]), int(p[4]))}
    return {}


def main():
    ap = argparse.ArgumentParser(
        description='EE vs service-area scale, protocol matched to '
                    'run_tf_planned.py (the EE-vs-Tframe rerun)')
    ap.add_argument('--n_mc', type=int, default=1000)
    ap.add_argument('--pmotor', type=float, default=PMOTOR)
    ap.add_argument('--scales', nargs='*', type=float, default=SCALES)
    ap.add_argument('--write-csv', action='store_true')
    ap.add_argument('--serial', action='store_true',
                    help='no multiprocessing: run in the MAIN process, the '
                         'same floating-point environment run_tf_planned.py '
                         'used (the parallel workers wobble the last bits of '
                         'the BLAS reductions, which can flip a marginal '
                         'deployment-search decision)')
    ap.add_argument('--report', default=REPORT)
    ap.add_argument('--out', default=CSV)
    args = ap.parse_args()

    scales = [float(s) for s in args.scales]
    jobs = [(s, args.n_mc, args.pmotor) for s in scales]
    print('PMOTOR %.1f W   Tframe %.1f   N %d   n_mc %d   model_dir %s'
          % (args.pmotor, TF, N_PA, args.n_mc, MODEL_DIR), flush=True)
    print('scales %s -> D = %s m'
          % (scales, ['%.2f' % D_of(s) for s in scales]), flush=True)

    t0 = time.perf_counter()
    results = []
    if args.serial:
        results = [run_scale(j) for j in jobs]
        for r in results:
            m = ', '.join('%s %.4f' % (a, r['res'][a][0])
                          for a in ALGOS if a in r['res'])
            print('  s=%.1f done (%.0fs): %s'
                  % (r['scale'], time.perf_counter() - t0, m), flush=True)
    else:
        try:
            from multiprocessing import Pool
            with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
                for r in pool.imap_unordered(run_scale, jobs):
                    results.append(r)
                    m = ', '.join('%s %.4f' % (a, r['res'][a][0])
                                  for a in ALGOS if a in r['res'])
                    print('  s=%.1f done (%.0fs): %s'
                          % (r['scale'], time.perf_counter() - t0, m),
                          flush=True)
        except Exception as exc:
            print('parallel failed (%s); serial fallback' % type(exc).__name__,
                  flush=True)
            results = [run_scale(j) for j in jobs]
    results.sort(key=lambda r: r['scale'])

    ref = read_tf_reference()
    ref.update(read_fpa_reference())
    s10 = next((r for r in results if abs(r['scale'] - 1.0) < 1e-9), None)

    L = []

    def A(s):
        L.append(s)

    A('=' * 78)
    A('EE vs SCALE OF THE SERVICE AREA   --   PMOTOR = %.1f W' % args.pmotor)
    A('=' * 78)
    A('generated : %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    A('command   : python run_area_sweep.py --n_mc %d%s'
      % (args.n_mc, ' --write-csv' if args.write_csv else ''))
    A('')
    A('settings  : Tframe = %.1f s (fixed), M = 3 waveguides, K = 3 users,' % TF)
    A('            N = %d PAs per waveguide, Pmax = 25 dBm' % N_PA)
    A('            service area = square of side D, area = 1600 * scale')
    A('            scale  : %s'
      % '  '.join('%.1f' % s for s in scales))
    A('            D (m)  : %s'
      % '  '.join('%.2f' % D_of(s) for s in scales))
    A('            (D = 40*sqrt(scale); all computed D lie inside the training')
    A('             range D ~ U(20, 60) m, so the saved models are used inside')
    A('             the range they were trained on)')
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
    A('             evaluator the Tframe figure uses.  FPA / SAC are')
    A('             single-stage.  EVERY scale incl. s = 1.0 is computed here.')
    A('')
    A('--- RESULTS (mean EE, Mbit/J) -------------------------------------')
    A('%-6s %s' % ('scale', ''.join('%-12s' % a for a in ALGOS)))
    for r in results:
        A('%-6.1f %s' % (r['scale'], ''.join(
            '%-12.4f' % r['res'][a][0] if a in r['res'] else '%-12s' % 'n/a'
            for a in ALGOS)))
    A('')
    A('std / n / M_d (m):')
    for a in ALGOS:
        for r in results:
            if a in r['res']:
                m, sd, n = r['res'][a]
                A('  %-5s s=%.1f  D=%-6.2f std %-9.4f n %-4d M_d %s'
                  % (a, r['scale'], D_of(r['scale']), sd, n,
                     '%.4f' % r['md'][a] if a in r['md'] else '-.----'))
    A('')
    A('--- CROSS-CHECK: s = 1.0 must equal the EE-vs-Tframe figure at')
    A('    Tframe = 1.0 (same scenario stream, same reporting pipeline);')
    A('    FPA is checked against n_sweep_data.csv (N = 4), because FPA is')
    A('    not in the Tframe figure.')
    if s10:
        worst = 0.0
        for a in ALGOS:
            if a in s10['res'] and a in ref:
                d = abs(s10['res'][a][0] - ref[a][0])
                worst = max(worst, d)
                A('  %-5s  %.6f  vs  %.6f  (|dev| = %.1e)  %s'
                  % (a, s10['res'][a][0], ref[a][0], d,
                     'OK' if d < 5e-4 else 'MISMATCH'))
        A('  worst |dev| = %.1e Mbit/J' % worst)
    A('=' * 78)
    with open(args.report, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))

    if args.write_csv:
        with open(args.out, 'w') as f:
            f.write('algo,scale,ee_mean,ee_std,n,src\n')
            for r in results:
                for a in ALGOS:
                    if a in r['res']:
                        m, sd, n = r['res'][a]
                        f.write('%s,%.1f,%.6f,%.6f,%d,run\n'
                                % (a, r['scale'], m, sd, n))
        print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
