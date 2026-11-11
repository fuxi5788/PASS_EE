"""EE versus Tframe, with the deployment planned ONCE at the nominal frame.

Scheme A ("deployment is a network-planning decision; the frame length is a
scheduling parameter"): for ADMM / DSD / BGAT the position search runs ONCE at
the nominal frame T0 = 1.0 s, the resulting deployment X* is then held FIXED,
and EE(T) is re-evaluated for every Tframe by changing only the time split

    T_m = M_d / V  (fixed, from X*),   T_d = Tframe - T_m,
    D_eff = P_motor*T_m/T_d + PF

and re-solving the transmit sub-problem (ZF + KKT/Dinkelbach) for that D_eff.
The rate therefore does NOT depend on Tframe for any scheme, and every curve
shows the same 'rise then saturate' shape as GNN-SAC -- whose trained policy
already behaves exactly like this (its run log shows M_d = 0.113 m at every
Tframe, so its own curve is the movement-amortisation term alone:
EE(T) = 38.25/(3.53 + 0.113/(T-0.113)), reproducing the measured column to
within 0.003 Mbit/J).

GNN-SAC is not re-run here: its published column already satisfies Scheme A.
Scenario stream: env seed 32, n_mc 100, M = 3, N = 4, K = 3, D = 40 m,
Pmax = 25 dBm; ADMM budget 400, DSD budget 200; BGAT = models/bgat_best.pt
inference + the dual-start refinement.

Output: tf_sweep_fixeddep.csv (algo,Tframe,ee_mean,ee_std,n) and a text report.
"""
import argparse
import time
from datetime import datetime

import numpy as np

T0 = 1.0
TF = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
ALGOS = ['ADMM', 'DSD', 'BGAT']
CSV = 'tf_sweep_fixeddep.csv'
REPORT = 'tf_sweep_fixeddep_report.txt'
ADMM_BUDGET = 400
DSD_BUDGET = 200
SEED = 32
N_MC = 1000


def make_env(T):
    import env as E
    from env import PASSMultiUserEnv
    E.PMOTOR = 1.0
    import benchmark_traditional as BT
    BT.PMOTOR = 1.0
    return PASSMultiUserEnv(seed=SEED, randomize_params=False, M=3, N=4, K=3,
                            D=40.0, Pmax_dbm=25.0, Tframe=T)


def run(env, n_mc):
    """One pass: per scenario, plan the deployment at T0 then evaluate all T."""
    import os
    import benchmark_traditional as BT
    from env import optimize_alpha_multiuser
    from train_bgat import load_bgat
    # The trained checkpoint lives in <project root>/models, i.e. ../models
    # relative to mulsim/ where this script runs.  load_bgat() SILENTLY falls
    # back to a randomly initialised network when the file is missing, which
    # silently produced a bogus BGAT column once already -- so verify the file
    # exists and abort rather than publish numbers from an untrained network.
    model_dir = 'models' if os.path.isdir('models') else '../models'
    _ckpt = os.path.join(model_dir, 'bgat_best.pt')
    if not os.path.exists(_ckpt):
        raise FileNotFoundError('BGAT checkpoint not found: %s' % _ckpt)
    model = load_bgat(model_dir=model_dir, device='cpu')
    BT.WSUB_BUDGET = ADMM_BUDGET
    BT.DSD_WSUB_BUDGET = DSD_BUDGET
    out = {a: {T: [] for T in TF} for a in ALGOS}
    mds = {a: [] for a in ALGOS}
    for i in range(n_mc):
        env.reset()                      # scenario starts at T0 = 1.0 s
        for a in ALGOS:
            # Plan at the NOMINAL frame.  env.reset() restores Tframe = T0, but
            # only the FIRST algorithm of a scenario benefits from it: the
            # previous algorithm's T-loop leaves env.Tframe at the last grid
            # point (1.4 s), and reset() is not called again until the next
            # scenario.  Without this line DSD / BGAT searched their deployment
            # at Tframe = 1.4 instead of the documented T0 = 1.0 (measured
            # effect: ~1 marginal scenario in 100 flips, +-0.023 Mbit/J on the
            # mean, which broke the bit-level agreement with the EE-vs-area
            # figure at s = 1.0).
            env.Tframe = T0
            if a == 'ADMM':
                _, X = BT.admm_benchmark(env, return_X=True)
            elif a == 'DSD':
                _, X = BT.dsd_benchmark(env, return_X=True)
            else:
                _, X = BT.bgat_benchmark(env, model=model, device='cpu',
                                         return_X=True)
            X = np.asarray(X, float)
            A = optimize_alpha_multiuser(X, env.user_pos, env.M, env.N,
                                         env.K, env.YM)
            Heff = BT._dsd_heff(X, A, env)
            md = float(np.max(np.abs(X - env.X0)))
            mds[a].append(md)
            for T in TF:
                env.Tframe = T
                Deff = BT._deff(X, env, X_ref=env.X0)  # movement vs X0
                if Deff is None:
                    continue
                ee, _ = BT._solve_wsub(Heff, env, Deff)
                if np.isfinite(ee):
                    out[a][T].append(ee)
    return out, mds


def main():
    global N_MC
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_mc', type=int, default=N_MC)
    ap.add_argument('--report', type=str, default=REPORT)
    ap.add_argument('--out', type=str, default=CSV)
    ap.add_argument('--write-csv', action='store_true')
    args = ap.parse_args()
    n_mc = args.n_mc

    t0 = time.perf_counter()
    print('planning at the nominal frame T0 = %.1f s, then evaluating %d '
          'Tframe points (n_mc = %d, seed %d)' % (T0, len(TF), n_mc, SEED),
          flush=True)
    env = make_env(T0)
    res, mds = run(env, n_mc)

    L = []

    def A(s):
        L.append(s)

    A('=' * 78)
    A('TF_SWEEP with PLANNED DEPLOYMENTS (Scheme A)   --   PMOTOR = 1.0 W')
    A('=' * 78)
    A('generated : %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    A('method    : ADMM / DSD / BGAT search their deployment ONCE at the')
    A('            nominal frame T0 = 1.0 s; the deployment is held fixed and')
    A('            EE(T) is re-evaluated with only the time split changing')
    A('            (T_m fixed, T_d = T - T_m), transmit sub-problem re-solved.')
    A('            GNN-SAC already satisfies this (M_d = 0.113 m at every T);')
    A('            its column is taken from tf_sweep_data.csv unchanged.')
    A('settings  : Tframe grid = %s s' % ', '.join('%.1f' % t for t in TF))
    A('            n_mc %d, env seed %d, M = 3, N = 4, K = 3, D = 40 m,' %
      (n_mc, SEED))
    A('            Pmax = 25 dBm; ADMM budget %d, DSD budget %d'
      % (ADMM_BUDGET, DSD_BUDGET))
    A('')
    A('--- RESULTS (mean EE, Mbit/J) -------------------------------------')
    A('%-8s %s' % ('Tframe', ''.join('%-12s' % a for a in ALGOS)))
    for T in TF:
        A('%-8.1f %s' % (T, ''.join('%-12.4f' % np.mean(res[a][T])
                                    for a in ALGOS)))
    A('')
    A('std / n:')
    for a in ALGOS:
        A('  %-5s std = [%s]  n = [%s]'
          % (a, ' '.join('%.4f' % np.std(res[a][T]) for T in TF),
             ' '.join('%d' % len(res[a][T]) for T in TF)))
    A('')
    A('--- mean M_d (m), fixed by construction ---------------------------')
    for a in ALGOS:
        A('  %-5s M_d = %.4f' % (a, float(np.mean(mds[a])) if mds[a] else 0.0))
    A('')
    A('--- TREND ---------------------------------------------------------')
    A('%-7s %-10s %-11s' % ('scheme', 'range', '+%'))
    for a in ALGOS:
        y = np.array([np.mean(res[a][T]) for T in TF])
        A('%-7s %-10.4f %-11.2f' % (a, y.max() - y.min(),
                                    100 * (y.max() - y.min()) / y.min()))
    A('=' * 78)
    with open(args.report, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))

    if args.write_csv:
        with open(args.out, 'w') as f:
            f.write('algo,Tframe,ee_mean,ee_std,n\n')
            for a in ALGOS:
                for T in TF:
                    f.write('%s,%.1f,%.6f,%.6f,%d\n'
                            % (a, T, np.mean(res[a][T]), np.std(res[a][T]),
                               len(res[a][T])))
        print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
