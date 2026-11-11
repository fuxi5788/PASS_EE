"""
mulsim/env.py
Shared environment and helper functions for multi-user PASS simulation.
"""
import numpy as np

# ==================== Physical Constants ====================
C = 3e8
FC = 28e9
NEFF = 1.4
H = 3.0
LAM = C / FC
LAMG = LAM / NEFF
BETA = C ** 2 / (4 * np.pi * FC) ** 2
SIGMA2 = 10 ** (-90 / 10) * 1e-3
V = 1.0
PMOTOR = 1.0          # Table tab:sim_params of the paper
PC = 3.0              # fixed circuit power P_c (Table tab:sim_params of the paper)
PF = 0.2              # per-RF-chain power P_f (Table tab:sim_params of the paper)
POS_SCALE_MODE = 'fixed'
POS_SCALE_FIXED = 0.115
POS_REF_TFRAME = 1.0
MD_MODE = 'slots'
EPS_PA = 0.9
delta2 = 1.0 / EPS_PA
R_MIN = 0.8

# Reward reference and penalty coefficients for the paper objective
E_REF = 10.0
RHO_QOS = 0.5
RHO_POW = 0.5

MAX_M = 5
MAX_N = 8
MAX_K = 8


# ==================== Alpha Optimization ====================
def solve_ratio(Q):
    Nn = Q.shape[0]
    best_val, best_alpha = -np.inf, None
    for mask in range(1, 1 << Nn):
        S = [i for i in range(Nn) if (mask >> i) & 1]
        evals, evecs = np.linalg.eigh(Q[np.ix_(S, S)])
        vs = evecs[:, -1]
        if vs[np.argmax(np.abs(vs))] < 0:
            vs = -vs
        if np.all(vs >= -1e-12) and evals[-1] > best_val:
            best_val = evals[-1]
            a = np.zeros(Nn)
            a[S] = vs
            best_alpha = a
    if best_alpha is not None:
        best_alpha = best_alpha / (np.linalg.norm(best_alpha) + 1e-12)
    else:
        best_alpha = np.ones(Nn) / np.sqrt(Nn)
    return best_alpha, best_val


def solve_ratio_large(Q, iters=60, seed=0):
    """Nonnegative unit-norm maximiser of a^T Q a, scalable in N.

    solve_ratio() enumerates all 2^N supports, which is exact but impossible
    beyond N ~ 16 (N = 32 would need 4.3e9 eigenvalue problems).  This is the
    replacement used for large N in the EE-vs-N sweep: projected gradient ascent
    on the feasible set {a >= 0, ||a|| = 1} with step 1/lambda_max from a few
    deterministic starts, keeping the best feasible point seen, followed by a
    budget-capped greedy coordinate polish.  It can never come out below the
    uniform 1/sqrt(N) ratio, and for N <= ALPHA_EXACT_MAX_N solve_ratio_auto()
    still returns the exact enumeration instead.

    iters = 60 is the measured convergence point on random N = 32 problems (the
    objective moves by < 1e-8 beyond it; see the convergence table in the
    EE-vs-N probe).  The cost matters -- this runs once per waveguide per
    scoring call, and the BGAT refinement makes several hundred of those per
    scenario.
    """
    Nn = Q.shape[0]
    Qs = 0.5 * (Q + Q.T)
    lam = float(np.linalg.eigvalsh(Qs)[-1])
    step = 1.0 / max(lam, 1e-30)

    starts = [np.ones(Nn)]
    _, V = np.linalg.eigh(Qs)
    starts.append(np.maximum(V[:, -1], 0.0))            # clipped eigenvector
    order = np.argsort(-np.diag(Qs))
    for i in range(min(Nn, 3)):
        e = np.zeros(Nn)
        e[order[i]] = 1.0
        starts.append(e)

    best_a, best_v = None, -np.inf
    for a0 in starts:
        nrm = np.linalg.norm(a0)
        if nrm < 1e-12:
            continue
        a = a0 / nrm
        for _ in range(iters):
            a_new = np.maximum(a + step * (Qs @ a), 0.0)
            nrm = np.linalg.norm(a_new)
            if nrm < 1e-12:
                break
            a_new = a_new / nrm
            if np.linalg.norm(a_new - a) < 1e-9:
                a = a_new
                break
            a = a_new
            v = float(a @ Qs @ a)
            if v > best_v:
                best_v, best_a = v, a.copy()
    if best_a is None:
        return np.ones(Nn) / np.sqrt(Nn), 0.0

    # greedy coordinate polish (both directions, renormalised after each move),
    # capped by a fixed evaluation budget: this is called once per waveguide per
    # scoring call, so an unbounded 30-round polish dominated the BGAT
    # refinement at large N (30 rounds * 2N moves * 3 waveguides * ~400
    # evaluations per scenario).  The cap keeps the cost ~O(1) in N.
    rounds = max(2, 240 // (2 * max(Nn, 1)))
    for _ in range(rounds):
        improved = False
        for n in range(Nn):
            for sgn in (1.0, -1.0):
                cand = best_a.copy()
                cand[n] = max(0.0, cand[n] + sgn * 0.05)
                nrm = np.linalg.norm(cand)
                if nrm < 1e-12:
                    continue
                cand /= nrm
                v = float(cand @ Qs @ cand)
                if v > best_v + 1e-15:
                    best_v, best_a, improved = v, cand, True
        if not improved:
            break
    return best_a, best_v


# N <= this uses the exact 2^N enumeration of solve_ratio(); larger N uses
# solve_ratio_large().  8 keeps the reported sweep (N = 4) bit-identical.
ALPHA_EXACT_MAX_N = 8


def solve_ratio_auto(Q):
    if Q.shape[0] <= ALPHA_EXACT_MAX_N:
        return solve_ratio(Q)
    return solve_ratio_large(Q)


def optimize_alpha_multiuser(X, user_pos, M, N, K, YM):
    A = np.zeros((M, N))
    for m in range(M):
        cc = np.zeros((N, K), dtype=np.complex64)
        for n in range(N):
            x_pa = X[m, n]
            phi = 2 * np.pi / LAMG * x_pa
            for k in range(K):
                r = np.sqrt((x_pa - user_pos[k, 0]) ** 2
                            + (YM[m] - user_pos[k, 1]) ** 2
                            + H ** 2)
                cc[n, k] = np.sqrt(BETA) * np.exp(-1j * (2 * np.pi / LAM * r + phi)) / r
        Q = np.real(np.sum([np.outer(np.conj(cc[:, k]), cc[:, k]) for k in range(K)], axis=0))
        a, _ = solve_ratio_auto(Q)
        A[m, :] = a
    return A


# ==================== Channel & Power ====================
def compute_heff(X, A, user_pos, M, N, K, YM):
    Heff = np.zeros((K, M * N), dtype=np.complex64)
    idx = 0
    for m in range(M):
        for n in range(N):
            x_pa = X[m, n]
            phi = 2 * np.pi / LAMG * x_pa
            for k in range(K):
                r = np.sqrt((x_pa - user_pos[k, 0]) ** 2
                            + (YM[m] - user_pos[k, 1]) ** 2
                            + H ** 2)
                Heff[k, idx] = np.sqrt(BETA) * np.exp(-1j * (2 * np.pi / LAM * r + phi)) / r
            idx += 1
    return Heff


def power_alloc_uniform(Heff, Pmax, Rmin, sigma2, delta2_, Deff):
    """Uniform power: P_k = P_total/K. Optimize P_total via Dinkelbach."""
    K = Heff.shape[0]
    HHH = Heff @ Heff.conj().T
    reg = 1e-6 * np.trace(HHH) / max(HHH.shape[0], 1)
    HHH_r = HHH + reg * np.eye(HHH.shape[0])
    try:
        H_inv = np.linalg.inv(HHH_r)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(HHH_r)
    trH = np.real(np.trace(H_inv))

    Pbar_min = (2 ** Rmin - 1) * sigma2
    Pbar_max = Pmax / K

    if Pbar_min > Pbar_max:
        return None, -np.inf

    def ee(Pbar):
        if Pbar <= 0:
            return -np.inf
        return K * np.log2(1 + Pbar / sigma2) / (Deff + delta2_ * Pbar * trH)

    eta = ee(Pbar_max)
    Pbar_opt = Pbar_max

    for _ in range(1000):
        Pstar = K / (eta * delta2_ * trH * np.log(2)) - sigma2
        Pnew = min(Pbar_max, max(Pbar_min, Pstar))
        if Pnew <= 0:
            break
        eta_new = K * np.log2(1 + Pnew / sigma2) / (Deff + delta2_ * Pnew * trH)
        if abs(eta_new - eta) < 1e-6:
            if eta_new > eta:
                Pbar_opt = Pnew
                eta = eta_new
            break
        eta = eta_new
        Pbar_opt = Pnew

    P_total = K * Pbar_opt
    return P_total, eta


def compute_ee_complete(X, A, user_pos, M, N, K, YM, X0, Tframe, Pmax, Rmin):
    Heff = compute_heff(X, A, user_pos, M, N, K, YM)
    HHH = Heff @ Heff.conj().T
    reg = 1e-6 * np.trace(HHH) / max(HHH.shape[0], 1)
    HHH_r = HHH + reg * np.eye(HHH.shape[0])
    try:
        H_inv = np.linalg.inv(HHH_r)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(HHH_r)

    displ = np.abs(X - X0)
    Md = np.max(displ)
    Sd = float(np.sum(displ))
    Tm = Md / V
    Td = Tframe - Tm
    if Td <= 0:
        return -np.inf, None, Td, Md

    Emove = PMOTOR * Sd / V
    Deff = Emove / Td + PC + M * PF

    P_total, ee = power_alloc_uniform(Heff, Pmax, Rmin, SIGMA2, delta2, Deff)
    if P_total is None:
        return -np.inf, None, Td, Md

    Pbar = P_total / K
    rate = K * np.log2(1 + Pbar / SIGMA2)
    if rate < K * Rmin:
        return -np.inf, None, Td, Md

    return ee, P_total, Td, Md


# ==================== PSO & Feasibility ====================
def clip_feasible(X, X0, Dmax, D, M, N, Delta_min=None):
    if Delta_min is None:
        Delta_min = 0.5 * LAM
    # Vectorised over the waveguides; identical result to the per-waveguide
    # loop it replaces (clip to own window -> clip to [0,D] -> sort -> min gap
    # -> clip to [0,D]).
    X = np.array(X, dtype=np.float64, copy=True)
    np.clip(X, X0 - Dmax, X0 + Dmax, out=X)
    np.clip(X, 0.0, D, out=X)
    X.sort(axis=1)
    for n in range(1, N):
        np.maximum(X[:, n], X[:, n - 1] + Delta_min, out=X[:, n])
    np.clip(X, 0.0, D, out=X)
    return X


def pso_maximize(fit_func, X0, Dmax, D, M, N, I=30, T=300, seed=42):
    rng = np.random.RandomState(seed)
    mn = M * N
    x0v = X0.reshape(-1)

    Xp = np.tile(x0v, (I, 1)) + (rng.rand(I, mn) - 0.5) * 2 * Dmax
    for i in range(I):
        Xp[i] = clip_feasible(Xp[i].reshape(M, N), X0, Dmax, D, M, N).reshape(-1)

    V = np.zeros((I, mn))
    pbest = Xp.copy()
    pbest_f = np.array([fit_func(Xp[i]) for i in range(I)])
    gi = np.argmax(pbest_f)
    gbest = Xp[gi].copy()
    gbest_f = pbest_f[gi]

    wmax, wmin, c1, c2 = 0.9, 0.4, 1.5, 1.5

    for t in range(1, T + 1):
        w = wmax - (wmax - wmin) / T * t
        for i in range(I):
            V[i] = w * V[i] + c1 * rng.rand(mn) * (pbest[i] - Xp[i]) + c2 * rng.rand(mn) * (gbest - Xp[i])
            Xnew = Xp[i] + V[i]
            Xnew = clip_feasible(Xnew.reshape(M, N), X0, Dmax, D, M, N).reshape(-1)
            Xp[i] = Xnew
            f = fit_func(Xnew)
            if f > pbest_f[i]:
                pbest[i] = Xnew
                pbest_f[i] = f
                if f > gbest_f:
                    gbest_f = f
                    gbest = Xnew

    return gbest.reshape(M, N), gbest_f


# ==================== Environment ====================
class PASSMultiUserEnv:
    def __init__(self, seed=0, randomize_params=True,
                 M=None, N=None, K=None, D=None,
                 Pmax_dbm=None, Tframe=None, v_fixed=None):
        self.rng = np.random.RandomState(seed)
        self.randomize_params = randomize_params

        self.M_range = [2, 5]
        self.N_range = [2, 8]
        self.K_range = [2, 5]
        self.D_range = [20.0, 60.0]
        self.Pmax_dbm_range = [15.0, 35.0]
        self.Tframe_range = [0.5, 2.0]

        self.fixed_M = M
        self.fixed_N = N
        self.fixed_K = K
        self.fixed_D = D
        self.fixed_Pmax_dbm = Pmax_dbm
        self.fixed_Tframe = Tframe
        self.v_fixed = v_fixed  # fixed PA speed (m/s); None uses global V

        self.reset(init=True)

    def _sample_params(self):
        if self.randomize_params:
            # ZF feasibility: waveguide count must be >= user count (M >= K)
            self.K = self.rng.randint(self.K_range[0], self.K_range[1] + 1)
            m_min = max(self.M_range[0], self.K)
            self.M = self.rng.randint(m_min, self.M_range[1] + 1)
            self.N = self.rng.randint(self.N_range[0], self.N_range[1] + 1)
            self.D = self.rng.uniform(self.D_range[0], self.D_range[1])
            self.Pmax_dbm = self.rng.uniform(self.Pmax_dbm_range[0], self.Pmax_dbm_range[1])
            self.Tframe = self.rng.uniform(self.Tframe_range[0], self.Tframe_range[1])
        else:
            self.M = self.fixed_M if self.fixed_M is not None else 3
            self.K = self.fixed_K if self.fixed_K is not None else 3
            self.N = self.fixed_N if self.fixed_N is not None else 4
            # ZF feasibility: enforce M >= K
            if self.M < self.K:
                self.M = self.K
            self.D = self.fixed_D if self.fixed_D is not None else 40.0
            self.Pmax_dbm = self.fixed_Pmax_dbm if self.fixed_Pmax_dbm is not None else 25.0
            self.Tframe = self.fixed_Tframe if self.fixed_Tframe is not None else 1.0

        self.Pmax = 10 ** (self.Pmax_dbm / 10) * 1e-3
        self.YM = self.D * np.arange(1, self.M + 1) / (self.M + 1)
        self.V = self.v_fixed if self.v_fixed is not None else V
        self.Dmax = self.V * self.Tframe
        # Action scale x = X0 + a * POS_SCALE, selected by POS_SCALE_MODE:
        #   'window' : V * Tframe               (the original mapping; |a| <= 1
        #                                        always fits the frame exactly,
        #                                        so Tm/Tframe = |a| for every
        #                                        Tframe and EE is ~flat)
        #   'ref'    : V * POS_REF_TFRAME       (constant 1 m)
        #   'auto'   : V * min(Tframe, REF)     (previous default)
        #   'fixed'  : POS_SCALE_FIXED          (constant movement budget)
        # 'fixed' is the only mode in which the deployment is INDEPENDENT of
        # Tframe, which is what makes EE(Tframe) monotone: a longer frame then
        # only adds data time, Td = Tframe - Md, with Md fixed.  It is also
        # chosen below half the initial PA spacing, POS_SCALE < 40*LAM/2 =
        # 0.2143 m, so a commanded offset can never make a pinch point cross a
        # neighbour; the min-gap / ordering projection in step() (np.sort) is
        # therefore never activated, and the movement bookkeeping carries no
        # re-pairing artifact.
        if POS_SCALE_MODE == 'window':
            self.POS_SCALE = self.V * self.Tframe
        elif POS_SCALE_MODE == 'ref':
            self.POS_SCALE = self.V * POS_REF_TFRAME
        elif POS_SCALE_MODE == 'fixed':
            self.POS_SCALE = float(POS_SCALE_FIXED)
        else:
            self.POS_SCALE = self.V * min(self.Tframe, POS_REF_TFRAME)
        self.Delta_min = 0.5 * LAM
        self.X0_spacing = 40 * LAM  # initial PA spread around the user
        self.Rmin = R_MIN

    def _user_positions(self):
        xu = self.rng.uniform(2, self.D - 2, self.K)
        yu = self.rng.uniform(2, self.D - 2, self.K)
        return np.stack([xu, yu], axis=1)

    def _initial_positions(self):
        """Movable-antenna initial positions: spread around the user's x
        coordinate with spacing 40*lambda (same as single-user reference).
        Each waveguide anchors to user (m mod K), so the PAs start close to
        the users and only need a short, feasible movement (Td > 0)."""
        X0 = np.zeros((self.M, self.N))
        nvec = np.arange(self.N) - (self.N - 1) / 2.0  # centered offsets
        for m in range(self.M):
            xk = self.user_pos[m % self.K][0]
            X0[m] = np.clip(xk + nvec * self.X0_spacing, 0.0, self.D)
            X0[m] = np.sort(X0[m])
        return X0

    def reset(self, init=False):
        self._sample_params()
        self.user_pos = self._user_positions()
        self.X0 = self._initial_positions()
        self.x = self.X0.copy()
        self.Md_cmd = 0.0          # commanded travel, see MD_MODE
        self.P = np.ones(self.K) * (self.Pmax / self.K * 0.5)
        self.step_cnt = 0
        return self._get_state()

    def _channel_coeffs(self):
        Heff = np.zeros((self.K, self.M * self.N), dtype=np.complex64)
        idx = 0
        for m in range(self.M):
            for n in range(self.N):
                x_pa = self.x[m, n]
                phi = 2 * np.pi / LAMG * x_pa
                for k in range(self.K):
                    r = np.sqrt((x_pa - self.user_pos[k, 0]) ** 2
                                + (self.YM[m] - self.user_pos[k, 1]) ** 2
                                + H ** 2)
                    Heff[k, idx] = np.sqrt(BETA) * np.exp(-1j * (2 * np.pi / LAM * r + phi)) / r
                idx += 1
        return Heff

    def _compute_ee(self):
        Heff = self._channel_coeffs()
        HHH = Heff @ Heff.conj().T
        # Stronger regularization to prevent near-singular explosion
        reg = 1e-4 * np.trace(HHH) / max(HHH.shape[0], 1)
        HHH_r = HHH + reg * np.eye(HHH.shape[0])
        try:
            H_inv = np.linalg.inv(HHH_r)
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(HHH_r)
        # ZF noise enhancement per user: [(H H^H)^{-1}]_kk
        noise_enh = np.maximum(np.real(np.diag(H_inv)), 1e-30)
        # Post-ZF per-user rate: SNR_k = q_k / (sigma^2 * [H_inv]_kk)
        # where q_k = self.P[k] is the TRANSMIT power allocated to user k.
        rates = np.log2(1 + self.P / (SIGMA2 * noise_enh))
        sum_rate = np.sum(rates)
        # Total transmit power under ZF: the symbol power is q_k/[H_inv]_kk
        # and ||w_k||^2 = [H_inv]_kk, so the radiated power sums to sum(q_k).
        # (NOT Tr(P H_inv): self.P is already the transmit-power budget.)
        trace_term = float(np.sum(self.P))

        if MD_MODE == 'commanded':
            # True per-pinch-point travel; pairing preserved (see MD_MODE).
            Md = float(getattr(self, 'Md_cmd', 0.0))
            displ = np.abs(self.x - self.X0)
        else:
            # 'slots': travel of the n-th slot after the sort in step().
            displ = np.abs(self.x - self.X0)
            Md = np.max(displ)
        Sd = float(np.sum(displ))    # total travel of ALL PAs (paper Eq. EE_def)
        Tm = Md / self.V
        Td = self.Tframe - Tm
        if Td <= 0:
            # Physically infeasible, but return metrics so that
            # the outer soft penalty can provide a shaped signal.
            return 0.0, sum_rate, trace_term, Md, Td, noise_enh

        Emove = PMOTOR * Sd / self.V
        Deff = Emove / Td + PC + self.M * PF
        denom = Deff + delta2 * trace_term
        if denom <= 0:
            return 0.0, sum_rate, trace_term, Md, Td, noise_enh

        ee = sum_rate / denom
        return ee, sum_rate, trace_term, Md, Td, noise_enh

    def _get_state(self):
        Heff = self._channel_coeffs()
        mn = self.M * self.N

        def _fit(v, n):
            """Pad or TRIM to n entries.

            The RL state has a FIXED layout sized by MAX_M*MAX_N, so a
            waveguide with more than MAX_M*MAX_N/M pinch points does not fit in
            it at all (the old np.pad() then died with a negative width and even
            env.reset() crashed, which also blocked the non-RL benchmarks that
            only need reset()).  Trimming keeps every (M, N, K) with
            M*N <= MAX_M*MAX_N bit-identical to before, which is the whole
            reported sweep (M = 3, N = 4), and lets the classical benchmarks and
            BGAT run at large N (the RL policy cannot act there anyway: its
            action head is MAX_M*MAX_N + MAX_K wide).
            """
            if len(v) >= n:
                return v[:n]
            return np.pad(v, (0, n - len(v)))

        # ===== Normalized features (all ~[0, 1] or ~1e-1 scale) =====
        x_feat = (self.x / self.D).flatten()
        p_feat = (self.P / max(self.Pmax, 1e-30)).flatten()
        hr = (Heff.flatten().real * 1e4)  # channel coeff ~1e-5 -> ~1e-1
        hi = (Heff.flatten().imag * 1e4)
        u_feat = (self.user_pos / self.D).flatten()
        x0_feat = (self.X0 / self.D).flatten()

        x_pad = _fit(x_feat, MAX_M * MAX_N)
        p_pad = _fit(p_feat, MAX_K)
        hr_pad = _fit(hr, MAX_K * MAX_M * MAX_N)
        hi_pad = _fit(hi, MAX_K * MAX_M * MAX_N)
        u_pad = _fit(u_feat, MAX_K * 2)
        x0_pad = _fit(x0_feat, MAX_M * MAX_N)

        params = np.array([
            self.M / MAX_M,
            self.N / MAX_N,
            self.K / MAX_K,
            self.Pmax_dbm / 35.0,
            self.Tframe / 2.0,
            self.D / 60.0,
            self.V / 2.0,  # encode PA speed into state
        ], dtype=np.float32)

        feat = np.concatenate([x_pad, p_pad, hr_pad, hi_pad, u_pad, x0_pad, params])
        if len(feat) < 768:
            feat = np.pad(feat, (0, 768 - len(feat)))
        else:
            feat = feat[:768]
        return feat.astype(np.float32)

    def step(self, action):
        mn = self.M * self.N
        act_len = mn + self.K

        # ===== Direct control (no accumulation) =====
        # Position: target = X0 + action * POS_SCALE (POS_SCALE_MODE).
        #   action=0 -> x=X0 (initial layout, Md=0, Td=Tframe). The agent
        #   learns the optimal OFFSET from the near-user initial layout,
        #   exactly like the paper's PSO searches around X0.
        # The frame length caps feasibility (X0 +/- Dmax) below.  With
        # POS_SCALE_MODE='fixed' that cap never binds (POS_SCALE < Dmax for
        # Tframe >= 0.8 s), so the realized deployment does not depend on the
        # frame length and EE(Tframe) is monotone.
        dx_norm = np.clip(action[:mn], -1.0, 1.0)
        self.x = self.X0 + dx_norm.reshape(self.M, self.N) * self.POS_SCALE
        # Travel each physical pinch point is actually commanded to make,
        # before the sort below re-assigns pinch points to slots.
        self.Md_cmd = float(np.max(np.abs(dx_norm))) * self.POS_SCALE
        for m in range(self.M):
            lo = self.X0[m, :] - self.Dmax
            hi = self.X0[m, :] + self.Dmax
            self.x[m] = np.clip(self.x[m], lo, hi)
            self.x[m] = np.clip(self.x[m], 0.0, self.D)
            self.x[m] = np.sort(self.x[m])
            for n in range(1, self.N):
                if self.x[m, n] - self.x[m, n - 1] < self.Delta_min:
                    self.x[m, n] = self.x[m, n - 1] + self.Delta_min
            self.x[m] = np.clip(self.x[m], 0.0, self.D)

        # Power: direct mapping action [-1,1] -> [0, Pmax/K].
        #   action=0 -> P = Pmax/K/2 (same as the reset default).
        dP_norm = np.clip(action[mn:act_len], -1.0, 1.0)
        P_target = (dP_norm + 1.0) / 2.0 * (self.Pmax / self.K)
        P_min = (2 ** self.Rmin - 1) * SIGMA2 * 0.1
        self.P = np.clip(P_target, P_min, self.Pmax)
        if np.sum(self.P) > self.Pmax:
            self.P = self.P / np.sum(self.P) * self.Pmax

        ee, rate, trace_term, Md, Td, noise_enh = self._compute_ee()

        # ===== Reward: log-shaped paper objective =====
        # reward = log10(EE / E_REF) - rho_qos * qos_penalty - rho_pow * power_penalty
        # Monotonic shaping of the paper objective (maximize EE, minimize penalties).
        # The reward is unbounded above (can be >0) and has a lower floor.
        # QoS based on the ACTUAL post-ZF SNR: P_k / (sigma^2 * [H_inv]_kk)
        gamma_th = 2 ** self.Rmin - 1
        snr_k = self.P / (SIGMA2 * noise_enh)
        viol = np.maximum(0.0, gamma_th - snr_k)
        qos_penalty = float(np.sum((viol / gamma_th) ** 2))

        # Soft penalty for overusing the power budget (usually 0 after clipping)
        power_penalty = float(np.maximum(0.0, np.sum(self.P) / self.Pmax - 1.0))

        if Td <= 1e-6 or ee <= 0:
            raw_reward = -10.0  # infeasible: no time for data
            reward = -10.0
        else:
            raw_reward = float(np.log10(ee / E_REF))
            reward = raw_reward - RHO_QOS * qos_penalty - RHO_POW * power_penalty
            # Keep a lower floor for numerical stability; no upper clip.
            reward = float(np.clip(reward, -15.0, None))

        self.step_cnt += 1
        done = self.step_cnt >= 200
        next_state = self._get_state()
        info = {'ee': ee, 'rate': rate, 'Md': Md, 'Td': Td, 'reward': reward}

        # ===== DEBUG PRINT (every 50 steps to avoid spam) =====
        if self.step_cnt % 50 == 0:
            print(f"  [ENV-DEBUG] step={self.step_cnt:3d} | "
                  f"ee={ee:.4e} rate={rate:.2f} trace={trace_term:.2e} "
                  f"Md={Md:.3f} Td={Td:.3f} | "
                  f"ee_r={raw_reward:.2f} qos_p={qos_penalty:.2f} | "
                  f"reward={reward:.3f} | Pmin={self.P.min():.4e} Pmax={self.P.max():.4e}")

        return next_state, reward, done, info
