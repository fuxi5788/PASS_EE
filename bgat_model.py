"""
mulsim/bgat_model.py
Bipartite Graph Attention Network (BGAT) for multi-user PASS EE maximization.
Based on: X. Xie, Y. Lu, and Z. Ding, "Graph Neural Network Enabled Pinching Antennas",
         arXiv:2502.05447, 2025.

Adapted to multi-waveguide scenario:
- Antenna nodes: M*N pinching antennas (flattened)
- User nodes: K single-antenna users
- Bipartite edges: fully connected between user and antenna nodes
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from env import (
    compute_heff, power_alloc_uniform, clip_feasible,
    optimize_alpha_multiuser,
    C, FC, NEFF, H, LAM, LAMG, BETA, SIGMA2,
    V, PMOTOR, PF, delta2, R_MIN, MAX_M, MAX_N, MAX_K
)


def _build_graph_features(X, user_pos, M, N, K, YM, Pmax):
    """
    Build bipartite graph node and edge features.
    Returns:
        user_feats: (K, 2)  -- user coordinates [x, y]
        ant_feats:  (M*N, 2) -- [interval_approx, power_init]
        edge_feats: (K, M*N) -- distances l_{k,mn}
    """
    # User node features: coordinates
    user_feats = user_pos[:, :2].copy()  # (K, 2)

    # Antenna node features: position-based interval + initial power
    mn = M * N
    ant_feats = np.zeros((mn, 2), dtype=np.float32)
    idx = 0
    for m in range(M):
        for n in range(N):
            x_pa = X[m, n]
            # Feature 1: normalized position
            ant_feats[idx, 0] = x_pa / 60.0
            # Feature 2: initial power guess
            ant_feats[idx, 1] = Pmax / mn
            idx += 1

    # Edge features: distances between users and antennas
    edge_feats = np.zeros((K, mn), dtype=np.float32)
    idx = 0
    for m in range(M):
        for n in range(N):
            x_pa = X[m, n]
            for k in range(K):
                dist = np.sqrt((x_pa - user_pos[k, 0])**2
                               + (YM[m] - user_pos[k, 1])**2 + H**2)
                edge_feats[k, idx] = dist
            idx += 1

    return user_feats, ant_feats, edge_feats


class GATLayer(nn.Module):
    """Single GAT layer with multi-head attention for bipartite graph."""
    def __init__(self, in_dim_u, in_dim_a, edge_dim, out_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // n_heads

        # Linear transforms for source/target nodes
        self.W_src_u = nn.Linear(in_dim_u, out_dim)
        self.W_tgt_a = nn.Linear(in_dim_a, out_dim)
        self.W_src_a = nn.Linear(in_dim_a, out_dim)
        self.W_tgt_u = nn.Linear(in_dim_u, out_dim)

        # Edge transform
        self.W_e = nn.Linear(edge_dim, out_dim)

        # Attention vector
        self.att_vec = nn.Parameter(torch.Tensor(1, n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.att_vec)

    def forward(self, user_feats, ant_feats, edge_feats):
        """
        user_feats: (B, K, in_dim_u)
        ant_feats:  (B, M*N, in_dim_a)
        edge_feats: (B, K, M*N, edge_dim)
        Returns updated user_feats, ant_feats
        """
        B, K, _ = user_feats.shape
        MN = ant_feats.shape[1]

        # Transform
        hu_src = self.W_src_u(user_feats).view(B, K, self.n_heads, self.head_dim)
        ha_tgt = self.W_tgt_a(ant_feats).view(B, MN, self.n_heads, self.head_dim)
        ha_src = self.W_src_a(ant_feats).view(B, MN, self.n_heads, self.head_dim)
        hu_tgt = self.W_tgt_u(user_feats).view(B, K, self.n_heads, self.head_dim)

        # Edge transform: (B, K, MN, out_dim) -> (B, K, MN, n_heads, head_dim)
        he = self.W_e(edge_feats).view(B, K, MN, self.n_heads, self.head_dim)

        # ---- User -> Antenna attention ----
        # hu_src: (B, K, H, D), ha_tgt: (B, MN, H, D)
        # Broadcast: hu_src[:, :, None, :, :] + ha_tgt[:, None, :, :, :]
        score_ua = hu_src[:, :, None, :, :] + ha_tgt[:, None, :, :, :]  # (B, K, MN, H, D)
        score_ua = score_ua + he
        score_ua = F.leaky_relu(score_ua, negative_slope=0.2)
        attn_ua = (self.att_vec * score_ua).sum(dim=-1, keepdim=True)  # (B, K, MN, H, 1)
        attn_ua = F.softmax(attn_ua, dim=1)  # softmax over users

        # Aggregate for antennas
        hu_exp = hu_src[:, :, None, :, :].expand(B, K, MN, self.n_heads, self.head_dim)
        agg_a = (attn_ua * hu_exp).sum(dim=1)  # (B, MN, H, D)

        # ---- Antenna -> User attention ----
        score_au = ha_src[:, None, :, :, :] + hu_tgt[:, :, None, :, :]  # (B, K, MN, H, D)
        score_au = score_au + he
        score_au = F.leaky_relu(score_au, negative_slope=0.2)
        attn_au = (self.att_vec * score_au).sum(dim=-1, keepdim=True)  # (B, K, MN, H, 1)
        attn_au = F.softmax(attn_au, dim=2)  # softmax over antennas

        ha_exp = ha_src[:, None, :, :, :].expand(B, K, MN, self.n_heads, self.head_dim)
        agg_u = (attn_au * ha_exp).sum(dim=2)  # (B, K, H, D)

        # Reshape back
        new_user = agg_u.view(B, K, self.out_dim)
        new_ant = agg_a.view(B, MN, self.out_dim)
        return new_user, new_ant


class BGATBlock(nn.Module):
    """One BGAT block: GAT + MLP + residual."""
    def __init__(self, feat_dim, edge_dim, n_heads=4, mlp_hidden=32):
        super().__init__()
        self.gat = GATLayer(feat_dim, feat_dim, edge_dim, feat_dim, n_heads)
        self.mlp_u = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, feat_dim),
            nn.ReLU(),
        )
        self.mlp_a = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, feat_dim),
            nn.ReLU(),
        )
        self.res_u = nn.Linear(feat_dim, feat_dim)
        self.res_a = nn.Linear(feat_dim, feat_dim)

    def forward(self, user_feats, ant_feats, edge_feats):
        u1, a1 = self.gat(user_feats, ant_feats, edge_feats)
        u2 = self.mlp_u(u1) + self.res_u(user_feats)
        a2 = self.mlp_a(a1) + self.res_a(ant_feats)
        return F.relu(u2), F.relu(a2)


class BGAT(nn.Module):
    """
    Bipartite Graph Attention Network for PASS EE maximization.
    Outputs: PA positions X (satisfying spacing/range) and power allocation P.
    """
    def __init__(self, n_blocks=5, feat_dim=16, n_heads=4, mlp_hidden=32):
        super().__init__()
        self.n_blocks = n_blocks
        self.feat_dim = feat_dim

        # Embedding layers
        self.embed_u = nn.Linear(2, feat_dim)
        self.embed_a = nn.Linear(2, feat_dim)
        self.embed_e = nn.Linear(3, 1)          # distance + cos/sin of the phase

        # BGAT blocks
        self.blocks = nn.ModuleList([
            BGATBlock(feat_dim, 1, n_heads, mlp_hidden)
            for _ in range(n_blocks)
        ])

        # Readout NN1: positions (displacements around the initial deployment)
        self.readout_pos = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 1),
        )
        # Global displacement scale in metres: the head predicts the PATTERN,
        # this scalar the magnitude, so the optimiser can slide a whole row by a
        # few cm without having to push every readout weight (the objective
        # oscillates on the lambda = 1 cm scale, which froze the head at X0).
        # The head itself keeps its random init: BGAT optimises the deployment
        # WITHOUT charging the movement energy, so its optimum sits 5-20 cm away
        # from X0 and initialising the readout at zero would pin it to X0.
        self.pos_scale = nn.Parameter(torch.tensor(0.05))

        # Readout NN2: power
        self.readout_pwr = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 1),
        )

    def forward_batch(self, X0, user_pos, YM, Pmax, D, Dmax, Delta_min, M, N):
        """
        Batched and DIFFERENTIABLE form of forward(): the output X is built with
        torch ops on the graph, so loss.backward() reaches every parameter.
        The graph features depend on X0 alone, hence carry no gradient.

        X0:        (B, M, N) initial positions
        user_pos:  (B, K, 2)
        YM:        (B, M)
        Pmax, D, Dmax, Delta_min: (B,) -- may differ within the batch
        M, N:      ints, shared by the whole batch

        Returns: X (B, M, N) torch, P (B, K) torch.
        """
        device = next(self.parameters()).device
        B = X0.shape[0]
        K = user_pos.shape[1]
        mn = M * N

        def _t(a):
            if torch.is_tensor(a):
                return a.to(dtype=torch.float32, device=device)
            a = np.asarray(a, dtype=np.float32)
            if a.ndim == 0:
                a = a[None]
            return torch.as_tensor(a, dtype=torch.float32, device=device)

        X0t, ut, ymt = _t(X0), _t(user_pos), _t(YM)
        pmt, dt, dmt, dlt = _t(Pmax), _t(D), _t(Dmax), _t(Delta_min)

        # --- graph features (same definitions as _build_graph_features) ---
        ant = torch.empty(B, mn, 2, device=device)
        ant[:, :, 0] = X0t.reshape(B, mn) / 60.0
        ant[:, :, 1] = (pmt / mn)[:, None]
        xr = X0t.reshape(B, 1, mn)                             # (B, 1, MN)
        ymn = ymt.repeat_interleave(N, dim=1).unsqueeze(1)     # (B, 1, MN)
        ux = ut[:, :, 0].unsqueeze(2)                          # (B, K, 1)
        uy = ut[:, :, 1].unsqueeze(2)
        edge_r = torch.sqrt((xr - ux) ** 2 + (ymn - uy) ** 2 + H * H)
        # The edge must carry the CHANNEL, not just the distance: with r alone
        # the network cannot represent the 2*pi*r/LAM phase (an MLP cannot fit a
        # function with a 1 cm period), and that phase is exactly what decides
        # where a pinch point has to sit.  cos/sin make the alignment rule
        # learnable, which is what lets the deployment beat X0 at all.
        edge_ph = (2 * np.pi / LAM) * edge_r + (2 * np.pi / LAMG) * xr
        edge = torch.stack([edge_r / 60.0, torch.cos(edge_ph),
                            torch.sin(edge_ph)], dim=-1)       # (B,K,MN,3)

        # --- embed + blocks ---
        uf = F.relu(self.embed_u(ut[:, :, :2].contiguous()))
        af = F.relu(self.embed_a(ant))
        ef = self.embed_e(edge)
        for block in self.blocks:
            uf, af = block(uf, af, ef)

        # --- position readout ---
        # Signed displacement per pinch point: tanh keeps the gradient alive at
        # initialisation (with relu the head sat at relu(0) = 0, the derivative
        # was zero and the positions never changed), pos_scale sets the metric
        # magnitude, and the row is kept inside its travel budget
        # Bmax = 2*Dmax - (N-1)*Delta_min.
        raw = torch.tanh(self.readout_pos(af).squeeze(-1))             # (B, MN)
        d = (self.pos_scale * raw).view(B, M, N)   # SIGNED displacement per PA
        # Row travel budget Bmax = 2*Dmax - (N-1)*Delta_min, applied to
        # sum_n |d_n|, the budget the other schemes respect.  The cumulative
        # (row shift/stretch) form used before could not express what actually
        # pays -- moving individual pinch points left or right to align their
        # phases -- so the network could only make the EE worse than X0.
        Bmax = 2 * dmt - (N - 1) * dlt if N > 1 else 2 * dmt
        Bmax = torch.where(Bmax > 0, Bmax, dmt)
        tot = d.abs().sum(dim=2, keepdim=True)                         # (B,M,1)
        d = torch.where(tot > Bmax[:, None, None],
                        d / (tot + 1e-12) * Bmax[:, None, None], d)

        # The network output is a DISPLACEMENT around the initial deployment:
        #   X[:, :, n] = X0[:, :, n] + d_n,
        # so d = 0 reproduces X0 exactly.
        X = X0t + d

        # Projection of env.clip_feasible: own window -> [0, D] -> min gap ->
        # [0, D].  (The sort inside clip_feasible is a no-op: the max-with-
        # previous step below already restores a non-decreasing row with every
        # gap >= Delta_min.)
        zero = torch.zeros((), device=device)
        lo = torch.minimum(torch.maximum(X0t - dmt[:, None, None], zero),
                           dt[:, None, None])
        hi = torch.minimum(torch.maximum(X0t + dmt[:, None, None], zero),
                           dt[:, None, None])
        X = torch.minimum(torch.maximum(X, lo), hi)
        for n in range(1, N):
            X = torch.cat([X[:, :, :n],
                           torch.maximum(X[:, :, n:n + 1],
                                         X[:, :, n - 1:n] + dlt[:, None, None]),
                           X[:, :, n + 1:]], dim=2)
        X = torch.minimum(torch.maximum(X, lo), hi)

        # --- power readout (uniform per user, as in the original) ---
        pwr = F.relu(self.readout_pwr(uf).squeeze(-1))                 # (B, K)
        s = pwr.sum(1, keepdim=True)
        P = torch.where(s > 0,
                        pwr / torch.maximum(s, pmt[:, None]) * pmt[:, None],
                        torch.full_like(pwr, 1.0) * (pmt / K)[:, None])
        return X, P

    def forward_tensor(self, X0, user_pos, M, N, K, YM, D, Dmax, Pmax,
                       Delta_min):
        """Single-sample (B = 1) version of forward_batch(); see there."""
        X, P = self.forward_batch(np.asarray(X0, float)[None],
                                  np.asarray(user_pos, float)[None],
                                  np.asarray(YM, float)[None],
                                  [float(Pmax)], [float(D)], [float(Dmax)],
                                  [float(Delta_min)], M, N)
        return X[0], P[0]

    def forward(self, X0, user_pos, M, N, K, YM, D, Dmax, Pmax, Delta_min):
        """
        NumPy interface used for inference/evaluation (detached).

        X0: (M, N) initial positions
        user_pos: (K, 2)
        Returns: X positions (M, N), P allocation (K,)
        """
        X, P = self.forward_tensor(X0, user_pos, M, N, K, YM, D, Dmax, Pmax,
                                   Delta_min)
        return X.detach().cpu().numpy(), P.detach().cpu().numpy()


def _heff_batch(X, user_pos, YM, M, N):
    """
    Differentiable torch counterpart of env.compute_heff():

        Heff[b,k,m*N+n] = sqrt(BETA) / r
                          * exp(-1j (2*pi/LAM * r + 2*pi/LAMG * x_bmn)),
        r = sqrt((x_bmn - x_bk)^2 + (y_bm - y_bk)^2 + H^2)

    X: (B, M, N) torch (part of the graph); user_pos: (B, K, 2) torch;
    YM: (B, M) torch.  Returns (B, K, M*N) complex.
    """
    B = X.shape[0]
    x = X.reshape(B, 1, M * N)                                 # (B, 1, MN)
    ym = YM.repeat_interleave(N, dim=1).unsqueeze(1)           # (B, 1, MN)
    ux = user_pos[:, :, 0].unsqueeze(2)                        # (B, K, 1)
    uy = user_pos[:, :, 1].unsqueeze(2)
    r = torch.sqrt((x - ux) ** 2 + (ym - uy) ** 2 + H * H)     # (B, K, MN)
    phase = (2 * np.pi / LAM) * r + (2 * np.pi / LAMG) * x
    return (np.sqrt(BETA) / r) * torch.exp(-1j * phase)


def _dsd_coef_batch(X, user_pos, YM, M, N):
    """
    (B, K, M, N) complex per-PA coefficients of the paper's coherent-combining
    channel, before the radiation ratios A are applied:

        coef[b,k,m,n] = sqrt(BETA)/r * exp(-1j (2*pi/LAM*r + 2*pi/LAMG*x_bmn)),
        r = sqrt((x_bmn - x_bk)^2 + (Y_bm - y_bk)^2 + H^2).

    X: (B,M,N) torch; user_pos: (B,K,2) torch; YM: (B,M) torch.
    """
    x = X.unsqueeze(1)                                          # (B,1,M,N)
    ux = user_pos[:, :, 0][:, :, None, None]                    # (B,K,1,1)
    uy = user_pos[:, :, 1][:, :, None, None]
    ym = YM[:, None, :, None]                                   # (B,1,M,1)
    r = torch.sqrt((x - ux) ** 2 + (ym - uy) ** 2 + H * H)       # (B,K,M,N)
    phase = (2 * np.pi / LAM) * r + (2 * np.pi / LAMG) * x
    return np.sqrt(BETA) * torch.exp(-1j * phase) / r


def _alpha_torch(coef):
    """
    (B, M, N) real radiation-ratio vectors, unit norm per waveguide, i.e. the
    top eigenvector of Re(sum_k c_{k,m,n} conj(c_{k,m,p})) -- the torch
    counterpart of env.solve_ratio()/optimize_alpha_multiuser(), which ADMM and
    DSD use.  A UNIFORM 1/sqrt(N) ratio (the previous BGAT setting) leaves the
    N per-waveguide contributions almost randomly phased and costs 8-13% EE.
    """
    Q = torch.real(torch.einsum('bkmn,bkmp->bmnp', coef, coef.conj()))
    _, v = torch.linalg.eigh(Q)                                 # ascending
    a = v[:, :, :, -1]
    return a / torch.clamp(torch.linalg.norm(a, dim=-1, keepdim=True),
                           min=1e-12)


def _dsd_heff_batch(X, user_pos, YM, M, N, A):
    """
    Differentiable torch counterpart of benchmark_traditional._dsd_heff():
    the K x M coherent-combining channel, one effective coefficient per
    waveguide (paper Eqs. (ch-gm), (ch-hmk)),

        Heff[b,k,m] = sum_n A[b,m,n] coef[b,k,m,n].

    A: (B,M,N) real radiation ratios; see _alpha_torch().  Returns (B,K,M).
    """
    coef = _dsd_coef_batch(X, user_pos, YM, M, N)
    return (coef * A[:, None, :, :]).sum(dim=3)


def _ee_wsub_torch(Heff, Deff, K, Pmax, Rmin, iters=12):
    """
    Differentiable counterpart of the transmit sub-problem in
    benchmark_traditional._solve_wsub() -- the model ADMM and DSD are scored
    with:

        ZF:   W_z = H^H (H H^H + reg I)^-1,   w_k = sqrt(q_k) W_z[:, k]
              (so h_k^H w_j = sqrt(q_k) delta_kj, no interference)
        R     = sum_k log2(1 + q_k/sigma^2)
        Ptx   = sum_k q_k n_k,   n_k = [(H H^H)^-1]_kk   (RADIATED power)
        EE    = R / (Deff + delta2*Ptx)

    q_k is the KKT/Dinkelbach solution
        q_k = max(q_QoS, 1/(ln2 * eta * delta2 * n_k) - sigma^2),
    with eta unrolled (eta <- R/Ptx); the Pmax cap is enforced by scaling q
    down.  reg = 1e-4*tr/K, i.e. exactly the noise-enhancement scale used by
    _solve_wsub() and env._compute_ee().  The unrolled Dinkelbach stops at 12
    iterations (the reachable EE is within ~1-3% of the converged numpy
    _solve_wsub()); the REPORTED numbers always come from the numpy path, so
    this only affects the training gradient.

    Heff: (B,K,M) complex; Deff: (B,); Pmax: (B,).  Returns EE (B,), 0 where
    the QoS floor is infeasible.
    """
    HHH = Heff @ Heff.conj().transpose(1, 2)                    # (B,K,K)
    reg = 1e-4 * torch.real(torch.diagonal(HHH, dim1=1, dim2=2)).sum(-1) / K
    eye = torch.eye(K, dtype=HHH.dtype, device=Heff.device)
    H_inv = torch.linalg.inv(HHH + reg[:, None, None] * eye)
    ne = torch.real(torch.diagonal(H_inv, dim1=1, dim2=2))       # (B,K)
    Wz = Heff.conj().transpose(1, 2) @ H_inv                     # (B,M,K)
    q_qos = (2.0 ** Rmin - 1.0) * SIGMA2
    Pmin = (q_qos * ne).sum(-1)
    ln2d2 = np.log(2.0) * delta2
    eta = (K * np.log2(1.0 + q_qos / SIGMA2)) / (Deff + delta2 * Pmin)
    best = torch.full_like(Deff, -1e30)
    for _ in range(iters):
        q = torch.maximum(1.0 / (ln2d2 * eta[:, None] * ne) - SIGMA2,
                          torch.full_like(ne, q_qos))
        P = (q * ne).sum(-1)
        q = q * torch.minimum(torch.ones_like(P),
                              Pmax / (P + 1e-300))[:, None]       # Ptx cap
        # Actual beamformer W = Wz sqrt(q) and its SINR, exactly as
        # benchmark_traditional._rate_power() (so the residual interference of
        # the regularised ZF is charged here too):
        #   SINR_k = |h_k^H w_k|^2 / (sum_{j!=k} |h_k^H w_j|^2 + sigma^2)
        W = Wz * torch.sqrt(torch.clamp(q, min=0.0))[:, None, :]     # (B,M,K)
        Hw = Heff @ W                                                # (B,K,K)
        sig = torch.abs(torch.diagonal(Hw, dim1=1, dim2=2)) ** 2
        tot = torch.sum(torch.abs(Hw) ** 2, dim=2)
        gamma = sig / (torch.clamp(tot - sig, min=0.0) + SIGMA2)
        R = torch.log2(1.0 + gamma).sum(-1)
        P = torch.sum(torch.abs(W) ** 2, dim=(1, 2))
        best = torch.maximum(best, R / (Deff + delta2 * P))
        eta = R / (P + 1e-30)
    feasible = Pmin <= Pmax
    return torch.where(feasible, best, torch.zeros_like(best))


def _ee_optimal_torch(trH, Deff, K, Pmax, sigma2, delta2_, Rmin, iters=10):
    """
    Differentiable counterpart of env.power_alloc_uniform(): maximise

        EE(Pbar) = K log2(1 + Pbar/sigma^2) / (Deff + delta2*Pbar*trH)

    over Pbar in [Pbar_min, Pbar_max] with the same Dinkelbach fixed point the
    numpy version uses, unrolled so the graph stays connected.  The iteration
    is a contraction near the optimum: 10 steps reproduce the numpy value to
    ~1e-4 relative here, while 6 steps are already 15% off.

    trH, Deff, Pmax: torch tensors (element-wise).  Returns (ee, Pbar) of the
    same shape; ee is 0 where the power budget cannot meet R_min.
    """
    Pbar_min = (2.0 ** Rmin - 1.0) * sigma2
    Pbar_max = Pmax / K
    pmin = torch.full_like(trH, Pbar_min)
    Pbar = Pbar_max + torch.zeros_like(trH)
    for _ in range(iters):
        eta = K * torch.log2(1 + Pbar / sigma2) / (Deff + delta2_ * Pbar * trH)
        Pstar = K / (eta * delta2_ * trH * np.log(2)) - sigma2
        Pbar = torch.minimum(torch.maximum(Pstar, pmin), Pbar_max)
    ee = K * torch.log2(1 + Pbar / sigma2) / (Deff + delta2_ * Pbar * trH)
    feasible = Pbar_max >= Pbar_min
    return torch.where(feasible, ee, torch.zeros_like(ee)), Pbar


def compute_bgat_loss(model, batch, device='cpu', alpha=None):
    """
    Unsupervised loss: negative EE averaged over batch.

    Differentiable form of the SAME objective the reported BGAT column is
    scored with -- the paper's (C) model, which ADMM and DSD also use:

        Heff  = benchmark_traditional._dsd_heff()      K x M coherent combining
                with radiation ratios A = 1/sqrt(N)
        EE    = R / (Deff + delta2*Ptx)
        R     = sum_k log2(1 + q_k/sigma^2)            (ZF, no interference)
        Ptx   = sum_k q_k [(Heff Heff^H)^-1]_kk        (RADIATED power)
        Deff  = PF + P_motor*M_d/(T_d*v)               (movement energy)

    so the trained network maximises exactly the quantity that is reported:
    gradients and score now refer to one and the same R, which the previous
    K x M*N ZF surrogate did not (there the rate ignored the noise enhancement
    while the power charged it, and a full-array ZF channel is nearly singular
    -- [(H H^H)^-1]_kk ~ 1e7 -- so its "optimal" power was ~1 nW).

    The position readout stays in torch so loss.backward() reaches the
    parameters.  The power head is a by-product (the transmit power is
    re-optimised here, as _solve_wsub() does) and receives no gradient.

    batch: list of dicts with keys X0, user_pos, M, N, K, YM, D, Dmax, Pmax,
           Tframe, Rmin, Delta_min -- all samples MUST share (M, N, K).
    Returns: (loss, mean EE) as torch scalars.
    """
    dev = next(model.parameters()).device
    B = len(batch)
    M, N, K = batch[0]['M'], batch[0]['N'], batch[0]['K']

    X0 = np.stack([np.asarray(s['X0'], float) for s in batch])          # (B,M,N)
    up = np.stack([np.asarray(s['user_pos'], float) for s in batch])    # (B,K,2)
    YM = np.stack([np.asarray(s['YM'], float) for s in batch])          # (B,M)
    Pmax = np.array([float(s['Pmax']) for s in batch])
    D = np.array([float(s['D']) for s in batch])
    Dmax = np.array([float(s['Dmax']) for s in batch])
    Delta = np.array([float(s['Delta_min']) for s in batch])
    Tf = torch.as_tensor([float(s['Tframe']) for s in batch],
                         dtype=torch.float32, device=dev)
    Rmin = float(batch[0]['Rmin'])

    X, _ = model.forward_batch(X0, up, YM, Pmax, D, Dmax, Delta, M, N)

    # Paper (C) channel: K x M coherent combining with EXACTLY the radiation
    # ratios the scorer uses (env.optimize_alpha_multiuser, the same subset
    # search ADMM and DSD run), so the loss sees the scored effective channel.
    # The ratios are detached: by the envelope theorem the omitted dA/dX term
    # vanishes at the alpha optimum, whereas the differentiable full-N
    # eigenvector proxy is ~3% optimistic on moved layouts and made the
    # network over-move (its scored EE fell below the untouched X0).
    coef = _dsd_coef_batch(
        X,
        torch.as_tensor(up, dtype=torch.float32, device=dev),
        torch.as_tensor(YM, dtype=torch.float32, device=dev),
        M, N)
    if alpha is None:
        Xn = X.detach().cpu().numpy()
        A_np = np.stack([optimize_alpha_multiuser(Xn[b], up[b], M, N, K, YM[b])
                         for b in range(B)])
    else:
        # Cached ratios (see train_bgat.build_alpha): the per-sample numpy
        # subset search would otherwise dominate the epoch time.
        A_np = np.asarray(alpha, dtype=np.float32)
    A = torch.as_tensor(A_np, device=dev)
    Heff = (coef * A[:, None, :, :]).sum(dim=3)

    # Denominator, identical to the scored one (see bgat_benchmark):
    # PF + P_motor*M_d/(T_d*v) + delta2*||W||^2, so the network maximises the
    # quantity that is reported.  Measured alternatives: charging the movement
    # here gives 9.042-9.047 with M_d ~ 3 cm; dropping the movement term (the
    # "optimise without the motion cost" variant) collapses to M_d ~ 0.5 mm and
    # 9.026-9.027, i.e. worse -- and a nearly flat column.
    X0t = torch.as_tensor(X0, dtype=torch.float32, device=dev)
    Tm = torch.amax(torch.abs(X - X0t), dim=(1, 2)) / V
    Td = torch.clamp(Tf - Tm, min=1e-3)
    Deff = PMOTOR * Tm / Td + PF

    ee = _ee_wsub_torch(Heff, Deff, K,
                        torch.as_tensor(Pmax, dtype=torch.float32, device=dev),
                        Rmin)
    return -ee.mean(), ee.mean()
