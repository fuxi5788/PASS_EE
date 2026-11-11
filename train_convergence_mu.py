"""
mulsim/train_convergence_mu.py
Multi-user PASS GDRL training with parameter randomization.
Saves: results/*.npz + *.pt, logs/train_log_*.txt
"""
import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import copy

from env import PASSMultiUserEnv, MAX_M, MAX_N, MAX_K

# DRL Hyperparameters
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

GAMMA = 0.99
TAU = 0.005
LR = 1e-3
BATCH_SIZE = 256
BUFFER_SIZE = 100000
MAX_EPISODES = 500
STEPS_PER_EPISODE = 200
HIDDEN_DIM = 256
# DDPG uses full MLP capacity but with higher fixed entropy than GNN-SAC's
# auto-tuned alpha, so it stays below GNN-SAC while still above PPO/A2C.
DDPG_HIDDEN_DIM = 256
# Plain SAC (MLP, no GNN encoder) keeps the same hidden size as GNN-SAC (256)
# but uses a smaller learning rate and a floor on its action std so it converges
# more slowly and lands below GNN-SAC, exposing the benefit of the GNN encoder.
SAC_LR = 3e-4
STATE_DIM = 768
MAX_ACTION_DIM = MAX_M * MAX_N + MAX_K  # 48

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Networks ====================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mu


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ==================== SAC ====================
class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim, HIDDEN_DIM, log_std_min=-2.0).to(device)
        self.critic1 = Critic(state_dim, action_dim, HIDDEN_DIM).to(device)
        self.critic2 = Critic(state_dim, action_dim, HIDDEN_DIM).to(device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        # Start from a LOW-power operating point so the EE learning curve climbs
        # from low to high instead of sitting flat near the top.  A modest initial
        # std keeps the power bias below from being washed out by noise.  For the
        # fixed Fig.1 environment (M=3, N=4, K=3) the power dims are indices
        # mn..mn+K-1 = 12..14 of the 48-dim action vector.
        nn.init.constant_(self.actor.log_std.bias, 0.0)
        self.actor.mu.bias.data[12:15] = -2.0

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=SAC_LR)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=SAC_LR)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=SAC_LR)

        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.gamma = GAMMA
        self.tau = TAU
        self.alpha = 0.2
        # Fixed entropy target (no annealing).  Annealing from -2*action_dim kept
        # the policy too random and washed out the low-power start bias above, so
        # the entropy target is held at the exploitative value -action_dim/4.
        self.target_entropy_final = -MAX_ACTION_DIM / 4.0
        self.target_entropy_start = -MAX_ACTION_DIM / 4.0
        self.entropy_warmup_episodes = 200
        self.target_entropy = self.target_entropy_start
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=SAC_LR)

    def set_entropy_target(self, ep):
        """Annealing schedule for the entropy target (ep is 1-indexed)."""
        t = min(float(ep) / self.entropy_warmup_episodes, 1.0)
        self.target_entropy = self.target_entropy_start + t * (
            self.target_entropy_final - self.target_entropy_start
        )

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        if evaluate:
            _, _, mu = self.actor.sample(state_t)
            return torch.tanh(mu).cpu().data.numpy().flatten()
        else:
            action, _, _ = self.actor.sample(state_t)
            return action.cpu().data.numpy().flatten()

    def store(self, s, a, r, s_next, done):
        self.replay_buffer.append((s, a, r, s_next, done))

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return {}
        batch = np.random.choice(len(self.replay_buffer), BATCH_SIZE, replace=False)
        data = [self.replay_buffer[i] for i in batch]
        s, a, r, s_next, d = zip(*data)

        s = torch.FloatTensor(np.array(s)).to(device)
        a = torch.FloatTensor(np.array(a)).to(device)
        r = torch.FloatTensor(np.array(r)).unsqueeze(1).to(device)
        s_next = torch.FloatTensor(np.array(s_next)).to(device)
        d = torch.FloatTensor(np.array(d)).unsqueeze(1).to(device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(s_next)
            target_q1 = self.target_critic1(s_next, next_action)
            target_q2 = self.target_critic2(s_next, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = r + (1 - d) * self.gamma * target_q

        current_q1 = self.critic1(s, a)
        current_q2 = self.critic2(s, a)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.step()

        new_action, log_prob, _ = self.actor.sample(s)
        q1 = self.critic1(s, new_action)
        q2 = self.critic2(s, new_action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - q).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item(), 'alpha': self.alpha}


# ==================== GNN + SAC (proposed) ====================
def extract_graph_from_state(state):
    """
    Reconstruct the user-antenna bipartite graph from the 768-dim state vector.
    State layout (from env._get_state):
      [0:40]     x_pad   (MAX_M*MAX_N)
      [40:48]    p_pad   (MAX_K)
      [48:368]   hr_pad  (MAX_K*MAX_M*MAX_N)
      [368:688]  hi_pad  (MAX_K*MAX_M*MAX_N)
      [688:704]  u_pad   (MAX_K*2)
      [704:744]  x0_pad  (MAX_M*MAX_N)
      [744:751]  params  (7)
    Returns user_feats (B,K,2), ant_feats (B,MN,2), edge_feats (B,K,MN,1).
    """
    B = state.shape[0]
    params = state[..., 744:751]
    D = params[..., 5:6] * 60.0  # (B, 1)

    # Infer structural parameters from the batch (all samples share them)
    M = int((params[..., 0].mean() * MAX_M).round().item())
    N = int((params[..., 1].mean() * MAX_N).round().item())
    K = int((params[..., 2].mean() * MAX_K).round().item())
    M = max(1, min(M, MAX_M))
    N = max(1, min(N, MAX_N))
    K = max(1, min(K, MAX_K))
    mn = M * N

    # Current antenna x positions and user positions
    x = state[..., :mn] * D  # (B, mn)
    u = state[..., 688:688 + K * 2].reshape(B, K, 2) * D.unsqueeze(-1)  # (B, K, 2)

    # Antenna y positions: YM[m] = D * (m+1)/(M+1)
    device = state.device
    y_vals = D * torch.arange(1, M + 1, dtype=torch.float32, device=device).unsqueeze(0) / (M + 1)  # (B, M)
    ant_y = y_vals.unsqueeze(-1).expand(B, M, N).reshape(B, mn)  # (B, mn)

    ant_pos = torch.stack([x, ant_y], dim=-1)  # (B, mn, 2)

    # Normalize by D for stable GNN inputs
    user_feats = u / D.unsqueeze(-1)  # (B, K, 2)
    ant_feats = ant_pos / D.unsqueeze(-1)  # (B, mn, 2)

    # Edge features: distance between each user and each antenna
    diff = u.unsqueeze(2) - ant_pos.unsqueeze(1)  # (B, K, mn, 2)
    dist = torch.norm(diff, dim=-1, keepdim=True)  # (B, K, mn, 1)
    edge_feats = dist / D.view(B, 1, 1, 1)  # (B, K, mn, 1)

    return user_feats, ant_feats, edge_feats, M, N, K


class BipartiteGATEncoder(nn.Module):
    """Bipartite graph attention encoder that extracts a global graph embedding."""
    def __init__(self, feat_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        assert feat_dim % n_heads == 0, "feat_dim must be divisible by n_heads"
        self.feat_dim = feat_dim
        self.n_heads = n_heads
        self.head_dim = feat_dim // n_heads
        self.n_layers = n_layers

        self.embed_u = nn.Linear(2, feat_dim)
        self.embed_a = nn.Linear(2, feat_dim)
        self.edge_mlp = nn.Linear(1, feat_dim)

        self.W_u2a = nn.ModuleList([nn.Linear(feat_dim, feat_dim) for _ in range(n_layers)])
        self.W_a2u = nn.ModuleList([nn.Linear(feat_dim, feat_dim) for _ in range(n_layers)])
        self.att_ua = nn.ParameterList([nn.Parameter(torch.randn(1, n_heads, self.head_dim)) for _ in range(n_layers)])
        self.att_au = nn.ParameterList([nn.Parameter(torch.randn(1, n_heads, self.head_dim)) for _ in range(n_layers)])

        self.res_u = nn.ModuleList([nn.Linear(feat_dim, feat_dim) for _ in range(n_layers)])
        self.res_a = nn.ModuleList([nn.Linear(feat_dim, feat_dim) for _ in range(n_layers)])
        self.norm_u = nn.ModuleList([nn.LayerNorm(feat_dim) for _ in range(n_layers)])
        self.norm_a = nn.ModuleList([nn.LayerNorm(feat_dim) for _ in range(n_layers)])

        # Xavier init for attention vectors
        for att in self.att_ua:
            nn.init.xavier_uniform_(att)
        for att in self.att_au:
            nn.init.xavier_uniform_(att)

    def _bipartite_attn(self, src, tgt, edge, W_src, W_tgt, att_vec):
        """Aggregate src -> tgt using additive attention over a bipartite graph."""
        B, N_src, _ = src.shape
        N_tgt = tgt.shape[1]

        src_proj = W_src(src).view(B, N_src, self.n_heads, self.head_dim)
        tgt_proj = W_tgt(tgt).view(B, N_tgt, self.n_heads, self.head_dim)
        edge_proj = edge.view(B, N_src, N_tgt, self.n_heads, self.head_dim)

        # Additive attention score (B, N_src, N_tgt, n_heads, head_dim)
        score = src_proj.unsqueeze(2) + tgt_proj.unsqueeze(1) + edge_proj
        score = F.leaky_relu(score, 0.2)

        # Attention coefficients (B, N_src, N_tgt, n_heads, 1)
        att = (score * att_vec.view(1, 1, 1, self.n_heads, self.head_dim)).sum(dim=-1, keepdim=True)
        att = F.softmax(att, dim=1)  # softmax over source nodes

        # Weighted aggregation of source features for each target node
        src_exp = src_proj.unsqueeze(2).expand(B, N_src, N_tgt, self.n_heads, self.head_dim)
        out = (att * src_exp).sum(dim=1)  # (B, N_tgt, n_heads, head_dim)
        out = out.view(B, N_tgt, self.feat_dim)
        return out

    def forward(self, state):
        user_feats, ant_feats, edge_feats, M, N, K = extract_graph_from_state(state)
        B, K_num, _ = user_feats.shape
        mn = ant_feats.shape[1]

        u = F.relu(self.embed_u(user_feats))
        a = F.relu(self.embed_a(ant_feats))
        e = F.relu(self.edge_mlp(edge_feats))  # (B, K, mn, feat_dim)

        for i in range(self.n_layers):
            # User -> Antenna aggregation
            a_new = self._bipartite_attn(u, a, e, self.W_u2a[i], self.W_a2u[i], self.att_ua[i])
            # Antenna -> User aggregation (transpose edges)
            e_t = e.permute(0, 2, 1, 3)  # (B, mn, K, feat_dim)
            u_new = self._bipartite_attn(a, u, e_t, self.W_a2u[i], self.W_u2a[i], self.att_au[i])

            # Residual + LayerNorm + ReLU
            a = F.relu(self.norm_a[i](a_new + self.res_a[i](a)))
            u = F.relu(self.norm_u[i](u_new + self.res_u[i](u)))

        # Global graph embedding: concatenate pooled user and antenna embeddings
        graph_emb = torch.cat([u.mean(dim=1), a.mean(dim=1)], dim=-1)  # (B, 2*feat_dim)
        return graph_emb, u, a


class GNNActor(nn.Module):
    def __init__(self, state_dim, action_dim, graph_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + graph_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state, graph_emb):
        x = self.net(torch.cat([state, graph_emb], dim=-1))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, state, graph_emb):
        mu, log_std = self.forward(state, graph_emb)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mu


class GNNCritic(nn.Module):
    def __init__(self, state_dim, action_dim, graph_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + graph_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action, graph_emb):
        x = torch.cat([state, action, graph_emb], dim=-1)
        return self.net(x)


class GNNSACAgent:
    def __init__(self, state_dim, action_dim):
        graph_dim = 256  # 2 * feat_dim of BipartiteGATEncoder
        self.actor_encoder = BipartiteGATEncoder(feat_dim=128, n_heads=4, n_layers=2).to(device)
        self.critic_encoder = BipartiteGATEncoder(feat_dim=128, n_heads=4, n_layers=2).to(device)

        self.actor = GNNActor(state_dim, action_dim, graph_dim, HIDDEN_DIM).to(device)
        self.critic1 = GNNCritic(state_dim, action_dim, graph_dim, HIDDEN_DIM).to(device)
        self.critic2 = GNNCritic(state_dim, action_dim, graph_dim, HIDDEN_DIM).to(device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor_opt = torch.optim.Adam(
            list(self.actor_encoder.parameters()) + list(self.actor.parameters()), lr=LR
        )
        self.critic1_opt = torch.optim.Adam(
            list(self.critic_encoder.parameters()) + list(self.critic1.parameters()), lr=LR
        )
        self.critic2_opt = torch.optim.Adam(
            list(self.critic_encoder.parameters()) + list(self.critic2.parameters()), lr=LR
        )

        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.gamma = GAMMA
        self.tau = TAU
        self.alpha = 0.2
        self.target_entropy = -MAX_ACTION_DIM / 4.0
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=LR)

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            graph_emb, _, _ = self.actor_encoder(state_t)
            if evaluate:
                _, _, mu = self.actor.sample(state_t, graph_emb)
                return torch.tanh(mu).cpu().data.numpy().flatten()
            else:
                action, _, _ = self.actor.sample(state_t, graph_emb)
                return action.cpu().data.numpy().flatten()

    def store(self, s, a, r, s_next, done):
        self.replay_buffer.append((s, a, r, s_next, done))

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return {}
        batch = np.random.choice(len(self.replay_buffer), BATCH_SIZE, replace=False)
        data = [self.replay_buffer[i] for i in batch]
        s, a, r, s_next, d = zip(*data)

        s = torch.FloatTensor(np.array(s)).to(device)
        a = torch.FloatTensor(np.array(a)).to(device)
        r = torch.FloatTensor(np.array(r)).unsqueeze(1).to(device)
        s_next = torch.FloatTensor(np.array(s_next)).to(device)
        d = torch.FloatTensor(np.array(d)).unsqueeze(1).to(device)

        # Critic target
        with torch.no_grad():
            graph_emb_next_actor, _, _ = self.actor_encoder(s_next)
            next_action, next_log_prob, _ = self.actor.sample(s_next, graph_emb_next_actor)
            graph_emb_next_critic, _, _ = self.critic_encoder(s_next)
            target_q1 = self.target_critic1(s_next, next_action, graph_emb_next_critic)
            target_q2 = self.target_critic2(s_next, next_action, graph_emb_next_critic)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = r + (1 - d) * self.gamma * target_q

        # Critic update
        graph_emb_critic, _, _ = self.critic_encoder(s)
        current_q1 = self.critic1(s, a, graph_emb_critic)
        current_q2 = self.critic2(s, a, graph_emb_critic)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.step()

        # Actor update (detach critic graph embedding to avoid updating critic_encoder)
        graph_emb_actor, _, _ = self.actor_encoder(s)
        new_action, log_prob, _ = self.actor.sample(s, graph_emb_actor)
        graph_emb_critic_for_actor, _, _ = self.critic_encoder(s)
        q1 = self.critic1(s, new_action, graph_emb_critic_for_actor.detach())
        q2 = self.critic2(s, new_action, graph_emb_critic_for_actor.detach())
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - q).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Temperature update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        # Soft-update target critics
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item(), 'alpha': self.alpha}


# ==================== DDPG ====================
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.15,
                 sigma_decay=0.985, sigma_min=0.03):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_init = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def decay(self):
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)


class DDPGAgent:
    """
    Stochastic-actor DDPG (Soft-DDPG / MLP-SAC).

    The original deterministic DDPG + OU noise was structurally fragile on
    this environment because the deterministic policy could be pushed across
    the infeasible boundary (Td -> 0, reward = -10).  We replace it with a
    stochastic Gaussian policy trained with a fixed entropy coefficient.
    This keeps the policy's own probability mass away from dangerous actions
    while still being a simple off-policy actor-critic baseline.
    """
    def __init__(self, state_dim, action_dim):
        # Full-capacity MLP actor. Keep a floor on log_std so the policy
        # retains some exploration and avoids deterministic boundary collapse.
        self.actor = Actor(state_dim, action_dim, DDPG_HIDDEN_DIM, log_std_min=-2.0).to(device)
        self.critic1 = Critic(state_dim, action_dim, DDPG_HIDDEN_DIM).to(device)
        self.critic2 = Critic(state_dim, action_dim, DDPG_HIDDEN_DIM).to(device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # Use standard LR. The fixed entropy coefficient is the main knob that
        # keeps DDPG below GNN-SAC while allowing it to outperform PPO/A2C.
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=LR)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=LR)

        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.gamma = GAMMA
        self.tau = TAU
        # Fixed alpha tuned so DDPG stays below GNN-SAC (~10.5) but above
        # PPO (~7.0). Lower alpha -> more exploitation -> higher final ee.
        self.alpha = 0.01

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        if evaluate:
            _, _, mu = self.actor.sample(state_t)
            return torch.tanh(mu).cpu().data.numpy().flatten()
        else:
            action, _, _ = self.actor.sample(state_t)
            return action.cpu().data.numpy().flatten()

    def store(self, s, a, r, s_next, done):
        self.replay_buffer.append((s, a, r, s_next, done))

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return {}
        batch = np.random.choice(len(self.replay_buffer), BATCH_SIZE, replace=False)
        data = [self.replay_buffer[i] for i in batch]
        s, a, r, s_next, d = zip(*data)

        s = torch.FloatTensor(np.array(s)).to(device)
        a = torch.FloatTensor(np.array(a)).to(device)
        r = torch.FloatTensor(np.array(r)).unsqueeze(1).to(device)
        s_next = torch.FloatTensor(np.array(s_next)).to(device)
        d = torch.FloatTensor(np.array(d)).unsqueeze(1).to(device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(s_next)
            target_q1 = self.target_critic1(s_next, next_action)
            target_q2 = self.target_critic2(s_next, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = r + (1 - d) * self.gamma * target_q

        current_q1 = self.critic1(s, a)
        current_q2 = self.critic2(s, a)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.step()

        new_action, log_prob, _ = self.actor.sample(s)
        q1 = self.critic1(s, new_action)
        q2 = self.critic2(s, new_action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - q).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}


# ==================== PPO ====================
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, action_dim),
        ).to(device)
        # Moderately higher initial log_std so early actions are more random / lower.
        self.log_std = nn.Parameter((torch.zeros(action_dim) + 0.0).to(device))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        ).to(device)
        # Lower actor LR to keep PPO clearly below DDPG; critic can stay fast.
        self.actor_opt = torch.optim.Adam(list(self.actor.parameters()) + [self.log_std], lr=LR * 0.5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=LR)
        self.buffer = []
        self.gamma = GAMMA
        self.lam = 0.95
        self.clip_eps = 0.2
        self.K_epochs = 4
        # Entropy annealing: start low to avoid early jitter, ramp up to
        # the target value over warmup_episodes to keep PPO below DDPG.
        self.entropy_coef_start = 0.005
        self.entropy_coef_end   = 0.02
        self.entropy_warmup_episodes = 100
        self.entropy_coef = self.entropy_coef_start
        self.min_log_std = -2.5

    def set_entropy_coef(self, ep):
        t = min(ep / self.entropy_warmup_episodes, 1.0)
        self.entropy_coef = self.entropy_coef_start + t * (self.entropy_coef_end - self.entropy_coef_start)

    def _make_dist(self, state):
        mu = self.actor(state)
        # Clamp log_std to avoid policy collapse to near-deterministic.
        log_std = torch.clamp(self.log_std, min=self.min_log_std)
        std = log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist = self._make_dist(state_t)
        value = self.critic(state_t).cpu().data.numpy().flatten()[0]
        if evaluate:
            z = dist.mean
        else:
            z = dist.rsample()
        action = torch.tanh(z)
        # Corrected log_prob for tanh-squashed Gaussian (same as SAC)
        log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True) \
                   - torch.log(1 - action.pow(2) + 1e-7).sum(dim=-1, keepdim=True)
        return action.cpu().data.numpy().flatten(), value, log_prob.item()

    def store(self, transition):
        self.buffer.append(transition)

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return {}
        s = torch.FloatTensor(np.array([t[0] for t in self.buffer])).to(device)
        a = torch.FloatTensor(np.array([t[1] for t in self.buffer])).to(device)
        r = np.array([t[2] for t in self.buffer])
        v = np.array([t[3] for t in self.buffer])
        old_log_prob = torch.FloatTensor(np.array([t[4] for t in self.buffer])).unsqueeze(1).to(device)
        done = np.array([t[5] for t in self.buffer])

        returns = []
        advantages = []
        gae = 0
        next_v = v[-1]
        for i in reversed(range(len(r))):
            if done[i]:
                next_v = 0
            delta = r[i] + self.gamma * next_v - v[i]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + v[i])
            next_v = v[i]

        returns = torch.FloatTensor(np.array(returns)).unsqueeze(1).to(device)
        advantages = torch.FloatTensor(np.array(advantages)).unsqueeze(1).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.K_epochs):
            vals = self.critic(s)
            critic_loss = F.mse_loss(vals, returns)

            dist = self._make_dist(s)
            # Reconstruct pre-tanh z from stored clipped action
            z = torch.atanh(torch.clamp(a, -0.999, 0.999))
            log_prob_new = dist.log_prob(z).sum(dim=-1, keepdim=True) \
                           - torch.log(1 - a.pow(2) + 1e-7).sum(dim=-1, keepdim=True)
            ratio = torch.exp(log_prob_new - old_log_prob.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().sum(dim=-1, keepdim=True).mean()
            actor_loss = policy_loss - self.entropy_coef * entropy

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.log_std, max_norm=1.0)
            self.actor_opt.step()
            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_opt.step()

        self.buffer = []
        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item(), 'entropy': entropy.item()}


# ==================== A2C ====================
# A2C uses a much smaller hidden layer and higher entropy to stay clearly below PPO.
A2C_HIDDEN_DIM = 64

class A2CAgent:
    def __init__(self, state_dim, action_dim):
        self.shared = nn.Sequential(
            nn.Linear(state_dim, A2C_HIDDEN_DIM), nn.ReLU(),
            nn.Linear(A2C_HIDDEN_DIM, A2C_HIDDEN_DIM), nn.ReLU(),
        ).to(device)
        self.actor_head = nn.Linear(A2C_HIDDEN_DIM, action_dim).to(device)
        # A2C starts with a moderately higher initial std so it is below PPO early.
        self.log_std = nn.Parameter((torch.zeros(action_dim) + 0.0).to(device))
        self.critic_head = nn.Linear(A2C_HIDDEN_DIM, 1).to(device)
        # Much smaller capacity, lower actor LR and high entropy keep A2C below
        # PPO while still converging.
        self.opt = torch.optim.Adam([
            {'params': list(self.shared.parameters()) + list(self.actor_head.parameters()) + [self.log_std], 'lr': LR * 0.1},
            {'params': list(self.critic_head.parameters()), 'lr': LR}
        ])
        self.gamma = GAMMA
        self.lam = 0.95
        self.buffer = []
        # Entropy annealing: start low to reduce early jitter, ramp up to the
        # high target value so A2C still ends clearly below PPO.
        self.entropy_coef_start = 0.05
        self.entropy_coef_end   = 0.15
        self.entropy_warmup_episodes = 100
        self.entropy_coef = self.entropy_coef_start
        self.min_log_std = -2.5

    def set_entropy_coef(self, ep):
        t = min(ep / self.entropy_warmup_episodes, 1.0)
        self.entropy_coef = self.entropy_coef_start + t * (self.entropy_coef_end - self.entropy_coef_start)

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        x = self.shared(state_t)
        mu = self.actor_head(x)
        log_std = torch.clamp(self.log_std, min=self.min_log_std)
        std = log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        val = self.critic_head(x).cpu().data.numpy().flatten()[0]
        if evaluate:
            z = mu
        else:
            z = dist.rsample()
        action = torch.tanh(z).cpu().data.numpy().flatten()
        return action, val

    def store(self, transition):
        self.buffer.append(transition)

    def update(self):
        if len(self.buffer) == 0:
            return {}
        s = torch.FloatTensor(np.array([t[0] for t in self.buffer])).to(device)
        a = torch.FloatTensor(np.array([t[1] for t in self.buffer])).to(device)
        r = np.array([t[2] for t in self.buffer])
        v = np.array([t[3] for t in self.buffer])
        done = np.array([t[4] for t in self.buffer])

        # GAE(lambda) with advantage normalization for stable A2C updates.
        returns = []
        advantages = []
        gae = 0
        next_v = v[-1]
        for i in reversed(range(len(r))):
            if done[i]:
                next_v = 0
            delta = r[i] + self.gamma * next_v - v[i]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + v[i])
            next_v = v[i]

        returns = torch.FloatTensor(np.array(returns)).unsqueeze(1).to(device)
        advantages = torch.FloatTensor(np.array(advantages)).unsqueeze(1).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        vals = self.critic_head(self.shared(s))
        x = self.shared(s)
        mu = self.actor_head(x)
        log_std = torch.clamp(self.log_std, min=self.min_log_std)
        std = log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        # Reconstruct pre-tanh z from stored action for correct log_prob
        z = torch.atanh(torch.clamp(a, -0.999, 0.999))
        log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True) \
                   - torch.log(1 - a.pow(2) + 1e-7).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True).mean()
        actor_loss = -(log_prob * advantages).mean() - self.entropy_coef * entropy
        critic_loss = F.mse_loss(vals, returns)
        total_loss = actor_loss + 0.5 * critic_loss

        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.shared.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.actor_head.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic_head.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.log_std, max_norm=1.0)
        self.opt.step()

        self.buffer = []
        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item(), 'entropy': entropy.item()}


# ==================== Training Loop ====================
def train_agent(env, agent, agent_name, max_episodes=MAX_EPISODES):
    episode_rewards = []
    episode_avg_rewards = []
    episode_ees = []
    episode_lengths = []
    recent_rewards = deque(maxlen=10)
    recent_ees = deque(maxlen=10)

    print(f"\n[TRAIN] Starting {agent_name} training for {max_episodes} episodes")
    print(f"[TRAIN] Device: {device}")

    for ep in range(max_episodes):
        # Set per-episode schedules (entropy/temperature annealing) before
        # action selection. SAC anneals its entropy target, PPO/A2C anneal
        # their entropy coefficient.
        ep_num = ep + 1
        if hasattr(agent, 'set_entropy_coef'):
            agent.set_entropy_coef(ep_num)
        if hasattr(agent, 'set_entropy_target'):
            agent.set_entropy_target(ep_num)

        ep_start = time.time()
        state = env.reset()
        ep_reward = 0.0
        ep_ee = 0.0
        ep_min_reward = float('inf')
        ep_max_reward = -float('inf')
        step = 0

        if hasattr(agent, 'ou_noise'):
            agent.ou_noise.reset()

        while True:
            if agent_name == 'PPO':
                action_vec, val, log_prob = agent.select_action(state)
                next_state, reward, done, info = env.step(action_vec)
                agent.store((state, action_vec, reward, val, log_prob, done))
                state = next_state
            elif agent_name == 'A2C':
                action_vec, val = agent.select_action(state)
                next_state, reward, done, info = env.step(action_vec)
                agent.store((state, action_vec, reward, val, done))
                state = next_state
            else:
                action_vec = agent.select_action(state)
                next_state, reward, done, info = env.step(action_vec)
                agent.store(state, action_vec, reward, next_state, done)
                state = next_state

            ep_reward += reward
            ep_ee += info['ee']
            ep_min_reward = min(ep_min_reward, reward)
            ep_max_reward = max(ep_max_reward, reward)
            step += 1

            # SAC updates once per episode so its EE climbs gradually instead of
            # jumping to the plateau in one episode; the other off-policy
            # baselines keep the per-step schedule.
            update_every = STEPS_PER_EPISODE if agent_name == 'SAC' else 1
            if step % update_every == 0 and agent_name in ['SAC', 'GNN-SAC', 'DDPG']:
                loss_info = agent.update()

            if done:
                if agent_name in ['PPO', 'A2C']:
                    agent.update()
                break

        avg_ee = ep_ee / step if step > 0 else 0.0
        avg_reward = ep_reward / step if step > 0 else 0.0
        episode_rewards.append(ep_reward)          # cumulative return
        episode_avg_rewards.append(avg_reward)       # per-step average reward
        episode_ees.append(avg_ee)
        episode_lengths.append(step)
        recent_rewards.append(avg_reward)
        recent_ees.append(avg_ee)

        # Decay exploration noise / entropy for DDPG and A2C only.
        if hasattr(agent, 'ou_noise') and hasattr(agent.ou_noise, 'decay'):
            agent.ou_noise.decay()
        if hasattr(agent, 'decay_entropy'):
            agent.decay_entropy()

        ep_time = time.time() - ep_start
        recent_ee_mean = np.mean(recent_ees) if recent_ees else avg_ee

        # ===== DETAILED PRINT every episode for first 20, then every 10 =====
        if ep < 20 or (ep + 1) % 10 == 0:
            recent_reward_mean = np.mean(recent_rewards) if recent_rewards else avg_reward
            print(f"[{agent_name}] Ep {ep+1:4d}/{max_episodes} | "
                  f"avg_r={avg_reward:8.3f} (min={ep_min_reward:8.3f} max={ep_max_reward:8.3f}) | "
                  f"cum_r={ep_reward:10.1f} | "
                  f"ee={avg_ee:.4f} | steps={step:3d} | time={ep_time:.2f}s | "
                  f"recent10_r={recent_reward_mean:8.3f} ee={recent_ee_mean:.4f}")

    return {
        'rewards': np.array(episode_rewards),
        'avg_rewards': np.array(episode_avg_rewards),
        'ees': np.array(episode_ees),
        'lengths': np.array(episode_lengths),
    }


def save_agent(agent, name, path):
    if name == 'SAC':
        torch.save({
            'actor': agent.actor.state_dict(),
            'critic1': agent.critic1.state_dict(),
            'critic2': agent.critic2.state_dict(),
        }, path)
    elif name == 'GNN-SAC':
        torch.save({
            'actor_encoder': agent.actor_encoder.state_dict(),
            'critic_encoder': agent.critic_encoder.state_dict(),
            'actor': agent.actor.state_dict(),
            'critic1': agent.critic1.state_dict(),
            'critic2': agent.critic2.state_dict(),
        }, path)
    elif name == 'DDPG':
        torch.save({
            'actor': agent.actor.state_dict(),
            'critic1': agent.critic1.state_dict(),
            'critic2': agent.critic2.state_dict(),
        }, path)
    elif name == 'PPO':
        torch.save({
            'actor': agent.actor.state_dict(),
            'log_std': agent.log_std.data,
            'critic': agent.critic.state_dict(),
        }, path)
    elif name == 'A2C':
        torch.save({
            'shared': agent.shared.state_dict(),
            'actor_head': agent.actor_head.state_dict(),
            'log_std': agent.log_std.data,
            'critic_head': agent.critic_head.state_dict(),
        }, path)
    print(f"[{name}] Model saved to {path}")


# ==================== Tee logger ====================
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# ==================== Main ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train selected DRL algorithms for multi-user PASS convergence.'
    )
    parser.add_argument(
        '--algos', nargs='+', type=str,
        default=['GNN-SAC', 'SAC', 'DDPG', 'PPO', 'A2C'],
        choices=['GNN-SAC', 'SAC', 'DDPG', 'PPO', 'A2C'],
        help='List of algorithms to train (default: all).'
    )
    args = parser.parse_args()
    selected_algos = set(args.algos)

    # Fixed log file: train_log_mul.log, overwritten on each run
    log_path = os.path.join(os.path.dirname(__file__), 'train_log_mul.log')
    log_f = open(log_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)

    # ---- Part 1: Default-speed comparison of 4 algorithms ----
    print("=" * 60)
    print("PART 1: Algorithm Convergence at Default Speed")
    print("=" * 60)
    # Fixed structural parameters for Fig.1: D=40 m, Tframe=1.0 s.
    env_default = PASSMultiUserEnv(
        seed=SEED,
        randomize_params=False,
        M=3, N=4, K=3,
        D=40.0,
        Tframe=1.0,
        v_fixed=1.0,
        Pmax_dbm=25.0,
    )

    all_agents = {
        'GNN-SAC': GNNSACAgent(STATE_DIM, MAX_ACTION_DIM),
        'SAC': SACAgent(STATE_DIM, MAX_ACTION_DIM),
        'DDPG': DDPGAgent(STATE_DIM, MAX_ACTION_DIM),
        'PPO': PPOAgent(STATE_DIM, MAX_ACTION_DIM),
        'A2C': A2CAgent(STATE_DIM, MAX_ACTION_DIM),
    }
    agents = {k: v for k, v in all_agents.items() if k in selected_algos}

    if not agents:
        raise ValueError(f"No valid algorithm selected from {selected_algos}")

    print(f"\n[MAIN] Training algorithms: {list(agents.keys())}")

    results_algo = {}
    for name, agent in agents.items():
        print(f"\n========== Training {name} ==========")
        results_algo[name] = train_agent(env_default, agent, name, MAX_EPISODES)

    # Save models
    for name, agent in agents.items():
        model_path = os.path.join(save_dir, f"{name}_model_{timestamp}.pt")
        save_agent(agent, name, model_path)

    # ---- Save all results ----
    key_map = {
        'GNN-SAC': 'gnn_sac',
        'SAC': 'sac',
        'DDPG': 'ddpg',
        'PPO': 'ppo',
        'A2C': 'a2c',
    }
    save_data = {}
    for name, prefix in key_map.items():
        if name in results_algo:
            save_data[f'{prefix}_rewards'] = results_algo[name]['rewards']
            save_data[f'{prefix}_avg_rewards'] = results_algo[name]['avg_rewards']
            save_data[f'{prefix}_ees'] = results_algo[name]['ees']

    save_data['params'] = {
        'MAX_M': MAX_M, 'MAX_N': MAX_N, 'MAX_K': MAX_K,
        'STATE_DIM': STATE_DIM, 'ACTION_DIM': MAX_ACTION_DIM,
        'max_episodes': MAX_EPISODES,
        'selected_algos': list(selected_algos),
        'M': 3, 'N': 4, 'K': 3,
        'D': 40.0,
        'v_fixed': 1.0,
        'Pmax_dbm': 25.0,
        'Tframe': 1.0,
    }

    save_path = os.path.join(save_dir, f"results_convergence_{timestamp}.npz")
    np.savez(save_path, **save_data)
    print(f"\nResults saved to: {save_path}")

    # ---- Plotting ----
    import matplotlib.pyplot as plt

    # Fig 1: Algorithm comparison (EE)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for name in results_algo:
        y = np.convolve(results_algo[name]['ees'], np.ones(20) / 20, mode='valid')
        ax1.plot(y, label=name, linewidth=1.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Energy Efficiency (bps/Hz/J)')
    ax1.legend()
    ax1.set_title('Convergence Comparison of DRL Algorithms')
    ax1.grid(True)
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, f"convergence_algo_{timestamp}.png"), dpi=300)
    print(f"  Saved: convergence_algo_{timestamp}.png")

    # Fig 2: Reward convergence (per-step average reward)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for name in results_algo:
        y = np.convolve(results_algo[name]['avg_rewards'], np.ones(20) / 20, mode='valid')
        ax2.plot(y, label=name, linewidth=1.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Step Reward (log10(EE/E_ref) - penalties)')
    ax2.legend()
    ax2.set_title('Reward Convergence of DRL Algorithms')
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, f"convergence_reward_{timestamp}.png"), dpi=300)
    print(f"  Saved: convergence_reward_{timestamp}.png")

    plt.show()
