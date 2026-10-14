"""Graph policies for autoregressive SFC placement (pure PyTorch, batched).

Inputs for a batch of B decision steps on one substrate:
    x   [B, N, F]   node features      (FogEnv.observe)
    ef  [B, 2E, Fe] directed edge features
    ctx [B, Fc]     request / VNF context
    pos [B]         VNF position in the chain (used by separate actors)
    edge_index [2, 2E] shared by the batch

Encoders
    mlp   node-wise MLP (no topology)
    gcn   GCN, symmetric normalisation, no edge features (original R6 encoder)
    gine  GINE (edge-feature aware message passing)
    gatv2 GATv2 attention with edge features
Actor
    shared   one scoring head for every chain position (MAPPO parameter sharing)
    separate one head per chain position (original R6 design, ablation)
Critic
    centralised value on pooled graph embedding + request context.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .catalog import MAX_CHAIN_LEN


def mlp(i, h, o, layers=2):
    mods, d = [], i
    for _ in range(layers - 1):
        mods += [nn.Linear(d, h), nn.ReLU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """src [B, M, H], index [M] -> [B, n, H]"""
    out = src.new_zeros(src.size(0), n, src.size(2))
    return out.index_add_(1, index, src)


class NodeMLP(nn.Module):
    def __init__(self, f, fe, h, layers):
        super().__init__()
        self.inp = nn.Linear(f, h)
        self.layers = nn.ModuleList([mlp(h, h, h) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)])

    def forward(self, x, ef, ei):
        h = F.relu(self.inp(x))
        for f, nrm in zip(self.layers, self.norms):
            h = nrm(h + F.relu(f(h)))
        return h


class GCN(nn.Module):
    def __init__(self, f, fe, h, layers):
        super().__init__()
        self.inp = nn.Linear(f, h)
        self.lins = nn.ModuleList([nn.Linear(h, h) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)])

    def forward(self, x, ef, ei):
        B, N, _ = x.shape
        loop = torch.arange(N, device=x.device)
        src = torch.cat([ei[0], loop])
        dst = torch.cat([ei[1], loop])
        deg = torch.zeros(N, device=x.device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        w = (deg[src] * deg[dst]).rsqrt().view(1, -1, 1)
        h = F.relu(self.inp(x))
        for lin, nrm in zip(self.lins, self.norms):
            m = lin(h)[:, src] * w
            h = nrm(h + F.relu(scatter_sum(m, dst, N)))
        return h


class GINE(nn.Module):
    def __init__(self, f, fe, h, layers):
        super().__init__()
        self.inp = nn.Linear(f, h)
        self.edge = nn.ModuleList([nn.Linear(fe, h) for _ in range(layers)])
        self.upd = nn.ModuleList([mlp(h, h, h) for _ in range(layers)])
        self.eps = nn.Parameter(torch.zeros(layers))
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)])

    def forward(self, x, ef, ei):
        N = x.size(1)
        h = F.relu(self.inp(x))
        for k, (le, up, nrm) in enumerate(zip(self.edge, self.upd, self.norms)):
            m = F.relu(h[:, ei[0]] + le(ef))
            agg = scatter_sum(m, ei[1], N)
            h = nrm(h + F.relu(up((1 + self.eps[k]) * h + agg)))
        return h


class GATv2(nn.Module):
    def __init__(self, f, fe, h, layers, heads=4):
        super().__init__()
        assert h % heads == 0
        self.h, self.heads = h, heads
        self.inp = nn.Linear(f, h)
        self.ws = nn.ModuleList([nn.Linear(h, h) for _ in range(layers)])
        self.wd = nn.ModuleList([nn.Linear(h, h) for _ in range(layers)])
        self.we = nn.ModuleList([nn.Linear(fe, h, bias=False) for _ in range(layers)])
        self.att = nn.ParameterList([nn.Parameter(torch.randn(heads, h // heads) * 0.1) for _ in range(layers)])
        self.out = nn.ModuleList([nn.Linear(h, h) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(layers)])

    def forward(self, x, ef, ei):
        B, N, _ = x.shape
        H, K = self.h, self.heads
        loop = torch.arange(N, device=x.device)
        src = torch.cat([ei[0], loop])
        dst = torch.cat([ei[1], loop])
        ef = torch.cat([ef, ef.new_zeros(B, N, ef.size(2))], dim=1)
        h = F.relu(self.inp(x))
        for ws, wd, we, a, out, nrm in zip(self.ws, self.wd, self.we, self.att, self.out, self.norms):
            hs = ws(h)
            z = F.leaky_relu(hs[:, src] + wd(h)[:, dst] + we(ef), 0.2).view(B, -1, K, H // K)
            logit = (z * a).sum(-1)                                     # [B, M, K]
            # Softmax over the incoming edges of every destination node.
            idx = dst.view(1, -1, 1).expand(B, -1, K)
            mx = logit.new_full((B, N, K), -1e9).scatter_reduce_(1, idx, logit, "amax", include_self=True)
            e = torch.exp(logit - mx[:, dst])
            den = logit.new_zeros(B, N, K).index_add_(1, dst, e)
            alpha = (e / den[:, dst].clamp_min(1e-12)).unsqueeze(-1)    # [B, M, K, 1]
            msg = (alpha * hs[:, src].view(B, -1, K, H // K)).view(B, -1, H)
            h = nrm(h + F.relu(out(scatter_sum(msg, dst, N))))
        return h


ENCODERS = {"mlp": NodeMLP, "gcn": GCN, "gine": GINE, "gatv2": GATv2}


class PlacementPolicy(nn.Module):
    def __init__(self, node_f: int, edge_f: int, ctx_f: int, encoder: str = "gine",
                 hidden: int = 64, layers: int = 3, actor: str = "shared"):
        super().__init__()
        self.encoder_name, self.actor_mode = encoder, actor
        self.enc = ENCODERS[encoder](node_f, edge_f, hidden, layers)
        self.ctx = nn.Sequential(nn.Linear(ctx_f, hidden), nn.ReLU())
        n_heads = 1 if actor == "shared" else MAX_CHAIN_LEN
        self.heads = nn.ModuleList([mlp(3 * hidden + node_f, 2 * hidden, 1, layers=3) for _ in range(n_heads)])
        self.critic = mlp(3 * hidden, 2 * hidden, 1, layers=3)

    def forward(self, x, ef, ctx, mask, pos, edge_index):
        B, N, _ = x.shape
        h = self.enc(x, ef, edge_index)                           # [B, N, H]
        c = self.ctx(ctx)                                         # [B, H]
        g = h.mean(1)
        inp = torch.cat([h, x, c.unsqueeze(1).expand(-1, N, -1), g.unsqueeze(1).expand(-1, N, -1)], -1)
        if len(self.heads) == 1:
            logits = self.heads[0](inp).squeeze(-1)
        else:
            all_l = torch.stack([hd(inp).squeeze(-1) for hd in self.heads], 1)   # [B, P, N]
            idx = pos.clamp(max=len(self.heads) - 1).view(B, 1, 1).expand(-1, 1, N)
            logits = all_l.gather(1, idx).squeeze(1)
        logits = logits.masked_fill(~mask, -1e9)
        value = self.critic(torch.cat([g, h.max(1).values, c], -1)).squeeze(-1)
        return logits, value
