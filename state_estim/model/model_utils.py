"""
Minimal model building blocks used by the initializer.

Only includes MCG_block, CG_stacked, and MLP_3.
"""

import torch
import torch.nn as nn


class MCG_block(nn.Module):
    """Multiplicative Context Gating block."""

    def __init__(self, hidden_dim):
        super(MCG_block, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, inp, context, mask):
        context = context.unsqueeze(1)
        mask = mask.unsqueeze(-1)

        inp = self.MLP(inp)
        inp = inp * context
        inp = inp.masked_fill(mask == 0, torch.tensor(-1e9))
        context = torch.max(inp, dim=1)[0]
        return inp, context


class CG_stacked(nn.Module):
    """Stack of MCG blocks with running-mean aggregation."""

    def __init__(self, stack_num, hidden_dim):
        super(CG_stacked, self).__init__()
        self.CGs = nn.ModuleList()
        self.stack_num = stack_num
        for _ in range(stack_num):
            self.CGs.append(MCG_block(hidden_dim))

    def forward(self, inp, context, mask):
        inp_, context_ = self.CGs[0](inp, context, mask)
        for i in range(1, self.stack_num):
            inp, context = self.CGs[i](inp_, context_, mask)
            inp_ = (inp_ * i + inp) / (i + 1)
            context_ = (context_ * i + context) / (i + 1)
        return inp_, context_


class MLP_3(nn.Module):
    """Three-layer MLP with LayerNorm + ReLU."""

    def __init__(self, dims):
        super(MLP_3, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.LayerNorm(dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.LayerNorm(dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], dims[3]),
        )

    def forward(self, x):
        return self.mlp(x)


