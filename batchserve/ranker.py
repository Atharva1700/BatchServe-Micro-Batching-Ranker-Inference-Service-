"""Ranker and item features. Weights are fixed random values (seeded): no trained model is available,
so this measures serving behaviour, not ranking quality. Popularity is a Zipf prior, used as fallback."""
import numpy as np
import torch
import torch.nn as nn

N_ITEMS = 50_000
N_FEAT = 32


class Ranker(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(N_FEAT, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, nn.Linear):
                    m.weight.copy_(torch.randn(m.weight.shape, generator=g) / m.in_features ** 0.5)
                    m.bias.zero_()

    def forward(self, x):
        return self.net(x).squeeze(-1)


def item_table(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_ITEMS, N_FEAT)).astype(np.float32)


def popularity(seed=0):
    rng = np.random.default_rng(seed + 1)
    ranks = rng.permutation(N_ITEMS) + 1
    return (1.0 / ranks).astype(np.float32)
