import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiscreteModel(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        K: int,
        hidden_dim: int,
        num_heads: int,
        layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisble by num_heads"
        self.num_layers = layers
        self.emb = nn.Parameter(torch.randn(K, hidden_dim) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, hidden_dim) * 0.02)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hidden_dim,
                    num_heads,
                    hidden_dim * 4,
                    dropout,
                    batch_first=True,
                    norm_first=True,
                    bias=False,
                )
                for i in range(layers)
            ]
        )
        self.classifier = nn.Parameter(torch.randn(hidden_dim, K) * 0.02)

        self.apply(self._init_weights)
        self._residual_scaling()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _residual_scaling(self):
        scale = 0.02 / (2 * self.num_layers) ** 0.5
        for layer in self.layers:
            if isinstance(layer, nn.TransformerEncoderLayer):
                # 1. MLP output projection (linear2)
                torch.nn.init.normal_(layer.linear2.weight, mean=0.0, std=scale)
                if layer.linear2.bias is not None:
                    torch.nn.init.zeros_(layer.linear2.bias)

                # 2. Attention output projection (self_attn.out_proj)
                # nn.MultiheadAttention stores the output projection in `out_proj`
                if hasattr(layer.self_attn, "out_proj"):
                    torch.nn.init.normal_(
                        layer.self_attn.out_proj.weight, mean=0.0, std=scale
                    )
                    if layer.self_attn.out_proj.bias is not None:
                        torch.nn.init.zeros_(layer.self_attn.out_proj.bias)

    def token_emb(self, x):
        return x @ self.emb

    def positional_emb(self, x):
        return x + self.pos_emb[: x.shape[1]]

    def time_emb(self, x, t):
        assert (
            t.ndim == 1
        ), f"time vector `t` should be vector of length batch_size. Got shape {t.shape} while x has shape {x.shape}"
        time_embedding = self.time_mlp(t)
        return x + time_embedding.unsqueeze(1)

    def forward(self, x, t):
        x = self.token_emb(x)
        x = self.positional_emb(x)
        x = self.time_emb(x, t)
        for i, l in enumerate(self.layers):
            x = l(x)
        pred = x @ self.classifier

        return pred
