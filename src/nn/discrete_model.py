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
        dec_max_seq_len: int,
        enc_max_seq_len: int,
        K: int,
        hidden_dim: int,
        num_heads: int,
        decoder_layers: int = 3,
        encoder_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisble by num_heads"
        self.num_decoder_layers = decoder_layers
        self.num_encoder_layers = encoder_layers

        self.dec_emb = nn.Parameter(torch.randn(K, hidden_dim) * 0.02)
        self.enc_emb = nn.Embedding(K, hidden_dim)

        self.dec_pos_emb = nn.Parameter(torch.randn(dec_max_seq_len, hidden_dim) * 0.02)
        self.enc_pos_emb = nn.Parameter(torch.randn(enc_max_seq_len, hidden_dim) * 0.02)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.encoder_layers = nn.ModuleList(
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
                for i in range(encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
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
                for i in range(decoder_layers)
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
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _residual_scaling(self):
        scale = 0.02 / (2 * self.num_decoder_layers) ** 0.5
        layers = self.encoder_layers + self.decoder_layers
        for layer in layers:
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
            elif isinstance(layer, nn.TransformerDecoderLayer):
                # 1. MLP output projection (linear2)
                torch.nn.init.normal_(layer.linear2.weight, mean=0.0, std=scale)
                if layer.linear2.bias is not None:
                    torch.nn.init.zeros_(layer.linear2.bias)

                # 2. Self-Attention output projection (self_attn.out_proj)
                if hasattr(layer.self_attn, "out_proj"):
                    torch.nn.init.normal_(
                        layer.self_attn.out_proj.weight, mean=0.0, std=scale
                    )
                    if layer.self_attn.out_proj.bias is not None:
                        torch.nn.init.zeros_(layer.self_attn.out_proj.bias)

                # 3. Cross-Attention output projection (multihead_attn.out_proj)
                if hasattr(layer.multihead_attn, "out_proj"):
                    torch.nn.init.normal_(
                        layer.multihead_attn.out_proj.weight, mean=0.0, std=scale
                    )
                    if layer.multihead_attn.out_proj.bias is not None:
                        torch.nn.init.zeros_(layer.multihead_attn.out_proj.bias)

    def dec_token_emb(self, x):
        return x @ self.dec_emb

    def enc_token_emb(self, x):
        return self.enc_emb(x)

    def dec_positional_emb(self, x):
        return x + self.dec_pos_emb[: x.shape[1]]

    def enc_positional_emb(self, x):
        return x + self.enc_pos_emb[: x.shape[1]]

    def time_emb(self, x, t):
        assert (
            t.ndim == 1
        ), f"time vector `t` should be vector of length batch_size. Got shape {t.shape} while x has shape {x.shape}"
        time_embedding = self.time_mlp(t)
        return x + time_embedding.unsqueeze(1)

    def forward(self, x, t, c):
        x = self.dec_token_emb(x)
        c = self.enc_token_emb(c)

        x = self.dec_positional_emb(x)
        c = self.enc_positional_emb(c)

        x = self.time_emb(x, t)

        for i, l in enumerate(self.encoder_layers):
            c = l(c)

        # there is an issue with cross attention, so we cannot use decoder layers
        # as a kludge, we combine x and c along the sequence dimension and use encoder layers
        b, x_seq, k = x.shape
        _, c_seq, _ = c.shape
        x = torch.cat([c, x], dim=1)

        assert x.shape == (b, x_seq + c_seq, k)

        for i, l in enumerate(self.decoder_layers):
            x = l(x)

        # however for prediction, we only take the part corresponding to x
        x = x[:, c_seq:, :]

        pred = x @ self.classifier

        return pred
