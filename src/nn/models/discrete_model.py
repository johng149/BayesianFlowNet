import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_


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
        self.emb = nn.Parameter(torch.randn(K, hidden_dim))
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, hidden_dim))
        self.time_vec = nn.Parameter(torch.randn(1, hidden_dim))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hidden_dim,
                    num_heads,
                    hidden_dim * 4,
                    dropout,
                    activation=F.leaky_relu,
                    batch_first=True,
                    bias=False,
                )
                for i in range(layers)
            ]
        )
        self.classifier = nn.Parameter(torch.randn(hidden_dim, K))

    def _init_weights(self):
        xavier_uniform_(self.emb)
        xavier_uniform_(self.pos_emb)
        xavier_uniform_(self.time_vec)
        xavier_uniform_(self.classifier)

        for layer in self.layers:
            for name, param in layer.named_parameters():
                if "weight" in name:
                    xavier_uniform_(param)

    def token_emb(self, x):
        return x @ self.emb

    def positional_emb(self, x):
        return x + self.pos_emb[: x.shape[1]]

    def time_emb(self, x, t):
        assert t.ndim == 1, "time vector `t` should be vector of length batch_size"
        # we need to first unsqueeze t to get it from shape (batch_size,)
        # to (batch_size, 1) so it is compatible with the time_vec's (1, hidden_dim)
        # the result is (batch_size, hidden_dim) however the x is
        # (batch_size, seq_len, hidden_dim) so we need a second unsqueeze
        return (t.unsqueeze(-1) @ self.time_vec).unsqueeze(-2) + x

    def forward(self, x, t):
        x = self.token_emb(x)
        x = self.positional_emb(x)
        x = self.time_emb(x, t)
        for i, l in enumerate(self.layers):
            x = l.forward(x)
        return x @ self.classifier
