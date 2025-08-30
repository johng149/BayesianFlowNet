import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_

from src.datasets.discrete_helper import theta, y_distribution
from src.nn.layers.learnable_schedule import LearnableBetaScheduleNI


class ModelBody(nn.Module):
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
        self.learnable_beta = LearnableBetaScheduleNI()
        self.body = ModelBody(max_seq_len, K, hidden_dim, num_heads, layers, dropout)

    def _init_weights(self):
        xavier_uniform_(self.body.emb)
        xavier_uniform_(self.body.pos_emb)
        xavier_uniform_(self.body.time_vec)
        xavier_uniform_(self.body.classifier)

        for layer in self.body.layers:
            for name, param in layer.named_parameters():
                if "weight" in name:
                    xavier_uniform_(param)

    def beta_1(self, K: int, device: str) -> float:
        return self.learnable_beta.beta_1(K, device)

    def scaling(self, K: int, device: str) -> float:
        return self.learnable_beta.scaling(K, device)

    def beta(self, t: Tensor, K: int) -> Tensor:
        return self.learnable_beta(t, K)

    def beta_and_alpha(
        self, t: Tensor, K: int, epsilon: float = 1e-8
    ) -> tuple[Tensor, Tensor]:
        return self.learnable_beta.get_alpha(t, K, epsilon)

    def theta_input(self, x: Tensor, t: Tensor, beta: Tensor) -> Tensor:
        """
        Args:
            x: Ground truth tensor of shape (batch_size, seq_len, K).
            t: A tensor representing the time step of shape (batch_size,).
            beta: Beta value at the given time step t of shape (batch_size,).
        Returns:
            Transformed version of x, which is the input to the model.
            The shape of the output tensor is the same as x, i.e., (batch_size, seq_len, K).
        """
        assert x.ndim == 3, "x should be a 3D tensor of shape (batch_size, seq_len, K)"
        assert t.ndim == 1, "t should be a 1D tensor of shape (batch_size,)"
        assert beta.ndim == 1, "beta should be a 1D tensor of shape (batch_size,)"

        y = y_distribution(beta, x.shape[-1], x)  # Shape: (batch_size, seq_len, K)
        theta_tensor = theta(y)  # Shape: (batch_size, seq_len, K)
        return theta_tensor

    def token_emb(self, x):
        return self.body.token_emb(x)

    def positional_emb(self, x):
        return self.body.positional_emb(x)

    def time_emb(self, x, t):
        return self.body.time_emb(x, t)

    def forward(self, x, t):
        """
        At this point in time, `x` is still the ground truth tensor. The model will
        create the appropriate inputs as a function of this ground truth and the current
        time step `t` in the denoising process

        Args:
            x: Ground truth tensor of shape (batch_size, seq_len, K).
            t: A tensor representing the time step of shape (batch_size,).
        Returns:
            The output logits of the model of shape (batch_size, seq_len, K).
        """
        batch_folds, seq_len, K = x.shape
        beta, alpha = self.beta_and_alpha(
            t, K
        )  # Shape: (batch_size * folds,) for each tensor
        x = self.theta_input(x, t, beta)  # Shape: (batch_size * folds, seq_len, K)
        x = self.body(x, t)
        return (x, alpha)
