import math

import torch
from mamba_ssm import Mamba2
from torch import Tensor, nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=t.dtype) * -emb)
        emb = t.unsqueeze(-1) * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def mamba_check(hidden_dim, num_heads, mamba_expand):
    headdim = hidden_dim // num_heads
    if (hidden_dim * mamba_expand / headdim) % 8 != 0:
        gcd = math.gcd(mamba_expand, 8)
        step = 8 // gcd
        n_low = (num_heads // step) * step
        n_high = n_low + step
        candidates = [n for n in [n_low, n_high] if n > 0]
        suggestions = []
        for n in candidates:
            h = int(round(hidden_dim / n) * n)
            suggestions.append((h, n))
        suggestions.sort(key=lambda x: (abs(x[0] - hidden_dim), abs(x[1] - num_heads)))
        best_h, best_n = suggestions[0]
        raise ValueError(
            f"Mamba packed sequence constraint failed: (hidden_dim * expand / headdim) % 8 != 0.\n"
            f"Current: hidden_dim={hidden_dim}, num_heads={num_heads}, expand={mamba_expand}.\n"
            f"Suggested fix: hidden_dim={best_h}, num_heads={best_n}."
        )


class EBM(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        K: int,
        hidden_dim: int,
        num_heads: int,
        mamba_expand: int = 2,
        dropout: float = 0.1,
        stepsize: float = 0.01,
        max_grad_change: float = 9.0,
        steps: int = 2,
        langevin_noise: float = 0.005,
    ):
        super().__init__()
        mamba_check(hidden_dim, num_heads, mamba_expand)
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisble by num_heads"
        assert steps >= 1, "steps must be at least 1"
        self.headdim = hidden_dim // num_heads
        self.max_grad_change = max_grad_change
        self.steps = steps
        self.langevin_noise = langevin_noise

        self.emb = nn.Parameter(torch.randn(K, hidden_dim) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, hidden_dim) * 0.02)
        self.time_rotary = SinusoidalTimeEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.RMSNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.mixer = torch.compiler.disable(
            Mamba2(
                d_model=hidden_dim,
                headdim=self.headdim,
                d_state=16,
                d_conv=4,
                expand=mamba_expand,
            )
        )

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.RMSNorm(hidden_dim)

        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 3),
            nn.GELU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
        )

        # the energy head outputs energy for each of the K discrete tokens
        # since lower energy means the token position is more correct, to get
        # the actual logits, we need to negate the energy outputs. For example,
        # if a token is assigned -10,000 energy it is very likely to be the correct token
        # which means the logit should be 10,000 (very large positive number)
        self.energy_head = nn.Linear(hidden_dim, K)
        self.mcmc_alpha = nn.Parameter(torch.tensor(stepsize), requires_grad=False)

    def token_emb(self, x):
        return x @ self.emb

    def positional_emb(self, x, doc_ids):
        is_new_group = torch.cat(
            [
                torch.ones_like(doc_ids[:, :1], dtype=torch.bool),
                doc_ids[:, 1:] != doc_ids[:, :-1],
            ],
            dim=1,
        )
        # Get the indices where new groups start
        group_start_indices = torch.where(is_new_group)[1]

        # Broadcast subtraction: subtract the start index of current group from position
        positions = (
            torch.arange(doc_ids.size(1), device=doc_ids.device)
            .unsqueeze(0)
            .expand_as(doc_ids)
        )
        group_starts = torch.zeros_like(positions)
        group_starts[:, is_new_group[0]] = group_start_indices
        group_starts = group_starts.cummax(dim=1)[0]  # Forward fill the start indices
        positions = positions - group_starts  # (1, seq_len)

        pos_embedding = self.pos_emb[positions]  # shape is (1, total_len, hidden_dim)
        return x + pos_embedding

    def time_emb(self, x, t, mask):
        assert (
            t.ndim == 2 and t.shape[0] == x.shape[0] and t.shape[1] == x.shape[1]
        ), f"time vector `t` should be vector of length (1, total_len). Got shape {t.shape} while x has shape {x.shape}"

        # for all positions that are not masked, set time to 1 (no noise)
        time_rotary = self.time_rotary(torch.where(mask == False, 1, t))
        time_embedding = self.time_mlp(
            time_rotary
        )  # shape is (1, total_len, hidden_dim)

        return x + time_embedding

    def compute_energy(self, original_x, t, mask, doc_ids) -> Tensor:
        x = self.token_emb(original_x)
        x = self.positional_emb(x, doc_ids)
        x = self.time_emb(x, t, mask)
        x = self.norm(x)

        residual = x

        x = self.dropout(x)
        x = self.mixer(x, seq_idx=doc_ids)  # pyright: ignore[reportCallIssue]
        x = x + residual
        x = self.dropout2(x)
        x = x + self.mlp_out(self.norm2(x))

        return self.energy_head(x)

    def forward(self, original_x, t, mask, doc_ids):
        with torch.enable_grad():
            doc_ids = doc_ids.int()
            energy = self.compute_energy(original_x, t, mask, doc_ids)
            logits = -energy
            for step in range(self.steps):
                logits = logits.requires_grad_(True)
                probs = F.softmax(logits, dim=-1)

                # recalculate energy based on current probs
                energy = self.compute_energy(probs, t, mask, doc_ids)
                expected_energy = (probs * energy).sum(dim=-1).mean()
                grad = torch.autograd.grad(
                    expected_energy, logits, create_graph=self.training
                )[0]

                if self.training:
                    # during training, jitter the step size a little bit
                    jitter = 1.0 + 0.1 * (
                        torch.rand(1, device=grad.device) - 0.5
                    )  # in [0.95, 1.05]
                    step_size = self.mcmc_alpha * jitter
                else:
                    step_size = self.mcmc_alpha
                min_max = self.max_grad_change / step_size
                clamped_grad = torch.clamp(grad, min=-min_max, max=min_max)
                logits = logits - step_size * clamped_grad

                if self.training:
                    noise_scale = self.langevin_noise * (1.0 - step / self.steps)
                    logits = logits + torch.randn_like(logits) * noise_scale

                logits = logits - logits.mean(dim=-1, keepdim=True)

            # the main BFN expects inputs to be between 0 and 1, and since updated_x is effectively
            # logits here, we just apply softmax
            return (
                torch.softmax(logits, dim=-1),
                logits,
                expected_energy,  # step is at least 1 so guaranteed to be defined
            )  # this updated_x is used for loss computation
