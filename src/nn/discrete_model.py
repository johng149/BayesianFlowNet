import math

import torch
from mamba_ssm import Mamba2
from torch import Tensor, nn
from torch.nn.attention.flex_attention import create_block_mask

from src.nn.chunker import PackDynamicSequenceChunker
from src.nn.flex_transformer import TransformerBlock, causal, generate_doc_mask_mod
from src.training.loss import LossContext
from src.training.loss import loss as loss_fn


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


class DiscreteModel(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        K: int,
        hidden_dim: int,
        num_heads: int,
        mamba_expand: int = 2,
        layers: int = 3,
        dropout: float = 0.1,
        use_chunkers: bool = True,
        mcmc_steps: int = 2,
        aux_loss_weight: float = 0.03,
        mcmc_step_size: float = 0.01,
    ):
        super().__init__()
        mamba_check(hidden_dim, num_heads, mamba_expand) if use_chunkers else None
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisble by num_heads"
        self.headdim = hidden_dim // num_heads
        self.use_chunkers = use_chunkers
        self.mcmc_steps = mcmc_steps
        self.aux_loss_weight = aux_loss_weight
        self.mcmc_step_size = mcmc_step_size

        self.mcmc_alpha = nn.Parameter(
            torch.tensor(self.mcmc_step_size), requires_grad=True
        )

        self.num_layers = layers + 2  # account for pre and post chunker layers
        self.emb = nn.Parameter(torch.randn(K, hidden_dim) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, hidden_dim) * 0.02)
        self.mcmc_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.time_rotary = SinusoidalTimeEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Pre-chunk Mamba
        self.pre_chunker = (
            torch.compiler.disable(
                Mamba2(
                    d_model=hidden_dim,
                    headdim=self.headdim,
                    d_state=16,
                    d_conv=4,
                    expand=mamba_expand,
                )
            )
            if use_chunkers
            else nn.Identity()
        )

        # Chunker
        self.chunker = (
            PackDynamicSequenceChunker(dim=hidden_dim)
            if use_chunkers
            else nn.Identity()
        )

        # Main Transformer (Flex Attention)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim,
                    heads=num_heads,
                    dim_head=self.headdim,
                    num_layers=layers,  # used for weight init scaling
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

        # Post-chunk Mamba
        self.post_chunker = (
            torch.compiler.disable(
                Mamba2(
                    d_model=hidden_dim,
                    headdim=self.headdim,
                    d_state=16,
                    d_conv=4,
                    expand=mamba_expand,
                )
            )
            if use_chunkers
            else nn.Identity()
        )

        # in this case, the classifier is actually the energy head. Taking the negative of
        # its output (the energy) gives the actual logits
        self.classifier = nn.Parameter(torch.randn(hidden_dim, K) * 0.02)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

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

    def mcmc_emb(self, x: Tensor, step: int) -> Tensor:
        return x + self.mcmc_embedding * step

    def forward(self, x, t, mask, doc_ids, targets: LossContext | None = None):
        batch_size, seq_len, K = x.shape
        assert mask.shape == (
            batch_size,
            seq_len,
        ), f"mask shape {mask.shape} does not match input shape {x.shape} on the batch and seq len dimensions"
        assert doc_ids.shape == (
            batch_size,
            seq_len,
        ), f"doc_ids shape {doc_ids.shape} does not match input shape {x.shape} on the batch and seq len dimensions"

        doc_ids = doc_ids.int()

        unique_doc_ids, seq_lens = torch.unique_consecutive(doc_ids, return_counts=True)

        x = self.token_emb(x)
        x = self.positional_emb(x, doc_ids)
        x = self.time_emb(x, t, mask)
        x = self.mcmc_emb(x, 0)

        loss = 0.0

        for step in range(self.mcmc_steps):
            x = x.detach().requires_grad_(True)

            # Pre Chunker
            x = self.pre_chunker(x, seq_idx=doc_ids) if self.use_chunkers else x  # type: ignore

            # Chunker
            outputs, intermediates = (
                self.chunker(x.squeeze(0), seq_lens=seq_lens, return_intermediates=True)
                if self.use_chunkers
                else (x.squeeze(0), None)
            )
            x_down = (
                outputs.downsampled.unsqueeze(  # pyright: ignore[reportAttributeAccessIssue]
                    0
                )
                if self.use_chunkers
                else outputs.unsqueeze(0)
            )  # (1, total_len, D)

            with torch.no_grad():
                if self.use_chunkers:
                    assert (
                        intermediates is not None
                    ), "Intermediates should not be None when using chunkers"
                    new_seq_lens = intermediates.new_seq_lens  # (Batch_Size,)
                    doc_ids_down = torch.repeat_interleave(unique_doc_ids, new_seq_lens)
                else:
                    doc_ids_down = doc_ids

            # Generate mask
            mask_mod = generate_doc_mask_mod(None, doc_ids_down)
            block_mask = create_block_mask(
                mask_mod,
                B=None,
                H=None,
                Q_LEN=x_down.shape[1],
                KV_LEN=x_down.shape[1],
                device=x_down.device,
            )

            for block in self.transformer_blocks:
                x_down = block(x_down, block_mask)

            packed_out = x_down.squeeze(0)  # (TotalChunks, D)

            # Upsample via Chunker
            x_up = (
                outputs.upsample_fn(  # pyright: ignore[reportAttributeAccessIssue]
                    packed_out
                )
                if self.use_chunkers
                else packed_out
            )  # (S, D)

            # Mamba Post
            x = self.post_chunker(x_up.unsqueeze(0), seq_idx=doc_ids) if self.use_chunkers else x_up.unsqueeze(0)  # type: ignore

            pred = x @ self.classifier

            energy, aux_loss = pred, (
                outputs.weighted_aux_ratio_loss  # pyright: ignore[reportAttributeAccessIssue]
                if self.use_chunkers
                else 0.0
            )
            logits = -energy

            grad = torch.autograd.grad(energy.sum(), x, create_graph=self.training)[0]
            x = x - self.mcmc_alpha * grad

            if targets is not None:
                loss += loss_fn(targets, logits, aux_loss, self.aux_loss_weight)
        return logits, loss / self.mcmc_steps
