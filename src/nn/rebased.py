# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
https://github.com/corl-team/rebased/blob/main/flash_linear_attention/fla/layers/rebased_fast.py
"""

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from fla.modules.feature_map import RebasedFeatureMap
from fla.ops.linear_attn.utils import normalize_output
from fla.ops.simple_gla import chunk_simple_gla
from torch import Tensor

_original_checkpoint = torch.utils.checkpoint.checkpoint


def _checkpoint_wrapper(*args, **kwargs):
    kwargs["use_reentrant"] = False
    return _original_checkpoint(*args, **kwargs)


torch.utils.checkpoint.checkpoint = _checkpoint_wrapper


@torch.compiler.disable
def chunk_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    normalize: bool = True,
    head_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        scale (Optional[float]):
            Scale factor for the linear attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[B, H, K, V]`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[B, H, K, V]`. Default: `False`.
        normalize (bool):
            Whether to normalize the output. Default: `True`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[B, H, K, V]` if `output_final_state=True` else `None`.
    """

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
        )
    if not head_first:
        if q.shape[1] < q.shape[2]:
            raise DeprecationWarning(
                f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
                "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
                "when head_first=False was specified. "
                "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = chunk_simple_gla(
        q=q,  # type: ignore
        k=k,  # type: ignore
        v=v,  # type: ignore
        scale=scale,  # type: ignore
        initial_state=initial_state,  # type: ignore
        output_final_state=output_final_state,  # type: ignore
        cu_seqlens=cu_seqlens,  # type: ignore
    )
    if normalize:
        o = normalize_output(q * scale, k, o)  # type: ignore
    return o, final_state


def doc_ids_to_cu_seqlen(doc_ids: Tensor) -> Tensor:
    """
    Convert document IDs to cumulative sequence lengths for use in attention mechanisms.

    Args:
        doc_ids (Tensor): A tensor of shape (batch_size, seq_len) containing document IDs.

    Returns:
        Tensor: A 1D tensor of cumulative sequence lengths.
    """
    batch_size, seq_len = doc_ids.shape
    # Find where groups change
    is_new_group = torch.cat(
        [
            torch.ones_like(doc_ids[:, :1], dtype=torch.bool),
            doc_ids[:, 1:] != doc_ids[:, :-1],
        ],
        dim=1,
    )

    # Get the indices where new groups start
    group_start_indices = torch.where(is_new_group)[1]
    return torch.cat(
        [group_start_indices, torch.tensor([seq_len], device=doc_ids.device)]
    )


class ReBasedLinearAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        l_max: int = 2048,
        feature_dim: int = 16,
        num_key_value_heads: int = 16,
        num_heads: int = 16,
        use_gamma: bool | None = True,
        use_beta: bool | None = True,
        normalize: bool | None = True,
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.l_max = l_max

        self.feature_dim = feature_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_key_value_heads
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.normalize = normalize
        self.eps = eps

        self.feature_map = RebasedFeatureMap(
            self.feature_dim, use_gamma, use_beta, normalize
        )
        self.q_proj = nn.Linear(
            self.hidden_size, self.feature_dim * self.num_heads, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.feature_dim * self.num_heads, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.dropout = nn.Identity()

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        q = rearrange(
            self.q_proj(hidden_states),
            "... (h d) -> ... h d",
            h=self.num_heads,
            d=self.feature_dim,
        )
        k = rearrange(
            self.k_proj(hidden_states),
            "... (h d) -> ... h d",
            h=self.num_heads,
            d=self.feature_dim,
        )
        v = rearrange(
            self.v_proj(hidden_states),
            "... (h d) -> ... h d",
            h=self.num_key_value_heads,
            d=self.head_dim,
        )
        cu_seqlens = kwargs.get("cu_seqlens", None)
        seq_idx = kwargs.get("seq_idx", None)
        if cu_seqlens is None and seq_idx is not None:
            cu_seqlens = doc_ids_to_cu_seqlen(seq_idx)
        q, k = self.feature_map(q, flatten=True), self.feature_map(k, flatten=True)
        o, _ = chunk_linear_attn(
            q=q,  # type: ignore
            k=k,  # type: ignore
            v=v,  # type: ignore
            normalize=True,  # type: ignore
            scale=1,  # type: ignore
            cu_seqlens=cu_seqlens,  # type: ignore
        )
        o = rearrange(o, "... h d -> ... (h d)")
        o = self.o_proj(o)
        o = self.dropout(o)
        return o
