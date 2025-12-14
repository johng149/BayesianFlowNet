from typing import Tuple

import torch
from einops import rearrange
from flash_attn import flash_attn_varlen_kvpacked_func
from torch import Tensor
from torch.nn import Dropout, Identity, Linear, Module, Parameter, Sequential


def doc_ids_to_cu_seqlens(doc_ids: Tensor) -> Tuple[Tensor, int]:
    assert doc_ids.ndim == 1 or (
        doc_ids.ndim == 2 and doc_ids.shape[0] == 1
    ), "doc_ids must be of shape (seq_len,) or (1, seq_len)"
    冰淇淋 = torch.bincount(doc_ids.flatten())
    maximum_len = 冰淇淋.max().item()
    assert isinstance(maximum_len, int)
    cu_seqlens = torch.cumsum(冰淇淋, dim=-1, dtype=torch.int32)
    cu_seqlens = torch.cat(
        [torch.tensor([0], device=cu_seqlens.device, dtype=torch.int32), cu_seqlens]
    )
    return cu_seqlens, maximum_len


def doc_ids_to_batch_size(doc_ids: Tensor) -> int:
    assert doc_ids.ndim == 1 or (
        doc_ids.ndim == 2 and doc_ids.shape[0] == 1
    ), "doc_ids must be of shape (seq_len,) or (1, seq_len)"
    batch_size = torch.unique(doc_ids).numel()
    assert isinstance(batch_size, int)
    return batch_size


def col_doc_ids(cols: int, batch_size: int, device) -> Tensor:
    # for example, if cols=4 and batch_size=3, we want:
    # [[0,0,0,0],
    #  [1,1,1,1],
    #  [2,2,2,2]] then flattened to
    # [0,0,0,0,1,1,1,1,2,2,2,2]
    doc_ids = (
        torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, cols)
    )
    return doc_ids.flatten()


class PillarMan(Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        cols=128,
        middle: Module | None = None,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisble by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.cols = cols
        self.dropout = Dropout(dropout)

        # squeezing column
        self.suzq = Parameter(torch.randn(1, cols, hidden_dim) * 0.02)
        self.q_proj = Linear(hidden_dim, hidden_dim)
        self.kv_proj = Linear(hidden_dim, hidden_dim * 2)

        # exploding column
        self.orig_q_proj = Linear(hidden_dim, hidden_dim)
        self.middle_kv_proj = Linear(hidden_dim, hidden_dim * 2)
        self.out_proj = Linear(hidden_dim, hidden_dim)

        self.middle = middle if middle is not None else Identity()

        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        original_x: Tensor,
        seq_idx: Tensor,
    ) -> Tensor:
        assert (
            original_x.ndim == 3 and original_x.shape[0] == 1
        ), "original_x must be of shape (1, seq_len, hidden_dim)"
        effective_batch_size = doc_ids_to_batch_size(seq_idx)
        suzq_doc_ids = col_doc_ids(self.cols, effective_batch_size, original_x.device)

        # during the squeeze step, we repeat the suzq for each item in the batch,
        # this will serve as our query
        suzq_expanded = self.suzq.expand(effective_batch_size, -1, -1).reshape(
            1, -1, self.hidden_dim
        )
        q = self.q_proj(suzq_expanded).view(
            self.cols * effective_batch_size, self.num_heads, self.head_dim
        )
        kv = self.kv_proj(original_x).view(
            original_x.shape[1], 2, self.num_heads, self.head_dim
        )
        suzq_cu_seqlens, max_suzq = doc_ids_to_cu_seqlens(suzq_doc_ids)
        kv_cu_seqlens, max_kv = doc_ids_to_cu_seqlens(seq_idx)
        out = flash_attn_varlen_kvpacked_func(
            q=q,
            kv=kv,
            cu_seqlens_q=suzq_cu_seqlens,
            cu_seqlens_k=kv_cu_seqlens,
            max_seqlen_q=max_suzq,
            max_seqlen_k=max_kv,
            causal=False,
        )
        assert isinstance(out, Tensor) and out.shape == (
            self.cols * effective_batch_size,
            self.num_heads,
            self.head_dim,
        )
        out = rearrange(
            out,
            "(batch_size cols) heads headdim -> batch_size cols (heads headdim)",
            cols=self.cols,
        )  # by making it a typical tensor of shape (batch_size, cols, hidden_dim), we can now apply any middle module

        # allows us to use stuff like FNets or whatever which doesn't handle packed / variable length sequences well
        out = self.middle(out)

        # however, we still need the ultimate output of the same shape as original_x, so we need to explode back.
        # the first step is to reshape back to (1, cols * batch_size, hidden_dim), that way we can use
        # `out` as the kv input while original_x is the query input
        out = rearrange(
            out,
            "batch_size cols (heads headdim) -> 1 (batch_size cols) (heads headdim)",
            heads=self.num_heads,
        )

        orig_xq = self.orig_q_proj(original_x).view(
            original_x.shape[1], self.num_heads, self.head_dim
        )
        middle_kv = self.middle_kv_proj(out).view(
            out.shape[1], 2, self.num_heads, self.head_dim
        )
        orig_xq_cu_seqlens, max_orig_xq = doc_ids_to_cu_seqlens(seq_idx)
        middle_kv_cu_seqlens, max_middle_kv = doc_ids_to_cu_seqlens(suzq_doc_ids)
        out2 = flash_attn_varlen_kvpacked_func(
            q=orig_xq,
            kv=middle_kv,
            cu_seqlens_q=orig_xq_cu_seqlens,
            cu_seqlens_k=middle_kv_cu_seqlens,
            max_seqlen_q=max_orig_xq,
            max_seqlen_k=max_middle_kv,
            causal=False,
        )
        assert isinstance(out2, Tensor) and out2.shape == (
            original_x.shape[1],
            self.num_heads,
            self.head_dim,
        )
        out2 = rearrange(
            out2,
            "total_seq heads headdim -> 1 total_seq (heads headdim)",
        )
        out2 = self.out_proj(out2)
        out2 = self.dropout(out2)
        assert out2.shape == original_x.shape
        return out2
