import math

import torch
from einops import rearrange
from torch import Tensor
from torch.nn import Linear, Module, Parameter, RMSNorm
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from src.nn.monarch_linear import MonarchLinear

# --- Flex Attention Utils ---


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def generate_doc_mask_mod(mask_mod, document_id):
    # can feed in another mask modifier function such as `causal` or None
    assert document_id.ndim == 1 or (
        document_id.ndim == 2 and document_id.shape[0] == 1
    )
    doc_id = document_id.view(-1)

    # Get unique document IDs and their counts
    _, counts = torch.unique_consecutive(doc_id, return_counts=True)
    # Create cumulative counts (offsets)
    offsets = torch.cat(
        [torch.tensor([0], device=doc_id.device), counts.cumsum(0)[:-1]]
    )

    if mask_mod is not None:

        def doc_mask_wrapper(b, h, q_idx, kv_idx):
            same_doc = doc_id[q_idx] == doc_id[kv_idx]
            q_logical = q_idx - offsets[doc_id[q_idx]]
            kv_logical = kv_idx - offsets[doc_id[kv_idx]]
            inner_mask = mask_mod(b, h, q_logical, kv_logical)
            return same_doc & inner_mask

        return doc_mask_wrapper

    else:

        def doc_mask_wrapper_solo(b, h, q_idx, kv_idx):
            same_doc = doc_id[q_idx] == doc_id[kv_idx]
            q_logical = q_idx - offsets[doc_id[q_idx]]
            kv_logical = kv_idx - offsets[doc_id[kv_idx]]
            return same_doc

        return doc_mask_wrapper_solo


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class TransformerBlock(Module):
    def __init__(self, dim, heads, dim_head, depth, num_layers=1, dropout=0.0):
        super().__init__()
        self.depth = depth
        self.num_layers = num_layers
        self.norm1 = RMSNorm(dim)
        self.heads = heads
        self.dim_head = dim_head
        self.head_dim = dim // heads

        self.to_qkv = MonarchLinear(dim, heads * dim_head * 3 * 2, bias=False)
        self.to_out = MonarchLinear(heads * dim_head, dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))

        self.diff_norm = RMSNorm(self.head_dim)

        self.norm2 = RMSNorm(dim)
        self.ff = torch.nn.Sequential(
            MonarchLinear(dim, dim * 4, bias=False),
            torch.nn.GELU(),
            MonarchLinear(dim * 4, dim, bias=False),
        )
        self.flex = torch.compile(flex_attention)
        self.dropout = torch.nn.Dropout(dropout)

        # self.init_weights()

    def init_weights(self):
        scale = 0.02 / (2 * self.num_layers) ** 0.5

        # # Attention output projection
        # torch.nn.init.normal_(self.to_out.weight, mean=0.0, std=scale)
        # if self.to_out.bias is not None:
        #     torch.nn.init.zeros_(self.to_out.bias)

        # # QKV projection
        # torch.nn.init.normal_(self.to_qkv.weight, mean=0.0, std=scale)
        # if self.to_qkv.bias is not None:
        #     torch.nn.init.zeros_(self.to_qkv.bias)

        # # MLP output projection
        # # self.ff is Sequential(Linear, GELU, Linear)
        # mlp_in = self.ff[0]
        # mlp_out = self.ff[2]

        # assert isinstance(mlp_in, Linear) and isinstance(mlp_out, Linear)

        # torch.nn.init.normal_(mlp_out.weight, mean=0.0, std=scale)
        # if mlp_out.bias is not None:
        #     torch.nn.init.zeros_(mlp_out.bias)

        # torch.nn.init.normal_(mlp_in.weight, mean=0.0, std=scale)
        # if mlp_in.bias is not None:
        #     torch.nn.init.zeros_(mlp_in.bias)

    def forward(self, x, block_mask):
        # x: (1, SeqLen, Dim) - treating packed as batch 1
        B, S, D = x.shape

        residual = x
        x = self.norm1(x)

        qkv = self.to_qkv(x)  # (B, S, 2 * 3 * H * Dh)
        qkv = rearrange(
            qkv, "b s (n t h d) -> n t b h s d", n=2, t=3, h=self.heads, d=self.dim_head
        )  # (2, 3, B, H, S, Dh)
        q1, k1, v1 = qkv[0]  # (B, H, S, Dh)
        q2, k2, v2 = qkv[1]  # (B, H, S, Dh)
        # q, k, v = rearrange(
        #     qkv, "b s (t h d) -> t b h s d", t=3, h=self.heads, d=self.dim_head
        # )

        # Flex Attention
        out = self.flex(q1, k1, v1, block_mask=block_mask)  # (B, S, H, D)
        out_diff = self.flex(q2, k2, v2, block_mask=block_mask)  # (B, S, H, D)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        assert isinstance(out, Tensor) and isinstance(out_diff, Tensor)

        out = out - lambda_full * out_diff

        out = self.diff_norm(out)
        out = out * (1 - self.lambda_init)

        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.to_out(out)

        x = residual + out

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        x = self.dropout(x)

        return x
