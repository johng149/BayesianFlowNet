import torch
from einops import rearrange
from torch.nn import Linear, Module
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

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


class TransformerBlock(Module):
    def __init__(self, dim, heads, dim_head, num_layers=1, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.norm1 = torch.nn.RMSNorm(dim)
        self.heads = heads
        self.dim_head = dim_head

        self.to_qkv = Linear(dim, heads * dim_head * 3, bias=False)
        self.to_out = Linear(heads * dim_head, dim, bias=False)

        self.norm2 = torch.nn.RMSNorm(dim)
        self.ff = torch.nn.Sequential(
            Linear(dim, dim * 4), torch.nn.GELU(), Linear(dim * 4, dim)
        )
        self.flex = torch.compile(flex_attention)
        self.dropout = torch.nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        scale = 0.02 / (2 * self.num_layers) ** 0.5

        # Attention output projection
        torch.nn.init.normal_(self.to_out.weight, mean=0.0, std=scale)
        if self.to_out.bias is not None:
            torch.nn.init.zeros_(self.to_out.bias)

        # QKV projection
        torch.nn.init.normal_(self.to_qkv.weight, mean=0.0, std=scale)
        if self.to_qkv.bias is not None:
            torch.nn.init.zeros_(self.to_qkv.bias)

        # MLP output projection
        # self.ff is Sequential(Linear, GELU, Linear)
        mlp_in = self.ff[0]
        mlp_out = self.ff[2]

        assert isinstance(mlp_in, Linear) and isinstance(mlp_out, Linear)

        torch.nn.init.normal_(mlp_out.weight, mean=0.0, std=scale)
        if mlp_out.bias is not None:
            torch.nn.init.zeros_(mlp_out.bias)

        torch.nn.init.normal_(mlp_in.weight, mean=0.0, std=scale)
        if mlp_in.bias is not None:
            torch.nn.init.zeros_(mlp_in.bias)

    def forward(self, x, block_mask):
        # x: (1, SeqLen, Dim) - treating packed as batch 1
        B, S, D = x.shape

        residual = x
        x = self.norm1(x)

        qkv = self.to_qkv(x)  # (B, S, 3 * H * Dh)
        q, k, v = rearrange(
            qkv, "b s (t h d) -> t b h s d", t=3, h=self.heads, d=self.dim_head
        )

        # Flex Attention
        out = self.flex(q, k, v, block_mask=block_mask)  # (B, S, H, D)

        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.to_out(out)

        x = residual + out

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        x = self.dropout(x)

        return x
