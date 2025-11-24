import torch
from torch import Tensor
from transformers import AutoTokenizer

from src.tokenizers.base import TokenizerBase


class ByT5Tokenizer(TokenizerBase):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/byt5-small", use_fast=False
        )

    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(self, text: str) -> Tensor:
        return self.tokenizer.encode(text, return_tensors="pt").squeeze(0)

    def _decode(self, tokens: Tensor) -> str:
        assert tokens.ndim in (1, 2), "Input tensor must be either 1D or 2D."
        if tokens.ndim == 2:
            # Convert one-hot to token IDs
            tokens = torch.argmax(tokens, dim=-1)
        return self.tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
