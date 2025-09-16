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

    def decode(self, tokens: Tensor, enc_tokens: Tensor | None = None) -> str:
        assert tokens.ndim == 2, "tokens should be a 2D tensor of shape (seq_len, K)"
        assert (
            enc_tokens is None or enc_tokens.ndim == 1
        ), "enc_tokens should be a 1D tensor of token IDs"
        seq_len, K = tokens.shape
        assert (
            K == self.vocab_size()
        ), f"Token dimension K ({K}) does not match vocab size ({self.vocab_size()})"

        decoder_seq: list[int] = torch.argmax(tokens, dim=-1).tolist()

        if enc_tokens is None:
            return self.tokenizer.decode(decoder_seq, skip_special_tokens=True)

        enc_subsequences = []
        current_subseq: list[int] = []
        for token_id in enc_tokens.tolist():
            if token_id != self.enc_mask_token_id():
                current_subseq.append(token_id)
            else:
                if current_subseq:
                    enc_subsequences.append(
                        self.tokenizer.decode(current_subseq, skip_special_tokens=True)
                    )
                    current_subseq = []
        if current_subseq:
            enc_subsequences.append(
                self.tokenizer.decode(current_subseq, skip_special_tokens=True)
            )

        # now iterate through the decoder_seq and replace mask tokens by popping from start of enc_subsequences
        result_parts: list[str] = []
        current_chunk: list[int] = []
        for token in decoder_seq:
            if token != self.dec_mask_token_id():
                current_chunk.append(token)
            else:
                if current_chunk:
                    result_parts.append(
                        self.tokenizer.decode(current_chunk, skip_special_tokens=True)
                    )
                    current_chunk = []
                if enc_subsequences:
                    result_parts.append(enc_subsequences.pop(0))
                else:
                    # no corresponding subsequence, just put empty string
                    pass
        if current_chunk:
            result_parts.append(
                self.tokenizer.decode(current_chunk, skip_special_tokens=True)
            )

        return "".join(result_parts)

    def enc_mask_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def dec_mask_token_id(self) -> int:
        return self.tokenizer.pad_token_id
