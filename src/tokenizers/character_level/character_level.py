import torch
from torch import Tensor

from src.tokenizers.base import TokenizerBase


class CharacterLevelTokenizer(TokenizerBase):
    # intended for tokenizing a-z, ';', ':', '?', ' ', '!', ',', '.', "'", '<UNK>'
    def __init__(self):
        super().__init__()
        self.unk_token = "<UNK>"
        self.vocab = {
            char: idx
            for idx, char in enumerate(
                list("abcdefghijklmnopqrstuvwxyz;:?!.' ,") + [self.unk_token]
            )
        }
        self.anti_vocab = {idx: char for char, idx in self.vocab.items()}

    def mask_idx(self) -> int:
        return self.vocab[self.unk_token]

    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> Tensor:
        token_ids = [
            self.vocab[char] if char in self.vocab else self.vocab[self.unk_token]
            for char in text.lower()
        ]
        return torch.tensor(token_ids, dtype=torch.long)

    def _decode(self, tokens: Tensor) -> str:
        chars = [
            self.anti_vocab[token.item()]  # pyright: ignore[reportArgumentType]
            for token in tokens
        ]
        return "".join(chars)
