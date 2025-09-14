import string

import torch
from torch import Tensor

from src.tokenizers.base import TokenizerBase


class ASCIITokenizer(TokenizerBase):
    """
    Tokenizes text into a sequence of integers based on a-z (case-insensitive), 0-9, space, and an UNK token.
    Add one additional special masking token for encoder and decoder each in downstream tasks.
    """

    def __init__(self):
        super().__init__()
        self.vocab = {"UNK": 0, " ": 1, "<MASK>": 2}

        # Add digits 0-9
        for i, digit in enumerate(string.digits, start=len(self.vocab)):
            self.vocab[digit] = i

        # Add letters a-z
        for i, letter in enumerate(string.ascii_lowercase, start=len(self.vocab)):
            self.vocab[letter] = i

        self.anti_vocab = {v: k for k, v in self.vocab.items()}
        self.unk_token_id = self.vocab["UNK"]
        self.enc_mask = self.vocab["<MASK>"]
        self.dec_mask = self.vocab["<MASK>"]

    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> Tensor:
        """
        Encodes a string into a tensor of token IDs.
        """
        lower_text = text.lower()
        res = [self.vocab.get(char, self.unk_token_id) for char in lower_text]
        return torch.tensor(res, dtype=torch.long)

    def decode(self, tokens: Tensor) -> str:
        """
        Decodes a tensor of one-hot encoded tokens back into a string.
        """
        assert tokens.ndim == 2, "tokens should be a 2D tensor of shape (seq_len, K)"
        seq_len, K = tokens.shape
        assert (
            K == self.vocab_size()
        ), f"Token dimension K ({K}) does not match vocab size ({self.vocab_size()})"

        cur_seq = []
        for i in range(seq_len):
            one_hot_encoding = tokens[i]
            value = torch.argmax(one_hot_encoding).item()
            assert isinstance(value, int)
            cur_seq.append(self.anti_vocab.get(value, ""))
        return "".join(cur_seq)

    def enc_mask_token_id(self) -> int:
        return self.enc_mask

    def dec_mask_token_id(self) -> int:
        return self.dec_mask
