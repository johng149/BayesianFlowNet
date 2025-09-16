import string
from calendar import c

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

    def decode(self, tokens: Tensor, enc_tokens: Tensor | None = None) -> str:
        """
        Decodes a tensor of one-hot encoded tokens back into a string.

        `tokens` is assumed to be coming from the model output, hence one-hot encoded.
        `enc_tokens` is assumed to be the encoder input tokens and consists of token IDs,
        in other words, not one-hot encoded

        The `tokens` should correspond with a decoded sequence such as:
        [dec_mask, 's', 'o', 'm', 'e', ' ', 't', 'e', 'x', dec_mask, '?']

        and similarly for the `enc_tokens`, though it need not be the same sequence length
        as `tokens`

        If `enc_tokens` is provided, the `tokens` will still be decoded as usual, but all
        mask tokens will be replaced with the corresponding subsequence from `enc_tokens`.
        """
        assert tokens.ndim == 2, "tokens should be a 2D tensor of shape (seq_len, K)"
        assert (
            enc_tokens is None or enc_tokens.ndim == 1
        ), "enc_tokens should be a 1D tensor of token IDs"
        seq_len, K = tokens.shape
        assert (
            K == self.vocab_size()
        ), f"Token dimension K ({K}) does not match vocab size ({self.vocab_size()})"

        decoder_seq: list[int] = []
        for i in range(seq_len):
            one_hot_encoding = tokens[i]
            value = torch.argmax(one_hot_encoding).item()
            assert isinstance(value, int)
            decoder_seq.append(value)
        # return "".join(cur_seq)

        enc_subsequences = []
        if enc_tokens is not None:
            current_subseq = []
            for token_id in enc_tokens:
                if token_id != self.enc_mask:
                    # we are still working on the current subsequence
                    value = token_id.item()
                    assert isinstance(value, int)
                    current_subseq.append(self.anti_vocab.get(value, ""))
                else:
                    # we hit subsequence boundary
                    if current_subseq:
                        enc_subsequences.append("".join(current_subseq))
                        current_subseq = []
            if current_subseq:
                enc_subsequences.append("".join(current_subseq))

        # now iterate through the decoder_seq and replace mask tokens by popping from start of enc_subsequences
        cur_seq = []
        if enc_tokens is not None:
            for token in decoder_seq:
                if token != self.dec_mask:
                    cur_seq.append(self.anti_vocab.get(token, ""))
                else:
                    if enc_subsequences:
                        cur_seq.append(enc_subsequences.pop(0))
                    else:
                        cur_seq.append(
                            ""
                        )  # no corresponding subsequence, just put empty string
        else:
            cur_seq = [self.anti_vocab.get(token, "") for token in decoder_seq]
        return "".join(cur_seq)

    def enc_mask_token_id(self) -> int:
        return self.enc_mask

    def dec_mask_token_id(self) -> int:
        return self.dec_mask
