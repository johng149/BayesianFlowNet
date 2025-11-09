import torch
from torch import Tensor

from src.tokenizers.base import TokenizerBase


class DiscreteSyntheticTokenizer(TokenizerBase):
    # only tokenizes strings like " 8 , 9 , 1 0 , 1 1 , 1 2 ,"
    # this is intended to be used only with the discrete synthetic dataset
    def __init__(self):
        super().__init__()
        self.vocab = {",": 10, "<MASK>": 11}
        for i in range(10):
            key = str(i)
            value = i
            self.vocab[key] = value

        self.anti_vocab = {}
        for k in self.vocab:
            self.anti_vocab[self.vocab[k]] = k

    def mask_idx(self) -> int:
        return self.vocab["<MASK>"]

    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> Tensor:
        splits = text.split()
        res = [self.vocab.get(s, 0) for s in splits]
        return torch.tensor(res, dtype=torch.long)

    def _decode(self, tokens: Tensor) -> str:
        assert tokens.ndim == 2, "tokens should be a 2D tensor of shape (seq_len, K)"
        seq_len, K = tokens.shape
        cur_seq = []
        for i in range(seq_len):
            one_hot_encoding = tokens[i]
            value = torch.argmax(one_hot_encoding)
            cur_seq.append(self.anti_vocab.get(value.item(), ""))
        return " ".join(cur_seq)
