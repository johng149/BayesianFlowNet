import torch
from torch import Tensor
from src.tokenizers.base import TokenizerBase
from transformers import AutoTokenizer

class GPT2Tokenizer(TokenizerBase):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size
    
    def encode(self, text: str) -> Tensor:
        return self.tokenizer.encode(text, return_tensors="pt").squeeze(0)
    
    def decode(self, tokens: Tensor) -> str:
        assert tokens.ndim == 2, "tokens should be a 2D tensor of shape (seq_len, K)"
        seq_len, K = tokens.shape
        cur_seq = []
        for i in range(seq_len):
            one_hot_encoding = tokens[i]
            value = torch.argmax(one_hot_encoding)
            cur_seq.append(value.item())
        return self.tokenizer.decode(cur_seq, skip_special_tokens=True)