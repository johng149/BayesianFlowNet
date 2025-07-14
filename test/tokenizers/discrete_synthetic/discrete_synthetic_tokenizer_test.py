import pytest
import torch
from torch.nn import functional as F
from src.tokenizers.discrete_synthetic.discrete_synthetic_tokenizer import DiscreteSyntheticTokenizer

def test_init():
    tokenizer = DiscreteSyntheticTokenizer()

def test_encode():
    tokenizer = DiscreteSyntheticTokenizer()
    text = "8 , 9 , 1 0 , 1 1 , 1 2 ,"
    encoded = tokenizer.encode(text)
    expected = torch.tensor([8, 10, 9, 10, 1, 0, 10, 1, 1, 10, 1, 2, 10], dtype=torch.long)
    assert torch.equal(encoded, expected)

def test_decode():
    tokenizer = DiscreteSyntheticTokenizer()
    encoded = torch.tensor([8, 10, 9, 10, 1, 0, 10, 1, 1, 10, 1, 2, 10], dtype=torch.long)
    one_hot = F.one_hot(encoded, num_classes=tokenizer.vocab_size())
    decoded = tokenizer.decode(one_hot)
    expected = "8 , 9 , 1 0 , 1 1 , 1 2 ,"
    assert decoded == expected
