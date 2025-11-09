import torch
from torch import Tensor
import pytest

from src.tokenizers.base import TokenizerBase


# A concrete implementation of TokenizerBase for testing purposes
class ConcreteTokenizer(TokenizerBase):
    def vocab_size(self) -> int:
        return 4

    def encode(self, text: str) -> Tensor:
        # Simple encoding for testing
        if text == "abc":
            return torch.tensor([0, 1, 2])
        return torch.tensor([3])  # unk

    def _decode(self, tokens: Tensor) -> str:
        # Simple decoding for testing
        mapping = {0: "a", 1: "b", 2: "c", 3: "UNK"}
        chars = [mapping[token.item()] for token in tokens]
        return "".join(chars)


def test_tokenizer_base_decode():
    tokenizer = ConcreteTokenizer()
    tokens = torch.tensor([0, 1, 2])
    decoded_text = tokenizer.decode(tokens)
    assert decoded_text == "abc"


def test_tokenizer_base_call_is_decode():
    tokenizer = ConcreteTokenizer()
    tokens = torch.tensor([0, 1, 2])
    # __call__ should be the same as decode
    decoded_text = tokenizer(tokens)
    assert decoded_text == "abc"


def test_tokenizer_base_abstract_methods():
    # Test that abstract methods raise NotImplementedError
    with pytest.raises(TypeError):
        TokenizerBase()  # Cannot instantiate abstract class

    class IncompleteTokenizer(TokenizerBase):
        def vocab_size(self) -> int:
            return 1

        def encode(self, text: str) -> Tensor:
            return torch.tensor([0])

        # Missing _decode

    with pytest.raises(TypeError):
        IncompleteTokenizer()


def test_tokenizer_base_decode_empty():
    tokenizer = ConcreteTokenizer()
    tokens = torch.tensor([])
    decoded_text = tokenizer.decode(tokens)
    assert decoded_text == ""
