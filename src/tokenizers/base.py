from abc import ABC, abstractmethod

import torch
from torch import Tensor


class TokenizerBase(ABC):

    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def encode(self, text: str) -> Tensor:
        pass

    @abstractmethod
    def _decode(self, tokens: Tensor) -> str:
        """
        Decodes a tensor of token IDs back into a string. The tensor should either be a
        1D tensor of shape (sequence_length,)

        Args:
            - tokens (Tensor) A 1D tensor of token IDs
        Returns:
            str: The decoded string.
        """
        pass

    def decode(self, tokens: Tensor) -> str:
        """
        Decodes a tensor of token IDs back into a string. The tensor should either be a
        1D tensor of shape (sequence_length,) or a 2D one-hot tensor of shape (sequence_length, vocab_size)

        Args:
            - tokens (Tensor) A 1D tensor of token IDs or a 2D one-hot tensor.
        Returns:
            str: The decoded string.
        """
        assert tokens.ndim in (1, 2), "Input tensor must be either 1D or 2D."
        if tokens.ndim == 2:
            # Convert one-hot to token IDs
            tokens = torch.argmax(tokens, dim=-1)
        return self._decode(tokens)

    def __call__(self, tokens: Tensor) -> str:
        return self.decode(tokens)
