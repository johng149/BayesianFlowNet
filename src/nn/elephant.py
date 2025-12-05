import torch
from einops import einsum
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


class ElephantActivation(Module):
    def __init__(self, d: int = 8, a: int = 2, isotropic: str | None = "relu") -> None:
        super().__init__()
        self.d = d
        self.a = a
        self.isotropic = isotropic

    def forward(self, x: Tensor) -> Tensor:
        """
        Elephant activation function as described in the paper:
        https://arxiv.org/pdf/2310.01365 - Elephant Neural Networks: Born to Be a Continual Learner

        $\text{Elephant}(x) = \frac{1}{1 + |x/a|^d}}$
        """
        if self.isotropic is None:
            return 1 / (1 + torch.abs(x / self.a) ** self.d)
        else:
            norm = torch.linalg.norm(x, dim=-1)
            normalized = einsum(x, 1 / norm, "... d, ... -> ... d")
            activated = F.relu(norm)
            return einsum(normalized, activated, "... d, ... -> ... d")
