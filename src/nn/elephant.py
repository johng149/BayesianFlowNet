import torch
from torch import Tensor
from torch.nn import Module


class ElephantActivation(Module):
    def __init__(self, d: int = 8, a: int = 2) -> None:
        super().__init__()
        self.d = d
        self.a = a

    def forward(self, x: Tensor) -> Tensor:
        """
        Elephant activation function as described in the paper:
        https://arxiv.org/pdf/2310.01365 - Elephant Neural Networks: Born to Be a Continual Learner

        $\text{Elephant}(x) = \frac{1}{1 + |x/a|^d}}$
        """
        return 1 / (1 + torch.abs(x / self.a) ** self.d)
