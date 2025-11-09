from typing import NewType

from torch import Tensor

Accuracy = NewType("Accuracy", Tensor)
Beta = NewType("Beta", Tensor)
