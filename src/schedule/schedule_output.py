from typing import TypedDict

from torch import Tensor

ScheduleOutput = TypedDict(
    "ScheduleOutput",
    {
        "beta": Tensor,
        "alpha": Tensor,
    },
)
