from abc import ABC, abstractmethod
from typing import TypedDict

from torch import Tensor

from src.common.data_prep import Beta

ScheduleOutput = TypedDict(
    "ScheduleOutput",
    {
        "beta": Beta,
        "alpha": Tensor,
    },
)


class Scheduler(ABC):
    @abstractmethod
    def forward(self, t: Tensor) -> ScheduleOutput:
        pass

    @abstractmethod
    def __call__(self, t: Tensor) -> ScheduleOutput:
        pass
