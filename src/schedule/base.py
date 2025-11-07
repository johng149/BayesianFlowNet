from abc import ABC, abstractmethod

from torch import Tensor

from src.schedule.schedule_output import ScheduleOutput


class Scheduler(ABC):
    @abstractmethod
    def forward(self, t: Tensor) -> ScheduleOutput:
        pass

    @abstractmethod
    def __call__(self, t: Tensor) -> ScheduleOutput:
        pass
