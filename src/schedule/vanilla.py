import torch
from torch import Tensor
from torch.nn import Module

from src.common.data_prep import Beta
from src.schedule.base import ScheduleOutput, Scheduler


class VanillaScheduler(Module, Scheduler):
    def __init__(self, beta_1: float):
        super().__init__()
        assert beta_1 > 0, "beta_1 must be positive"
        self.beta_1 = beta_1

    def forward(self, t: Tensor) -> ScheduleOutput:
        beta = self.beta_1 * (t**2)
        alpha = self.beta_1 * 2 * t
        return ScheduleOutput(beta=Beta(beta), alpha=alpha)
