from typing import NewType

from torch import Tensor

"""
DiscreteFormattedLoss is used by `format_discrete_loss` function

It describes Tensors that have the shape (batch_size, folds)

Folds is the number of different timesteps sampled per input sample.

For example, if we have two input samples and we sample three timesteps
for each, the resulting tensor would have the shape (2, 3). This means
the model will see the same input sample 3 times, each with a different
timesteps. This is useful for estimating the variance of the loss
at each timestep for a given input sample.
"""
DiscreteFormattedLoss = NewType("DiscreteFormattedLoss", Tensor)
