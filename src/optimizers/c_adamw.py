# copy dependencies from transformers/optimization.py
import math
import warnings
from itertools import chain
from typing import Callable, Generator, Iterable, List, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.optim import Optimizer
from torch.optim.optimizer import (
    DeviceDict,
    DeviceDtypeDict,
    Optimizer,
    ParamsT,
    _capturable_doc,
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _fused_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _stack_if_compiling,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
)

from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)


@torch.compile(fullgraph=False)
def c_adamw_update_foreach_async(
    X: List[Tensor],  # Model weights (modified in place)
    G: List[Tensor],  # Gradient
    M: List[Tensor],  # Momentum buffer (modified in place)
    V: List[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
):
    """
    C AdamW optimizer algorithm (async foreach implementation).
    """
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)
    assert batch_size == len(V)

    M_dtype = M[0].dtype
    V_dtype = V[0].dtype

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    G = [g.to(dtype=M_dtype) for g in G]
    torch._foreach_lerp_(M, G, [1 - beta1] * batch_size)

    # V = beta2 * V + (1 - beta2) * G * G
    G_square = torch._foreach_mul(G, G)
    G_square = [g.to(dtype=V_dtype) for g in G_square]
    torch._foreach_lerp_(V, G_square, [1 - beta2] * batch_size)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # Compute the denominator for the weight update
    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = torch._foreach_sqrt(V)
    torch._foreach_div_(denom, bias_correction2_sqrt)
    torch._foreach_add_(denom, [epsilon] * batch_size)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1
    mask = torch._foreach_mul(M, G)
    mask = [m.gt(0.0).to(e.dtype) for m, e in zip(mask, M)]
    mask_mean = [m.mean().clamp(min=1e-3) for m in mask]
    M = torch._foreach_mul(M, mask)  # type: ignore
    M_div = torch._foreach_div(M, denom)

    # Apply weight decay
    torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Weight update
    torch._foreach_mul_(M_div, adj_lr)
    torch._foreach_div_(M_div, mask_mean)
    torch._foreach_sub_(X, M_div)
    yield


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
        foreach: bool = True,
        fused: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self.init_lr = lr
        self._foreach = foreach
        try:
            self._world_size = dist.get_world_size()
        except:
            self._world_size = 1

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])
        return has_complex

    def _get_or_initialize_state(self, param):
        state = self.state[param]
        if not state:
            state["momentum"] = torch.zeros_like(param)
            state["variance"] = torch.zeros_like(param)
        return state

    def _create_tasks(self, param_groups):
        for group in param_groups:
            for params in create_param_batches(
                group["params"], batch_size=self._world_size
            ):
                params = [p for p in params if p.grad is not None]
                if not params:
                    continue
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p) for p in params]
                momentums = [s["momentum"] for s in states]
                variances = [s["variance"] for s in states]
                lr = torch.tensor(group["lr"])
                beta1 = torch.tensor(group["betas"][0])
                beta2 = torch.tensor(group["betas"][1])
                weight_decay = torch.tensor(group["weight_decay"])
                epsilon = torch.tensor(group["eps"])
                step = torch.tensor(group["step"])
                yield AsyncTask(
                    c_adamw_update_foreach_async(
                        X=pad_batch(params, self._world_size),
                        G=pad_batch(gradients, self._world_size),  # type: ignore
                        M=pad_batch(momentums, self._world_size),
                        V=pad_batch(variances, self._world_size),
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        weight_decay=weight_decay,
                        step=step,  # type: ignore
                        epsilon=epsilon,  # type: ignore
                    )
                )

    @torch.no_grad()
    def step(self, closure: Callable = None):  # type: ignore
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if self._foreach:
            # collect grouped tensors for multi-tensor version
            """
            params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps = [], [], [], [], []
            max_exp_avg_sqs = []

            for group in self.param_groups:

                has_complex = self._init_group(
                    group,
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                )

                beta1, beta2 = group["betas"]
                if len(params_with_grad) > 0:
                    _multi_tensor_c_adam(
                        params=params_with_grad,
                        grads=grads,
                        exp_avgs=exp_avgs,
                        exp_avg_sqs=exp_avg_sqs,
                        max_exp_avg_sqs=max_exp_avg_sqs,
                        state_steps=state_steps,
                        grad_scale=None,
                        found_inf=None,
                        amsgrad=group.get("amsgrad", False),
                        has_complex=False,
                        beta1=beta1,
                        beta2=beta2,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        eps=group["eps"],
                        maximize=False,
                        capturable=False,
                        differentiable=False,
                        decoupled_weight_decay=True,
                        cautious=True,  # <<< enable cautious logic
                    )
            return loss
            """

            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            groups = []
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                group["step"] += 1
                groups.append(group)
            tasks = self._create_tasks(groups)
            all_tasks = chain(tasks)
            runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)  # type: ignore
            runtime.run()

            return loss

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # apply weight decay
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                # compute norm gradient
                if type(grad) is DTensor:
                    mask = (exp_avg.full_tensor() * grad.full_tensor() > 0).to(
                        grad.dtype
                    )
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    mask = distribute_tensor(
                        mask, device_mesh=grad.device_mesh, placements=grad.placements
                    )
                else:
                    mask = (exp_avg * grad > 0).to(grad.dtype)
                    # mask = mask * (mask.numel() / (mask.sum() + 1)) ## original implementation, leaving it here for record
                    mask.div_(
                        mask.mean().clamp_(min=1e-3)
                    )  # https://huggingface.co/rwightman/timm-optim-caution found this implementation is more favoarable in many cases
                norm_grad = (exp_avg * mask) / denom
                p.add_(norm_grad, alpha=-step_size)
        return loss
