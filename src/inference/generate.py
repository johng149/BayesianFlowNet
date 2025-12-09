from typing import Callable, NewType

import torch
from torch import Tensor
from torch.distributions import Categorical as TorchCategorical
from torch.nn import Module
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.common.data_prep import accuracy, dis_t, sample_model_output, theta, y
from src.schedule.base import Scheduler
from src.tokenizers.base import TokenizerBase


def bayesian_update(y: Tensor, model_input: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Args:
        - y (Tensor) Noisy version of the sampled one-hot tensor of shape (batch_size, seq_len, K).
        - model_input (Tensor) Input to the model of shape (batch_size, seq_len, K).
    Returns:
        Tensor: Resulting tensor after applying Bayesian update to the model input based on the noisy output y.
    """
    log_model_input = torch.log(model_input + eps)  # add eps to avoid log(0)
    z = y + log_model_input
    log_new_probs = F.log_softmax(z, dim=-1)
    res = torch.exp(log_new_probs)
    return res


def bayesian_inference(
    model_input: Tensor,
    model_output_logits: Tensor,
    i: Tensor,
    n: Tensor,
    scheduler: Scheduler,
) -> Tensor:
    """
    Args:
        - model_input (Tensor) Input to the model of shape (batch_size, seq_len, K).
        - model_output_logits (Tensor) Model output logits of shape (batch_size, seq_len, K).
        - i (Tensor) Current iteration number of shape (batch_size,).
        - n (Tensor) Total number of iterations of shape (batch_size,).
        - beta_1 (Tensor) Maximum possible accuracy (reached when t=1) of shape (batch_size,).
    Returns:
        Tensor: Resulting tensor after performing Bayesian inference.
    """
    acc = accuracy(i, n, scheduler)
    sampled = sample_model_output(model_output_logits)
    noisy_y = y(sampled, acc)

    # we need to do `(model_input + 1) / 2` to convert the input from [-1, 1] to [0, 1]
    # if we did not, the parameters of the distribution wouldn't produce a valid probability distribution
    # and so the `bayesian_update` may end up with NaN values
    # however, upon returning, we need to convert it back to [-1, 1] as that is what the model is trained on
    return bayesian_update(noisy_y, model_input)


def generative_prior(
    batch_size: int,
    seq_len: int,
    K: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Create an uninformative prior tensor used to start the inference process.
    Args:
        - batch_size (int) Batch size.
        - seq_len (int) Sequence length.
        - K (int) Number of categories.
        - device (torch.device) Device to create the tensor on.
        - dtype (torch.dtype) Data type of the tensor.
    Returns:
        Tensor: Prior tensor of shape (batch_size, seq_len, K) with uniform probabilities.
    """
    uniform = torch.full(
        (batch_size, seq_len, K),
        fill_value=1.0 / K,
        device=device,
        dtype=dtype,
    )
    return theta(uniform)


class TextBFNSolver:
    def __init__(
        self,
        unet: torch.nn.Module,
        class_num: int = 27,
        num_steps: int = 100,
        max_sqrt_beta: float = 0.75,
        eta: float = 1e-5,
        callback=None,
    ):
        self.unet = unet
        self.eta = eta
        self.callback = callback

        self.max_sqrt_beta = max_sqrt_beta
        self.K = class_num

        self.num_steps = num_steps
        self.steps = torch.flip(torch.arange(num_steps + 1), [0])
        self.times = self.steps.to(torch.float64) / num_steps * (1 - eta)
        self.delta_t = (1 - eta) / num_steps

        # f g
        self.f_t = -2 / (1 - self.times)
        self.g_t = (2 * self.K * (1 - self.times)) ** 0.5 * self.max_sqrt_beta

        # beta alpha
        self.beta_t = (self.max_sqrt_beta * (1 - self.times)) ** 2
        self.alpha_t = 2 * (1 - self.times) * self.max_sqrt_beta**2

    def sde_euler_update(
        self,
        x_s,
        step,
        mask,
        model_input,
        doc_ids,
        last_drop=False,
        cate_samp=False,
        addi_step=False,
    ):
        # x_s -> x_t
        t = torch.ones((x_s.shape[0], x_s.shape[1]), device=x_s.device) * (
            1 - self.times[step]
        )

        g = self.g_t[step]

        noise = torch.randn_like(x_s, device=x_s.device)

        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            theta = torch.where(mask.unsqueeze(-1), theta, model_input)
            logits, _, _, energy = self.unet(theta, t, mask, doc_ids)
            if self.callback is not None:
                logits = self.callback(logits)
            data_pred = F.softmax(logits, -1)
            if cate_samp == True:
                categorical = TorchCategorical(logits=logits, validate_args=False)
                data_pred = categorical.sample()
                data_pred = F.one_hot(data_pred.long(), self.K)

            if last_drop == True and step == self.num_steps - 1:
                return logits, data_pred, energy
            elif addi_step == True and step == self.num_steps - 1:
                x_t = (
                    x_s
                    + g**2 * (data_pred - 1 / self.K) * self.delta_t
                    + g * self.delta_t**0.5 * noise
                )
                theta = F.softmax(x_t, -1)
                t = torch.ones((x_s.shape[0], x_s.shape[1]), device=x_s.device) * (
                    1 - self.times[step + 1]
                )
                theta = torch.where(mask.unsqueeze(-1), theta, model_input)
                logits, _, _, energy = self.unet(theta, t, mask, doc_ids)
                data_pred = F.softmax(logits, -1)
                return logits, data_pred, energy
            else:
                x_t = (
                    x_s
                    + g**2 * (data_pred - 1 / self.K) * self.delta_t
                    + g * self.delta_t**0.5 * noise
                )  # FIXME: do we return logits or x_t here???
                return logits, data_pred, energy

    def ode_euler_update(
        self,
        x_s,
        step,
        mask,
        model_input,
        doc_ids,
        last_drop=False,
        cate_samp=False,
        addi_step=False,
    ):
        # x_s -> x_t
        t = torch.ones((x_s.shape[0], x_s.shape[1]), device=x_s.device) * (
            1 - self.times[step]
        )

        f = self.f_t[step]
        g = self.g_t[step]
        beta_s = self.beta_t[step]

        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            theta = torch.where(mask.unsqueeze(-1), theta, model_input)
            logits, _, _, energy = self.unet(theta, t, mask, doc_ids)
            data_pred = F.softmax(logits, -1)
            if cate_samp == True:
                categorical = TorchCategorical(logits=logits, validate_args=False)
                data_pred = categorical.sample()
                data_pred = F.one_hot(data_pred.long(), self.K)
            if last_drop == True and step == self.num_steps - 1:
                return logits, data_pred, energy
            elif addi_step == True and step == self.num_steps - 1:
                x_t = (
                    x_s
                    - (
                        (f + (g**2) / (2 * self.K * beta_s)) * x_s
                        - 0.5 * g**2 * (data_pred - 1 / self.K)
                    )
                    * self.delta_t
                )
                theta = F.softmax(x_t, -1)
                t = torch.ones((x_s.shape[0], x_s.shape[1]), device=x_s.device) * (
                    1 - self.times[step + 1]
                )
                theta = torch.where(mask.unsqueeze(-1), theta, model_input)
                logits, _, _, energy = self.unet(theta, t, mask, doc_ids)
                data_pred = F.softmax(logits, -1)
                return logits, data_pred, energy
            else:
                x_t = (
                    x_s
                    - (
                        (f + (g**2) / (2 * self.K * beta_s)) * x_s
                        - 0.5 * g**2 * (data_pred - 1 / self.K)
                    )
                    * self.delta_t
                )
                return x_t, data_pred, energy

    def ode_bfnsolver1_update(
        self, x_s, step, mask, model_input, doc_ids, last_drop=False
    ):
        # x_s -> x_t
        t = torch.ones((x_s.shape[0], x_s.shape[1]), device=x_s.device) * (
            1 - self.times[step]
        )
        t_t, t_s = self.times[step + 1], self.times[step]
        c_t = self.K * self.max_sqrt_beta**2 * (1 - t_t)
        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            theta = torch.where(mask.unsqueeze(-1), theta, model_input)
            logits, _, _, energy = self.unet(theta, t, mask, doc_ids)
            data_pred = F.softmax(logits, -1)

            if last_drop == True and step == self.num_steps - 1:
                return logits, data_pred, energy
            else:
                x_t = (1 - t_t) / (1 - t_s) * x_s + c_t * (t_t - t_s) * (
                    1 / self.K - data_pred
                )
                return x_t, data_pred, energy

    def ode_bfnsolver2_multi_step_update(
        self,
        x_s,
        step,
        mask,
        model_input,
        doc_ids,
        data_pred_last=None,
        last_drop=False,
    ):
        t = torch.ones((x_s.shape[0], x_s.shape[1]), device=x_s.device) * (
            1 - self.times[step]
        )
        t_t, t_s = self.times[step + 1], self.times[step]
        c_t = self.K * self.max_sqrt_beta**2 * (1 - t_t)
        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            theta = torch.where(mask.unsqueeze(-1), theta, model_input)
            logits, _, _, energy = self.unet(theta, t, mask, doc_ids)
            if self.callback is not None:
                logits = self.callback(logits)
            data_pred = F.softmax(logits, -1)
            if step == 0:
                x_t = (1 - t_t) / (1 - t_s) * x_s + c_t * (t_t - t_s) * (
                    1 / self.K - data_pred
                )
                return x_t, data_pred, energy
            elif last_drop == True and step == self.num_steps - 1:
                return logits, data_pred, energy
            else:
                assert isinstance(data_pred_last, Tensor)
                t_r = self.times[step - 1]
                # x_t = x_s +
                A = (1 - t_t) / (1 - t_s) * x_s + c_t / self.K * (t_t - t_s)
                B = -c_t * (t_t - t_s) * data_pred
                D1 = (data_pred - data_pred_last) / (t_s - t_r)
                C = -c_t * (t_t - t_s) ** 2 / 2 * D1
                x_t = A + B + C
                return A + B + C, data_pred, energy

    def ode_bfnsolver2_single_step_update(
        self, x_s, step, mask, model_input, doc_ids, last_drop=False
    ):
        # x_s -> x_t
        t = torch.ones((x_s.shape[0], x_s.shape[1]), device=x_s.device) * (
            1 - self.times[step]
        )
        t_t, t_s = self.times[step + 1], self.times[step]
        t_r = (t_t + t_s) / 2
        c_r = self.K * self.max_sqrt_beta**2 * (1 - t_r)
        c_t = self.K * self.max_sqrt_beta**2 * (1 - t_t)

        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            theta = torch.where(mask.unsqueeze(-1), theta, model_input)
            logits, _, _, energy = self.unet(theta, t, mask, doc_ids)
            if self.callback is not None:
                logits = self.callback(logits)
            data_pred_s = F.softmax(logits, -1)

            # x_r
            x_r = (1 - t_r) / (1 - t_s) * x_s + c_r * (t_r - t_s) * (
                1 / self.K - data_pred_s
            )
            t = torch.ones((x_s.shape[0], x_s.shape[1]), device=x_s.device) * (1 - t_r)
            theta = F.softmax(x_r, -1)
            theta = torch.where(mask.unsqueeze(-1), theta, model_input)
            logits, _, _, energy = self.unet(theta, t, mask, doc_ids)
            data_pred_r = F.softmax(logits, -1)
            if last_drop == True and step == self.num_steps - 1:
                return logits, data_pred_r, energy
            else:
                A = (1 - t_t) / (1 - t_s) * x_s + c_t / self.K * (t_t - t_s)
                B = -c_t * (t_t - t_s) * data_pred_s
                D1 = (data_pred_r - data_pred_s) / (t_r - t_s)
                C = -c_t * (t_t - t_s) ** 2 / 2 * D1
                x_t = A + B + C
                return x_t, data_pred_r, energy

    def sde_bfnsolver2_multi_step_update(
        self,
        x_s,
        step,
        mask,
        model_input,
        doc_ids,
        data_pred_last=None,
        last_drop=False,
    ):
        t = torch.ones((x_s.shape[0], x_s.shape[1]), device=x_s.device) * (
            1 - self.times[step]
        )
        t_t, t_s = self.times[step + 1], self.times[step]
        beta_s = self.max_sqrt_beta**2 * (1 - t_s) ** 2
        beta_t = self.max_sqrt_beta**2 * (1 - t_t) ** 2
        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            theta = torch.where(mask.unsqueeze(-1), theta, model_input)
            logits, _, _, energy = self.unet(theta, t, mask, doc_ids)
            if self.callback is not None:
                logits = self.callback(logits)
            data_pred_s = F.softmax(logits, -1)
            if step == 0:
                noise = torch.randn_like(x_s, device=x_s.device)
                x_t = (
                    x_s
                    + (beta_t - beta_s) * (self.K * data_pred_s - 1)
                    + (self.K * (beta_t - beta_s)) ** 0.5 * noise
                )
                return x_t, data_pred_s, energy
            elif last_drop == True and step == self.num_steps - 1:
                return logits, data_pred_s, energy
            else:
                assert isinstance(data_pred_last, Tensor)
                noise = torch.randn_like(x_s, device=x_s.device)
                t_r = self.times[step - 1]
                D1 = (data_pred_last - data_pred_s) / (t_r - t_s)
                # x_t_ = x_s + (beta_t - beta_s) * (self.K * data_pred_s - 1)\
                #     + (2*self.K*self.max_sqrt_beta**2*( ((t_t**2)/2 - (t_t**3)/3) - ((t_s**2)/2-(t_s**3)/3 ) ) + t_s * self.K * (beta_t - beta_s)) * D1 \
                #         + (self.K * (beta_t - beta_s))**0.5 * noise

                x_t = (
                    x_s
                    + (beta_t - beta_s) * (self.K * data_pred_s - 1)
                    + 1
                    / 3
                    * self.K
                    * self.max_sqrt_beta**2
                    * (t_t - t_s) ** 2
                    * (t_s + 2 * t_t - 3)
                    * D1
                    + (self.K * (beta_t - beta_s)) ** 0.5 * noise
                )
                return x_t, data_pred_s, energy

    def sde_bfnsolver1_update(
        self, x_s, step, mask, model_input, doc_ids, last_drop=False, cate_samp=False
    ):
        t = torch.ones((x_s.shape[0], x_s.shape[1]), device=x_s.device) * (
            1 - self.times[step]
        )
        t_t, t_s = self.times[step + 1], self.times[step]
        beta_s = self.max_sqrt_beta**2 * (1 - t_s) ** 2
        beta_t = self.max_sqrt_beta**2 * (1 - t_t) ** 2
        with torch.no_grad():
            theta = F.softmax(x_s, -1)
            theta = torch.where(mask.unsqueeze(-1), theta, model_input)
            logits, _, _, energy = self.unet(theta, t, mask, doc_ids)
            if self.callback is not None:
                logits = self.callback(logits)
            data_pred = F.softmax(logits, -1)
            if cate_samp == True:
                data_pred = TorchCategorical(
                    logits=logits, validate_args=False
                ).sample()
                data_pred = F.one_hot(data_pred, self.K).to(torch.float32)
            if last_drop == True and step == self.num_steps - 1:
                return logits, data_pred, energy
            else:
                noise = torch.randn_like(x_s, device=x_s.device)
                x_t = (
                    x_s
                    + (beta_t - beta_s) * (self.K * data_pred - 1)
                    + (self.K * (beta_t - beta_s)) ** 0.5 * noise
                )
                return x_t, data_pred, energy


def sample(
    solver: TextBFNSolver,
    batch_size,
    seq_len,
    K,
    mask,
    model_input,
    doc_ids,
    device,
    steps: int = 100,
    algorithm: str = "sde_euler",
    tk: TokenizerBase | None = None,
):
    beta_t = (solver.max_sqrt_beta * solver.eta) ** 2
    std_t = (K * beta_t) ** 0.5
    prior = torch.randn(batch_size, seq_len, K, device=device) * std_t
    xt = prior
    data_pred_last = None
    ebm_energy = None
    for step in range(steps):
        if algorithm == "sde_euler":
            xt, _, ebm_energy = solver.sde_euler_update(
                xt, step, mask, model_input, doc_ids
            )
        elif algorithm == "ode_euler":
            xt, _, ebm_energy = solver.ode_euler_update(
                xt, step, mask, model_input, doc_ids
            )
        elif algorithm == "ode_bfnsolver1":
            xt, _, ebm_energy = solver.ode_bfnsolver1_update(
                xt, step, mask, model_input, doc_ids
            )
        elif algorithm == "ode_bfnsolver2_single_step":
            xt, _, ebm_energy = solver.ode_bfnsolver2_single_step_update(
                xt, step, mask, model_input, doc_ids
            )
        elif algorithm == "ode_bfnsolver2_multi_step":
            xt, data_pred_last, ebm_energy = solver.ode_bfnsolver2_multi_step_update(
                xt, step, mask, model_input, doc_ids, data_pred_last
            )
        elif algorithm == "sde_bfnsolver1":
            xt, _, ebm_energy = solver.sde_bfnsolver1_update(
                xt, step, mask, model_input, doc_ids
            )
        elif algorithm == "sde_bfnsolver2_multi_step":
            xt, data_pred_last, ebm_energy = solver.sde_bfnsolver2_multi_step_update(
                xt, step, mask, model_input, doc_ids, data_pred_last
            )
        else:
            raise NotImplementedError
        if tk is not None and (step % (steps // 10) == 0 or step == steps - 1):
            print(f"Step {step + 1}: {tk.decode(torch.argmax(xt, dim=-1)[0].cpu())}")
    return xt, ebm_energy


def inference(
    model: Module,
    scheduler: Scheduler,
    num_steps: int,
    batch_size: int,
    seq_len: int,
    K: int,
    mask: Tensor,
    masked_input: Tensor,
    doc_ids: Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    tk: TokenizerBase | None = None,
):
    solver = TextBFNSolver(
        model, class_num=K, num_steps=num_steps, max_sqrt_beta=(20.4054 / K) ** 0.5
    )
    xt, ebm_energy = sample(
        solver,
        batch_size,
        seq_len,
        K,
        mask,
        masked_input,
        doc_ids,
        device,
        steps=num_steps,
        algorithm="sde_euler",
        tk=tk,
    )
    assert isinstance(ebm_energy, Tensor)
    return xt, ebm_energy
