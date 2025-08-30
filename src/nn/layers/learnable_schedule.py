import torch
from torch import Tensor, nn

from src.nn.layers.unconstrained_montonic_nn import MonotonicNN


class LearnableBetaScheduleNI(nn.Module):
    def __init__(
        self, in_d: int = 2, hidden_layers=[128, 128], nb_steps=50, encoding_dim=256
    ):
        """
        NI stands for Neural Integral.
        This module learns a beta schedule beta(t) that satisfies:
        1. beta(0) = 0
        2. beta(t) is strictly monotonically increasing.
        """
        super().__init__()
        # The input to MonotonicNN is the integration variable (time, dim 1)
        # and conditioning variables. We have no conditioning variables, so `in_d=1`.
        self.monotonic_nn = MonotonicNN(
            in_d=in_d,
            hidden_layers=hidden_layers,
            nb_steps=nb_steps,
            encoding_dim=encoding_dim,
        )

    def beta_1(self, K: int, device) -> float:
        return self.monotonic_nn(
            torch.tensor([[1.0]], device=device),
            torch.tensor([[1.0]], device=device) * K,
        ).item()

    def scaling(self, K: int, device) -> float:
        """
        Returns the scaling factor for the beta schedule.
        This is used to ensure that beta(t) is positive and strictly increasing.

        Args:
            K: The vocabulary size
            device: The device to run the computation on (e.g., "cpu", "cuda").

        Returns:
            A tuple containing the scaling factor
        """
        h = torch.tensor([[1.0]], device=device) * K
        return self.monotonic_nn.scaling(h).item()

    def forward(self, t: Tensor, K: int) -> Tensor:
        """
        Args:
            t: A tensor of time steps of shape (batch_size,).
            K: The vocabulary size
        Returns:
            beta values at given time step t, of shape (batch_size,)
        """
        assert t.ndim == 1, "t should be a 1D tensor"

        # Reshape t to (batch_size, 1) for the MonotonicNN
        t_reshaped = t.unsqueeze(-1)

        h = torch.ones_like(t_reshaped, device=t.device) * K

        # 1. Compute the monotonic function F(t) = integral from 0 to t
        integral_t = self.monotonic_nn(t_reshaped, h).squeeze(-1)

        return integral_t

    def get_alpha(
        self, t: Tensor, K: int, epsilon: float = 1e-8
    ) -> tuple[Tensor, Tensor]:
        """
        Calculates alpha = d(beta)/dt.
        This is efficient as it doesn't require backpropagation through the integral.
        """
        t.requires_grad_(True)
        beta = self.forward(t, K)

        alpha = torch.autograd.grad(
            outputs=beta,
            inputs=t,
            grad_outputs=torch.ones_like(beta),
            create_graph=True,
            retain_graph=True,
        )[0]

        return beta, alpha + epsilon  # add epsilon for stability

    def visualize(self, K: int, points: int = 150, device="cpu"):
        """
        Generates the beta schedule curve for visualization.

        Args:
            points (int): The number of points to sample between t=0 and t=1.
            device (str): The device to run the computation on (e.g., "cpu", "cuda").

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing t_values and beta_values as NumPy arrays.
        """
        # Ensure no gradients are computed for this utility function
        with torch.no_grad():

            # Create the time steps for plotting
            t_values = torch.linspace(0.0, 1.0, points, device=device)

            # Call the forward pass to get the beta values
            beta_values = self.forward(t_values, K=K)

            # Return as numpy arrays for easy plotting with matplotlib
            return t_values.cpu().detach().numpy(), beta_values.cpu().detach().numpy()

    def visualize_alpha(self, K: int, points: int = 150, device="cpu"):
        """
        Generates the alpha schedule curve for visualization. This cannot be run from within
        a `torch.no_grad` context

        Args:
            points (int): The number of points to sample between t=0 and t=1.
            device (str): The device to run the computation on (e.g., "cpu", "cuda").

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing t_values and alpha_values as NumPy arrays.
        """
        # Create the time steps for plotting
        t_values = torch.linspace(0.0, 1.0, points, device=device)

        # Call the get_alpha method to get the alpha values
        _, alpha_values = self.get_alpha(t=t_values, K=K)

        # Return as numpy arrays for easy plotting with matplotlib
        return t_values.cpu().detach().numpy(), alpha_values.cpu().detach().numpy()
