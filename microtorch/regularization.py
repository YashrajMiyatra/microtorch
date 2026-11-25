"""
Advanced regularization techniques for deep neural networks in MicroTorch.
Includes Gradient Penalty (WGAN-GP) for stabilizing adversarial and generative models.
"""

import math
import random
from typing import Callable, Optional
from .tensor import Tensor


def compute_gradient_penalty(
    critic: Callable[[Tensor], Tensor],
    real: Tensor,
    fake: Tensor,
    lambda_gp: float = 10.0,
    eps_val: Optional[float] = None,
) -> Tensor:
    """
    Computes Wasserstein GAN Gradient Penalty (Gulrajani et al., 2017).
    GP = lambda * (||grad||_2 - 1)^2
    """
    if real.shape != fake.shape:
        raise ValueError(f"Shape mismatch: real {real.shape} vs fake {fake.shape}")

    eps = float(eps_val) if eps_val is not None else random.random()

    interp_flat = [eps * r + (1.0 - eps) * f for r, f in zip(real._flat, fake._flat)]

    # Unflatten into the same nested shape as real.data
    def _unflatten(flat: list, shape: tuple):
        if not shape:
            return flat[0]
        if len(shape) == 1:
            return flat[: shape[0]]
        sub_size = 1
        for s in shape[1:]:
            sub_size *= s
        return [_unflatten(flat[i * sub_size : (i + 1) * sub_size], shape[1:]) for i in range(shape[0])]

    nested_data = _unflatten(interp_flat, real.shape)
    interpolated = Tensor(nested_data, requires_grad=True)

    critic_score = critic(interpolated)
    critic_score.backward()

    if interpolated.grad is None:
        raise RuntimeError("Critic backward pass did not generate gradients for interpolated input.")

    grad_norm = math.sqrt(sum(g * g for g in interpolated.grad._flat) + 1e-12)
    gp_value = lambda_gp * ((grad_norm - 1.0) ** 2)

    return Tensor(gp_value, requires_grad=False)
