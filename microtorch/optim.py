"""
MicroTorch Optimizers: SGD (with Momentum), Adam, and AdamW.
"""

import math
from typing import List, Tuple
from .nn import Parameter


class Optimizer:
    def __init__(self, params: List[Parameter], lr: float) -> None:
        self.params = params
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum."""

    def __init__(self, params: List[Parameter], lr: float = 0.01, momentum: float = 0.0) -> None:
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocities = [[0.0] * len(p._flat) for p in self.params]

    def step(self) -> None:
        for p, v in zip(self.params, self.velocities):
            if p.grad is None:
                continue
            for i in range(len(p._flat)):
                grad = p.grad._flat[i]
                if self.momentum > 0.0:
                    v[i] = self.momentum * v[i] + grad
                    p._flat[i] -= self.lr * v[i]
                else:
                    p._flat[i] -= self.lr * grad


class Adam(Optimizer):
    """Adam Optimizer (Adaptive Moment Estimation)."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [[0.0] * len(p._flat) for p in self.params]
        self.v = [[0.0] * len(p._flat) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            m = self.m[idx]
            v = self.v[idx]
            for i in range(len(p._flat)):
                grad = p.grad._flat[i]
                # Update biased 1st and 2nd moment estimates
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * (grad ** 2)

                # Compute bias-corrected moments
                m_hat = m[i] / (1.0 - (self.beta1 ** self.t))
                v_hat = v[i] / (1.0 - (self.beta2 ** self.t))

                # Update parameter
                p._flat[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """AdamW Optimizer with decoupled weight decay."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [[0.0] * len(p._flat) for p in self.params]
        self.v = [[0.0] * len(p._flat) for p in self.params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            m = self.m[idx]
            v = self.v[idx]
            for i in range(len(p._flat)):
                grad = p.grad._flat[i]
                # Decoupled weight decay
                if self.weight_decay != 0.0:
                    p._flat[i] -= self.lr * self.weight_decay * p._flat[i]

                # Update biased 1st and 2nd moment estimates
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * (grad ** 2)

                # Compute bias-corrected moments
                m_hat = m[i] / (1.0 - (self.beta1 ** self.t))
                v_hat = v[i] / (1.0 - (self.beta2 ** self.t))

                # Update parameter
                p._flat[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


class Adagrad(Optimizer):
    """
    Adagrad Optimizer (Duchi et al., 2011).
    Adapts learning rate per parameter by dividing by root of accumulated squared gradients:
    G_t = G_{t-1} + g_t^2
    theta_t = theta_{t-1} - (lr / (sqrt(G_t) + eps)) * (g_t + weight_decay * theta_{t-1})
    """

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.01,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
    ) -> None:
        super().__init__(params, lr)
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        self.state_sum = [
            [initial_accumulator_value] * len(p._flat) for p in self.params
        ]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        clr = self.lr / (1.0 + (self.t - 1) * self.lr_decay)
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            state = self.state_sum[idx]
            for i in range(len(p._flat)):
                grad = p.grad._flat[i]
                if self.weight_decay != 0.0:
                    grad += self.weight_decay * p._flat[i]
                state[i] += grad * grad
                std = math.sqrt(state[i]) + self.eps
                p._flat[i] -= clr * grad / std


def clip_grad_norm_(parameters: List[Parameter], max_norm: float, norm_type: float = 2.0) -> float:
    """
    Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were concatenated into a single vector.
    Gradients are modified in-place.
    Returns total norm.
    """
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if math.isinf(norm_type):
        total_norm = max(max(abs(g) for g in p.grad._flat) for p in params)
    else:
        total_norm = 0.0
        for p in params:
            for g in p.grad._flat:
                total_norm += abs(g) ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            for i in range(len(p.grad._flat)):
                p.grad._flat[i] *= clip_coef

    return total_norm


def clip_grad_value_(parameters: List[Parameter], clip_value: float) -> None:
    """
    Clips gradient values of parameters at specified value in-place:
    grad is clamped to [-clip_value, clip_value].
    """
    clip_value = abs(float(clip_value))
    for p in parameters:
        if p.grad is None:
            continue
        for i in range(len(p.grad._flat)):
            p.grad._flat[i] = max(-clip_value, min(clip_value, p.grad._flat[i]))


class GradientAccumulator:
    """
    Utility wrapper that accumulates gradients across multiple micro-batches
    before executing an optimizer step and resetting gradients.
    
    This enables training with virtually enlarged effective batch sizes
    without increasing peak memory usage.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        accumulation_steps: int = 4,
        max_norm: Optional[float] = None,
    ) -> None:
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be at least 1, got {accumulation_steps}")
        self.optimizer = optimizer
        self.accumulation_steps = int(accumulation_steps)
        self.max_norm = float(max_norm) if max_norm is not None else None
        self.step_count = 0

    def accumulate(self, loss: Tensor) -> bool:
        """
        Scales loss by 1/accumulation_steps, invokes backward(), and conditionally steps
        the optimizer when accumulation_steps micro-steps are reached.
        
        Returns:
            bool: True if an optimizer update was performed, False otherwise.
        """
        scaled_loss = loss / float(self.accumulation_steps)
        scaled_loss.backward()
        self.step_count += 1

        if self.step_count % self.accumulation_steps == 0:
            if self.max_norm is not None:
                clip_grad_norm_(self.optimizer.params, self.max_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            return True
        return False

    def flush(self) -> bool:
        """
        Executes optimizer step on any remaining uncommitted gradients and resets step count.
        
        Returns:
            bool: True if an uncommitted gradient step was executed, False if nothing was pending.
        """
        remainder = self.step_count % self.accumulation_steps
        if remainder != 0:
            if self.max_norm is not None:
                clip_grad_norm_(self.optimizer.params, self.max_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step_count = 0
            return True
        return False

    def zero_grad(self) -> None:
        """Zeros gradients on underlying optimizer parameters and resets step count."""
        self.optimizer.zero_grad()
        self.step_count = 0


