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
