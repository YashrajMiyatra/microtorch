"""
MicroTorch Learning Rate Schedulers: StepLR, ExponentialLR, and CosineAnnealingLR.
"""

from __future__ import annotations
import math
from typing import List
from .optim import Optimizer


class LRScheduler:
    """Base class for all learning rate schedulers."""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch
        self.step()

    def get_lr(self) -> float:
        raise NotImplementedError

    def step(self) -> None:
        self.last_epoch += 1
        new_lr = self.get_lr()
        self.optimizer.lr = new_lr


class StepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every step_size epochs."""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1) -> None:
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch == 0:
            return self.base_lr
        if self.last_epoch % self.step_size == 0:
            return self.optimizer.lr * self.gamma
        return self.optimizer.lr


class ExponentialLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch."""

    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1) -> None:
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch == 0:
            return self.base_lr
        return self.optimizer.lr * self.gamma


class CosineAnnealingLR(LRScheduler):
    """
    Set the learning rate using a cosine annealing schedule:
    eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T_max))
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch == 0:
            return self.base_lr
        if self.last_epoch >= self.T_max:
            return self.eta_min
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + math.cos(math.pi * self.last_epoch / self.T_max))


class CosineAnnealingWarmupLR(LRScheduler):
    """
    Sets learning rate with linear warmup phase followed by cosine annealing decay:
    - Phase 1 (epoch < warmup_steps): linear increase from warmup_lr to base_lr.
    - Phase 2 (epoch >= warmup_steps): cosine decay from base_lr down to min_lr.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 0.0,
        warmup_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if total_steps <= warmup_steps:
            raise ValueError("total_steps must be strictly greater than warmup_steps")
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        self.warmup_lr = float(warmup_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch < self.warmup_steps:
            progress = self.last_epoch / max(1, self.warmup_steps)
            return self.warmup_lr + progress * (self.base_lr - self.warmup_lr)
        elif self.last_epoch >= self.total_steps:
            return self.min_lr
        else:
            decay_epochs = self.total_steps - self.warmup_steps
            current_decay_epoch = self.last_epoch - self.warmup_steps
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * current_decay_epoch / decay_epochs))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_factor

