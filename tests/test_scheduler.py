"""
Unit tests for MicroTorch Learning Rate Schedulers.
"""

import math
import unittest
from microtorch.tensor import Tensor
from microtorch.nn import Parameter
from microtorch.optim import SGD
from microtorch.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR


class TestLRSchedulers(unittest.TestCase):
    def setUp(self):
        self.p = Parameter(Tensor([1.0, 2.0]))
        self.opt = SGD([self.p], lr=0.1)

    def test_step_lr(self):
        scheduler = StepLR(self.opt, step_size=2, gamma=0.5)
        self.assertAlmostEqual(self.opt.lr, 0.1)

        scheduler.step()  # epoch 1
        self.assertAlmostEqual(self.opt.lr, 0.1)

        scheduler.step()  # epoch 2 (decay)
        self.assertAlmostEqual(self.opt.lr, 0.05)

        scheduler.step()  # epoch 3
        self.assertAlmostEqual(self.opt.lr, 0.05)

        scheduler.step()  # epoch 4 (decay)
        self.assertAlmostEqual(self.opt.lr, 0.025)

    def test_exponential_lr(self):
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        self.assertAlmostEqual(self.opt.lr, 0.1)

        scheduler.step()
        self.assertAlmostEqual(self.opt.lr, 0.09)

        scheduler.step()
        self.assertAlmostEqual(self.opt.lr, 0.081)

    def test_cosine_annealing_lr(self):
        scheduler = CosineAnnealingLR(self.opt, T_max=4, eta_min=0.01)
        self.assertAlmostEqual(self.opt.lr, 0.1)

        scheduler.step()  # epoch 1
        expected_1 = 0.01 + 0.5 * (0.1 - 0.01) * (1.0 + math.cos(math.pi * 1 / 4))
        self.assertAlmostEqual(self.opt.lr, expected_1, places=5)

        scheduler.step()  # epoch 2
        expected_2 = 0.01 + 0.5 * (0.1 - 0.01) * (1.0 + math.cos(math.pi * 2 / 4))
        self.assertAlmostEqual(self.opt.lr, expected_2, places=5)

        scheduler.step()  # epoch 3
        scheduler.step()  # epoch 4 (T_max reached)
        self.assertAlmostEqual(self.opt.lr, 0.01, places=5)


if __name__ == "__main__":
    unittest.main()
