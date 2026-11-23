import unittest
from microtorch.tensor import Tensor
from microtorch.nn import SmoothL1Loss


class TestSmoothL1Loss(unittest.TestCase):
    def test_smooth_l1_loss_forward(self):
        loss_fn = SmoothL1Loss(beta=1.0, reduction="mean")
        # diff 0.5 < 1.0 -> 0.5 * 0.25 = 0.125
        # diff 2.0 >= 1.0 -> 2.0 - 0.5 = 1.5
        # mean = (0.125 + 1.5) / 2 = 0.8125
        pred = Tensor([0.5, 3.0], requires_grad=True)
        target = Tensor([0.0, 1.0])

        loss = loss_fn(pred, target)
        self.assertAlmostEqual(loss.item(), 0.8125, places=5)

    def test_smooth_l1_loss_backward(self):
        loss_fn = SmoothL1Loss(beta=1.0, reduction="mean")
        pred = Tensor([0.5, 3.0], requires_grad=True)
        target = Tensor([0.0, 1.0])

        loss = loss_fn(pred, target)
        loss.backward()

        # grad 0: 0.5 / (1.0 * 2) = 0.25
        # grad 1: 1.0 / 2 = 0.5
        self.assertAlmostEqual(pred.grad._flat[0], 0.25, places=5)
        self.assertAlmostEqual(pred.grad._flat[1], 0.5, places=5)

    def test_invalid_beta(self):
        with self.assertRaises(ValueError):
            SmoothL1Loss(beta=-0.5)


if __name__ == "__main__":
    unittest.main()
