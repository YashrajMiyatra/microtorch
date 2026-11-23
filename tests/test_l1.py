import unittest
from microtorch.tensor import Tensor
from microtorch.nn import L1Loss


class TestL1Loss(unittest.TestCase):
    def test_l1_loss_mean_forward(self):
        loss_fn = L1Loss(reduction="mean")
        pred = Tensor([1.0, 5.0, 2.0], requires_grad=True)
        target = Tensor([2.0, 3.0, 2.0])

        loss = loss_fn(pred, target)
        # |1-2| + |5-3| + |2-2| = 1 + 2 + 0 = 3. Mean = 3 / 3 = 1.0
        self.assertAlmostEqual(loss.item(), 1.0, places=5)

    def test_l1_loss_sum_forward(self):
        loss_fn = L1Loss(reduction="sum")
        pred = Tensor([1.0, 5.0, 2.0], requires_grad=True)
        target = Tensor([2.0, 3.0, 2.0])

        loss = loss_fn(pred, target)
        self.assertAlmostEqual(loss.item(), 3.0, places=5)

    def test_l1_loss_backward(self):
        loss_fn = L1Loss(reduction="mean")
        pred = Tensor([1.0, 5.0], requires_grad=True)
        target = Tensor([3.0, 2.0])

        loss = loss_fn(pred, target)
        loss.backward()

        # (1-3) < 0 -> sign = -1 / 2 = -0.5
        # (5-2) > 0 -> sign = +1 / 2 = +0.5
        self.assertAlmostEqual(pred.grad._flat[0], -0.5, places=5)
        self.assertAlmostEqual(pred.grad._flat[1], 0.5, places=5)

    def test_l1_loss_invalid_reduction(self):
        with self.assertRaises(ValueError):
            L1Loss(reduction="invalid")


if __name__ == "__main__":
    unittest.main()
