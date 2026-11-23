import math
import unittest
from microtorch.tensor import Tensor
from microtorch.nn import BCEWithLogitsLoss


class TestBCEWithLogitsLoss(unittest.TestCase):
    def test_bce_loss_forward(self):
        loss_fn = BCEWithLogitsLoss()
        pred = Tensor([0.0, 2.0, -2.0], requires_grad=True)
        target = Tensor([0.0, 1.0, 0.0])

        loss = loss_fn(pred, target)
        # Expected calculation:
        # x=0, y=0: max(0,0) - 0 + log(1 + exp(0)) = log(2) ~= 0.693147
        # x=2, y=1: max(2,0) - 2*1 + log(1 + exp(-2)) = 2 - 2 + log(1 + exp(-2)) ~= 0.126928
        # x=-2, y=0: max(-2,0) - (-2)*0 + log(1 + exp(-2)) = 0 + log(1 + exp(-2)) ~= 0.126928
        # mean ~= (0.693147 + 0.126928 + 0.126928) / 3 ~= 0.315668
        expected = (math.log(2.0) + 2 * math.log(1.0 + math.exp(-2.0))) / 3.0
        self.assertAlmostEqual(loss.data, expected, places=5)

    def test_bce_loss_backward(self):
        loss_fn = BCEWithLogitsLoss()
        pred = Tensor([0.0], requires_grad=True)
        target = Tensor([1.0])

        loss = loss_fn(pred, target)
        loss.backward()

        # sigmoid(0) = 0.5. d(loss)/dx = (0.5 - 1.0) / 1 = -0.5
        self.assertAlmostEqual(pred.grad.item(), -0.5, places=5)

    def test_bce_loss_shape_mismatch(self):
        loss_fn = BCEWithLogitsLoss()
        pred = Tensor([0.0, 1.0])
        target = Tensor([0.0])
        with self.assertRaises(ValueError):
            loss_fn(pred, target)


if __name__ == "__main__":
    unittest.main()
