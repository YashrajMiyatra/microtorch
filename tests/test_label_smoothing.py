import unittest
from microtorch.tensor import Tensor
from microtorch.nn import LabelSmoothingCrossEntropyLoss


class TestLabelSmoothing(unittest.TestCase):
    def test_forward_computation(self):
        # 1 sample, 3 classes: logits [2.0, 1.0, 0.1], target 0
        loss_fn = LabelSmoothingCrossEntropyLoss(smoothing=0.1)
        logits = Tensor([[2.0, 1.0, 0.1]])
        target = Tensor([0])
        loss = loss_fn(logits, target)
        self.assertGreater(loss.data, 0.0)

    def test_zero_smoothing_equals_standard_cross_entropy(self):
        loss_fn = LabelSmoothingCrossEntropyLoss(smoothing=0.0)
        logits = Tensor([[2.0, 1.0, 0.1]])
        target = Tensor([0])
        loss = loss_fn(logits, target)

        import math
        exp_sum = math.exp(2.0) + math.exp(1.0) + math.exp(0.1)
        p0 = math.exp(2.0) / exp_sum
        expected = -math.log(p0)
        self.assertAlmostEqual(loss.data, expected, places=4)

    def test_backward_gradient(self):
        loss_fn = LabelSmoothingCrossEntropyLoss(smoothing=0.2)
        logits = Tensor([[1.0, 2.0, 0.5]], requires_grad=True)
        target = Tensor([1])
        loss = loss_fn(logits, target)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertEqual(len(logits.grad._flat), 3)
        # Sum of gradients for softmax cross-entropy should be ~0
        self.assertAlmostEqual(sum(logits.grad._flat), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
