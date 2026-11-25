import unittest
from microtorch.tensor import Tensor
from microtorch.nn import TripletMarginLoss


class TestTripletMarginLoss(unittest.TestCase):
    def test_zero_loss_case(self):
        loss_fn = TripletMarginLoss(margin=1.0)
        # Anchor and positive close (dist=0), anchor and negative far (dist=10)
        # d(a, p) - d(a, n) + 1.0 = 0 - 10 + 1 = -9 <= 0 -> loss = 0
        a = Tensor([[0.0, 0.0]])
        p = Tensor([[0.0, 0.0]])
        n = Tensor([[10.0, 0.0]])
        loss = loss_fn(a, p, n)
        self.assertAlmostEqual(loss.data, 0.0, places=5)

    def test_positive_loss_case(self):
        loss_fn = TripletMarginLoss(margin=1.0)
        # Anchor at 0, positive at 2 (dist=2), negative at 1 (dist=1)
        # d(a, p) - d(a, n) + margin = 2 - 1 + 1 = 2.0
        a = Tensor([[0.0]])
        p = Tensor([[2.0]])
        n = Tensor([[1.0]])
        loss = loss_fn(a, p, n)
        self.assertAlmostEqual(loss.data, 2.0, places=4)

    def test_backward_pass(self):
        loss_fn = TripletMarginLoss(margin=1.0)
        a = Tensor([[0.0, 0.0]], requires_grad=True)
        p = Tensor([[2.0, 0.0]], requires_grad=True)
        n = Tensor([[1.0, 0.0]], requires_grad=True)
        loss = loss_fn(a, p, n)
        loss.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(p.grad)
        self.assertIsNotNone(n.grad)
        self.assertEqual(len(a.grad._flat), 2)
        self.assertEqual(len(p.grad._flat), 2)
        self.assertEqual(len(n.grad._flat), 2)


if __name__ == "__main__":
    unittest.main()
