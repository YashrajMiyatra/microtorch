import unittest
from microtorch.tensor import Tensor
from microtorch.nn import CosineEmbeddingLoss


class TestCosineEmbeddingLoss(unittest.TestCase):
    def test_positive_label_forward(self):
        loss_fn = CosineEmbeddingLoss()
        # Identical vectors: cosine similarity = 1.0, loss = 1 - 1 = 0.0
        x1 = Tensor([[1.0, 2.0, 3.0]])
        x2 = Tensor([[1.0, 2.0, 3.0]])
        y = Tensor([1.0])
        loss = loss_fn(x1, x2, y)
        self.assertAlmostEqual(loss.data, 0.0, places=5)

    def test_positive_label_orthogonal(self):
        loss_fn = CosineEmbeddingLoss()
        # Orthogonal vectors: cosine similarity = 0.0, loss = 1 - 0 = 1.0
        x1 = Tensor([[1.0, 0.0]])
        x2 = Tensor([[0.0, 1.0]])
        y = Tensor([1.0])
        loss = loss_fn(x1, x2, y)
        self.assertAlmostEqual(loss.data, 1.0, places=5)

    def test_negative_label_margin(self):
        loss_fn = CosineEmbeddingLoss(margin=0.5)
        # Cosine similarity = 1.0 > margin 0.5: loss = max(0, 1.0 - 0.5) = 0.5
        x1 = Tensor([[2.0, 0.0]])
        x2 = Tensor([[4.0, 0.0]])
        y = Tensor([-1.0])
        loss = loss_fn(x1, x2, y)
        self.assertAlmostEqual(loss.data, 0.5, places=5)

        # Orthogonal: cosine similarity = 0.0 <= margin 0.5: loss = 0.0
        x1 = Tensor([[1.0, 0.0]])
        x2 = Tensor([[0.0, 1.0]])
        loss = loss_fn(x1, x2, y)
        self.assertAlmostEqual(loss.data, 0.0, places=5)

    def test_backward_pass(self):
        loss_fn = CosineEmbeddingLoss()
        x1 = Tensor([[1.0, 2.0]], requires_grad=True)
        x2 = Tensor([[3.0, 4.0]], requires_grad=True)
        y = Tensor([1.0])
        loss = loss_fn(x1, x2, y)
        loss.backward()

        self.assertIsNotNone(x1.grad)
        self.assertIsNotNone(x2.grad)
        self.assertEqual(len(x1.grad._flat), 2)
        self.assertEqual(len(x2.grad._flat), 2)


if __name__ == "__main__":
    unittest.main()
