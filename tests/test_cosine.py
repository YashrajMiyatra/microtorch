import math
import unittest
from microtorch.tensor import Tensor
from microtorch.nn import CosineSimilarity


class TestCosineSimilarity(unittest.TestCase):
    def test_cosine_similarity_orthogonal(self):
        cos = CosineSimilarity()
        v1 = Tensor([1.0, 0.0], requires_grad=True)
        v2 = Tensor([0.0, 1.0])
        sim = cos(v1, v2)
        self.assertAlmostEqual(sim.item(), 0.0, places=5)

    def test_cosine_similarity_identical(self):
        cos = CosineSimilarity()
        v1 = Tensor([2.0, 3.0], requires_grad=True)
        v2 = Tensor([4.0, 6.0])
        sim = cos(v1, v2)
        self.assertAlmostEqual(sim.item(), 1.0, places=5)

    def test_cosine_similarity_backward(self):
        cos = CosineSimilarity()
        v1 = Tensor([1.0, 0.0], requires_grad=True)
        v2 = Tensor([0.0, 2.0], requires_grad=True)
        sim = cos(v1, v2)
        sim.backward()

        # At orthogonal vectors (1, 0) and (0, 2):
        # dot = 0. denom = 1 * 2 = 2.
        # d/dv1 = v2 / denom = [0, 2] / 2 = [0, 1]
        self.assertAlmostEqual(v1.grad._flat[0], 0.0, places=5)
        self.assertAlmostEqual(v1.grad._flat[1], 1.0, places=5)

    def test_shape_mismatch_raises(self):
        cos = CosineSimilarity()
        with self.assertRaises(ValueError):
            cos(Tensor([1.0, 2.0]), Tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
