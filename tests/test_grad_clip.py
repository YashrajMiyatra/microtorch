import math
import unittest
from microtorch.tensor import Tensor
from microtorch.nn import Parameter
from microtorch.optim import clip_grad_norm_, clip_grad_value_


class TestGradientClipping(unittest.TestCase):
    def test_clip_grad_norm_l2(self):
        # Two parameters with grads [3.0, 4.0] -> norm is sqrt(9 + 16) = 5.0
        p1 = Parameter([[1.0]])
        p1.grad = Tensor([[3.0]])
        p2 = Parameter([[1.0]])
        p2.grad = Tensor([[4.0]])

        total_norm = clip_grad_norm_([p1, p2], max_norm=2.5, norm_type=2.0)
        self.assertAlmostEqual(total_norm, 5.0, places=4)

        # Gradients should be scaled by 2.5 / 5.0 = 0.5
        self.assertAlmostEqual(p1.grad._flat[0], 1.5, places=4)
        self.assertAlmostEqual(p2.grad._flat[0], 2.0, places=4)

    def test_clip_grad_value(self):
        p1 = Parameter([[1.0, 2.0]])
        p1.grad = Tensor([[10.0, -8.0]])

        clip_grad_value_([p1], clip_value=3.5)
        self.assertEqual(p1.grad._flat[0], 3.5)
        self.assertEqual(p1.grad._flat[1], -3.5)


if __name__ == "__main__":
    unittest.main()
