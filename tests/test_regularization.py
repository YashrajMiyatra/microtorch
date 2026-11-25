import unittest
from microtorch.tensor import Tensor
from microtorch.regularization import compute_gradient_penalty


class TestRegularization(unittest.TestCase):
    def test_gradient_penalty_linear_critic(self):
        # Linear critic f(x) = sum(x) has gradient [1.0, 1.0, ...] -> norm = sqrt(sum(1))
        # For 1D scalar x, grad = 1.0, norm = 1.0 -> GP = lambda * (1 - 1)^2 = 0.0
        def critic_1d(x: Tensor) -> Tensor:
            return x * 1.0

        real = Tensor([[2.0]])
        fake = Tensor([[-1.0]])
        gp = compute_gradient_penalty(critic_1d, real, fake, lambda_gp=10.0, eps_val=0.5)
        self.assertAlmostEqual(gp.data, 0.0, places=4)

    def test_gradient_penalty_steeper_critic(self):
        # Critic f(x) = 3.0 * x has grad = 3.0 -> norm = 3.0
        # GP = 10.0 * (3.0 - 1.0)^2 = 10.0 * 4.0 = 40.0
        def critic_steep(x: Tensor) -> Tensor:
            return x * 3.0

        real = Tensor([[1.0]])
        fake = Tensor([[0.0]])
        gp = compute_gradient_penalty(critic_steep, real, fake, lambda_gp=10.0, eps_val=0.5)
        self.assertAlmostEqual(gp.data, 40.0, places=4)

    def test_shape_mismatch(self):
        real = Tensor([[1.0, 2.0]])
        fake = Tensor([[1.0]])
        with self.assertRaises(ValueError):
            compute_gradient_penalty(lambda x: x, real, fake)


if __name__ == "__main__":
    unittest.main()
