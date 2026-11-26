import unittest
from microtorch.tensor import Tensor
from microtorch.nn import Parameter
from microtorch.optim import Adagrad


class TestAdagrad(unittest.TestCase):
    def test_single_step(self):
        # theta = [1.0], grad = [2.0], lr = 0.1, eps = 1e-10
        # G_1 = 4.0, std = 2.0, update = 0.1 * 2.0 / 2.0 = 0.1
        # theta_1 = 1.0 - 0.1 = 0.9
        p = Parameter([[1.0]])
        p.grad = Tensor([[2.0]])
        opt = Adagrad([p], lr=0.1)
        opt.step()

        self.assertAlmostEqual(p._flat[0], 0.9, places=5)

    def test_decaying_step_size(self):
        # Successive updates with constant gradient become smaller due to accumulator growth
        p = Parameter([[5.0]])
        opt = Adagrad([p], lr=1.0)
        updates = []
        for _ in range(3):
            p.grad = Tensor([[1.0]])
            before = p._flat[0]
            opt.step()
            updates.append(before - p._flat[0])

        self.assertGreater(updates[0], updates[1])
        self.assertGreater(updates[1], updates[2])


if __name__ == "__main__":
    unittest.main()
