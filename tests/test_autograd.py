import unittest
from microtorch.tensor import Tensor


class TestAutograd(unittest.TestCase):
    def test_scalar_operations(self):
        # f(x, y) = (x * y + 3) ** 2
        x = Tensor(2.0, requires_grad=True)
        y = Tensor(3.0, requires_grad=True)
        z = (x * y + 3.0) ** 2.0
        z.backward()

        # df/dx = 2 * (x * y + 3) * y = 2 * (6 + 3) * 3 = 54
        # df/dy = 2 * (x * y + 3) * x = 2 * (6 + 3) * 2 = 36
        self.assertEqual(z.item(), 81.0)
        self.assertAlmostEqual(x.grad.item(), 54.0)
        self.assertAlmostEqual(y.grad.item(), 36.0)

    def test_matrix_multiplication_grad(self):
        # A: (2, 2), B: (2, 2)
        A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        B = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        C = A @ B
        s = C.sum()
        s.backward()

        # d(sum(A @ B)) / dA_ik = sum_j B_kj
        # Row 0: [sum(B[:, 0]), sum(B[:, 1])] = [11, 15]
        # For A: grad should be B summed along cols = [11, 15] for each row
        self.assertEqual(A.grad.shape, (2, 2))
        self.assertEqual(B.grad.shape, (2, 2))

    def test_relu_gradient(self):
        x = Tensor([-2.0, 0.5, 3.0], requires_grad=True)
        y = x.relu().sum()
        y.backward()
        self.assertEqual(x.grad.data, [0.0, 1.0, 1.0])


    def test_tanh_gradient(self):
        x = Tensor(0.0, requires_grad=True)
        y = x.tanh()
        y.backward()
        self.assertAlmostEqual(y.item(), 0.0)
        self.assertAlmostEqual(x.grad.item(), 1.0)

    def test_sigmoid_gradient(self):
        x = Tensor(0.0, requires_grad=True)
        y = x.sigmoid()
        y.backward()
        self.assertAlmostEqual(y.item(), 0.5)
        self.assertAlmostEqual(x.grad.item(), 0.25)


if __name__ == "__main__":

    unittest.main()
