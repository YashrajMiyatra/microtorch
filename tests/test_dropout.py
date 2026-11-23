"""
Unit tests for Dropout module and train/eval mode switching in MicroTorch.
"""

import unittest
from microtorch.tensor import Tensor
from microtorch.nn import Dropout, Linear, Sequential


class TestDropout(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            Dropout(p=1.0)
        with self.assertRaises(ValueError):
            Dropout(p=-0.1)

    def test_eval_mode_is_identity(self):
        drop = Dropout(p=0.5)
        drop.eval()
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        out = drop(x)
        self.assertEqual(out.data, x.data)

        # Backward in eval mode
        loss = out.sum()
        loss.backward()
        self.assertEqual(x.grad.data, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_train_mode_scaling(self):
        drop = Dropout(p=0.5)
        drop.train()
        x = Tensor([[2.0, 2.0, 2.0, 2.0]], requires_grad=True)
        out = drop(x)
        # Any non-zero element should be scaled by 1 / (1 - 0.5) = 2.0 -> value = 4.0
        for val in out._flat:
            self.assertIn(val, [0.0, 4.0])

    def test_sequential_train_eval_propagation(self):
        model = Sequential(Linear(2, 4), Dropout(0.2), Linear(4, 1))
        self.assertTrue(model.training)
        model.eval()
        self.assertFalse(model.training)
        self.assertFalse(model.layers[1].training)
        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.layers[1].training)


if __name__ == "__main__":
    unittest.main()
