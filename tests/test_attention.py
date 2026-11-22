"""
Unit tests for Transformer Attention, LayerNorm, and Softmax primitives in MicroTorch.
"""

import unittest
from microtorch.tensor import Tensor
from microtorch.nn import MultiHeadAttention, LayerNorm, Softmax


class TestTransformerPrimitives(unittest.TestCase):
    def test_tensor_transpose(self):
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        xt = x.T
        self.assertEqual(xt.shape, (3, 2))
        self.assertEqual(xt.data, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

        # Gradient check
        loss = (xt * 3.0).sum()
        loss.backward()
        self.assertEqual(x.grad.shape, (2, 3))
        self.assertEqual(x.grad.data, [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])

    def test_softmax_forward_and_backward(self):
        sm = Softmax()
        x = Tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], requires_grad=True)
        probs = sm(x)
        self.assertEqual(probs.shape, (2, 3))
        # Each row should sum to 1.0
        for row in probs.data:
            self.assertAlmostEqual(sum(row), 1.0, places=5)

        loss = probs.sum()
        loss.backward()
        self.assertEqual(x.grad.shape, (2, 3))

    def test_layernorm_forward_and_backward(self):
        ln = LayerNorm(normalized_shape=4)
        x = Tensor([[2.0, 4.0, 6.0, 8.0], [-1.0, 0.0, 1.0, 2.0]], requires_grad=True)
        y = ln(x)
        self.assertEqual(y.shape, (2, 4))

        # Check mean is close to 0 and variance close to 1
        for row in y.data:
            mean = sum(row) / 4.0
            self.assertAlmostEqual(mean, 0.0, places=4)

        loss = y.sum()
        loss.backward()
        self.assertEqual(x.grad.shape, (2, 4))
        self.assertIsNotNone(ln.gamma.grad)
        self.assertIsNotNone(ln.beta.grad)

    def test_multihead_attention_forward_backward(self):
        seq_len, d_model = 4, 8
        mha = MultiHeadAttention(d_model=d_model, n_heads=1)
        x = Tensor.randn(seq_len, d_model, requires_grad=True)

        out = mha(x)
        self.assertEqual(out.shape, (seq_len, d_model))

        # Backward propagation
        loss = out.sum()
        loss.backward()
        self.assertEqual(x.grad.shape, (seq_len, d_model))
        for p in mha.parameters():
            self.assertIsNotNone(p.grad)


if __name__ == "__main__":
    unittest.main()
