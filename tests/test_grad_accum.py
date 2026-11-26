import unittest
from microtorch import Tensor, Parameter, SGD, GradientAccumulator


class TestGradientAccumulator(unittest.TestCase):
    def test_accumulation_step_cycle(self):
        # Param initialized to 10.0
        p = Parameter([[10.0]])
        opt = SGD([p], lr=0.1)
        accum = GradientAccumulator(opt, accumulation_steps=3)

        # Micro-step 1: loss = p * 2 -> grad = 2 / 3
        loss1 = p * 2.0
        stepped = accum.accumulate(loss1)
        self.assertFalse(stepped)
        self.assertEqual(accum.step_count, 1)
        self.assertAlmostEqual(p.data[0][0], 10.0)

        # Micro-step 2: loss = p * 1 -> grad = 1 / 3
        loss2 = p * 1.0
        stepped = accum.accumulate(loss2)
        self.assertFalse(stepped)
        self.assertEqual(accum.step_count, 2)
        self.assertAlmostEqual(p.data[0][0], 10.0)

        # Micro-step 3: loss = p * 3 -> grad = 3 / 3
        # Total accumulated grad = (2 + 1 + 3) / 3 = 6 / 3 = 2.0
        # Weight update = 10.0 - (0.1 * 2.0) = 9.8
        loss3 = p * 3.0
        stepped = accum.accumulate(loss3)
        self.assertTrue(stepped)
        self.assertEqual(accum.step_count, 3)
        self.assertAlmostEqual(p.data[0][0], 9.8)
        # Verify gradients were reset
        self.assertIsNone(p.grad)

    def test_flush_remaining_steps(self):
        p = Parameter([[5.0]])
        opt = SGD([p], lr=0.1)
        accum = GradientAccumulator(opt, accumulation_steps=4)

        loss1 = p * 2.0  # scaled grad = 2 / 4 = 0.5
        accum.accumulate(loss1)
        loss2 = p * 2.0  # scaled grad = 2 / 4 = 0.5
        accum.accumulate(loss2)

        self.assertEqual(accum.step_count, 2)
        self.assertAlmostEqual(p.data[0][0], 5.0)

        # Flush prematurely
        flushed = accum.flush()
        self.assertTrue(flushed)
        # Update = 5.0 - (0.1 * 1.0) = 4.9
        self.assertAlmostEqual(p.data[0][0], 4.9)
        self.assertEqual(accum.step_count, 0)

        # Subsequent flush with nothing pending returns False
        self.assertFalse(accum.flush())

    def test_accumulation_with_gradient_clipping(self):
        p = Parameter([[10.0]])
        opt = SGD([p], lr=1.0)
        # Max norm = 1.0, accumulation_steps = 2
        accum = GradientAccumulator(opt, accumulation_steps=2, max_norm=1.0)

        # Micro-step 1: grad = 10.0 / 2 = 5.0
        accum.accumulate(p * 10.0)
        # Micro-step 2: grad = 10.0 / 2 = 5.0
        # Total grad before clip = 10.0, clipped to 1.0
        stepped = accum.accumulate(p * 10.0)
        self.assertTrue(stepped)
        # p = 10.0 - 1.0 * (1.0 / (10.0 + 1e-6) * 10.0) ~ 9.0
        self.assertAlmostEqual(p.data[0][0], 9.0, places=5)

    def test_zero_grad_reset(self):
        p = Parameter([[1.0]])
        opt = SGD([p], lr=0.1)
        accum = GradientAccumulator(opt, accumulation_steps=3)
        accum.accumulate(p * 4.0)
        self.assertEqual(accum.step_count, 1)

        accum.zero_grad()
        self.assertEqual(accum.step_count, 0)
        self.assertIsNone(p.grad)

    def test_invalid_accumulation_steps(self):
        p = Parameter([[1.0]])
        opt = SGD([p], lr=0.1)
        with self.assertRaises(ValueError):
            GradientAccumulator(opt, accumulation_steps=0)


if __name__ == "__main__":
    unittest.main()
