import unittest
import math
from microtorch import Tensor, Parameter, SGD, GradScaler


class TestGradScaler(unittest.TestCase):
    def test_normal_scale_and_unscale_step(self):
        p = Parameter([[2.0]])
        opt = SGD([p], lr=0.1)
        scaler = GradScaler(init_scale=1024.0, growth_interval=2)

        loss = p * 4.0  # loss = 8.0, raw grad = 4.0
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        # Scaled grad is 4.0 * 1024.0 = 4096.0
        self.assertAlmostEqual(p.grad.data[0][0], 4096.0)

        # Step unscales back to 4.0 and performs update: 2.0 - 0.1 * 4.0 = 1.6
        stepped = scaler.step(opt)
        self.assertTrue(stepped)
        self.assertAlmostEqual(p.data[0][0], 1.6)

        # Scale update
        scaler.update()
        self.assertEqual(scaler.get_scale(), 1024.0)

    def test_growth_after_clean_intervals(self):
        p = Parameter([[1.0]])
        opt = SGD([p], lr=0.01)
        scaler = GradScaler(init_scale=100.0, growth_factor=2.0, growth_interval=2)

        # Step 1 clean
        opt.zero_grad()
        scaler.scale(p * 1.0).backward()
        scaler.step(opt)
        scaler.update()
        self.assertEqual(scaler.get_scale(), 100.0)

        # Step 2 clean -> hits growth interval -> scale doubles
        opt.zero_grad()
        scaler.scale(p * 1.0).backward()
        scaler.step(opt)
        scaler.update()
        self.assertEqual(scaler.get_scale(), 200.0)

    def test_inf_gradient_skips_step_and_backs_off(self):
        p = Parameter([[5.0]])
        opt = SGD([p], lr=0.1)
        scaler = GradScaler(init_scale=1000.0, backoff_factor=0.5)

        opt.zero_grad()
        scaled = scaler.scale(p * 1.0)
        scaled.backward()

        # Manually inject inf into gradient
        p.grad._flat[0] = float("inf")

        stepped = scaler.step(opt)
        self.assertFalse(stepped)
        # Weight unchanged
        self.assertEqual(p.data[0][0], 5.0)

        # Update reduces scale
        scaler.update()
        self.assertEqual(scaler.get_scale(), 500.0)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            GradScaler(init_scale=0)
        with self.assertRaises(ValueError):
            GradScaler(growth_factor=1.0)
        with self.assertRaises(ValueError):
            GradScaler(backoff_factor=1.5)
        with self.assertRaises(ValueError):
            GradScaler(growth_interval=0)


if __name__ == "__main__":
    unittest.main()
