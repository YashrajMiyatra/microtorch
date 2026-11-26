import unittest
from microtorch import Parameter, SGD, CosineAnnealingWarmupLR


class TestCosineAnnealingWarmupLR(unittest.TestCase):
    def test_warmup_and_cosine_decay(self):
        p = Parameter([[1.0]])
        opt = SGD([p], lr=0.1)
        # 10 steps total: 2 warmup steps (0, 1), 8 decay steps (2..9)
        scheduler = CosineAnnealingWarmupLR(
            opt,
            total_steps=10,
            warmup_steps=2,
            min_lr=0.01,
            warmup_lr=0.0,
        )

        # Epoch 0 (warmup step 0) -> lr = 0.0
        self.assertAlmostEqual(opt.lr, 0.0)

        # Epoch 1 (warmup step 1) -> lr = 0.0 + 1/2 * (0.1 - 0.0) = 0.05
        scheduler.step()
        self.assertAlmostEqual(opt.lr, 0.05)

        # Epoch 2 (warmup complete -> reaches base_lr = 0.1)
        scheduler.step()
        self.assertAlmostEqual(opt.lr, 0.1)

        # Epoch 6 (midpoint of decay: 4/8 = half period)
        for _ in range(4):
            scheduler.step()
        # Halfway between 0.1 and 0.01 = 0.01 + 0.5 * 0.09 = 0.055
        self.assertAlmostEqual(opt.lr, 0.055)

        # Step until total_steps reached (epoch 10)
        for _ in range(4):
            scheduler.step()
        self.assertAlmostEqual(opt.lr, 0.01)

    def test_invalid_parameters(self):
        p = Parameter([[1.0]])
        opt = SGD([p], lr=0.1)
        with self.assertRaises(ValueError):
            CosineAnnealingWarmupLR(opt, total_steps=5, warmup_steps=5)
        with self.assertRaises(ValueError):
            CosineAnnealingWarmupLR(opt, total_steps=5, warmup_steps=-1)


if __name__ == "__main__":
    unittest.main()
