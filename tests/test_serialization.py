"""
Unit tests for model weights serialization and state_dict save/load in MicroTorch.
"""

import os
import tempfile
import unittest
from microtorch.tensor import Tensor
from microtorch.nn import Linear, Sequential, ReLU


class TestSerialization(unittest.TestCase):
    def test_state_dict_and_load(self):
        m1 = Sequential(
            Linear(2, 3),
            ReLU(),
            Linear(3, 1)
        )
        state = m1.state_dict()
        self.assertIn("layers.0.weight", state)
        self.assertIn("layers.0.bias", state)
        self.assertIn("layers.2.weight", state)
        self.assertIn("layers.2.bias", state)

        # Create a fresh model with different initial weights
        m2 = Sequential(
            Linear(2, 3),
            ReLU(),
            Linear(3, 1)
        )
        x = Tensor([[1.0, 2.0]])
        out1 = m1(x)
        out2_before = m2(x)
        self.assertNotEqual(out1.data, out2_before.data)

        # Load weights into m2
        m2.load_state_dict(state)
        out2_after = m2(x)
        self.assertEqual(out1.data, out2_after.data)

    def test_save_and_load_checkpoint(self):
        model = Sequential(Linear(4, 2), ReLU(), Linear(2, 1))
        x = Tensor([[0.5, -1.0, 2.0, 1.5]])
        orig_out = model(x).data

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "model.json")
            model.save(ckpt)
            self.assertTrue(os.path.exists(ckpt))

            restored = Sequential(Linear(4, 2), ReLU(), Linear(2, 1))
            restored.load(ckpt)
            restored_out = restored(x).data
            self.assertEqual(orig_out, restored_out)


if __name__ == "__main__":
    unittest.main()
