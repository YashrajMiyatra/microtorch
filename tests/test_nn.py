import unittest
from microtorch.tensor import Tensor
from microtorch.nn import Linear, Sequential, ReLU, MSELoss
from microtorch.optim import SGD, AdamW


class TestNeuralNetwork(unittest.TestCase):
    def test_linear_forward_backward(self):
        fc = Linear(2, 3)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        out = fc(x)
        self.assertEqual(out.shape, (2, 3))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(fc.weight.grad)
        self.assertEqual(fc.weight.grad.shape, (2, 3))

    def test_adamw_optimizer(self):
        model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
        optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
        criterion = MSELoss()

        x = Tensor([[1.0, 1.0]])
        y = Tensor([[2.0]])

        pred0 = model(x)
        loss0 = criterion(pred0, y).item()

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        pred1 = model(x)
        loss1 = criterion(pred1, y).item()
        self.assertLess(loss1, loss0)

    def test_sequential_training_step(self):
        model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
        optimizer = SGD(model.parameters(), lr=0.01)
        criterion = MSELoss()

        x = Tensor([[1.0, 1.0]])
        y = Tensor([[2.0]])

        # Initial loss
        pred0 = model(x)
        loss0 = criterion(pred0, y).item()

        # Step
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        # Loss should decrease
        pred1 = model(x)
        loss1 = criterion(pred1, y).item()
        self.assertLess(loss1, loss0)


if __name__ == "__main__":
    unittest.main()
