<div align="center">

# 🔥 MicroTorch

### A Lightweight, Zero-Dependency Tensor Autograd Engine & Neural Network Framework from Scratch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dependencies](https://img.shields.io/badge/dependencies-zero-success.svg)](https://github.com)

*A ground-up implementation of automatic differentiation and deep learning primitives inspired by PyTorch and Micrograd.*

</div>

---

## 💡 Why MicroTorch?

Most tutorials implement scalar-only autograd (like Andrej Karpathy's `micrograd`), which cannot do batched matrix operations or train real neural networks efficiently.

**MicroTorch** implements **full N-dimensional Tensors with reverse-mode automatic differentiation**, matrix multiplication broadcasting, modular neural network layers, Adam optimization, and DAG graph visualizers with **zero external dependencies** (no NumPy, no PyTorch).

---

## 🌟 Key Features

- ⚡ **Reverse-Mode Autograd**: Dynamic computation graph construction with topological sort backpropagation.
- 📐 **Batched Matrix Operations**: Matrix multiplication (`@`), broadcasting, and reductions (`sum`, `mean`).
- 🧠 **PyTorch-Style `nn` Library**: `Module`, `Parameter`, `Linear`, `Sequential`, `ReLU`, `Sigmoid`, `Tanh`, `MSELoss`, and `CrossEntropyLoss`.
- 🚀 **First-Class Optimizers**: `SGD` (with momentum) and `Adam` (with adaptive moments and bias correction).
- 📊 **Visual Computation Graphs**: Trace and render computational DAGs in ASCII trees or GitHub-ready Mermaid diagrams.
- 🧪 **100% Zero Dependencies**: Runs out of the box with standard Python 3.

---

## 🚀 Quickstart

### 1. Minimal Autograd Example
```python
from microtorch import Tensor

# Define tensors with gradient tracking
x = Tensor(2.0, requires_grad=True)
y = Tensor(3.0, requires_grad=True)

# Build dynamic computation graph
z = (x * y + 3.0) ** 2.0

# Backpropagation
z.backward()

print(f"z value : {z.item()}")     # 81.0
print(f"dz / dx : {x.grad.item()}") # 54.0
print(f"dz / dy : {y.grad.item()}") # 36.0
```

### 2. Training a Neural Network
```python
from microtorch import Tensor, Sequential, Linear, ReLU, MSELoss, Adam

# XOR Dataset
X = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = Tensor([[0.0], [1.0], [1.0], [0.0]])

# Define Model (2 -> 8 -> 1)
model = Sequential(
    Linear(2, 8),
    ReLU(),
    Linear(8, 1)
)

criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=0.05)

# Training loop
for epoch in range(150):
    loss = criterion(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Final Loss: {loss.item():.6f}")
print("Predictions:", model(X).data)
```

---

## 📊 Computation Graph Visualizer

Print ASCII computational DAGs directly in your terminal:

```python
from microtorch import to_ascii_tree
print(to_ascii_tree(loss))
```

Output:
```text
└── Tensor(shape=() [op: *])
    ├── Tensor(shape=() [op: sum])
    │   └── Tensor(shape=(4, 1) [op: *])
    │       └── Tensor(shape=(4, 1) [op: +])
    │           ├── Tensor(shape=(1, 1))
    │           └── Tensor(shape=(4, 1) [op: @])
    │               ├── Tensor(shape=(8, 1))
    │               └── Tensor(shape=(4, 8) [op: ReLU])
    ...
```

---

## 🧪 Testing

Run the automated test suite:
```bash
python3 -m unittest discover -s tests -v
```

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
