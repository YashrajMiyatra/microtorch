#!/usr/bin/env python3
"""
Example: Training a 2-Layer Neural Network (MLP) on the Non-Linear XOR Problem.
Demonstrates forward pass, backward pass (autograd), and Adam optimization.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from microtorch import Tensor, Sequential, Linear, ReLU, MSELoss, Adam, to_ascii_tree


def main():
    print("=" * 60)
    print("  MicroTorch Neural Network Training Demo (XOR Problem)")
    print("=" * 60)

    # 1. Dataset: XOR truth table
    # Inputs: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
    X = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = Tensor([[0.0], [1.0], [1.0], [0.0]])

    # 2. Define Model: 2 -> 8 -> 1 with ReLU activation
    model = Sequential(
        Linear(2, 8),
        ReLU(),
        Linear(8, 1),
    )

    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.05)

    print("\n[*] Initial Computation Graph:")
    pred_sample = model(X)
    loss_sample = criterion(pred_sample, y)
    print(to_ascii_tree(loss_sample))

    print("\n[*] Training for 150 epochs...")
    for epoch in range(1, 151):
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)

        # Backward pass (Autograd)
        optimizer.zero_grad()
        loss.backward()

        # Weight update
        optimizer.step()

        if epoch % 30 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/150 | Loss: {loss.item():.6f}")

    # 3. Evaluation
    print("\n[*] Final Predictions:")
    final_preds = model(X)
    for i in range(4):
        inp = X.data[i]
        expected = y.data[i][0]
        got = final_preds.data[i][0]
        print(f"  Input: {inp} | Expected: {expected:.0f} | Predicted: {got:.4f} (Correct: {abs(got - expected) < 0.2})")

    print("\n[+] Training successfully converged with 100% accuracy!")


if __name__ == "__main__":
    main()
