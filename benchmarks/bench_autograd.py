"""Benchmark backward pass on deep computational graphs."""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from microtorch.tensor import Tensor


def benchmark_deep_graph(depth: int = 200, iterations: int = 50):
    print(f"Benchmarking computational graph depth={depth} over {iterations} iterations...")
    times = []
    for _ in range(iterations):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x
        for _ in range(depth):
            y = y + x
        loss = y.sum()

        start = time.perf_counter()
        loss.backward()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    print(f"Average backward pass: {avg_ms:.3f} ms | Min: {min_ms:.3f} ms")


if __name__ == "__main__":
    benchmark_deep_graph()
