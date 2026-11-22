"""
MicroTorch Core Tensor Engine.
Provides multi-dimensional Tensors with reverse-mode automatic differentiation (Autograd).
Zero external dependencies required.

Mathematical Autograd Architecture:
- Forward pass constructs dynamic computation DAG nodes on the fly.
- Reverse-mode backward pass topologically sorts DAG nodes via Kahn's / DFS algorithm.
- Multi-dimensional gradient broadcasting propagates adjoint vectors across matrix operations.
"""

from __future__ import annotations
import math
import random
from typing import Any, Callable, List, Optional, Set, Tuple, Union


def _shape_of(data: Any) -> Tuple[int, ...]:
    """Recursively computes shape of nested list."""
    if isinstance(data, (int, float)):
        return ()
    if not isinstance(data, list):
        raise TypeError(f"Unsupported data type: {type(data)}")
    if len(data) == 0:
        return (0,)
    return (len(data),) + _shape_of(data[0])


def _flatten(data: Any) -> List[float]:
    """Flattens nested lists into a 1D list of floats."""
    if isinstance(data, (int, float)):
        return [float(data)]
    out = []
    for item in data:
        out.extend(_flatten(item))
    return out


def _unflatten(flat: List[float], shape: Tuple[int, ...]) -> Any:
    """Reconstructs nested list from flat buffer given shape."""
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return flat[: shape[0]]
    size = 1
    for dim in shape[1:]:
        size *= dim
    return [_unflatten(flat[i * size : (i + 1) * size], shape[1:]) for i in range(shape[0])]


def _zeros(shape: Tuple[int, ...]) -> List[float]:
    size = 1
    for s in shape:
        size *= s
    return [0.0] * size


class Tensor:
    """
    A multi-dimensional array with automatic differentiation.
    """

    def __init__(
        self,
        data: Union[int, float, List[Any]],
        requires_grad: bool = False,
        _children: Tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        if isinstance(data, (int, float)):
            self.shape: Tuple[int, ...] = ()
            self._flat: List[float] = [float(data)]
        elif isinstance(data, list):
            self.shape = _shape_of(data)
            self._flat = _flatten(data)
        elif isinstance(data, Tensor):
            self.shape = data.shape
            self._flat = list(data._flat)
        else:
            raise TypeError(f"Invalid tensor input type: {type(data)}")

        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Tensor] = set(_children)
        self._op = _op

    @property
    def data(self) -> Any:
        return _unflatten(self._flat, self.shape)

    @classmethod
    def zeros(cls, *shape: int, requires_grad: bool = False) -> Tensor:
        flat = _zeros(shape)
        t = cls.__new__(cls)
        t.shape = shape
        t._flat = flat
        t.requires_grad = requires_grad
        t.grad = None
        t._backward = lambda: None
        t._prev = set()
        t._op = "zeros"
        return t

    @classmethod
    def ones(cls, *shape: int, requires_grad: bool = False) -> Tensor:
        size = 1
        for s in shape:
            size *= s
        flat = [1.0] * size
        t = cls.__new__(cls)
        t.shape = shape
        t._flat = flat
        t.requires_grad = requires_grad
        t.grad = None
        t._backward = lambda: None
        t._prev = set()
        t._op = "ones"
        return t

    @classmethod
    def randn(cls, *shape: int, requires_grad: bool = False) -> Tensor:
        size = 1
        for s in shape:
            size *= s
        # Box-Muller transform or standard Gaussian random
        flat = [random.gauss(0.0, 1.0) for _ in range(size)]
        t = cls.__new__(cls)
        t.shape = shape
        t._flat = flat
        t.requires_grad = requires_grad
        t.grad = None
        t._backward = lambda: None
        t._prev = set()
        t._op = "randn"
        return t

    def item(self) -> float:
        if len(self._flat) != 1:
            raise ValueError("item() can only be called on single-element tensors")
        return self._flat[0]

    def zero_grad(self) -> None:
        self.grad = None

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.data}{', requires_grad=True' if self.requires_grad else ''})"

    # ------------------ Mathematical Operations ------------------ #

    def __add__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        # Elementwise or scalar broadcasting
        if self.shape == other.shape:
            out_flat = [a + b for a, b in zip(self._flat, other._flat)]
            out = Tensor(_unflatten(out_flat, self.shape), requires_grad=(self.requires_grad or other.requires_grad), _children=(self, other), _op="+")

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = Tensor.zeros(*self.shape)
                    for i in range(len(self._flat)):
                        self.grad._flat[i] += out.grad._flat[i]
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = Tensor.zeros(*other.shape)
                    for i in range(len(other._flat)):
                        other.grad._flat[i] += out.grad._flat[i]

            out._backward = _backward
            return out
        elif other.shape == ():  # Scalar addition
            val = other._flat[0]
            out_flat = [a + val for a in self._flat]
            out = Tensor(_unflatten(out_flat, self.shape), requires_grad=(self.requires_grad or other.requires_grad), _children=(self, other), _op="+")

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = Tensor.zeros(*self.shape)
                    for i in range(len(self._flat)):
                        self.grad._flat[i] += out.grad._flat[i]
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = Tensor.zeros()
                    other.grad._flat[0] += sum(out.grad._flat)

            out._backward = _backward
            return out
        elif len(self.shape) == 2 and len(other.shape) == 2 and self.shape[1] == other.shape[1] and other.shape[0] == 1:
            # (M, N) + (1, N) broadcasting
            M, N = self.shape
            out_flat = [0.0] * (M * N)
            for i in range(M):
                for j in range(N):
                    out_flat[i * N + j] = self._flat[i * N + j] + other._flat[j]
            out = Tensor(_unflatten(out_flat, self.shape), requires_grad=(self.requires_grad or other.requires_grad), _children=(self, other), _op="+")

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = Tensor.zeros(*self.shape)
                    for k in range(len(self._flat)):
                        self.grad._flat[k] += out.grad._flat[k]
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = Tensor.zeros(*other.shape)
                    for j in range(N):
                        col_sum = sum(out.grad._flat[i * N + j] for i in range(M))
                        other.grad._flat[j] += col_sum

            out._backward = _backward
            return out
        else:
            raise ValueError(f"Shape mismatch in add: {self.shape} vs {other.shape}")

    def __radd__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self + other

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self + (-other)

    def __rsub__(self, other: Union[Tensor, float, int]) -> Tensor:
        return (-self) + other

    def __mul__(self, other: Union[Tensor, float, int]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.shape == other.shape:
            out_flat = [a * b for a, b in zip(self._flat, other._flat)]
            out = Tensor(_unflatten(out_flat, self.shape), requires_grad=(self.requires_grad or other.requires_grad), _children=(self, other), _op="*")

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = Tensor.zeros(*self.shape)
                    for i in range(len(self._flat)):
                        self.grad._flat[i] += other._flat[i] * out.grad._flat[i]
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = Tensor.zeros(*other.shape)
                    for i in range(len(other._flat)):
                        other.grad._flat[i] += self._flat[i] * out.grad._flat[i]

            out._backward = _backward
            return out
        elif other.shape == ():  # Scalar multiplication
            val = other._flat[0]
            out_flat = [a * val for a in self._flat]
            out = Tensor(_unflatten(out_flat, self.shape), requires_grad=(self.requires_grad or other.requires_grad), _children=(self, other), _op="*")

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = Tensor.zeros(*self.shape)
                    for i in range(len(self._flat)):
                        self.grad._flat[i] += val * out.grad._flat[i]
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = Tensor.zeros()
                    other.grad._flat[0] += sum(a * g for a, g in zip(self._flat, out.grad._flat))

            out._backward = _backward
            return out
        else:
            raise ValueError(f"Shape mismatch in mul: {self.shape} vs {other.shape}")

    def __rmul__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self * other

    def __truediv__(self, other: Union[Tensor, float, int]) -> Tensor:
        return self * (other ** -1.0)

    def __rtruediv__(self, other: Union[Tensor, float, int]) -> Tensor:
        return Tensor(other) / self

    def __pow__(self, power: Union[int, float]) -> Tensor:
        out_flat = [a ** power for a in self._flat]
        out = Tensor(_unflatten(out_flat, self.shape), requires_grad=self.requires_grad, _children=(self,), _op=f"**{power}")

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor.zeros(*self.shape)
                for i in range(len(self._flat)):
                    self.grad._flat[i] += power * (self._flat[i] ** (power - 1)) * out.grad._flat[i]

        out._backward = _backward
        return out

    # ------------------ Matrix Multiplication ------------------ #

    def matmul(self, other: Tensor) -> Tensor:
        """2D Matrix Multiplication: (M, K) x (K, N) -> (M, N)."""
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError(f"matmul requires 2D tensors, got {self.shape} and {other.shape}")

        M, K1 = self.shape
        K2, N = other.shape
        if K1 != K2:
            raise ValueError(f"Cannot multiply matrices with shapes ({M}, {K1}) and ({K2}, {N})")

        out_flat = [0.0] * (M * N)
        for i in range(M):
            for k in range(K1):
                a_ik = self._flat[i * K1 + k]
                for j in range(N):
                    out_flat[i * N + j] += a_ik * other._flat[k * N + j]

        out = Tensor(_unflatten(out_flat, (M, N)), requires_grad=(self.requires_grad or other.requires_grad), _children=(self, other), _op="@")

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor.zeros(*self.shape)
                # dL/dA = dL/dOut @ B^T : (M, N) @ (N, K) -> (M, K)
                for i in range(M):
                    for k in range(K1):
                        grad_sum = 0.0
                        for j in range(N):
                            grad_sum += out.grad._flat[i * N + j] * other._flat[k * N + j]
                        self.grad._flat[i * K1 + k] += grad_sum

            if other.requires_grad:
                if other.grad is None:
                    other.grad = Tensor.zeros(*other.shape)
                # dL/dB = A^T @ dL/dOut : (K, M) @ (M, N) -> (K, N)
                for k in range(K2):
                    for j in range(N):
                        grad_sum = 0.0
                        for i in range(M):
                            grad_sum += self._flat[i * K1 + k] * out.grad._flat[i * N + j]
                        other.grad._flat[k * N + j] += grad_sum

        out._backward = _backward
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.matmul(other)

    def transpose(self) -> Tensor:
        """Transposes a 2D tensor: (M, N) -> (N, M)."""
        if len(self.shape) != 2:
            raise ValueError(f"transpose requires a 2D tensor, got shape {self.shape}")
        M, N = self.shape
        out_flat = [0.0] * (M * N)
        for i in range(M):
            for j in range(N):
                out_flat[j * M + i] = self._flat[i * N + j]
        out = Tensor(_unflatten(out_flat, (N, M)), requires_grad=self.requires_grad, _children=(self,), _op="T")

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor.zeros(*self.shape)
                for i in range(M):
                    for j in range(N):
                        self.grad._flat[i * N + j] += out.grad._flat[j * M + i]

        out._backward = _backward
        return out

    @property
    def T(self) -> Tensor:
        """Shorthand transpose property: A.T."""
        return self.transpose()

    # ------------------ Non-linearities & Reductions ------------------ #

    def relu(self) -> Tensor:
        out_flat = [max(0.0, x) for x in self._flat]
        out = Tensor(_unflatten(out_flat, self.shape), requires_grad=self.requires_grad, _children=(self,), _op="ReLU")

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor.zeros(*self.shape)
                for i in range(len(self._flat)):
                    self.grad._flat[i] += (out.grad._flat[i] if self._flat[i] > 0 else 0.0)

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        out_flat = [math.tanh(x) for x in self._flat]
        out = Tensor(_unflatten(out_flat, self.shape), requires_grad=self.requires_grad, _children=(self,), _op="Tanh")

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor.zeros(*self.shape)
                for i in range(len(self._flat)):
                    t = out_flat[i]
                    self.grad._flat[i] += (1.0 - t * t) * out.grad._flat[i]

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        out_flat = [1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0))) for x in self._flat]
        out = Tensor(_unflatten(out_flat, self.shape), requires_grad=self.requires_grad, _children=(self,), _op="Sigmoid")

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor.zeros(*self.shape)
                for i in range(len(self._flat)):
                    s = out_flat[i]
                    self.grad._flat[i] += s * (1.0 - s) * out.grad._flat[i]

        out._backward = _backward
        return out

    def sum(self) -> Tensor:
        total = sum(self._flat)
        out = Tensor(total, requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = Tensor.zeros(*self.shape)
                g = out.grad._flat[0]
                for i in range(len(self._flat)):
                    self.grad._flat[i] += g

        out._backward = _backward
        return out

    def mean(self) -> Tensor:
        return self.sum() * (1.0 / len(self._flat))

    # ------------------ Reverse-Mode Autograd Engine ------------------ #

    def backward(self) -> None:
        """
        Executes reverse-mode automatic differentiation.
        Topologically sorts the DAG and executes backpropagation.
        """
        topo: List[Tensor] = []
        visited: Set[Tensor] = set()

        def build_topo(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed gradient of the root node with 1.0
        self.grad = Tensor.ones(*self.shape)

        # Call _backward in reverse topological order
        for node in reversed(topo):
            node._backward()
