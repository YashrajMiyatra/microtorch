"""
MicroTorch Neural Network (nn) Library.
Defines base Module, Parameters, Layers (Linear, Sequential, Activations), and Loss Functions.
"""

from __future__ import annotations
import math
from typing import Any, List, Optional
from .tensor import Tensor


class Parameter(Tensor):
    """A trainable parameter within a neural network."""

    def __init__(self, data: Any) -> None:
        super().__init__(data, requires_grad=True)


class Module:
    """Base class for all neural network modules."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def parameters(self) -> List[Parameter]:
        params = []
        for _, value in self.__dict__.items():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Parameter):
                        params.append(item)
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    """Fully connected linear layer: Y = X @ W + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.in_features = in_features
        self.out_features = out_features

        # Xavier / Glorot initialization: N(0, sqrt(2 / (in + out)))
        std = math.sqrt(2.0 / (in_features + out_features))
        raw_w = Tensor.randn(in_features, out_features) * std
        self.weight = Parameter(raw_w.data)

        if bias:
            raw_b = Tensor.zeros(1, out_features)
            self.bias: Optional[Parameter] = Parameter(raw_b.data)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Sequential(Module):
    """Chains multiple modules sequentially."""

    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class MSELoss(Module):
    """Mean Squared Error loss: (1/N) * sum((pred - target)^2)."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        sq = diff * diff
        return sq.mean()


class CrossEntropyLoss(Module):
    """
    Softmax Cross-Entropy loss for multi-class classification.
    Expects pred of shape (Batch, Classes) and one-hot target of shape (Batch, Classes).
    """

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        batch_size, num_classes = pred.shape
        loss_sum = 0.0

        # Stable Log-Sum-Exp Softmax
        for b in range(batch_size):
            logits = pred._flat[b * num_classes : (b + 1) * num_classes]
            targets = target._flat[b * num_classes : (b + 1) * num_classes]

            max_logit = max(logits)
            exp_sum = sum(math.exp(x - max_logit) for x in logits)
            log_sum_exp = max_logit + math.log(exp_sum)

            for c in range(num_classes):
                if targets[c] > 0:
                    loss_sum += targets[c] * (log_sum_exp - logits[c])

        mean_loss = loss_sum / batch_size
        out = Tensor(mean_loss, requires_grad=True, _children=(pred,), _op="CrossEntropy")

        def _backward():
            if pred.requires_grad:
                if pred.grad is None:
                    pred.grad = Tensor.zeros(*pred.shape)
                for b in range(batch_size):
                    logits = pred._flat[b * num_classes : (b + 1) * num_classes]
                    targets = target._flat[b * num_classes : (b + 1) * num_classes]
                    max_logit = max(logits)
                    exps = [math.exp(x - max_logit) for x in logits]
                    exp_sum = sum(exps)
                    probs = [e / exp_sum for e in exps]

                    for c in range(num_classes):
                        # dL/dz = (prob - target) / batch_size
                        grad_val = (probs[c] - targets[c]) / batch_size
                        pred.grad._flat[b * num_classes + c] += grad_val * out.grad._flat[0]

        out._backward = _backward
        return out
