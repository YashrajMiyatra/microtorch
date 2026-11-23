"""
MicroTorch Neural Network (nn) Library.
Defines base Module, Parameters, Layers (Linear, Sequential, Activations), and Loss Functions.
"""

from __future__ import annotations
import math
from typing import Any, List, Optional
from .tensor import Tensor, _unflatten


class Parameter(Tensor):
    """A trainable parameter within a neural network."""

    def __init__(self, data: Any) -> None:
        super().__init__(data, requires_grad=True)


class Module:
    """Base class for all neural network modules."""

    def __init__(self) -> None:
        self.training = True

    def train(self, mode: bool = True) -> Module:
        """Sets the module in training or evaluation mode."""
        self.training = mode
        for _, value in self.__dict__.items():
            if isinstance(value, Module):
                value.train(mode)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self) -> Module:
        """Sets the module in evaluation mode."""
        return self.train(False)

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

    def state_dict(self, prefix: str = "") -> dict[str, Any]:
        """Returns a dictionary containing a whole state of the module."""
        state = {}
        for name, value in self.__dict__.items():
            full_name = f"{prefix}{name}" if prefix else name
            if isinstance(value, Parameter):
                state[full_name] = value.data
            elif isinstance(value, Module):
                state.update(value.state_dict(prefix=f"{full_name}."))
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    item_prefix = f"{full_name}.{idx}."
                    if isinstance(item, Module):
                        state.update(item.state_dict(prefix=item_prefix))
                    elif isinstance(item, Parameter):
                        state[f"{full_name}.{idx}"] = item.data
        return state

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Copies parameters from state_dict into this module and its descendants."""
        curr_state = self.state_dict()
        if strict:
            missing = set(curr_state.keys()) - set(state_dict.keys())
            unexpected = set(state_dict.keys()) - set(curr_state.keys())
            if missing or unexpected:
                raise KeyError(f"Error loading state_dict: missing={missing}, unexpected={unexpected}")

        def _assign(mod: Module, prefix: str):
            for name, value in mod.__dict__.items():
                full_name = f"{prefix}{name}" if prefix else name
                if isinstance(value, Parameter) and full_name in state_dict:
                    new_val = state_dict[full_name]
                    # Update parameter data in-place
                    value.data = new_val
                    value.shape = Parameter(new_val).shape
                    value._flat = Parameter(new_val)._flat
                elif isinstance(value, Module):
                    _assign(value, prefix=f"{full_name}.")
                elif isinstance(value, list):
                    for idx, item in enumerate(value):
                        item_prefix = f"{full_name}.{idx}"
                        if isinstance(item, Parameter) and item_prefix in state_dict:
                            new_val = state_dict[item_prefix]
                            item.data = new_val
                            item.shape = Parameter(new_val).shape
                            item._flat = Parameter(new_val)._flat
                        elif isinstance(item, Module):
                            _assign(item, prefix=f"{item_prefix}.")

        _assign(self, "")

    def save(self, filepath: str) -> None:
        """Saves model state_dict to a JSON checkpoint."""
        import json, os
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.state_dict(), f, indent=2)

    def load(self, filepath: str) -> None:
        """Loads model state_dict from a JSON checkpoint."""
        import json
        with open(filepath, "r") as f:
            state = json.load(f)
        self.load_state_dict(state)


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
        super().__init__()
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


class Softmax(Module):
    """Row-wise Softmax activation for 2D tensors."""

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) != 2:
            raise ValueError(f"Softmax expects a 2D tensor, got {x.shape}")
        M, N = x.shape
        out_flat = [0.0] * (M * N)
        for i in range(M):
            row = x._flat[i * N : (i + 1) * N]
            max_val = max(row)
            exps = [math.exp(v - max_val) for v in row]
            sum_exp = sum(exps)
            for j in range(N):
                out_flat[i * N + j] = exps[j] / sum_exp

        out = Tensor(
            _unflatten(out_flat, (M, N)),
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="Softmax",
        )

        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = Tensor.zeros(*x.shape)
                for i in range(M):
                    s = out._flat[i * N : (i + 1) * N]
                    g = out.grad._flat[i * N : (i + 1) * N]
                    dot = sum(s[k] * g[k] for k in range(N))
                    for j in range(N):
                        x.grad._flat[i * N + j] += s[j] * (g[j] - dot)

        out._backward = _backward
        return out


class LayerNorm(Module):
    """
    Layer Normalization over the last feature dimension:
    y = ((x - E[x]) / sqrt(Var[x] + eps)) * gamma + beta
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Parameter(Tensor.ones(1, normalized_shape).data)
        self.beta = Parameter(Tensor.zeros(1, normalized_shape).data)

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) != 2:
            raise ValueError(f"LayerNorm requires a 2D tensor, got {x.shape}")
        M, N = x.shape
        if N != self.normalized_shape:
            raise ValueError(f"Expected last dimension {self.normalized_shape}, got {N}")

        out_flat = [0.0] * (M * N)
        for i in range(M):
            row = x._flat[i * N : (i + 1) * N]
            mean = sum(row) / N
            var = sum((v - mean) ** 2 for v in row) / N
            std = math.sqrt(var + self.eps)
            for j in range(N):
                norm = (row[j] - mean) / std
                out_flat[i * N + j] = norm * self.gamma._flat[j] + self.beta._flat[j]

        from .tensor import _unflatten
        out = Tensor(_unflatten(out_flat, (M, N)), requires_grad=True, _children=(x, self.gamma, self.beta), _op="LayerNorm")

        def _backward():
            for i in range(M):
                row = x._flat[i * N : (i + 1) * N]
                mean = sum(row) / N
                var = sum((v - mean) ** 2 for v in row) / N
                std = math.sqrt(var + self.eps)
                dout = out.grad._flat[i * N : (i + 1) * N]
                dnorm = [dout[j] * self.gamma._flat[j] for j in range(N)]

                if self.gamma.requires_grad:
                    if self.gamma.grad is None:
                        self.gamma.grad = Tensor.zeros(1, N)
                    for j in range(N):
                        self.gamma.grad._flat[j] += dout[j] * ((row[j] - mean) / std)

                if self.beta.requires_grad:
                    if self.beta.grad is None:
                        self.beta.grad = Tensor.zeros(1, N)
                    for j in range(N):
                        self.beta.grad._flat[j] += dout[j]

                if x.requires_grad:
                    if x.grad is None:
                        x.grad = Tensor.zeros(*x.shape)
                    sum_dnorm = sum(dnorm)
                    sum_dnorm_x = sum(dnorm[j] * (row[j] - mean) for j in range(N))
                    for j in range(N):
                        dx = (1.0 / (N * std)) * (N * dnorm[j] - sum_dnorm - ((row[j] - mean) / (std ** 2)) * sum_dnorm_x)
                        x.grad._flat[i * N + j] += dx

        out._backward = _backward
        return out


class MultiHeadAttention(Module):
    """
    Scaled Dot-Product Multi-Head Self-Attention primitive for sequence modeling:
    Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) V
    """

    def __init__(self, d_model: int, n_heads: int = 1) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)
        self.softmax = Softmax()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for self-attention on sequence tensor (SeqLen, DModel)."""
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scale = 1.0 / math.sqrt(self.d_model)
        scores = (Q @ K.T) * scale
        weights = self.softmax(scores)
        context = weights @ V
        return self.out_proj(context)


class Dropout(Module):
    """
    Applies Inverted Dropout to input tensor:
    Zeroes elements with probability p during training and scales remaining activations by 1 / (1 - p).
    Acts as identity in eval() mode.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        import random
        scale = 1.0 / (1.0 - self.p)
        mask = [1.0 if random.random() >= self.p else 0.0 for _ in range(len(x._flat))]
        out_flat = [v * m * scale for v, m in zip(x._flat, mask)]
        out = Tensor(_unflatten(out_flat, x.shape), requires_grad=x.requires_grad, _children=(x,), _op="Dropout")

        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = Tensor.zeros(*x.shape)
                for i in range(len(x._flat)):
                    x.grad._flat[i] += out.grad._flat[i] * mask[i] * scale

        out._backward = _backward
        return out


class BCEWithLogitsLoss(Module):
    """
    Binary Cross Entropy with Logits Loss.
    Combines Sigmoid activation and BCELoss using the numerically stable log-sum-exp formulation:
    loss = max(x, 0) - x * y + log(1 + exp(-|x|))
    Gradient:
    d(loss) / dx = (sigmoid(x) - y) / N
    """

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred shape {pred.shape} vs target shape {target.shape}")

        n = len(pred._flat)
        loss_sum = 0.0
        for x, y in zip(pred._flat, target._flat):
            m = max(x, 0.0)
            loss_sum += m - x * y + math.log(1.0 + math.exp(-abs(x)))

        mean_loss = loss_sum / n
        out = Tensor(mean_loss, requires_grad=pred.requires_grad, _children=(pred,), _op="BCEWithLogits")

        def _backward():
            if pred.requires_grad:
                if pred.grad is None:
                    pred.grad = Tensor.zeros(*pred.shape)
                for i in range(n):
                    x = pred._flat[i]
                    y = target._flat[i]
                    sig = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
                    pred.grad._flat[i] += out.grad._flat[0] * (sig - y) / n

        out._backward = _backward
        return out
