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


class L1Loss(Module):
    """
    Mean Absolute Error (L1) loss:
    loss = (1/N) * sum(|pred - target|) (if reduction == 'mean')
    loss = sum(|pred - target|) (if reduction == 'sum')
    Gradient:
    d(loss)/dx = sign(pred - target) / N (or sign(pred - target) for sum)
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Invalid reduction mode: {reduction}. Supported: 'mean', 'sum'")
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred shape {pred.shape} vs target shape {target.shape}")

        n = len(pred._flat)
        diffs = [p - t for p, t in zip(pred._flat, target._flat)]
        abs_diffs = [abs(d) for d in diffs]
        total_loss = sum(abs_diffs)
        loss_val = (total_loss / n) if self.reduction == "mean" else total_loss

        out = Tensor(loss_val, requires_grad=pred.requires_grad, _children=(pred,), _op="L1Loss")

        def _backward():
            if pred.requires_grad:
                if pred.grad is None:
                    pred.grad = Tensor.zeros(*pred.shape)
                scale = (1.0 / n) if self.reduction == "mean" else 1.0
                out_g = out.grad._flat[0] if out.grad is not None else 1.0
                for i in range(n):
                    d = diffs[i]
                    sgn = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
                    pred.grad._flat[i] += out_g * sgn * scale
        out._backward = _backward
        return out


class SmoothL1Loss(Module):
    """
    Smooth L1 (Huber) Loss with threshold beta:
    If |x - y| < beta: 0.5 * (x - y)^2 / beta
    Else: |x - y| - 0.5 * beta
    Gradient:
    If |x - y| < beta: (x - y) / (beta * N)
    Else: sign(x - y) / N
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        if beta <= 0.0:
            raise ValueError(f"beta must be strictly positive, got {beta}")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Invalid reduction mode: {reduction}. Supported: 'mean', 'sum'")
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred shape {pred.shape} vs target shape {target.shape}")

        n = len(pred._flat)
        diffs = [p - t for p, t in zip(pred._flat, target._flat)]
        losses = []
        for d in diffs:
            ad = abs(d)
            if ad < self.beta:
                losses.append(0.5 * (d * d) / self.beta)
            else:
                losses.append(ad - 0.5 * self.beta)

        total_loss = sum(losses)
        loss_val = (total_loss / n) if self.reduction == "mean" else total_loss
        out = Tensor(loss_val, requires_grad=pred.requires_grad, _children=(pred,), _op="SmoothL1Loss")

        def _backward():
            if pred.requires_grad:
                if pred.grad is None:
                    pred.grad = Tensor.zeros(*pred.shape)
                scale = (1.0 / n) if self.reduction == "mean" else 1.0
                out_g = out.grad._flat[0] if out.grad is not None else 1.0
                for i in range(n):
                    d = diffs[i]
                    if abs(d) < self.beta:
                        grad_d = d / self.beta
                    else:
                        grad_d = 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)
                    pred.grad._flat[i] += out_g * grad_d * scale
        out._backward = _backward
        return out


class CosineSimilarity(Module):
    """
    Computes cosine similarity between tensors along the flattened dimension:
    sim(x1, x2) = (x1 . x2) / (max(||x1||_2, eps) * max(||x2||_2, eps))
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        if x1.shape != x2.shape:
            raise ValueError(f"Shape mismatch: x1 shape {x1.shape} vs x2 shape {x2.shape}")

        dot = sum(a * b for a, b in zip(x1._flat, x2._flat))
        norm1 = math.sqrt(sum(a * a for a in x1._flat))
        norm2 = math.sqrt(sum(b * b for b in x2._flat))
        denom = max(norm1 * norm2, self.eps)
        sim = dot / denom

        out = Tensor(sim, requires_grad=(x1.requires_grad or x2.requires_grad), _children=(x1, x2), _op="CosineSimilarity")

        def _backward():
            out_g = out.grad._flat[0] if out.grad is not None else 1.0
            if x1.requires_grad:
                if x1.grad is None:
                    x1.grad = Tensor.zeros(*x1.shape)
                for i in range(len(x1._flat)):
                    term1 = x2._flat[i] / denom
                    term2 = (dot * x1._flat[i]) / (max(norm1 ** 3 * norm2, self.eps))
                    x1.grad._flat[i] += out_g * (term1 - term2)

            if x2.requires_grad:
                if x2.grad is None:
                    x2.grad = Tensor.zeros(*x2.shape)
                for i in range(len(x2._flat)):
                    term1 = x1._flat[i] / denom
                    term2 = (dot * x2._flat[i]) / (max(norm2 ** 3 * norm1, self.eps))
                    x2.grad._flat[i] += out_g * (term1 - term2)

        out._backward = _backward
        return out


class CosineEmbeddingLoss(Module):
    """
    Cosine Embedding Loss:
    Measures loss given input tensors x1, x2 and label Tensor y (containing 1 or -1).
    loss(x1, x2, y) = 1 - cos(x1, x2) if y == 1
                    = max(0, cos(x1, x2) - margin) if y == -1
    """

    def __init__(self, margin: float = 0.0, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.eps = eps
        self._cos = CosineSimilarity(eps=eps)

    def forward(self, x1: Tensor, x2: Tensor, target: Tensor) -> Tensor:
        sim = self._cos(x1, x2)
        y = target._flat[0]
        if y not in (1.0, -1.0, 1, -1):
            raise ValueError(f"Target must be 1 or -1, got {y}")

        cos_val = sim._flat[0]
        if y == 1:
            loss_val = 1.0 - cos_val
        else:
            loss_val = max(0.0, cos_val - self.margin)

        out = Tensor(loss_val, requires_grad=sim.requires_grad, _children=(sim,), _op="CosineEmbeddingLoss")

        def _backward():
            out_g = out.grad._flat[0] if out.grad is not None else 1.0
            if sim.requires_grad:
                if sim.grad is None:
                    sim.grad = Tensor.zeros(*sim.shape)
                if y == 1:
                    sim.grad._flat[0] += -1.0 * out_g
                else:
                    if cos_val > self.margin:
                        sim.grad._flat[0] += 1.0 * out_g

        out._backward = _backward
        return out


class TripletMarginLoss(Module):
    """
    Triplet Margin Loss for metric learning:
    L(a, p, n) = max(0, d(a, p) - d(a, n) + margin)
    where d(x, y) is Euclidean distance.
    """

    def __init__(self, margin: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        if anchor.shape != positive.shape or anchor.shape != negative.shape:
            raise ValueError(f"Shape mismatch: anchor {anchor.shape}, pos {positive.shape}, neg {negative.shape}")

        d_ap_sq = sum((a - p) ** 2 for a, p in zip(anchor._flat, positive._flat))
        d_an_sq = sum((a - n) ** 2 for a, n in zip(anchor._flat, negative._flat))
        d_ap = math.sqrt(d_ap_sq + self.eps)
        d_an = math.sqrt(d_an_sq + self.eps)

        val = d_ap - d_an + self.margin
        loss_val = max(0.0, val)

        out = Tensor(
            loss_val,
            requires_grad=(anchor.requires_grad or positive.requires_grad or negative.requires_grad),
            _children=(anchor, positive, negative),
            _op="TripletMarginLoss"
        )

        def _backward():
            out_g = out.grad._flat[0] if out.grad is not None else 1.0
            if loss_val > 0.0:
                if anchor.requires_grad:
                    if anchor.grad is None:
                        anchor.grad = Tensor.zeros(*anchor.shape)
                    for i in range(len(anchor._flat)):
                        g = (anchor._flat[i] - positive._flat[i]) / d_ap - (anchor._flat[i] - negative._flat[i]) / d_an
                        anchor.grad._flat[i] += out_g * g

                if positive.requires_grad:
                    if positive.grad is None:
                        positive.grad = Tensor.zeros(*positive.shape)
                    for i in range(len(positive._flat)):
                        g = -(anchor._flat[i] - positive._flat[i]) / d_ap
                        positive.grad._flat[i] += out_g * g

                if negative.requires_grad:
                    if negative.grad is None:
                        negative.grad = Tensor.zeros(*negative.shape)
                    for i in range(len(negative._flat)):
                        g = (anchor._flat[i] - negative._flat[i]) / d_an
                        negative.grad._flat[i] += out_g * g

        out._backward = _backward
        return out


class LabelSmoothingCrossEntropyLoss(Module):
    """
    Cross Entropy Loss with Label Smoothing regularization:
    q(k) = (1 - smoothing) * I(k == target) + smoothing / K
    L = - sum_k q(k) * log(softmax(logits)_k)
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        # logits shape: (N, K) or (K,)
        if len(logits.shape) == 1:
            batch_logits = [logits._flat]
            targets = [int(target._flat[0])]
        else:
            n_batch = logits.shape[0]
            k_classes = logits.shape[1]
            batch_logits = [logits._flat[i * k_classes : (i + 1) * k_classes] for i in range(n_batch)]
            targets = [int(t) for t in target._flat]

        n = len(batch_logits)
        k = len(batch_logits[0])
        total_loss = 0.0
        all_probs = []

        for row, y in zip(batch_logits, targets):
            max_val = max(row)
            exp_row = [math.exp(v - max_val) for v in row]
            sum_exp = sum(exp_row)
            probs = [e / sum_exp for e in exp_row]
            all_probs.append(probs)

            # q(i) = (1 - alpha) * (1 if i == y else 0) + alpha / k
            log_probs = [math.log(max(p, 1e-12)) for p in probs]
            loss_i = 0.0
            for i in range(k):
                q_i = ((1.0 - self.smoothing) if i == y else 0.0) + (self.smoothing / k)
                loss_i -= q_i * log_probs[i]
            total_loss += loss_i

        loss_val = (total_loss / n) if self.reduction == "mean" else total_loss
        out = Tensor(loss_val, requires_grad=logits.requires_grad, _children=(logits,), _op="LabelSmoothingCrossEntropyLoss")

        def _backward():
            out_g = out.grad._flat[0] if out.grad is not None else 1.0
            scale = (out_g / n) if self.reduction == "mean" else out_g
            if logits.requires_grad:
                if logits.grad is None:
                    logits.grad = Tensor.zeros(*logits.shape)
                for b_idx, (probs, y) in enumerate(zip(all_probs, targets)):
                    for i in range(k):
                        q_i = ((1.0 - self.smoothing) if i == y else 0.0) + (self.smoothing / k)
                        grad_z = probs[i] - q_i
                        logits.grad._flat[b_idx * k + i] += grad_z * scale

        out._backward = _backward
        return out
