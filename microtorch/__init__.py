"""
MicroTorch: A lightweight, clean Tensor Autograd engine and Neural Network framework from scratch.
"""

__version__ = "1.0.0"

from .tensor import Tensor
from .nn import (
    Module,
    Parameter,
    Linear,
    Sequential,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LayerNorm,
    MultiHeadAttention,
    Dropout,
    MSELoss,
    CrossEntropyLoss,
    BCEWithLogitsLoss,
    L1Loss,
    SmoothL1Loss,
    CosineSimilarity,
    CosineEmbeddingLoss,
    TripletMarginLoss,
    LabelSmoothingCrossEntropyLoss,
)
from .optim import SGD, Adam, AdamW, Adagrad, GradientAccumulator, clip_grad_norm_, clip_grad_value_
from .regularization import compute_gradient_penalty
from .lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
from .viz import to_mermaid, to_ascii_tree

__all__ = [
    "Tensor",
    "Module",
    "Parameter",
    "Linear",
    "Sequential",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "LayerNorm",
    "MultiHeadAttention",
    "Dropout",
    "MSELoss",
    "CrossEntropyLoss",
    "BCEWithLogitsLoss",
    "L1Loss",
    "SmoothL1Loss",
    "CosineSimilarity",
    "CosineEmbeddingLoss",
    "TripletMarginLoss",
    "LabelSmoothingCrossEntropyLoss",
    "SGD",
    "Adam",
    "AdamW",
    "Adagrad",
    "GradientAccumulator",
    "clip_grad_norm_",
    "clip_grad_value_",
    "compute_gradient_penalty",
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "to_mermaid",
    "to_ascii_tree",
]
