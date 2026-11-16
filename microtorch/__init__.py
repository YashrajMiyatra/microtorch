"""
MicroTorch: A lightweight, clean Tensor Autograd engine and Neural Network framework from scratch.
"""

__version__ = "1.0.0"

from .tensor import Tensor
from .nn import Module, Parameter, Linear, Sequential, ReLU, Sigmoid, Tanh, MSELoss, CrossEntropyLoss
from .optim import SGD, Adam, AdamW
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
    "MSELoss",
    "CrossEntropyLoss",
    "SGD",
    "Adam",
    "AdamW",
    "to_mermaid",
    "to_ascii_tree",
]
