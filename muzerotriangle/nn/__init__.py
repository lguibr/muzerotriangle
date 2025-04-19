"""
Neural Network module for the MuZeroTriangle agent.
Contains the MuZero model definition (h, g, f) and a wrapper interface.
"""

from .model import MuZeroNet  # Changed from AlphaTriangleNet
from .network import NeuralNetwork

__all__ = [
    "MuZeroNet",  # Changed from AlphaTriangleNet
    "NeuralNetwork",
]
