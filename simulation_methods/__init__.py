"""
Simulation methods for multi-asset return modeling.

Exports:
- CholeskyMethod: Correlated return simulation using Cholesky decomposition
- ReturnSimulation: Abstract base class for simulation interfaces
"""

from .base_simulator import ReturnSimulation
from .cholesky import CholeskyMethod

__all__ = ["CholeskyMethod", "ReturnSimulation"]
