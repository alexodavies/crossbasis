"""
Distributed Lag Non-linear Models (DLNMs) for Python.

Based on Gasparrini, Armstrong & Kenward (2010), Statistics in Medicine, 29:2224–2234.
"""

from .basis import ns_basis, bs_basis, poly_basis, linear_basis
from .crossbasis import CrossBasis
from .crosspred import CrossPred, PredictionResult
from .utils import logknots, equalknots

__all__ = [
    "CrossBasis",
    "CrossPred",
    "PredictionResult",
    "ns_basis",
    "bs_basis",
    "poly_basis",
    "linear_basis",
    "logknots",
    "equalknots",
]
