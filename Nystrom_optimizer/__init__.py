"""
Nys-Newton: Nyström-Approximated Curvature for Stochastic Optimization

A PyTorch implementation of the Nys-Newton optimizer that efficiently 
approximates second-order curvature information using the Nyström method.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .optimizer import NysNewtonOptimizer
from .nystrom_approximation import NystromApproximator
from .utils import (
    compute_gradient_norm,
    safe_matrix_sqrt,
    check_convergence,
    log_optimizer_stats
)

__all__ = [
    "NysNewtonOptimizer",
    "NystromApproximator",
    "compute_gradient_norm",
    "safe_matrix_sqrt",
    "check_convergence",
    "log_optimizer_stats"
]