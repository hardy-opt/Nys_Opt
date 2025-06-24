"""
Utility functions for the Nys-Newton optimizer implementation.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

def compute_gradient_norm(parameters) -> float:
    """
    Compute the L2 norm of gradients across all parameters.
    
    Args:
        parameters: Iterator of model parameters
        
    Returns:
        L2 norm of gradients
    """
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def safe_matrix_sqrt(matrix: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Compute matrix square root with numerical stability.
    
    Args:
        matrix: Input positive definite matrix
        epsilon: Small value for numerical stability
        
    Returns:
        Matrix square root
    """
    eigenvals, eigenvecs = torch.symeig(matrix, eigenvectors=True)
    eigenvals = torch.clamp(eigenvals, min=epsilon)
    return eigenvecs @ torch.diag(torch.sqrt(eigenvals)) @ eigenvecs.t()

def check_convergence(loss_history: List[float], 
                     tolerance: float = 1e-6, 
                     patience: int = 10) -> bool:
    """
    Check if optimization has converged based on loss history.
    
    Args:
        loss_history: List of recent loss values
        tolerance: Convergence tolerance
        patience: Number of steps to wait for improvement
        
    Returns:
        True if converged, False otherwise
    """
    if len(loss_history) < patience + 1:
        return False
    
    recent_losses = loss_history[-patience:]
    return all(abs(recent_losses[i] - recent_losses[i-1]) < tolerance 
              for i in range(1, len(recent_losses)))

def log_optimizer_stats(optimizer, step: int, loss: float, 
                       grad_norm: float, logger: logging.Logger) -> None:
    """
    Log optimizer statistics for monitoring.
    
    Args:
        optimizer: The optimizer instance
        step: Current optimization step
        loss: Current loss value
        grad_norm: Gradient norm
        logger: Logger instance
    """
    logger.info(f"Step {step}: Loss={loss:.6f}, GradNorm={grad_norm:.6f}")
    
    # Log optimizer-specific stats
    if hasattr(optimizer, 'nystrom'):
        rank = optimizer.nystrom.rank
        logger.info(f"Nystrom rank: {rank}")

def plot_convergence(loss_history: Dict[str, List[float]], 
                    save_path: Optional[str] = None) -> None:
    """
    Plot convergence curves for different optimizers.
    
    Args:
        loss_history: Dictionary mapping optimizer names to loss histories
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for name, losses in loss_history.items():
        plt.plot(losses, label=name, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Optimizer Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_synthetic_problem(n_features: int = 100, 
                           n_samples: int = 1000,
                           condition_number: float = 100.0,
                           noise_std: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a synthetic optimization problem for testing.
    
    Args:
        n_features: Number of features
        n_samples: Number of samples
        condition_number: Condition number of the problem
        noise_std: Standard deviation of noise
        
    Returns:
        Tuple of (features, targets)
    """
    # Create ill-conditioned problem
    U = torch.randn(n_features, n_features)
    Q = torch.qr(U)[0]  # Orthogonal matrix
    
    # Create eigenvalues with specified condition number
    eigenvals = torch.logspace(0, np.log10(condition_number), n_features)
    A = Q @ torch.diag(eigenvals) @ Q.t()
    
    # Generate data
    X = torch.randn(n_samples, n_features)
    true_weights = torch.randn(n_features)
    y = X @ true_weights + noise_std * torch.randn(n_samples)
    
    return X, y

class ExperimentLogger:
    """
    Logger for tracking experiment results and metrics.
    """
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        self.metrics = {}
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.log_dir}/experiment.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("NysNewton")
    
    def log_metric(self, name: str, value: float, step: int):
        """Log a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((step, value))
        self.logger.info(f"{name}: {value:.6f} at step {step}")
    
    def save_metrics(self, filepath: str):
        """Save all metrics to file."""
        torch.save(self.metrics, filepath)
        self.logger.info(f"Metrics saved to {filepath}")