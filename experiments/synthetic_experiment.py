"""
Synthetic optimization problem experiment for testing Nys-Newton.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import argparse

from nys_newton import NysNewtonOptimizer
from nys_newton.utils import create_synthetic_problem, plot_convergence

class QuadraticProblem:
    """
    Quadratic optimization problem: min_x (1/2) x^T A x - b^T x
    where A is positive definite with specified condition number.
    """
    
    def __init__(self, n_dim: int = 100, condition_number: float = 100.0):
        self.n_dim = n_dim
        self.condition_number = condition_number
        self.setup_problem()
    
    def setup_problem(self):
        """Setup the quadratic problem."""
        # Create well-conditioned to ill-conditioned matrix
        eigenvals = torch.logspace(0, np.log10(self.condition_number), self.n_dim)
        Q = torch.qr(torch.randn(self.n_dim, self.n_dim))[0]
        self.A = Q @ torch.diag(eigenvals) @ Q.t()
        self.b = torch.randn(self.n_dim)
        
        # Analytical solution
        self.x_optimal = torch.solve(self.b.unsqueeze(1), self.A)[0].squeeze()
        self.f_optimal = -0.5 * self.b.dot(self.x_optimal)
    
    def objective(self, x: torch.Tensor) -> torch.Tensor:
        """Compute objective value."""
        return 0.5 * x.dot(self.A @ x) - self.b.dot(x)
    
    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient."""
        return self.A @ x - self.b
    
    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Hessian (constant for quadratic)."""
        return self.A

class SyntheticExperiment:
    """Runner for synthetic optimization experiments."""
    
    def __init__(self, n_dim: int = 100, condition_number: float = 100.0):
        self.problem = QuadraticProblem(n_dim, condition_number)
        self.optimizers = {}
        self.results = {}
    
    def setup_optimizers(self, lr: float = 0.01):
        """Setup different optimizers to compare."""
        # Starting point
        x0 = torch.randn(self.problem.n_dim, requires_grad=True)
        
        self.optimizers = {
            'NysNewton': {
                'param': x0.clone().detach().requires_grad_(True),
                'optimizer': None  # Will be created with parameter
            },
            'Adam': {
                'param': x0.clone().detach().requires_grad_(True),
                'optimizer': None
            },
            'SGD': {
                'param': x0.clone().detach().requires_grad_(True),
                'optimizer': None
            }
        }
        
        # Create optimizers
        for name, opt_dict in self.optimizers.items():
            param = opt_dict['param']
            if name == 'NysNewton':
                opt_dict['optimizer'] = NysNewtonOptimizer([param], lr=lr, rank=20)
            elif name == 'Adam':
                opt_dict['optimizer'] = torch.optim.Adam([param], lr=lr)
            elif name == 'SGD':
                opt_dict['optimizer'] = torch.optim.SGD([param], lr=lr)
    
    def run_optimization(self, max_iterations: int = 1000, tolerance: float = 1e-8):
        """Run optimization with all optimizers."""
        self.results = {name: {'loss': [], 'error': []} for name in self.optimizers.keys()}
        
        for name, opt_dict in self.optimizers.items():
            param = opt_dict['param']
            optimizer = opt_dict['optimizer']
            
            print(f"Running {name} optimizer...")
            
            for i in range(max_iterations):
                optimizer.zero_grad()
                
                # Compute loss
                loss = self.problem.objective(param)
                loss.backward()
                
                # Optimization step
                optimizer.step()
                
                # Record metrics
                loss_val = loss.item()
                error = torch.norm(param - self.problem.x_optimal).item()
                
                self.results[name]['loss'].append(loss_val)
                self.results[name]['error'].append(error)
                
                # Check convergence
                if error < tolerance:
                    print(f"{name} converged at iteration {i}")
                    break
                
                if i % 100 == 0:
                    print(f"{name} - Iteration {i}: Loss = {loss_val:.6f}, Error = {error:.6f}")
    
    def plot_results(self, save_path: str = None):
        """Plot convergence results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss convergence
        for name, results in self.results.items():
            ax1.plot(results['loss'], label=name, linewidth=2)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Objective Value Convergence')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot error convergence
        for name, results in self.results.items():
            ax2.plot(results['error'], label=name, linewidth=2)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Distance to Optimum')
        ax2.set_title('Distance to Optimum')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """Print experiment summary."""
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Problem dimension: {self.problem.n_dim}")
        print(f"Condition number: {self.problem.condition_number:.2f}")
        print(f"Optimal value: {self.problem.f_optimal:.6f}")
        
        for name, results in self.results.items():
            final_loss = results['loss'][-1]
            final_error = results['error'][-1]
            iterations = len(results['loss'])
            print(f"\n{name}:")
            print(f"  Final loss: {final_loss:.6f}")
            print(f"  Final error: {final_error:.6f}")
            print(f"  Iterations: {iterations}")

def main():
    parser = argparse.ArgumentParser(description='Synthetic Optimization Experiment')
    parser.add_argument('--dim', type=int, default=100, help='Problem dimension')
    parser.add_argument('--condition', type=float, default=100.0, help='Condition number')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations')
    parser.add_argument('--save_plot', type=str, help='Path to save convergence plot')
    
    args = parser.parse_args()
    
    # Run experiment
    experiment = SyntheticExperiment(args.dim, args.condition)
    experiment.setup_optimizers(args.lr)
    experiment.run_optimization(args.max_iter)
    
    # Plot and summarize results
    experiment.plot_results(args.save_plot)
    experiment.print_summary()

if __name__ == '__main__':
    main()