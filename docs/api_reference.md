# API Reference

## NysNewtonOptimizer

The main optimizer class implementing the Nys-Newton algorithm.

### Constructor

```python
NysNewtonOptimizer(params, lr=0.01, rank=50, damping=1e-6, update_freq=10)

Parameters:

params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
lr (float, optional): Learning rate (default: 0.01)
rank (int, optional): Rank for Nyström approximation (default: 50)
damping (float, optional): Damping factor for numerical stability (default: 1e-6)
update_freq (int, optional): Frequency of curvature matrix updates (default: 10)

Methods
step(closure=None)
Performs a single optimization step.
Parameters:

closure (callable, optional): A closure that reevaluates the model and returns the loss

Returns:

Loss value if closure is provided

zero_grad()
Clears the gradients of all optimized parameters.
Example Usage
pythonimport torch
import torch.nn as nn
from nys_newton import NysNewtonOptimizer

# Define model
model = nn.Linear(10, 1)

# Initialize optimizer
optimizer = NysNewtonOptimizer(model.parameters(), lr=0.01, rank=50)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward pass
    output = model(input_data)
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Optimization step
    optimizer.step()
NystromApproximator
Low-level class for computing Nyström approximations.
Constructor
pythonNystromApproximator(rank=50, epsilon=1e-6)
Parameters:

rank (int): Rank of the approximation
epsilon (float): Regularization parameter for numerical stability

Methods
fit(matrix)
Compute the Nyström approximation of a matrix.
Parameters:

matrix (torch.Tensor): Input matrix to approximate

Returns:

torch.Tensor: Low-rank approximation of the input matrix

solve(b)
Solve a linear system using the computed approximation.
Parameters:

b (torch.Tensor): Right-hand side vector

Returns:

torch.Tensor: Solution vector

Utility Functions
compute_gradient_norm(parameters)
Compute the L2 norm of gradients across all parameters.
Parameters:

parameters (iterable): Iterator of model parameters

Returns:

float: L2 norm of gradients

safe_matrix_sqrt(matrix, epsilon=1e-10)
Compute matrix square root with numerical stability.
Parameters:

matrix (torch.Tensor): Input positive definite matrix
epsilon (float): Small value for numerical stability

Returns:

torch.Tensor: Matrix square root

check_convergence(loss_history, tolerance=1e-6, patience=10)
Check if optimization has converged based on loss history.
Parameters:

loss_history (list): List of recent loss values
tolerance (float): Convergence tolerance
patience (int): Number of steps to wait for improvement

Returns:

bool: True if converged, False otherwise

create_synthetic_problem(n_features=100, n_samples=1000, condition_number=100.0, noise_std=0.1)
Create a synthetic optimization problem for testing.
Parameters:

n_features (int): Number of features
n_samples (int): Number of samples
condition_number (float): Condition number of the problem
noise_std (float): Standard deviation of noise

Returns:

tuple: (features, targets) as torch tensors# Incomplete
