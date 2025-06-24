"""
Unit tests for utility functions.
"""

import unittest
import torch
import numpy as np
from nys_newton.utils import (
    compute_gradient_norm,
    safe_matrix_sqrt,
    check_convergence,
    create_synthetic_problem
)

class TestUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions."""
    
    def test_compute_gradient_norm(self):
        """Test gradient norm computation."""
        # Create parameters with known gradients
        param1 = torch.tensor([1.0, 2.0], requires_grad=True)
        param2 = torch.tensor([3.0, 4.0], requires_grad=True)
        
        # Set gradients manually
        param1.grad = torch.tensor([1.0, 0.0])
        param2.grad = torch.tensor([0.0, 1.0])
        
        # Expected norm: sqrt(1^2 + 0^2 + 0^2 + 1^2) = sqrt(2)
        expected_norm = np.sqrt(2.0)
        computed_norm = compute_gradient_norm([param1, param2])
        
        self.assertAlmostEqual(computed_norm, expected_norm, places=6)
    
    def test_safe_matrix_sqrt(self):
        """Test safe matrix square root computation."""
        # Create positive definite matrix
        A = torch.randn(5, 5)
        pos_def = A @ A.t() + torch.eye(5)
        
        # Compute square root
        sqrt_A = safe_matrix_sqrt(pos_def)
        
        # Verify: sqrt_A @ sqrt_A ≈ A
        reconstructed = sqrt_A @ sqrt_A
        error = torch.norm(pos_def - reconstructed, 'fro')
        
        self.assertLess(error, 1e-6)
    
    def test_check_convergence(self):
        """Test convergence checking."""
        # Test converged case
        converged_history = [1.0, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01]
        self.assertTrue(check_convergence(converged_history, tolerance=1e-2, patience=5))
        
        # Test non-converged case
        non_converged_history = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        self.assertFalse(check_convergence(non_converged_history, tolerance=1e-3, patience=3))
        
        # Test insufficient history
        short_history = [1.0, 0.5]
        self.assertFalse(check_convergence(short_history, patience=5))
    
    def test_create_synthetic_problem(self):
        """Test synthetic problem generation."""
        n_features, n_samples = 50, 100
        X, y = create_synthetic_problem(n_features, n_samples)
        
        # Check dimensions
        self.assertEqual(X.shape, (n_samples, n_features))
        self.assertEqual(y.shape, (n_samples,))
        
        # Check that problem is not trivial
        self.assertGreater(torch.var(y), 1e-6)

if __name__ == '__main__':
    unittest.main()