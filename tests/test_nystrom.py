"""
Unit tests for Nyström approximation implementation.
"""

import unittest
import torch
import numpy as np
from nys_newton.nystrom_approximation import NystromApproximator

class TestNystromApproximator(unittest.TestCase):
    """Test suite for Nyström approximation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_dim = 100
        self.rank = 20
        self.nystrom = NystromApproximator(rank=self.rank)
        
        # Create test matrix (positive definite)
        A = torch.randn(self.n_dim, self.n_dim)
        self.test_matrix = A @ A.t() + torch.eye(self.n_dim)
    
    def test_initialization(self):
        """Test Nyström approximator initialization."""
        self.assertEqual(self.nystrom.rank, self.rank)
        self.assertIsNone(self.nystrom.W)
        self.assertIsNone(self.nystrom.U)
        self.assertIsNone(self.nystrom.S)
    
    def test_fit_dimensions(self):
        """Test that fit produces correct dimensions."""
        approximation = self.nystrom.fit(self.test_matrix)
        
        self.assertEqual(approximation.shape, self.test_matrix.shape)
        self.assertIsNotNone(self.nystrom.W)
        self.assertIsNotNone(self.nystrom.U)
        self.assertIsNotNone(self.nystrom.S)
    
    def test_approximation_quality(self):
        """Test quality of Nyström approximation."""
        approximation = self.nystrom.fit(self.test_matrix)
        
        # Compute approximation error
        error = torch.norm(self.test_matrix - approximation, 'fro')
        relative_error = error / torch.norm(self.test_matrix, 'fro')
        
        # For a well-conditioned matrix, relative error should be reasonable
        self.assertLess(relative_error, 0.5)  # Somewhat loose bound
    
    def test_solve_system(self):
        """Test linear system solving."""
        self.nystrom.fit(self.test_matrix)
        
        # Create test right-hand side
        b = torch.randn(self.n_dim)
        
        # Solve using Nyström approximation
        x_approx = self.nystrom.solve(b)
        
        # Solve exactly for comparison
        x_exact = torch.solve(b.unsqueeze(1), self.test_matrix)[0].squeeze()
        
        # Check solution quality
        relative_error = torch.norm(x_approx - x_exact) / torch.norm(x_exact)
        self.assertLess(relative_error, 0.1)  # 10% relative error threshold
    
    def test_rank_deficient_case(self):
        """Test behavior with rank-deficient matrices."""
        # Create rank-deficient matrix
        rank_def = 10
        U = torch.randn(self.n_dim, rank_def)
        rank_def_matrix = U @ U.t() + 1e-6 * torch.eye(self.n_dim)
        
        nystrom_small = NystromApproximator(rank=min(rank_def + 5, self.n_dim))
        approximation = nystrom_small.fit(rank_def_matrix)
        
        # Should not crash and produce reasonable approximation
        self.assertEqual(approximation.shape, rank_def_matrix.shape)

if __name__ == '__main__':
    unittest.main()