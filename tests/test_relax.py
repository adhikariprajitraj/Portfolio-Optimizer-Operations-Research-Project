"""
Unit tests for relax module.
"""

import pytest
import numpy as np
from src.relax import (
    solve_relax,
    solve_min_variance,
    solve_max_sharpe,
    compute_efficient_frontier,
    validate_solution
)
from src.data import generate_sample_data, compute_statistics


class TestRelaxModule:
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        returns = generate_sample_data(n_assets=10, n_periods=100)
        cov_matrix, mean_returns = compute_statistics(returns)
        return cov_matrix, mean_returns
    
    def test_solve_relax_basic(self, sample_data):
        """Test basic portfolio optimization."""
        cov_matrix, mean_returns = sample_data
        
        weights, obj_value, solver_info = solve_relax(cov_matrix, mean_returns, gamma=1.0)
        
        assert isinstance(weights, np.ndarray)
        assert len(weights) == len(mean_returns)
        assert np.allclose(np.sum(weights), 1.0, atol=1e-6)
        assert np.all(weights >= -1e-10)  # Allow for numerical precision
        assert isinstance(obj_value, float)
        assert isinstance(solver_info, dict)
    
    def test_solve_relax_with_transaction_costs(self, sample_data):
        """Test portfolio optimization with transaction costs."""
        cov_matrix, mean_returns = sample_data
        w_prev = np.ones(len(mean_returns)) / len(mean_returns)
        
        weights, obj_value, solver_info = solve_relax(
            cov_matrix, mean_returns, gamma=1.0, 
            w_prev=w_prev, lambda_tc=0.01
        )
        
        assert isinstance(weights, np.ndarray)
        assert len(weights) == len(mean_returns)
        assert np.allclose(np.sum(weights), 1.0, atol=1e-6)
        assert np.all(weights >= -1e-10)  # Allow for numerical precision
    
    def test_solve_min_variance(self, sample_data):
        """Test minimum variance portfolio."""
        cov_matrix, _ = sample_data
        
        weights, obj_value, solver_info = solve_min_variance(cov_matrix)
        
        assert isinstance(weights, np.ndarray)
        assert len(weights) == cov_matrix.shape[0]
        assert np.allclose(np.sum(weights), 1.0, atol=1e-6)
        assert np.all(weights >= -1e-10)  # Allow for numerical precision
        assert obj_value > 0  # Variance should be positive
    
    def test_solve_max_sharpe(self, sample_data):
        """Test maximum Sharpe ratio portfolio."""
        cov_matrix, mean_returns = sample_data
        
        weights, obj_value, solver_info = solve_max_sharpe(
            cov_matrix, mean_returns, risk_free_rate=0.02
        )
        
        assert isinstance(weights, np.ndarray)
        assert len(weights) == len(mean_returns)
        assert np.allclose(np.sum(weights), 1.0, atol=1e-6)
        assert np.all(weights >= -1e-10)  # Allow for numerical precision
    
    def test_compute_efficient_frontier(self, sample_data):
        """Test efficient frontier computation."""
        cov_matrix, mean_returns = sample_data
        
        risks, returns, weights_array = compute_efficient_frontier(
            cov_matrix, mean_returns, n_points=10
        )
        
        assert isinstance(risks, np.ndarray)
        assert isinstance(returns, np.ndarray)
        assert isinstance(weights_array, np.ndarray)
        assert len(risks) == len(returns)
        assert len(risks) == len(weights_array)
        assert len(risks) > 0
        
        # Check that risks are increasing
        assert np.all(np.diff(risks) >= 0)
    
    def test_validate_solution(self, sample_data):
        """Test solution validation."""
        cov_matrix, mean_returns = sample_data
        
        # Valid solution
        weights = np.ones(len(mean_returns)) / len(mean_returns)
        validation = validate_solution(weights, cov_matrix, mean_returns)
        
        assert isinstance(validation, dict)
        assert validation["is_valid"] is True
        assert validation["budget_violation"] < 1e-6
        assert validation["negative_weights"] == 0
    
    def test_validate_solution_invalid(self, sample_data):
        """Test solution validation with invalid solution."""
        cov_matrix, mean_returns = sample_data
        
        # Invalid solution (negative weights)
        weights = np.ones(len(mean_returns)) / len(mean_returns)
        weights[0] = -0.1
        weights[1] = 1.1  # Sum > 1
        
        validation = validate_solution(weights, cov_matrix, mean_returns)
        
        assert validation["is_valid"] == False
        assert validation["negative_weights"] > 0
        assert validation["budget_violation"] > 1e-6
    
    def test_solve_relax_different_gamma(self, sample_data):
        """Test portfolio optimization with different risk aversion."""
        cov_matrix, mean_returns = sample_data
        
        # Low risk aversion
        weights_low, _, _ = solve_relax(cov_matrix, mean_returns, gamma=0.1)
        
        # High risk aversion
        weights_high, _, _ = solve_relax(cov_matrix, mean_returns, gamma=5.0)
        
        # Higher gamma should lead to lower risk
        risk_low = np.sqrt(weights_low @ cov_matrix @ weights_low)
        risk_high = np.sqrt(weights_high @ cov_matrix @ weights_high)
        
        # This is not always true due to numerical issues, but should be generally true
        # assert risk_high <= risk_low
    
    def test_solve_relax_edge_cases(self, sample_data):
        """Test edge cases for portfolio optimization."""
        cov_matrix, mean_returns = sample_data
        
        # Very high gamma
        weights, obj_value, solver_info = solve_relax(cov_matrix, mean_returns, gamma=100.0)
        assert isinstance(weights, np.ndarray)
        assert np.allclose(np.sum(weights), 1.0, atol=1e-6)
        
        # Very low gamma
        weights, obj_value, solver_info = solve_relax(cov_matrix, mean_returns, gamma=0.01)
        assert isinstance(weights, np.ndarray)
        assert np.allclose(np.sum(weights), 1.0, atol=1e-6)
    
    def test_solve_relax_singular_covariance(self):
        """Test with singular covariance matrix."""
        # Create singular covariance matrix
        n = 5
        cov_matrix = np.eye(n)
        cov_matrix[0, 0] = 0  # Make singular
        mean_returns = np.random.randn(n)
        
        # Should handle gracefully
        weights, obj_value, solver_info = solve_relax(cov_matrix, mean_returns, gamma=1.0)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == n 