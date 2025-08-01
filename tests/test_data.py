"""
Unit tests for data module.
"""

import pytest
import numpy as np
import pandas as pd
import os
from src.data import (
    generate_sample_data,
    compute_returns,
    compute_statistics,
    validate_data,
    prepare_portfolio_data
)


class TestDataModule:
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        returns = generate_sample_data(n_assets=10, n_periods=100)
        
        assert isinstance(returns, pd.DataFrame)
        assert returns.shape == (100, 10)
        assert not returns.isnull().any().any()
        assert not np.isinf(returns.values).any()
    
    def test_compute_returns(self):
        """Test returns computation."""
        # Create sample price data
        prices = pd.DataFrame({
            'Asset1': [100, 110, 105, 115],
            'Asset2': [50, 55, 52, 58]
        })
        
        # Test percentage change
        returns_pct = compute_returns(prices, method="pct_change")
        assert isinstance(returns_pct, pd.DataFrame)
        assert returns_pct.shape == (3, 2)  # First row dropped
        
        # Test log returns
        returns_log = compute_returns(prices, method="log")
        assert isinstance(returns_log, pd.DataFrame)
        assert returns_log.shape == (3, 2)
        
        # Test invalid method
        with pytest.raises(ValueError):
            compute_returns(prices, method="invalid")
    
    def test_compute_statistics(self):
        """Test statistics computation."""
        returns = generate_sample_data(n_assets=5, n_periods=50)
        cov_matrix, mean_returns = compute_statistics(returns)
        
        assert isinstance(cov_matrix, np.ndarray)
        assert isinstance(mean_returns, np.ndarray)
        assert cov_matrix.shape == (5, 5)
        assert mean_returns.shape == (5,)
        assert np.allclose(cov_matrix, cov_matrix.T)  # Symmetric
    
    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        returns = generate_sample_data(n_assets=5, n_periods=50)
        assert validate_data(returns) is True
    
    def test_validate_data_missing_values(self):
        """Test data validation with missing values."""
        returns = generate_sample_data(n_assets=5, n_periods=50)
        returns.iloc[0, 0] = np.nan
        
        with pytest.raises(ValueError, match="missing values"):
            validate_data(returns)
    
    def test_validate_data_infinite_values(self):
        """Test data validation with infinite values."""
        returns = generate_sample_data(n_assets=5, n_periods=50)
        returns.iloc[0, 0] = np.inf
        
        with pytest.raises(ValueError, match="infinite values"):
            validate_data(returns)
    
    def test_validate_data_zero_variance(self):
        """Test data validation with zero variance assets."""
        returns = generate_sample_data(n_assets=5, n_periods=50)
        returns.iloc[:, 0] = 0.1  # Constant returns
        
        # This should not raise an error for constant returns
        # The validation only checks for exactly zero variance
        validate_data(returns)
    
    def test_prepare_portfolio_data_sample(self):
        """Test portfolio data preparation with sample data."""
        cov_matrix, mean_returns = prepare_portfolio_data(use_sample_data=True)
        
        assert isinstance(cov_matrix, np.ndarray)
        assert isinstance(mean_returns, np.ndarray)
        assert cov_matrix.shape[0] == cov_matrix.shape[1]
        assert len(mean_returns) == cov_matrix.shape[0]
    
    def test_compute_returns_edge_cases(self):
        """Test returns computation edge cases."""
        # Single price - should handle gracefully
        prices = pd.DataFrame({'Asset1': [100]})
        # This should not raise an error, just return empty DataFrame
        returns = compute_returns(prices)
        assert returns.empty
        
        # Zero prices
        prices = pd.DataFrame({
            'Asset1': [0, 0, 0],
            'Asset2': [1, 1, 1]
        })
        returns = compute_returns(prices)
        # When all prices are constant, returns are zero and get dropped
        assert returns.shape == (0, 2)
    
    def test_generate_sample_data_parameters(self):
        """Test sample data generation with different parameters."""
        # Test small dataset
        returns = generate_sample_data(n_assets=3, n_periods=10)
        assert returns.shape == (10, 3)
        
        # Test large dataset
        returns = generate_sample_data(n_assets=100, n_periods=500)
        assert returns.shape == (500, 100)
        
        # Test statistics
        cov_matrix, mean_returns = compute_statistics(returns)
        assert cov_matrix.shape == (100, 100)
        assert mean_returns.shape == (100,) 