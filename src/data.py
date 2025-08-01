"""
Data ingestion and cleaning module for portfolio optimization.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Optional, List
import os


def download_stock_data(tickers: List[str], 
                       start_date: str = "2018-01-01", 
                       end_date: str = "2025-01-01",
                       save_path: str = "data/stock_data.csv") -> pd.DataFrame:
    """
    Download stock data using yfinance and save to CSV.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date for data download
        end_date: End date for data download
        save_path: Path to save the CSV file
        
    Returns:
        DataFrame with adjusted close prices
    """
    print(f"Downloading data for {len(tickers)} stocks...")
    
    # Download data
    raw_data = yf.download(tickers, start=start_date, end=end_date)
    if isinstance(raw_data, pd.DataFrame):
        # Single ticker case
        raw_data = raw_data["Adj Close"]
    else:
        # Multiple tickers case
        raw_data = raw_data["Adj Close"]
    
    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    raw_data.to_csv(save_path)
    print(f"Data saved to {save_path}")
    
    return raw_data


def load_returns(file_path: str = "data/returns.csv") -> pd.DataFrame:
    """
    Load returns data from CSV file.
    
    Args:
        file_path: Path to the CSV file containing returns
        
    Returns:
        DataFrame with returns data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Returns file not found: {file_path}")
    
    returns = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return returns


def compute_returns(prices: pd.DataFrame, method: str = "pct_change") -> pd.DataFrame:
    """
    Compute returns from price data.
    
    Args:
        prices: DataFrame with price data
        method: Method to compute returns ("pct_change" or "log")
        
    Returns:
        DataFrame with returns
    """
    if len(prices) < 2:
        return pd.DataFrame()  # Return empty DataFrame for insufficient data
    
    if method == "pct_change":
        returns = prices.pct_change().dropna()
    elif method == "log":
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("Method must be 'pct_change' or 'log'")
    
    return returns


def compute_statistics(returns: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute covariance matrix and mean returns.
    
    Args:
        returns: DataFrame with returns data
        
    Returns:
        Tuple of (covariance_matrix, mean_returns)
    """
    cov_matrix = returns.cov().values
    mean_returns = returns.mean().values
    
    return cov_matrix, mean_returns


def generate_sample_data(n_assets: int = 50, 
                        n_periods: int = 252, 
                        save_path: str = "data/sample_returns.csv") -> pd.DataFrame:
    """
    Generate synthetic returns data for testing.
    
    Args:
        n_assets: Number of assets
        n_periods: Number of time periods
        save_path: Path to save the sample data
        
    Returns:
        DataFrame with synthetic returns
    """
    np.random.seed(42)
    
    # Generate random covariance matrix
    A = np.random.randn(n_assets, n_assets)
    cov_matrix = A @ A.T + np.eye(n_assets) * 0.1
    
    # Generate random mean returns
    mean_returns = np.random.normal(0.08, 0.15, n_assets)
    
    # Generate returns from multivariate normal
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)
    
    # Convert to DataFrame
    asset_names = [f"Asset_{i}" for i in range(n_assets)]
    returns_df = pd.DataFrame(returns, columns=asset_names)
    
    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    returns_df.to_csv(save_path)
    
    return returns_df


def validate_data(returns: pd.DataFrame) -> bool:
    """
    Validate returns data for portfolio optimization.
    
    Args:
        returns: DataFrame with returns data
        
    Returns:
        True if data is valid, raises ValueError otherwise
    """
    # Check for missing values
    if returns.isnull().any().any():
        raise ValueError("Returns data contains missing values")
    
    # Check for infinite values
    if np.isinf(returns.values).any():
        raise ValueError("Returns data contains infinite values")
    
    # Check for zero variance assets (exactly zero, not just small)
    zero_var_assets = returns.var() == 0
    if zero_var_assets.any():
        raise ValueError(f"Assets with zero variance: {zero_var_assets[zero_var_assets].index.tolist()}")
    
    # Check for perfect correlation
    corr_matrix = returns.corr()
    np.fill_diagonal(corr_matrix.values, 0)
    if (corr_matrix == 1).any().any():
        raise ValueError("Returns data contains perfectly correlated assets")
    
    return True


def prepare_portfolio_data(tickers: Optional[List[str]] = None,
                          use_sample_data: bool = True,
                          file_path: str = "data/returns.csv") -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for portfolio optimization.
    
    Args:
        tickers: List of stock tickers (if None, uses sample data)
        use_sample_data: Whether to use sample data
        file_path: Path to returns CSV file
        
    Returns:
        Tuple of (covariance_matrix, mean_returns)
    """
    if use_sample_data or tickers is None:
        print("Using sample data for portfolio optimization...")
        returns = generate_sample_data()
    else:
        print(f"Downloading data for tickers: {tickers}")
        prices = download_stock_data(tickers)
        returns = compute_returns(prices)
        returns.to_csv(file_path)
    
    # Validate data
    validate_data(returns)
    
    # Compute statistics
    cov_matrix, mean_returns = compute_statistics(returns)
    
    print(f"Data prepared: {len(returns.columns)} assets, {len(returns)} periods")
    print(f"Mean return range: [{mean_returns.min():.4f}, {mean_returns.max():.4f}]")
    print(f"Volatility range: [{np.sqrt(np.diag(cov_matrix)).min():.4f}, {np.sqrt(np.diag(cov_matrix)).max():.4f}]")
    
    return cov_matrix, mean_returns 