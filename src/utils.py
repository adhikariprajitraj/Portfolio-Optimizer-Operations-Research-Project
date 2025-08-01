"""
Shared utilities for portfolio optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import warnings


def compute_portfolio_statistics(weights: np.ndarray,
                               cov_matrix: np.ndarray,
                               mean_returns: np.ndarray,
                               w_prev: Optional[np.ndarray] = None) -> Dict:
    """
    Compute comprehensive portfolio statistics.
    
    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix
        mean_returns: Mean returns
        w_prev: Previous portfolio weights
        
    Returns:
        Dictionary with portfolio statistics
    """
    # Basic statistics
    portfolio_return = mean_returns @ weights
    portfolio_risk = np.sqrt(weights @ cov_matrix @ weights)
    sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
    
    # Diversification metrics
    n_assets = len(weights)
    effective_n = 1 / np.sum(weights ** 2)  # Effective number of assets
    concentration = np.sum(weights ** 2)  # Herfindahl index
    
    # Risk decomposition
    marginal_contributions = cov_matrix @ weights
    risk_contributions = weights * marginal_contributions
    total_risk = np.sum(risk_contributions)
    
    # Turnover
    turnover = 0
    if w_prev is not None:
        turnover = np.sum(np.abs(weights - w_prev))
    
    # Maximum drawdown approximation (simplified)
    max_weight = np.max(weights)
    min_weight = np.min(weights[weights > 0]) if np.any(weights > 0) else 0
    
    statistics = {
        "portfolio_return": portfolio_return,
        "portfolio_risk": portfolio_risk,
        "sharpe_ratio": sharpe_ratio,
        "effective_n": effective_n,
        "concentration": concentration,
        "max_weight": max_weight,
        "min_weight": min_weight,
        "turnover": turnover,
        "risk_contributions": risk_contributions,
        "marginal_contributions": marginal_contributions
    }
    
    return statistics


def plot_efficient_frontier(risks: np.ndarray,
                           returns: np.ndarray,
                           weights_array: np.ndarray,
                           save_path: Optional[str] = None) -> None:
    """
    Plot efficient frontier.
    
    Args:
        risks: Portfolio risks
        returns: Portfolio returns
        weights_array: Array of portfolio weights
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot efficient frontier
    plt.plot(risks, returns, 'b-', linewidth=2, label='Efficient Frontier')
    
    # Mark minimum variance and maximum Sharpe points
    min_var_idx = np.argmin(risks)
    max_sharpe_idx = np.argmax(returns / risks)
    
    plt.scatter(risks[min_var_idx], returns[min_var_idx], 
               color='red', s=100, zorder=5, label='Minimum Variance')
    plt.scatter(risks[max_sharpe_idx], returns[max_sharpe_idx], 
               color='green', s=100, zorder=5, label='Maximum Sharpe')
    
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Portfolio Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_weight_distribution(weights: np.ndarray,
                           asset_names: Optional[List[str]] = None,
                           save_path: Optional[str] = None) -> None:
    """
    Plot portfolio weight distribution.
    
    Args:
        weights: Portfolio weights
        asset_names: Asset names
        save_path: Path to save the plot
    """
    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(len(weights))]
    
    # Sort weights in descending order
    sorted_indices = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_indices]
    sorted_names = [asset_names[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    bars = plt.bar(range(len(weights)), sorted_weights, alpha=0.7)
    
    # Color bars by weight size
    colors = plt.cm.viridis(sorted_weights / np.max(sorted_weights))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.title('Portfolio Weight Distribution')
    plt.xticks(range(len(weights)), sorted_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_risk_contribution(weights: np.ndarray,
                          cov_matrix: np.ndarray,
                          asset_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> None:
    """
    Plot risk contribution by asset.
    
    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix
        asset_names: Asset names
        save_path: Path to save the plot
    """
    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(len(weights))]
    
    # Compute risk contributions
    marginal_contributions = cov_matrix @ weights
    risk_contributions = weights * marginal_contributions
    
    # Sort by risk contribution
    sorted_indices = np.argsort(risk_contributions)[::-1]
    sorted_contributions = risk_contributions[sorted_indices]
    sorted_names = [asset_names[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    bars = plt.bar(range(len(weights)), sorted_contributions, alpha=0.7)
    
    # Color bars by contribution size
    colors = plt.cm.Reds(sorted_contributions / np.max(sorted_contributions))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Assets')
    plt.ylabel('Risk Contribution')
    plt.title('Portfolio Risk Contribution by Asset')
    plt.xticks(range(len(weights)), sorted_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_turnover_analysis(weights_history: List[np.ndarray],
                          dates: Optional[List] = None,
                          save_path: Optional[str] = None) -> None:
    """
    Plot portfolio turnover over time.
    
    Args:
        weights_history: List of portfolio weights over time
        dates: Dates for x-axis
        save_path: Path to save the plot
    """
    if dates is None:
        dates = range(len(weights_history))
    
    # Compute turnover
    turnovers = []
    for i in range(1, len(weights_history)):
        turnover = np.sum(np.abs(weights_history[i] - weights_history[i-1]))
        turnovers.append(turnover)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(dates[1:], turnovers, 'b-', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Turnover')
    plt.title('Portfolio Turnover Over Time')
    plt.grid(True, alpha=0.3)
    
    # Add average turnover line
    avg_turnover = np.mean(turnovers)
    plt.axhline(y=avg_turnover, color='red', linestyle='--', 
                label=f'Average Turnover: {avg_turnover:.4f}')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_portfolios(portfolio_results: Dict[str, Dict],
                      save_path: Optional[str] = None) -> None:
    """
    Compare multiple portfolio strategies.
    
    Args:
        portfolio_results: Dictionary with portfolio results
        save_path: Path to save the plot
    """
    strategies = list(portfolio_results.keys())
    
    # Extract metrics
    returns = [portfolio_results[s]['return'] for s in strategies]
    risks = [portfolio_results[s]['risk'] for s in strategies]
    sharpe_ratios = [portfolio_results[s]['sharpe_ratio'] for s in strategies]
    turnovers = [portfolio_results[s]['turnover'] for s in strategies]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Return comparison
    axes[0, 0].bar(strategies, returns, alpha=0.7)
    axes[0, 0].set_title('Portfolio Returns')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Risk comparison
    axes[0, 1].bar(strategies, risks, alpha=0.7, color='orange')
    axes[0, 1].set_title('Portfolio Risk')
    axes[0, 1].set_ylabel('Risk')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Sharpe ratio comparison
    axes[1, 0].bar(strategies, sharpe_ratios, alpha=0.7, color='green')
    axes[1, 0].set_title('Sharpe Ratios')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Turnover comparison
    axes[1, 1].bar(strategies, turnovers, alpha=0.7, color='red')
    axes[1, 1].set_title('Portfolio Turnover')
    axes[1, 1].set_ylabel('Turnover')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def generate_report(portfolio_results: Dict[str, Dict],
                   save_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive portfolio report.
    
    Args:
        portfolio_results: Dictionary with portfolio results
        save_path: Path to save the report
        
    Returns:
        Report as string
    """
    report = []
    report.append("=" * 60)
    report.append("PORTFOLIO OPTIMIZATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    for strategy, results in portfolio_results.items():
        report.append(f"Strategy: {strategy}")
        report.append("-" * 40)
        
        if 'error' in results:
            report.append(f"Error: {results['error']}")
        else:
            report.append(f"Return: {results['return']:.4f}")
            report.append(f"Risk: {results['risk']:.4f}")
            report.append(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
            report.append(f"Turnover: {results['turnover']:.4f}")
            
            if 'info' in results:
                report.append(f"Solver Status: {results['info'].get('status', 'N/A')}")
                if 'solve_time' in results['info']:
                    report.append(f"Solve Time: {results['info']['solve_time']:.4f}s")
        
        report.append("")
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text


def validate_inputs(cov_matrix: np.ndarray,
                   mean_returns: np.ndarray,
                   weights: Optional[np.ndarray] = None) -> bool:
    """
    Validate input matrices and vectors.
    
    Args:
        cov_matrix: Covariance matrix
        mean_returns: Mean returns vector
        weights: Portfolio weights (optional)
        
    Returns:
        True if inputs are valid
    """
    # Check dimensions
    n = len(mean_returns)
    
    if cov_matrix.shape != (n, n):
        raise ValueError(f"Covariance matrix shape {cov_matrix.shape} doesn't match mean returns length {n}")
    
    if weights is not None and len(weights) != n:
        raise ValueError(f"Weights length {len(weights)} doesn't match mean returns length {n}")
    
    # Check for NaN and inf values
    if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
        raise ValueError("Covariance matrix contains NaN or inf values")
    
    if np.any(np.isnan(mean_returns)) or np.any(np.isinf(mean_returns)):
        raise ValueError("Mean returns contains NaN or inf values")
    
    if weights is not None:
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            raise ValueError("Weights contains NaN or inf values")
    
    # Check covariance matrix properties
    if not np.allclose(cov_matrix, cov_matrix.T):
        raise ValueError("Covariance matrix is not symmetric")
    
    # Use a more robust check for positive semi-definiteness
    try:
        eigenvals = np.linalg.eigvals(cov_matrix)
        if np.any(eigenvals < -1e-8):  # Slightly more tolerant
            raise ValueError("Covariance matrix is not positive semi-definite")
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is not valid")
    
    return True


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize weights to sum to 1.
    
    Args:
        weights: Portfolio weights
        
    Returns:
        Normalized weights
    """
    if np.sum(weights) == 0:
        warnings.warn("All weights are zero, returning equal weights")
        return np.ones(len(weights)) / len(weights)
    
    return weights / np.sum(weights) 