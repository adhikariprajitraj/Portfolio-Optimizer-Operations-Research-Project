"""
Robust optimization module for portfolio optimization.
Implements worst-case scenario analysis, distributional ambiguity, and stress testing.
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, Dict, List, Optional
import warnings
from scipy.stats import norm, t
from sklearn.covariance import MinCovDet, LedoitWolf
from .relax import solve_relax


def worst_case_optimization(cov_matrix: np.ndarray,
                          mean_returns: np.ndarray,
                          uncertainty_set: Dict,
                          gamma: float = 1.0) -> Tuple[np.ndarray, float, dict]:
    """
    Robust optimization with worst-case scenario analysis.
    
    Args:
        cov_matrix: Base covariance matrix
        mean_returns: Base mean returns
        uncertainty_set: Dictionary defining uncertainty sets
        gamma: Risk aversion parameter
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n = len(mean_returns)
    
    # Decision variables
    w = cp.Variable(n)
    
    # Robust objective (simplified version)
    risk_term = cp.quad_form(w, cov_matrix)
    return_term = -gamma * mean_returns @ w
    
    # Add uncertainty penalty
    uncertainty_penalty = 0
    if 'return_uncertainty' in uncertainty_set:
        radius = uncertainty_set['return_uncertainty']
        uncertainty_penalty += radius * cp.norm(w, 1)
    
    if 'covariance_uncertainty' in uncertainty_set:
        radius = uncertainty_set['covariance_uncertainty']
        uncertainty_penalty += radius * cp.quad_form(w, np.eye(n))
    
    objective = risk_term + return_term + uncertainty_penalty
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    # Solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        result = problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            weights = w.value
            obj_value = problem.value
            
            solver_info = {
                "status": problem.status,
                "solve_time": problem.solver_stats.solve_time,
                "method": "worst_case_robust"
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"Robust optimization failed: {e}")
        # Fallback to standard optimization
        return solve_relax(cov_matrix, mean_returns, gamma)


def distributional_robust_optimization(returns_data: np.ndarray,
                                     confidence_level: float = 0.95,
                                     gamma: float = 1.0) -> Tuple[np.ndarray, float, dict]:
    """
    Distributionally robust optimization using Wasserstein ambiguity sets.
    
    Args:
        returns_data: Historical returns data (T x n)
        confidence_level: Confidence level for ambiguity set
        gamma: Risk aversion parameter
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    T, n = returns_data.shape
    
    # Estimate empirical distribution
    empirical_mean = np.mean(returns_data, axis=0)
    empirical_cov = np.cov(returns_data.T)
    
    # Wasserstein radius based on confidence level
    radius = norm.ppf(confidence_level) / np.sqrt(T)
    
    # Decision variables
    w = cp.Variable(n)
    
    # Dual variables for Wasserstein ambiguity
    lambda_var = cp.Variable()
    mu_var = cp.Variable(n)
    
    # Objective function
    objective = lambda_var * radius + empirical_mean @ mu_var + gamma * cp.quad_form(w, empirical_cov)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        lambda_var >= 0,
        cp.norm(mu_var, 2) <= lambda_var
    ]
    
    # Solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        result = problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            weights = w.value
            obj_value = problem.value
            
            solver_info = {
                "status": problem.status,
                "confidence_level": confidence_level,
                "wasserstein_radius": radius,
                "method": "distributional_robust"
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"Distributional robust optimization failed: {e}")
        # Fallback to empirical mean-variance
        return solve_relax(empirical_cov, empirical_mean, gamma)


def stress_test_portfolio(weights: np.ndarray,
                        cov_matrix: np.ndarray,
                        mean_returns: np.ndarray,
                        stress_scenarios: List[Dict]) -> Dict:
    """
    Stress test portfolio under various scenarios.
    
    Args:
        weights: Portfolio weights
        cov_matrix: Base covariance matrix
        mean_returns: Base mean returns
        stress_scenarios: List of stress scenarios
        
    Returns:
        Dictionary with stress test results
    """
    results = {}
    
    for i, scenario in enumerate(stress_scenarios):
        # Apply scenario shocks
        scenario_cov = cov_matrix.copy()
        scenario_returns = mean_returns.copy()
        
        if 'covariance_shock' in scenario:
            shock = scenario['covariance_shock']
            scenario_cov *= (1 + shock)
        
        if 'return_shock' in scenario:
            shock = scenario['return_shock']
            scenario_returns *= (1 + shock)
        
        # Compute scenario statistics
        scenario_return = scenario_returns @ weights
        scenario_risk = np.sqrt(weights @ scenario_cov @ weights)
        scenario_sharpe = scenario_return / scenario_risk if scenario_risk > 0 else 0
        
        results[f'scenario_{i}'] = {
            'return': scenario_return,
            'risk': scenario_risk,
            'sharpe': scenario_sharpe,
            'description': scenario.get('description', f'Scenario {i}')
        }
    
    return results


def robust_covariance_estimation(returns_data: np.ndarray,
                                method: str = 'ledoit_wolf') -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust covariance estimation using various methods.
    
    Args:
        returns_data: Historical returns data
        method: Estimation method ('ledoit_wolf', 'min_cov_det', 'empirical')
        
    Returns:
        Tuple of (robust_covariance, robust_mean)
    """
    if method == 'ledoit_wolf':
        estimator = LedoitWolf()
        robust_cov = estimator.fit(returns_data).covariance_
        robust_mean = np.mean(returns_data, axis=0)
        
    elif method == 'min_cov_det':
        estimator = MinCovDet()
        robust_cov = estimator.fit(returns_data).covariance_
        robust_mean = np.mean(returns_data, axis=0)
        
    elif method == 'empirical':
        robust_cov = np.cov(returns_data.T)
        robust_mean = np.mean(returns_data, axis=0)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return robust_cov, robust_mean


def conditional_value_at_risk_optimization(returns_data: np.ndarray,
                                         alpha: float = 0.05,
                                         gamma: float = 1.0) -> Tuple[np.ndarray, float, dict]:
    """
    Conditional Value at Risk (CVaR) optimization.
    
    Args:
        returns_data: Historical returns data (T x n)
        alpha: Confidence level for CVaR
        gamma: Risk aversion parameter
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    T, n = returns_data.shape
    
    # Decision variables
    w = cp.Variable(n)
    beta = cp.Variable()  # VaR
    z = cp.Variable(T)    # Auxiliary variables
    
    # Objective function
    mean_return = np.mean(returns_data, axis=0)
    objective = -gamma * mean_return @ w + beta + (1/alpha) * cp.sum(z) / T
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        z >= 0
    ]
    
    # Add constraints for each time period
    for t in range(T):
        constraints.append(z[t] >= -returns_data[t] @ w - beta)
    
    # Solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        result = problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            weights = w.value
            obj_value = problem.value
            var_value = beta.value
            
            solver_info = {
                "status": problem.status,
                "alpha": alpha,
                "var": var_value,
                "method": "cvar_optimization"
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"CVaR optimization failed: {e}")
        # Fallback to mean-variance
        cov_matrix = np.cov(returns_data.T)
        mean_returns = np.mean(returns_data, axis=0)
        return solve_relax(cov_matrix, mean_returns, gamma)


def regime_dependent_optimization(returns_data: np.ndarray,
                                regime_indicators: np.ndarray,
                                gamma: float = 1.0) -> Tuple[np.ndarray, float, dict]:
    """
    Regime-dependent portfolio optimization.
    
    Args:
        returns_data: Historical returns data
        regime_indicators: Regime indicators (0, 1, 2, etc.)
        gamma: Risk aversion parameter
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n_regimes = len(np.unique(regime_indicators))
    n_assets = returns_data.shape[1]
    
    # Estimate regime-specific parameters
    regime_params = {}
    for regime in range(n_regimes):
        regime_mask = regime_indicators == regime
        regime_returns = returns_data[regime_mask]
        
        if len(regime_returns) > 0:
            regime_params[regime] = {
                'mean': np.mean(regime_returns, axis=0),
                'cov': np.cov(regime_returns.T),
                'weight': len(regime_returns) / len(returns_data)
            }
    
    # Decision variables
    w = cp.Variable(n_assets)
    
    # Objective function (weighted across regimes)
    objective = 0
    for regime, params in regime_params.items():
        weight = params['weight']
        mean = params['mean']
        cov = params['cov']
        
        objective += weight * (cp.quad_form(w, cov) - gamma * mean @ w)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    # Solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        result = problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            weights = w.value
            obj_value = problem.value
            
            solver_info = {
                "status": problem.status,
                "n_regimes": n_regimes,
                "regime_weights": {k: v['weight'] for k, v in regime_params.items()},
                "method": "regime_dependent"
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"Regime-dependent optimization failed: {e}")
        # Fallback to overall mean-variance
        cov_matrix = np.cov(returns_data.T)
        mean_returns = np.mean(returns_data, axis=0)
        return solve_relax(cov_matrix, mean_returns, gamma) 