"""
Convex relaxation module for portfolio optimization.
Implements the basic mean-variance optimization problem.
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, Optional
import warnings


def solve_relax(cov_matrix: np.ndarray, 
                mean_returns: np.ndarray, 
                gamma: float = 1.0,
                w_prev: Optional[np.ndarray] = None,
                lambda_tc: float = 0.0) -> Tuple[np.ndarray, float, dict]:
    """
    Solve the convex relaxation of the portfolio optimization problem.
    
    Basic problem:
        min_w w^T Σ w - γ μ^T w
        subject to sum(w) = 1, w >= 0
    
    With transaction costs:
        min_w w^T Σ w - γ μ^T w + λ ||w - w_prev||_1
        subject to sum(w) = 1, w >= 0
    
    Args:
        cov_matrix: Covariance matrix (n x n)
        mean_returns: Mean returns vector (n,)
        gamma: Risk aversion parameter
        w_prev: Previous portfolio weights (for transaction costs)
        lambda_tc: Transaction cost parameter
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n = len(mean_returns)
    
    # Decision variable
    w = cp.Variable(n)
    
    # Objective function
    risk_term = cp.quad_form(w, cov_matrix)
    return_term = -gamma * mean_returns @ w
    
    if w_prev is not None and lambda_tc > 0:
        # Add transaction cost term
        transaction_cost = lambda_tc * cp.norm(w - w_prev, 1)
        objective = risk_term + return_term + transaction_cost
    else:
        objective = risk_term + return_term
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,  # Budget constraint
        w >= 0           # Long-only constraint
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
                "setup_time": problem.solver_stats.setup_time,
                "num_iters": problem.solver_stats.num_iters
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"CVXPY solver failed: {e}")
        # Fallback to simple equal-weight portfolio
        weights = np.ones(n) / n
        obj_value = weights @ cov_matrix @ weights - gamma * mean_returns @ weights
        solver_info = {"status": "fallback", "error": str(e)}
        
        return weights, obj_value, solver_info


def solve_min_variance(cov_matrix: np.ndarray) -> Tuple[np.ndarray, float, dict]:
    """
    Solve minimum variance portfolio.
    
    Args:
        cov_matrix: Covariance matrix (n x n)
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n = cov_matrix.shape[0]
    
    # Decision variable
    w = cp.Variable(n)
    
    # Objective: minimize variance
    objective = cp.quad_form(w, cov_matrix)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,  # Budget constraint
        w >= 0           # Long-only constraint
    ]
    
    # Solve
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        result = problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            weights = w.value
            obj_value = problem.value
            
            solver_info = {
                "status": problem.status,
                "solve_time": problem.solver_stats.solve_time,
                "setup_time": problem.solver_stats.setup_time,
                "num_iters": problem.solver_stats.num_iters
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"CVXPY solver failed: {e}")
        # Fallback to equal-weight portfolio
        weights = np.ones(n) / n
        obj_value = weights @ cov_matrix @ weights
        solver_info = {"status": "fallback", "error": str(e)}
        
        return weights, obj_value, solver_info


def solve_max_sharpe(cov_matrix: np.ndarray, 
                     mean_returns: np.ndarray,
                     risk_free_rate: float = 0.02) -> Tuple[np.ndarray, float, dict]:
    """
    Solve maximum Sharpe ratio portfolio using a simplified approach.
    
    Args:
        cov_matrix: Covariance matrix (n x n)
        mean_returns: Mean returns vector (n,)
        risk_free_rate: Risk-free rate
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n = len(mean_returns)
    
    # Decision variable
    w = cp.Variable(n)
    
    # Excess returns
    excess_returns = mean_returns - risk_free_rate
    
    # Use a linear approximation for Sharpe ratio optimization
    # Maximize excess return while minimizing risk
    objective = cp.quad_form(w, cov_matrix) - 2 * (excess_returns @ w)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,  # Budget constraint
        w >= 0           # Long-only constraint
    ]
    
    # Solve
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        result = problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            weights = w.value
            obj_value = problem.value
            
            solver_info = {
                "status": problem.status,
                "solve_time": problem.solver_stats.solve_time,
                "setup_time": problem.solver_stats.setup_time,
                "num_iters": problem.solver_stats.num_iters
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"CVXPY solver failed: {e}")
        # Fallback to equal-weight portfolio
        weights = np.ones(n) / n
        excess_return = excess_returns @ weights
        variance = weights @ cov_matrix @ weights
        obj_value = variance - 2 * excess_return
        solver_info = {"status": "fallback", "error": str(e)}
        
        return weights, obj_value, solver_info


def compute_efficient_frontier(cov_matrix: np.ndarray, 
                             mean_returns: np.ndarray,
                             n_points: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute efficient frontier by varying the risk aversion parameter.
    
    Args:
        cov_matrix: Covariance matrix (n x n)
        mean_returns: Mean returns vector (n,)
        n_points: Number of points on the frontier
        
    Returns:
        Tuple of (risks, returns, weights_array)
    """
    # Generate gamma values (risk aversion parameters)
    gamma_min = 0.1
    gamma_max = 10.0
    gamma_values = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_points)
    
    risks = []
    returns = []
    weights_array = []
    
    for gamma in gamma_values:
        try:
            weights, obj_value, _ = solve_relax(cov_matrix, mean_returns, gamma)
            
            # Compute portfolio statistics
            portfolio_return = mean_returns @ weights
            portfolio_risk = np.sqrt(weights @ cov_matrix @ weights)
            
            risks.append(portfolio_risk)
            returns.append(portfolio_return)
            weights_array.append(weights)
            
        except Exception as e:
            warnings.warn(f"Failed to solve for gamma={gamma}: {e}")
            continue
    
    return np.array(risks), np.array(returns), np.array(weights_array)


def validate_solution(weights: np.ndarray, 
                     cov_matrix: np.ndarray, 
                     mean_returns: np.ndarray) -> dict:
    """
    Validate portfolio solution.
    
    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix
        mean_returns: Mean returns
        
    Returns:
        Dictionary with validation results
    """
    # Check budget constraint
    budget_violation = abs(np.sum(weights) - 1.0)
    
    # Check non-negativity
    negative_weights = np.sum(weights < 0)
    
    # Compute portfolio statistics
    portfolio_return = mean_returns @ weights
    portfolio_risk = np.sqrt(weights @ cov_matrix @ weights)
    sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
    
    # Check for numerical issues
    has_nan = np.any(np.isnan(weights))
    has_inf = np.any(np.isinf(weights))
    
    validation = {
        "budget_violation": budget_violation,
        "negative_weights": negative_weights,
        "portfolio_return": portfolio_return,
        "portfolio_risk": portfolio_risk,
        "sharpe_ratio": sharpe_ratio,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "is_valid": (budget_violation < 1e-6 and 
                    negative_weights == 0 and 
                    not has_nan and 
                    not has_inf)
    }
    
    return validation 