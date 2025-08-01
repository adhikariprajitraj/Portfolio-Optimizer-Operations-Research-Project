"""
SQP (Sequential Quadratic Programming) solver wrapper for portfolio optimization.
Implements transaction cost-aware portfolio optimization using SciPy's SLSQP.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional, Callable
import warnings


def solve_sqp(cov_matrix: np.ndarray, 
              mean_returns: np.ndarray, 
              gamma: float = 1.0,
              lambda_tc: float = 0.0,
              w_prev: Optional[np.ndarray] = None,
              bounds: Optional[list] = None) -> Tuple[np.ndarray, float, dict]:
    """
    Solve portfolio optimization using SQP (SLSQP).
    
    Problem:
        min_w w^T Σ w - γ μ^T w + λ ||w - w_prev||_1
        subject to sum(w) = 1, w >= 0
    
    Args:
        cov_matrix: Covariance matrix (n x n)
        mean_returns: Mean returns vector (n,)
        gamma: Risk aversion parameter
        lambda_tc: Transaction cost parameter
        w_prev: Previous portfolio weights
        bounds: Bounds for each weight [(0, 1), ...]
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n = len(mean_returns)
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = [(0.0, 1.0)] * n
    
    # Set default previous weights if not provided
    if w_prev is None:
        w_prev = np.ones(n) / n
    
    # Objective function
    def objective(w):
        risk_term = w @ cov_matrix @ w
        return_term = -gamma * mean_returns @ w
        
        if lambda_tc > 0:
            transaction_cost = lambda_tc * np.sum(np.abs(w - w_prev))
            return risk_term + return_term + transaction_cost
        else:
            return risk_term + return_term
    
    # Gradient of objective function
    def gradient(w):
        grad_risk = 2 * cov_matrix @ w
        grad_return = -gamma * mean_returns
        
        if lambda_tc > 0:
            # Gradient of L1 norm
            grad_tc = lambda_tc * np.sign(w - w_prev)
            return grad_risk + grad_return + grad_tc
        else:
            return grad_risk + grad_return
    
    # Constraint: sum of weights equals 1
    def constraint_sum(w):
        return np.sum(w) - 1.0
    
    # Jacobian of constraint
    def constraint_jacobian(w):
        return np.ones(n)
    
    # Initial guess (equal weights)
    w0 = np.ones(n) / n
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_jacobian}
    ]
    
    # Solve using SLSQP
    try:
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-8, 'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
            obj_value = result.fun
            
            solver_info = {
                "status": "success",
                "message": result.message,
                "nit": result.nit,
                "nfev": result.nfev,
                "njev": result.njev
            }
            
            return weights, obj_value, solver_info
        else:
            warnings.warn(f"SQP solver failed: {result.message}")
            # Fallback to equal weights
            weights = np.ones(n) / n
            obj_value = objective(weights)
            solver_info = {"status": "failed", "message": result.message}
            
            return weights, obj_value, solver_info
            
    except Exception as e:
        warnings.warn(f"SQP solver error: {e}")
        # Fallback to equal weights
        weights = np.ones(n) / n
        obj_value = objective(weights)
        solver_info = {"status": "error", "error": str(e)}
        
        return weights, obj_value, solver_info


def solve_sqp_with_constraints(cov_matrix: np.ndarray,
                              mean_returns: np.ndarray,
                              gamma: float = 1.0,
                              lambda_tc: float = 0.0,
                              w_prev: Optional[np.ndarray] = None,
                              max_weight: float = 0.3,
                              min_weight: float = 0.0) -> Tuple[np.ndarray, float, dict]:
    """
    Solve portfolio optimization with additional constraints.
    
    Args:
        cov_matrix: Covariance matrix (n x n)
        mean_returns: Mean returns vector (n,)
        gamma: Risk aversion parameter
        lambda_tc: Transaction cost parameter
        w_prev: Previous portfolio weights
        max_weight: Maximum weight per asset
        min_weight: Minimum weight per asset
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n = len(mean_returns)
    
    # Set bounds with max/min weight constraints
    bounds = [(min_weight, max_weight)] * n
    
    return solve_sqp(cov_matrix, mean_returns, gamma, lambda_tc, w_prev, bounds)


def solve_sqp_sector_constraints(cov_matrix: np.ndarray,
                                mean_returns: np.ndarray,
                                sector_labels: list,
                                sector_limits: dict,
                                gamma: float = 1.0,
                                lambda_tc: float = 0.0,
                                w_prev: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, dict]:
    """
    Solve portfolio optimization with sector constraints.
    
    Args:
        cov_matrix: Covariance matrix (n x n)
        mean_returns: Mean returns vector (n,)
        sector_labels: List of sector labels for each asset
        sector_limits: Dictionary of sector limits {sector: max_weight}
        gamma: Risk aversion parameter
        lambda_tc: Transaction cost parameter
        w_prev: Previous portfolio weights
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n = len(mean_returns)
    
    # Set default previous weights if not provided
    if w_prev is None:
        w_prev = np.ones(n) / n
    
    # Objective function
    def objective(w):
        risk_term = w @ cov_matrix @ w
        return_term = -gamma * mean_returns @ w
        
        if lambda_tc > 0:
            transaction_cost = lambda_tc * np.sum(np.abs(w - w_prev))
            return risk_term + return_term + transaction_cost
        else:
            return risk_term + return_term
    
    # Gradient of objective function
    def gradient(w):
        grad_risk = 2 * cov_matrix @ w
        grad_return = -gamma * mean_returns
        
        if lambda_tc > 0:
            grad_tc = lambda_tc * np.sign(w - w_prev)
            return grad_risk + grad_return + grad_tc
        else:
            return grad_risk + grad_return
    
    # Budget constraint
    def constraint_sum(w):
        return np.sum(w) - 1.0
    
    # Sector constraints
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    
    # Add sector constraints
    for sector, max_weight in sector_limits.items():
        sector_indices = [i for i, s in enumerate(sector_labels) if s == sector]
        
        def sector_constraint(w, indices=sector_indices, limit=max_weight):
            return limit - np.sum(w[indices])
        
        constraints.append({'type': 'ineq', 'fun': sector_constraint})
    
    # Initial guess
    w0 = np.ones(n) / n
    
    # Bounds
    bounds = [(0.0, 1.0)] * n
    
    # Solve
    try:
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-8, 'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
            obj_value = result.fun
            
            solver_info = {
                "status": "success",
                "message": result.message,
                "nit": result.nit,
                "nfev": result.nfev,
                "njev": result.njev
            }
            
            return weights, obj_value, solver_info
        else:
            warnings.warn(f"SQP solver failed: {result.message}")
            # Fallback to equal weights
            weights = np.ones(n) / n
            obj_value = objective(weights)
            solver_info = {"status": "failed", "message": result.message}
            
            return weights, obj_value, solver_info
            
    except Exception as e:
        warnings.warn(f"SQP solver error: {e}")
        # Fallback to equal weights
        weights = np.ones(n) / n
        obj_value = objective(weights)
        solver_info = {"status": "error", "error": str(e)}
        
        return weights, obj_value, solver_info


def compare_solvers(cov_matrix: np.ndarray,
                   mean_returns: np.ndarray,
                   gamma: float = 1.0,
                   lambda_tc: float = 0.01,
                   w_prev: Optional[np.ndarray] = None) -> dict:
    """
    Compare different solvers for portfolio optimization.
    
    Args:
        cov_matrix: Covariance matrix (n x n)
        mean_returns: Mean returns vector (n,)
        gamma: Risk aversion parameter
        lambda_tc: Transaction cost parameter
        w_prev: Previous portfolio weights
        
    Returns:
        Dictionary with comparison results
    """
    n = len(mean_returns)
    
    if w_prev is None:
        w_prev = np.ones(n) / n
    
    results = {}
    
    # Test SQP solver
    try:
        weights_sqp, obj_sqp, info_sqp = solve_sqp(
            cov_matrix, mean_returns, gamma, lambda_tc, w_prev
        )
        
        results['sqp'] = {
            'weights': weights_sqp,
            'objective': obj_sqp,
            'info': info_sqp,
            'return': mean_returns @ weights_sqp,
            'risk': np.sqrt(weights_sqp @ cov_matrix @ weights_sqp),
            'turnover': np.sum(np.abs(weights_sqp - w_prev))
        }
    except Exception as e:
        results['sqp'] = {'error': str(e)}
    
    # Test convex relaxation (without transaction costs)
    try:
        from .relax import solve_relax
        weights_relax, obj_relax, info_relax = solve_relax(
            cov_matrix, mean_returns, gamma
        )
        
        results['relax'] = {
            'weights': weights_relax,
            'objective': obj_relax,
            'info': info_relax,
            'return': mean_returns @ weights_relax,
            'risk': np.sqrt(weights_relax @ cov_matrix @ weights_relax),
            'turnover': np.sum(np.abs(weights_relax - w_prev))
        }
    except Exception as e:
        results['relax'] = {'error': str(e)}
    
    # Equal weight benchmark
    weights_eq = np.ones(n) / n
    obj_eq = weights_eq @ cov_matrix @ weights_eq - gamma * mean_returns @ weights_eq
    
    results['equal_weight'] = {
        'weights': weights_eq,
        'objective': obj_eq,
        'return': mean_returns @ weights_eq,
        'risk': np.sqrt(weights_eq @ cov_matrix @ weights_eq),
        'turnover': np.sum(np.abs(weights_eq - w_prev))
    }
    
    return results 