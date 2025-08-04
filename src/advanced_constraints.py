"""
Advanced constraints module for real-world portfolio optimization.
Implements sector limits, leverage constraints, lot sizes, and other practical constraints.
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, Dict, List, Optional
import warnings


def sector_constrained_optimization(cov_matrix: np.ndarray,
                                  mean_returns: np.ndarray,
                                  sector_labels: List[str],
                                  sector_limits: Dict[str, float],
                                  gamma: float = 1.0) -> Tuple[np.ndarray, float, dict]:
    """
    Portfolio optimization with sector constraints.
    
    Args:
        cov_matrix: Covariance matrix
        mean_returns: Mean returns vector
        sector_labels: List of sector labels for each asset
        sector_limits: Dictionary of sector limits (e.g., {'TECH': 0.3})
        gamma: Risk aversion parameter
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n = len(mean_returns)
    
    # Decision variables
    w = cp.Variable(n)
    
    # Objective function
    risk_term = cp.quad_form(w, cov_matrix)
    return_term = -gamma * mean_returns @ w
    objective = risk_term + return_term
    
    # Basic constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    
    # Sector constraints
    unique_sectors = list(set(sector_labels))
    for sector in unique_sectors:
        if sector in sector_limits:
            sector_mask = [i for i, s in enumerate(sector_labels) if s == sector]
            sector_weight = cp.sum(w[sector_mask])
            constraints.append(sector_weight <= sector_limits[sector])
    
    # Solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        result = problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            weights = w.value
            obj_value = problem.value
            
            # Compute sector allocations
            sector_allocations = {}
            for sector in unique_sectors:
                sector_mask = [i for i, s in enumerate(sector_labels) if s == sector]
                sector_allocations[sector] = np.sum(weights[sector_mask])
            
            solver_info = {
                "status": problem.status,
                "sector_allocations": sector_allocations,
                "method": "sector_constrained"
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"Sector constrained optimization failed: {e}")
        # Fallback to standard optimization
        from .relax import solve_relax
        return solve_relax(cov_matrix, mean_returns, gamma)


def leverage_constrained_optimization(cov_matrix: np.ndarray,
                                   mean_returns: np.ndarray,
                                   max_leverage: float = 1.5,
                                   gamma: float = 1.0) -> Tuple[np.ndarray, float, dict]:
    """
    Portfolio optimization with leverage constraints.
    
    Args:
        cov_matrix: Covariance matrix
        mean_returns: Mean returns vector
        max_leverage: Maximum leverage allowed
        gamma: Risk aversion parameter
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n = len(mean_returns)
    
    # Decision variables (long and short positions)
    w_long = cp.Variable(n)   # Long positions
    w_short = cp.Variable(n)  # Short positions
    
    # Net weights
    w_net = w_long - w_short
    
    # Objective function
    risk_term = cp.quad_form(w_net, cov_matrix)
    return_term = -gamma * mean_returns @ w_net
    objective = risk_term + return_term
    
    # Constraints
    constraints = [
        cp.sum(w_net) == 1,           # Budget constraint
        w_long >= 0,                  # Long positions non-negative
        w_short >= 0,                 # Short positions non-negative
        cp.sum(w_long) + cp.sum(w_short) <= max_leverage  # Leverage constraint
    ]
    
    # Solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        result = problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            weights = w_net.value
            obj_value = problem.value
            
            # Compute leverage metrics
            total_long = np.sum(w_long.value)
            total_short = np.sum(w_short.value)
            gross_leverage = total_long + total_short
            net_leverage = total_long - total_short
            
            solver_info = {
                "status": problem.status,
                "gross_leverage": gross_leverage,
                "net_leverage": net_leverage,
                "total_long": total_long,
                "total_short": total_short,
                "method": "leverage_constrained"
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"Leverage constrained optimization failed: {e}")
        # Fallback to standard optimization
        from .relax import solve_relax
        return solve_relax(cov_matrix, mean_returns, gamma)


def lot_size_constrained_optimization(cov_matrix: np.ndarray,
                                    mean_returns: np.ndarray,
                                    lot_sizes: np.ndarray,
                                    portfolio_value: float = 1000000,
                                    gamma: float = 1.0) -> Tuple[np.ndarray, float, dict]:
    """
    Portfolio optimization with lot size constraints.
    
    Args:
        cov_matrix: Covariance matrix
        mean_returns: Mean returns vector
        lot_sizes: Lot sizes for each asset
        portfolio_value: Total portfolio value
        gamma: Risk aversion parameter
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n = len(mean_returns)
    
    # Decision variables (integer lots)
    lots = cp.Variable(n, integer=True)
    
    # Convert lots to weights
    lot_values = lot_sizes * portfolio_value
    w = lots * lot_sizes / portfolio_value
    
    # Objective function
    risk_term = cp.quad_form(w, cov_matrix)
    return_term = -gamma * mean_returns @ w
    objective = risk_term + return_term
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,           # Budget constraint
        lots >= 0,                # Non-negative lots
        cp.sum(lots * lot_sizes) <= portfolio_value  # Portfolio value constraint
    ]
    
    # Solve the problem (mixed-integer)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        result = problem.solve(solver='ECOS_BB')  # Branch and bound solver
        
        if problem.status not in ["infeasible", "unbounded"]:
            weights = w.value
            obj_value = problem.value
            
            # Compute lot allocation
            lot_allocation = lots.value
            total_lots = np.sum(lot_allocation)
            
            solver_info = {
                "status": problem.status,
                "total_lots": total_lots,
                "lot_allocation": lot_allocation,
                "method": "lot_size_constrained"
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"Lot size constrained optimization failed: {e}")
        # Fallback to standard optimization
        from .relax import solve_relax
        return solve_relax(cov_matrix, mean_returns, gamma)


def concentration_constrained_optimization(cov_matrix: np.ndarray,
                                        mean_returns: np.ndarray,
                                        max_concentration: float = 0.1,
                                        min_concentration: float = 0.01,
                                        gamma: float = 1.0) -> Tuple[np.ndarray, float, dict]:
    """
    Portfolio optimization with concentration constraints.
    
    Args:
        cov_matrix: Covariance matrix
        mean_returns: Mean returns vector
        max_concentration: Maximum weight per asset
        min_concentration: Minimum weight per asset (if > 0)
        gamma: Risk aversion parameter
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n = len(mean_returns)
    
    # Decision variables
    w = cp.Variable(n)
    
    # Objective function
    risk_term = cp.quad_form(w, cov_matrix)
    return_term = -gamma * mean_returns @ w
    objective = risk_term + return_term
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_concentration  # Maximum concentration
    ]
    
    # Add minimum concentration if specified
    if min_concentration > 0:
        # Use binary variables for minimum concentration
        z = cp.Variable(n, boolean=True)
        constraints.extend([
            w >= min_concentration * z,
            w <= max_concentration * z
        ])
    
    # Solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        result = problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            weights = w.value
            obj_value = problem.value
            
            # Compute concentration metrics
            max_weight = np.max(weights)
            min_weight = np.min(weights[weights > 0]) if np.any(weights > 0) else 0
            effective_n = 1 / np.sum(weights ** 2)
            
            solver_info = {
                "status": problem.status,
                "max_weight": max_weight,
                "min_weight": min_weight,
                "effective_n": effective_n,
                "method": "concentration_constrained"
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"Concentration constrained optimization failed: {e}")
        # Fallback to standard optimization
        from .relax import solve_relax
        return solve_relax(cov_matrix, mean_returns, gamma)


def tracking_error_constrained_optimization(cov_matrix: np.ndarray,
                                         mean_returns: np.ndarray,
                                         benchmark_weights: np.ndarray,
                                         max_tracking_error: float = 0.05,
                                         gamma: float = 1.0) -> Tuple[np.ndarray, float, dict]:
    """
    Portfolio optimization with tracking error constraints.
    
    Args:
        cov_matrix: Covariance matrix
        mean_returns: Mean returns vector
        benchmark_weights: Benchmark portfolio weights
        max_tracking_error: Maximum tracking error
        gamma: Risk aversion parameter
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n = len(mean_returns)
    
    # Decision variables
    w = cp.Variable(n)
    
    # Active weights
    w_active = w - benchmark_weights
    
    # Objective function
    risk_term = cp.quad_form(w, cov_matrix)
    return_term = -gamma * mean_returns @ w
    objective = risk_term + return_term
    
    # Tracking error constraint
    tracking_error = cp.sqrt(cp.quad_form(w_active, cov_matrix))
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        tracking_error <= max_tracking_error
    ]
    
    # Solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        result = problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            weights = w.value
            obj_value = problem.value
            
            # Compute tracking error
            actual_tracking_error = np.sqrt(w_active.value @ cov_matrix @ w_active.value)
            
            solver_info = {
                "status": problem.status,
                "tracking_error": actual_tracking_error,
                "active_weights": w_active.value,
                "method": "tracking_error_constrained"
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"Tracking error constrained optimization failed: {e}")
        # Fallback to standard optimization
        from .relax import solve_relax
        return solve_relax(cov_matrix, mean_returns, gamma)


def multi_period_optimization(returns_data: np.ndarray,
                            n_periods: int = 4,
                            gamma: float = 1.0,
                            discount_factor: float = 0.95) -> Tuple[np.ndarray, float, dict]:
    """
    Multi-period portfolio optimization.
    
    Args:
        returns_data: Historical returns data
        n_periods: Number of periods to optimize
        gamma: Risk aversion parameter
        discount_factor: Discount factor for future periods
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    T, n = returns_data.shape
    
    # Estimate parameters for each period
    period_length = T // n_periods
    period_params = []
    
    for i in range(n_periods):
        start_idx = i * period_length
        end_idx = min((i + 1) * period_length, T)
        period_data = returns_data[start_idx:end_idx]
        
        period_params.append({
            'mean': np.mean(period_data, axis=0),
            'cov': np.cov(period_data.T)
        })
    
    # Decision variables for each period
    w_periods = [cp.Variable(n) for _ in range(n_periods)]
    
    # Multi-period objective
    objective = 0
    for t in range(n_periods):
        params = period_params[t]
        discount = discount_factor ** t
        
        risk_term = cp.quad_form(w_periods[t], params['cov'])
        return_term = -gamma * params['mean'] @ w_periods[t]
        
        objective += discount * (risk_term + return_term)
    
    # Constraints for each period
    constraints = []
    for t in range(n_periods):
        constraints.extend([
            cp.sum(w_periods[t]) == 1,
            w_periods[t] >= 0
        ])
    
    # Solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        result = problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            weights = w_periods[0].value  # Return first period weights
            obj_value = problem.value
            
            solver_info = {
                "status": problem.status,
                "n_periods": n_periods,
                "discount_factor": discount_factor,
                "method": "multi_period"
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"Multi-period optimization failed: {e}")
        # Fallback to standard optimization
        cov_matrix = np.cov(returns_data.T)
        mean_returns = np.mean(returns_data, axis=0)
        from .relax import solve_relax
        return solve_relax(cov_matrix, mean_returns, gamma) 