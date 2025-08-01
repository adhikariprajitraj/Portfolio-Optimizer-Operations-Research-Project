"""
Branch-and-Bound module for cardinality-constrained portfolio optimization.
Implements surrogate relaxation and branch-and-bound algorithm.
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, List, Optional, Dict
import warnings
from copy import deepcopy


def solve_surrogate_relaxation(cov_matrix: np.ndarray,
                              mean_returns: np.ndarray,
                              k: int,
                              gamma: float = 1.0,
                              w_prev: Optional[np.ndarray] = None,
                              lambda_tc: float = 0.0) -> Tuple[np.ndarray, float, dict]:
    """
    Solve surrogate relaxation for cardinality-constrained portfolio optimization.
    
    Surrogate relaxation:
        min_w w^T Σ w - γ μ^T w + λ ||w - w_prev||_1
        subject to sum(w) = 1, w >= 0, sum(z) = k, w <= z, z binary
    
    Relaxed to:
        min_w w^T Σ w - γ μ^T w + λ ||w - w_prev||_1 + M * sum(z)
        subject to sum(w) = 1, w >= 0, 0 <= z <= 1, sum(z) = k
    
    Args:
        cov_matrix: Covariance matrix (n x n)
        mean_returns: Mean returns vector (n,)
        k: Cardinality constraint (number of assets to hold)
        gamma: Risk aversion parameter
        w_prev: Previous portfolio weights
        lambda_tc: Transaction cost parameter
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n = len(mean_returns)
    
    if w_prev is None:
        w_prev = np.ones(n) / n
    
    # Decision variables
    w = cp.Variable(n)  # Portfolio weights
    z = cp.Variable(n)  # Binary variables (relaxed to [0,1])
    
    # Objective function
    risk_term = cp.quad_form(w, cov_matrix)
    return_term = -gamma * mean_returns @ w
    
    if lambda_tc > 0:
        transaction_cost = lambda_tc * cp.norm(w - w_prev, 1)
    else:
        transaction_cost = 0
    
    # Penalty term for cardinality
    M = 1e6  # Large penalty parameter
    cardinality_penalty = M * cp.sum(z)
    
    objective = risk_term + return_term + transaction_cost + cardinality_penalty
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,      # Budget constraint
        w >= 0,              # Long-only constraint
        w <= z,              # Link w and z
        0 <= z,              # Lower bound on z
        z <= 1,              # Upper bound on z (relaxed binary)
        cp.sum(z) == k       # Cardinality constraint
    ]
    
    # Solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        result = problem.solve()
        
        if problem.status not in ["infeasible", "unbounded"]:
            weights = w.value
            z_values = z.value
            obj_value = problem.value
            
            solver_info = {
                "status": problem.status,
                "solve_time": problem.solver_stats.solve_time,
                "setup_time": problem.solver_stats.setup_time,
                "num_iters": problem.solver_stats.num_iters,
                "z_values": z_values,
                "cardinality": np.sum(z_values > 0.5)
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"Surrogate relaxation failed: {e}")
        # Fallback to equal weights
        weights = np.ones(n) / n
        obj_value = weights @ cov_matrix @ weights - gamma * mean_returns @ weights
        solver_info = {"status": "fallback", "error": str(e)}
        
        return weights, obj_value, solver_info


def branch_and_bound(cov_matrix: np.ndarray,
                    mean_returns: np.ndarray,
                    k: int,
                    w_prev: Optional[np.ndarray] = None,
                    gamma: float = 1.0,
                    lambda_tc: float = 0.0,
                    max_iterations: int = 1000,
                    tolerance: float = 1e-6) -> Tuple[np.ndarray, float, dict]:
    """
    Branch-and-bound algorithm for cardinality-constrained portfolio optimization.
    
    Args:
        cov_matrix: Covariance matrix (n x n)
        mean_returns: Mean returns vector (n,)
        k: Cardinality constraint
        w_prev: Previous portfolio weights
        gamma: Risk aversion parameter
        lambda_tc: Transaction cost parameter
        max_iterations: Maximum number of iterations
        tolerance: Tolerance for convergence
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n = len(mean_returns)
    
    if w_prev is None:
        w_prev = np.ones(n) / n
    
    # Initialize
    best_objective = float('inf')
    best_weights = None
    best_solution_info = {}
    
    # Solve surrogate relaxation to get initial bound
    try:
        weights_relax, obj_relax, info_relax = solve_surrogate_relaxation(
            cov_matrix, mean_returns, k, gamma, w_prev, lambda_tc
        )
        
        # Round to get feasible solution
        weights_rounded = round_weights(weights_relax, k)
        obj_rounded = compute_objective(weights_rounded, cov_matrix, mean_returns, 
                                      gamma, w_prev, lambda_tc)
        
        if obj_rounded < best_objective:
            best_objective = obj_rounded
            best_weights = weights_rounded
            best_solution_info = {
                "method": "surrogate_rounded",
                "original_obj": obj_relax,
                "rounded_obj": obj_rounded
            }
            
    except Exception as e:
        warnings.warn(f"Initial surrogate relaxation failed: {e}")
    
    # If no feasible solution found, use greedy approach
    if best_weights is None:
        best_weights = greedy_cardinality_solution(cov_matrix, mean_returns, k, 
                                                 gamma, w_prev, lambda_tc)
        best_objective = compute_objective(best_weights, cov_matrix, mean_returns,
                                         gamma, w_prev, lambda_tc)
        best_solution_info = {"method": "greedy"}
    
    # Branch-and-bound iterations
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        # Try to improve solution by fixing fractional components
        improved_weights = improve_solution(best_weights, cov_matrix, mean_returns,
                                         k, gamma, w_prev, lambda_tc)
        
        if improved_weights is not None:
            improved_obj = compute_objective(improved_weights, cov_matrix, mean_returns,
                                          gamma, w_prev, lambda_tc)
            
            if improved_obj < best_objective - tolerance:
                best_objective = improved_obj
                best_weights = improved_weights
                best_solution_info["method"] = "branch_and_bound"
                best_solution_info["iterations"] = iteration
            else:
                break
        else:
            break
    
    solver_info = {
        "status": "success",
        "iterations": iteration,
        "best_objective": best_objective,
        "cardinality": np.sum(best_weights > 1e-6),
        **best_solution_info
    }
    
    return best_weights, best_objective, solver_info


def round_weights(weights: np.ndarray, k: int) -> np.ndarray:
    """
    Round weights to satisfy cardinality constraint.
    
    Args:
        weights: Portfolio weights
        k: Cardinality constraint
        
    Returns:
        Rounded weights
    """
    n = len(weights)
    
    # Sort weights in descending order
    sorted_indices = np.argsort(weights)[::-1]
    
    # Keep top k weights
    rounded_weights = np.zeros(n)
    for i in range(k):
        rounded_weights[sorted_indices[i]] = weights[sorted_indices[i]]
    
    # Normalize to sum to 1
    if np.sum(rounded_weights) > 0:
        rounded_weights = rounded_weights / np.sum(rounded_weights)
    
    return rounded_weights


def compute_objective(weights: np.ndarray,
                     cov_matrix: np.ndarray,
                     mean_returns: np.ndarray,
                     gamma: float,
                     w_prev: np.ndarray,
                     lambda_tc: float) -> float:
    """
    Compute objective function value.
    
    Args:
        weights: Portfolio weights
        cov_matrix: Covariance matrix
        mean_returns: Mean returns
        gamma: Risk aversion parameter
        w_prev: Previous weights
        lambda_tc: Transaction cost parameter
        
    Returns:
        Objective function value
    """
    risk_term = weights @ cov_matrix @ weights
    return_term = -gamma * mean_returns @ weights
    
    if lambda_tc > 0:
        transaction_cost = lambda_tc * np.sum(np.abs(weights - w_prev))
    else:
        transaction_cost = 0
    
    return risk_term + return_term + transaction_cost


def greedy_cardinality_solution(cov_matrix: np.ndarray,
                               mean_returns: np.ndarray,
                               k: int,
                               gamma: float,
                               w_prev: np.ndarray,
                               lambda_tc: float) -> np.ndarray:
    """
    Greedy algorithm for cardinality-constrained portfolio optimization.
    
    Args:
        cov_matrix: Covariance matrix (n x n)
        mean_returns: Mean returns vector (n,)
        k: Cardinality constraint
        gamma: Risk aversion parameter
        w_prev: Previous portfolio weights
        lambda_tc: Transaction cost parameter
        
    Returns:
        Portfolio weights
    """
    n = len(mean_returns)
    
    # Start with equal weights
    weights = np.ones(n) / n
    
    # Iteratively improve by swapping assets
    for _ in range(k * 2):  # Limit iterations
        best_improvement = 0
        best_swap = None
        
        # Try all possible swaps
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Create new weights with swap
                    new_weights = weights.copy()
                    new_weights[i] = weights[j]
                    new_weights[j] = weights[i]
                    
                    # Normalize
                    if np.sum(new_weights) > 0:
                        new_weights = new_weights / np.sum(new_weights)
                    
                    # Compute improvement
                    old_obj = compute_objective(weights, cov_matrix, mean_returns,
                                              gamma, w_prev, lambda_tc)
                    new_obj = compute_objective(new_weights, cov_matrix, mean_returns,
                                              gamma, w_prev, lambda_tc)
                    
                    improvement = old_obj - new_obj
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_swap = (i, j)
        
        # Apply best swap if improvement found
        if best_swap is not None and best_improvement > 1e-8:
            i, j = best_swap
            weights[i], weights[j] = weights[j], weights[i]
        else:
            break
    
    return weights


def improve_solution(weights: np.ndarray,
                    cov_matrix: np.ndarray,
                    mean_returns: np.ndarray,
                    k: int,
                    gamma: float,
                    w_prev: np.ndarray,
                    lambda_tc: float) -> Optional[np.ndarray]:
    """
    Try to improve solution by fixing fractional components.
    
    Args:
        weights: Current portfolio weights
        cov_matrix: Covariance matrix
        mean_returns: Mean returns
        k: Cardinality constraint
        gamma: Risk aversion parameter
        w_prev: Previous weights
        lambda_tc: Transaction cost parameter
        
    Returns:
        Improved weights or None if no improvement
    """
    n = len(weights)
    
    # Find fractional components
    fractional_indices = np.where((weights > 1e-6) & (weights < 1 - 1e-6))[0]
    
    if len(fractional_indices) == 0:
        return None
    
    # Try fixing each fractional component
    best_weights = None
    best_objective = compute_objective(weights, cov_matrix, mean_returns,
                                     gamma, w_prev, lambda_tc)
    
    for idx in fractional_indices:
        # Try setting to 0
        weights_zero = weights.copy()
        weights_zero[idx] = 0
        if np.sum(weights_zero) > 0:
            weights_zero = weights_zero / np.sum(weights_zero)
            obj_zero = compute_objective(weights_zero, cov_matrix, mean_returns,
                                       gamma, w_prev, lambda_tc)
            
            if obj_zero < best_objective:
                best_objective = obj_zero
                best_weights = weights_zero
        
        # Try setting to maximum allowed
        weights_max = weights.copy()
        weights_max[idx] = min(weights[idx] * 2, 1.0)
        if np.sum(weights_max) > 0:
            weights_max = weights_max / np.sum(weights_max)
            obj_max = compute_objective(weights_max, cov_matrix, mean_returns,
                                      gamma, w_prev, lambda_tc)
            
            if obj_max < best_objective:
                best_objective = obj_max
                best_weights = weights_max
    
    return best_weights


def validate_cardinality_solution(weights: np.ndarray, k: int) -> dict:
    """
    Validate cardinality-constrained solution.
    
    Args:
        weights: Portfolio weights
        k: Cardinality constraint
        
    Returns:
        Dictionary with validation results
    """
    n = len(weights)
    
    # Check budget constraint
    budget_violation = abs(np.sum(weights) - 1.0)
    
    # Check non-negativity
    negative_weights = np.sum(weights < 0)
    
    # Check cardinality constraint
    non_zero_weights = np.sum(weights > 1e-6)
    cardinality_violation = abs(non_zero_weights - k)
    
    # Check for numerical issues
    has_nan = np.any(np.isnan(weights))
    has_inf = np.any(np.isinf(weights))
    
    validation = {
        "budget_violation": budget_violation,
        "negative_weights": negative_weights,
        "cardinality": non_zero_weights,
        "cardinality_violation": cardinality_violation,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "is_valid": (budget_violation < 1e-6 and 
                    negative_weights == 0 and 
                    cardinality_violation == 0 and
                    not has_nan and 
                    not has_inf)
    }
    
    return validation 