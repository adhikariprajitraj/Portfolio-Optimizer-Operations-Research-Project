"""
Machine Learning enhanced portfolio optimization.
Implements factor models, neural networks, and ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
import warnings
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import cvxpy as cp


def factor_model_optimization(returns_data: np.ndarray,
                            n_factors: int = 3,
                            gamma: float = 1.0) -> Tuple[np.ndarray, float, dict]:
    """
    Factor model portfolio optimization using PCA.
    
    Args:
        returns_data: Historical returns data (T x n)
        n_factors: Number of factors to extract
        gamma: Risk aversion parameter
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    T, n = returns_data.shape
    
    # Extract factors using PCA
    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(returns_data)
    factor_loadings = pca.components_.T  # n x k
    
    # Estimate factor covariance and mean
    factor_cov = np.cov(factors.T)
    factor_mean = np.mean(factors, axis=0)
    
    # Specific variance (diagonal matrix)
    residuals = returns_data - factors @ factor_loadings.T
    specific_var = np.var(residuals, axis=0)
    D = np.diag(specific_var)
    
    # Decision variables
    w = cp.Variable(n)
    
    # Factor model risk decomposition
    factor_risk = cp.quad_form(factor_loadings.T @ w, factor_cov)
    specific_risk = cp.quad_form(w, D)
    total_risk = factor_risk + specific_risk
    
    # Expected return (using factor model)
    asset_mean = factor_loadings @ factor_mean
    expected_return = asset_mean @ w
    
    # Objective function
    objective = total_risk - gamma * expected_return
    
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
                "n_factors": n_factors,
                "explained_variance": pca.explained_variance_ratio_,
                "method": "factor_model"
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"Factor model optimization failed: {e}")
        # Fallback to standard optimization
        cov_matrix = np.cov(returns_data.T)
        mean_returns = np.mean(returns_data, axis=0)
        from .relax import solve_relax
        return solve_relax(cov_matrix, mean_returns, gamma)


def neural_network_optimization(returns_data: np.ndarray,
                              features: Optional[np.ndarray] = None,
                              gamma: float = 1.0,
                              hidden_layers: Tuple[int, ...] = (50, 25)) -> Tuple[np.ndarray, float, dict]:
    """
    Neural network enhanced portfolio optimization.
    
    Args:
        returns_data: Historical returns data (T x n)
        features: Additional features (T x m)
        gamma: Risk aversion parameter
        hidden_layers: Neural network architecture
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    T, n = returns_data.shape
    
    # Prepare features
    if features is None:
        # Use lagged returns as features
        features = np.roll(returns_data, 1, axis=0)
        features[0, :] = 0  # First row has no lag
    
    # Train neural network for return prediction
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split data for time series validation
    split_point = int(0.8 * T)
    X_train = features_scaled[:split_point]
    y_train = returns_data[:split_point]
    X_test = features_scaled[split_point:]
    y_test = returns_data[split_point:]
    
    # Train neural network
    nn_model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        max_iter=1000,
        random_state=42
    )
    
    # Train separate model for each asset
    predicted_returns = np.zeros_like(returns_data)
    
    for i in range(n):
        nn_model.fit(X_train, y_train.iloc[:, i])
        predicted_returns[:, i] = nn_model.predict(features_scaled)
    
    # Use predicted returns for optimization
    predicted_mean = np.mean(predicted_returns, axis=0)
    predicted_cov = np.cov(predicted_returns.T)
    
    # Decision variables
    w = cp.Variable(n)
    
    # Objective function
    risk_term = cp.quad_form(w, predicted_cov)
    return_term = -gamma * predicted_mean @ w
    
    objective = risk_term + return_term
    
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
                "nn_architecture": hidden_layers,
                "training_samples": len(X_train),
                "method": "neural_network"
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"Neural network optimization failed: {e}")
        # Fallback to standard optimization
        cov_matrix = np.cov(returns_data.T)
        mean_returns = np.mean(returns_data, axis=0)
        from .relax import solve_relax
        return solve_relax(cov_matrix, mean_returns, gamma)


def ensemble_optimization(returns_data: np.ndarray,
                         methods: List[str] = ['mean_variance', 'factor_model', 'neural_network'],
                         gamma: float = 1.0) -> Tuple[np.ndarray, float, dict]:
    """
    Ensemble portfolio optimization combining multiple methods.
    
    Args:
        returns_data: Historical returns data
        methods: List of optimization methods to combine
        gamma: Risk aversion parameter
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    n_assets = returns_data.shape[1]
    ensemble_weights = []
    method_results = {}
    
    # Run each method
    for method in methods:
        try:
            if method == 'mean_variance':
                cov_matrix = np.cov(returns_data.T)
                mean_returns = np.mean(returns_data, axis=0)
                from .relax import solve_relax
                weights, obj_val, info = solve_relax(cov_matrix, mean_returns, gamma)
                
            elif method == 'factor_model':
                weights, obj_val, info = factor_model_optimization(returns_data, n_factors=3, gamma=gamma)
                
            elif method == 'neural_network':
                weights, obj_val, info = neural_network_optimization(returns_data, gamma=gamma)
                
            else:
                continue
                
            ensemble_weights.append(weights)
            method_results[method] = {
                'weights': weights,
                'objective': obj_val,
                'info': info
            }
            
        except Exception as e:
            warnings.warn(f"Method {method} failed: {e}")
            continue
    
    if not ensemble_weights:
        raise ValueError("No methods succeeded")
    
    # Combine weights (simple average)
    combined_weights = np.mean(ensemble_weights, axis=0)
    combined_weights = combined_weights / np.sum(combined_weights)  # Normalize
    
    # Compute combined objective
    cov_matrix = np.cov(returns_data.T)
    mean_returns = np.mean(returns_data, axis=0)
    combined_obj = combined_weights @ cov_matrix @ combined_weights - gamma * mean_returns @ combined_weights
    
    solver_info = {
        "status": "success",
        "methods_used": list(method_results.keys()),
        "n_methods": len(method_results),
        "method": "ensemble"
    }
    
    return combined_weights, combined_obj, solver_info


def dynamic_optimization(returns_data: np.ndarray,
                        window_size: int = 252,
                        gamma: float = 1.0) -> Tuple[List[np.ndarray], List[float], dict]:
    """
    Dynamic portfolio optimization with rolling windows.
    
    Args:
        returns_data: Historical returns data
        window_size: Rolling window size
        gamma: Risk aversion parameter
        
    Returns:
        Tuple of (weights_history, objective_history, solver_info)
    """
    T, n = returns_data.shape
    weights_history = []
    objective_history = []
    
    for t in range(window_size, T):
        # Use data from t-window_size to t-1
        window_data = returns_data[t-window_size:t]
        
        # Estimate parameters
        cov_matrix = np.cov(window_data.T)
        mean_returns = np.mean(window_data, axis=0)
        
        # Solve optimization
        from .relax import solve_relax
        weights, obj_val, info = solve_relax(cov_matrix, mean_returns, gamma)
        
        weights_history.append(weights)
        objective_history.append(obj_val)
    
    solver_info = {
        "status": "success",
        "window_size": window_size,
        "n_periods": len(weights_history),
        "method": "dynamic_optimization"
    }
    
    return weights_history, objective_history, solver_info


def clustering_optimization(returns_data: np.ndarray,
                          n_clusters: int = 5,
                          gamma: float = 1.0) -> Tuple[np.ndarray, float, dict]:
    """
    Clustering-based portfolio optimization.
    
    Args:
        returns_data: Historical returns data
        n_clusters: Number of clusters
        gamma: Risk aversion parameter
        
    Returns:
        Tuple of (optimal_weights, objective_value, solver_info)
    """
    from sklearn.cluster import KMeans
    
    # Cluster assets based on return characteristics
    asset_features = np.column_stack([
        np.mean(returns_data, axis=0),  # Mean return
        np.std(returns_data, axis=0),   # Volatility
        np.percentile(returns_data, 5, axis=0),  # 5th percentile
        np.percentile(returns_data, 95, axis=0)  # 95th percentile
    ])
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(asset_features)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Compute cluster-specific parameters
    cluster_params = {}
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_returns = returns_data.iloc[:, cluster_mask]
        
        if np.sum(cluster_mask) > 0:
            cluster_params[cluster_id] = {
                'mean': np.mean(cluster_returns, axis=0),
                'cov': np.cov(cluster_returns.T),
                'size': np.sum(cluster_mask)
            }
    
    # Decision variables
    w = cp.Variable(returns_data.shape[1])
    
    # Objective function (weighted by cluster)
    objective = 0
    total_weight = 0
    
    for cluster_id, params in cluster_params.items():
        weight = params['size'] / len(returns_data)
        mean = params['mean']
        cov = params['cov']
        
        # Extract weights for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_weights = w[cluster_indices]
        
        objective += weight * (cp.quad_form(cluster_weights, cov) - gamma * mean @ cluster_weights)
        total_weight += weight
    
    # Normalize objective
    objective = objective / total_weight
    
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
                "n_clusters": n_clusters,
                "cluster_sizes": {k: v['size'] for k, v in cluster_params.items()},
                "method": "clustering"
            }
            
            return weights, obj_value, solver_info
        else:
            raise ValueError(f"Problem is {problem.status}")
            
    except Exception as e:
        warnings.warn(f"Clustering optimization failed: {e}")
        # Fallback to standard optimization
        cov_matrix = np.cov(returns_data.T)
        mean_returns = np.mean(returns_data, axis=0)
        from .relax import solve_relax
        return solve_relax(cov_matrix, mean_returns, gamma) 