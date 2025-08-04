"""
Advanced Portfolio Optimization Example
Demonstrates robust optimization, ML-enhanced methods, and advanced constraints.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings

# Import our modules
from src.data import generate_sample_data, prepare_portfolio_data
from src.relax import solve_relax, solve_min_variance, solve_max_sharpe
from src.bnb import branch_and_bound
from src.utils import compute_portfolio_statistics, plot_efficient_frontier, compare_portfolios
from src.robust_optimization import (
    worst_case_optimization, distributional_robust_optimization,
    stress_test_portfolio, robust_covariance_estimation,
    conditional_value_at_risk_optimization, regime_dependent_optimization
)
from src.ml_enhanced import (
    factor_model_optimization, neural_network_optimization,
    ensemble_optimization, clustering_optimization
)
from src.advanced_constraints import (
    sector_constrained_optimization, leverage_constrained_optimization,
    concentration_constrained_optimization, tracking_error_constrained_optimization
)


def run_advanced_example():
    """Run comprehensive advanced portfolio optimization example."""
    
    print("=" * 60)
    print("ADVANCED PORTFOLIO OPTIMIZATION EXAMPLE")
    print("=" * 60)
    
    # 1. Generate comprehensive data
    print("\n1. Preparing advanced data...")
    returns_data = generate_sample_data(n_assets=50, n_periods=500)
    cov_matrix, mean_returns = prepare_portfolio_data(returns_data)
    asset_names = [f"Asset_{i}" for i in range(len(mean_returns))]
    
    print(f"Data prepared: {len(asset_names)} assets, {returns_data.shape[0]} periods")
    print(f"Mean return range: [{mean_returns.min():.4f}, {mean_returns.max():.4f}]")
    print(f"Volatility range: [{np.sqrt(np.diag(cov_matrix)).min():.4f}, {np.sqrt(np.diag(cov_matrix)).max():.4f}]")
    
    # 2. Robust Optimization Methods
    print("\n2. Robust Optimization Methods...")
    
    # Worst-case optimization
    uncertainty_set = {
        'return_uncertainty': 0.1,
        'covariance_uncertainty': 0.2
    }
    weights_worst_case, obj_worst_case, info_worst_case = worst_case_optimization(
        cov_matrix, mean_returns, uncertainty_set, gamma=1.0
    )
    
    # CVaR optimization
    weights_cvar, obj_cvar, info_cvar = conditional_value_at_risk_optimization(
        returns_data, alpha=0.05, gamma=1.0
    )
    
    print(f"Worst-case - Return: {mean_returns @ weights_worst_case:.4f}, Risk: {np.sqrt(weights_worst_case @ cov_matrix @ weights_worst_case):.4f}")
    print(f"CVaR - Return: {mean_returns @ weights_cvar:.4f}, Risk: {np.sqrt(weights_cvar @ cov_matrix @ weights_cvar):.4f}")
    
    # 3. Machine Learning Enhanced Methods
    print("\n3. Machine Learning Enhanced Methods...")
    
    # Factor model optimization
    weights_factor, obj_factor, info_factor = factor_model_optimization(
        returns_data, n_factors=5, gamma=1.0
    )
    
    # Neural network optimization
    weights_nn, obj_nn, info_nn = neural_network_optimization(
        returns_data, gamma=1.0, hidden_layers=(30, 15)
    )
    
    # Ensemble optimization
    weights_ensemble, obj_ensemble, info_ensemble = ensemble_optimization(
        returns_data, methods=['mean_variance', 'factor_model'], gamma=1.0
    )
    
    # Clustering optimization
    weights_cluster, obj_cluster, info_cluster = clustering_optimization(
        returns_data, n_clusters=6, gamma=1.0
    )
    
    print(f"Factor Model - Return: {mean_returns @ weights_factor:.4f}, Risk: {np.sqrt(weights_factor @ cov_matrix @ weights_factor):.4f}")
    print(f"Neural Network - Return: {mean_returns @ weights_nn:.4f}, Risk: {np.sqrt(weights_nn @ cov_matrix @ weights_nn):.4f}")
    print(f"Ensemble - Return: {mean_returns @ weights_ensemble:.4f}, Risk: {np.sqrt(weights_ensemble @ cov_matrix @ weights_ensemble):.4f}")
    print(f"Clustering - Return: {mean_returns @ weights_cluster:.4f}, Risk: {np.sqrt(weights_cluster @ cov_matrix @ weights_cluster):.4f}")
    
    # 4. Advanced Constraints
    print("\n4. Advanced Constraints...")
    
    # Sector constraints
    sector_labels = ['TECH'] * 10 + ['FIN'] * 10 + ['HEALTH'] * 10 + ['CONS'] * 10 + ['IND'] * 10
    sector_limits = {'TECH': 0.3, 'FIN': 0.25, 'HEALTH': 0.25, 'CONS': 0.2, 'IND': 0.2}
    
    weights_sector, obj_sector, info_sector = sector_constrained_optimization(
        cov_matrix, mean_returns, sector_labels, sector_limits, gamma=1.0
    )
    
    # Leverage constraints
    weights_leverage, obj_leverage, info_leverage = leverage_constrained_optimization(
        cov_matrix, mean_returns, max_leverage=1.3, gamma=1.0
    )
    
    # Concentration constraints
    weights_concentration, obj_concentration, info_concentration = concentration_constrained_optimization(
        cov_matrix, mean_returns, max_concentration=0.08, gamma=1.0
    )
    
    # Tracking error constraints
    benchmark_weights = np.ones(len(mean_returns)) / len(mean_returns)  # Equal weight benchmark
    weights_tracking, obj_tracking, info_tracking = tracking_error_constrained_optimization(
        cov_matrix, mean_returns, benchmark_weights, max_tracking_error=0.03, gamma=1.0
    )
    
    print(f"Sector Constrained - Return: {mean_returns @ weights_sector:.4f}, Risk: {np.sqrt(weights_sector @ cov_matrix @ weights_sector):.4f}")
    print(f"Leverage Constrained - Return: {mean_returns @ weights_leverage:.4f}, Risk: {np.sqrt(weights_leverage @ cov_matrix @ weights_leverage):.4f}")
    print(f"Concentration Constrained - Return: {mean_returns @ weights_concentration:.4f}, Risk: {np.sqrt(weights_concentration @ cov_matrix @ weights_concentration):.4f}")
    print(f"Tracking Error Constrained - Return: {mean_returns @ weights_tracking:.4f}, Risk: {np.sqrt(weights_tracking @ cov_matrix @ weights_tracking):.4f}")
    
    # 5. Stress Testing
    print("\n5. Stress Testing...")
    
    # Define stress scenarios
    stress_scenarios = [
        {'description': 'Market Crash', 'return_shock': -0.3, 'covariance_shock': 0.5},
        {'description': 'Volatility Spike', 'covariance_shock': 0.8},
        {'description': 'Sector Rotation', 'return_shock': 0.2},
        {'description': 'Liquidity Crisis', 'return_shock': -0.15, 'covariance_shock': 0.3}
    ]
    
    # Test a portfolio under stress
    test_portfolio = weights_ensemble  # Use ensemble portfolio for stress testing
    stress_results = stress_test_portfolio(
        test_portfolio, cov_matrix, mean_returns, stress_scenarios
    )
    
    print("Stress Test Results:")
    for scenario, results in stress_results.items():
        print(f"  {results['description']}: Return={results['return']:.4f}, Risk={results['risk']:.4f}, Sharpe={results['sharpe']:.4f}")
    
    # 6. Robust Covariance Estimation
    print("\n6. Robust Covariance Estimation...")
    
    # Compare different covariance estimation methods
    methods = ['empirical', 'ledoit_wolf', 'min_cov_det']
    robust_results = {}
    
    for method in methods:
        robust_cov, robust_mean = robust_covariance_estimation(returns_data, method=method)
        weights_robust, obj_robust, info_robust = solve_relax(robust_cov, robust_mean, gamma=1.0)
        
        robust_results[method] = {
            'weights': weights_robust,
            'return': robust_mean @ weights_robust,
            'risk': np.sqrt(weights_robust @ robust_cov @ weights_robust)
        }
        
        print(f"{method.title()} - Return: {robust_results[method]['return']:.4f}, Risk: {robust_results[method]['risk']:.4f}")
    
    # 7. Regime-Dependent Optimization
    print("\n7. Regime-Dependent Optimization...")
    
    # Create regime indicators (simplified)
    regime_indicators = np.zeros(len(returns_data))
    regime_indicators[len(returns_data)//3:2*len(returns_data)//3] = 1  # Middle third is regime 1
    regime_indicators[2*len(returns_data)//3:] = 2  # Last third is regime 2
    
    weights_regime, obj_regime, info_regime = regime_dependent_optimization(
        returns_data, regime_indicators, gamma=1.0
    )
    
    print(f"Regime-Dependent - Return: {mean_returns @ weights_regime:.4f}, Risk: {np.sqrt(weights_regime @ cov_matrix @ weights_regime):.4f}")
    print(f"Regime distribution: {info_regime['regime_weights']}")
    
    # 8. Compile Results
    print("\n8. Compiling Results...")
    
    # Collect all results
    all_results = {
        'Standard Mean-Variance': {
            'weights': solve_relax(cov_matrix, mean_returns, gamma=1.0)[0],
            'method': 'Standard'
        },
        'Worst-Case Robust': {
            'weights': weights_worst_case,
            'method': 'Robust'
        },

        'CVaR Optimization': {
            'weights': weights_cvar,
            'method': 'Risk'
        },
        'Factor Model': {
            'weights': weights_factor,
            'method': 'ML'
        },
        'Neural Network': {
            'weights': weights_nn,
            'method': 'ML'
        },
        'Ensemble': {
            'weights': weights_ensemble,
            'method': 'ML'
        },
        'Clustering': {
            'weights': weights_cluster,
            'method': 'ML'
        },
        'Sector Constrained': {
            'weights': weights_sector,
            'method': 'Constraint'
        },
        'Leverage Constrained': {
            'weights': weights_leverage,
            'method': 'Constraint'
        },
        'Concentration Constrained': {
            'weights': weights_concentration,
            'method': 'Constraint'
        },
        'Tracking Error Constrained': {
            'weights': weights_tracking,
            'method': 'Constraint'
        },
        'Regime-Dependent': {
            'weights': weights_regime,
            'method': 'Advanced'
        }
    }
    
    # Compute statistics for all methods
    for name, result in all_results.items():
        weights = result['weights']
        stats = compute_portfolio_statistics(weights, cov_matrix, mean_returns)
        result.update(stats)
    
    # 9. Generate Visualizations
    print("\n9. Generating Advanced Visualizations...")
    
    # Create comparison plot
    plt.figure(figsize=(15, 10))
    
    # Color coding by method type
    colors = {'Standard': 'blue', 'Robust': 'red', 'ML': 'green', 'Constraint': 'orange', 'Risk': 'purple', 'Advanced': 'brown'}
    
    for name, result in all_results.items():
        method_type = result['method']
        color = colors.get(method_type, 'gray')
        
        plt.scatter(result['portfolio_risk'], result['portfolio_return'], 
                   c=color, s=100, alpha=0.7, label=f"{name} ({method_type})")
    
    plt.xlabel('Portfolio Risk')
    plt.ylabel('Portfolio Return')
    plt.title('Advanced Portfolio Optimization Methods Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/advanced_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create method type comparison
    method_types = {}
    for name, result in all_results.items():
        method_type = result['method']
        if method_type not in method_types:
            method_types[method_type] = []
        method_types[method_type].append({
            'name': name,
            'return': result['portfolio_return'],
            'risk': result['portfolio_risk'],
            'sharpe': result['sharpe_ratio']
        })
    
    # Plot by method type
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (method_type, methods) in enumerate(method_types.items()):
        if i < len(axes):
            ax = axes[i]
            
            returns = [m['return'] for m in methods]
            risks = [m['risk'] for m in methods]
            names = [m['name'] for m in methods]
            
            ax.scatter(risks, returns, c=colors.get(method_type, 'gray'), s=80, alpha=0.8)
            
            for j, name in enumerate(names):
                ax.annotate(name.split()[0], (risks[j], returns[j]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_xlabel('Risk')
            ax.set_ylabel('Return')
            ax.set_title(f'{method_type} Methods')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/method_types_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 10. Generate Report
    print("\n10. Generating Advanced Report...")
    
    report = generate_advanced_report(all_results, stress_results, robust_results)
    
    with open('results/advanced_portfolio_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + "=" * 60)
    print("ADVANCED PORTFOLIO OPTIMIZATION COMPLETED!")
    print("Check the 'results/' directory for generated plots and reports.")
    print("=" * 60)


def generate_advanced_report(all_results: Dict, stress_results: Dict, robust_results: Dict) -> str:
    """Generate comprehensive advanced portfolio report."""
    
    report = """
ADVANCED PORTFOLIO OPTIMIZATION REPORT
=====================================

This report compares various advanced portfolio optimization methods including:
- Robust optimization techniques
- Machine learning enhanced methods
- Advanced constraint handling
- Stress testing results
- Risk management approaches

METHOD COMPARISON
================
"""
    
    # Sort by Sharpe ratio
    sorted_results = sorted(all_results.items(), 
                           key=lambda x: x[1]['sharpe_ratio'], reverse=True)
    
    for i, (name, result) in enumerate(sorted_results, 1):
        report += f"""
{i}. {name}
   Return: {result['portfolio_return']:.4f}
   Risk: {result['portfolio_risk']:.4f}
   Sharpe Ratio: {result['sharpe_ratio']:.4f}
   Effective N: {result['effective_n']:.2f}
   Max Weight: {result['max_weight']:.4f}
   Method Type: {result['method']}
"""
    
    # Method type summary
    method_summary = {}
    for name, result in all_results.items():
        method_type = result['method']
        if method_type not in method_summary:
            method_summary[method_type] = []
        method_summary[method_type].append(result['sharpe_ratio'])
    
    report += "\nMETHOD TYPE SUMMARY\n==================\n"
    for method_type, sharpes in method_summary.items():
        avg_sharpe = np.mean(sharpes)
        max_sharpe = np.max(sharpes)
        report += f"{method_type}: Avg Sharpe = {avg_sharpe:.4f}, Max Sharpe = {max_sharpe:.4f}\n"
    
    # Stress testing results
    report += "\nSTRESS TESTING RESULTS\n====================\n"
    for scenario, results in stress_results.items():
        report += f"{results['description']}:\n"
        report += f"  Return: {results['return']:.4f}\n"
        report += f"  Risk: {results['risk']:.4f}\n"
        report += f"  Sharpe: {results['sharpe']:.4f}\n\n"
    
    # Robust covariance results
    report += "ROBUST COVARIANCE ESTIMATION\n============================\n"
    for method, results in robust_results.items():
        report += f"{method.title()}:\n"
        report += f"  Return: {results['return']:.4f}\n"
        report += f"  Risk: {results['risk']:.4f}\n\n"
    
    return report


if __name__ == "__main__":
    run_advanced_example() 