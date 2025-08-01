#!/usr/bin/env python3
"""
Example script demonstrating the Robust Portfolio Optimizer.

This script shows how to use the different modules for portfolio optimization
including basic mean-variance optimization, transaction costs, and cardinality constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.data import prepare_portfolio_data, generate_sample_data
from src.relax import solve_relax, compute_efficient_frontier, solve_min_variance, solve_max_sharpe
from src.sqp import solve_sqp, compare_solvers
from src.bnb import branch_and_bound, validate_cardinality_solution
from src.utils import (
    compute_portfolio_statistics, 
    plot_efficient_frontier, 
    plot_weight_distribution,
    compare_portfolios,
    generate_report
)


def main():
    """Main example function."""
    print("=" * 60)
    print("ROBUST PORTFOLIO OPTIMIZER - EXAMPLE")
    print("=" * 60)
    
    # 1. Prepare data
    print("\n1. Preparing data...")
    cov_matrix, mean_returns = prepare_portfolio_data(use_sample_data=True)
    print(f"Data prepared: {len(mean_returns)} assets")
    
    # 2. Basic portfolio optimization
    print("\n2. Basic portfolio optimization...")
    
    # Equal weight benchmark
    n_assets = len(mean_returns)
    weights_eq = np.ones(n_assets) / n_assets
    stats_eq = compute_portfolio_statistics(weights_eq, cov_matrix, mean_returns)
    
    # Mean-variance optimization
    weights_mv, obj_mv, info_mv = solve_relax(cov_matrix, mean_returns, gamma=1.0)
    stats_mv = compute_portfolio_statistics(weights_mv, cov_matrix, mean_returns)
    
    # Minimum variance portfolio
    weights_minvar, obj_minvar, info_minvar = solve_min_variance(cov_matrix)
    stats_minvar = compute_portfolio_statistics(weights_minvar, cov_matrix, mean_returns)
    
    # Maximum Sharpe ratio portfolio
    weights_maxsharpe, obj_maxsharpe, info_maxsharpe = solve_max_sharpe(
        cov_matrix, mean_returns, risk_free_rate=0.02
    )
    stats_maxsharpe = compute_portfolio_statistics(weights_maxsharpe, cov_matrix, mean_returns)
    
    print(f"Equal Weight - Return: {stats_eq['portfolio_return']:.4f}, Risk: {stats_eq['portfolio_risk']:.4f}")
    print(f"Mean-Variance - Return: {stats_mv['portfolio_return']:.4f}, Risk: {stats_mv['portfolio_risk']:.4f}")
    print(f"Min Variance - Return: {stats_minvar['portfolio_return']:.4f}, Risk: {stats_minvar['portfolio_risk']:.4f}")
    print(f"Max Sharpe - Return: {stats_maxsharpe['portfolio_return']:.4f}, Risk: {stats_maxsharpe['portfolio_risk']:.4f}")
    
    # 3. Transaction cost optimization
    print("\n3. Transaction cost optimization...")
    
    # Previous portfolio (equal weights)
    w_prev = np.ones(n_assets) / n_assets
    
    # Without transaction costs
    weights_no_tc, obj_no_tc, info_no_tc = solve_sqp(
        cov_matrix, mean_returns, gamma=1.0, lambda_tc=0.0, w_prev=w_prev
    )
    stats_no_tc = compute_portfolio_statistics(weights_no_tc, cov_matrix, mean_returns, w_prev)
    
    # With transaction costs
    weights_with_tc, obj_with_tc, info_with_tc = solve_sqp(
        cov_matrix, mean_returns, gamma=1.0, lambda_tc=0.01, w_prev=w_prev
    )
    stats_with_tc = compute_portfolio_statistics(weights_with_tc, cov_matrix, mean_returns, w_prev)
    
    print(f"No TC - Return: {stats_no_tc['portfolio_return']:.4f}, Turnover: {stats_no_tc['turnover']:.4f}")
    print(f"With TC - Return: {stats_with_tc['portfolio_return']:.4f}, Turnover: {stats_with_tc['turnover']:.4f}")
    
    # 4. Cardinality-constrained optimization
    print("\n4. Cardinality-constrained optimization...")
    
    k_values = [5, 10, 15, 20]
    cardinality_results = {}
    
    for k in k_values:
        if k <= n_assets:
            weights_card, obj_card, info_card = branch_and_bound(
                cov_matrix, mean_returns, k=k, w_prev=w_prev, gamma=1.0, lambda_tc=0.01
            )
            stats_card = compute_portfolio_statistics(weights_card, cov_matrix, mean_returns, w_prev)
            validation = validate_cardinality_solution(weights_card, k)
            
            cardinality_results[f"k={k}"] = {
                'weights': weights_card,
                'return': stats_card['portfolio_return'],
                'risk': stats_card['portfolio_risk'],
                'sharpe_ratio': stats_card['sharpe_ratio'],
                'turnover': stats_card['turnover'],
                'cardinality': validation['cardinality'],
                'is_valid': validation['is_valid']
            }
            
            print(f"k={k} - Return: {stats_card['portfolio_return']:.4f}, "
                  f"Risk: {stats_card['portfolio_risk']:.4f}, "
                  f"Cardinality: {validation['cardinality']}")
    
    # 5. Efficient frontier
    print("\n5. Computing efficient frontier...")
    risks, returns, weights_array = compute_efficient_frontier(cov_matrix, mean_returns, n_points=20)
    
    # 6. Solver comparison
    print("\n6. Comparing solvers...")
    comparison_results = compare_solvers(cov_matrix, mean_returns, gamma=1.0, lambda_tc=0.01, w_prev=w_prev)
    
    # 7. Generate comprehensive results
    print("\n7. Generating results...")
    
    # Combine all results
    all_results = {
        'Equal Weight': {
            'weights': weights_eq,
            'return': stats_eq['portfolio_return'],
            'risk': stats_eq['portfolio_risk'],
            'sharpe_ratio': stats_eq['sharpe_ratio'],
            'turnover': stats_eq['turnover']
        },
        'Mean-Variance': {
            'weights': weights_mv,
            'return': stats_mv['portfolio_return'],
            'risk': stats_mv['portfolio_risk'],
            'sharpe_ratio': stats_mv['sharpe_ratio'],
            'turnover': stats_mv['turnover']
        },
        'Min Variance': {
            'weights': weights_minvar,
            'return': stats_minvar['portfolio_return'],
            'risk': stats_minvar['portfolio_risk'],
            'sharpe_ratio': stats_minvar['sharpe_ratio'],
            'turnover': stats_minvar['turnover']
        },
        'Max Sharpe': {
            'weights': weights_maxsharpe,
            'return': stats_maxsharpe['portfolio_return'],
            'risk': stats_maxsharpe['portfolio_risk'],
            'sharpe_ratio': stats_maxsharpe['sharpe_ratio'],
            'turnover': stats_maxsharpe['turnover']
        },
        'No Transaction Costs': {
            'weights': weights_no_tc,
            'return': stats_no_tc['portfolio_return'],
            'risk': stats_no_tc['portfolio_risk'],
            'sharpe_ratio': stats_no_tc['sharpe_ratio'],
            'turnover': stats_no_tc['turnover']
        },
        'With Transaction Costs': {
            'weights': weights_with_tc,
            'return': stats_with_tc['portfolio_return'],
            'risk': stats_with_tc['portfolio_risk'],
            'sharpe_ratio': stats_with_tc['sharpe_ratio'],
            'turnover': stats_with_tc['turnover']
        }
    }
    
    # Add cardinality results
    all_results.update(cardinality_results)
    
    # 8. Generate plots and reports
    print("\n8. Generating visualizations...")
    
    # Plot efficient frontier
    plot_efficient_frontier(risks, returns, weights_array, save_path="results/efficient_frontier.png")
    
    # Plot weight distributions for different strategies
    strategies_to_plot = ['Equal Weight', 'Mean-Variance', 'Min Variance', 'Max Sharpe']
    for strategy in strategies_to_plot:
        if strategy in all_results:
            plot_weight_distribution(
                all_results[strategy]['weights'],
                save_path=f"results/weights_{strategy.lower().replace(' ', '_')}.png"
            )
    
    # Compare portfolios
    compare_portfolios(all_results, save_path="results/portfolio_comparison.png")
    
    # Generate report
    report = generate_report(all_results, save_path="results/portfolio_report.txt")
    print("\n" + report)
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("Check the 'results/' directory for generated plots and reports.")
    print("=" * 60)


if __name__ == "__main__":
    main() 