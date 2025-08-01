#!/usr/bin/env python3
"""
Real Market Portfolio Optimization Example

This script demonstrates the portfolio optimizer using real market data
from yfinance, including popular stocks and ETFs.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.data import prepare_portfolio_data, download_stock_data, compute_returns
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
    """Main function using real market data."""
    print("=" * 60)
    print("REAL MARKET PORTFOLIO OPTIMIZATION")
    print("=" * 60)
    
    # Define real stock tickers
    tech_stocks = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "NFLX"]
    financial_stocks = ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK"]
    healthcare_stocks = ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR"]
    
    # Combine all stocks
    all_tickers = tech_stocks + financial_stocks + healthcare_stocks
    print(f"\nUsing {len(all_tickers)} real stocks:")
    print(f"Tech: {tech_stocks}")
    print(f"Financial: {financial_stocks}")
    print(f"Healthcare: {healthcare_stocks}")
    
    # 1. Download real market data
    print("\n1. Downloading real market data...")
    try:
        # Download data for the last 2 years
        prices = download_stock_data(
            tickers=all_tickers,
            start_date="2022-01-01",
            end_date="2024-01-01",
            save_path="data/real_market_data.csv"
        )
        
        # Compute returns
        returns = compute_returns(prices, method="pct_change")
        returns.to_csv("data/real_market_returns.csv")
        
        print(f"Downloaded {len(returns)} days of data for {len(returns.columns)} stocks")
        print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Falling back to sample data...")
        prices = None
        returns = None
    
    # 2. Prepare portfolio data
    print("\n2. Preparing portfolio data...")
    if returns is not None:
        # Use real data
        cov_matrix, mean_returns = prepare_portfolio_data(
            tickers=None,  # Don't download again
            use_sample_data=False,
            file_path="data/real_market_returns.csv"
        )
    else:
        # Fallback to sample data
        cov_matrix, mean_returns = prepare_portfolio_data(use_sample_data=True)
    
    print(f"Data prepared: {len(mean_returns)} assets")
    print(f"Mean return range: [{mean_returns.min():.4f}, {mean_returns.max():.4f}]")
    print(f"Volatility range: [{np.sqrt(np.diag(cov_matrix)).min():.4f}, {np.sqrt(np.diag(cov_matrix)).max():.4f}]")
    
    # 3. Basic portfolio optimization
    print("\n3. Basic portfolio optimization...")
    
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
    
    # 4. Transaction cost optimization
    print("\n4. Transaction cost optimization...")
    
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
    
    # 5. Cardinality-constrained optimization
    print("\n5. Cardinality-constrained optimization...")
    
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
    
    # 6. Efficient frontier
    print("\n6. Computing efficient frontier...")
    risks, returns, weights_array = compute_efficient_frontier(cov_matrix, mean_returns, n_points=20)
    
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
    plot_efficient_frontier(risks, returns, weights_array, save_path="results/real_market_efficient_frontier.png")
    
    # Plot weight distributions for different strategies
    strategies_to_plot = ['Equal Weight', 'Mean-Variance', 'Min Variance', 'Max Sharpe']
    for strategy in strategies_to_plot:
        if strategy in all_results:
            plot_weight_distribution(
                all_results[strategy]['weights'],
                asset_names=all_tickers if len(all_tickers) == len(all_results[strategy]['weights']) else None,
                save_path=f"results/real_market_weights_{strategy.lower().replace(' ', '_')}.png"
            )
    
    # Compare portfolios
    compare_portfolios(all_results, save_path="results/real_market_portfolio_comparison.png")
    
    # Generate report
    report = generate_report(all_results, save_path="results/real_market_portfolio_report.txt")
    print("\n" + report)
    
    # 9. Show top holdings for each strategy
    print("\n9. Top holdings analysis:")
    
    for strategy_name, strategy_data in all_results.items():
        if 'weights' in strategy_data:
            weights = strategy_data['weights']
            # Get top 5 holdings
            top_indices = np.argsort(weights)[-5:][::-1]
            top_weights = weights[top_indices]
            
            print(f"\n{strategy_name} - Top 5 Holdings:")
            for i, (idx, weight) in enumerate(zip(top_indices, top_weights)):
                ticker = all_tickers[idx] if idx < len(all_tickers) else f"Asset_{idx}"
                print(f"  {i+1}. {ticker}: {weight:.4f} ({weight*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("REAL MARKET ANALYSIS COMPLETED!")
    print("Check the 'results/' directory for generated plots and reports.")
    print("=" * 60)


if __name__ == "__main__":
    main() 