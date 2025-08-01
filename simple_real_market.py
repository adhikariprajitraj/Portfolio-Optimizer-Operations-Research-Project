#!/usr/bin/env python3
"""
Simple Real Market Portfolio Optimization

This script demonstrates the portfolio optimizer using real market data
from yfinance with proper error handling.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from src.relax import solve_relax, solve_min_variance, solve_max_sharpe
from src.sqp import solve_sqp
from src.bnb import branch_and_bound
from src.utils import compute_portfolio_statistics, plot_weight_distribution


def download_real_data():
    """Download real market data from yfinance."""
    # Define a smaller set of popular stocks
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "PFE"]
    
    print(f"Downloading data for {len(tickers)} stocks...")
    
    try:
        # Download data for the last 2 years
        data = yf.download(tickers, start="2022-01-01", end="2024-01-01", progress=False)
        
        # Handle the new yfinance data structure
        if len(tickers) == 1:
            # Single ticker case
            if ('Close', tickers[0]) in data.columns:
                prices = data[('Close', tickers[0])]
            else:
                # Fallback to regular column names
                prices = data['Close']
        else:
            # Multiple tickers case - data has tuple columns
            close_columns = [col for col in data.columns if col[0] == 'Close']
            if close_columns:
                prices = data[close_columns]
                # Rename columns to just ticker names
                prices.columns = [col[1] for col in close_columns]
            else:
                # Fallback to regular column names
                prices = data['Close']
        
        # Compute returns
        returns = prices.pct_change().dropna()
        
        print(f"Successfully downloaded {len(returns)} days of data")
        print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
        print(f"Stocks: {list(returns.columns)}")
        
        return returns, list(returns.columns)
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Trying alternative approach...")
        
        # Try downloading one by one
        try:
            all_prices = []
            valid_tickers = []
            
            for ticker in tickers:
                try:
                    stock_data = yf.download(ticker, start="2022-01-01", end="2024-01-01", progress=False)
                    if not stock_data.empty:
                        # Handle the new data structure
                        if ('Close', ticker) in stock_data.columns:
                            prices = stock_data[('Close', ticker)]
                        else:
                            prices = stock_data['Close']
                        
                        all_prices.append(prices)
                        valid_tickers.append(ticker)
                        print(f"Downloaded {ticker}")
                except Exception as e:
                    print(f"Failed to download {ticker}: {e}")
            
            if all_prices:
                # Combine all prices
                combined_prices = pd.concat(all_prices, axis=1)
                combined_prices.columns = valid_tickers
                
                # Compute returns
                returns = combined_prices.pct_change().dropna()
                
                print(f"Successfully downloaded {len(returns)} days of data for {len(valid_tickers)} stocks")
                print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
                print(f"Stocks: {valid_tickers}")
                
                return returns, valid_tickers
            else:
                print("No data could be downloaded")
                return None, None
                
        except Exception as e2:
            print(f"Alternative approach also failed: {e2}")
            return None, None


def main():
    """Main function using real market data."""
    print("=" * 60)
    print("REAL MARKET PORTFOLIO OPTIMIZATION")
    print("=" * 60)
    
    # Download real data
    returns, tickers = download_real_data()
    
    if returns is None:
        print("Failed to download real data. Exiting.")
        return
    
    # Compute covariance matrix and mean returns
    cov_matrix = returns.cov().values
    mean_returns = returns.mean().values
    
    print(f"\nData prepared: {len(mean_returns)} assets")
    print(f"Mean return range: [{mean_returns.min():.4f}, {mean_returns.max():.4f}]")
    print(f"Volatility range: [{np.sqrt(np.diag(cov_matrix)).min():.4f}, {np.sqrt(np.diag(cov_matrix)).max():.4f}]")
    
    # Basic portfolio optimization
    print("\n1. Basic portfolio optimization...")
    
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
    
    # Transaction cost optimization
    print("\n2. Transaction cost optimization...")
    
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
    
    # Cardinality-constrained optimization
    print("\n3. Cardinality-constrained optimization...")
    
    k_values = [3, 5, 7]
    for k in k_values:
        if k <= n_assets:
            weights_card, obj_card, info_card = branch_and_bound(
                cov_matrix, mean_returns, k=k, w_prev=w_prev, gamma=1.0, lambda_tc=0.01
            )
            stats_card = compute_portfolio_statistics(weights_card, cov_matrix, mean_returns, w_prev)
            
            print(f"k={k} - Return: {stats_card['portfolio_return']:.4f}, "
                  f"Risk: {stats_card['portfolio_risk']:.4f}")
    
    # Show top holdings for each strategy
    print("\n4. Top holdings analysis:")
    
    strategies = [
        ("Equal Weight", weights_eq),
        ("Mean-Variance", weights_mv),
        ("Min Variance", weights_minvar),
        ("Max Sharpe", weights_maxsharpe),
        ("No Transaction Costs", weights_no_tc),
        ("With Transaction Costs", weights_with_tc)
    ]
    
    for strategy_name, weights in strategies:
        # Get top 5 holdings
        top_indices = np.argsort(weights)[-5:][::-1]
        top_weights = weights[top_indices]
        
        print(f"\n{strategy_name} - Top 5 Holdings:")
        for i, (idx, weight) in enumerate(zip(top_indices, top_weights)):
            ticker = tickers[idx] if idx < len(tickers) else f"Asset_{idx}"
            print(f"  {i+1}. {ticker}: {weight:.4f} ({weight*100:.2f}%)")
    
    # Plot weight distributions
    print("\n5. Generating visualizations...")
    
    for strategy_name, weights in strategies:
        plot_weight_distribution(
            weights,
            asset_names=tickers,
            save_path=f"results/real_market_{strategy_name.lower().replace(' ', '_')}.png"
        )
    
    print("\n" + "=" * 60)
    print("REAL MARKET ANALYSIS COMPLETED!")
    print("Check the 'results/' directory for generated plots.")
    print("=" * 60)


if __name__ == "__main__":
    main() 