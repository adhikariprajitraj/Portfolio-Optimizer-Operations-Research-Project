#!/usr/bin/env python3
"""
Debug script to understand yfinance data structure
"""

import yfinance as yf
import pandas as pd

def debug_yfinance():
    """Debug yfinance data structure."""
    ticker = "AAPL"
    
    print(f"Downloading data for {ticker}...")
    
    try:
        # Download data
        data = yf.download(ticker, start="2022-01-01", end="2024-01-01", progress=False)
        
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")
        print(f"Data head:\n{data.head()}")
        
        # Check if 'Adj Close' exists
        if 'Adj Close' in data.columns:
            print("'Adj Close' column found!")
            prices = data['Adj Close']
            print(f"Prices shape: {prices.shape}")
            print(f"Prices head:\n{prices.head()}")
        else:
            print("'Adj Close' column not found!")
            print("Available columns:", data.columns.tolist())
            
            # Try using 'Close' instead
            if 'Close' in data.columns:
                print("Using 'Close' column instead...")
                prices = data['Close']
                print(f"Prices shape: {prices.shape}")
                print(f"Prices head:\n{prices.head()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_yfinance() 