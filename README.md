# Robust Portfolio Optimizer

A modular Python toolkit for building and benchmarking a transaction-aware, cardinality-constrained portfolio optimizer.

## Project Structure

```
portfolio-optimizer/
├── data/              # raw & processed data
├── notebooks/         # exploratory analysis & visualization
├── src/               # modular solver code
│   ├── data.py        # data ingestion & cleaning
│   ├── relax.py       # convex relaxation module
│   ├── sqp.py         # SQP solver wrapper
│   ├── bnb.py         # branch-and-bound logic
│   └── utils.py       # shared utilities
├── tests/             # unit tests for each module
├── results/           # benchmarks, plots, reports
├── README.md
└── requirements.txt
```

## Setup

1. **Create & activate virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Data Acquisition

* **YFinance**
  ```python
  import yfinance as yf
  import pandas as pd

  tickers = ["AAPL","MSFT","GOOG","AMZN","TSLA"]
  raw = yf.download(tickers, start="2018-01-01", end="2025-01-01")["Adj Close"]
  returns = raw.pct_change().dropna()
  returns.to_csv("data/returns.csv")
  ```

* **Alternative sources**:
  * Kaggle "S&P 500 Stock Data" CSV
  * Quandl / FRED for indices or macro factors

## Modules

### 1. Convex Relaxation (`src/relax.py`)
* **Function**: `solve_relax(Σ, μ, γ)`
* **Problem**:
  ```
    min_w w^T Σ w - γ μ^T w
    subject to sum(w)=1, w>=0
  ```

### 2. Transaction Costs Extension
* **Objective**:
  ```
    min_w w^T Σ w - γ μ^T w + λ ||w - w_prev||_1
  ```
* **Adjust** `solve_relax` signature to accept `w_prev` and `λ`

### 3. SQP Wrapper (`src/sqp.py`)
* **Function**: `solve_sqp(Σ, μ, γ, λ, w_prev)`
* **Implementation**: SciPy's `minimize(method='SLSQP')` with analytic gradient and constraints

### 4. Cardinality Handling (`src/bnb.py`)
* **Surrogate Relaxation**: auxiliary variables for ℓ₀
* **Branch-and-Bound**: recursive solve + fix fractional components
* **Interface**: `branch_and_bound(Σ, μ, k, w_prev)`

### 5. Benchmarking & Visualization
* **Backtests**: rolling windows, annual rebalancing
* **Comparisons**: CVXOPT, OSQP, genetic algorithms (DEAP)
* **Plots**:
  * Efficient frontier (σ vs. E[R])
  * Turnover vs. risk
  * Cardinality k vs. objective value

## Reporting & Insights

* **Notebooks**: method overview, code snippets, embedded charts
* **Interpretation**:
  * Dual variables & KKT sensitivity
  * Impact of transaction cost parameter λ
  * Cardinality vs. diversification trade-off

## Extensions

* **Stochastic returns**: bootstrap / scenario CVaR
* **Robust optimization**: distributional ambiguity
* **Real-world constraints**: lot sizes, sector caps, leverage

## Usage

```python
from src.data import load_returns
from src.relax import solve_relax
from src.sqp import solve_sqp
from src.bnb import branch_and_bound

# Load data
returns = load_returns("data/returns.csv")
cov_matrix = returns.cov()
mean_returns = returns.mean()

# Solve basic portfolio
weights = solve_relax(cov_matrix.values, mean_returns.values, gamma=1.0)

# Solve with transaction costs
prev_weights = np.ones(len(returns.columns)) / len(returns.columns)
weights = solve_sqp(cov_matrix.values, mean_returns.values, gamma=1.0, lambda_tc=0.01, w_prev=prev_weights)

# Solve with cardinality constraint
weights = branch_and_bound(cov_matrix.values, mean_returns.values, k=10, w_prev=prev_weights)
```

## Testing

```bash
pytest tests/
```

## License

MIT License 