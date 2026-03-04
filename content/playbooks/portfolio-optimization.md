---
slug: portfolio-optimization
title: "Portfolio Optimization"
time: "20 min"
color: green
desc: "Convex optimization for portfolio allocation with cvxpy"
tags: [data science, finance]
spark: "Portfolio Optimization"
category: data-science
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Use Python's scientific stack to perform convex optimization for financial portfolio allocation. This is a cross-platform workflow that runs on Apple Silicon with no GPU requirements.

## Prerequisites

- macOS (any version)
- Python 3.9+
- Internet connection for data download

## Time & risk

- **Duration:** 20 minutes
- **Risk level:** None

<!-- tab: Setup -->
## Install dependencies

```bash
pip install numpy scipy pandas yfinance cvxpy matplotlib
```

<!-- tab: Examples -->
## Mean-variance portfolio optimization

```python
import numpy as np
import cvxpy as cp
import yfinance as yf

# Download historical data
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]
data = yf.download(tickers, start="2023-01-01", end="2025-01-01")["Close"]
returns = data.pct_change().dropna()

# Compute expected returns and covariance
mu = returns.mean().values * 252  # annualized
sigma = returns.cov().values * 252  # annualized covariance

# Define optimization problem
w = cp.Variable(len(tickers))
ret = mu @ w
risk = cp.quad_form(w, sigma)

# Maximize Sharpe ratio (maximize return - 0.5 * risk)
prob = cp.Problem(
    cp.Maximize(ret - 0.5 * risk),
    [cp.sum(w) == 1, w >= 0]
)
prob.solve()
print(dict(zip(tickers, np.round(w.value, 4))))
```

## Efficient frontier

```python
import matplotlib.pyplot as plt

risks, returns_list = [], []
for target_return in np.linspace(mu.min(), mu.max(), 50):
    w = cp.Variable(len(tickers))
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, sigma)),
        [mu @ w == target_return, cp.sum(w) == 1, w >= 0]
    )
    prob.solve()
    if prob.status == "optimal":
        risks.append(np.sqrt(cp.quad_form(w, sigma).value))
        returns_list.append(target_return)

plt.plot(risks, returns_list)
plt.xlabel("Risk (std dev)")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier")
plt.savefig("efficient_frontier.png")
```

<!-- tab: Extensions -->
## Risk constraints

```python
# Maximum risk constraint
max_risk = 0.15  # 15% annual volatility
w = cp.Variable(len(tickers))
prob = cp.Problem(
    cp.Maximize(mu @ w),
    [
        cp.sum(w) == 1,
        w >= 0,
        cp.quad_form(w, sigma) <= max_risk**2
    ]
)
prob.solve()
```

## Sector constraints

```python
# Limit tech sector (indices 0, 2) to 40%
w = cp.Variable(len(tickers))
prob = cp.Problem(
    cp.Maximize(ret - 0.5 * risk),
    [
        cp.sum(w) == 1,
        w >= 0,
        w[0] + w[2] <= 0.4,  # tech cap
    ]
)
```
