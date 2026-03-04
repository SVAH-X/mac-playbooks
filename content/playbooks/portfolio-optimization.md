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

Portfolio optimization finds the allocation of capital across assets that maximizes expected return for a given level of risk — or equivalently, minimizes risk for a given target return. This is the Markowitz mean-variance framework (1952), and it reduces to a quadratic program (QP): minimize **w**ᵀΣ**w** subject to **μ**ᵀ**w** ≥ target, Σwᵢ = 1, and wᵢ ≥ 0. `cvxpy` provides a clean Python interface to convex solvers (OSQP, SCS) that solve these QPs efficiently and reliably on your Mac — no GPU required.

## What you'll accomplish

A working portfolio optimizer that downloads real historical stock data via `yfinance`, computes annualized returns and the covariance matrix, solves for the maximum Sharpe ratio portfolio and the minimum variance portfolio, plots the efficient frontier, and saves a `matplotlib` visualization — all running locally with no cloud API.

## What to know before starting

- **Mean-variance optimization** — The Markowitz framework says a rational investor cares only about a portfolio's expected return and variance. Given those two numbers for every asset and every pair of assets, we can find the "best" portfolio mathematically.
- **Covariance matrix (Σ)** — A square matrix where the diagonal entry for asset i is its variance (risk squared) and the off-diagonal entry for assets i,j is their covariance (how much they move together). Diversification works because negatively correlated assets reduce overall variance.
- **Sharpe ratio** — (Portfolio return − Risk-free rate) / Portfolio volatility. The ratio measures return per unit of risk. The maximum-Sharpe portfolio is the one tangent to the efficient frontier from the risk-free rate.
- **Convex optimization** — A problem is convex if the objective and feasible set are both convex. The key property: any local minimum is the global minimum. `cvxpy` checks your problem is convex (DCP rules) before solving, so the solution is guaranteed optimal.
- **Efficient frontier** — The curve of portfolios that achieve maximum return for each level of risk. No portfolio above the frontier is feasible; any portfolio below is suboptimal (you could do better).
- **cvxpy DCP rules** — Disciplined Convex Programming requires your objective and constraints to be built from DCP-compliant expressions. `cp.quad_form(w, sigma)` is convex; `cp.sum(w) == 1` is affine. If `cvxpy` raises a DCP error, the problem as written is not recognized as convex.

## Prerequisites

- macOS (any version)
- Python 3.9+
- Internet connection for yfinance data download

## Time & risk

- **Duration:** 20 minutes
- **Risk level:** None — pure Python, no system changes
- **Rollback:** Delete the virtual environment; nothing is installed system-wide

<!-- tab: Setup -->
## Step 1: Install the scientific stack

Each library has a specific role. Installing them together avoids version conflicts.

```bash
pip install numpy scipy pandas yfinance cvxpy matplotlib
# numpy  — fast array math (matrix multiply, eigenvalues)
# scipy  — scientific computing (used internally by cvxpy solvers)
# pandas — DataFrame for time-series price data
# yfinance — Yahoo Finance API client; downloads OHLCV data
# cvxpy  — convex optimization modeling language; calls OSQP/SCS solvers
# matplotlib — 2D plotting for the efficient frontier chart
```

Verify `cvxpy` can find its solvers:

```python
import cvxpy as cp
print(cp.installed_solvers())
# Expected: ['CLARABEL', 'OSQP', 'SCS', ...] — at least one must appear
```

## Step 2: Download historical price data

`yfinance.download()` fetches daily OHLCV data from Yahoo Finance. We use closing prices and convert to daily returns.

```python
import yfinance as yf
import pandas as pd

# Five large-cap stocks across different sectors for diversification
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]

# Download two years of daily close prices
data = yf.download(tickers, start="2023-01-01", end="2025-01-01")["Close"]
# data shape: (~500 rows × 5 columns) — one close price per ticker per trading day

# Daily return = (price_today - price_yesterday) / price_yesterday
returns = data.pct_change().dropna()
# dropna() removes row 0, which has no previous day to compare against
print(f"Returns shape: {returns.shape}")  # e.g. (499, 5)
print(returns.describe())                 # sanity check: mean ~0, std ~0.01-0.03
```

## Step 3: Compute annualized statistics

We convert daily statistics to annual by multiplying by the number of trading days (252). This makes the numbers interpretable as yearly return percentages.

```python
import numpy as np

n = len(tickers)

# Annualized expected return vector (shape: n,)
mu = returns.mean().values * 252
# Each entry: expected annual return for that stock (e.g. 0.25 = 25% per year)

# Annualized covariance matrix (shape: n × n)
sigma = returns.cov().values * 252
# Diagonal: variance of each stock (risk²)
# Off-diagonal: covariance between pairs — positive = move together, negative = hedge

print("Expected annual returns:", dict(zip(tickers, np.round(mu, 3))))
print("Covariance matrix condition number:", np.linalg.cond(sigma))
# Condition number >> 1000 means near-singular — assets are highly correlated
```

## Step 4: Set up the optimization problem

`cvxpy` lets you write the math almost directly as Python. The Variable represents the portfolio weights we're solving for.

```python
import cvxpy as cp

# Decision variable: portfolio weight for each asset
w = cp.Variable(n)
# w[i] will be the fraction of capital allocated to ticker[i]

# Portfolio return (scalar): dot product of weights and expected returns
portfolio_return = mu @ w

# Portfolio variance (scalar): w^T * Sigma * w — the quadratic form
portfolio_variance = cp.quad_form(w, sigma)
# quad_form is DCP-compliant and recognized as convex by cvxpy

# Constraints
constraints = [
    cp.sum(w) == 1,  # weights must sum to 100% (fully invested)
    w >= 0,          # no short selling — each weight is non-negative
]

# The optimization problem is now fully specified
# We'll solve different objectives in the Examples tab
print("Problem is DCP:", cp.Problem(cp.Minimize(portfolio_variance), constraints).is_dcp())
# Must print True — if False, cvxpy won't solve it
```

<!-- tab: Examples -->
## Step 1: Maximum Sharpe ratio portfolio

The maximum Sharpe ratio portfolio is the Markowitz "optimal" portfolio — the one that gives the best return per unit of risk. We maximize (return − risk_free_rate) / volatility, which is equivalent to maximizing return − λ·variance for the right λ.

```python
import numpy as np
import cvxpy as cp
import yfinance as yf

tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]
data = yf.download(tickers, start="2023-01-01", end="2025-01-01")["Close"]
returns = data.pct_change().dropna()
mu = returns.mean().values * 252
sigma = returns.cov().values * 252
n = len(tickers)

risk_free_rate = 0.05  # ~5% annual (approximate US T-bill rate)

w = cp.Variable(n)
# Maximize Sharpe: equivalently maximize (mu^T w - rf) / sqrt(w^T Sigma w)
# We parameterize by fixing denominator = 1 and maximizing numerator
# This is the Sharpe-maximizing QP via the Markowitz reformulation
sharpe_problem = cp.Problem(
    cp.Maximize(mu @ w - risk_free_rate),
    [cp.sum(w) == 1, w >= 0, cp.quad_form(w, sigma) <= 1]
)
sharpe_problem.solve(solver=cp.OSQP)

weights = w.value
port_return = mu @ weights
port_vol = np.sqrt(weights @ sigma @ weights)
sharpe = (port_return - risk_free_rate) / port_vol

print("Maximum Sharpe Portfolio:")
for ticker, weight in zip(tickers, weights):
    print(f"  {ticker}: {weight:.1%}")
print(f"  Expected return: {port_return:.1%}")
print(f"  Volatility:      {port_vol:.1%}")
print(f"  Sharpe ratio:    {sharpe:.2f}")
# Sharpe > 1.0 is good; > 2.0 is excellent for a 2-year window
```

## Step 2: Minimum variance portfolio

The most conservative allocation — minimize risk regardless of expected return. This is the leftmost point on the efficient frontier.

```python
w_mv = cp.Variable(n)
min_var_problem = cp.Problem(
    cp.Minimize(cp.quad_form(w_mv, sigma)),  # minimize portfolio variance
    [cp.sum(w_mv) == 1, w_mv >= 0]
)
min_var_problem.solve()

mv_weights = w_mv.value
mv_vol = np.sqrt(mv_weights @ sigma @ mv_weights)
mv_return = mu @ mv_weights

print("Minimum Variance Portfolio:")
for ticker, weight in zip(tickers, mv_weights):
    print(f"  {ticker}: {weight:.1%}")
print(f"  Volatility: {mv_vol:.1%}  Return: {mv_return:.1%}")
# Expect high allocation to the least volatile, least correlated stocks
```

## Step 3: Plot the efficient frontier

The efficient frontier is traced by solving the minimum-variance problem at each target return level. Every point on the curve is a portfolio you cannot improve without accepting more risk.

```python
import matplotlib.pyplot as plt

target_returns = np.linspace(mu.min(), mu.max(), 60)
frontier_vols = []
frontier_rets = []

for target in target_returns:
    w_f = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w_f, sigma)),
        [
            mu @ w_f == target,  # fix return at this target level
            cp.sum(w_f) == 1,
            w_f >= 0,
        ]
    )
    prob.solve()
    if prob.status == "optimal":
        frontier_vols.append(np.sqrt(cp.quad_form(w_f, sigma).value))
        frontier_rets.append(target)

plt.figure(figsize=(8, 5))
plt.plot(frontier_vols, frontier_rets, "b-", lw=2, label="Efficient Frontier")
plt.scatter([port_vol], [port_return], color="red", s=100, zorder=5, label="Max Sharpe")
plt.scatter([mv_vol], [mv_return], color="green", s=100, zorder=5, label="Min Variance")
plt.xlabel("Annualized Volatility (Risk)")
plt.ylabel("Annualized Expected Return")
plt.title("Efficient Frontier — 5 Large-Cap Stocks")
plt.legend()
plt.tight_layout()
plt.savefig("efficient_frontier.png", dpi=150)
print("Saved efficient_frontier.png")
# Open the file — the curve bows to the left (diversification benefit)
```

## Step 4: Risk-constrained optimization

Limit the portfolio to 15% annual volatility maximum. Useful when you have a mandate like "target return, subject to VaR limit."

```python
max_vol = 0.15  # 15% annual volatility ceiling

w_rc = cp.Variable(n)
risk_constrained = cp.Problem(
    cp.Maximize(mu @ w_rc),                         # maximize return...
    [
        cp.sum(w_rc) == 1,
        w_rc >= 0,
        cp.quad_form(w_rc, sigma) <= max_vol**2,    # ...subject to vol <= 15%
    ]
)
risk_constrained.solve()

rc_vol = np.sqrt(w_rc.value @ sigma @ w_rc.value)
print(f"Risk-constrained portfolio vol: {rc_vol:.1%} (limit: {max_vol:.1%})")
# If unconstrained max-return portfolio has vol > 15%, the constraint will bind
```

## Step 5: Black-Litterman model (advanced)

The Black-Litterman model lets you blend market equilibrium returns with your own views. The prior comes from market-cap weights; you express views as a matrix P and view returns q.

```python
# Market-cap weights (example; in practice pull from market data)
w_market = np.array([0.25, 0.20, 0.25, 0.18, 0.12])

# Implied equilibrium returns: pi = delta * Sigma * w_market
delta = 2.5  # risk aversion coefficient (typical value)
pi = delta * sigma @ w_market

# View: NVDA will outperform AAPL by 5% (one relative view)
P = np.zeros((1, n))
P[0, tickers.index("NVDA")] = 1
P[0, tickers.index("AAPL")] = -1
q = np.array([0.05])

# Uncertainty in views (tau * Sigma scales the prior)
tau = 0.05
omega = tau * P @ sigma @ P.T  # diagonal matrix of view variances

# Black-Litterman posterior return
M = np.linalg.inv(np.linalg.inv(tau * sigma) + P.T @ np.linalg.inv(omega) @ P)
mu_bl = M @ (np.linalg.inv(tau * sigma) @ pi + P.T @ np.linalg.inv(omega) @ q)

print("BL posterior returns:", dict(zip(tickers, np.round(mu_bl, 3))))
# Now optimize using mu_bl instead of mu — your views are incorporated
```

<!-- tab: Extensions -->
## Sector constraints

Cap the technology sector to 40% of the portfolio. Define sector membership as a binary matrix and add a linear constraint.

```python
# Indices: AAPL=0, GOOGL=1, MSFT=2, AMZN=3, NVDA=4
# Tech sector: AAPL, MSFT, NVDA
tech_indices = [0, 2, 4]

w_s = cp.Variable(n)
sector_prob = cp.Problem(
    cp.Maximize(mu @ w_s - 0.5 * cp.quad_form(w_s, sigma)),
    [
        cp.sum(w_s) == 1,
        w_s >= 0,
        cp.sum(w_s[tech_indices]) <= 0.40,   # tech cap at 40%
        w_s <= 0.30,                          # single-stock cap at 30%
    ]
)
sector_prob.solve()
print(dict(zip(tickers, np.round(w_s.value, 3))))
```

## Transaction cost penalty

Add L1 regularization to penalize large trades away from a current portfolio. Useful for rebalancing — favors smaller changes.

```python
w_current = np.array([0.20, 0.20, 0.20, 0.20, 0.20])  # current equal-weight
lambda_tc = 0.001  # transaction cost weight (tune based on actual costs)

w_tc = cp.Variable(n)
tc_prob = cp.Problem(
    cp.Maximize(mu @ w_tc - 0.5 * cp.quad_form(w_tc, sigma)
                - lambda_tc * cp.norm1(w_tc - w_current)),  # penalize trades
    [cp.sum(w_tc) == 1, w_tc >= 0]
)
tc_prob.solve()
# Result: weights will shift from equal-weight but stay closer than unconstrained
```

## Monte Carlo simulation for return uncertainty

Instead of using point estimates for expected returns, sample from a distribution and solve the optimization many times to understand sensitivity.

```python
n_simulations = 200
sharpe_ratios = []
weight_samples = []

for _ in range(n_simulations):
    # Sample expected returns from a multivariate normal (uncertainty in mu)
    mu_sample = np.random.multivariate_normal(mu, sigma / 10)
    w_mc = cp.Variable(n)
    prob_mc = cp.Problem(
        cp.Maximize(mu_sample @ w_mc - 0.5 * cp.quad_form(w_mc, sigma)),
        [cp.sum(w_mc) == 1, w_mc >= 0]
    )
    prob_mc.solve()
    if prob_mc.status == "optimal":
        weight_samples.append(w_mc.value)

weight_array = np.array(weight_samples)
print("Weight mean across simulations:", np.round(weight_array.mean(axis=0), 3))
print("Weight std (sensitivity):", np.round(weight_array.std(axis=0), 3))
# High std on a ticker means the optimal weight is sensitive to return estimates
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `yfinance` returns empty DataFrame | Rate limiting or ticker delisted | Add `time.sleep(1)` between downloads; verify ticker on finance.yahoo.com |
| `cvxpy` raises `DCPError` | Problem is not DCP-compliant (e.g., dividing by variable) | Reformulate: use `quad_form` for quadratic terms; check cvxpy DCP rules docs |
| `prob.status == "infeasible"` | Target return is outside feasible range | Check `mu.min()` and `mu.max()`; the target must lie within this range |
| Covariance matrix is singular | Two assets are perfectly correlated | Add Tikhonov regularization: `sigma += 1e-4 * np.eye(n)` |
| Weights near 0/1 (very concentrated) | No diversification constraint | Add `w <= 0.30` (max 30% per asset) to constraints |
| Negative weights despite `w >= 0` | Numerical solver tolerance | Use `np.maximum(w.value, 0); renormalize` to clip negatives |
| `solve()` very slow (> 30s) | Default solver struggling | Try `solver=cp.SCS` or `solver=cp.CLARABEL` explicitly |

### cvxpy DCP rules explained

cvxpy enforces Disciplined Convex Programming — every expression must be provably convex. The most common violations:

- **Minimizing a concave function**: `cp.Minimize(cp.log(x))` — `log` is concave, so you must `cp.Maximize` it
- **Multiplying two variables**: `x * y` is not convex in general — use `cp.multiply` with a constant, or reformulate
- **Non-affine equality constraint**: `cp.quad_form(w, sigma) == c` — quadratic equality is not convex (it's the intersection of two non-convex sets)

### Singular covariance matrix

When assets have near-perfect correlation (e.g., two ETFs tracking the same index), the covariance matrix becomes singular (determinant ≈ 0) and the solver may fail or return extreme weights. Fix with Tikhonov regularization, which adds a tiny amount to the diagonal:

```python
epsilon = 1e-4  # regularization strength — increase if still singular
sigma_reg = sigma + epsilon * np.eye(n)
# This ensures all eigenvalues are >= epsilon (positive definite)
# The optimization result changes negligibly for small epsilon
```

### yfinance rate limiting

Yahoo Finance limits request frequency. For portfolios with many tickers:

```python
import time
all_data = {}
for ticker in tickers:
    all_data[ticker] = yf.download(ticker, start="2023-01-01", end="2025-01-01")["Close"]
    time.sleep(0.5)  # 500ms pause between requests
data = pd.DataFrame(all_data)
```
