# Quant Finance Projects

Projects from a master's-level quant finance curriculum. Each one implements the math from
first principles, then checks it against closed forms, independent reference implementations
and textbook identities.

| # | Project | Subject | Status |
|---|---|---|---|
| 1 | Option pricing engine (Black-Scholes, binomial, Monte Carlo, Greeks) | Derivatives, stochastic calculus | Planned |
| 2 | Implied volatility surface | Volatility modeling | Planned |
| 3 | **[Portfolio optimizer](docs/portfolio-optimizer.md)** (Markowitz, Ledoit-Wolf, Black-Litterman, risk parity, HRP) | Portfolio theory | **Done** |
| 4 | VaR and Expected Shortfall engine | Risk management | Planned |
| 5 | Yield curve and bond analytics | Fixed income | Planned |
| 6 | GARCH volatility forecasting | Financial econometrics | Planned |
| 7 | Pairs trading / statistical arbitrage | Time series, algorithmic trading | Planned |
| 8 | Factor models (Fama-French, Fama-MacBeth) | Asset pricing | Planned |
| 9 | Event-driven backtesting engine | Algorithmic trading | Planned |
| 10 | Delta hedging simulator | Derivatives, risk | Planned |
| 11 | Heston stochastic volatility | Volatility modeling | Planned |
| 12 | Finite-difference PDE solver | Numerical methods | Planned |
| 13 | Exotic options by Monte Carlo | Numerical methods | Planned |
| 14 | Short-rate models (Vasicek, CIR, Hull-White) | Fixed income | Planned |
| 15 | Credit risk (Merton, CDS hazard rates, copulas) | Credit risk | Planned |
| 16 | Market making and optimal execution | Market microstructure | Planned |
| 17 | Machine learning for finance, done properly | Machine learning | Planned |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data]"     # data = yfinance for real market prices
python -m pytest                  # run the test suite
```

Code lives in `src/quantfin/`, one subpackage per project. Tests live in `tests/`.

## Project 3: Portfolio optimizer

```bash
python -m quantfin.portfolio frontier     # efficient frontier + 7 portfolio types on 12 ETFs
python -m quantfin.portfolio backtest     # walk-forward comparison of 9 strategies, net of costs
python -m quantfin.portfolio experiment   # how estimation error hurts each method (known truth)
```

The commands download prices from Yahoo Finance. Add `--source synthetic` to run offline, or
`--source csv --csv prices.csv` to use your own data.

The key lesson, measured in a Monte Carlo with known true parameters: with one year of
daily data, the max-Sharpe optimizer *believes* its portfolio has a Sharpe ratio of 1.83.
Its true Sharpe ratio is 0.45, worse than naive 1/N at 0.54. Ledoit-Wolf shrinkage doesn't
fix that, because the noisy expected returns are the problem. Shrinkage does help when
covariance error dominates. With 100 assets and six months of data, it cuts the excess
risk of the minimum-variance portfolio from +122% to +15%.

Theory, methodology, results and references: [docs/portfolio-optimizer.md](docs/portfolio-optimizer.md).
