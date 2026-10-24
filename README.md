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
