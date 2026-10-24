"""Project 3: portfolio optimization (Markowitz, Black-Litterman, risk parity, HRP).

Quick tour::

    from quantfin.portfolio import data, estimators, markowitz

    returns = data.prices_to_returns(data.load_prices_yahoo())  # or data.synthetic_market().returns
    mu = estimators.mean_historical(returns)
    cov = estimators.cov_ledoit_wolf(returns)
    weights = markowitz.max_sharpe(mu, cov, rf=0.02)

Modules: ``data``, ``estimators``, ``markowitz``, ``black_litterman``, ``risk_parity``,
``hrp``, ``metrics``, ``backtest``, ``strategies``, ``experiments``, ``plots``, ``cli``.
Command line: ``python -m quantfin.portfolio {frontier,backtest,experiment} --help``.
"""

from quantfin.portfolio import (
    backtest,
    black_litterman,
    data,
    estimators,
    experiments,
    hrp,
    markowitz,
    metrics,
    risk_parity,
    strategies,
)
from quantfin.portfolio._common import TRADING_DAYS, OptimizationError
from quantfin.portfolio.backtest import BacktestResult, walk_forward_backtest
from quantfin.portfolio.black_litterman import black_litterman as black_litterman_posterior
from quantfin.portfolio.estimators import cov_ledoit_wolf, cov_sample, mean_historical
from quantfin.portfolio.hrp import hrp_weights
from quantfin.portfolio.markowitz import (
    efficient_frontier,
    max_sharpe,
    min_variance,
)
from quantfin.portfolio.risk_parity import equal_risk_contribution

__all__ = [
    "TRADING_DAYS",
    "BacktestResult",
    "OptimizationError",
    "backtest",
    "black_litterman",
    "black_litterman_posterior",
    "cov_ledoit_wolf",
    "cov_sample",
    "data",
    "efficient_frontier",
    "equal_risk_contribution",
    "estimators",
    "experiments",
    "hrp",
    "hrp_weights",
    "markowitz",
    "max_sharpe",
    "mean_historical",
    "metrics",
    "min_variance",
    "risk_parity",
    "strategies",
    "walk_forward_backtest",
]
