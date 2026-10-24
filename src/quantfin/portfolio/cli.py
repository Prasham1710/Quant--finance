"""Command-line entry point: ``python -m quantfin.portfolio <command>``.

Commands:

* ``frontier``   — estimate inputs, draw the efficient frontier, compare portfolios.
* ``backtest``   — walk-forward comparison of every strategy, net of trading costs.
* ``experiment`` — Monte Carlo with known truth: how estimation error hurts each method.

Data sources (``--source``): ``yahoo`` downloads adjusted prices with yfinance,
``csv`` reads a wide price file, and ``synthetic`` generates a factor-model market
whose true parameters are known. Synthetic results illustrate the method only; they
say nothing about real markets.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from quantfin.portfolio import (
    backtest,
    data,
    estimators,
    experiments,
    hrp,
    markowitz,
    metrics,
    plots,
    risk_parity,
    strategies,
)

_SYNTHETIC_BANNER = (
    "NOTE: synthetic data. These numbers illustrate the method on a simulated market; "
    "they are not evidence about real markets. Use --source yahoo or --source csv for that."
)

_PERCENT_COLUMNS = (
    "CAGR",
    "Exp. return",
    "Volatility",
    "Max drawdown",
    "Max weight",
    "VaR 95%",
    "ES 95%",
    "Turnover (annual)",
)


def _load_returns(args: argparse.Namespace) -> tuple[pd.DataFrame, data.SyntheticMarket | None]:
    """Daily returns for the chosen source, plus the synthetic truth when there is one."""
    if args.source == "synthetic":
        market = data.synthetic_market(n_assets=args.n_assets, n_years=args.years, seed=args.seed)
        return market.returns, market
    if args.source == "csv":
        if not args.csv:
            raise SystemExit("--csv PATH is required with --source csv")
        prices = data.load_prices_csv(args.csv)
    else:
        prices = data.load_prices_yahoo(args.tickers, start=args.start, end=args.end)
    return data.prices_to_returns(prices), None


def _describe(returns: pd.DataFrame, args: argparse.Namespace) -> None:
    start, end = returns.index[0], returns.index[-1]
    print(
        f"Data: {args.source}, {returns.shape[1]} assets, {returns.shape[0]} days "
        f"({start:%Y-%m-%d} to {end:%Y-%m-%d})"
        if isinstance(start, pd.Timestamp)
        else f"Data: {args.source}, {returns.shape[1]} assets, {returns.shape[0]} days"
    )
    if args.source == "synthetic":
        print(_SYNTHETIC_BANNER)
    print()


def _format_table(table: pd.DataFrame) -> str:
    out = table.copy()
    for col in out.columns:
        if col in _PERCENT_COLUMNS:
            out[col] = out[col].map(lambda v: f"{v:.1%}")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: f"{v:.2f}")
    return out.to_string()


def _portfolio_stats(
    portfolios: dict[str, pd.Series], mu: pd.Series, cov: pd.DataFrame, rf: float
) -> pd.DataFrame:
    rows = {}
    for name, w in portfolios.items():
        rows[name] = {
            "Exp. return": markowitz.portfolio_return(w, mu),
            "Volatility": markowitz.portfolio_volatility(w, cov),
            "Sharpe": markowitz.portfolio_sharpe(w, mu, cov, rf),
            "Effective N": metrics.effective_number_of_assets(w),
            "Diversification ratio": metrics.diversification_ratio(w, cov),
            "Max weight": float(w.max()),
        }
    return pd.DataFrame(rows).T


def cmd_frontier(args: argparse.Namespace) -> int:
    """Efficient frontier with sample and Ledoit-Wolf inputs, plus every portfolio type."""
    returns, _ = _load_returns(args)
    _describe(returns, args)
    out = Path(args.out) / "frontier"
    mu = estimators.mean_historical(returns)
    cov = estimators.cov_sample(returns)
    cov_lw = estimators.cov_ledoit_wolf(returns)
    _, _, delta = estimators.ledoit_wolf_shrinkage(returns)
    print(f"Ledoit-Wolf shrinkage intensity toward constant correlation: {delta:.3f}\n")

    frontier = markowitz.efficient_frontier(mu, cov, n_points=args.points, rf=args.rf)
    frontier_lw = markowitz.efficient_frontier(mu, cov_lw, n_points=args.points, rf=args.rf)
    equal = pd.Series(1.0 / len(mu), index=mu.index)
    portfolios = {
        "Equal weight": equal,
        "Min variance": markowitz.min_variance(cov),
        "Max Sharpe": markowitz.max_sharpe(mu, cov, rf=args.rf),
        "Max Sharpe (Ledoit-Wolf)": markowitz.max_sharpe(mu, cov_lw, rf=args.rf),
        "Black-Litterman (momentum views)": strategies.black_litterman_strategy(rf=args.rf)(
            returns
        ),
        "Risk parity (ERC)": risk_parity.equal_risk_contribution(cov),
        "HRP": hrp.hrp_weights(cov),
    }
    stats = _portfolio_stats(portfolios, mu, cov, args.rf)
    print("In-sample portfolio statistics (full-sample estimates; optimistic by construction):")
    print(_format_table(stats))
    print("\nWeights:")
    weights = pd.DataFrame(portfolios)
    print(weights.map(lambda v: f"{v:.1%}").to_string())

    out.mkdir(parents=True, exist_ok=True)
    weights.to_csv(out / "weights.csv")
    stats.to_csv(out / "stats.csv")
    frontier.to_csv(out / "frontier_sample.csv", index=False)
    frontier_lw.to_csv(out / "frontier_ledoit_wolf.csv", index=False)
    plots.save_figure(
        plots.plot_efficient_frontier(
            frontier,
            mu,
            cov,
            portfolios,
            rf=args.rf,
            extra_frontiers={"Frontier with Ledoit-Wolf covariance": frontier_lw},
        ),
        out / "frontier.png",
    )
    plots.save_figure(
        plots.plot_weights(portfolios, title="Portfolio weights"), out / "weights.png"
    )
    plots.save_figure(
        plots.plot_risk_contributions(portfolios, cov), out / "risk_contributions.png"
    )
    print(f"\nSaved tables and charts to {out}/")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Walk-forward comparison of all strategies, net of transaction costs."""
    returns, _ = _load_returns(args)
    _describe(returns, args)
    out = Path(args.out) / "backtest"
    momentum = min(252, args.lookback)
    strategy_set = strategies.default_strategies(rf=args.rf, momentum_lookback=momentum)
    print(
        f"Walk-forward: {args.lookback}-day estimation window, rebalance every "
        f"{args.rebalance} days, {args.cost_bps:g} bp cost per unit traded.\n"
    )
    results = backtest.run_backtests(
        returns,
        strategy_set,
        lookback=args.lookback,
        rebalance_every=args.rebalance,
        cost_bps=args.cost_bps,
    )
    table = backtest.compare_backtests(results, rf=args.rf)
    print(_format_table(table))
    for name, strategy in strategy_set.items():
        if getattr(strategy, "fallbacks", 0):
            print(
                f"\n{name}: fell back to min variance in {strategy.fallbacks} rebalances "
                "(no asset had a positive expected excess return)."
            )

    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "summary.csv")
    pd.DataFrame({n: r.returns for n, r in results.items()}).to_csv(out / "daily_returns.csv")
    plots.save_figure(plots.plot_cumulative_returns(results), out / "cumulative_returns.png")
    plots.save_figure(plots.plot_drawdowns(results), out / "drawdowns.png")
    for name in ("Max Sharpe (sample)", "Max Sharpe (Ledoit-Wolf)", "HRP", "Risk parity (ERC)"):
        if name in results:
            slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            plots.save_figure(
                plots.plot_weights_over_time(results[name]), out / f"weights_{slug}.png"
            )
    print(f"\nSaved tables and charts to {out}/")
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    """Monte Carlo estimation-error study with known true parameters."""
    returns, market = _load_returns(args)
    _describe(returns, args)
    out = Path(args.out) / "experiment"
    if market is not None:
        true_mu, true_cov = market.true_mu, market.true_cov
        print("Truth: the synthetic market's exact parameters.")
    else:
        true_mu, true_cov = experiments.calibrate_truth(returns)
        print("Truth: full-sample estimates from the data, treated as known parameters.")
    sizes = tuple(args.sample_sizes)
    print(f"Simulating {args.trials} histories for each of {sizes} days...\n")
    summary = experiments.estimation_error_experiment(
        true_mu, true_cov, sample_sizes=sizes, n_trials=args.trials, rf=args.rf, seed=args.seed
    )
    columns = ["believed_sharpe", "true_sharpe", "optimism_gap", "pct_of_optimum", "fallback_rate"]
    view = summary.set_index(["n_obs", "portfolio", "estimator"])[columns]
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 120):
        print(view.to_string())

    print(
        f"\nCovariance shrinkage: unconstrained minimum variance, {len(true_mu)} assets "
        "(extra true volatility over the true minimum):\n"
    )
    shrink = experiments.shrinkage_experiment(
        true_cov, sample_sizes=sizes, n_trials=args.trials, seed=args.seed
    )
    table = shrink.pivot(index="n_obs", columns="estimator", values="excess_volatility")
    table["shrinkage δ"] = shrink.groupby("n_obs")["shrinkage"].max()
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(table.to_string())
    if len(true_mu) < 50:
        print(
            "(Shrinkage matters far more with many assets: try --source synthetic --n-assets 100.)"
        )

    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "estimation_error.csv", index=False)
    shrink.to_csv(out / "shrinkage.csv", index=False)
    plots.save_figure(plots.plot_estimation_error(summary), out / "estimation_error.png")
    plots.save_figure(plots.plot_shrinkage(shrink), out / "shrinkage.png")
    for estimator in ("sample", "ledoit_wolf"):
        fe = experiments.frontier_experiment(
            true_mu,
            true_cov,
            n_obs=args.frontier_days,
            seed=args.seed,
            rf=args.rf,
            cov_estimator=estimator,
        )
        plots.save_figure(
            plots.plot_frontier_experiment(
                fe, title=f"Estimation error with {fe.n_obs} days of data ({estimator} covariance)"
            ),
            out / f"frontiers_{estimator}.png",
        )
    print(f"\nSaved tables and charts to {out}/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for all sub-commands."""
    parser = argparse.ArgumentParser(
        prog="python -m quantfin.portfolio",
        description="Portfolio optimization: Markowitz, Ledoit-Wolf, Black-Litterman, "
        "risk parity and HRP.",
    )
    common = argparse.ArgumentParser(add_help=False)
    src = common.add_argument_group("data")
    src.add_argument("--source", choices=("yahoo", "csv", "synthetic"), default="yahoo")
    src.add_argument(
        "--tickers",
        nargs="+",
        default=list(data.DEFAULT_UNIVERSE),
        help="tickers for --source yahoo (default: 12 multi-asset ETFs)",
    )
    src.add_argument("--start", default=data.DEFAULT_START)
    src.add_argument("--end", default=None)
    src.add_argument("--csv", default=None, help="wide price CSV for --source csv")
    src.add_argument("--n-assets", type=int, default=12, help="synthetic market size")
    src.add_argument("--years", type=float, default=16.0, help="synthetic history length")
    src.add_argument("--seed", type=int, default=0)
    common.add_argument("--rf", type=float, default=0.0, help="annual risk-free rate, e.g. 0.02")
    common.add_argument("--out", default="results", help="output directory")

    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("frontier", parents=[common], help=cmd_frontier.__doc__)
    p.add_argument("--points", type=int, default=50)
    p.set_defaults(func=cmd_frontier)

    p = sub.add_parser("backtest", parents=[common], help=cmd_backtest.__doc__)
    p.add_argument("--lookback", type=int, default=504, help="estimation window in days")
    p.add_argument("--rebalance", type=int, default=21, help="days between rebalances")
    p.add_argument("--cost-bps", type=float, default=10.0, help="cost per unit of turnover")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("experiment", parents=[common], help=cmd_experiment.__doc__)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--sample-sizes", type=int, nargs="+", default=[63, 126, 252, 504, 1260])
    p.add_argument("--frontier-days", type=int, default=252)
    p.set_defaults(func=cmd_experiment)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
