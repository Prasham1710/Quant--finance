"""Charts for the portfolio optimizer.

Every function builds and returns a :class:`matplotlib.figure.Figure` without touching
pyplot's global state, so the module works the same in scripts, tests, and notebooks
(in a notebook, display the returned figure). :func:`save_figure` writes it to disk.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from quantfin.portfolio import markowitz, metrics, risk_parity
from quantfin.portfolio.backtest import BacktestResult
from quantfin.portfolio.experiments import FrontierExperiment

_PCT = PercentFormatter(xmax=1.0)


def _label_sample_sizes(ax: object, sizes: pd.Series) -> None:
    """Log-scale x axis with ticks at the simulated history lengths."""
    ticks = sorted({int(n) for n in sizes})
    ax.set_xscale("log")
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Days of history used for estimation")


def _new_figure(width: float = 9.0, height: float = 5.5) -> tuple[Figure, object]:
    fig = Figure(figsize=(width, height), layout="constrained")
    return fig, fig.add_subplot()


def save_figure(fig: Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save ``fig`` to ``path`` (parent directories are created) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    mu: pd.Series | None = None,
    cov: pd.DataFrame | None = None,
    portfolios: Mapping[str, pd.Series] | None = None,
    rf: float | None = None,
    extra_frontiers: Mapping[str, pd.DataFrame] | None = None,
    title: str = "Efficient frontier",
) -> Figure:
    """Plot a frontier with optional assets, named portfolios and capital market line.

    Args:
        frontier: output of :func:`markowitz.efficient_frontier`.
        mu, cov: annualized parameters; when given, individual assets are plotted and
            ``portfolios`` are placed at their (volatility, return) under these parameters.
        portfolios: name -> weights of portfolios to mark.
        rf: when given, draw the capital market line through the frontier's max-Sharpe point.
        extra_frontiers: label -> other frontiers to overlay (e.g. Ledoit-Wolf inputs).
        title: chart title.
    """
    fig, ax = _new_figure()
    ax.plot(frontier["volatility"], frontier["return"], lw=2.2, label="Efficient frontier")
    for label, other in (extra_frontiers or {}).items():
        ax.plot(other["volatility"], other["return"], lw=1.5, ls="--", label=label)
    if mu is not None and cov is not None:
        vols = np.sqrt(np.diag(np.asarray(cov, dtype=float)))
        ax.scatter(vols, np.asarray(mu, dtype=float), s=22, color="0.45", zorder=3, label="Assets")
        for name, x, y in zip(mu.index, vols, mu.to_numpy(), strict=True):
            ax.annotate(str(name), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
        markers = iter("o^sDvP*Xh")
        for name, w in (portfolios or {}).items():
            ax.scatter(
                markowitz.portfolio_volatility(w, cov),
                markowitz.portfolio_return(w, mu),
                s=70,
                marker=next(markers, "o"),
                edgecolor="black",
                zorder=4,
                label=name,
            )
    if rf is not None:
        best = frontier.loc[frontier["sharpe"].idxmax()]
        x_max = float(frontier["volatility"].max()) * 1.05
        ax.plot(
            [0.0, x_max],
            [rf, rf + best["sharpe"] * x_max],
            lw=1,
            color="0.3",
            ls=":",
            label="Capital market line",
        )
    ax.xaxis.set_major_formatter(_PCT)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_xlim(left=0.0)
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    return fig


def plot_weights(
    weights: Mapping[str, pd.Series] | pd.DataFrame,
    title: str = "Weights",
    ylabel: str = "Weight",
) -> Figure:
    """Stacked bar chart of several portfolios' weights, one bar per portfolio."""
    frame = pd.DataFrame(weights) if not isinstance(weights, pd.DataFrame) else weights
    fig, ax = _new_figure(10.0, 5.5)
    frame.T.plot.bar(ax=ax, stacked=True, width=0.75, colormap="tab20", legend=False)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    return fig


def plot_risk_contributions(
    weights: Mapping[str, pd.Series], cov: pd.DataFrame, title: str = "Share of portfolio risk"
) -> Figure:
    """Stacked bars of each portfolio's relative risk contributions (sum to 100%)."""
    shares = pd.DataFrame(
        {name: risk_parity.risk_contributions(w, cov, relative=True) for name, w in weights.items()}
    )
    return plot_weights(shares, title=title, ylabel="Share of portfolio volatility")


def plot_weights_over_time(result: BacktestResult, title: str | None = None) -> Figure:
    """Stacked area of a backtest's target weights at each rebalance date."""
    fig, ax = _new_figure(10.0, 5.0)
    w = result.weights.fillna(0.0)
    ax.stackplot(w.index, w.T.to_numpy(), labels=[str(c) for c in w.columns], step="post")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_title(title or f"{result.name}: weights at each rebalance")
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    return fig


def _net_returns(results: Mapping[str, BacktestResult] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results
    return pd.DataFrame({name: r.returns for name, r in results.items()})


def plot_cumulative_returns(
    results: Mapping[str, BacktestResult] | pd.DataFrame,
    log_scale: bool = True,
    title: str = "Growth of $1 (net of costs)",
) -> Figure:
    """Wealth paths of several strategies, starting at 1."""
    wealth = 1.0 + metrics.cumulative_returns(_net_returns(results))
    fig, ax = _new_figure(10.0, 5.5)
    for name in wealth.columns:
        ax.plot(wealth.index, wealth[name], lw=1.4, label=name)
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel("Wealth")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return fig


def plot_drawdowns(
    results: Mapping[str, BacktestResult] | pd.DataFrame, title: str = "Drawdowns"
) -> Figure:
    """Drawdown paths (fall from the running peak) of several strategies."""
    dd = metrics.drawdowns(_net_returns(results))
    fig, ax = _new_figure(10.0, 4.5)
    for name in dd.columns:
        ax.plot(dd.index, dd[name], lw=1.1, label=name)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig


def plot_frontier_experiment(fe: FrontierExperiment, title: str | None = None) -> Figure:
    """True vs estimated vs realized frontiers: the error-maximization picture."""
    fig, ax = _new_figure()
    ax.plot(fe.true["volatility"], fe.true["return"], lw=2.2, label="True frontier")
    ax.plot(
        fe.estimated["volatility"],
        fe.estimated["return"],
        lw=1.8,
        ls="--",
        label="Estimated frontier (what the optimizer believes)",
    )
    ax.plot(
        fe.realized["volatility"],
        fe.realized["return"],
        lw=1.8,
        ls="-.",
        label="Realized frontier (estimated weights, true parameters)",
    )
    ax.xaxis.set_major_formatter(_PCT)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.set_title(title or f"Estimation error with {fe.n_obs} days of data")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig


def plot_estimation_error(summary: pd.DataFrame, title: str | None = None) -> Figure:
    """Believed vs true Sharpe ratio as the estimation window grows.

    Args:
        summary: output of :func:`experiments.estimation_error_experiment`.
    """
    fig = Figure(figsize=(11.0, 5.0), layout="constrained")
    # Separate y scales: the believed Sharpe ratios are several times the true ones.
    ax_believed, ax_true = fig.subplots(1, 2)
    optimized = summary[summary["estimator"] != "-"]
    for (portfolio, estimator), group in optimized.groupby(["portfolio", "estimator"], sort=False):
        label = f"{portfolio} ({estimator})"
        ax_believed.plot(group["n_obs"], group["believed_sharpe"], marker="o", label=label)
        ax_true.plot(group["n_obs"], group["true_sharpe"], marker="o", label=label)
    for name, style in (("True optimum", "-"), ("Equal weight", ":")):
        ref = summary[summary["portfolio"] == name]
        for ax in (ax_believed, ax_true):
            ax.plot(ref["n_obs"], ref["true_sharpe"], color="black", ls=style, lw=1.2, label=name)
    for ax, sub in ((ax_believed, "Sharpe the optimizer believes"), (ax_true, "True Sharpe")):
        _label_sample_sizes(ax, summary["n_obs"])
        ax.set_title(sub)
        ax.grid(alpha=0.3)
    ax_believed.set_ylabel("Annualized Sharpe ratio")
    handles, labels = ax_true.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(title or "Mean-variance optimization magnifies estimation error")
    return fig


def plot_shrinkage(summary: pd.DataFrame, title: str | None = None) -> Figure:
    """Excess true risk of minimum variance (sample vs Ledoit-Wolf) and shrinkage intensity.

    Args:
        summary: output of :func:`experiments.shrinkage_experiment`.
    """
    fig = Figure(figsize=(11.0, 4.5), layout="constrained")
    ax_risk, ax_delta = fig.subplots(1, 2)
    labels = {"sample": "Sample covariance", "ledoit_wolf": "Ledoit-Wolf"}
    for name, group in summary.groupby("estimator", sort=False):
        ok = group["valid_trials"] > 0
        ax_risk.plot(
            group.loc[ok, "n_obs"],
            group.loc[ok, "excess_volatility"],
            marker="o",
            label=labels.get(name, name),
        )
    lw = summary[summary["estimator"] == "ledoit_wolf"]
    ax_delta.plot(lw["n_obs"], lw["shrinkage"], marker="o", color="0.3")
    ax_risk.set_title("Extra true volatility over the true minimum")
    ax_risk.yaxis.set_major_formatter(_PCT)
    ax_risk.legend(fontsize=8)
    ax_delta.set_title("Ledoit-Wolf shrinkage intensity δ")
    ax_delta.set_ylim(0.0, 1.0)
    for ax in (ax_risk, ax_delta):
        _label_sample_sizes(ax, summary["n_obs"])
        ax.grid(alpha=0.3)
    fig.suptitle(title or "Covariance shrinkage and the minimum-variance portfolio")
    return fig


__all__ = [
    "plot_cumulative_returns",
    "plot_drawdowns",
    "plot_efficient_frontier",
    "plot_estimation_error",
    "plot_frontier_experiment",
    "plot_risk_contributions",
    "plot_shrinkage",
    "plot_weights",
    "plot_weights_over_time",
    "save_figure",
]
