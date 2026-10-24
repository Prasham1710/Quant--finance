"""End-to-end tests of the command-line interface on small synthetic markets."""

from __future__ import annotations

import pandas as pd
import pytest

from quantfin.portfolio import cli, data

SMALL = ["--source", "synthetic", "--n-assets", "5", "--years", "3", "--seed", "4"]


def test_frontier_command_writes_outputs(tmp_path, capsys):
    assert cli.main(["frontier", *SMALL, "--points", "12", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "synthetic data" in out
    assert "Ledoit-Wolf shrinkage intensity" in out
    for name in (
        "weights.csv",
        "stats.csv",
        "frontier.png",
        "weights.png",
        "risk_contributions.png",
    ):
        assert (tmp_path / "frontier" / name).exists(), name
    weights = pd.read_csv(tmp_path / "frontier" / "weights.csv", index_col=0)
    assert weights.sum().round(8).eq(1.0).all()


def test_backtest_command_writes_outputs(tmp_path, capsys):
    args = ["backtest", *SMALL, "--lookback", "252", "--rebalance", "21", "--cost-bps", "5"]
    assert cli.main([*args, "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Walk-forward: 252-day estimation window" in out
    summary = pd.read_csv(tmp_path / "backtest" / "summary.csv", index_col=0)
    assert len(summary) == 9 and "Sharpe" in summary.columns
    assert (tmp_path / "backtest" / "cumulative_returns.png").exists()
    daily = pd.read_csv(tmp_path / "backtest" / "daily_returns.csv", index_col=0)
    assert len(daily) == 3 * 252 - 252


def test_experiment_command_writes_outputs(tmp_path, capsys):
    args = ["experiment", *SMALL, "--trials", "3", "--sample-sizes", "63", "126"]
    assert cli.main([*args, "--frontier-days", "126", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "synthetic market's exact parameters" in out
    table = pd.read_csv(tmp_path / "experiment" / "estimation_error.csv")
    assert set(table["n_obs"]) == {63, 126}
    assert "Covariance shrinkage" in out
    for name in (
        "estimation_error.png",
        "frontiers_sample.png",
        "frontiers_ledoit_wolf.png",
        "shrinkage.png",
        "shrinkage.csv",
    ):
        assert (tmp_path / "experiment" / name).exists(), name


def test_csv_source(tmp_path, capsys):
    market = data.synthetic_market(n_assets=4, n_years=2, seed=9)
    path = tmp_path / "prices.csv"
    market.prices.to_csv(path)
    assert (
        cli.main(["frontier", "--source", "csv", "--csv", str(path), "--out", str(tmp_path)]) == 0
    )
    assert "csv, 4 assets" in capsys.readouterr().out


def test_csv_source_requires_path(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["frontier", "--source", "csv", "--out", str(tmp_path)])


def test_yahoo_source_uses_downloaded_prices(tmp_path, monkeypatch, capsys):
    market = data.synthetic_market(n_assets=3, n_years=2, seed=2)
    seen = {}

    def fake_download(tickers, start, end):
        seen.update(tickers=tickers, start=start, end=end)
        return market.prices.set_axis(list(tickers), axis=1)

    monkeypatch.setattr(data, "load_prices_yahoo", fake_download)
    argv = ["frontier", "--tickers", "AAA", "BBB", "CCC", "--start", "2015-01-01"]
    assert cli.main([*argv, "--out", str(tmp_path)]) == 0
    assert seen == {"tickers": ["AAA", "BBB", "CCC"], "start": "2015-01-01", "end": None}
    out = capsys.readouterr().out
    assert "synthetic data" not in out and "AAA" in out


def test_download_errors_become_exit_code_1(tmp_path, monkeypatch, capsys):
    def failing(*args, **kwargs):
        raise RuntimeError("no data returned for SPY")

    monkeypatch.setattr(data, "load_prices_yahoo", failing)
    assert cli.main(["frontier", "--out", str(tmp_path)]) == 1
    assert "no data returned" in capsys.readouterr().err
