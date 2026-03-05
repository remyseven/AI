"""
Round 3 backtests — Novel BTC cycle strategies.
  V11: Halving Cycle Momentum
  V12: Pi Cycle Top Detector
  V13: Supertrend (ATR Bands)
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest_engine import BacktestConfig, run_backtest
from strategies_v3 import get_round3_strategies, BTC_HALVINGS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    path = os.path.join(DATA_DIR, "btc_usd_daily.csv")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def plot_results(results, df):
    """Plot equity curves with halving dates marked."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 16),
                             gridspec_kw={"height_ratios": [2.5, 1, 1]})
    colors = ["#00BCD4", "#E91E63", "#8BC34A"]

    ax1 = axes[0]
    bh_equity = 10000 * df["Close"] / df["Close"].iloc[0]
    bh_ret = (bh_equity.iloc[-1] / 10000 - 1) * 100
    ax1.plot(df.index, bh_equity, color="#888888", linewidth=2, linestyle="--",
             label=f"Buy & Hold ({bh_ret:.0f}%)", alpha=0.7)

    for i, r in enumerate(results):
        eq = r.equity_curve
        label = (f"{r.strategy_name}  |  "
                 f"Return: {r.total_return_pct:.0f}%  CAGR: {r.cagr_pct:.1f}%  "
                 f"MaxDD: {r.max_drawdown_pct:.1f}%  Sharpe: {r.sharpe_ratio:.2f}")
        ax1.plot(eq.index, eq.values, color=colors[i], linewidth=2, label=label)

    # Mark halvings
    for h in BTC_HALVINGS:
        if df.index[0] <= h <= df.index[-1]:
            ax1.axvline(h, color="#F7931A", linestyle=":", linewidth=1.5, alpha=0.8)

    ax1.set_yscale("log")
    ax1.set_ylabel("Portfolio Value (log scale)", fontsize=12)
    ax1.set_title("BTC/USD — Round 3: Novel Strategies", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # BTC price
    ax2 = axes[1]
    ax2.plot(df.index, df["Close"], color="#F7931A", linewidth=1.2)
    for h in BTC_HALVINGS:
        if df.index[0] <= h <= df.index[-1]:
            ax2.axvline(h, color="#F7931A", linestyle=":", linewidth=1, alpha=0.5)
            ax2.text(h, df["Close"].max() * 0.6, f"↑{h.year}", color="#F7931A",
                     fontsize=8, ha="center")
    ax2.set_yscale("log")
    ax2.set_ylabel("BTC Price (log)")
    ax2.set_title("BTC/USD Price  (orange dotted = halvings)")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Drawdowns
    ax3 = axes[2]
    for i, r in enumerate(results):
        eq = r.equity_curve
        peak = eq.expanding().max()
        dd = (eq - peak) / peak * 100
        ax3.fill_between(dd.index, dd.values, 0, color=colors[i], alpha=0.25)
        ax3.plot(dd.index, dd.values, color=colors[i], linewidth=1,
                 label=r.strategy_name.split(":")[0], alpha=0.85)

    ax3.set_ylabel("Drawdown %")
    ax3.set_title("Strategy Drawdowns")
    ax3.legend(loc="lower left", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "round3_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved round3_comparison.png")


def main():
    print("=" * 70)
    print("  BTC/USD — ROUND 3: NOVEL STRATEGIES")
    print("  V11: Halving Cycle Momentum")
    print("  V12: Pi Cycle Top Detector")
    print("  V13: Supertrend (ATR Bands)")
    print("=" * 70)

    df = load_data()
    print(f"\n  Data: {df.index[0].date()} to {df.index[-1].date()}  ({len(df)} candles)")

    config = BacktestConfig(
        initial_capital=10_000,
        commission_pct=0.001,   # 0.1% per trade
        slippage_pct=0.0005,    # 0.05% slippage
    )

    strategies = get_round3_strategies()
    results = []

    for strat in strategies:
        print(f"\n  Running: {strat.name}...")
        result = run_backtest(strat, df, config)
        results.append(result)
        print(result.summary())

        if result.trades:
            print(f"  Trade log (first 10 of {result.total_trades}):")
            for j, t in enumerate(result.trades[:10]):
                icon = "+" if t.pnl_pct > 0 else ""
                print(f"    #{j+1}: {t.entry_date.date()} @ ${t.entry_price:,.0f}"
                      f" -> {t.exit_date.date()} @ ${t.exit_price:,.0f}"
                      f"  {icon}{t.pnl_pct:.1f}%")
            if result.total_trades > 10:
                print(f"    ...and {result.total_trades - 10} more")

    # Comparison table
    print("\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    rows = []
    for r in results:
        rows.append({
            "Strategy": r.strategy_name,
            "Return%": round(r.total_return_pct, 1),
            "CAGR%": round(r.cagr_pct, 2),
            "MaxDD%": round(r.max_drawdown_pct, 1),
            "Sharpe": round(r.sharpe_ratio, 2),
            "Sortino": round(r.sortino_ratio, 2),
            "Calmar": round(r.calmar_ratio, 2),
            "PF": round(r.profit_factor, 2),
            "WinRate%": round(r.win_rate_pct, 1),
            "Trades": r.total_trades,
            "AvgTrade%": round(r.avg_trade_pct, 2),
            "TimeInMkt%": round(r.time_in_market_pct, 1),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(os.path.join(RESULTS_DIR, "round3_results.csv"), index=False)

    bh = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    print(f"\n  Buy & Hold (full period): {bh:.1f}%")

    # Rank by Calmar (return/drawdown)
    print("\n" + "=" * 70)
    print("  RANKING by Calmar Ratio (CAGR / Max Drawdown)")
    print("=" * 70)
    ranked = sorted(results, key=lambda r: r.calmar_ratio, reverse=True)
    for rank, r in enumerate(ranked, 1):
        marker = "  *** BEST ***" if rank == 1 else ""
        print(f"  #{rank}: {r.strategy_name}{marker}")
        print(f"       Return: {r.total_return_pct:.0f}%  CAGR: {r.cagr_pct:.1f}%"
              f"  MaxDD: {r.max_drawdown_pct:.1f}%  Calmar: {r.calmar_ratio:.2f}"
              f"  Sharpe: {r.sharpe_ratio:.2f}")

    print("\nGenerating charts...")
    plot_results(results, df)

    print(f"\n{'='*70}")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    results = main()
