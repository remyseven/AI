"""
Round 2 backtests - Enhanced strategies.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest_engine import BacktestConfig, run_backtest
from strategies import get_all_strategies
from strategies_v2 import get_round2_strategies

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    path = os.path.join(DATA_DIR, "btc_usd_daily.csv")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def plot_all_equity_curves(results, df, filename="equity_comparison_all.png"):
    """Plot equity curves for all strategies (round 1 + round 2)."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 20), gridspec_kw={"height_ratios": [2.5, 1, 1]})

    ax1 = axes[0]
    colors = [
        "#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800",  # R1
        "#00BCD4", "#E91E63", "#8BC34A", "#3F51B5", "#795548",  # R2
    ]

    # Buy & Hold
    bh_equity = 10000 * df["Close"] / df["Close"].iloc[0]
    ax1.plot(df.index, bh_equity, color="#888888", linewidth=2, linestyle="--",
             label=f"Buy & Hold ({(bh_equity.iloc[-1]/10000-1)*100:.0f}%)", alpha=0.7)

    for i, r in enumerate(results):
        eq = r.equity_curve
        label = f"{r.strategy_name} ({r.total_return_pct:.0f}%)"
        ax1.plot(eq.index, eq.values, color=colors[i % len(colors)],
                 linewidth=1.5 if i < 5 else 2.0, label=label,
                 alpha=0.6 if i < 5 else 0.9,
                 linestyle=":" if i < 5 else "-")

    ax1.set_yscale("log")
    ax1.set_ylabel("Portfolio Value (log scale)", fontsize=12)
    ax1.set_title("BTC/USD 1D - All Strategies Comparison", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # BTC price
    ax2 = axes[1]
    ax2.plot(df.index, df["Close"], color="#F7931A", linewidth=1.2)
    ax2.set_yscale("log")
    ax2.set_ylabel("BTC Price (log)")
    ax2.set_title("BTC/USD Price")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Drawdowns for R2 only
    ax3 = axes[2]
    for i, r in enumerate(results[5:], 5):
        eq = r.equity_curve
        peak = eq.expanding().max()
        dd = (eq - peak) / peak * 100
        ax3.fill_between(dd.index, dd.values, 0, color=colors[i % len(colors)], alpha=0.25)
        ax3.plot(dd.index, dd.values, color=colors[i % len(colors)], linewidth=0.8,
                 label=r.strategy_name, alpha=0.7)

    ax3.set_ylabel("Drawdown %")
    ax3.set_title("Round 2 Strategy Drawdowns")
    ax3.legend(loc="lower left", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def main():
    print("=" * 70)
    print("  BTC/USD 1D CYCLE TRADING - ROUND 2 (ALL STRATEGIES)")
    print("=" * 70)

    df = load_data()
    print(f"\n  Data: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} candles)")

    config = BacktestConfig(
        initial_capital=10000,
        commission_pct=0.001,
        slippage_pct=0.0005,
    )

    # Run ALL strategies (R1 + R2)
    all_strategies = get_all_strategies() + get_round2_strategies()
    results = []

    for strat in all_strategies:
        print(f"\n  Running: {strat.name}...")
        result = run_backtest(strat, df, config)
        results.append(result)

    # Print comparison table
    print("\n" + "=" * 70)
    print("  FULL COMPARISON TABLE")
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

    # Advanced scoring
    print("\n" + "=" * 70)
    print("  RANKING BY COMPOSITE SCORE")
    print("=" * 70)

    scored = []
    for r in results:
        # Penalize negative returns heavily
        if r.total_return_pct < 0:
            scored.append((r, -10))
            continue

        # Risk-adjusted return emphasis
        cagr_score = min(r.cagr_pct / 15, 4.0)  # Cap at 60% CAGR
        dd_score = max(0, 2 - abs(r.max_drawdown_pct) / 25)  # 0 at -50%
        sharpe_score = min(r.sharpe_ratio / 1.0, 4.0)
        sortino_score = min(r.sortino_ratio / 1.5, 3.0)
        calmar_score = min(r.calmar_ratio / 0.5, 4.0)
        pf_score = min(r.profit_factor / 3.0, 3.0)
        wr_score = r.win_rate_pct / 50
        consistency = min(r.total_trades / 5, 2.0)  # Reward having enough trades

        composite = (
            cagr_score * 0.20 +
            dd_score * 0.20 +
            sharpe_score * 0.20 +
            sortino_score * 0.10 +
            calmar_score * 0.10 +
            pf_score * 0.08 +
            wr_score * 0.07 +
            consistency * 0.05
        )
        scored.append((r, composite))

    scored.sort(key=lambda x: x[1], reverse=True)

    for rank, (r, score) in enumerate(scored, 1):
        marker = " *** BEST ***" if rank == 1 else ""
        print(f"  #{rank:2d}  Score: {score:.3f}  |  {r.strategy_name}"
              f"  |  Return: {r.total_return_pct:.0f}%  CAGR: {r.cagr_pct:.1f}%"
              f"  MaxDD: {r.max_drawdown_pct:.1f}%  Sharpe: {r.sharpe_ratio:.2f}{marker}")

    # Detailed summary of top 3
    print("\n" + "=" * 70)
    print("  TOP 3 STRATEGIES - DETAILED ANALYSIS")
    print("=" * 70)
    for rank, (r, score) in enumerate(scored[:3], 1):
        print(r.summary())
        if r.trades:
            print(f"  Trade details ({len(r.trades)} trades):")
            for j, t in enumerate(r.trades[:10]):
                pnl_icon = "+" if t.pnl_pct > 0 else ""
                print(f"    #{j+1}: {t.entry_date.date()} @ ${t.entry_price:,.0f} -> "
                      f"{t.exit_date.date()} @ ${t.exit_price:,.0f} = {pnl_icon}{t.pnl_pct:.1f}%")
            if len(r.trades) > 10:
                print(f"    ... and {len(r.trades) - 10} more trades")

    # Plot
    print("\nGenerating comparison chart...")
    plot_all_equity_curves(results, df)

    # Save full results
    table.to_csv(os.path.join(RESULTS_DIR, "comparison_all.csv"), index=False)

    # Identify the best strategy
    best = scored[0][0]
    print(f"\n{'='*70}")
    print(f"  FINAL WINNER: {best.strategy_name}")
    print(f"  Return: {best.total_return_pct:.1f}%  |  CAGR: {best.cagr_pct:.2f}%")
    print(f"  Max DD: {best.max_drawdown_pct:.1f}%  |  Sharpe: {best.sharpe_ratio:.2f}")
    print(f"  Sortino: {best.sortino_ratio:.2f}  |  Calmar: {best.calmar_ratio:.2f}")
    print(f"{'='*70}")

    return results, scored


if __name__ == "__main__":
    results, scored = main()
