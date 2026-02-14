"""
Run backtests for all BTC/USD cycle trading strategies and compare results.
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

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    path = os.path.join(DATA_DIR, "btc_usd_daily.csv")
    if not os.path.exists(path):
        print("Data file not found. Generating...")
        from generate_data import generate_btc_daily_data
        df = generate_btc_daily_data()
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(path)
    else:
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def plot_equity_curves(results, df, filename="equity_comparison.png"):
    """Plot equity curves for all strategies."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 18), gridspec_kw={"height_ratios": [2, 1, 1]})

    # --- Top panel: Equity curves ---
    ax1 = axes[0]
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    # Plot buy & hold
    bh_equity = 10000 * df["Close"] / df["Close"].iloc[0]
    ax1.plot(df.index, bh_equity, color="#888888", linewidth=1.5, linestyle="--",
             label=f"Buy & Hold ({(bh_equity.iloc[-1]/10000-1)*100:.0f}%)", alpha=0.7)

    for i, r in enumerate(results):
        eq = r.equity_curve
        label = f"{r.strategy_name} ({r.total_return_pct:.0f}%)"
        ax1.plot(eq.index, eq.values, color=colors[i % len(colors)],
                 linewidth=1.5, label=label, alpha=0.85)

    ax1.set_yscale("log")
    ax1.set_ylabel("Portfolio Value (log scale)", fontsize=12)
    ax1.set_title("BTC/USD 1D Strategy Comparison - Equity Curves", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # --- Middle panel: BTC price with cycle annotations ---
    ax2 = axes[1]
    ax2.plot(df.index, df["Close"], color="#F7931A", linewidth=1.2, label="BTC/USD")
    ax2.set_yscale("log")
    ax2.set_ylabel("BTC Price (log)", fontsize=12)
    ax2.set_title("BTC/USD Price", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    # --- Bottom panel: Drawdowns ---
    ax3 = axes[2]
    for i, r in enumerate(results):
        eq = r.equity_curve
        peak = eq.expanding().max()
        dd = (eq - peak) / peak * 100
        ax3.fill_between(dd.index, dd.values, 0, color=colors[i % len(colors)],
                         alpha=0.3, label=r.strategy_name)
        ax3.plot(dd.index, dd.values, color=colors[i % len(colors)],
                 linewidth=0.8, alpha=0.6)

    ax3.set_ylabel("Drawdown %", fontsize=12)
    ax3.set_title("Strategy Drawdowns", fontsize=12)
    ax3.legend(loc="lower left", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_trade_analysis(result, df, filename=None):
    """Plot trade entry/exit points on price chart for a single strategy."""
    if filename is None:
        filename = f"trades_{result.strategy_name.replace(' ', '_').replace(':', '')}.png"

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(df.index, df["Close"], color="#888888", linewidth=0.8, alpha=0.7)

    # Plot trades
    for trade in result.trades:
        color = "#4CAF50" if trade.pnl_pct > 0 else "#F44336"
        ax1.scatter(trade.entry_date, trade.entry_price, marker="^", color="#2196F3", s=60, zorder=5)
        ax1.scatter(trade.exit_date, trade.exit_price, marker="v", color=color, s=60, zorder=5)

    ax1.set_yscale("log")
    ax1.set_ylabel("BTC Price (log)", fontsize=12)
    ax1.set_title(f"{result.strategy_name} - Trade Entry/Exit Points", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Per-trade PnL bars
    ax2 = axes[1]
    if result.trades:
        trade_dates = [t.exit_date for t in result.trades]
        trade_pnls = [t.pnl_pct for t in result.trades]
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in trade_pnls]
        ax2.bar(trade_dates, trade_pnls, color=colors, width=15, alpha=0.7)
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_ylabel("Trade P&L %", fontsize=12)

    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


def create_comparison_table(results):
    """Create a comparison DataFrame of all strategy metrics."""
    rows = []
    for r in results:
        rows.append({
            "Strategy": r.strategy_name,
            "Total Return %": round(r.total_return_pct, 1),
            "CAGR %": round(r.cagr_pct, 2),
            "Max Drawdown %": round(r.max_drawdown_pct, 1),
            "Sharpe": round(r.sharpe_ratio, 2),
            "Sortino": round(r.sortino_ratio, 2),
            "Calmar": round(r.calmar_ratio, 2),
            "Profit Factor": round(r.profit_factor, 2),
            "Win Rate %": round(r.win_rate_pct, 1),
            "Total Trades": r.total_trades,
            "Avg Trade %": round(r.avg_trade_pct, 2),
            "Avg Win %": round(r.avg_win_pct, 2),
            "Avg Loss %": round(r.avg_loss_pct, 2),
            "Max Consec Losses": r.max_consecutive_losses,
            "Time in Market %": round(r.time_in_market_pct, 1),
        })
    table = pd.DataFrame(rows)
    return table


def score_strategies(results):
    """
    Score strategies using a weighted composite metric.
    Higher is better. Considers risk-adjusted returns, consistency, and practicality.
    """
    scores = []
    for r in results:
        # Normalize metrics to comparable scales
        cagr_score = r.cagr_pct / 10  # normalize around 10% CAGR
        dd_score = max(0, 1 - abs(r.max_drawdown_pct) / 50)  # penalize >50% DD
        sharpe_score = r.sharpe_ratio / 1.5  # normalize around 1.5 Sharpe
        sortino_score = r.sortino_ratio / 2.0
        calmar_score = min(r.calmar_ratio / 1.0, 3.0)  # cap at 3
        pf_score = min(r.profit_factor / 2.0, 2.5)  # cap at 2.5
        wr_score = r.win_rate_pct / 60  # normalize around 60% win rate
        # Lower time in market is better (more efficient)
        efficiency = r.total_return_pct / max(r.time_in_market_pct, 1)

        # Weighted composite
        composite = (
            cagr_score * 0.20 +
            dd_score * 0.15 +
            sharpe_score * 0.20 +
            sortino_score * 0.10 +
            calmar_score * 0.10 +
            pf_score * 0.10 +
            wr_score * 0.10 +
            min(efficiency / 50, 2) * 0.05
        )
        scores.append((r, composite))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def main():
    print("=" * 70)
    print("  BTC/USD 1D CYCLE TRADING STRATEGY BACKTEST")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"  Period: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Candles: {len(df)}")
    print(f"  Price range: ${df['Close'].min():,.0f} - ${df['Close'].max():,.0f}")

    # Configuration
    config = BacktestConfig(
        initial_capital=10000,
        commission_pct=0.001,
        slippage_pct=0.0005,
    )
    print(f"\n  Initial Capital: ${config.initial_capital:,.0f}")
    print(f"  Commission: {config.commission_pct*100:.2f}%")
    print(f"  Slippage: {config.slippage_pct*100:.3f}%")

    # Run backtests
    strategies = get_all_strategies()
    results = []

    for strat in strategies:
        print(f"\nRunning backtest: {strat.name}...")
        result = run_backtest(strat, df, config)
        results.append(result)
        print(result.summary())

    # Buy & Hold reference
    bh_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    print(f"\n  Buy & Hold Return: {bh_return:.1f}%")

    # Comparison table
    print("\n" + "=" * 70)
    print("  STRATEGY COMPARISON TABLE")
    print("=" * 70)
    table = create_comparison_table(results)
    print(table.to_string(index=False))

    # Save comparison table
    table.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"), index=False)
    print(f"\n  Saved comparison.csv")

    # Score and rank strategies
    print("\n" + "=" * 70)
    print("  STRATEGY RANKING (Composite Score)")
    print("=" * 70)
    scored = score_strategies(results)
    for rank, (r, score) in enumerate(scored, 1):
        indicator = " <-- BEST" if rank == 1 else ""
        print(f"  #{rank}: {r.strategy_name} (Score: {score:.3f}){indicator}")

    best_result = scored[0][0]
    print(f"\n  WINNER: {best_result.strategy_name}")
    print(f"    Return: {best_result.total_return_pct:.1f}%")
    print(f"    CAGR: {best_result.cagr_pct:.2f}%")
    print(f"    Sharpe: {best_result.sharpe_ratio:.2f}")
    print(f"    Max DD: {best_result.max_drawdown_pct:.1f}%")

    # Generate plots
    print("\nGenerating charts...")
    plot_equity_curves(results, df)
    for r in results:
        plot_trade_analysis(r, df)

    # Plot best strategy detail
    plot_trade_analysis(best_result, df, "BEST_strategy_trades.png")

    print(f"\nAll results saved to {RESULTS_DIR}/")
    print("Done!")

    return results, scored


if __name__ == "__main__":
    results, scored = main()
