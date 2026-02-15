"""
Backtest runner for V11 Intraday Day Trading Strategy.
Generates 5-minute data, computes intraday indicators, and runs backtest.
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
from strategies_daytrading import (
    V11_IntradayVWAP,
    compute_intraday_indicators,
    get_daytrading_strategies,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 5-minute bars: 288 per day * 365 days = 105,120 per year
PERIODS_PER_YEAR_5MIN = 105120


def load_or_generate_intraday():
    """Load 5-minute data, generating it if needed."""
    path_5min = os.path.join(DATA_DIR, "btc_usd_5min.csv")

    if os.path.exists(path_5min):
        print("  Loading existing 5-minute data...")
        df = pd.read_csv(path_5min, index_col="Datetime", parse_dates=True)
        return df

    # Generate from daily data
    print("  5-minute data not found. Generating from daily data...")
    from generate_intraday_data import generate_intraday_from_daily

    daily_path = os.path.join(DATA_DIR, "btc_usd_daily.csv")
    if not os.path.exists(daily_path):
        print("  ERROR: Daily data not found. Run generate_data.py first.")
        sys.exit(1)

    daily = pd.read_csv(daily_path, index_col="Date", parse_dates=True)
    start = daily.index[-180]  # Last 6 months
    intraday = generate_intraday_from_daily(daily, start_date=start)
    intraday.to_csv(path_5min)
    print(f"  Generated {len(intraday):,} bars -> {path_5min}")
    return intraday


def plot_daytrading_results(result, df, filename="daytrading_v11.png"):
    """Plot day trading backtest results with 4 panels."""
    fig, axes = plt.subplots(4, 1, figsize=(18, 22),
                              gridspec_kw={"height_ratios": [2, 1.2, 0.8, 0.8]})

    # Panel 1: Equity curve
    ax1 = axes[0]
    eq = result.equity_curve
    ax1.plot(eq.index, eq.values, color="#2196F3", linewidth=1.2,
             label=f"{result.strategy_name} ({result.total_return_pct:.1f}%)")

    # Buy & Hold comparison
    bh = 10000 * df["Close"] / df["Close"].iloc[0]
    ax1.plot(df.index, bh, color="#888888", linewidth=1, linestyle="--",
             label=f"Buy & Hold ({(bh.iloc[-1]/10000-1)*100:.1f}%)", alpha=0.7)

    ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax1.set_title(f"BTC/USD 5min - {result.strategy_name}", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Price with trade markers
    ax2 = axes[1]
    # Downsample price for readability (every 12 bars = 1 hour)
    price_hourly = df["Close"].resample("1h").last().dropna()
    ax2.plot(price_hourly.index, price_hourly.values, color="#F7931A", linewidth=0.8)

    # Plot trade entry/exit points
    for trade in result.trades:
        color = "#4CAF50" if trade.pnl_pct > 0 else "#F44336"
        ax2.scatter(trade.entry_date, trade.entry_price, color="#2196F3",
                    marker="^", s=15, zorder=5, alpha=0.7)
        if trade.exit_date is not None:
            ax2.scatter(trade.exit_date, trade.exit_price, color=color,
                        marker="v", s=15, zorder=5, alpha=0.7)

    ax2.set_ylabel("BTC Price ($)")
    ax2.set_title("Price Chart with Trade Markers (^ entry, v exit)")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Drawdown
    ax3 = axes[2]
    peak = eq.expanding().max()
    dd = (eq - peak) / peak * 100
    ax3.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.3)
    ax3.plot(dd.index, dd.values, color="#F44336", linewidth=0.8)
    ax3.set_ylabel("Drawdown %")
    ax3.set_title(f"Drawdown (Max: {result.max_drawdown_pct:.1f}%)")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Per-trade P&L
    ax4 = axes[3]
    if result.trades:
        pnls = [t.pnl_pct for t in result.trades]
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
        ax4.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=0.8)
        ax4.axhline(0, color="black", linewidth=0.5)
        ax4.set_xlabel("Trade #")
        ax4.set_ylabel("P&L %")
        ax4.set_title(f"Per-Trade P&L ({result.total_trades} trades, "
                       f"{result.win_rate_pct:.0f}% win rate)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def plot_daily_pnl(result, filename="daytrading_daily_pnl.png"):
    """Plot daily P&L summary."""
    if not result.trades:
        return

    # Group trades by date
    trade_data = []
    for t in result.trades:
        trade_data.append({
            "date": t.entry_date.date(),
            "pnl_pct": t.pnl_pct,
            "pnl_abs": t.pnl_abs,
        })
    trade_df = pd.DataFrame(trade_data)
    daily_pnl = trade_df.groupby("date").agg(
        total_pnl=("pnl_abs", "sum"),
        num_trades=("pnl_pct", "count"),
        avg_pnl_pct=("pnl_pct", "mean"),
    )

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Daily $ P&L
    ax1 = axes[0]
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in daily_pnl["total_pnl"]]
    ax1.bar(range(len(daily_pnl)), daily_pnl["total_pnl"], color=colors, alpha=0.7)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Daily P&L ($)")
    ax1.set_title("Daily P&L Distribution")
    ax1.grid(True, alpha=0.3)

    # Trades per day
    ax2 = axes[1]
    ax2.bar(range(len(daily_pnl)), daily_pnl["num_trades"], color="#2196F3", alpha=0.6)
    ax2.set_xlabel("Trading Day")
    ax2.set_ylabel("# Trades")
    ax2.set_title("Trades Per Day")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def main():
    print("=" * 70)
    print("  BTC/USD 5min - DAY TRADING BACKTEST (V11)")
    print("=" * 70)

    # Load / generate data
    df_raw = load_or_generate_intraday()
    print(f"  Data: {df_raw.index[0]} to {df_raw.index[-1]} ({len(df_raw):,} bars)")
    print(f"  Price range: ${df_raw['Close'].min():,.0f} - ${df_raw['Close'].max():,.0f}")

    # Compute intraday indicators
    print("\n  Computing intraday indicators...")
    df = compute_intraday_indicators(df_raw)

    # Day trading config (higher commissions for intraday)
    config = BacktestConfig(
        initial_capital=10000,
        commission_pct=0.0006,  # 0.06% maker fee (typical for Binance VIP)
        slippage_pct=0.0003,   # 0.03% slippage (tighter on 5min)
        position_size=1.0,
    )

    # Run strategies
    strategies = get_daytrading_strategies()
    results = []

    for strat in strategies:
        print(f"\n  Running: {strat.name}...")
        result = run_backtest(
            strat, df, config,
            skip_indicators=True,
            periods_per_year=PERIODS_PER_YEAR_5MIN,
        )
        results.append(result)

    # Print results
    for result in results:
        print(result.summary())

        if result.trades:
            # Trade statistics
            pnls = [t.pnl_pct for t in result.trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            print(f"  Day Trading Statistics:")
            print(f"  {'─'*50}")

            # Avg hold time
            hold_times = []
            for t in result.trades:
                if t.exit_date and t.entry_date:
                    hold_minutes = (t.exit_date - t.entry_date).total_seconds() / 60
                    hold_times.append(hold_minutes)

            if hold_times:
                print(f"  Avg hold time:     {np.mean(hold_times):.0f} minutes")
                print(f"  Median hold time:  {np.median(hold_times):.0f} minutes")
                print(f"  Max hold time:     {max(hold_times):.0f} minutes")

            # Daily breakdown
            trade_dates = set()
            for t in result.trades:
                trade_dates.add(t.entry_date.date())
            trading_days = len(trade_dates)
            total_days = (df.index[-1] - df.index[0]).days
            if total_days > 0:
                print(f"  Active trading days: {trading_days} / {total_days} "
                      f"({trading_days/total_days*100:.0f}%)")
            if trading_days > 0:
                print(f"  Avg trades/day:    {len(result.trades)/trading_days:.1f}")

            # Best/worst trades
            print(f"\n  Best trade:  {max(pnls):+.2f}%")
            print(f"  Worst trade: {min(pnls):+.2f}%")
            if wins:
                print(f"  Avg winner:  {np.mean(wins):+.2f}%")
            if losses:
                print(f"  Avg loser:   {np.mean(losses):+.2f}%")

            # Show sample trades
            print(f"\n  Sample trades (first 15):")
            for j, t in enumerate(result.trades[:15]):
                hold = ""
                if t.exit_date and t.entry_date:
                    mins = (t.exit_date - t.entry_date).total_seconds() / 60
                    hold = f" ({mins:.0f}min)"
                icon = "+" if t.pnl_pct > 0 else ""
                print(f"    #{j+1:3d}: {t.entry_date.strftime('%Y-%m-%d %H:%M')} "
                      f"@ ${t.entry_price:,.0f} -> "
                      f"{t.exit_date.strftime('%H:%M')} "
                      f"@ ${t.exit_price:,.0f} = {icon}{t.pnl_pct:.2f}%{hold}")
            if len(result.trades) > 15:
                print(f"    ... and {len(result.trades) - 15} more trades")

    # Generate charts
    print("\n  Generating charts...")
    for result in results:
        plot_daytrading_results(result, df_raw)
        plot_daily_pnl(result)

    # Save comparison CSV
    rows = []
    for r in results:
        rows.append({
            "Strategy": r.strategy_name,
            "Return%": round(r.total_return_pct, 2),
            "CAGR%": round(r.cagr_pct, 2),
            "MaxDD%": round(r.max_drawdown_pct, 2),
            "Sharpe": round(r.sharpe_ratio, 2),
            "Sortino": round(r.sortino_ratio, 2),
            "Calmar": round(r.calmar_ratio, 2),
            "PF": round(r.profit_factor, 2),
            "WinRate%": round(r.win_rate_pct, 1),
            "Trades": r.total_trades,
            "AvgTrade%": round(r.avg_trade_pct, 3),
            "TimeInMkt%": round(r.time_in_market_pct, 1),
        })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, "daytrading_results.csv"), index=False)
    print(f"\n  Results saved to results/daytrading_results.csv")

    print(f"\n{'='*70}")
    print(f"  DAY TRADING BACKTEST COMPLETE")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    results = main()
