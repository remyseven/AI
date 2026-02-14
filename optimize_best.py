"""
Final optimization of the best strategy (V8: Weekly Momentum + Daily Entry).
Tests variations of key parameters to find the optimal configuration.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import product

from backtest_engine import BacktestConfig, run_backtest, compute_indicators, Strategy

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class V8_Optimized(Strategy):
    """V8 with configurable parameters for optimization."""

    def __init__(self, weekly_fast=50, weekly_slow=100, macro_ma=200,
                 rsi_entry_low=40, rsi_entry_high=65,
                 trail_pct_high=0.78, trail_pct_mid=0.82,
                 trail_gain_high=0.50, trail_gain_mid=0.20,
                 stop_loss=-0.18, rsi_blowoff=82, roc_blowoff=80):
        self.name = (f"V8_opt(f{weekly_fast}/s{weekly_slow}/m{macro_ma}/"
                     f"rsi{rsi_entry_low}-{rsi_entry_high}/"
                     f"trail{trail_pct_high}/{trail_pct_mid}/"
                     f"sl{stop_loss})")
        super().__init__({
            "weekly_fast": weekly_fast,
            "weekly_slow": weekly_slow,
            "macro_ma": macro_ma,
            "rsi_entry_low": rsi_entry_low,
            "rsi_entry_high": rsi_entry_high,
            "trail_pct_high": trail_pct_high,
            "trail_pct_mid": trail_pct_mid,
            "trail_gain_high": trail_gain_high,
            "trail_gain_mid": trail_gain_mid,
            "stop_loss": stop_loss,
            "rsi_blowoff": rsi_blowoff,
            "roc_blowoff": roc_blowoff,
        })

    def generate_signals(self, df):
        df["signal"] = 0
        p = self.params

        wf = p["weekly_fast"]
        ws = p["weekly_slow"]
        mm = p["macro_ma"]

        sma_fast = df["Close"].rolling(wf).mean() if f"SMA_{wf}" not in df.columns else df[f"SMA_{wf}"]
        sma_slow = df["Close"].rolling(ws).mean() if f"SMA_{ws}" not in df.columns else df[f"SMA_{ws}"]
        sma_macro = df["Close"].rolling(mm).mean() if f"SMA_{mm}" not in df.columns else df[f"SMA_{mm}"]

        rsi = df["RSI_14"]
        rsi_21 = df["RSI_21"]
        macd_hist = df["MACD_hist"]
        roc_50 = df["ROC_50"]

        in_position = False
        entry_price = 0.0
        highest = 0.0

        start = max(wf, ws, mm) + 10
        for i in range(start, len(df)):
            close = df["Close"].iloc[i]
            prev_close = df["Close"].iloc[i - 1]

            weekly_bull = sma_fast.iloc[i] > sma_slow.iloc[i]
            weekly_bear = sma_fast.iloc[i] < sma_slow.iloc[i]
            macro_bull = close > sma_macro.iloc[i]

            curr_rsi = rsi.iloc[i]
            curr_macd = macd_hist.iloc[i]
            prev_macd = macd_hist.iloc[i - 1]
            curr_roc50 = roc_50.iloc[i]

            if not in_position:
                weekly_turn = weekly_bull and not (sma_fast.iloc[i - 1] > sma_slow.iloc[i - 1])

                pullback = (
                    weekly_bull and macro_bull and
                    curr_rsi < p["rsi_entry_low"] and
                    curr_macd > prev_macd
                )

                reclaim = (
                    macro_bull and not (prev_close > sma_macro.iloc[i - 1]) and
                    curr_rsi > 45 and curr_rsi < p["rsi_entry_high"]
                )

                if weekly_turn or pullback or reclaim:
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                    in_position = True
                    entry_price = close
                    highest = close
            else:
                highest = max(highest, close)
                pnl = (close - entry_price) / entry_price

                weekly_turn_bear = weekly_bear and not (sma_fast.iloc[i - 1] < sma_slow.iloc[i - 1])

                blow_off = (
                    curr_rsi > p["rsi_blowoff"] and
                    curr_roc50 > p["roc_blowoff"] and
                    curr_macd < prev_macd
                )

                trail_triggered = False
                if pnl > p["trail_gain_high"]:
                    trail_triggered = close < highest * p["trail_pct_high"]
                elif pnl > p["trail_gain_mid"]:
                    trail_triggered = close < highest * p["trail_pct_mid"]

                death_cross = (
                    close < sma_macro.iloc[i] and
                    sma_fast.iloc[i] < sma_macro.iloc[i] and
                    curr_rsi < 45
                )

                hard_stop = pnl < p["stop_loss"]

                if weekly_turn_bear or blow_off or trail_triggered or death_cross or hard_stop:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    in_position = False

        return df


def run_parameter_sweep():
    """Test key parameter combinations."""
    df = pd.read_csv(os.path.join(DATA_DIR, "btc_usd_daily.csv"),
                     index_col="Date", parse_dates=True)
    config = BacktestConfig(initial_capital=10000, commission_pct=0.001, slippage_pct=0.0005)

    # Parameter grid - test key dimensions
    param_grid = {
        "weekly_fast": [40, 50, 60],
        "weekly_slow": [90, 100, 120],
        "rsi_entry_low": [35, 40, 45],
        "trail_pct_high": [0.75, 0.78, 0.82],
        "trail_pct_mid": [0.80, 0.82, 0.85],
        "stop_loss": [-0.15, -0.18, -0.22],
    }

    # Fixed params
    fixed = {
        "macro_ma": 200,
        "rsi_entry_high": 65,
        "trail_gain_high": 0.50,
        "trail_gain_mid": 0.20,
        "rsi_blowoff": 82,
        "roc_blowoff": 80,
    }

    # Generate combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    print(f"Testing {len(combinations)} parameter combinations...")

    results = []
    best_score = -999
    best_result = None
    best_params = None

    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        params.update(fixed)

        # Skip invalid combos
        if params["weekly_fast"] >= params["weekly_slow"]:
            continue
        if params["trail_pct_high"] > params["trail_pct_mid"]:
            continue

        strat = V8_Optimized(**params)
        result = run_backtest(strat, df, config)

        # Score
        if result.total_return_pct < 0:
            score = -10
        else:
            cagr_s = min(result.cagr_pct / 15, 4.0)
            dd_s = max(0, 2 - abs(result.max_drawdown_pct) / 25)
            sharpe_s = min(result.sharpe_ratio / 1.0, 4.0)
            sortino_s = min(result.sortino_ratio / 1.5, 3.0)
            calmar_s = min(result.calmar_ratio / 0.5, 4.0)
            pf_s = min(result.profit_factor / 3.0, 3.0)

            score = (cagr_s * 0.25 + dd_s * 0.20 + sharpe_s * 0.25 +
                     sortino_s * 0.10 + calmar_s * 0.10 + pf_s * 0.10)

        results.append((params, result, score))

        if score > best_score:
            best_score = score
            best_result = result
            best_params = params

        if (idx + 1) % 50 == 0:
            print(f"  Tested {idx+1}/{len(combinations)}... Best so far: score={best_score:.3f}")

    # Sort by score
    results.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTested {len(results)} valid combinations")
    print(f"\n{'='*70}")
    print(f"  TOP 10 PARAMETER COMBINATIONS")
    print(f"{'='*70}")

    for rank, (params, result, score) in enumerate(results[:10], 1):
        print(f"\n  #{rank} Score: {score:.3f}")
        print(f"    Params: fast={params['weekly_fast']} slow={params['weekly_slow']} "
              f"rsi_low={params['rsi_entry_low']} trail_h={params['trail_pct_high']} "
              f"trail_m={params['trail_pct_mid']} sl={params['stop_loss']}")
        print(f"    Return: {result.total_return_pct:.0f}%  CAGR: {result.cagr_pct:.1f}%  "
              f"MaxDD: {result.max_drawdown_pct:.1f}%  Sharpe: {result.sharpe_ratio:.2f}  "
              f"Trades: {result.total_trades}  WinRate: {result.win_rate_pct:.0f}%")

    # Save best params
    print(f"\n{'='*70}")
    print(f"  BEST PARAMETERS")
    print(f"{'='*70}")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"\n  Score: {best_score:.3f}")
    print(best_result.summary())

    return best_params, best_result, results


def plot_best_strategy(best_params, df):
    """Generate detailed analysis plot for the best strategy."""
    config = BacktestConfig(initial_capital=10000, commission_pct=0.001, slippage_pct=0.0005)
    strat = V8_Optimized(**best_params)
    strat.name = "BEST: Optimized Weekly Momentum + Daily Entry"
    result = run_backtest(strat, df, config)

    fig, axes = plt.subplots(4, 1, figsize=(18, 24),
                             gridspec_kw={"height_ratios": [3, 1.5, 1, 1]})

    # Panel 1: Equity curve vs Buy & Hold
    ax1 = axes[0]
    eq = result.equity_curve
    bh = 10000 * df["Close"] / df["Close"].iloc[0]

    ax1.plot(eq.index, eq.values, color="#2196F3", linewidth=2, label=f"Strategy ({result.total_return_pct:.0f}%)")
    ax1.plot(df.index, bh, color="#888888", linewidth=1.5, linestyle="--",
             label=f"Buy & Hold ({result.buy_hold_return_pct:.0f}%)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Portfolio Value ($, log)", fontsize=12)
    ax1.set_title("BEST STRATEGY: Optimized BTC/USD 1D Weekly Momentum + Daily Entry",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Shade trade periods
    for trade in result.trades:
        color = "#4CAF5030" if trade.pnl_pct > 0 else "#F4433630"
        ax1.axvspan(trade.entry_date, trade.exit_date, color=color, alpha=0.2)

    # Panel 2: BTC price with entry/exit markers
    ax2 = axes[1]
    ax2.plot(df.index, df["Close"], color="#F7931A", linewidth=1)
    for trade in result.trades:
        ax2.scatter(trade.entry_date, trade.entry_price, marker="^", color="#2196F3", s=80, zorder=5)
        c = "#4CAF50" if trade.pnl_pct > 0 else "#F44336"
        ax2.scatter(trade.exit_date, trade.exit_price, marker="v", color=c, s=80, zorder=5)
    ax2.set_yscale("log")
    ax2.set_ylabel("BTC Price ($)", fontsize=12)
    ax2.set_title("Trade Entry (▲ blue) / Exit (▼ green=win, red=loss)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Drawdown
    ax3 = axes[2]
    peak = eq.expanding().max()
    dd = (eq - peak) / peak * 100
    ax3.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.3)
    ax3.plot(dd.index, dd.values, color="#F44336", linewidth=0.8)
    ax3.set_ylabel("Drawdown %", fontsize=12)
    ax3.set_title(f"Strategy Drawdown (Max: {result.max_drawdown_pct:.1f}%)", fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Per-trade P&L
    ax4 = axes[3]
    if result.trades:
        dates = [t.exit_date for t in result.trades]
        pnls = [t.pnl_pct for t in result.trades]
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
        ax4.bar(dates, pnls, color=colors, width=20, alpha=0.7)
        ax4.axhline(y=0, color="black", linewidth=0.5)
    ax4.set_ylabel("Trade P&L %", fontsize=12)
    ax4.set_title(f"Individual Trade Returns ({result.total_trades} trades, {result.win_rate_pct:.0f}% win rate)", fontsize=11)
    ax4.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "BEST_strategy_detailed.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved BEST_strategy_detailed.png")

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("  PARAMETER OPTIMIZATION - V8: Weekly Momentum + Daily Entry")
    print("=" * 70)

    best_params, best_result, all_results = run_parameter_sweep()

    df = pd.read_csv(os.path.join(DATA_DIR, "btc_usd_daily.csv"),
                     index_col="Date", parse_dates=True)

    print("\nGenerating best strategy chart...")
    final_result = plot_best_strategy(best_params, df)

    # Print final trade list
    print(f"\n{'='*70}")
    print(f"  COMPLETE TRADE LIST - BEST STRATEGY")
    print(f"{'='*70}")
    for j, t in enumerate(final_result.trades):
        days = (t.exit_date - t.entry_date).days
        icon = "WIN " if t.pnl_pct > 0 else "LOSS"
        print(f"  {icon} #{j+1:2d}: {t.entry_date.date()} -> {t.exit_date.date()} "
              f"({days:4d}d)  ${t.entry_price:>10,.0f} -> ${t.exit_price:>10,.0f}  "
              f"P&L: {'+' if t.pnl_pct > 0 else ''}{t.pnl_pct:.1f}%")
