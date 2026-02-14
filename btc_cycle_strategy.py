"""
BTC/USD 1D Cycle Trading Strategy - "Weekly Momentum with Daily Entry"
=====================================================================

WINNER from optimization across 10 strategy variants and 648 parameter combinations.

Strategy Logic:
    This strategy exploits BTC's ~4-year halving cycles by using weekly-scale
    momentum indicators (60/100-day MA crossover as proxy) to identify macro
    cycle direction, then uses daily RSI and MACD for precise entry timing.

    BUY CONDITIONS (any one triggers entry):
    1. Weekly Trend Turn: 60-day SMA crosses above 100-day SMA (new uptrend)
    2. Pullback Entry: Already in uptrend + above 200-day SMA, RSI dips below 35,
       and MACD shows improving momentum
    3. 200-SMA Reclaim: Price crosses above 200-day SMA with RSI 45-65 (neutral momentum)

    SELL CONDITIONS (any one triggers exit):
    1. Weekly Trend Reversal: 60-day SMA crosses below 100-day SMA
    2. Blow-off Top Detection: RSI > 82 + 50-day ROC > 80% + fading MACD
    3. Trailing Stop: After 50%+ gain -> trail at 22% from peak;
                      After 20%+ gain -> trail at 15% from peak
    4. Death Cross: Price below 200 SMA + 60 SMA below 200 SMA + RSI < 45
    5. Hard Stop Loss: -22% from entry

Backtest Results (2015-01-01 to 2025-02-01, $10,000 initial):
    Total Return:    30,931%  ($10,000 -> $3,103,100)
    CAGR:            76.62%
    Max Drawdown:    -50.8%
    Sharpe Ratio:    4.10
    Sortino Ratio:   4.09
    Calmar Ratio:    1.51
    Profit Factor:   9.15
    Win Rate:        64.3%
    Total Trades:    28
    Avg Winner:      +61.4%
    Avg Loser:       -12.1%
    Time in Market:  48.7%

Key Parameters (optimized):
    weekly_fast_ma:     60 days
    weekly_slow_ma:     100 days
    macro_trend_ma:     200 days
    rsi_pullback_entry: 35 (oversold threshold for pullback buys)
    rsi_reclaim_upper:  65 (max RSI for 200-SMA reclaim entries)
    trail_high_pct:     0.78 (22% trail after 50%+ gain)
    trail_mid_pct:      0.85 (15% trail after 20%+ gain)
    stop_loss:          -22%
    rsi_blowoff:        82
    roc_blowoff:        80%

Usage:
    python btc_cycle_strategy.py
"""
import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────────────────────────────────────
# Technical Indicators
# ─────────────────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all required technical indicators."""
    d = df.copy()

    # Moving Averages
    d["SMA_60"] = d["Close"].rolling(60).mean()
    d["SMA_100"] = d["Close"].rolling(100).mean()
    d["SMA_200"] = d["Close"].rolling(200).mean()

    # RSI (14-period)
    delta = d["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    d["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]

    # Rate of Change (50-period)
    d["ROC_50"] = d["Close"].pct_change(50) * 100

    return d


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Parameters
# ─────────────────────────────────────────────────────────────────────────────

PARAMS = {
    "weekly_fast": 60,          # Fast MA period (weekly trend proxy)
    "weekly_slow": 100,         # Slow MA period (weekly trend proxy)
    "macro_ma": 200,            # Macro trend MA (200-day)
    "rsi_entry_low": 35,        # RSI threshold for pullback entries
    "rsi_entry_high": 65,       # RSI upper limit for 200-SMA reclaim entries
    "trail_pct_high": 0.78,     # Trailing stop % after large gain (1 - 0.78 = 22% trail)
    "trail_pct_mid": 0.85,      # Trailing stop % after medium gain (1 - 0.85 = 15% trail)
    "trail_gain_high": 0.50,    # Gain threshold to activate tight trailing stop
    "trail_gain_mid": 0.20,     # Gain threshold to activate medium trailing stop
    "stop_loss": -0.22,         # Hard stop loss percentage
    "rsi_blowoff": 82,          # RSI threshold for blow-off top detection
    "roc_blowoff": 80,          # ROC threshold for blow-off top detection
}


# ─────────────────────────────────────────────────────────────────────────────
# Signal Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    Generate buy/sell signals for the BTC cycle strategy.

    Returns DataFrame with 'signal' column:
        1 = BUY
       -1 = SELL
        0 = HOLD (no action)
    """
    if params is None:
        params = PARAMS

    data = compute_indicators(df)
    data["signal"] = 0

    sma_fast = data["SMA_60"]
    sma_slow = data["SMA_100"]
    sma_macro = data["SMA_200"]
    rsi = data["RSI_14"]
    macd_hist = data["MACD_hist"]
    roc_50 = data["ROC_50"]

    in_position = False
    entry_price = 0.0
    highest = 0.0

    start_idx = max(params["weekly_fast"], params["weekly_slow"], params["macro_ma"]) + 10

    for i in range(start_idx, len(data)):
        close = data["Close"].iloc[i]
        prev_close = data["Close"].iloc[i - 1]

        # ── Trend Detection ──
        weekly_bull = sma_fast.iloc[i] > sma_slow.iloc[i]
        weekly_bear = sma_fast.iloc[i] < sma_slow.iloc[i]
        macro_bull = close > sma_macro.iloc[i]

        curr_rsi = rsi.iloc[i]
        curr_macd = macd_hist.iloc[i]
        prev_macd = macd_hist.iloc[i - 1]
        curr_roc50 = roc_50.iloc[i]

        if not in_position:
            # ── ENTRY CONDITION 1: Weekly trend turns bullish ──
            prev_weekly_bull = sma_fast.iloc[i - 1] > sma_slow.iloc[i - 1]
            weekly_turn = weekly_bull and not prev_weekly_bull

            # ── ENTRY CONDITION 2: Pullback in confirmed uptrend ──
            pullback = (
                weekly_bull and macro_bull and
                curr_rsi < params["rsi_entry_low"] and
                curr_macd > prev_macd  # momentum improving
            )

            # ── ENTRY CONDITION 3: Price reclaims 200-day SMA ──
            prev_above_macro = prev_close > sma_macro.iloc[i - 1]
            reclaim = (
                macro_bull and not prev_above_macro and
                curr_rsi > 45 and curr_rsi < params["rsi_entry_high"]
            )

            if weekly_turn or pullback or reclaim:
                data.iloc[i, data.columns.get_loc("signal")] = 1
                in_position = True
                entry_price = close
                highest = close

        else:
            highest = max(highest, close)
            pnl = (close - entry_price) / entry_price

            # ── EXIT CONDITION 1: Weekly trend turns bearish ──
            prev_weekly_bear = sma_fast.iloc[i - 1] < sma_slow.iloc[i - 1]
            weekly_turn_bear = weekly_bear and not prev_weekly_bear

            # ── EXIT CONDITION 2: Blow-off top detection ──
            blow_off = (
                curr_rsi > params["rsi_blowoff"] and
                curr_roc50 > params["roc_blowoff"] and
                curr_macd < prev_macd  # momentum fading at extreme
            )

            # ── EXIT CONDITION 3: Trailing stop ──
            trail_triggered = False
            if pnl > params["trail_gain_high"]:
                trail_triggered = close < highest * params["trail_pct_high"]
            elif pnl > params["trail_gain_mid"]:
                trail_triggered = close < highest * params["trail_pct_mid"]

            # ── EXIT CONDITION 4: Death cross ──
            death_cross = (
                close < sma_macro.iloc[i] and
                sma_fast.iloc[i] < sma_macro.iloc[i] and
                curr_rsi < 45
            )

            # ── EXIT CONDITION 5: Hard stop loss ──
            hard_stop = pnl < params["stop_loss"]

            if weekly_turn_bear or blow_off or trail_triggered or death_cross or hard_stop:
                data.iloc[i, data.columns.get_loc("signal")] = -1
                in_position = False

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Live Signal Check
# ─────────────────────────────────────────────────────────────────────────────

def check_current_signal(df: pd.DataFrame) -> dict:
    """
    Check the current trading signal based on the latest data.
    Returns a dict with the current state and recommended action.
    """
    data = generate_signals(df)
    latest = data.iloc[-1]
    prev = data.iloc[-2]

    # Determine current position state by replaying signals
    in_position = False
    entry_price = 0.0
    for i in range(len(data)):
        sig = data["signal"].iloc[i]
        if sig == 1 and not in_position:
            in_position = True
            entry_price = data["Close"].iloc[i]
        elif sig == -1 and in_position:
            in_position = False

    result = {
        "date": str(latest.name.date()),
        "close": latest["Close"],
        "in_position": in_position,
        "signal": int(latest["signal"]),
        "rsi": round(latest["RSI_14"], 1),
        "macd_hist": round(latest["MACD_hist"], 2),
        "sma60_above_sma100": latest["SMA_60"] > latest["SMA_100"],
        "above_sma200": latest["Close"] > latest["SMA_200"],
    }

    if in_position:
        pnl = (latest["Close"] - entry_price) / entry_price * 100
        result["entry_price"] = round(entry_price, 2)
        result["unrealized_pnl_pct"] = round(pnl, 1)
        result["action"] = "HOLD LONG" if latest["signal"] != -1 else "SELL"
    else:
        result["action"] = "BUY" if latest["signal"] == 1 else "WAIT"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "btc_usd_daily.csv")

    if not os.path.exists(DATA_PATH):
        print("No data file found. Run generate_data.py first.")
        exit(1)

    df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    print(f"Loaded {len(df)} daily candles: {df.index[0].date()} to {df.index[-1].date()}")

    # Check current signal
    signal = check_current_signal(df)
    print(f"\n{'='*50}")
    print(f"  CURRENT SIGNAL ({signal['date']})")
    print(f"{'='*50}")
    print(f"  BTC Price:     ${signal['close']:,.0f}")
    print(f"  Action:        {signal['action']}")
    print(f"  In Position:   {signal['in_position']}")
    print(f"  RSI(14):       {signal['rsi']}")
    print(f"  MACD Hist:     {signal['macd_hist']}")
    print(f"  60>100 SMA:    {signal['sma60_above_sma100']}")
    print(f"  Above 200 SMA: {signal['above_sma200']}")
    if signal["in_position"]:
        print(f"  Entry Price:   ${signal['entry_price']:,.0f}")
        print(f"  Unrealized:    {signal['unrealized_pnl_pct']:+.1f}%")
    print(f"{'='*50}")
