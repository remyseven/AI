"""
BTC/USD Intraday Day Trading Strategy - V11.
Designed for 5-minute timeframe with tight entries, quick exits,
and session-aware logic. Trades are opened and closed within the same day.

Key indicators:
- VWAP (Volume Weighted Average Price) - institutional anchor
- EMA 9/21 crossover - fast trend detection
- RSI(7) - short-period momentum
- Volume spike detection - confirmation of breakouts
- ATR-based stops - volatility-adjusted risk management
- No session restriction — trades can execute 24/7

NOTE: This strategy is designed for REAL exchange data (e.g., Binance 5min candles).
Synthetic data (generate_intraday_data.py) uses Brownian bridge interpolation which
lacks the microstructure, order flow, and session patterns that intraday strategies
exploit. Backtest on synthetic data will underperform. Use real data via:
  - Binance API: GET /api/v3/klines?symbol=BTCUSDT&interval=5m
  - CryptoDataDownload or similar services for historical 5min OHLCV
"""
import numpy as np
import pandas as pd
from backtest_engine import Strategy


def compute_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add intraday-specific indicators to a 5-minute OHLCV dataframe."""
    d = df.copy()

    # Fast EMAs for intraday trend
    for period in [9, 21, 50]:
        d[f"EMA_{period}"] = d["Close"].ewm(span=period, adjust=False).mean()

    # RSI (7-period for fast responsiveness)
    for period in [7, 14]:
        delta = d["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss
        d[f"RSI_{period}"] = 100 - (100 / (1 + rs))

    # MACD (fast settings: 8/17/9 instead of 12/26/9)
    ema_fast = d["Close"].ewm(span=8, adjust=False).mean()
    ema_slow = d["Close"].ewm(span=17, adjust=False).mean()
    d["MACD"] = ema_fast - ema_slow
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]

    # ATR (14-period on 5min bars = ~70 minutes of data)
    high_low = d["High"] - d["Low"]
    high_close = (d["High"] - d["Close"].shift()).abs()
    low_close = (d["Low"] - d["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d["ATR_14"] = tr.rolling(14).mean()
    d["ATR_pct"] = d["ATR_14"] / d["Close"] * 100

    # VWAP (reset daily)
    d["Date"] = d.index.date
    d["TypicalPrice"] = (d["High"] + d["Low"] + d["Close"]) / 3
    d["TP_Volume"] = d["TypicalPrice"] * d["Volume"]
    d["Cum_TP_Vol"] = d.groupby("Date")["TP_Volume"].cumsum()
    d["Cum_Vol"] = d.groupby("Date")["Volume"].cumsum()
    d["VWAP"] = d["Cum_TP_Vol"] / d["Cum_Vol"]

    # VWAP bands (1 and 2 standard deviations)
    d["VWAP_sq_diff"] = ((d["TypicalPrice"] - d["VWAP"]) ** 2) * d["Volume"]
    d["Cum_sq_diff"] = d.groupby("Date")["VWAP_sq_diff"].cumsum()
    d["VWAP_std"] = np.sqrt(d["Cum_sq_diff"] / d["Cum_Vol"])
    d["VWAP_upper1"] = d["VWAP"] + d["VWAP_std"]
    d["VWAP_lower1"] = d["VWAP"] - d["VWAP_std"]
    d["VWAP_upper2"] = d["VWAP"] + 2 * d["VWAP_std"]
    d["VWAP_lower2"] = d["VWAP"] - 2 * d["VWAP_std"]

    # Volume analysis
    d["Vol_SMA_20"] = d["Volume"].rolling(20).mean()
    d["Vol_ratio"] = d["Volume"] / d["Vol_SMA_20"]

    # Bollinger Bands (20-period on 5min)
    d["BB_mid"] = d["Close"].rolling(20).mean()
    bb_std = d["Close"].rolling(20).std()
    d["BB_upper"] = d["BB_mid"] + 2 * bb_std
    d["BB_lower"] = d["BB_mid"] - 2 * bb_std

    # Candle body ratio (for pin bar / doji detection)
    body = (d["Close"] - d["Open"]).abs()
    full_range = d["High"] - d["Low"]
    d["BodyRatio"] = body / full_range.replace(0, np.nan)

    # Clean up temp columns
    d.drop(columns=["Date", "TypicalPrice", "TP_Volume", "Cum_TP_Vol",
                     "Cum_Vol", "VWAP_sq_diff", "Cum_sq_diff"], inplace=True)

    return d


# =============================================================================
# V11: Intraday VWAP + EMA Momentum Day Trading
# =============================================================================
# Designed for 5-minute BTC/USD charts. Combines:
#   - VWAP as institutional price anchor
#   - 9/21 EMA crossover for fast trend detection
#   - RSI(7) for momentum confirmation
#   - Volume spike detection for breakout confirmation
#   - ATR-based dynamic stops and targets
#   - No session restriction — trades can execute 24/7
#
# Entry types:
#   1. VWAP Bounce: Price pulls back to VWAP, bounces with volume
#   2. EMA Momentum: 9 EMA crosses above 21 EMA with RSI confirmation
#   3. Volume Breakout: Price breaks above resistance with 2x+ volume
#
# Exit types:
#   1. Take profit at 1.5-2x ATR from entry
#   2. Stop loss at 1x ATR from entry (1.5:1 to 2:1 R:R)
#   3. RSI exhaustion (>78 for longs)
#   4. Trailing stop after 1x ATR profit reached
# =============================================================================
class V11_IntradayVWAP(Strategy):
    name = "V11: Intraday VWAP + EMA Day Trading"

    def __init__(self, atr_tp_mult=3.0, atr_sl_mult=1.2, vol_spike=2.0,
                 rsi_entry_low=35, rsi_entry_high=62, rsi_exit=80,
                 trail_activation_atr=1.5, trail_pct=0.003,
                 max_trades_per_day=3):
        super().__init__({
            "atr_tp_mult": atr_tp_mult,
            "atr_sl_mult": atr_sl_mult,
            "vol_spike": vol_spike,
            "rsi_entry_low": rsi_entry_low,
            "rsi_entry_high": rsi_entry_high,
            "rsi_exit": rsi_exit,
            "trail_activation_atr": trail_activation_atr,
            "trail_pct": trail_pct,
            "max_trades_per_day": max_trades_per_day,
        })

    def generate_signals(self, df):
        p = self.params
        df["signal"] = 0

        # Use intraday indicators
        ema9 = df["EMA_9"]
        ema21 = df["EMA_21"]
        ema50 = df["EMA_50"]
        rsi = df["RSI_7"]
        macd_hist = df["MACD_hist"]
        atr = df["ATR_14"]
        vwap = df["VWAP"]
        vol_ratio = df["Vol_ratio"]
        bb_lower = df["BB_lower"]
        bb_upper = df["BB_upper"]

        in_position = False
        entry_price = 0.0
        highest = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        trailing_active = False
        trailing_stop = 0.0
        current_date = None
        trades_today = 0
        losses_today = 0
        cooldown_bars = 0  # Wait N bars after a loss before re-entering

        warmup = 50  # Need enough bars for indicators

        for i in range(warmup, len(df)):
            close = df["Close"].iloc[i]
            bar_date = df.index[i].date()

            # Reset daily counters
            if bar_date != current_date:
                current_date = bar_date
                trades_today = 0
                losses_today = 0

            # Cooldown decrement
            if cooldown_bars > 0:
                cooldown_bars -= 1

            curr_atr = atr.iloc[i]
            if pd.isna(curr_atr) or curr_atr <= 0:
                continue

            curr_rsi = rsi.iloc[i]
            curr_ema9 = ema9.iloc[i]
            curr_ema21 = ema21.iloc[i]
            curr_ema50 = ema50.iloc[i]
            curr_vwap = vwap.iloc[i]
            curr_vol = vol_ratio.iloc[i]
            curr_macd = macd_hist.iloc[i]
            prev_macd = macd_hist.iloc[i - 1] if i > 0 else 0
            if not in_position:
                # Skip if max daily trades reached
                if trades_today >= p["max_trades_per_day"]:
                    continue
                # Cooldown after losses (avoid revenge trading)
                if cooldown_bars > 0:
                    continue
                # Stop trading for the day after 2 losses
                if losses_today >= 2:
                    continue

                # ---- ENTRY SIGNAL 1: VWAP Bounce ----
                # Price pulled back to VWAP in established trend, bouncing with volume
                vwap_bounce = (
                    close > curr_vwap and
                    df["Low"].iloc[i] <= curr_vwap * 1.002 and  # Touched VWAP
                    curr_ema9 > curr_ema21 and  # Short-term trend up
                    curr_rsi > 40 and curr_rsi < p["rsi_entry_high"] and
                    curr_vol > 1.3 and  # Above-average volume
                    curr_macd > 0  # MACD positive (momentum aligned)
                )

                # ---- ENTRY SIGNAL 2: EMA Momentum Crossover ----
                prev_ema9 = ema9.iloc[i - 1]
                prev_ema21 = ema21.iloc[i - 1]
                ema_cross = (
                    curr_ema9 > curr_ema21 and
                    prev_ema9 <= prev_ema21 and  # Fresh cross
                    close > curr_vwap and  # Above VWAP (institutional bias)
                    close > curr_ema50 and  # Above 50 EMA (trend filter)
                    curr_rsi > 45 and curr_rsi < p["rsi_entry_high"] and
                    curr_macd > prev_macd and  # MACD improving
                    curr_vol > 1.2  # Needs some volume confirmation
                )

                # ---- ENTRY SIGNAL 3: Volume Breakout ----
                # Big volume spike above recent high with strong trend
                recent_high = df["High"].iloc[max(0, i-12):i].max()
                vol_breakout = (
                    close > recent_high and
                    curr_vol >= p["vol_spike"] and  # Volume spike
                    close > curr_vwap and
                    curr_ema9 > curr_ema21 and
                    curr_ema21 > curr_ema50 and  # Strong trend filter
                    curr_rsi > 50 and curr_rsi < 72
                )

                # ---- ENTRY SIGNAL 4: Bollinger Band Mean Reversion ----
                # Price touched lower BB, now bouncing back with volume
                curr_bb_lower = bb_lower.iloc[i]
                curr_bb_upper = bb_upper.iloc[i]
                bb_reversion = (
                    df["Low"].iloc[i] <= curr_bb_lower and  # Touched lower band
                    close > curr_bb_lower and  # Bouncing back
                    close > df["Open"].iloc[i] and  # Bullish candle
                    curr_rsi < 35 and  # Oversold
                    curr_vol > 1.5 and  # Volume confirmation
                    curr_ema21 > curr_ema50  # Overall trend still up
                )

                if vwap_bounce or ema_cross or vol_breakout or bb_reversion:
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                    in_position = True
                    entry_price = close
                    highest = close
                    stop_loss = close - p["atr_sl_mult"] * curr_atr
                    take_profit = close + p["atr_tp_mult"] * curr_atr
                    trailing_active = False
                    trailing_stop = 0.0
                    trades_today += 1

            else:
                # Position management
                highest = max(highest, close)
                pnl = (close - entry_price) / entry_price

                # Activate trailing stop after reaching 1x ATR profit
                entry_atr = atr.iloc[i]
                if not trailing_active and (close - entry_price) >= p["trail_activation_atr"] * entry_atr:
                    trailing_active = True
                    trailing_stop = close * (1 - p["trail_pct"])

                if trailing_active:
                    trailing_stop = max(trailing_stop, close * (1 - p["trail_pct"]))

                # ---- EXIT CONDITIONS ----

                # 1. Take profit hit
                tp_hit = close >= take_profit

                # 2. Stop loss hit
                sl_hit = close <= stop_loss

                # 3. Trailing stop hit
                trail_hit = trailing_active and close <= trailing_stop

                # 4. RSI exhaustion
                rsi_exit = curr_rsi >= p["rsi_exit"]

                # 5. EMA death cross while in solid profit (trend reversal)
                ema_reversal = (
                    curr_ema9 < curr_ema21 and
                    ema9.iloc[i - 1] >= ema21.iloc[i - 1] and
                    pnl > 0.003  # Only exit on cross if meaningful profit
                )

                # 6. Price drops below VWAP lower band with momentum failing
                vwap_lower = df["VWAP_lower1"].iloc[i]
                vwap_fail = (
                    close < vwap_lower and
                    pnl < -0.003 and
                    curr_macd < prev_macd  # Momentum confirming weakness
                )

                if tp_hit or sl_hit or trail_hit or rsi_exit or ema_reversal or vwap_fail:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    in_position = False
                    # Track losses and set cooldown
                    if pnl < 0:
                        losses_today += 1
                        cooldown_bars = 6  # Wait 30 minutes (6 x 5min) after a loss

        return df


def get_daytrading_strategies():
    """Return list of day trading strategies."""
    return [V11_IntradayVWAP()]
