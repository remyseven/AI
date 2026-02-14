"""
BTC/USD 1D Cycle Trading Strategies - Round 2 (Enhanced versions).
Based on learnings from Round 1:
- V1 (Dual MA) had best returns but terrible drawdown (82%)
- V3 (MACD+RSI) had best risk-adjusted returns (7.2% DD, 2.28 Sharpe)
- Need to balance return capture with drawdown control
"""
import numpy as np
import pandas as pd
from backtest_engine import Strategy


# =============================================================================
# V6: Enhanced Dual MA with ATR Trailing Stop
# =============================================================================
# Based on V1 but adds an ATR-based trailing stop to cap drawdowns.
# Also adds a regime filter to avoid buying into late-stage blow-off tops.
# =============================================================================
class V6_DualMA_TrailingStop(Strategy):
    name = "V6: Dual MA + ATR Trailing Stop"

    def __init__(self, fast=50, slow=100, atr_mult=3.0):
        super().__init__({"fast": fast, "slow": slow, "atr_mult": atr_mult})

    def generate_signals(self, df):
        fast = self.params["fast"]
        slow = self.params["slow"]
        atr_mult = self.params["atr_mult"]

        df["signal"] = 0

        fast_ma = df["Close"].rolling(fast).mean()
        slow_ma = df["Close"].rolling(slow).mean()
        sma200 = df["SMA_200"]
        atr = df["ATR_14"]

        in_position = False
        trailing_stop = 0.0
        highest_since_entry = 0.0

        for i in range(max(slow, 200), len(df)):
            close = df["Close"].iloc[i]
            prev_fast = fast_ma.iloc[i - 1]
            prev_slow = slow_ma.iloc[i - 1]
            curr_fast = fast_ma.iloc[i]
            curr_slow = slow_ma.iloc[i]
            curr_atr = atr.iloc[i]
            rsi = df["RSI_14"].iloc[i]

            if not in_position:
                # Buy: fast MA crosses above slow MA, price > 200 SMA
                # Filter: Don't buy if RSI already extremely overbought (blow-off top)
                if (curr_fast > curr_slow and prev_fast <= prev_slow and
                        close > sma200.iloc[i] and rsi < 75):
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                    in_position = True
                    highest_since_entry = close
                    trailing_stop = close - atr_mult * curr_atr
            else:
                # Update trailing stop
                if close > highest_since_entry:
                    highest_since_entry = close
                    trailing_stop = max(trailing_stop, close - atr_mult * curr_atr)

                # Sell conditions:
                # 1. Trailing stop hit
                # 2. Fast MA crosses below slow MA (original signal)
                # 3. Emergency: price drops 25% from high
                emergency_stop = highest_since_entry * 0.75

                if (close < trailing_stop or
                        (curr_fast < curr_slow and prev_fast >= prev_slow) or
                        close < emergency_stop):
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    in_position = False

        return df


# =============================================================================
# V7: MACD Momentum with Trend Confirmation (Enhanced V3)
# =============================================================================
# Based on V3 but holds positions longer using a trailing mechanism
# instead of selling on every MACD cross-down. Only exits on confirmed
# trend reversals, catching more of the big moves.
# =============================================================================
class V7_MACD_Trend(Strategy):
    name = "V7: MACD Momentum + Trend Hold"

    def __init__(self):
        super().__init__({})

    def generate_signals(self, df):
        df["signal"] = 0

        macd_hist = df["MACD_hist"]
        rsi = df["RSI_14"]
        sma50 = df["SMA_50"]
        sma200 = df["SMA_200"]
        atr = df["ATR_14"]

        in_position = False
        entry_price = 0.0
        highest = 0.0

        for i in range(200, len(df)):
            close = df["Close"].iloc[i]
            curr_macd = macd_hist.iloc[i]
            prev_macd = macd_hist.iloc[i - 1]
            curr_rsi = rsi.iloc[i]
            curr_sma50 = sma50.iloc[i]
            curr_sma200 = sma200.iloc[i]

            if not in_position:
                # Buy: MACD histogram turns positive + price above 200 SMA
                # OR: MACD histogram turns positive + 50 SMA > 200 SMA (confirmed uptrend)
                macd_cross_up = curr_macd > 0 and prev_macd <= 0

                trend_ok = (close > curr_sma200) or (curr_sma50 > curr_sma200)
                rsi_ok = curr_rsi > 35 and curr_rsi < 72

                if macd_cross_up and trend_ok and rsi_ok:
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                    in_position = True
                    entry_price = close
                    highest = close
            else:
                highest = max(highest, close)
                pnl = (close - entry_price) / entry_price

                # Sell conditions (more patient than V3):
                # 1. Price drops below 200 SMA AND MACD bearish (confirmed bear)
                bear_confirmed = (close < curr_sma200 and curr_macd < 0 and
                                  curr_sma50 < curr_sma200)

                # 2. RSI extreme overbought + MACD divergence (topping)
                topping = curr_rsi > 80 and curr_macd < prev_macd

                # 3. Trailing stop: give back max 20% from peak, but only
                #    activate after 10% gain to avoid premature stops
                trailing_triggered = False
                if pnl > 0.10:
                    trail_stop = highest * 0.80
                    trailing_triggered = close < trail_stop

                # 4. Hard stop loss at -15%
                hard_stop = pnl < -0.15

                if bear_confirmed or topping or trailing_triggered or hard_stop:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    in_position = False

        return df


# =============================================================================
# V8: Weekly Momentum with Daily Entry (Cycle-focused)
# =============================================================================
# Uses weekly-scale momentum (30/100 day MAs as proxy) to identify BTC's
# ~4-year cycles, then uses daily RSI + MACD for optimal entry timing.
# Designed to ride the big bull runs and exit before deep bears.
# =============================================================================
class V8_WeeklyMomentum(Strategy):
    name = "V8: Weekly Momentum + Daily Entry"

    def __init__(self):
        super().__init__({})

    def generate_signals(self, df):
        df["signal"] = 0

        # Weekly-scale indicators (approximated from daily)
        sma30 = df["SMA_50"]  # ~7-week
        sma100 = df["SMA_100"]  # ~20-week
        sma200 = df["SMA_200"]  # ~40-week

        rsi = df["RSI_14"]
        rsi_21 = df["RSI_21"]
        macd_hist = df["MACD_hist"]
        bb_pct = df["BB_pct"]
        roc_50 = df["ROC_50"]

        in_position = False
        entry_price = 0.0
        highest = 0.0
        days_in_trade = 0

        for i in range(200, len(df)):
            close = df["Close"].iloc[i]
            prev_close = df["Close"].iloc[i - 1]

            # Weekly trend: 50 SMA vs 100 SMA
            weekly_bull = sma30.iloc[i] > sma100.iloc[i]
            weekly_bear = sma30.iloc[i] < sma100.iloc[i]

            # Macro trend: above/below 200 SMA
            macro_bull = close > sma200.iloc[i]

            curr_rsi = rsi.iloc[i]
            curr_rsi21 = rsi_21.iloc[i]
            curr_macd = macd_hist.iloc[i]
            prev_macd = macd_hist.iloc[i - 1]
            curr_roc50 = roc_50.iloc[i]

            if not in_position:
                # === ENTRY CONDITIONS ===
                # Primary: Weekly trend turns bullish + macro confirmation
                weekly_turn = weekly_bull and not (sma30.iloc[i - 1] > sma100.iloc[i - 1])

                # Secondary: Already in weekly uptrend, daily pullback entry
                pullback = (
                    weekly_bull and macro_bull and
                    curr_rsi < 40 and  # daily oversold
                    curr_macd > prev_macd  # MACD turning up
                )

                # Tertiary: Early cycle accumulation
                # Price just reclaimed 200 SMA with improving momentum
                reclaim_200 = (
                    macro_bull and not (prev_close > sma200.iloc[i - 1]) and
                    curr_rsi > 45 and curr_rsi < 65
                )

                if weekly_turn or pullback or reclaim_200:
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                    in_position = True
                    entry_price = close
                    highest = close
                    days_in_trade = 0
            else:
                days_in_trade += 1
                highest = max(highest, close)
                pnl = (close - entry_price) / entry_price

                # === EXIT CONDITIONS ===
                # 1. Weekly trend turns bearish (primary exit)
                weekly_turn_bear = weekly_bear and not (sma30.iloc[i - 1] < sma100.iloc[i - 1])

                # 2. Extreme blow-off top detection
                blow_off = (
                    curr_rsi > 82 and
                    curr_roc50 > 80 and  # 80%+ gain in 50 days
                    curr_macd < prev_macd  # momentum fading
                )

                # 3. Trailing stop: tighter as gains increase
                trail_triggered = False
                if pnl > 0.50:  # After 50% gain
                    trail_stop = highest * 0.78  # 22% trail
                    trail_triggered = close < trail_stop
                elif pnl > 0.20:  # After 20% gain
                    trail_stop = highest * 0.82  # 18% trail
                    trail_triggered = close < trail_stop

                # 4. Death cross + price below 200 SMA
                death_cross = (
                    close < sma200.iloc[i] and
                    sma30.iloc[i] < sma200.iloc[i] and
                    curr_rsi < 45
                )

                # 5. Stop loss at -18%
                hard_stop = pnl < -0.18

                if weekly_turn_bear or blow_off or trail_triggered or death_cross or hard_stop:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    in_position = False

        return df


# =============================================================================
# V9: Hybrid Best-Of: MA Crossover + MACD confirmation + Dynamic Stops
# =============================================================================
# Combines the best elements from all previous versions:
# - MA crossover for cycle direction (from V1)
# - MACD confirmation for timing (from V3)
# - ATR trailing stops for risk management (from V6)
# - Regime awareness for position sizing (from V5)
# =============================================================================
class V9_Hybrid(Strategy):
    name = "V9: Hybrid (MA + MACD + Dynamic Stops)"

    def __init__(self):
        super().__init__({})

    def generate_signals(self, df):
        df["signal"] = 0

        sma50 = df["SMA_50"]
        sma100 = df["SMA_100"]
        sma200 = df["SMA_200"]
        rsi = df["RSI_14"]
        macd = df["MACD"]
        macd_signal = df["MACD_signal"]
        macd_hist = df["MACD_hist"]
        atr = df["ATR_14"]
        bb_pct = df["BB_pct"]
        vol_20 = df["Volatility_20"]

        in_position = False
        entry_price = 0.0
        highest = 0.0

        for i in range(200, len(df)):
            close = df["Close"].iloc[i]

            # Core trend indicators
            trend_bull = sma50.iloc[i] > sma100.iloc[i]
            macro_bull = close > sma200.iloc[i]
            curr_rsi = rsi.iloc[i]
            curr_macd_hist = macd_hist.iloc[i]
            prev_macd_hist = macd_hist.iloc[i - 1]
            curr_atr = atr.iloc[i]

            # Volatility regime
            curr_vol = vol_20.iloc[i] if not pd.isna(vol_20.iloc[i]) else 50
            high_vol = curr_vol > 80  # Annualized vol > 80%

            if not in_position:
                # === ENTRY: Need trend + momentum + not overbought ===

                # Signal 1: Golden cross (50 > 100 SMA) with MACD confirmation
                golden_cross = trend_bull and not (sma50.iloc[i - 1] > sma100.iloc[i - 1])
                macd_bullish = curr_macd_hist > 0 or (curr_macd_hist > prev_macd_hist)

                # Signal 2: Trend pullback (already bullish, oversold RSI bounce)
                pullback_entry = (
                    trend_bull and macro_bull and
                    curr_rsi < 42 and curr_rsi > 25 and  # oversold but not crashing
                    curr_macd_hist > prev_macd_hist  # MACD improving
                )

                # Signal 3: 200 SMA reclaim with momentum
                reclaim = (
                    macro_bull and
                    not (df["Close"].iloc[i - 1] > sma200.iloc[i - 1]) and
                    curr_macd_hist > 0 and
                    curr_rsi > 50 and curr_rsi < 68
                )

                # Don't buy if extremely overbought or in blow-off territory
                not_overheated = curr_rsi < 72 and not (high_vol and curr_rsi > 65)

                if ((golden_cross and macd_bullish) or pullback_entry or reclaim) and not_overheated:
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                    in_position = True
                    entry_price = close
                    highest = close
            else:
                highest = max(highest, close)
                pnl = (close - entry_price) / entry_price

                # === EXIT: Multiple exit conditions ===

                # Dynamic trailing stop based on ATR and gain size
                if pnl > 0.30:
                    # After 30% gain, trail at 2.5x ATR
                    trail_stop = highest - 2.5 * curr_atr
                elif pnl > 0.10:
                    # After 10% gain, trail at 3.5x ATR
                    trail_stop = highest - 3.5 * curr_atr
                else:
                    trail_stop = entry_price - 4.0 * curr_atr  # Wide initial stop

                trailing_hit = close < trail_stop

                # Death cross exit (50 crosses below 100)
                death_cross = (
                    not trend_bull and
                    (sma50.iloc[i - 1] > sma100.iloc[i - 1]) and
                    close < sma200.iloc[i]
                )

                # Blow-off top detection
                blow_off = (
                    curr_rsi > 80 and
                    high_vol and
                    curr_macd_hist < prev_macd_hist and
                    pnl > 0.20  # Only after decent gain
                )

                # Hard stop at -20%
                hard_stop = pnl < -0.20

                # Bearish MACD divergence in overbought territory
                macd_divergence = (
                    curr_rsi > 70 and
                    curr_macd_hist < prev_macd_hist and
                    macd_hist.iloc[i - 1] < macd_hist.iloc[i - 2] and  # 2 consecutive declines
                    not macro_bull  # lost 200 SMA
                )

                if trailing_hit or death_cross or blow_off or hard_stop or macd_divergence:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    in_position = False

        return df


# =============================================================================
# V10: Simple 200 SMA Trend Follower (Optimized Baseline)
# =============================================================================
# Sometimes simple is best. Buy when weekly close > 200 SMA and 50 SMA
# is rising. Sell when weekly close < 200 SMA. Uses the 200 SMA as the
# ultimate cycle indicator for BTC.
# =============================================================================
class V10_Simple200SMA(Strategy):
    name = "V10: Simple 200 SMA Cycle Follower"

    def __init__(self, ma=200, confirmation_days=3):
        super().__init__({"ma": ma, "confirmation_days": confirmation_days})

    def generate_signals(self, df):
        ma = self.params["ma"]
        conf_days = self.params["confirmation_days"]

        df["signal"] = 0

        sma = df["Close"].rolling(ma).mean()
        sma_slope = sma.diff(10)  # 10-day slope of 200 SMA

        in_position = False
        above_count = 0
        below_count = 0

        for i in range(ma + 10, len(df)):
            close = df["Close"].iloc[i]
            curr_sma = sma.iloc[i]
            slope = sma_slope.iloc[i]

            if close > curr_sma:
                above_count += 1
                below_count = 0
            else:
                below_count += 1
                above_count = 0

            if not in_position:
                # Buy after N consecutive closes above 200 SMA + rising SMA
                if above_count >= conf_days and slope > 0:
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                    in_position = True
            else:
                # Sell after N consecutive closes below 200 SMA
                if below_count >= conf_days:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    in_position = False

        return df


def get_round2_strategies():
    return [
        V6_DualMA_TrailingStop(),
        V7_MACD_Trend(),
        V8_WeeklyMomentum(),
        V9_Hybrid(),
        V10_Simple200SMA(),
    ]
