"""
BTC/USD Round 3 Strategies - Novel approaches.

V11: Halving Cycle Momentum
  - Uses BTC's known 4-year halving dates to phase entry aggressiveness and
    exit timing. Accumulation phase = buy dips. Bull phase = ride trends.
    Distribution phase = tighter stops.

V12: Pi Cycle Top Detector
  - The Pi Cycle Top indicator (111-day MA vs 350-day MA × 2) has historically
    predicted BTC market tops within days. Uses it as the primary exit signal
    with technical entries.

V13: Supertrend (ATR Bands)
  - Classic Supertrend indicator: ATR-multiple dynamic bands that flip when
    price breaks through. Mechanical, adaptive trend following.
"""
import numpy as np
import pandas as pd
from backtest_engine import Strategy


# Known BTC halving dates
BTC_HALVINGS = [
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-19"),
]


def get_cycle_phase(date):
    """
    Returns (months_since_halving, phase) based on BTC's ~4-year cycle.

    Phases:
        1: Accumulation    (0–12 months post-halving)
        2: Bull run        (12–24 months post-halving)
        3: Distribution    (24–36 months post-halving)
        4: Bear/pre-halving (36+ months post-halving)
    """
    past_halvings = [h for h in BTC_HALVINGS if h <= date]
    if not past_halvings:
        return 0, 1  # Before first halving → treat as accumulation

    last_halving = max(past_halvings)
    months_since = (date - last_halving).days / 30.44

    if months_since < 12:
        phase = 1
    elif months_since < 24:
        phase = 2
    elif months_since < 36:
        phase = 3
    else:
        phase = 4

    return months_since, phase


# =============================================================================
# V11: Halving Cycle Momentum
# =============================================================================
class V11_HalvingCycle(Strategy):
    name = "V11: Halving Cycle Momentum"

    def __init__(self):
        super().__init__({})

    def generate_signals(self, df):
        df["signal"] = 0

        rsi = df["RSI_14"]
        macd_hist = df["MACD_hist"]
        sma50 = df["SMA_50"]
        sma200 = df["SMA_200"]

        in_position = False
        entry_price = 0.0
        highest = 0.0

        for i in range(200, len(df)):
            date = df.index[i]
            close = df["Close"].iloc[i]
            months_since, phase = get_cycle_phase(date)

            curr_rsi = rsi.iloc[i]
            curr_macd = macd_hist.iloc[i]
            prev_macd = macd_hist.iloc[i - 1]
            curr_sma50 = sma50.iloc[i]
            curr_sma200 = sma200.iloc[i]

            macro_bull = close > curr_sma200
            trend_bull = curr_sma50 > curr_sma200

            if not in_position:
                # Entry aggressiveness scales with cycle phase
                if phase == 1:  # Accumulation: buy dips actively
                    entry = (macro_bull and curr_rsi < 55 and curr_macd > prev_macd) or (
                        not macro_bull and curr_rsi < 40 and
                        curr_macd > 0 and curr_macd > prev_macd
                    )
                elif phase == 2:  # Bull run: buy pullbacks + breakouts
                    entry = (
                        macro_bull and trend_bull and
                        curr_rsi < 65 and curr_macd > prev_macd
                    )
                elif phase == 3:  # Distribution: only buy deep dips
                    entry = (
                        macro_bull and trend_bull and
                        curr_rsi < 50 and curr_macd > 0 and curr_macd > prev_macd
                    )
                else:  # Phase 4 bear: very selective
                    entry = (
                        macro_bull and trend_bull and
                        curr_rsi < 45 and curr_macd > prev_macd and curr_macd > 0
                    )

                if entry:
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                    in_position = True
                    entry_price = close
                    highest = close

            else:
                highest = max(highest, close)
                pnl = (close - entry_price) / entry_price

                # Exit rules vary by phase
                if phase == 2:
                    trail_pct, overbought_rsi = 0.25, 82
                elif phase == 3:
                    trail_pct, overbought_rsi = 0.15, 75
                else:
                    trail_pct, overbought_rsi = 0.20, 78

                # Trailing stop activates after 10% gain
                trail_stop = highest * (1 - trail_pct) if pnl > 0.10 else entry_price * 0.85
                trailing_hit = close < trail_stop

                # Overbought + momentum fading
                overbought = curr_rsi > overbought_rsi and curr_macd < prev_macd

                # Bear confirmation
                bear_signal = (
                    not macro_bull and not trend_bull and
                    curr_macd < 0 and curr_rsi < 45
                )

                # Hard stop at -18%
                hard_stop = pnl < -0.18

                if trailing_hit or overbought or bear_signal or hard_stop:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    in_position = False

        return df


# =============================================================================
# V12: Pi Cycle Top Detector
# =============================================================================
# The Pi Cycle Top uses 111-day MA vs 350-day MA × 2.
# When 111 DMA crosses ABOVE 350 DMA × 2, a major top is imminent.
# Historically accurate: 2013, 2017, 2021 tops all flagged within days.
# =============================================================================
class V12_PiCycleTop(Strategy):
    name = "V12: Pi Cycle Top Detector"

    def __init__(self):
        super().__init__({})

    def generate_signals(self, df):
        df["signal"] = 0

        # Pi Cycle components (computed inline — not in base indicators)
        sma111 = df["Close"].rolling(111).mean()
        sma350_x2 = df["Close"].rolling(350).mean() * 2

        sma50 = df["SMA_50"]
        sma200 = df["SMA_200"]
        rsi = df["RSI_14"]
        macd_hist = df["MACD_hist"]

        in_position = False
        entry_price = 0.0
        highest = 0.0

        for i in range(350, len(df)):
            close = df["Close"].iloc[i]

            curr_sma111 = sma111.iloc[i]
            prev_sma111 = sma111.iloc[i - 1]
            curr_sma350x2 = sma350_x2.iloc[i]
            prev_sma350x2 = sma350_x2.iloc[i - 1]

            curr_sma50 = sma50.iloc[i]
            curr_sma200 = sma200.iloc[i]
            curr_rsi = rsi.iloc[i]
            curr_macd = macd_hist.iloc[i]
            prev_macd = macd_hist.iloc[i - 1]

            # Pi Cycle top crossover
            pi_top = curr_sma111 >= curr_sma350x2 and prev_sma111 < prev_sma350x2
            # Warning zone: 111 DMA within 5% of crossing
            pi_warning = curr_sma111 > curr_sma350x2 * 0.95 and curr_rsi > 70

            macro_bull = close > curr_sma200
            trend_bull = curr_sma50 > curr_sma200

            if not in_position:
                # Entry: well below Pi Cycle top zone, recovering with momentum
                not_in_top_zone = curr_sma111 < curr_sma350x2 * 0.85

                recovery = (
                    macro_bull and trend_bull and not_in_top_zone and
                    curr_rsi > 40 and curr_rsi < 65 and curr_macd > prev_macd
                )

                # Deep accumulation: below 200 SMA, RSI very oversold, MACD turning
                deep_accum = (
                    not macro_bull and curr_rsi < 35 and
                    curr_macd > prev_macd and not_in_top_zone
                )

                if recovery or deep_accum:
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                    in_position = True
                    entry_price = close
                    highest = close

            else:
                highest = max(highest, close)
                pnl = (close - entry_price) / entry_price

                # PRIMARY EXIT: Pi Cycle top triggered
                if pi_top or pi_warning:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    in_position = False
                    continue

                # Trailing stop: 20% from peak (activates after 15% gain)
                if pnl > 0.15 and close < highest * 0.80:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    in_position = False
                    continue

                # Bear confirmation exit
                if (not macro_bull and not trend_bull and
                        curr_macd < 0 and curr_rsi < 42):
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    in_position = False
                    continue

                # Hard stop at -15%
                if pnl < -0.15:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    in_position = False

        return df


# =============================================================================
# V13: Supertrend (ATR Bands)
# =============================================================================
# Classic Supertrend: dynamic upper/lower bands based on ATR.
# Flips bullish when price breaks above upper band, bearish when it breaks below.
# One of the cleanest mechanical trend-following approaches.
# =============================================================================
class V13_Supertrend(Strategy):
    name = "V13: Supertrend (ATR Bands)"

    def __init__(self, atr_period=10, atr_mult=3.0):
        super().__init__({"atr_period": atr_period, "atr_mult": atr_mult})

    def generate_signals(self, df):
        atr_period = self.params["atr_period"]
        atr_mult = self.params["atr_mult"]

        df["signal"] = 0

        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        hl2 = (high + low) / 2

        # ATR via EWM
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=atr_period, adjust=False).mean()

        # Raw bands
        upper_raw = hl2 + atr_mult * atr
        lower_raw = hl2 - atr_mult * atr

        # Compute Supertrend direction iteratively
        direction = [0] * len(df)
        final_upper = upper_raw.copy()
        final_lower = lower_raw.copy()

        start = atr_period + 1
        direction[start - 1] = -1  # start bearish

        for i in range(start, len(df)):
            curr_u = upper_raw.iloc[i]
            curr_l = lower_raw.iloc[i]
            prev_u = final_upper.iloc[i - 1]
            prev_l = final_lower.iloc[i - 1]
            prev_close = close.iloc[i - 1]
            curr_close = close.iloc[i]
            prev_dir = direction[i - 1]

            # Upper band: only tighten, never widen while below price
            fu = curr_u if (curr_u < prev_u or prev_close > prev_u) else prev_u
            # Lower band: only rise, never drop while above price
            fl = curr_l if (curr_l > prev_l or prev_close < prev_l) else prev_l

            final_upper.iloc[i] = fu
            final_lower.iloc[i] = fl

            if prev_dir == -1:
                direction[i] = 1 if curr_close > fu else -1
            else:
                direction[i] = -1 if curr_close < fl else 1

        # Generate signals: direction flips + 200 SMA filter
        sma200 = df["SMA_200"]
        rsi = df["RSI_14"]
        in_position = False

        for i in range(max(start, 200), len(df)):
            curr_dir = direction[i]
            prev_dir = direction[i - 1]
            curr_close = close.iloc[i]
            curr_sma200 = sma200.iloc[i]
            curr_rsi = rsi.iloc[i]

            if not in_position:
                # Buy: Supertrend flips bullish, price above 200 SMA, RSI not extreme
                if (curr_dir == 1 and prev_dir == -1 and
                        curr_close > curr_sma200 and curr_rsi < 75):
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                    in_position = True
            else:
                # Sell: Supertrend flips bearish
                if curr_dir == -1 and prev_dir == 1:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    in_position = False

        return df


def get_round3_strategies():
    return [
        V11_HalvingCycle(),
        V12_PiCycleTop(),
        V13_Supertrend(),
    ]
