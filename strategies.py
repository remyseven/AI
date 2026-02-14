"""
BTC/USD 1D Cycle Trading Strategies.
All strategies aim to buy low in bear/accumulation phases and sell high in distribution/euphoria phases.
"""
import numpy as np
import pandas as pd
from backtest_engine import Strategy


# =============================================================================
# VERSION 1: Dual Moving Average Crossover with 200-day Trend Filter
# =============================================================================
# Logic: Use fast/slow MA crossover for entries/exits but only buy when price
# is above the 200-day SMA (confirmed uptrend). This avoids buying in deep
# bear markets while capturing the major uptrend moves.
# =============================================================================
class V1_DualMA(Strategy):
    name = "V1: Dual MA Crossover + Trend Filter"

    def __init__(self, fast=50, slow=100, trend=200):
        super().__init__({"fast": fast, "slow": slow, "trend": trend})

    def generate_signals(self, df):
        fast = self.params["fast"]
        slow = self.params["slow"]
        trend = self.params["trend"]

        df["signal"] = 0

        fast_ma = df[f"SMA_{fast}"] if f"SMA_{fast}" in df.columns else df["Close"].rolling(fast).mean()
        slow_ma = df[f"SMA_{slow}"] if f"SMA_{slow}" in df.columns else df["Close"].rolling(slow).mean()
        trend_ma = df[f"SMA_{trend}"] if f"SMA_{trend}" in df.columns else df["Close"].rolling(trend).mean()

        # Buy when fast MA crosses above slow MA AND price > trend MA
        buy_signal = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1)) & (df["Close"] > trend_ma)
        # Sell when fast MA crosses below slow MA
        sell_signal = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1
        return df


# =============================================================================
# VERSION 2: RSI + Bollinger Bands Mean Reversion
# =============================================================================
# Logic: Buy when RSI is oversold AND price touches lower Bollinger Band
# (mean reversion in oversold conditions). Sell when RSI is overbought AND
# price touches upper Bollinger Band. This buys dips and sells rips.
# =============================================================================
class V2_RSI_BB(Strategy):
    name = "V2: RSI + Bollinger Bands Mean Reversion"

    def __init__(self, rsi_buy=35, rsi_sell=70, bb_period=20, bb_std=2.0):
        super().__init__({
            "rsi_buy": rsi_buy, "rsi_sell": rsi_sell,
            "bb_period": bb_period, "bb_std": bb_std
        })

    def generate_signals(self, df):
        rsi_buy = self.params["rsi_buy"]
        rsi_sell = self.params["rsi_sell"]

        df["signal"] = 0

        rsi = df["RSI_14"]

        # Buy: RSI oversold AND price near/below lower Bollinger Band
        buy_signal = (rsi < rsi_buy) & (df["Close"] <= df["BB_lower"] * 1.02)
        # Sell: RSI overbought AND price near/above upper Bollinger Band
        sell_signal = (rsi > rsi_sell) & (df["Close"] >= df["BB_upper"] * 0.98)

        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1
        return df


# =============================================================================
# VERSION 3: MACD + RSI Momentum Cycle Strategy
# =============================================================================
# Logic: Use MACD for cycle detection (histogram turning positive = new up cycle)
# combined with RSI confirmation. Buy when MACD histogram turns positive and
# RSI is not overbought. Sell when MACD histogram turns negative or RSI
# reaches extreme overbought levels.
# =============================================================================
class V3_MACD_RSI(Strategy):
    name = "V3: MACD + RSI Momentum Cycles"

    def __init__(self, rsi_upper=75, rsi_lower=40):
        super().__init__({"rsi_upper": rsi_upper, "rsi_lower": rsi_lower})

    def generate_signals(self, df):
        rsi_upper = self.params["rsi_upper"]
        rsi_lower = self.params["rsi_lower"]

        df["signal"] = 0

        macd_hist = df["MACD_hist"]
        rsi = df["RSI_14"]

        # MACD histogram crossing above zero (momentum shifting bullish)
        macd_cross_up = (macd_hist > 0) & (macd_hist.shift(1) <= 0)
        # MACD histogram crossing below zero (momentum shifting bearish)
        macd_cross_down = (macd_hist < 0) & (macd_hist.shift(1) >= 0)

        # Buy: MACD turns bullish AND RSI not overbought
        buy_signal = macd_cross_up & (rsi < rsi_upper) & (rsi > rsi_lower)
        # Sell: MACD turns bearish OR RSI extremely overbought
        sell_signal = macd_cross_down | (rsi > 80)

        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1
        return df


# =============================================================================
# VERSION 4: Multi-Period Momentum (200d MA + RSI + Volume)
# =============================================================================
# Logic: A trend-following approach that uses the 200-day MA as the primary
# cycle indicator, RSI for timing, and volume for confirmation. Buy when
# price reclaims the 200-day MA with RSI momentum and volume confirmation.
# Sell when price loses the 200-day MA or shows distribution.
# =============================================================================
class V4_MultiMomentum(Strategy):
    name = "V4: Multi-Period Momentum + Volume"

    def __init__(self, ma_period=200, rsi_period=14, vol_mult=1.2):
        super().__init__({"ma_period": ma_period, "rsi_period": rsi_period, "vol_mult": vol_mult})

    def generate_signals(self, df):
        ma_period = self.params["ma_period"]
        vol_mult = self.params["vol_mult"]

        df["signal"] = 0

        sma200 = df[f"SMA_{ma_period}"]
        rsi = df["RSI_14"]
        vol_avg = df["Volume"].rolling(50).mean()

        # Price crossing above 200 SMA
        price_above_ma = df["Close"] > sma200
        price_was_below = df["Close"].shift(1) <= sma200.shift(1)

        # Recently crossed above (within last 5 days)
        crossed_above = price_above_ma & price_was_below
        recently_crossed = crossed_above.rolling(5).sum() > 0

        # Volume confirmation
        high_volume = df["Volume"] > vol_avg * vol_mult

        # Buy: Price reclaims 200 SMA with RSI momentum and volume
        buy_signal = recently_crossed & (rsi > 45) & (rsi < 70) & high_volume
        # Also buy on strong pullbacks to 200 SMA in uptrend
        pullback_buy = (
            price_above_ma &
            (df["Close"] < sma200 * 1.05) &
            (df["Close"] > sma200 * 0.99) &
            (rsi < 45) &
            (df["Close"] > df["Close"].shift(1))  # turning up
        )
        buy_signal = buy_signal | pullback_buy

        # Sell: Price drops below 200 SMA with confirmation
        price_below_ma = df["Close"] < sma200
        sell_signal = price_below_ma & (rsi < 45)
        # Also sell on extreme overbought
        sell_signal = sell_signal | ((rsi > 80) & (df["Close"] > sma200 * 1.5))

        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1
        return df


# =============================================================================
# VERSION 5: Adaptive Cycle Strategy with Dynamic Thresholds
# =============================================================================
# Logic: This is the most sophisticated strategy. It adapts to market regimes
# using volatility-adjusted indicators. It identifies accumulation zones
# (low volatility + oversold) for buying and distribution zones (high
# volatility + overbought) for selling. Uses multiple confirmation signals.
# =============================================================================
class V5_AdaptiveCycle(Strategy):
    name = "V5: Adaptive Cycle (Dynamic Thresholds)"

    def __init__(self):
        super().__init__({})

    def generate_signals(self, df):
        df["signal"] = 0

        # Market regime detection via volatility
        vol_20 = df["Volatility_20"]
        vol_median = vol_20.rolling(100).median()
        low_vol = vol_20 < vol_median  # Accumulation regime
        high_vol = vol_20 > vol_median * 1.5  # Distribution regime

        # Trend detection
        sma50 = df["SMA_50"]
        sma200 = df["SMA_200"]
        uptrend = sma50 > sma200
        downtrend = sma50 < sma200

        # Momentum
        rsi = df["RSI_14"]
        macd_hist = df["MACD_hist"]
        stoch_k = df["StochRSI_K"]

        # Adaptive RSI thresholds based on volatility
        rsi_buy_threshold = np.where(low_vol, 40, 30)  # More lenient in low vol
        rsi_sell_threshold = np.where(high_vol, 65, 75)  # Tighter in high vol

        # === BUY CONDITIONS ===
        # Condition 1: Accumulation zone (low vol, oversold, trend turning)
        accumulation_buy = (
            (rsi < rsi_buy_threshold) &
            (macd_hist > macd_hist.shift(1)) &  # MACD improving
            (df["Close"] > df["Close"].shift(3))  # Price turning up
        )

        # Condition 2: Trend confirmation (uptrend with pullback)
        trend_buy = (
            uptrend &
            (rsi < 45) &
            (stoch_k < 0.3) &
            (df["Close"] > sma200)
        )

        # Condition 3: Golden cross with momentum
        golden_cross = uptrend & (~uptrend.shift(1).fillna(False))
        momentum_buy = golden_cross & (rsi > 40) & (rsi < 65)

        buy_signal = accumulation_buy | trend_buy | momentum_buy

        # === SELL CONDITIONS ===
        # Condition 1: Distribution zone (high vol, overbought)
        distribution_sell = (
            high_vol &
            (rsi > rsi_sell_threshold) &
            (macd_hist < macd_hist.shift(1))  # MACD deteriorating
        )

        # Condition 2: Trend breakdown
        death_cross = downtrend & (~downtrend.shift(1).fillna(True))
        trend_sell = death_cross & (rsi < 55)

        # Condition 3: Extreme overbought with volume divergence
        extreme_sell = (
            (rsi > 78) &
            (stoch_k > 0.8) &
            (df["Close"] > df["BB_upper"])
        )

        # Condition 4: Stop-loss - price drops below key support
        support_break = (
            (df["Close"] < sma200) &
            (df["Close"].shift(1) >= sma200.shift(1)) &
            (rsi < 40)
        )

        sell_signal = distribution_sell | trend_sell | extreme_sell | support_break

        df.loc[buy_signal, "signal"] = 1
        df.loc[sell_signal, "signal"] = -1
        return df


# =============================================================================
# Get all strategies for comparison
# =============================================================================
def get_all_strategies():
    return [
        V1_DualMA(),
        V2_RSI_BB(),
        V3_MACD_RSI(),
        V4_MultiMomentum(),
        V5_AdaptiveCycle(),
    ]
