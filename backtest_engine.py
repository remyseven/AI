"""
Custom backtesting engine for BTC/USD trading strategies.
Supports long-only strategies on daily timeframe with realistic simulation.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp = None
    exit_price: float = None
    size: float = 1.0  # fraction of capital
    side: str = "long"
    pnl_pct: float = 0.0
    pnl_abs: float = 0.0


@dataclass
class BacktestConfig:
    initial_capital: float = 10000.0
    commission_pct: float = 0.001  # 0.1% per trade (realistic for crypto)
    slippage_pct: float = 0.0005  # 0.05% slippage
    position_size: float = 1.0  # fraction of capital per trade
    allow_short: bool = False


@dataclass
class BacktestResult:
    strategy_name: str
    config: BacktestConfig
    trades: list = field(default_factory=list)
    equity_curve: pd.Series = None
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    max_consecutive_losses: int = 0
    time_in_market_pct: float = 0.0
    buy_hold_return_pct: float = 0.0
    calmar_ratio: float = 0.0

    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"  Strategy: {self.strategy_name}\n"
            f"{'='*60}\n"
            f"  Total Return:       {self.total_return_pct:>10.1f}%\n"
            f"  Buy & Hold Return:  {self.buy_hold_return_pct:>10.1f}%\n"
            f"  CAGR:               {self.cagr_pct:>10.2f}%\n"
            f"  Max Drawdown:       {self.max_drawdown_pct:>10.1f}%\n"
            f"  Sharpe Ratio:       {self.sharpe_ratio:>10.2f}\n"
            f"  Sortino Ratio:      {self.sortino_ratio:>10.2f}\n"
            f"  Calmar Ratio:       {self.calmar_ratio:>10.2f}\n"
            f"  Profit Factor:      {self.profit_factor:>10.2f}\n"
            f"  Win Rate:           {self.win_rate_pct:>10.1f}%\n"
            f"  Total Trades:       {self.total_trades:>10d}\n"
            f"  Avg Trade:          {self.avg_trade_pct:>10.2f}%\n"
            f"  Avg Win:            {self.avg_win_pct:>10.2f}%\n"
            f"  Avg Loss:           {self.avg_loss_pct:>10.2f}%\n"
            f"  Max Consec. Losses: {self.max_consecutive_losses:>10d}\n"
            f"  Time in Market:     {self.time_in_market_pct:>10.1f}%\n"
            f"{'='*60}\n"
        )


class Strategy:
    """Base class for all trading strategies."""

    name = "BaseStrategy"

    def __init__(self, params=None):
        self.params = params or {}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.
        Must add a 'signal' column to df:
          1 = buy/go long
         -1 = sell/go flat (close long)
          0 = no action / hold current position
        """
        raise NotImplementedError


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to dataframe."""
    d = df.copy()

    # Simple Moving Averages
    for period in [10, 20, 50, 100, 150, 200]:
        d[f"SMA_{period}"] = d["Close"].rolling(period).mean()

    # Exponential Moving Averages
    for period in [10, 12, 20, 21, 26, 50, 100, 200]:
        d[f"EMA_{period}"] = d["Close"].ewm(span=period, adjust=False).mean()

    # RSI
    for period in [7, 14, 21]:
        delta = d["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss
        d[f"RSI_{period}"] = 100 - (100 / (1 + rs))

    # MACD
    d["MACD"] = d["EMA_12"] - d["EMA_26"]
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_signal"]

    # Bollinger Bands (20-period)
    d["BB_mid"] = d["SMA_20"]
    bb_std = d["Close"].rolling(20).std()
    d["BB_upper"] = d["BB_mid"] + 2 * bb_std
    d["BB_lower"] = d["BB_mid"] - 2 * bb_std
    d["BB_pct"] = (d["Close"] - d["BB_lower"]) / (d["BB_upper"] - d["BB_lower"])

    # ATR (14-period)
    high_low = d["High"] - d["Low"]
    high_close = (d["High"] - d["Close"].shift()).abs()
    low_close = (d["Low"] - d["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d["ATR_14"] = tr.rolling(14).mean()
    d["ATR_pct"] = d["ATR_14"] / d["Close"] * 100

    # Stochastic RSI
    rsi14 = d["RSI_14"]
    stoch_period = 14
    rsi_low = rsi14.rolling(stoch_period).min()
    rsi_high = rsi14.rolling(stoch_period).max()
    d["StochRSI"] = (rsi14 - rsi_low) / (rsi_high - rsi_low)
    d["StochRSI_K"] = d["StochRSI"].rolling(3).mean()
    d["StochRSI_D"] = d["StochRSI_K"].rolling(3).mean()

    # On-Balance Volume
    obv = (np.sign(d["Close"].diff()) * d["Volume"]).fillna(0).cumsum()
    d["OBV"] = obv
    d["OBV_SMA_20"] = d["OBV"].rolling(20).mean()

    # Rate of Change
    for period in [10, 20, 50]:
        d[f"ROC_{period}"] = d["Close"].pct_change(period) * 100

    # Volatility (rolling std of returns)
    d["Volatility_20"] = d["Close"].pct_change().rolling(20).std() * np.sqrt(365) * 100

    # Distance from 200-day MA (percent)
    d["Dist_SMA200_pct"] = (d["Close"] - d["SMA_200"]) / d["SMA_200"] * 100

    # Weekly and Monthly performance
    d["Return_7d"] = d["Close"].pct_change(7) * 100
    d["Return_30d"] = d["Close"].pct_change(30) * 100

    # Accumulation/Distribution oscillator
    clv = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / (d["High"] - d["Low"])
    clv = clv.fillna(0)
    d["ADL"] = (clv * d["Volume"]).cumsum()

    return d


def run_backtest(
    strategy: Strategy, df: pd.DataFrame, config: BacktestConfig = None,
    skip_indicators: bool = False, periods_per_year: int = 365
) -> BacktestResult:
    """Run a backtest for the given strategy on the provided data.

    Args:
        periods_per_year: For annualization. 365 for daily, ~105120 for 5min crypto.
    """
    if config is None:
        config = BacktestConfig()

    # Generate signals
    if skip_indicators:
        data = df.copy()
    else:
        data = compute_indicators(df.copy())
    data = strategy.generate_signals(data)

    # Drop NaN rows (from indicator warmup)
    data = data.dropna(subset=["signal"])

    capital = config.initial_capital
    position = 0  # 0 = flat, 1 = long
    entry_price = 0.0
    entry_date = None
    trades = []
    equity = [capital]
    equity_dates = [data.index[0]]
    days_in_market = 0

    for i in range(len(data)):
        date = data.index[i]
        close = data["Close"].iloc[i]
        signal = data["signal"].iloc[i]

        if position == 1:
            days_in_market += 1

        # Entry signal
        if signal == 1 and position == 0:
            entry_price = close * (1 + config.slippage_pct)  # slippage on entry
            entry_cost = capital * config.commission_pct
            capital -= entry_cost
            entry_date = date
            position = 1

        # Exit signal
        elif signal == -1 and position == 1:
            exit_price = close * (1 - config.slippage_pct)  # slippage on exit
            pnl_pct = (exit_price - entry_price) / entry_price
            pnl_abs = capital * config.position_size * pnl_pct
            exit_cost = (capital + pnl_abs) * config.commission_pct
            capital = capital + pnl_abs - exit_cost

            trade = Trade(
                entry_date=entry_date,
                entry_price=entry_price,
                exit_date=date,
                exit_price=exit_price,
                pnl_pct=pnl_pct * 100,
                pnl_abs=pnl_abs,
            )
            trades.append(trade)
            position = 0

        # Track equity
        if position == 1:
            unrealized_pnl = capital * config.position_size * (close - entry_price) / entry_price
            equity.append(capital + unrealized_pnl)
        else:
            equity.append(capital)
        equity_dates.append(date)

    # Close any open position at the end
    if position == 1:
        close = data["Close"].iloc[-1]
        exit_price = close * (1 - config.slippage_pct)
        pnl_pct = (exit_price - entry_price) / entry_price
        pnl_abs = capital * config.position_size * pnl_pct
        capital = capital + pnl_abs - (capital + pnl_abs) * config.commission_pct
        trades.append(
            Trade(
                entry_date=entry_date,
                entry_price=entry_price,
                exit_date=data.index[-1],
                exit_price=exit_price,
                pnl_pct=pnl_pct * 100,
                pnl_abs=pnl_abs,
            )
        )

    equity_series = pd.Series(equity, index=equity_dates[:len(equity)])
    equity_series = equity_series[~equity_series.index.duplicated(keep="last")]

    # Compute metrics
    result = BacktestResult(
        strategy_name=strategy.name,
        config=config,
        trades=trades,
        equity_curve=equity_series,
    )

    # Total return
    final_capital = equity[-1]
    result.total_return_pct = (final_capital / config.initial_capital - 1) * 100

    # Buy & Hold return
    result.buy_hold_return_pct = (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100

    # CAGR
    years = (data.index[-1] - data.index[0]).days / 365.25
    if years > 0 and final_capital > 0:
        result.cagr_pct = (pow(final_capital / config.initial_capital, 1 / years) - 1) * 100

    # Max Drawdown
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    result.max_drawdown_pct = drawdown.min() * 100

    # Sharpe Ratio (returns annualized)
    daily_returns = equity_series.pct_change().dropna()
    annualize_factor = np.sqrt(periods_per_year)
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        result.sharpe_ratio = daily_returns.mean() / daily_returns.std() * annualize_factor

    # Sortino Ratio
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        result.sortino_ratio = daily_returns.mean() / downside_returns.std() * annualize_factor

    # Calmar Ratio
    if result.max_drawdown_pct != 0:
        result.calmar_ratio = result.cagr_pct / abs(result.max_drawdown_pct)

    # Trade statistics
    result.total_trades = len(trades)
    if trades:
        pnls = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.win_rate_pct = len(wins) / len(pnls) * 100
        result.avg_trade_pct = np.mean(pnls)

        if wins:
            result.avg_win_pct = np.mean(wins)
        if losses:
            result.avg_loss_pct = np.mean(losses)

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max consecutive losses
        consec = 0
        max_consec = 0
        for p in pnls:
            if p <= 0:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0
        result.max_consecutive_losses = max_consec

    # Time in market
    total_days = len(data)
    if total_days > 0:
        result.time_in_market_pct = days_in_market / total_days * 100

    return result
