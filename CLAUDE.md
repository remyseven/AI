# CLAUDE.md

## Project Overview

Bitcoin (BTC/USD) quantitative trading strategy research repository. Develops, backtests, and optimizes algorithmic trading strategies designed to exploit BTC's ~4-year halving cycles. Contains both Python backtesting implementations and TradingView Pine Script versions for live deployment.

## Repository Structure

```
AI/
├── backtest_engine.py            # Core backtesting engine (Trade, BacktestConfig, BacktestResult, Strategy base class)
├── strategies.py                 # Round 1 strategies (V1-V5): Dual MA, RSI+BB, MACD+RSI, Multi-Momentum, Adaptive Cycle
├── strategies_v2.py              # Round 2 strategies (V6-V10): Enhanced with trailing stops, trend filters, risk management
├── strategies_daytrading.py      # V11 intraday VWAP-based strategy for 5-min bars
├── btc_cycle_strategy.py         # Standalone optimized V8 strategy (winner)
├── run_backtest.py               # Round 1 backtest runner + comparison
├── run_backtest_v2.py            # Round 2 backtest runner (all 10 strategies)
├── run_backtest_daytrading.py    # Intraday backtest runner for V11
├── optimize_best.py              # Parameter sweep optimizer (648 combinations)
├── fetch_data.py                 # CoinGecko API data fetcher
├── generate_data.py              # Synthetic daily OHLCV data generator
├── generate_intraday_data.py     # Synthetic 5-min bar expansion from daily data
├── btc_cycle_strategy.pine       # Pine Script v6 – V8 cycle strategy for TradingView
├── daytrading_strategy.pine      # Pine Script v6 – V11 intraday strategy for TradingView
├── data/
│   ├── btc_usd_daily.csv         # Daily OHLCV 2015-02-2025 (~10 years)
│   └── btc_usd_5min.csv          # 5-minute OHLCV 2024-08 to present
└── results/                      # Generated backtest output (PNG charts, CSV metrics)
```

## Tech Stack

- **Language:** Python 3
- **Key Libraries:** pandas, numpy, matplotlib, requests
- **TradingView:** Pine Script v6 for live deployment
- **Data Source:** CoinGecko API, synthetic generation

There is no `requirements.txt`. Dependencies are: `pandas`, `numpy`, `matplotlib`, `requests`.

## Architecture

### Strategy Pattern

All strategies inherit from the `Strategy` base class defined in `backtest_engine.py`:

```
Strategy (base)
├── generate_signals(df) → sets df["signal"] to 1 (buy), -1 (sell), or 0 (hold)
└── Implementations: V1_DualMA, V2_RSI_BB, V3_MACD_RSI, ..., V11_IntradayVWAP
```

### Data Pipeline

```
Raw OHLCV (CSV) → compute_indicators() → generate_signals() → run_backtest() → metrics + charts
```

- `compute_indicators()` in `backtest_engine.py` adds 30+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, etc.)
- Each strategy's `generate_signals(df)` sets a `signal` column using vectorized pandas operations
- `run_backtest()` simulates trade execution with commissions (0.1%) and slippage (0.05%)

### Backtest Configuration Defaults

- Initial capital: $10,000
- Commission: 0.1% per trade
- Slippage: 0.05%
- Position sizing: 100% of capital
- Long-only (no shorting)

## How to Run

```bash
# Fetch live data from CoinGecko
python fetch_data.py

# Generate synthetic data (no API needed)
python generate_data.py
python generate_intraday_data.py

# Run backtests
python run_backtest.py              # Round 1 (V1-V5)
python run_backtest_v2.py           # All strategies (V1-V10)
python run_backtest_daytrading.py   # Intraday V11

# Optimize parameters
python optimize_best.py

# Run standalone optimized strategy
python btc_cycle_strategy.py
```

Output goes to `results/` as PNG charts and CSV metrics files.

## Naming Conventions

- **Strategies:** `V{N}_{Approach}` classes (e.g., `V3_MACD_RSI`, `V11_IntradayVWAP`)
- **Functions:** snake_case (`compute_indicators`, `generate_signals`, `run_backtest`)
- **Constants:** UPPER_SNAKE_CASE (`DATA_DIR`, `RESULTS_DIR`)
- **Classes:** PascalCase for dataclasses (`BacktestConfig`, `BacktestResult`, `Trade`)
- **Result files:** descriptive names in `results/` (e.g., `equity_comparison_all.png`, `comparison.csv`)

## Key Metrics Tracked

Sharpe ratio, Sortino ratio, Calmar ratio, CAGR, max drawdown, win rate, profit factor, average trade %, time in market, max consecutive losses, buy-and-hold comparison.

## Strategy Versions

| Version | Type | Description |
|---------|------|-------------|
| V1 | Daily | Dual MA Crossover + 200-day Trend Filter |
| V2 | Daily | RSI + Bollinger Bands Mean Reversion |
| V3 | Daily | MACD + RSI Momentum |
| V4 | Daily | Multi-Indicator Momentum Composite |
| V5 | Daily | Adaptive Cycle Detection |
| V6 | Daily | Enhanced MA Crossover with Trailing Stops |
| V7 | Daily | RSI Divergence + Volume Confirmation |
| V8 | Daily | **Winner** – Weekly Momentum + Daily Entry (30,931% return over 10yr) |
| V9 | Daily | Breakout + Mean Reversion Hybrid |
| V10 | Daily | Ensemble Voting System |
| V11 | Intraday | VWAP + EMA 9/21 Crossover + RSI(7) for 5-min bars |

## Development Guidelines

- Strategies should subclass `Strategy` from `backtest_engine.py` and implement `generate_signals(df)`
- Use vectorized pandas/numpy operations for indicator and signal computation where possible
- All strategies are long-only by default; set `allow_short=True` in `BacktestConfig` to enable shorting
- Pine Script versions should mirror Python logic and use Pine Script v6 syntax
- Charts use matplotlib with the Agg backend (headless rendering to PNG)
- Data CSV format: columns must include `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

## Testing

There is no automated test suite. Validation is done through backtesting against historical data and visual inspection of equity curves and trade plots in `results/`.

## Git Workflow

- Single branch development
- Commits authored by Claude (noreply@anthropic.com)
- No CI/CD pipeline configured
