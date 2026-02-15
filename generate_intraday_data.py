"""
Generate realistic BTC/USD 5-minute OHLCV data from daily milestones.
Expands daily candles into intraday bars with realistic session patterns,
volume distribution, and microstructure.
"""
import numpy as np
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def generate_intraday_from_daily(daily_df, start_date=None, end_date=None, seed=42):
    """
    Expand daily OHLCV data into 5-minute bars.

    Each day produces 288 five-minute bars (24h crypto market).
    Intraday price action is generated with:
    - Session-based volume patterns (London/NY peaks)
    - Mean-reverting microstructure within the daily range
    - Realistic bid-ask spread simulation
    """
    np.random.seed(seed)

    if start_date:
        daily_df = daily_df[daily_df.index >= start_date]
    if end_date:
        daily_df = daily_df[daily_df.index <= end_date]

    bars_per_day = 288  # 24h * 12 bars/hour

    # Session volume weights (UTC hours 0-23)
    # Peaks during London (8-16 UTC) and NY (14-21 UTC) sessions
    hourly_vol_weights = np.array([
        0.3, 0.25, 0.2, 0.2, 0.2, 0.25, 0.3, 0.5,   # 00-07: Asia tail-off
        0.8, 0.9, 1.0, 0.9, 0.85, 1.2, 1.5, 1.4,     # 08-15: London + overlap
        1.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35,     # 16-23: NY tail-off
    ])
    # Expand to 5-min resolution (12 bars per hour)
    bar_vol_weights = np.repeat(hourly_vol_weights, 12)
    bar_vol_weights = bar_vol_weights / bar_vol_weights.sum()

    all_rows = []

    for day_idx in range(len(daily_df)):
        row = daily_df.iloc[day_idx]
        day_open = row["Open"]
        day_high = row["High"]
        day_low = row["Low"]
        day_close = row["Close"]
        day_volume = row["Volume"]
        day_date = daily_df.index[day_idx]

        # Generate 5-min timestamps for this day
        timestamps = pd.date_range(
            start=day_date, periods=bars_per_day, freq="5min"
        )

        # Generate intraday price path using a bridge process
        # (Brownian bridge from open to close, bounded by high/low)
        daily_range = day_high - day_low
        if daily_range <= 0:
            daily_range = day_open * 0.01

        # Random walk from open to close
        increments = np.random.normal(0, 1, bars_per_day)
        # Brownian bridge: force path to end at close
        bridge = np.cumsum(increments)
        bridge = bridge - np.linspace(0, bridge[-1], bars_per_day)

        # Scale to fit within daily range
        if bridge.max() - bridge.min() > 0:
            bridge = (bridge - bridge.min()) / (bridge.max() - bridge.min())
        else:
            bridge = np.full(bars_per_day, 0.5)

        # Map bridge to price range
        price_range_used = daily_range * 0.85  # Use 85% of range for close prices
        price_offset = day_low + (daily_range - price_range_used) / 2

        close_prices = price_offset + bridge * price_range_used

        # Force first bar open = day_open, last bar close = day_close
        close_prices[0] = day_open + (day_close - day_open) * (1 / bars_per_day)
        close_prices[-1] = day_close

        # Smooth transition
        adjustment = np.linspace(
            day_open - close_prices[0],
            0,
            bars_per_day
        )
        close_prices = close_prices + adjustment

        # Generate OHLC for each 5-min bar
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = day_open

        # High/Low with small wicks
        wick_factor = daily_range / bars_per_day * np.random.uniform(0.5, 2.0, bars_per_day)
        high_prices = np.maximum(open_prices, close_prices) + np.abs(wick_factor * 0.5)
        low_prices = np.minimum(open_prices, close_prices) - np.abs(wick_factor * 0.5)

        # Ensure day high/low are touched at some point
        high_bar = np.random.randint(0, bars_per_day)
        low_bar = np.random.randint(0, bars_per_day)
        high_prices[high_bar] = max(high_prices[high_bar], day_high)
        low_prices[low_bar] = min(low_prices[low_bar], day_low)

        # Ensure OHLC consistency
        high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
        low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

        # Distribute volume according to session weights
        bar_volumes = day_volume * bar_vol_weights
        # Add noise to volume
        vol_noise = np.random.lognormal(0, 0.3, bars_per_day)
        bar_volumes = bar_volumes * vol_noise
        # Rescale so total matches daily volume
        bar_volumes = bar_volumes / bar_volumes.sum() * day_volume
        bar_volumes = bar_volumes.astype(int)
        bar_volumes = np.maximum(bar_volumes, 1)

        for j in range(bars_per_day):
            all_rows.append({
                "Datetime": timestamps[j],
                "Open": round(open_prices[j], 2),
                "High": round(high_prices[j], 2),
                "Low": round(low_prices[j], 2),
                "Close": round(close_prices[j], 2),
                "Volume": int(bar_volumes[j]),
            })

    df = pd.DataFrame(all_rows)
    df.set_index("Datetime", inplace=True)
    df.index.name = "Datetime"
    return df


if __name__ == "__main__":
    # Load daily data
    daily_path = os.path.join(DATA_DIR, "btc_usd_daily.csv")
    if not os.path.exists(daily_path):
        print("Daily data not found. Run generate_data.py first.")
        exit(1)

    daily = pd.read_csv(daily_path, index_col="Date", parse_dates=True)

    # Generate intraday for last 6 months (manageable size)
    # 6 months * 30 days * 288 bars = ~51,840 bars
    start = daily.index[-180]  # Last 180 days
    print(f"Generating 5-minute data from {start.date()} to {daily.index[-1].date()}...")

    intraday = generate_intraday_from_daily(daily, start_date=start)

    out_path = os.path.join(DATA_DIR, "btc_usd_5min.csv")
    intraday.to_csv(out_path)

    print(f"Generated {len(intraday):,} five-minute candles")
    print(f"Date range: {intraday.index[0]} to {intraday.index[-1]}")
    print(f"Price range: ${intraday['Low'].min():,.0f} - ${intraday['High'].max():,.0f}")
    print(f"File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")
    print(f"\nSample data (last 10 bars):")
    print(intraday.tail(10))
