"""
Generate realistic BTC/USD daily OHLCV data based on known historical milestones.
Uses cubic spline interpolation between major price points with added volatility.
"""
import numpy as np
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Known BTC/USD price milestones (date, approximate close price)
MILESTONES = [
    ("2015-01-01", 315),
    ("2015-07-01", 260),
    ("2015-10-15", 255),
    ("2016-01-01", 430),
    ("2016-06-01", 530),
    ("2016-06-18", 760),
    ("2016-08-01", 590),
    ("2016-10-01", 610),
    ("2016-12-31", 960),
    ("2017-03-01", 1190),
    ("2017-05-25", 2400),
    ("2017-07-16", 1930),
    ("2017-09-01", 4900),
    ("2017-09-15", 3200),
    ("2017-11-01", 6700),
    ("2017-12-17", 19800),
    ("2018-02-06", 6900),
    ("2018-03-05", 11500),
    ("2018-04-01", 6800),
    ("2018-05-05", 9800),
    ("2018-06-24", 6100),
    ("2018-07-25", 8300),
    ("2018-08-14", 6200),
    ("2018-09-22", 6700),
    ("2018-11-14", 6300),
    ("2018-11-25", 3700),
    ("2018-12-15", 3200),
    ("2019-02-08", 3400),
    ("2019-04-02", 4900),
    ("2019-06-26", 13800),
    ("2019-09-25", 8300),
    ("2019-10-26", 9500),
    ("2019-12-18", 6600),
    ("2020-01-01", 7200),
    ("2020-02-13", 10300),
    ("2020-03-13", 5000),
    ("2020-05-08", 10000),
    ("2020-07-27", 11070),
    ("2020-09-05", 10200),
    ("2020-10-21", 12800),
    ("2020-11-30", 19700),
    ("2020-12-31", 29000),
    ("2021-01-08", 40700),
    ("2021-01-27", 30400),
    ("2021-02-21", 57500),
    ("2021-02-28", 45100),
    ("2021-03-13", 61200),
    ("2021-04-14", 64800),
    ("2021-05-19", 30000),
    ("2021-06-08", 33500),
    ("2021-06-22", 29000),
    ("2021-07-20", 29500),
    ("2021-08-07", 44500),
    ("2021-09-07", 52700),
    ("2021-09-21", 40000),
    ("2021-10-20", 66000),
    ("2021-11-10", 69000),
    ("2021-12-04", 42000),
    ("2021-12-27", 50500),
    ("2022-01-24", 33000),
    ("2022-02-10", 44000),
    ("2022-03-28", 47500),
    ("2022-04-28", 39000),
    ("2022-05-09", 30000),
    ("2022-05-12", 26700),
    ("2022-06-13", 22500),
    ("2022-06-18", 17600),
    ("2022-08-15", 24400),
    ("2022-09-13", 20200),
    ("2022-11-05", 21100),
    ("2022-11-09", 15800),
    ("2022-11-21", 16200),
    ("2022-12-31", 16500),
    ("2023-01-14", 21100),
    ("2023-02-16", 24600),
    ("2023-03-10", 20100),
    ("2023-03-20", 28000),
    ("2023-04-14", 30400),
    ("2023-05-01", 29200),
    ("2023-06-22", 30700),
    ("2023-07-13", 31500),
    ("2023-08-17", 26000),
    ("2023-09-11", 25800),
    ("2023-10-16", 28500),
    ("2023-10-24", 34100),
    ("2023-11-09", 37000),
    ("2023-12-04", 42000),
    ("2023-12-31", 42300),
    ("2024-01-11", 46600),
    ("2024-02-12", 49700),
    ("2024-03-05", 66800),
    ("2024-03-14", 73800),
    ("2024-03-20", 63800),
    ("2024-04-08", 71500),
    ("2024-04-17", 61300),
    ("2024-05-06", 62700),
    ("2024-05-21", 70600),
    ("2024-06-24", 61300),
    ("2024-07-29", 66700),
    ("2024-08-05", 49500),
    ("2024-08-25", 64000),
    ("2024-09-06", 53900),
    ("2024-09-27", 65600),
    ("2024-10-29", 72300),
    ("2024-11-06", 75000),
    ("2024-11-12", 89500),
    ("2024-11-22", 99000),
    ("2024-12-05", 96800),
    ("2024-12-17", 108000),
    ("2024-12-31", 93500),
    ("2025-01-07", 100000),
    ("2025-01-20", 109000),
    ("2025-01-28", 101500),
    ("2025-02-01", 104500),
]


def generate_btc_daily_data(seed=42):
    """Generate realistic daily BTC/USD OHLCV data."""
    np.random.seed(seed)

    # Parse milestones
    dates = pd.to_datetime([m[0] for m in MILESTONES])
    prices = np.array([m[1] for m in MILESTONES], dtype=float)

    # Create daily date range
    all_dates = pd.date_range(start=dates[0], end=dates[-1], freq="D")

    # Interpolate close prices using log-space cubic interpolation
    log_prices = np.log(prices)
    milestone_days = (dates - dates[0]).days.values.astype(float)
    all_days = (all_dates - dates[0]).days.values.astype(float)

    # Piecewise linear interpolation in log space for base trend
    log_interp = np.interp(all_days, milestone_days, log_prices)

    # Add daily noise (BTC typical daily vol ~3-4%)
    daily_vol = 0.025
    noise = np.random.normal(0, daily_vol, len(all_days))
    # Smooth noise a bit to avoid too much jitter
    from scipy.ndimage import uniform_filter1d
    try:
        noise = uniform_filter1d(noise, size=3)
    except ImportError:
        pass

    # Blend noise into prices but anchor to milestones
    # Create weight that's 0 at milestones and higher between them
    log_close = log_interp + noise * 0.3  # dampened noise
    close_prices = np.exp(log_close)

    # Force exact milestone values
    for i, md in enumerate(milestone_days):
        idx = np.searchsorted(all_days, md)
        if idx < len(close_prices):
            close_prices[idx] = prices[i]

    # Generate OHLV from close
    daily_range_pct = np.abs(np.random.normal(0.02, 0.015, len(all_dates)))
    daily_range_pct = np.clip(daily_range_pct, 0.005, 0.08)

    # Open: previous close with small gap
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0] * 0.998
    gap = np.random.normal(0, 0.003, len(all_dates))
    open_prices = open_prices * (1 + gap)

    # Determine if bullish or bearish candle
    bullish = close_prices > open_prices

    # High and Low
    high_prices = np.where(
        bullish,
        close_prices * (1 + daily_range_pct * np.random.uniform(0.2, 0.8, len(all_dates))),
        open_prices * (1 + daily_range_pct * np.random.uniform(0.2, 0.8, len(all_dates))),
    )
    low_prices = np.where(
        bullish,
        open_prices * (1 - daily_range_pct * np.random.uniform(0.2, 0.8, len(all_dates))),
        close_prices * (1 - daily_range_pct * np.random.uniform(0.2, 0.8, len(all_dates))),
    )

    # Ensure OHLC consistency
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))

    # Generate volume (higher during volatile periods)
    price_changes = np.abs(np.diff(close_prices) / close_prices[:-1])
    price_changes = np.append(price_changes, price_changes[-1])
    base_volume = close_prices * np.random.uniform(500, 2000, len(all_dates))
    vol_multiplier = 1 + price_changes * 20  # More volume on big moves
    volume = base_volume * vol_multiplier
    volume = volume.astype(int)

    df = pd.DataFrame(
        {
            "Open": np.round(open_prices, 2),
            "High": np.round(high_prices, 2),
            "Low": np.round(low_prices, 2),
            "Close": np.round(close_prices, 2),
            "Volume": volume,
        },
        index=all_dates,
    )
    df.index.name = "Date"
    return df


if __name__ == "__main__":
    df = generate_btc_daily_data()
    out_path = os.path.join(DATA_DIR, "btc_usd_daily.csv")
    df.to_csv(out_path)
    print(f"Generated {len(df)} daily candles")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Price range: ${df['Low'].min():,.0f} - ${df['High'].max():,.0f}")
    print(f"\nSample data (last 10 rows):")
    print(df.tail(10))
    print(f"\nYearly summary:")
    yearly = df.groupby(df.index.year).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        AvgVolume=("Volume", "mean"),
    )
    print(yearly)
