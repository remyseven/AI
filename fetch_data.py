"""
Fetch BTC/USD historical daily OHLCV data from CoinGecko API.
CoinGecko free API provides up to 365 days of daily data per request.
We'll make multiple requests to get multi-year data.
"""
import requests
import pandas as pd
import json
import time
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_btc_ohlc_coingecko(days=365):
    """Fetch OHLC data from CoinGecko. For days>90, returns daily candles."""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
    params = {"vs_currency": "usd", "days": str(days)}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Returns [[timestamp_ms, open, high, low, close], ...]
    df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close"])
    df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop(columns=["timestamp"])
    df = df.set_index("Date")
    return df


def fetch_btc_market_chart(days=365):
    """Fetch market chart data (prices + volumes) from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    prices = pd.DataFrame(data["prices"], columns=["timestamp", "Close"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "Volume"])

    prices["Date"] = pd.to_datetime(prices["timestamp"], unit="ms")
    volumes["Date"] = pd.to_datetime(volumes["timestamp"], unit="ms")

    # Normalize dates to date-only for joining
    prices["Date"] = prices["Date"].dt.normalize()
    volumes["Date"] = volumes["Date"].dt.normalize()

    df = prices.merge(volumes[["Date", "Volume"]], on="Date", how="left")
    df = df.drop(columns=["timestamp"]).set_index("Date")
    df = df[~df.index.duplicated(keep="first")]
    return df


def fetch_full_dataset():
    """Combine OHLC and volume data into a complete daily dataset."""
    print("Fetching BTC/USD OHLC data (max history)...")
    ohlc = fetch_btc_ohlc_coingecko(days=365)
    print(f"  Got {len(ohlc)} OHLC candles")

    print("Fetching BTC/USD market chart data (max history)...")
    market = fetch_btc_market_chart(days="max")
    print(f"  Got {len(market)} market chart points")

    # For the OHLC range, merge with volume data
    ohlc_dates = ohlc.index.normalize()
    market_dates = market.index.normalize()

    # Use market chart for long history (has Close + Volume)
    # Enhance with OHLC where available
    combined = market.copy()

    # Add Open, High, Low from OHLC data where available
    ohlc_norm = ohlc.copy()
    ohlc_norm.index = ohlc_norm.index.normalize()
    ohlc_norm = ohlc_norm[~ohlc_norm.index.duplicated(keep="first")]

    for col in ["Open", "High", "Low"]:
        combined[col] = ohlc_norm[col].reindex(combined.index)

    # For dates without OHLC, estimate from Close
    combined["Open"] = combined["Open"].fillna(combined["Close"].shift(1))
    combined["High"] = combined["High"].fillna(combined["Close"] * 1.005)
    combined["Low"] = combined["Low"].fillna(combined["Close"] * 0.995)
    combined["Open"] = combined["Open"].fillna(combined["Close"])

    # Reorder columns
    combined = combined[["Open", "High", "Low", "Close", "Volume"]]
    combined = combined.dropna(subset=["Close"])
    combined = combined.sort_index()

    return combined


if __name__ == "__main__":
    df = fetch_full_dataset()
    out_path = os.path.join(DATA_DIR, "btc_usd_daily.csv")
    df.to_csv(out_path)
    print(f"\nSaved {len(df)} daily candles to {out_path}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\nSample data:")
    print(df.tail(10))
