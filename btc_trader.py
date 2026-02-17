"""
BTC/USD 5min Intraday Day Trading Bot - V11
Alpaca API implementation of the Pine Script VWAP + EMA Momentum strategy.

Usage:
    pip install requests python-dotenv numpy
    python btc_trader.py              # paper trading (default)
    python btc_trader.py --live       # live trading (use with caution)
"""

import os
import time
import math
import logging
import argparse
from datetime import datetime, timezone, timedelta
from collections import deque
from dotenv import load_dotenv
import requests
import numpy as np

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (mirrors Pine Script inputs)
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Entry settings
    "ema_fast_len": 9,
    "ema_mid_len": 21,
    "ema_slow_len": 50,
    "rsi_len": 7,
    "rsi_entry_hi": 62,
    "vol_spike": 2.0,
    "bb_len": 20,
    "bb_mult": 2.0,

    # Exit settings
    "atr_len": 14,
    "atr_tp_mult": 3.0,
    "atr_sl_mult": 1.2,
    "trail_act_atr": 1.5,
    "trail_pct": 0.3 / 100,  # 0.3% as decimal
    "rsi_exit_lvl": 80,

    # Risk management
    "max_trades": 3,
    "max_losses": 2,
    "cooldown_bars": 6,

    # Session settings (UTC hours)
    "london_open": 7,
    "ny_close": 22,
    "session_end": 21,

    # Bot settings
    "symbol": "BTC/USD",
    "bar_interval_sec": 300,  # 5 minutes
    "qty_pct": 1.0,           # 100% of available cash
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("btc_trader")

# ─────────────────────────────────────────────────────────────────────────────
# ALPACA API CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaClient:
    """Minimal Alpaca REST client for crypto trading."""

    def __init__(self, base_url, api_key, secret_key):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        })
        # Data API for market data (bars, quotes)
        if "paper" in base_url:
            self.data_url = "https://data.alpaca.markets/v1beta3/crypto/us"
        else:
            self.data_url = "https://data.alpaca.markets/v1beta3/crypto/us"

    def _request(self, method, url, **kwargs):
        resp = self.session.request(method, url, **kwargs)
        if resp.status_code not in (200, 204):
            log.error("API %s %s -> %s: %s", method, url, resp.status_code, resp.text)
            resp.raise_for_status()
        return resp.json() if resp.content else {}

    # Account
    def get_account(self):
        return self._request("GET", f"{self.base_url}/account")

    # Positions
    def get_position(self, symbol):
        sym = symbol.replace("/", "")
        resp = self.session.get(f"{self.base_url}/positions/{sym}")
        if resp.status_code == 404:
            return None  # No open position — expected
        if resp.status_code not in (200, 204):
            log.error("API GET positions/%s -> %s: %s", sym, resp.status_code, resp.text)
            resp.raise_for_status()
        return resp.json()

    # Orders
    def submit_order(self, symbol, qty, side, order_type="market", time_in_force="gtc"):
        payload = {
            "symbol": symbol.replace("/", ""),
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        log.info("Submitting order: %s", payload)
        return self._request("POST", f"{self.base_url}/orders", json=payload)

    def close_position(self, symbol):
        sym = symbol.replace("/", "")
        log.info("Closing position: %s", sym)
        return self._request("DELETE", f"{self.base_url}/positions/{sym}")

    # Market data — crypto bars
    def get_bars(self, symbol, timeframe="5Min", limit=100):
        """Fetch historical crypto bars from Alpaca data API."""
        # Build URL manually — requests would encode the slash in BTC/USD
        url = f"{self.data_url}/bars?symbols={symbol}&timeframe={timeframe}&limit={limit}&sort=asc"
        data = self._request("GET", url)
        return data.get("bars", {}).get(symbol, [])

    def get_latest_quote(self, symbol):
        # Build URL manually to preserve slash in BTC/USD
        url = f"{self.data_url}/latest/quotes?symbols={symbol}"
        data = self._request("GET", url)
        return data.get("quotes", {}).get(symbol)


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def ema(values, period):
    """Exponential moving average."""
    if len(values) < period:
        return [np.nan] * len(values)
    result = [np.nan] * (period - 1)
    k = 2.0 / (period + 1)
    sma = np.mean(values[:period])
    result.append(sma)
    for v in values[period:]:
        sma = v * k + sma * (1 - k)
        result.append(sma)
    return result


def sma(values, period):
    """Simple moving average."""
    result = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(np.nan)
        else:
            result.append(np.mean(values[i - period + 1:i + 1]))
    return result


def rsi(closes, period):
    """Relative Strength Index."""
    if len(closes) < period + 1:
        return [np.nan] * len(closes)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    result = [np.nan] * period
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        result.append(100.0)
    else:
        rs = avg_gain / avg_loss
        result.append(100 - 100 / (1 + rs))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100 - 100 / (1 + rs))

    return result


def macd(closes, fast=8, slow=17, signal=9):
    """MACD with histogram. Returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = [f - s if not (np.isnan(f) or np.isnan(s)) else np.nan
                 for f, s in zip(ema_fast, ema_slow)]

    valid_macd = [v for v in macd_line if not np.isnan(v)]
    signal_line_vals = ema(valid_macd, signal) if len(valid_macd) >= signal else [np.nan] * len(valid_macd)

    # Align signal line with macd_line
    signal_line = [np.nan] * (len(macd_line) - len(signal_line_vals)) + signal_line_vals

    histogram = [m - s if not (np.isnan(m) or np.isnan(s)) else np.nan
                 for m, s in zip(macd_line, signal_line)]

    return macd_line, signal_line, histogram


def atr(highs, lows, closes, period):
    """Average True Range."""
    if len(highs) < 2:
        return [np.nan] * len(highs)
    trs = [highs[0] - lows[0]]
    for i in range(1, len(highs)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    # RMA (Wilder's smoothing) for ATR
    result = [np.nan] * (period - 1)
    avg = np.mean(trs[:period])
    result.append(avg)
    for i in range(period, len(trs)):
        avg = (avg * (period - 1) + trs[i]) / period
        result.append(avg)
    return result


def stdev(values, period):
    """Rolling standard deviation."""
    result = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(np.nan)
        else:
            window = values[i - period + 1:i + 1]
            result.append(np.std(window, ddof=0))
    return result


def vwap(highs, lows, closes, volumes):
    """Session VWAP (cumulative from start of bars)."""
    hlc3 = [(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)]
    cum_vol = 0.0
    cum_pv = 0.0
    result = []
    for i in range(len(hlc3)):
        cum_vol += volumes[i]
        cum_pv += hlc3[i] * volumes[i]
        result.append(cum_pv / cum_vol if cum_vol > 0 else hlc3[i])
    return result


def highest(values, period):
    """Rolling highest value (shifted by 1 to avoid lookahead)."""
    result = [np.nan]
    for i in range(1, len(values)):
        start = max(0, i - period)
        result.append(max(values[start:i]))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class StrategyState:
    """Tracks position and risk management state across bars."""

    def __init__(self):
        self.entry_price = None
        self.highest_since = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_active = False
        self.trailing_stop = None
        self.trades_today = 0
        self.losses_today = 0
        self.cooldown_left = 0
        self.in_position = False
        self.last_trade_date = None
        self.prev_ema9 = None
        self.prev_ema21 = None


def compute_signals(bars, state, cfg):
    """
    Given OHLCV bars and current state, compute entry/exit signals
    for the latest bar. Returns a dict with signal info.
    """
    if len(bars) < cfg["ema_slow_len"] + 10:
        return {"buy": False, "sell": False, "reason": "insufficient data"}

    opens = [b["o"] for b in bars]
    highs = [b["h"] for b in bars]
    lows = [b["l"] for b in bars]
    closes = [b["c"] for b in bars]
    volumes = [b["v"] for b in bars]

    # Current bar index
    i = -1

    # Indicators
    ema9 = ema(closes, cfg["ema_fast_len"])
    ema21 = ema(closes, cfg["ema_mid_len"])
    ema50 = ema(closes, cfg["ema_slow_len"])
    rsi7 = rsi(closes, cfg["rsi_len"])
    macd_l, macd_s, macd_h = macd(closes, 8, 17, 9)
    atr14 = atr(highs, lows, closes, cfg["atr_len"])
    vwap_vals = vwap(highs, lows, closes, volumes)
    bb_basis = sma(closes, cfg["bb_len"])
    bb_dev_vals = stdev(closes, cfg["bb_len"])
    vol_sma20 = sma(volumes, 20)
    recent_high = highest(highs, 12)

    # Current values
    c = closes[i]
    o = opens[i]
    h = highs[i]
    lo = lows[i]
    v = volumes[i]

    ema9_v = ema9[i]
    ema21_v = ema21[i]
    ema50_v = ema50[i]
    rsi_v = rsi7[i]
    macd_hist = macd_h[i]
    macd_hist_prev = macd_h[-2] if len(macd_h) >= 2 else np.nan
    atr_v = atr14[i]
    vwap_v = vwap_vals[i]
    vol_ratio = v / vol_sma20[i] if vol_sma20[i] and not np.isnan(vol_sma20[i]) and vol_sma20[i] > 0 else 0

    # VWAP bands
    vwap_close_diff = [closes[j] - vwap_vals[j] for j in range(len(closes))]
    vwap_dev_vals = stdev(vwap_close_diff, cfg["bb_len"])
    vwap_lower1 = vwap_v - (vwap_dev_vals[i] if not np.isnan(vwap_dev_vals[i]) else 0)

    # Bollinger Bands
    bb_lower = bb_basis[i] - cfg["bb_mult"] * bb_dev_vals[i] if not np.isnan(bb_dev_vals[i]) else 0
    recent_high_v = recent_high[i] if not np.isnan(recent_high[i]) else 0

    # Detect new day — reset counters
    bar_time = bars[-1].get("t", "")
    if bar_time:
        today = bar_time[:10]
        if state.last_trade_date != today:
            state.trades_today = 0
            state.losses_today = 0
            state.cooldown_left = 0
            state.last_trade_date = today

    # Decrement cooldown
    if state.cooldown_left > 0:
        state.cooldown_left -= 1

    # Session detection
    utc_hour = datetime.now(timezone.utc).hour
    in_london = utc_hour >= cfg["london_open"] and utc_hour < 16
    in_ny = utc_hour >= 13 and utc_hour < cfg["ny_close"]
    high_liq = in_london or in_ny
    near_session_end = utc_hour >= (cfg["session_end"] - 1)
    at_session_end = utc_hour >= cfg["session_end"]

    # NaN checks
    if any(np.isnan(x) for x in [ema9_v, ema21_v, ema50_v, rsi_v, macd_hist, atr_v, vwap_v]):
        return {"buy": False, "sell": False, "reason": "indicators not ready"}

    # ── ENTRY SIGNALS ──
    can_trade = (high_liq and not near_session_end
                 and state.trades_today < cfg["max_trades"]
                 and state.losses_today < cfg["max_losses"]
                 and state.cooldown_left == 0)

    # Signal 1: VWAP Bounce
    vwap_bounce = (c > vwap_v and lo <= vwap_v * 1.002
                   and ema9_v > ema21_v and rsi_v > 40 and rsi_v < cfg["rsi_entry_hi"]
                   and vol_ratio > 1.3 and macd_hist > 0)

    # Signal 2: EMA Crossover
    ema_cross_raw = (state.prev_ema9 is not None
                     and state.prev_ema9 <= state.prev_ema21
                     and ema9_v > ema21_v)
    ema_cross = (ema_cross_raw and c > vwap_v and c > ema50_v
                 and rsi_v > 45 and rsi_v < cfg["rsi_entry_hi"]
                 and macd_hist > macd_hist_prev and vol_ratio > 1.2)

    # Signal 3: Volume Breakout
    vol_breakout = (c > recent_high_v and vol_ratio >= cfg["vol_spike"]
                    and c > vwap_v and ema9_v > ema21_v and ema21_v > ema50_v
                    and rsi_v > 50 and rsi_v < 72)

    # Signal 4: BB Mean Reversion
    bb_reversion = (lo <= bb_lower and c > bb_lower and c > o
                    and rsi_v < 35 and vol_ratio > 1.5 and ema21_v > ema50_v)

    any_entry = vwap_bounce or ema_cross or vol_breakout or bb_reversion
    buy_signal = can_trade and not state.in_position and any_entry

    entry_type = ("VWAP Bounce" if vwap_bounce else
                  "EMA Cross" if ema_cross else
                  "Vol Breakout" if vol_breakout else
                  "BB Reversion" if bb_reversion else "Unknown")

    # ── EXIT SIGNALS ──
    sell_signal = False
    exit_reason = ""

    if state.in_position and state.entry_price:
        pnl = (c - state.entry_price) / state.entry_price
        state.highest_since = max(state.highest_since or c, c)

        # Trailing stop activation
        if not state.trailing_active and (c - state.entry_price) >= cfg["trail_act_atr"] * atr_v:
            state.trailing_active = True
            state.trailing_stop = c * (1 - cfg["trail_pct"])

        if state.trailing_active:
            state.trailing_stop = max(state.trailing_stop or 0, c * (1 - cfg["trail_pct"]))

        # EMA crossunder (death cross)
        ema_cross_down = (state.prev_ema9 is not None
                          and state.prev_ema9 >= state.prev_ema21
                          and ema9_v < ema21_v)

        # Exit conditions
        tp_hit = c >= state.take_profit
        sl_hit = c <= state.stop_loss
        trail_hit = state.trailing_active and state.trailing_stop and c <= state.trailing_stop
        rsi_exit = rsi_v >= cfg["rsi_exit_lvl"]
        session_close = at_session_end
        ema_reversal = ema_cross_down and pnl > 0.003
        vwap_fail = (c < vwap_lower1 and pnl < -0.003
                     and not np.isnan(macd_hist_prev) and macd_hist < macd_hist_prev)

        if tp_hit:
            sell_signal, exit_reason = True, "Take Profit"
        elif sl_hit:
            sell_signal, exit_reason = True, "Stop Loss"
        elif trail_hit:
            sell_signal, exit_reason = True, "Trail Stop"
        elif rsi_exit:
            sell_signal, exit_reason = True, "RSI Exit"
        elif session_close:
            sell_signal, exit_reason = True, "Session Close"
        elif ema_reversal:
            sell_signal, exit_reason = True, "EMA Reversal"
        elif vwap_fail:
            sell_signal, exit_reason = True, "VWAP Fail"

    # Save EMA values for crossover detection on next bar
    state.prev_ema9 = ema9_v
    state.prev_ema21 = ema21_v

    return {
        "buy": buy_signal,
        "sell": sell_signal,
        "entry_type": entry_type if buy_signal else None,
        "exit_reason": exit_reason if sell_signal else None,
        "price": c,
        "atr": atr_v,
        "rsi": rsi_v,
        "vol_ratio": vol_ratio,
        "ema9": ema9_v,
        "ema21": ema21_v,
        "vwap": vwap_v,
        "macd_hist": macd_hist,
        "session": "London/NY" if (in_london and in_ny) else "London" if in_london else "NY" if in_ny else "Off",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRADING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_bot(client, cfg):
    state = StrategyState()
    symbol = cfg["symbol"]

    # Sync position state with broker on startup
    pos = client.get_position(symbol)
    if pos:
        state.in_position = True
        state.entry_price = float(pos["avg_entry_price"])
        state.highest_since = float(pos["current_price"])
        state.stop_loss = state.entry_price - cfg["atr_sl_mult"] * 100  # rough estimate
        state.take_profit = state.entry_price + cfg["atr_tp_mult"] * 100
        log.info("Existing position detected: entry=%.2f qty=%s", state.entry_price, pos["qty"])

    log.info("Bot started — symbol=%s, interval=%ds", symbol, cfg["bar_interval_sec"])
    log.info("Risk: max %d trades/day, max %d losses/day, %d-bar cooldown",
             cfg["max_trades"], cfg["max_losses"], cfg["cooldown_bars"])

    while True:
        try:
            # Fetch enough bars for the slowest indicator (EMA 50 + margin)
            bars_needed = cfg["ema_slow_len"] + cfg["bb_len"] + 20
            raw_bars = client.get_bars(symbol, timeframe="5Min", limit=bars_needed)

            if not raw_bars:
                log.warning("No bars returned, retrying in 30s...")
                time.sleep(30)
                continue

            # Normalize bar keys
            bars = []
            for b in raw_bars:
                bars.append({
                    "t": b.get("t", ""),
                    "o": float(b["o"]),
                    "h": float(b["h"]),
                    "l": float(b["l"]),
                    "c": float(b["c"]),
                    "v": float(b["v"]),
                })

            signals = compute_signals(bars, state, cfg)

            # Log current state
            log.info(
                "Bar: price=%.2f | RSI=%.1f | Vol=%.1fx | MACD_H=%.2f | EMA9=%.2f EMA21=%.2f | VWAP=%.2f | Session=%s",
                signals["price"], signals.get("rsi", 0), signals.get("vol_ratio", 0),
                signals.get("macd_hist", 0), signals.get("ema9", 0), signals.get("ema21", 0),
                signals.get("vwap", 0), signals.get("session", "?"),
            )

            # ── EXECUTE BUY ──
            if signals["buy"] and not state.in_position:
                acct = client.get_account()
                cash = float(acct["cash"])
                price = signals["price"]
                qty = round((cash * cfg["qty_pct"]) / price, 4)

                if qty > 0:
                    log.info(">>> BUY SIGNAL: %s | price=%.2f | qty=%.4f",
                             signals["entry_type"], price, qty)
                    client.submit_order(symbol, qty, "buy")

                    state.in_position = True
                    state.entry_price = price
                    state.highest_since = price
                    state.stop_loss = price - cfg["atr_sl_mult"] * signals["atr"]
                    state.take_profit = price + cfg["atr_tp_mult"] * signals["atr"]
                    state.trailing_active = False
                    state.trailing_stop = None
                    state.trades_today += 1

                    log.info("    SL=%.2f | TP=%.2f | ATR=%.2f",
                             state.stop_loss, state.take_profit, signals["atr"])

            # ── EXECUTE SELL ──
            elif signals["sell"] and state.in_position:
                pnl = (signals["price"] - state.entry_price) / state.entry_price * 100
                log.info(">>> SELL SIGNAL: %s | price=%.2f | PnL=%.2f%%",
                         signals["exit_reason"], signals["price"], pnl)

                client.close_position(symbol)

                if (signals["price"] - state.entry_price) < 0:
                    state.losses_today += 1
                    state.cooldown_left = cfg["cooldown_bars"]
                    log.info("    Loss recorded. Losses today: %d. Cooldown: %d bars",
                             state.losses_today, state.cooldown_left)

                state.in_position = False
                state.entry_price = None
                state.highest_since = None
                state.stop_loss = None
                state.take_profit = None
                state.trailing_active = False
                state.trailing_stop = None

            elif state.in_position:
                pnl = (signals["price"] - state.entry_price) / state.entry_price * 100
                log.info("    In position: PnL=%.2f%% | SL=%.2f | TP=%.2f | Trail=%s",
                         pnl, state.stop_loss, state.take_profit,
                         f"{state.trailing_stop:.2f}" if state.trailing_stop else "inactive")

            # Wait for next bar
            now = datetime.now(timezone.utc)
            seconds_into_bar = (now.minute * 60 + now.second) % cfg["bar_interval_sec"]
            sleep_sec = cfg["bar_interval_sec"] - seconds_into_bar + 5  # +5s buffer for bar to close
            log.info("Next check in %ds (at %s UTC)",
                     sleep_sec, (now + timedelta(seconds=sleep_sec)).strftime("%H:%M:%S"))
            time.sleep(sleep_sec)

        except KeyboardInterrupt:
            log.info("Bot stopped by user.")
            break
        except Exception:
            log.exception("Error in main loop, retrying in 60s...")
            time.sleep(60)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BTC/USD Day Trading Bot (V11)")
    parser.add_argument("--live", action="store_true", help="Use live trading endpoint")
    args = parser.parse_args()

    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        log.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env or environment")
        return

    if args.live:
        base_url = "https://api.alpaca.markets/v2"
        log.warning("*** LIVE TRADING MODE — real money at risk ***")
    else:
        log.info("Paper trading mode")

    client = AlpacaClient(base_url, api_key, secret_key)

    # Verify connection
    acct = client.get_account()
    log.info("Connected — Account: %s | Cash: $%s | Status: %s",
             acct["id"][:8], acct["cash"], acct["status"])

    run_bot(client, CONFIG)


if __name__ == "__main__":
    main()
