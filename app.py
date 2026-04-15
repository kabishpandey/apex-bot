"""
APEX QQQ Bot — Python version of your Pine Script v3
Runs 24/7 on Render.com, pulls live data, trades on Alpaca paper account.
"""

import time
import os
import logging
from datetime import datetime
import pytz
import traceback

import pandas as pd
import ta as talib
import yfinance as yf
import alpaca_trade_api as tradeapi
from flask import Flask, jsonify
import threading

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Flask app (keeps Render alive) ─────────────────────────────────────
app = Flask(__name__)

# Track bot health for the status endpoint
bot_status = {
    "last_heartbeat": None,
    "last_bar_processed": None,
    "last_error": None,
    "loop_count": 0,
    "status": "starting",
}

@app.route("/")
def home():
    return jsonify({
        "status":               "APEX Bot Running",
        "time":                 str(datetime.now()),
        "bot_status":           bot_status["status"],
        "last_heartbeat":       str(bot_status["last_heartbeat"]),
        "last_bar_processed":   str(bot_status["last_bar_processed"]),
        "last_error":           str(bot_status["last_error"]),
        "loop_count":           bot_status["loop_count"],
    }), 200

# ── Alpaca connection ───────────────────────────────────────────────────
API_KEY    = os.environ.get("ALPACA_KEY",    "YOUR_KEY_HERE")
API_SECRET = os.environ.get("ALPACA_SECRET", "YOUR_SECRET_HERE")
BASE_URL   = "https://paper-api.alpaca.markets"

alpaca = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# ── Settings (mirrors your Pine Script inputs) ─────────────────────────
SYMBOL       = "QQQ"
TIMEFRAME    = "4h"
SCORE_NEEDED = 4
EMA_FAST     = 21
EMA_SLOW     = 55
SMA_50       = 50
SMA_200      = 200
ST_LEN       = 10
ST_MULT      = 3.0
RSI_LEN      = 15
RSI_MAX      = 65.0
ATR_LEN      = 14
SL_ATR       = 2.0
RR           = 2.5
RISK_PCT     = 1.5
PART_PCT     = 0.40
VOL_MULT     = 2.0
SKIP_SEPT    = True
CB_LOSSES    = 3


# ══════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════

def get_data(symbol=SYMBOL, period="180d", interval="1h"):
    """Fetch OHLCV data from Yahoo Finance."""
    log.info(f"Fetching {symbol} data...")
    try:
        df = yf.download(symbol, period=period, interval=interval,
                         auto_adjust=True, progress=False)
    except Exception as e:
        log.error(f"Yahoo Finance download error: {e}")
        return None

    if df is None or df.empty:
        log.error("No data returned from Yahoo Finance!")
        return None

    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    df.dropna(inplace=True)

    # Resample 1h → 4h
    df_4h = df.resample("4h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum"
    }).dropna()

    log.info(f"Got {len(df_4h)} 4H bars")
    return df_4h


def get_daily_ema50(symbol=SYMBOL):
    """Fetch daily EMA50 for HTF filter."""
    try:
        df = yf.download(symbol, period="120d", interval="1d",
                         auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        ema = talib.trend.ema_indicator(df["close"], window=50)
        return float(ema.iloc[-1])
    except Exception as e:
        log.error(f"Daily EMA50 error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════════════

def calc_supertrend(df, length=ST_LEN, multiplier=ST_MULT):
    hl2 = (df["high"] + df["low"]) / 2

    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_st     = tr.ewm(alpha=1/length, adjust=False).mean()

    upper = hl2 + multiplier * atr_st
    lower = hl2 - multiplier * atr_st

    st_upper  = [float("nan")] * len(df)
    st_lower  = [float("nan")] * len(df)
    direction = [1] * len(df)

    for i in range(1, len(df)):
        pu = st_upper[i-1] if not pd.isna(st_upper[i-1]) else upper.iloc[i]
        pl = st_lower[i-1] if not pd.isna(st_lower[i-1]) else lower.iloc[i]

        st_upper[i] = max(upper.iloc[i], pu) if df["close"].iloc[i-1] > pu else upper.iloc[i]
        st_lower[i] = min(lower.iloc[i], pl) if df["close"].iloc[i-1] < pl else lower.iloc[i]

        if direction[i-1] == -1:
            direction[i] = 1 if df["close"].iloc[i] > pu else -1
        else:
            direction[i] = -1 if df["close"].iloc[i] < pl else 1

    df["st_dir"]  = direction
    df["st_bull"] = df["st_dir"] == 1
    df["st_line"] = [st_lower[i] if direction[i] == 1 else st_upper[i]
                     for i in range(len(df))]
    return df


def add_indicators(df):
    df["ema21"]  = talib.trend.ema_indicator(df["close"], window=EMA_FAST)
    df["ema55"]  = talib.trend.ema_indicator(df["close"], window=EMA_SLOW)
    df["sma50"]  = talib.trend.sma_indicator(df["close"], window=SMA_50)
    df["sma200"] = talib.trend.sma_indicator(df["close"], window=SMA_200)

    df["atr"]       = talib.volatility.average_true_range(df["high"], df["low"], df["close"], window=ATR_LEN)
    df["atr_20avg"] = df["atr"].rolling(20).mean()

    df["rsi"] = talib.momentum.rsi(df["close"], window=RSI_LEN)

    macd_line   = talib.trend.macd(df["close"])
    signal_line = talib.trend.macd_signal(df["close"])
    df["macd_hist"] = macd_line - signal_line

    df["vol_avg"] = df["volume"].rolling(20).mean()
    df["adx"]     = talib.trend.adx_neg(df["high"], df["low"], df["close"], window=14)

    df = calc_supertrend(df)
    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════
#  SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════

def score_bar(row, prev_rows, d_ema50):
    close = row["close"]

    ema_touch = any(
        prev_rows["low"].iloc[-(i+1)] <= prev_rows["ema21"].iloc[-(i+1)] * 1.005
        for i in range(min(4, len(prev_rows)))
    )

    c1 = bool(row["ema21"] > row["ema55"])
    c2 = bool(row["st_bull"])
    c3 = bool(d_ema50 is None or close > d_ema50)
    c4 = bool(row["rsi"] < RSI_MAX)
    c5 = bool(row["macd_hist"] > prev_rows["macd_hist"].iloc[-1])
    c6 = bool(row["volume"] > row["vol_avg"])
    c7 = bool(ema_touch)

    score = sum([c1, c2, c3, c4, c5, c6, c7])

    conditions = {
        "C1_ema_stack":  c1,
        "C2_supertrend": c2,
        "C3_htf":        c3,
        "C4_rsi":        c4,
        "C5_macd":       c5,
        "C6_volume":     c6,
        "C7_ema_dip":    c7,
        "score":         score,
    }
    return score, conditions


def check_filters(row, month):
    vol_spike = bool(row["atr"] > row["atr_20avg"] * VOL_MULT)
    sept_skip = SKIP_SEPT and month == 9

    notes = []
    if vol_spike:  notes.append("VOL SPIKE blocked")
    if sept_skip:  notes.append("SEPTEMBER blocked")

    return not vol_spike, not sept_skip, notes


# ══════════════════════════════════════════════════════════════════════
#  POSITION SIZING
# ══════════════════════════════════════════════════════════════════════

def get_equity():
    account = alpaca.get_account()
    return float(account.equity)


def calc_qty(equity, atr, consec_losses):
    stop_dist = atr * SL_ATR
    risk_dol  = equity * RISK_PCT / 100.0
    cb_mult   = 0.5 if consec_losses >= CB_LOSSES else 1.0
    raw_qty   = risk_dol * cb_mult / stop_dist if stop_dist > 0 else 1.0
    return max(1, round(raw_qty)), cb_mult


# ══════════════════════════════════════════════════════════════════════
#  TRADE EXECUTION
# ══════════════════════════════════════════════════════════════════════

def get_position():
    try:
        pos = alpaca.get_position(SYMBOL)
        return int(pos.qty)
    except:
        return 0


def enter_long(qty, sl, tp1, tp2, score):
    log.info(f"🟢 LONG ENTRY — qty={qty}  SL={sl:.2f}  TP1={tp1:.2f}  TP2={tp2:.2f}  score={score}/7")
    alpaca.submit_order(
        symbol        = SYMBOL,
        qty           = qty,
        side          = "buy",
        type          = "market",
        time_in_force = "day"
    )


def exit_position(reason="exit"):
    try:
        pos_qty = get_position()
        if pos_qty > 0:
            log.info(f"🔴 CLOSING POSITION — reason: {reason}")
            alpaca.close_position(SYMBOL)
    except Exception as e:
        log.error(f"Exit error: {e}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════

state = {
    "in_trade":       False,
    "entry_price":    0.0,
    "sl":             0.0,
    "tp1":            0.0,
    "tp2":            0.0,
    "partial_done":   False,
    "consec_losses":  0,
    "last_bar_time":  None,
}


def is_market_hours():
    """
    Returns True during extended market window.
    We use 9:00 AM – 8:00 PM ET so we never miss a 4H bar close
    (e.g. a bar opening at 1:30 PM closes at 5:30 PM, after the 4 PM hard close).
    """
    et  = pytz.timezone("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5:   # Saturday/Sunday
        return False
    market_open  = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    market_close = now.replace(hour=20, minute=0,  second=0, microsecond=0)
    return market_open <= now <= market_close


def get_last_closed_4h_bar(df):
    """Return the last FULLY CLOSED 4H bar."""
    now_et = datetime.now(pytz.timezone("America/New_York"))

    closed_bars = []
    for ts, _ in df.iterrows():
        bar_close_time = ts + pd.Timedelta(hours=4)
        if bar_close_time.astimezone(pytz.timezone("America/New_York")) <= now_et:
            closed_bars.append(ts)

    if not closed_bars:
        return None, None

    last_closed_ts = closed_bars[-1]
    return df.loc[last_closed_ts], last_closed_ts


def run_bot():
    """Main bot loop — checks every 5 minutes, only acts on confirmed closed 4H bars."""
    log.info("🚀 APEX Bot started — waiting for 4H bar closes...")
    bot_status["status"] = "running"

    while True:
        try:
            bot_status["last_heartbeat"] = datetime.now()
            bot_status["loop_count"]    += 1

            if not is_market_hours():
                log.info("Market closed — sleeping 15 min...")
                bot_status["status"] = "sleeping (market closed)"
                time.sleep(900)
                continue

            bot_status["status"] = "active"

            # ── Fetch data ──────────────────────────────────────────
            df = get_data()
            if df is None or len(df) < 50:
                log.warning("Not enough data — retrying in 5 min...")
                time.sleep(300)
                continue

            df = add_indicators(df)

            if len(df) < 10:
                log.warning("Not enough bars after indicators — retrying...")
                time.sleep(300)
                continue

            # ── Only use confirmed closed 4H bars ──────────────────
            row, bar_ts = get_last_closed_4h_bar(df)
            if row is None:
                log.info("No confirmed closed 4H bar yet — waiting...")
                time.sleep(300)
                continue

            # Skip if we already processed this bar
            if state["last_bar_time"] == bar_ts:
                log.info(f"Already processed bar {bar_ts} — next check in 5 min...")
                time.sleep(300)
                continue

            # New bar confirmed — process it
            log.info(f"✅ New confirmed 4H bar: {bar_ts}")
            state["last_bar_time"] = bar_ts
            bot_status["last_bar_processed"] = str(bar_ts)

            prev   = df.loc[:bar_ts].iloc[:-1]
            now_et = datetime.now(pytz.timezone("America/New_York"))
            month  = now_et.month

            d_ema50            = get_daily_ema50()
            score, conds       = score_bar(row, prev, d_ema50)
            vol_ok, sept_ok, filter_notes = check_filters(row, month)

            log.info(
                f"📊 Score={score}/7  "
                f"RSI={row['rsi']:.1f}  "
                f"ADX={row['adx']:.1f}  "
                f"ST={'Bull' if row['st_bull'] else 'Bear'}  "
                f"Filters={'OK' if (vol_ok and sept_ok) else str(filter_notes)}"
            )
            log.info(f"Conditions: {conds}")

            # ── Check existing position ─────────────────────────────
            pos_qty = get_position()
            state["in_trade"] = pos_qty > 0

            if state["in_trade"]:
                close = float(row["close"])
                ep    = state["entry_price"]
                sl    = state["sl"]
                tp1   = state["tp1"]
                tp2   = state["tp2"]

                if not row["st_bull"]:
                    exit_position("ST Flip")
                    state["in_trade"]      = False
                    state["consec_losses"] += 1

                elif row["ema21"] < row["ema55"]:
                    exit_position("EMA Flip")
                    state["in_trade"]      = False
                    state["consec_losses"] += 1

                elif row["sma50"] < row["sma200"]:
                    exit_position("Death Cross")
                    state["in_trade"]      = False
                    state["consec_losses"] += 1

                elif close <= sl:
                    exit_position("Stop Loss")
                    state["in_trade"]      = False
                    state["consec_losses"] += 1

                elif close >= tp2:
                    exit_position("Take Profit")
                    state["in_trade"]      = False
                    state["consec_losses"] = 0

                elif close >= tp1 and not state["partial_done"]:
                    part_qty = max(1, round(pos_qty * PART_PCT))
                    log.info(f"🟡 PARTIAL EXIT at 1R — selling {part_qty} shares")
                    alpaca.submit_order(
                        symbol=SYMBOL, qty=part_qty, side="sell",
                        type="market", time_in_force="day"
                    )
                    state["partial_done"] = True
                    state["sl"]           = max(sl, ep)

            else:
                # ── Look for new entry ──────────────────────────────
                bull_regime = row["close"] > row["sma50"]
                long_signal = (
                    bull_regime and
                    score >= SCORE_NEEDED and
                    vol_ok and
                    sept_ok
                )

                log.info(
                    f"Entry check — bull_regime={bull_regime}  "
                    f"score={score}>={SCORE_NEEDED}  "
                    f"vol_ok={vol_ok}  sept_ok={sept_ok}  "
                    f"→ signal={'YES 🟢' if long_signal else 'NO'}"
                )

                if long_signal:
                    equity  = get_equity()
                    atr     = float(row["atr"])
                    qty, cb = calc_qty(equity, atr, state["consec_losses"])

                    stop_dist = atr * SL_ATR
                    close_p   = float(row["close"])
                    sl        = close_p - stop_dist
                    tp1       = close_p + stop_dist
                    tp2       = close_p + stop_dist * RR

                    enter_long(qty, sl, tp1, tp2, score)

                    state["in_trade"]     = True
                    state["entry_price"]  = close_p
                    state["sl"]           = sl
                    state["tp1"]          = tp1
                    state["tp2"]          = tp2
                    state["partial_done"] = False

                    if cb < 1.0:
                        log.info(f"⚠️  Circuit breaker active — {state['consec_losses']} losses, half size")

        except Exception as e:
            err_msg = f"{e}\n{traceback.format_exc()}"
            log.error(f"Bot loop error: {err_msg}")
            bot_status["last_error"]  = str(e)
            bot_status["status"]      = "error — recovering"
            time.sleep(60)   # short sleep then retry instead of dying

        # Check every 5 minutes
        time.sleep(300)


# ══════════════════════════════════════════════════════════════════════
#  WATCHDOG — restarts bot thread if it ever dies
# ══════════════════════════════════════════════════════════════════════

def watchdog():
    """Monitors the bot thread and restarts it if it dies."""
    global bot_thread
    log.info("👁️  Watchdog started")
    while True:
        time.sleep(120)   # check every 2 minutes
        if not bot_thread.is_alive():
            log.error("⚠️  Bot thread died — restarting!")
            bot_status["status"] = "restarting"
            bot_thread = threading.Thread(target=run_bot, daemon=True)
            bot_thread.start()
            log.info("✅ Bot thread restarted")


# ══════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Start bot thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    # Start watchdog thread
    wd_thread = threading.Thread(target=watchdog, daemon=True)
    wd_thread.start()

    # Flask keeps Render alive
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
