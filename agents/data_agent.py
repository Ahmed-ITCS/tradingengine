"""
EvoTrade AI - Data & Technical Analysis Agent
Pulls OHLCV from ccxt (Binance) and computes all TA indicators via pandas-ta.
"""
from __future__ import annotations
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from core.state import trading_state


def get_exchange():
    """Public Binance spot client for market data (no API key required)."""
    try:
        import ccxt
        return ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
    except Exception as e:
        trading_state.add_log("DataAgent", f"Exchange init error: {e}", level="error")
        return None


def fetch_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 200) -> Optional[pd.DataFrame]:
    """Fetch live OHLCV from Binance public API. Returns None if unavailable."""
    exchange = get_exchange()
    if exchange is None:
        trading_state.add_log("DataAgent", "Cannot fetch OHLCV — exchange unavailable", level="error")
        return None

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            trading_state.add_log("DataAgent", f"No candles returned for {symbol} [{timeframe}]", level="error")
            return None
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        trading_state.add_log("DataAgent", f"Fetched {len(df)} candles for {symbol} [{timeframe}] (Binance live)")
        return df
    except Exception as e:
        trading_state.add_log("DataAgent", f"OHLCV fetch failed for {symbol} [{timeframe}]: {e}", level="error")
        return None


def _register_pandas_ta_accessor() -> None:
    """Import pandas-ta or community fork; registers ``DataFrame.ta`` as a side effect."""
    try:
        import pandas_ta  # noqa: F401
    except ImportError:
        import pandas_ta_classic  # noqa: F401


def compute_indicators(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute comprehensive TA indicators using pandas-ta."""
    used_manual = False
    try:
        _register_pandas_ta_accessor()
    except ImportError:
        trading_state.add_log(
            "DataAgent",
            "TA library missing — using manual indicators. Install in this same Python: "
            "`pip install pandas-ta-classic` (or use the project `.venv` after `pip install -r requirements.txt`).",
            level="warn",
        )
        df = _compute_manual_indicators(df)
        used_manual = True

    if not used_manual:
        try:
            # Trend
            df.ta.ema(length=20, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.ema(length=200, append=True)
            df.ta.sma(length=20, append=True)

            # Momentum
            df.ta.rsi(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.stoch(append=True)
            df.ta.cci(length=20, append=True)

            # Volatility
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.atr(length=14, append=True)

            # Volume
            df.ta.obv(append=True)
            df.ta.mfi(length=14, append=True)
            df.ta.vwap(append=True)

            # Additional
            df.ta.adx(length=14, append=True)
            df.ta.ichimoku(append=True)

        except Exception as e:
            trading_state.add_log(
                "DataAgent",
                f"pandas-ta computation error: {e}, using manual indicators",
                level="warn",
            )
            df = _compute_manual_indicators(df)

    # Extract latest values as clean dict
    latest = df.iloc[-1]
    summary = _extract_indicator_summary(df, latest)

    return df, summary


def _compute_manual_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback manual indicator computation."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # EMA
    df["EMA_20"] = close.ewm(span=20).mean()
    df["EMA_50"] = close.ewm(span=50).mean()
    df["EMA_200"] = close.ewm(span=200).mean()
    df["SMA_20"] = close.rolling(20).mean()

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["MACD_12_26_9"] = ema12 - ema26
    df["MACDs_12_26_9"] = df["MACD_12_26_9"].ewm(span=9).mean()
    df["MACDh_12_26_9"] = df["MACD_12_26_9"] - df["MACDs_12_26_9"]

    # Bollinger Bands
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["BBU_20_2.0"] = sma + 2 * std
    df["BBM_20_2.0"] = sma
    df["BBL_20_2.0"] = sma - 2 * std

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["ATRr_14"] = tr.rolling(14).mean()

    # ADX (simplified)
    df["ADX_14"] = tr.rolling(14).mean() / close.rolling(14).mean() * 100

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    return df


def _extract_indicator_summary(df: pd.DataFrame, latest: pd.Series) -> Dict[str, Any]:
    """Extract a clean summary of current indicator values."""
    def safe_get(key_patterns, default=None):
        for pattern in key_patterns if isinstance(key_patterns, list) else [key_patterns]:
            for col in df.columns:
                if pattern.upper() in col.upper():
                    val = latest.get(col, default)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        return float(val)
        return default

    close = float(latest["close"])
    ema20 = safe_get("EMA_20", close)
    ema50 = safe_get("EMA_50", close)
    ema200 = safe_get("EMA_200", close)

    return {
        "close": close,
        "open": float(latest["open"]),
        "high": float(latest["high"]),
        "low": float(latest["low"]),
        "volume": float(latest["volume"]),
        # Trend
        "ema_20": ema20,
        "ema_50": ema50,
        "ema_200": ema200,
        "trend": "BULLISH" if close > ema20 > ema50 else ("BEARISH" if close < ema20 < ema50 else "NEUTRAL"),
        # Momentum
        "rsi": safe_get("RSI_14", 50),
        "macd": safe_get("MACD_12_26_9", 0),
        "macd_signal": safe_get("MACDs_12_26_9", 0),
        "macd_hist": safe_get("MACDh_12_26_9", 0),
        # Volatility
        "bb_upper": safe_get("BBU_20_2", close * 1.02),
        "bb_middle": safe_get("BBM_20_2", close),
        "bb_lower": safe_get("BBL_20_2", close * 0.98),
        "atr": safe_get("ATRr_14", close * 0.01),
        "adx": safe_get("ADX_14", 25),
        # Volume
        "obv": safe_get("OBV", 0),
        "mfi": safe_get("MFI_14", 50),
        "vwap": safe_get("VWAP_D", close),
    }


def get_chart_data(df: pd.DataFrame, indicators: Dict) -> Dict[str, Any]:
    """Prepare chart data for the frontend."""
    # OHLCV for candlestick
    candles = []
    for ts, row in df.tail(100).iterrows():
        candles.append({
            "time": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"])
        })

    # EMA lines
    def safe_series(col_pattern):
        for col in df.columns:
            if col_pattern.upper() in col.upper():
                series = df[col].tail(100)
                return [
                    {"time": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                     "value": float(v)}
                    for ts, v in zip(series.index, series.values)
                    if not (isinstance(v, float) and np.isnan(v))
                ]
        return []

    return {
        "candles": candles,
        "ema_20": safe_series("EMA_20"),
        "ema_50": safe_series("EMA_50"),
        "bb_upper": safe_series("BBU_20_2"),
        "bb_lower": safe_series("BBL_20_2"),
        "rsi": safe_series("RSI_14"),
        "macd_hist": safe_series("MACDh_12_26_9"),
        "volume": [c["volume"] for c in candles],
    }


def run_data_agent(symbol: str, timeframe: str) -> Dict[str, Any]:
    """Main entry point for the Data & TA Agent."""
    trading_state.add_log("DataAgent", f"Fetching OHLCV: {symbol} [{timeframe}]")

    df = fetch_ohlcv(symbol, timeframe, limit=200)
    if df is None or len(df) < 50:
        trading_state.add_log("DataAgent", "Insufficient data", level="error")
        return {}

    df, indicators = compute_indicators(df)
    chart_data = get_chart_data(df, indicators)

    # Update global state
    trading_state.current_price = indicators["close"]
    trading_state.last_ohlcv = df
    trading_state.last_indicators = indicators
    trading_state.update_price(indicators["close"])
    trading_state.update_unrealized_pnl()

    trading_state.add_log(
        "DataAgent",
        f"Price=${indicators['close']:,.2f} | RSI={indicators['rsi']:.1f} | Trend={indicators['trend']}",
        data=indicators,
        level="success"
    )

    return {
        "indicators": indicators,
        "chart_data": chart_data,
        "symbol": symbol,
        "timeframe": timeframe,
        "candles_count": len(df)
    }
