"""
EvoTrade AI - Scalping Decision Agent
Rule-based, low-latency scalp signal engine for 1m/3m/5m crypto timeframes.

Four auto-selected setups (engine picks the highest-scoring one each cycle):
  1. momentum_breakout  — ADX trending + volume spike + price breaks recent high/low
  2. ema_pullback       — Clear EMA trend, price bounces off EMA20
  3. vwap_reversion     — Price >1% away from VWAP + RSI/MFI extreme
  4. rsi_bb_extreme     — RSI oversold/overbought at Bollinger Band edge

Risk gate: every signal is checked against ScalpingRiskManager before firing.
Position sizing is ATR-aware via ScalpingRiskManager.calculate_position_size().
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings
from core.state import trading_state, TradeDecision, TradeSignal
from core.scalping_risk import scalping_risk
from core.database import db


# ── Setup result ─────────────────────────────────────────────────────────────

@dataclass
class SetupResult:
    """Outcome of evaluating one scalping setup."""
    name: str               # momentum_breakout | ema_pullback | vwap_reversion | rsi_bb_extreme
    signal: str             # BUY | SELL | HOLD
    score: float            # 0–100; higher = more conviction
    confidence: float       # 0–1 for TradeDecision
    sl_price: float
    tp_price: float
    reasoning: str


# ── ATR-based SL/TP helper ────────────────────────────────────────────────────

def _atr_sl_tp(
    close: float,
    signal: str,
    atr: float,
    atr_mult: float,
    tp_mult: float,
    min_sl_pct: float,
) -> tuple:
    """
    Compute dynamic SL and TP driven by ATR.

    SL distance = max(ATR × atr_mult, close × min_sl_pct)
    TP distance = SL distance × tp_mult

    min_sl_pct is a safety floor so SL never collapses to zero
    during very-low-volatility candles.

    Returns (sl_price, tp_price).
    """
    sl_dist = max(atr * atr_mult, close * min_sl_pct)
    tp_dist = sl_dist * tp_mult
    if signal == "BUY":
        return close - sl_dist, close + tp_dist
    elif signal == "SELL":
        return close + sl_dist, close - tp_dist
    return close, close


# ── Individual setup scorers ──────────────────────────────────────────────────

def _score_momentum_breakout(df: pd.DataFrame, ind: Dict) -> SetupResult:
    """
    Breakout above 20-bar high (BUY) or below 20-bar low (SELL).
    Requires ADX trend + volume confirmation.
    """
    close = ind.get("close", 0)
    adx = ind.get("adx", 0)
    atr = ind.get("atr", close * 0.005)
    volume = ind.get("volume", 1)

    score = 0.0
    signal = "HOLD"
    reasons = []

    if len(df) >= 21:
        recent_high = float(df["high"].iloc[-21:-1].max())
        recent_low = float(df["low"].iloc[-21:-1].min())
        avg_vol = float(df["volume"].iloc[-20:].mean()) if len(df) >= 20 else volume

        vol_ratio = volume / avg_vol if avg_vol > 0 else 1.0

        # Trend strength
        if adx > 30:
            score += 30
            reasons.append(f"ADX={adx:.1f} strong trend")
        elif adx > 20:
            score += 15
            reasons.append(f"ADX={adx:.1f} mild trend")

        # Volume confirmation
        if vol_ratio >= 2.0:
            score += 25
            reasons.append(f"Vol x{vol_ratio:.1f} spike")
        elif vol_ratio >= 1.5:
            score += 15
            reasons.append(f"Vol x{vol_ratio:.1f} above avg")

        # MACD momentum
        macd_hist = ind.get("macd_hist", 0)
        if macd_hist > 0:
            score += 10
            reasons.append("MACD bullish")
        elif macd_hist < 0:
            score += 10
            reasons.append("MACD bearish")

        # Breakout direction
        if close > recent_high * 1.001:
            score += 20
            signal = "BUY"
            reasons.append(f"Broke above {recent_high:.2f}")
        elif close < recent_low * 0.999:
            score += 20
            signal = "SELL"
            reasons.append(f"Broke below {recent_low:.2f}")
    else:
        score = 0

    if signal == "HOLD":
        score = max(score - 30, 0)

    # ATR-based SL: 0.8× ATR (tight — we're trading with the trend so less wiggle room needed)
    sl, tp = _atr_sl_tp(
        close, signal, atr,
        atr_mult=settings.SCALPING_ATR_SL_MULT * 0.8,
        tp_mult=settings.SCALPING_TP_MULTIPLIER,
        min_sl_pct=settings.SCALPING_SL_PCT,
    )

    return SetupResult(
        name="momentum_breakout",
        signal=signal,
        score=score,
        confidence=min(score / 100, 0.92),
        sl_price=sl,
        tp_price=tp,
        reasoning=f"[Breakout] {' | '.join(reasons) or 'No breakout detected'} | ATR={atr:.4f} SL={sl:.4f} TP={tp:.4f}",
    )


def _score_ema_pullback(df: pd.DataFrame, ind: Dict) -> SetupResult:
    """
    Price pulls back to EMA20 in a clear trend.
    Long in uptrend (close > EMA20 > EMA50), short in downtrend.
    """
    close = ind.get("close", 0)
    ema20 = ind.get("ema_20", close)
    ema50 = ind.get("ema_50", close)
    rsi = ind.get("rsi", 50)
    atr = ind.get("atr", close * 0.005)

    score = 0.0
    signal = "HOLD"
    reasons = []

    ema20_dist_pct = abs(close - ema20) / ema20 if ema20 > 0 else 1.0

    # Trend clarity
    uptrend = ema20 > ema50 and close > ema50
    downtrend = ema20 < ema50 and close < ema50

    if uptrend:
        score += 25
        reasons.append("Uptrend (EMA20>EMA50)")
        signal = "BUY"
    elif downtrend:
        score += 25
        reasons.append("Downtrend (EMA20<EMA50)")
        signal = "SELL"

    # Proximity to EMA20 (pullback zone: within 0.8%)
    if ema20_dist_pct < 0.003:
        score += 35
        reasons.append(f"At EMA20 ({ema20_dist_pct*100:.2f}% away)")
    elif ema20_dist_pct < 0.008:
        score += 20
        reasons.append(f"Near EMA20 ({ema20_dist_pct*100:.2f}% away)")

    # RSI not extreme (has room to run)
    if signal == "BUY" and 40 <= rsi <= 62:
        score += 20
        reasons.append(f"RSI={rsi:.1f} room to rally")
    elif signal == "SELL" and 38 <= rsi <= 60:
        score += 20
        reasons.append(f"RSI={rsi:.1f} room to fall")

    # MACD confirmation
    macd_hist = ind.get("macd_hist", 0)
    if signal == "BUY" and macd_hist > 0:
        score += 10
        reasons.append("MACD bullish")
    elif signal == "SELL" and macd_hist < 0:
        score += 10
        reasons.append("MACD bearish")

    if signal == "HOLD":
        score = 0

    # ATR-based SL: 1.0× ATR, anchored just beyond EMA20 for extra confluence
    sl_atr, tp_atr = _atr_sl_tp(
        close, signal, atr,
        atr_mult=settings.SCALPING_ATR_SL_MULT * 1.0,
        tp_mult=settings.SCALPING_TP_MULTIPLIER,
        min_sl_pct=settings.SCALPING_SL_PCT,
    )
    if signal == "BUY":
        # Use whichever gives the wider (safer) stop: ATR-based or just below EMA20
        sl = min(sl_atr, ema20 - atr * 0.3)
        tp = tp_atr
    elif signal == "SELL":
        sl = max(sl_atr, ema20 + atr * 0.3)
        tp = tp_atr
    else:
        sl = tp = close

    return SetupResult(
        name="ema_pullback",
        signal=signal,
        score=score,
        confidence=min(score / 100, 0.90),
        sl_price=sl,
        tp_price=tp,
        reasoning=f"[EMA Pullback] {' | '.join(reasons) or 'No clear pullback'} | ATR={atr:.4f} SL={sl:.4f} TP={tp:.4f}",
    )


def _score_vwap_reversion(df: pd.DataFrame, ind: Dict) -> SetupResult:
    """
    Mean-reversion to VWAP when price is significantly extended.
    BUY when price is >1% below VWAP + RSI oversold.
    SELL when price is >1% above VWAP + RSI overbought.
    """
    close = ind.get("close", 0)
    vwap = ind.get("vwap", close)
    rsi = ind.get("rsi", 50)
    mfi = ind.get("mfi", 50)
    atr = ind.get("atr", close * 0.005)

    score = 0.0
    signal = "HOLD"
    reasons = []

    if vwap <= 0:
        return SetupResult("vwap_reversion", "HOLD", 0, 0, close, close, "VWAP unavailable")

    vwap_dev_pct = (close - vwap) / vwap

    # VWAP deviation threshold
    if vwap_dev_pct < -0.015:
        score += 40
        signal = "BUY"
        reasons.append(f"Price {abs(vwap_dev_pct)*100:.2f}% below VWAP")
    elif vwap_dev_pct < -0.008:
        score += 25
        signal = "BUY"
        reasons.append(f"Price {abs(vwap_dev_pct)*100:.2f}% below VWAP")
    elif vwap_dev_pct > 0.015:
        score += 40
        signal = "SELL"
        reasons.append(f"Price {vwap_dev_pct*100:.2f}% above VWAP")
    elif vwap_dev_pct > 0.008:
        score += 25
        signal = "SELL"
        reasons.append(f"Price {vwap_dev_pct*100:.2f}% above VWAP")

    # RSI extreme alignment
    if signal == "BUY" and rsi < 35:
        score += 25
        reasons.append(f"RSI={rsi:.1f} oversold")
    elif signal == "SELL" and rsi > 65:
        score += 25
        reasons.append(f"RSI={rsi:.1f} overbought")
    elif signal == "BUY" and rsi < 45:
        score += 10
    elif signal == "SELL" and rsi > 55:
        score += 10

    # MFI alignment
    if signal == "BUY" and mfi < 35:
        score += 15
        reasons.append(f"MFI={mfi:.1f} oversold")
    elif signal == "SELL" and mfi > 65:
        score += 15
        reasons.append(f"MFI={mfi:.1f} overbought")

    # Bollinger band touch adds conviction
    bb_lower = ind.get("bb_lower", close * 0.98)
    bb_upper = ind.get("bb_upper", close * 1.02)
    if signal == "BUY" and close <= bb_lower * 1.003:
        score += 10
        reasons.append("At BB lower")
    elif signal == "SELL" and close >= bb_upper * 0.997:
        score += 10
        reasons.append("At BB upper")

    if signal == "HOLD":
        score = 0

    # ATR-based SL: 1.2× ATR (mean reversion can overshoot before snapping back)
    sl_atr, tp_atr = _atr_sl_tp(
        close, signal, atr,
        atr_mult=settings.SCALPING_ATR_SL_MULT * 1.2,
        tp_mult=settings.SCALPING_TP_MULTIPLIER,
        min_sl_pct=settings.SCALPING_SL_PCT,
    )
    # TP: target VWAP, but at least the ATR-derived TP (whichever is further from entry)
    if signal == "BUY":
        sl = sl_atr
        tp = max(tp_atr, vwap) if vwap > close else tp_atr
    elif signal == "SELL":
        sl = sl_atr
        tp = min(tp_atr, vwap) if vwap < close else tp_atr
    else:
        sl = tp = close

    return SetupResult(
        name="vwap_reversion",
        signal=signal,
        score=score,
        confidence=min(score / 100, 0.88),
        sl_price=sl,
        tp_price=tp,
        reasoning=f"[VWAP Rev] {' | '.join(reasons) or 'No VWAP extension'} | ATR={atr:.4f} SL={sl:.4f} TP={tp:.4f}",
    )


def _score_rsi_bb_extreme(df: pd.DataFrame, ind: Dict) -> SetupResult:
    """
    RSI extreme at Bollinger Band edge — classic mean-reversion scalp.
    Highest conviction when RSI <28 at BB lower or RSI >72 at BB upper.
    """
    close = ind.get("close", 0)
    rsi = ind.get("rsi", 50)
    bb_lower = ind.get("bb_lower", close * 0.98)
    bb_upper = ind.get("bb_upper", close * 1.02)
    bb_middle = ind.get("bb_middle", close)
    atr = ind.get("atr", close * 0.005)
    mfi = ind.get("mfi", 50)

    score = 0.0
    signal = "HOLD"
    reasons = []

    # RSI extremes
    if rsi < 25:
        score += 45
        signal = "BUY"
        reasons.append(f"RSI={rsi:.1f} severely oversold")
    elif rsi < 30:
        score += 30
        signal = "BUY"
        reasons.append(f"RSI={rsi:.1f} oversold")
    elif rsi > 75:
        score += 45
        signal = "SELL"
        reasons.append(f"RSI={rsi:.1f} severely overbought")
    elif rsi > 70:
        score += 30
        signal = "SELL"
        reasons.append(f"RSI={rsi:.1f} overbought")

    # BB band touch
    if signal == "BUY" and bb_lower > 0 and close <= bb_lower * 1.005:
        score += 25
        reasons.append(f"Touching BB lower ({bb_lower:.2f})")
    elif signal == "SELL" and bb_upper > 0 and close >= bb_upper * 0.995:
        score += 25
        reasons.append(f"Touching BB upper ({bb_upper:.2f})")

    # MFI confirmation
    if signal == "BUY" and mfi < 30:
        score += 15
        reasons.append(f"MFI={mfi:.1f} oversold")
    elif signal == "SELL" and mfi > 70:
        score += 15
        reasons.append(f"MFI={mfi:.1f} overbought")

    # MACD histogram reversing
    if len(df) >= 3:
        hist_col = next((c for c in df.columns if "MACDh" in c or "macdh" in c.lower()), None)
        if hist_col:
            hist_prev = float(df[hist_col].iloc[-2]) if not pd.isna(df[hist_col].iloc[-2]) else 0
            hist_now = float(df[hist_col].iloc[-1]) if not pd.isna(df[hist_col].iloc[-1]) else 0
            if signal == "BUY" and hist_now > hist_prev:
                score += 10
                reasons.append("MACD hist turning up")
            elif signal == "SELL" and hist_now < hist_prev:
                score += 10
                reasons.append("MACD hist turning down")

    if signal == "HOLD":
        score = 0

    # ATR-based SL: 1.5× ATR (widest — RSI reversals can spike hard before reversing)
    sl_atr, tp_atr = _atr_sl_tp(
        close, signal, atr,
        atr_mult=settings.SCALPING_ATR_SL_MULT * 1.5,
        tp_mult=settings.SCALPING_TP_MULTIPLIER,
        min_sl_pct=settings.SCALPING_SL_PCT,
    )
    # TP: target the BB middle band; fall back to ATR-derived TP if band is unreachable
    if signal == "BUY":
        sl = sl_atr
        tp = bb_middle if (bb_middle > close and bb_middle <= close + (close - sl_atr) * settings.SCALPING_TP_MULTIPLIER * 1.5) else tp_atr
    elif signal == "SELL":
        sl = sl_atr
        tp = bb_middle if (bb_middle < close and bb_middle >= close - (sl_atr - close) * settings.SCALPING_TP_MULTIPLIER * 1.5) else tp_atr
    else:
        sl = tp = close

    return SetupResult(
        name="rsi_bb_extreme",
        signal=signal,
        score=score,
        confidence=min(score / 100, 0.90),
        sl_price=sl,
        tp_price=tp,
        reasoning=f"[RSI/BB] {' | '.join(reasons) or 'No extreme reading'} | ATR={atr:.4f} SL={sl:.4f} TP={tp:.4f}",
    )


# ── Setup selector ────────────────────────────────────────────────────────────

def select_best_setup(df: pd.DataFrame, ind: Dict) -> Optional[SetupResult]:
    """
    Score all setups and return the best non-HOLD one above MIN_CONFIDENCE.
    Returns None if no setup clears the confidence threshold.
    """
    candidates = [
        _score_momentum_breakout(df, ind),
        _score_ema_pullback(df, ind),
        _score_vwap_reversion(df, ind),
        _score_rsi_bb_extreme(df, ind),
    ]

    actionable = [c for c in candidates if c.signal != "HOLD" and c.score > 0]
    if not actionable:
        return None

    best = max(actionable, key=lambda c: c.score)
    min_conf = settings.SCALPING_MIN_CONFIDENCE
    if best.confidence < min_conf:
        trading_state.add_log(
            "ScalpingAgent",
            f"Best setup '{best.name}' confidence {best.confidence:.0%} < min {min_conf:.0%} — HOLD",
            level="info",
        )
        return None

    return best


# ── Auto timeframe selection ──────────────────────────────────────────────────

def _tf_noise_ratio(indicators: Dict) -> float:
    """ATR / price — a normalised volatility proxy for TF quality filtering."""
    close = indicators.get("close", 1)
    atr   = indicators.get("atr",   close * 0.005)
    return atr / close if close > 0 else 1.0


def run_scalping_agent_auto(symbol: str, portfolio_dict: Dict) -> "TradeDecision":
    """
    Multi-timeframe scalp decision.

    Steps:
    1. Iterate over SCALPING_TF_OPTIONS (e.g. 1m, 3m, 5m)
    2. Fetch OHLCV + compute indicators for each
    3. Skip any TF where ATR/price > SCALPING_TF_NOISE_FILTER (too choppy)
    4. Score all 4 setups on every eligible TF
    5. Pick the (TF, setup) pair with the highest setup score
    6. Route through the standard risk gate + position sizer
    7. Update trading_state with the winning TF's OHLCV/indicators

    Falls back to the first eligible TF's HOLD if nothing clears confidence.
    """
    from agents.data_agent import fetch_ohlcv, compute_indicators

    tfs = [t.strip() for t in settings.SCALPING_TF_OPTIONS.split(",") if t.strip()]
    noise_cap = settings.SCALPING_TF_NOISE_FILTER

    best_setup:  "SetupResult | None" = None
    best_tf:     str = tfs[0] if tfs else settings.SCALPING_TIMEFRAME
    best_df:     "pd.DataFrame | None" = None
    best_ind:    "Dict | None" = None
    fallback_ind: "Dict | None" = None
    fallback_df:  "pd.DataFrame | None" = None
    fallback_tf:  str = best_tf

    tf_results = []  # for logging

    for tf in tfs:
        try:
            df = fetch_ohlcv(symbol, tf, limit=150)
            if df is None or len(df) < 30:
                tf_results.append(f"{tf}:no_data")
                continue

            df, ind = compute_indicators(df)
            noise = _tf_noise_ratio(ind)

            # Store first valid TF as fallback regardless of noise
            if fallback_df is None:
                fallback_df  = df
                fallback_ind = ind
                fallback_tf  = tf

            if noise > noise_cap:
                tf_results.append(f"{tf}:noisy({noise*100:.1f}%)")
                continue

            setup = select_best_setup(df, ind)
            score = setup.score if setup else 0.0
            tf_results.append(f"{tf}:score={score:.0f}({setup.name if setup else 'HOLD'})")

            if setup and (best_setup is None or setup.score > best_setup.score):
                best_setup = setup
                best_tf    = tf
                best_df    = df
                best_ind   = ind

        except Exception as exc:
            tf_results.append(f"{tf}:err({str(exc)[:30]})")
            continue

    trading_state.add_log(
        "ScalpingAgent",
        f"Auto-TF scan [{symbol}]: {' | '.join(tf_results)}",
        level="info",
    )

    # If no TF produced an actionable setup, use fallback data for a HOLD
    if best_df is None or best_ind is None:
        best_df  = fallback_df
        best_ind = fallback_ind
        best_tf  = fallback_tf

    # Update global state with the winning TF's data
    if best_ind and best_df is not None:
        trading_state.current_price = best_ind.get("close", trading_state.current_price)
        trading_state.last_ohlcv    = best_df
        trading_state.last_indicators = best_ind
        trading_state.update_price(best_ind["close"])
        trading_state.update_unrealized_pnl()

    if best_setup and best_ind:
        trading_state.add_log(
            "ScalpingAgent",
            f"✅ Selected TF={best_tf} | setup={best_setup.name} | "
            f"score={best_setup.score:.0f} conf={best_setup.confidence:.0%}",
            level="success",
        )
        return run_scalping_agent(symbol, best_ind, best_df, portfolio_dict,
                                  _tf_override=best_tf)

    # Nothing cleared confidence on any TF
    if best_ind:
        return run_scalping_agent(symbol, best_ind, best_df, portfolio_dict,
                                  _tf_override=best_tf)

    return _hold_decision(symbol, {}, "Auto-TF: no valid data on any timeframe")


# ── Main entry point ──────────────────────────────────────────────────────────

def run_scalping_agent(
    symbol: str,
    indicators: Dict,
    df: pd.DataFrame,
    portfolio_dict: Dict,
    _tf_override: str = "",
) -> TradeDecision:
    """
    Core scalping decision function (single-TF).
    Called directly or via run_scalping_agent_auto() after TF selection.

    Steps:
    1. Check ScalpingRiskManager kill switch / daily budget
    2. Score all 4 setups and pick the winner
    3. Size position via ATR-aware calculator
    4. Return TradeDecision (HOLD if blocked or no setup)
    """
    tf_tag = f"[{_tf_override}]" if _tf_override else f"[{settings.SCALPING_TIMEFRAME}]"
    trading_state.add_log("ScalpingAgent", f"Evaluating scalp setups for {symbol} {tf_tag}...")

    # ── 1. Risk gate ──────────────────────────────────────────────────────────
    allowed, reason = scalping_risk.is_trading_allowed()
    if not allowed:
        trading_state.add_log("ScalpingAgent", f"BLOCKED: {reason}", level="warn")
        return _hold_decision(symbol, indicators, f"Risk gate: {reason}")

    # ── 2. Setup selection ────────────────────────────────────────────────────
    best = select_best_setup(df, indicators)

    if best is None:
        trading_state.add_log("ScalpingAgent", "No high-confidence setup found — HOLD", level="info")
        return _hold_decision(symbol, indicators, "No setup cleared confidence threshold")

    trading_state.add_log(
        "ScalpingAgent",
        f"Setup selected: {best.name} | {best.signal} | score={best.score:.0f} conf={best.confidence:.0%}",
        level="success",
    )

    # ── 3. Position sizing ────────────────────────────────────────────────────
    sizing = scalping_risk.calculate_position_size(
        entry_price=indicators.get("close", 1),
        sl_price=best.sl_price,
        current_equity=portfolio_dict.get("equity", settings.SCALPING_ACCOUNT_SIZE),
    )

    # ── 4. Build TradeDecision ────────────────────────────────────────────────
    decision = TradeDecision(
        id=str(uuid.uuid4())[:8],
        timestamp=datetime.now(timezone.utc).isoformat(),
        symbol=symbol,
        signal=TradeSignal(best.signal),
        confidence=best.confidence,
        size_usdt=sizing["size_usdt"],
        entry_price=indicators.get("close", 0),
        stop_loss=best.sl_price,
        take_profit=best.tp_price,
        reasoning=(
            best.reasoning
            + f" | tf={_tf_override or settings.SCALPING_TIMEFRAME}"
            + f" | risk=${sizing['risk_usdt']:.2f}"
            + f" | budget_left=${sizing['max_loss_today_remaining']:.2f}"
        ),
        agent_contributions={
            "setup":     best.name,
            "timeframe": _tf_override or settings.SCALPING_TIMEFRAME,
            "score":     str(round(best.score, 1)),
            "daily_pnl": str(round(scalping_risk.daily_status()["daily_pnl_usdt"], 2)),
        },
    )

    db.save_decision(decision)
    trading_state.add_decision(decision)

    emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(best.signal, "⚪")
    trading_state.add_log(
        "ScalpingAgent",
        f"{emoji} SCALP {best.signal} | {best.name} | "
        f"conf={best.confidence:.0%} size=${sizing['size_usdt']:.0f} "
        f"SL={best.sl_price:.4f} TP={best.tp_price:.4f}",
        data={
            "signal":     best.signal,
            "confidence": best.confidence,
            "setup":      best.name,
            "size_usdt":  sizing["size_usdt"],
            "reasoning":  best.reasoning,
        },
        level="success",
    )

    return decision


def _hold_decision(symbol: str, indicators: Dict, reason: str) -> TradeDecision:
    decision = TradeDecision(
        id=str(uuid.uuid4())[:8],
        timestamp=datetime.now(timezone.utc).isoformat(),
        symbol=symbol,
        signal=TradeSignal.HOLD,
        confidence=0.0,
        size_usdt=0.0,
        entry_price=indicators.get("close", 0),
        stop_loss=None,
        take_profit=None,
        reasoning=reason,
    )
    trading_state.add_decision(decision)
    return decision


# ── All-positions close helper ────────────────────────────────────────────────

def close_all_scalp_positions(reason: str = "EOD close-all"):
    """
    Force-close every open position. Called at EOD or on kill-switch trigger.
    """
    price = trading_state.current_price
    if price <= 0:
        trading_state.add_log("ScalpingAgent", "close-all skipped: no current price", level="warn")
        return []

    closed_ids = []
    with trading_state._lock:
        open_ids = [t.id for t in trading_state.open_trades]

    for tid in open_ids:
        trading_state.close_trade(tid, price)
        scalping_risk.record_trade_close(tid, price, 0)   # pnl already in state
        closed_ids.append(tid)

    if closed_ids:
        trading_state.add_log(
            "ScalpingAgent",
            f"EOD close-all: closed {len(closed_ids)} position(s) @ {price:.4f} — {reason}",
            level="warn",
        )
    return closed_ids
