"""
EvoTrade AI - Swing Trading Decision Agent
4h timeframe chart pattern engine for multi-day holds.

Scans ALL classical chart patterns on 4h candles:
  double top/bottom, H&S, triangles, flags, wedges, S/R breakout,
  hammer, engulfing, morning/evening star, etc.

Optional LLM overlay for final BUY/SELL/HOLD decision.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import settings
from core.state import trading_state, TradeDecision, TradeSignal
from core.swing_risk import swing_risk
from core.database import db
from core.llm import get_llm_response
from agents.chart_patterns import (
    PatternResult,
    detect_all_patterns,
    select_best_pattern,
    patterns_summary,
    PATTERN_NAMES,
)


_signal_state: Dict[str, Dict[str, object]] = {}


def _atr_sl_tp(
    close: float,
    signal: str,
    atr: float,
    sl_price: float,
    tp_price: float,
) -> tuple:
    """Validate swing SL/TP with ATR floor."""
    min_sl_pct = settings.SWING_SL_PCT
    min_dist = max(atr * settings.SWING_ATR_SL_MULT, close * min_sl_pct)

    if signal == "BUY":
        sl = min(sl_price, close - min_dist) if sl_price < close else close - min_dist
        tp_dist = max(tp_price - close, min_dist * settings.SWING_TP_MULTIPLIER) if tp_price > close else min_dist * settings.SWING_TP_MULTIPLIER
        tp = close + tp_dist
    elif signal == "SELL":
        sl = max(sl_price, close + min_dist) if sl_price > close else close + min_dist
        tp_dist = max(close - tp_price, min_dist * settings.SWING_TP_MULTIPLIER) if tp_price < close else min_dist * settings.SWING_TP_MULTIPLIER
        tp = close - tp_dist
    else:
        return close, close
    return sl, tp


def _trend_confluence(ind: Dict, signal: str) -> Tuple[float, str]:
    """Boost score when pattern aligns with EMA trend on 4h."""
    ema20 = ind.get("ema_20", 0)
    ema50 = ind.get("ema_50", 0)
    close = ind.get("close", 0)
    adx = ind.get("adx", 0)
    bonus = 0.0
    notes = []

    if signal == "BUY" and ema20 > ema50 and close > ema50:
        bonus += 12
        notes.append("4h uptrend (EMA20>EMA50)")
    elif signal == "SELL" and ema20 < ema50 and close < ema50:
        bonus += 12
        notes.append("4h downtrend (EMA20<EMA50)")

    if adx > 25:
        bonus += 8
        notes.append(f"ADX={adx:.0f} trending")
    elif adx > 18:
        bonus += 4

    return bonus, " | ".join(notes) if notes else ""


def evaluate_patterns(
    df: pd.DataFrame,
    ind: Dict,
) -> List[PatternResult]:
    """Detect all 4h chart patterns and apply trend confluence bonus."""
    raw = detect_all_patterns(df)
    enriched: List[PatternResult] = []

    for p in raw:
        bonus, trend_note = _trend_confluence(ind, p.signal)
        new_score = min(p.score + bonus, 98)
        new_conf = min(new_score / 100, 0.92)
        reasoning = p.reasoning
        if trend_note:
            reasoning += f" | {trend_note}"
        enriched.append(PatternResult(
            name=p.name,
            signal=p.signal,
            score=new_score,
            confidence=new_conf,
            sl_price=p.sl_price,
            tp_price=p.tp_price,
            reasoning=reasoning,
            neckline=p.neckline,
            target_price=p.target_price,
            pattern_bars=p.pattern_bars,
            meta=p.meta,
        ))
    return enriched


def select_best_pattern_setup(
    df: pd.DataFrame,
    ind: Dict,
    *,
    silent: bool = False,
) -> Optional[PatternResult]:
    patterns = evaluate_patterns(df, ind)
    actionable = [p for p in patterns if p.signal != "HOLD" and p.score > 0]
    if not actionable:
        return None

    best = max(actionable, key=lambda p: p.score)
    if best.confidence < settings.SWING_MIN_CONFIDENCE:
        if not silent:
            trading_state.add_log(
                "SwingAgent",
                f"Best pattern '{best.name}' confidence {best.confidence:.0%} < min {settings.SWING_MIN_CONFIDENCE:.0%} — HOLD",
                level="info",
            )
        return None
    return best


SWING_LLM_SYSTEM_PROMPT = """SWING_LLM_AGENT
You are EvoTrade AI's swing trading decision engine for 4h crypto charts.
You must output STRICT JSON only.

Goal:
- Decide BUY / SELL / HOLD for a multi-day swing trade.
- Use detected chart patterns on the 4h timeframe.
- Prefer high-conviction pattern + trend alignment.

Rules:
1) Prefer HOLD when no clear pattern or weak confluence.
2) Confidence must be between 0 and 1.
3) If BUY/SELL, provide stop_loss and take_profit (wider than scalping).
4) Reference the pattern name from the candidate list.
5) Swing trades hold days — avoid noise entries.

Output JSON:
{
  "signal": "BUY|SELL|HOLD",
  "confidence": 0.0,
  "pattern": "pattern_name|none",
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "reasoning": "..."
}
"""


def _build_swing_llm_prompt(
    symbol: str,
    indicators: Dict,
    portfolio_dict: Dict,
    patterns: List[PatternResult],
    rule_best: Optional[PatternResult],
) -> str:
    lines = []
    for p in patterns[:12]:
        lines.append(
            f"- {p.name}: {p.signal} score={p.score:.0f} conf={p.confidence:.2f} "
            f"SL={p.sl_price:.4f} TP={p.tp_price:.4f} — {p.reasoning[:80]}"
        )
    return f"""Swing decision for {symbol} on 4h timeframe.

INDICATORS:
- close={indicators.get('close')}
- ema20={indicators.get('ema_20')} ema50={indicators.get('ema_50')}
- rsi={indicators.get('rsi')} adx={indicators.get('adx')}
- atr={indicators.get('atr')} trend={indicators.get('trend')}

PORTFOLIO:
- equity={portfolio_dict.get('equity')}
- open_trades={portfolio_dict.get('open_trades_count')}
- min_confidence={settings.SWING_MIN_CONFIDENCE}

ALL DETECTED 4H PATTERNS ({len(patterns)} found):
{chr(10).join(lines) if lines else '- none'}

BEST RULE PATTERN:
{rule_best.name if rule_best else 'none'} | {rule_best.signal if rule_best else 'HOLD'} | conf={rule_best.confidence if rule_best else 0}

Return JSON only."""


def _parse_swing_llm_response(raw: str, patterns: List[PatternResult]) -> Dict:
    pattern_map = {p.name: p for p in patterns}
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON")
        data = json.loads(match.group())
    except Exception as exc:
        trading_state.add_log("SwingAgent", f"LLM parse error: {exc}", level="warn")
        return {}

    signal = str(data.get("signal", "HOLD")).upper().strip()
    if signal not in ("BUY", "SELL", "HOLD"):
        signal = "HOLD"
    conf = max(0.0, min(float(data.get("confidence", 0) or 0), 1.0))
    pattern_name = str(data.get("pattern", "none")).strip()
    chosen = pattern_map.get(pattern_name)
    sl = data.get("stop_loss")
    tp = data.get("take_profit")
    return {
        "signal": signal,
        "confidence": conf,
        "pattern": pattern_name,
        "chosen_pattern": chosen,
        "stop_loss": float(sl) if sl is not None else (chosen.sl_price if chosen else None),
        "take_profit": float(tp) if tp is not None else (chosen.tp_price if chosen else None),
        "reasoning": str(data.get("reasoning", "")).strip(),
    }


def _passes_signal_stability(symbol: str, signal: str) -> Tuple[bool, str]:
    if signal not in ("BUY", "SELL"):
        return True, "ok"

    state = _signal_state.get(symbol, {"last_signal": None, "streak": 0})
    last = state.get("last_signal")
    streak = int(state.get("streak", 0))
    streak = streak + 1 if last == signal else 1
    _signal_state[symbol] = {"last_signal": signal, "streak": streak}

    need = max(1, settings.SWING_SIGNAL_CONFIRMATION_CYCLES)
    if streak < need:
        return False, f"awaiting 4h confirmation: {signal} streak {streak}/{need}"

    return True, "ok"


def run_swing_agent(
    symbol: str,
    indicators: Dict,
    df: pd.DataFrame,
    portfolio_dict: Dict,
) -> TradeDecision:
    """
    Core swing decision on 4h chart patterns.
    """
    tf = settings.SWING_TIMEFRAME
    trading_state.add_log(
        "SwingAgent",
        f"Scanning all chart patterns on {symbol} [{tf}]...",
    )

    allowed, reason = swing_risk.is_trading_allowed()
    if not allowed:
        trading_state.add_log("SwingAgent", f"BLOCKED: {reason}", level="warn")
        return _hold_decision(symbol, indicators, f"Risk gate: {reason}")

    all_patterns = evaluate_patterns(df, indicators)
    best = select_best_pattern_setup(df, indicators)

    summary = patterns_summary(all_patterns)
    trading_state.add_log(
        "SwingAgent",
        f"4h pattern scan: {summary['count']} detected — "
        + ", ".join(p["name"] for p in summary.get("patterns", [])[:5])
        or "none",
        level="info",
    )

    llm_out = {}
    if settings.SWING_USE_LLM and (all_patterns or best):
        try:
            raw = get_llm_response(
                _build_swing_llm_prompt(symbol, indicators, portfolio_dict, all_patterns, best),
                system=SWING_LLM_SYSTEM_PROMPT,
                max_tokens=500,
            )
            llm_out = _parse_swing_llm_response(raw, all_patterns)
        except Exception as exc:
            trading_state.add_log("SwingAgent", f"LLM failed: {exc} — using rules", level="warn")

    source = "RULE"
    final_signal = best.signal if best else "HOLD"
    final_conf = best.confidence if best else 0.0
    final_pattern = best
    final_sl = best.sl_price if best else None
    final_tp = best.tp_price if best else None
    final_reasoning = best.reasoning if best else "No 4h pattern cleared confidence threshold"

    if llm_out:
        source = "LLM"
        final_signal = llm_out.get("signal", final_signal)
        final_conf = llm_out.get("confidence", final_conf)
        if llm_out.get("chosen_pattern"):
            final_pattern = llm_out["chosen_pattern"]
        elif final_pattern is None and final_signal in ("BUY", "SELL"):
            matching = [p for p in all_patterns if p.signal == final_signal]
            final_pattern = max(matching, key=lambda p: p.score) if matching else None
        final_sl = llm_out.get("stop_loss", final_sl)
        final_tp = llm_out.get("take_profit", final_tp)
        final_reasoning = llm_out.get("reasoning") or final_reasoning

    entry = indicators.get("close", 0)
    atr = indicators.get("atr", entry * 0.02)
    if final_signal in ("BUY", "SELL") and final_sl and final_tp:
        final_sl, final_tp = _atr_sl_tp(entry, final_signal, atr, final_sl, final_tp)

    if final_signal in ("BUY", "SELL") and final_pattern is None and not settings.SWING_LLM_FINAL_AUTHORITY:
        return _hold_decision(symbol, indicators, f"No rule-supported pattern for {final_signal}")

    if final_signal == "HOLD" or final_conf < settings.SWING_MIN_CONFIDENCE:
        return _hold_decision(
            symbol, indicators,
            f"No trade: signal={final_signal} conf={final_conf:.0%} min={settings.SWING_MIN_CONFIDENCE:.0%}",
        )

    ok, stability_reason = _passes_signal_stability(symbol, final_signal)
    if not ok:
        return _hold_decision(symbol, indicators, stability_reason)

    if final_sl is None or final_tp is None:
        return _hold_decision(symbol, indicators, "Missing SL/TP after pattern analysis")

    sizing = swing_risk.calculate_position_size(
        entry_price=entry,
        sl_price=float(final_sl),
        current_equity=portfolio_dict.get("equity", settings.SWING_ACCOUNT_SIZE),
    )

    pattern_names = [p.name for p in all_patterns]
    decision = TradeDecision(
        id=str(uuid.uuid4())[:8],
        timestamp=datetime.now(timezone.utc).isoformat(),
        symbol=symbol,
        signal=TradeSignal(final_signal),
        confidence=final_conf,
        size_usdt=sizing["size_usdt"],
        entry_price=entry,
        stop_loss=float(final_sl),
        take_profit=float(final_tp),
        reasoning=(
            final_reasoning
            + f" | tf=4h"
            + f" | pattern={final_pattern.name if final_pattern else 'llm'}"
            + f" | source={source}"
            + f" | patterns_seen={len(all_patterns)}"
            + f" | risk=${sizing['risk_usdt']:.2f}"
        ),
        agent_contributions={
            "pattern": final_pattern.name if final_pattern else "llm_inferred",
            "timeframe": settings.SWING_TIMEFRAME,
            "score": str(round(final_pattern.score, 1)) if final_pattern else "n/a",
            "source": source,
            "patterns_detected": ",".join(pattern_names[:8]),
            "weekly_pnl": str(round(swing_risk.weekly_status()["weekly_pnl_usdt"], 2)),
        },
    )

    db.save_decision(decision)
    trading_state.add_decision(decision)

    emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(final_signal, "⚪")
    trading_state.add_log(
        "SwingAgent",
        f"{emoji} SWING {final_signal} | {(final_pattern.name if final_pattern else 'llm')} | "
        f"{source} | conf={final_conf:.0%} | {len(all_patterns)} patterns on 4h | "
        f"SL={float(final_sl):.2f} TP={float(final_tp):.2f}",
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


def close_all_swing_positions(reason: str = "Manual close-all"):
    price = trading_state.current_price
    if price <= 0:
        return []

    closed_ids = []
    with trading_state._lock:
        open_ids = [t.id for t in trading_state.open_trades]

    for tid in open_ids:
        trading_state.close_trade(tid, price)
        swing_risk.record_trade_close(tid, price, 0)
        closed_ids.append(tid)

    if closed_ids:
        trading_state.add_log(
            "SwingAgent",
            f"Close-all: {len(closed_ids)} position(s) @ {price:.4f} — {reason}",
            level="warn",
        )
    return closed_ids


def get_detected_patterns_for_symbol(symbol: str) -> Dict:
    """API helper: fetch 4h data and return all detected patterns."""
    from agents.data_agent import fetch_ohlcv, compute_indicators

    df = fetch_ohlcv(symbol, settings.SWING_TIMEFRAME, limit=settings.SWING_OHLCV_LIMIT)
    if df is None or len(df) < 30:
        return {"symbol": symbol, "timeframe": "4h", "patterns": [], "count": 0}

    df, ind = compute_indicators(df)
    patterns = evaluate_patterns(df, ind)
    return {
        "symbol": symbol,
        "timeframe": settings.SWING_TIMEFRAME,
        "count": len(patterns),
        "patterns": patterns_summary(patterns),
        "pattern_catalog": PATTERN_NAMES,
    }
