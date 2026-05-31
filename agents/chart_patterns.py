"""
EvoTrade AI - Chart Pattern Detection (4h swing timeframe)

Detects classical chart patterns on OHLCV data:
  - Double top / double bottom
  - Head & shoulders / inverse H&S
  - Ascending / descending / symmetrical triangles
  - Bull flag / bear flag
  - Rising / falling wedge
  - Support & resistance breakout
  - Candlestick: engulfing, hammer, shooting star, morning/evening star
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PatternResult:
    """A detected chart pattern with trade direction and confidence."""
    name: str
    signal: str          # BUY | SELL | HOLD
    score: float         # 0–100
    confidence: float    # 0–1
    sl_price: float
    tp_price: float
    reasoning: str
    neckline: Optional[float] = None
    target_price: Optional[float] = None
    pattern_bars: int = 0
    meta: Dict = field(default_factory=dict)


def _swing_highs_lows(
    df: pd.DataFrame,
    window: int = 5,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Return (index, price) lists of local swing highs and lows."""
    highs, lows = [], []
    n = len(df)
    if n < window * 2 + 1:
        return highs, lows

    h = df["high"].values
    l = df["low"].values

    for i in range(window, n - window):
        if h[i] == max(h[i - window : i + window + 1]):
            highs.append((i, float(h[i])))
        if l[i] == min(l[i - window : i + window + 1]):
            lows.append((i, float(l[i])))
    return highs, lows


def _pct_diff(a: float, b: float) -> float:
    if b == 0:
        return 1.0
    return abs(a - b) / abs(b)


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return float(df["close"].iloc[-1]) * 0.02
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return float(val) if not pd.isna(val) else float(df["close"].iloc[-1]) * 0.02


def _sl_tp_from_pattern(
    close: float,
    signal: str,
    sl: float,
    tp: float,
    atr: float,
    min_sl_pct: float = 0.015,
    tp_mult: float = 3.0,
) -> Tuple[float, float]:
    """Ensure SL/TP are valid; widen with ATR floor for swing trades."""
    if signal == "HOLD":
        return close, close

    min_dist = max(atr * 2.0, close * min_sl_pct)
    if signal == "BUY":
        sl = min(sl, close - min_dist) if sl < close else close - min_dist
        tp_dist = max(tp - close, min_dist * tp_mult) if tp > close else min_dist * tp_mult
        tp = close + tp_dist
    else:
        sl = max(sl, close + min_dist) if sl > close else close + min_dist
        tp_dist = max(close - tp, min_dist * tp_mult) if tp < close else min_dist * tp_mult
        tp = close - tp_dist
    return sl, tp


# ── Individual pattern detectors ──────────────────────────────────────────────

def detect_double_top_bottom(df: pd.DataFrame, atr: float) -> Optional[PatternResult]:
    """Double top (SELL) or double bottom (BUY) on recent swing points."""
    close = float(df["close"].iloc[-1])
    highs, lows = _swing_highs_lows(df, window=4)

    if len(highs) >= 2:
        h1, h2 = highs[-2], highs[-1]
        if _pct_diff(h1[1], h2[1]) < 0.012 and h2[0] > h1[0]:
            neckline = min(float(df["low"].iloc[h1[0]:h2[0] + 1].min()), close)
            height = h2[1] - neckline
            if close < neckline * 0.998 and height > atr * 0.5:
                sl = max(h2[1], close + atr)
                tp = close - height
                sl, tp = _sl_tp_from_pattern(close, "SELL", sl, tp, atr)
                score = 70 + min(20, height / atr * 5)
                return PatternResult(
                    name="double_top",
                    signal="SELL",
                    score=score,
                    confidence=min(score / 100, 0.88),
                    sl_price=sl,
                    tp_price=tp,
                    reasoning=f"Double top at {h1[1]:.2f}/{h2[1]:.2f}, neckline break {neckline:.2f}",
                    neckline=neckline,
                    target_price=tp,
                    pattern_bars=h2[0] - h1[0],
                )

    if len(lows) >= 2:
        l1, l2 = lows[-2], lows[-1]
        if _pct_diff(l1[1], l2[1]) < 0.012 and l2[0] > l1[0]:
            neckline = max(float(df["high"].iloc[l1[0]:l2[0] + 1].max()), close)
            height = neckline - l2[1]
            if close > neckline * 1.002 and height > atr * 0.5:
                sl = min(l2[1], close - atr)
                tp = close + height
                sl, tp = _sl_tp_from_pattern(close, "BUY", sl, tp, atr)
                score = 70 + min(20, height / atr * 5)
                return PatternResult(
                    name="double_bottom",
                    signal="BUY",
                    score=score,
                    confidence=min(score / 100, 0.88),
                    sl_price=sl,
                    tp_price=tp,
                    reasoning=f"Double bottom at {l1[1]:.2f}/{l2[1]:.2f}, neckline break {neckline:.2f}",
                    neckline=neckline,
                    target_price=tp,
                    pattern_bars=l2[0] - l1[0],
                )
    return None


def detect_head_shoulders(df: pd.DataFrame, atr: float) -> Optional[PatternResult]:
    """Head & shoulders (SELL) or inverse H&S (BUY)."""
    close = float(df["close"].iloc[-1])
    highs, lows = _swing_highs_lows(df, window=3)

    if len(highs) >= 3:
        ls, head, rs = highs[-3], highs[-2], highs[-1]
        if head[1] > ls[1] * 1.005 and head[1] > rs[1] * 1.005:
            if _pct_diff(ls[1], rs[1]) < 0.02:
                neckline = float(df["low"].iloc[ls[0]:rs[0] + 1].min())
                height = head[1] - neckline
                if close < neckline * 0.997 and height > atr:
                    sl = head[1]
                    tp = close - height
                    sl, tp = _sl_tp_from_pattern(close, "SELL", sl, tp, atr)
                    score = 75 + min(15, height / atr * 3)
                    return PatternResult(
                        name="head_shoulders",
                        signal="SELL",
                        score=score,
                        confidence=min(score / 100, 0.90),
                        sl_price=sl,
                        tp_price=tp,
                        reasoning=f"H&S head={head[1]:.2f}, shoulders {ls[1]:.2f}/{rs[1]:.2f}, neckline {neckline:.2f}",
                        neckline=neckline,
                        target_price=tp,
                        pattern_bars=rs[0] - ls[0],
                    )

    if len(lows) >= 3:
        ls, head, rs = lows[-3], lows[-2], lows[-1]
        if head[1] < ls[1] * 0.995 and head[1] < rs[1] * 0.995:
            if _pct_diff(ls[1], rs[1]) < 0.02:
                neckline = float(df["high"].iloc[ls[0]:rs[0] + 1].max())
                height = neckline - head[1]
                if close > neckline * 1.003 and height > atr:
                    sl = head[1]
                    tp = close + height
                    sl, tp = _sl_tp_from_pattern(close, "BUY", sl, tp, atr)
                    score = 75 + min(15, height / atr * 3)
                    return PatternResult(
                        name="inverse_head_shoulders",
                        signal="BUY",
                        score=score,
                        confidence=min(score / 100, 0.90),
                        sl_price=sl,
                        tp_price=tp,
                        reasoning=f"Inverse H&S head={head[1]:.2f}, shoulders {ls[1]:.2f}/{rs[1]:.2f}",
                        neckline=neckline,
                        target_price=tp,
                        pattern_bars=rs[0] - ls[0],
                    )
    return None


def detect_triangles(df: pd.DataFrame, atr: float) -> Optional[PatternResult]:
    """Ascending, descending, or symmetrical triangle breakout."""
    if len(df) < 40:
        return None

    close = float(df["close"].iloc[-1])
    window = df.iloc[-40:]
    highs, lows = _swing_highs_lows(window, window=3)
    if len(highs) < 2 or len(lows) < 2:
        return None

    high_prices = [p for _, p in highs[-4:]]
    low_prices = [p for _, p in lows[-4:]]

    high_slope = (high_prices[-1] - high_prices[0]) / max(len(high_prices) - 1, 1)
    low_slope = (low_prices[-1] - low_prices[0]) / max(len(low_prices) - 1, 1)

    recent_high = max(high_prices[-2:])
    recent_low = min(low_prices[-2:])

    # Ascending triangle: flat top, rising lows
    if abs(high_slope) < atr * 0.02 and low_slope > atr * 0.01:
        if close > recent_high * 1.002:
            sl = recent_low - atr * 0.5
            height = recent_high - min(low_prices)
            tp = close + height
            sl, tp = _sl_tp_from_pattern(close, "BUY", sl, tp, atr)
            score = 68 + min(20, (close - recent_high) / atr * 10)
            return PatternResult(
                name="ascending_triangle",
                signal="BUY",
                score=score,
                confidence=min(score / 100, 0.85),
                sl_price=sl,
                tp_price=tp,
                reasoning=f"Ascending triangle breakout above {recent_high:.2f}",
                neckline=recent_high,
                target_price=tp,
                pattern_bars=40,
            )

    # Descending triangle: flat bottom, falling highs
    if abs(low_slope) < atr * 0.02 and high_slope < -atr * 0.01:
        if close < recent_low * 0.998:
            sl = recent_high + atr * 0.5
            height = max(high_prices) - recent_low
            tp = close - height
            sl, tp = _sl_tp_from_pattern(close, "SELL", sl, tp, atr)
            score = 68 + min(20, (recent_low - close) / atr * 10)
            return PatternResult(
                name="descending_triangle",
                signal="SELL",
                score=score,
                confidence=min(score / 100, 0.85),
                sl_price=sl,
                tp_price=tp,
                reasoning=f"Descending triangle breakdown below {recent_low:.2f}",
                neckline=recent_low,
                target_price=tp,
                pattern_bars=40,
            )

    # Symmetrical: converging highs and lows
    if high_slope < 0 and low_slope > 0:
        mid = (recent_high + recent_low) / 2
        if close > recent_high * 1.002:
            sl = recent_low - atr
            height = recent_high - recent_low
            tp = close + height
            sl, tp = _sl_tp_from_pattern(close, "BUY", sl, tp, atr)
            score = 62 + min(18, height / atr * 4)
            return PatternResult(
                name="symmetrical_triangle",
                signal="BUY",
                score=score,
                confidence=min(score / 100, 0.82),
                sl_price=sl,
                tp_price=tp,
                reasoning=f"Symmetrical triangle bullish breakout above {recent_high:.2f}",
                target_price=tp,
                pattern_bars=40,
            )
        if close < recent_low * 0.998:
            sl = recent_high + atr
            height = recent_high - recent_low
            tp = close - height
            sl, tp = _sl_tp_from_pattern(close, "SELL", sl, tp, atr)
            score = 62 + min(18, height / atr * 4)
            return PatternResult(
                name="symmetrical_triangle",
                signal="SELL",
                score=score,
                confidence=min(score / 100, 0.82),
                sl_price=sl,
                tp_price=tp,
                reasoning=f"Symmetrical triangle bearish breakdown below {recent_low:.2f}",
                target_price=tp,
                pattern_bars=40,
            )
    return None


def detect_flags(df: pd.DataFrame, atr: float) -> Optional[PatternResult]:
    """Bull flag (BUY) or bear flag (SELL) after strong impulse move."""
    if len(df) < 30:
        return None

    close = float(df["close"].iloc[-1])
    impulse = df.iloc[-30:-10]
    flag = df.iloc[-10:]

    impulse_move = float(impulse["close"].iloc[-1]) - float(impulse["close"].iloc[0])
    impulse_pct = impulse_move / float(impulse["close"].iloc[0]) if impulse["close"].iloc[0] else 0

    flag_range = float(flag["high"].max()) - float(flag["low"].min())
    flag_slope = float(flag["close"].iloc[-1]) - float(flag["close"].iloc[0])

    # Bull flag: strong up impulse, slight downward/sideways flag
    if impulse_pct > 0.03 and flag_range < abs(impulse_move) * 0.5 and flag_slope <= 0:
        pole_top = float(impulse["high"].max())
        flag_high = float(flag["high"].max())
        if close > flag_high * 1.001:
            sl = float(flag["low"].min()) - atr * 0.3
            tp = close + abs(impulse_move)
            sl, tp = _sl_tp_from_pattern(close, "BUY", sl, tp, atr)
            score = 72 + min(18, impulse_pct * 200)
            return PatternResult(
                name="bull_flag",
                signal="BUY",
                score=score,
                confidence=min(score / 100, 0.87),
                sl_price=sl,
                tp_price=tp,
                reasoning=f"Bull flag breakout, impulse +{impulse_pct*100:.1f}%, flag break {flag_high:.2f}",
                target_price=tp,
                pattern_bars=30,
            )

    # Bear flag
    if impulse_pct < -0.03 and flag_range < abs(impulse_move) * 0.5 and flag_slope >= 0:
        flag_low = float(flag["low"].min())
        if close < flag_low * 0.999:
            sl = float(flag["high"].max()) + atr * 0.3
            tp = close - abs(impulse_move)
            sl, tp = _sl_tp_from_pattern(close, "SELL", sl, tp, atr)
            score = 72 + min(18, abs(impulse_pct) * 200)
            return PatternResult(
                name="bear_flag",
                signal="SELL",
                score=score,
                confidence=min(score / 100, 0.87),
                sl_price=sl,
                tp_price=tp,
                reasoning=f"Bear flag breakdown, impulse {impulse_pct*100:.1f}%, break {flag_low:.2f}",
                target_price=tp,
                pattern_bars=30,
            )
    return None


def detect_wedges(df: pd.DataFrame, atr: float) -> Optional[PatternResult]:
    """Rising wedge (bearish) or falling wedge (bullish)."""
    if len(df) < 35:
        return None

    close = float(df["close"].iloc[-1])
    window = df.iloc[-35:]
    highs, lows = _swing_highs_lows(window, window=3)
    if len(highs) < 2 or len(lows) < 2:
        return None

    hp = [p for _, p in highs[-3:]]
    lp = [p for _, p in lows[-3:]]

    high_rise = hp[-1] - hp[0]
    low_rise = lp[-1] - lp[0]

    # Rising wedge: both rising but highs rise slower → bearish
    if high_rise > 0 and low_rise > 0 and low_rise > high_rise * 1.2:
        support = lp[-1]
        if close < support * 0.998:
            sl = hp[-1] + atr * 0.5
            height = hp[-1] - lp[0]
            tp = close - height
            sl, tp = _sl_tp_from_pattern(close, "SELL", sl, tp, atr)
            score = 65 + min(20, height / atr * 3)
            return PatternResult(
                name="rising_wedge",
                signal="SELL",
                score=score,
                confidence=min(score / 100, 0.84),
                sl_price=sl,
                tp_price=tp,
                reasoning=f"Rising wedge breakdown below support {support:.2f}",
                target_price=tp,
                pattern_bars=35,
            )

    # Falling wedge: both falling but lows fall slower → bullish
    if high_rise < 0 and low_rise < 0 and abs(high_rise) > abs(low_rise) * 1.2:
        resistance = hp[-1]
        if close > resistance * 1.002:
            sl = lp[-1] - atr * 0.5
            height = hp[0] - lp[-1]
            tp = close + height
            sl, tp = _sl_tp_from_pattern(close, "BUY", sl, tp, atr)
            score = 65 + min(20, height / atr * 3)
            return PatternResult(
                name="falling_wedge",
                signal="BUY",
                score=score,
                confidence=min(score / 100, 0.84),
                sl_price=sl,
                tp_price=tp,
                reasoning=f"Falling wedge breakout above resistance {resistance:.2f}",
                target_price=tp,
                pattern_bars=35,
            )
    return None


def detect_sr_breakout(df: pd.DataFrame, atr: float) -> Optional[PatternResult]:
    """Swing high/low support & resistance breakout (50-bar lookback)."""
    if len(df) < 55:
        return None

    close = float(df["close"].iloc[-1])
    lookback = df.iloc[-51:-1]
    resistance = float(lookback["high"].max())
    support = float(lookback["low"].min())

    if close > resistance * 1.003:
        sl = support
        height = resistance - support
        tp = close + height * 0.618
        sl, tp = _sl_tp_from_pattern(close, "BUY", sl, tp, atr)
        score = 60 + min(25, (close - resistance) / atr * 8)
        return PatternResult(
            name="resistance_breakout",
            signal="BUY",
            score=score,
            confidence=min(score / 100, 0.83),
            sl_price=sl,
            tp_price=tp,
            reasoning=f"Resistance breakout above 50-bar high {resistance:.2f}",
            neckline=resistance,
            target_price=tp,
            pattern_bars=50,
        )

    if close < support * 0.997:
        sl = resistance
        height = resistance - support
        tp = close - height * 0.618
        sl, tp = _sl_tp_from_pattern(close, "SELL", sl, tp, atr)
        score = 60 + min(25, (support - close) / atr * 8)
        return PatternResult(
            name="support_breakdown",
            signal="SELL",
            score=score,
            confidence=min(score / 100, 0.83),
            sl_price=sl,
            tp_price=tp,
            reasoning=f"Support breakdown below 50-bar low {support:.2f}",
            neckline=support,
            target_price=tp,
            pattern_bars=50,
        )
    return None


def detect_candlestick_patterns(df: pd.DataFrame, atr: float) -> Optional[PatternResult]:
    """Single/multi-candle reversal patterns on latest bars."""
    if len(df) < 3:
        return None

    close = float(df["close"].iloc[-1])
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    body = abs(c[-1] - o[-1])
    range_ = h[-1] - l[-1]
    if range_ <= 0:
        return None

    lower_wick = min(o[-1], c[-1]) - l[-1]
    upper_wick = h[-1] - max(o[-1], c[-1])

    # Hammer (bullish)
    if lower_wick > body * 2 and upper_wick < body * 0.5 and c[-1] > o[-1]:
        sl = l[-1] - atr * 0.3
        tp = close + atr * 3
        sl, tp = _sl_tp_from_pattern(close, "BUY", sl, tp, atr)
        score = 58
        return PatternResult(
            name="hammer",
            signal="BUY",
            score=score,
            confidence=0.58,
            sl_price=sl,
            tp_price=tp,
            reasoning="Bullish hammer candle at swing low",
            pattern_bars=1,
        )

    # Shooting star (bearish)
    if upper_wick > body * 2 and lower_wick < body * 0.5 and c[-1] < o[-1]:
        sl = h[-1] + atr * 0.3
        tp = close - atr * 3
        sl, tp = _sl_tp_from_pattern(close, "SELL", sl, tp, atr)
        score = 58
        return PatternResult(
            name="shooting_star",
            signal="SELL",
            score=score,
            confidence=0.58,
            sl_price=sl,
            tp_price=tp,
            reasoning="Bearish shooting star at swing high",
            pattern_bars=1,
        )

    # Bullish engulfing
    if len(c) >= 2:
        prev_bear = c[-2] < o[-2]
        curr_bull = c[-1] > o[-1]
        if prev_bear and curr_bull and o[-1] <= c[-2] and c[-1] >= o[-2]:
            sl = l[-1] - atr * 0.5
            tp = close + (close - sl) * 2.5
            sl, tp = _sl_tp_from_pattern(close, "BUY", sl, tp, atr)
            score = 62
            return PatternResult(
                name="bullish_engulfing",
                signal="BUY",
                score=score,
                confidence=0.62,
                sl_price=sl,
                tp_price=tp,
                reasoning="Bullish engulfing candle pattern",
                pattern_bars=2,
            )

        prev_bull = c[-2] > o[-2]
        curr_bear = c[-1] < o[-1]
        if prev_bull and curr_bear and o[-1] >= c[-2] and c[-1] <= o[-2]:
            sl = h[-1] + atr * 0.5
            tp = close - (sl - close) * 2.5
            sl, tp = _sl_tp_from_pattern(close, "SELL", sl, tp, atr)
            score = 62
            return PatternResult(
                name="bearish_engulfing",
                signal="SELL",
                score=score,
                confidence=0.62,
                sl_price=sl,
                tp_price=tp,
                reasoning="Bearish engulfing candle pattern",
                pattern_bars=2,
            )

    # Morning star (3-candle bullish)
    if len(c) >= 3:
        first_bear = c[-3] < o[-3] and abs(c[-3] - o[-3]) > atr * 0.3
        small_body = abs(c[-2] - o[-2]) < atr * 0.4
        third_bull = c[-1] > o[-1] and c[-1] > (o[-3] + c[-3]) / 2
        if first_bear and small_body and third_bull:
            sl = min(l[-3], l[-2], l[-1]) - atr * 0.3
            tp = close + (close - sl) * 2.5
            sl, tp = _sl_tp_from_pattern(close, "BUY", sl, tp, atr)
            return PatternResult(
                name="morning_star",
                signal="BUY",
                score=68,
                confidence=0.68,
                sl_price=sl,
                tp_price=tp,
                reasoning="Morning star 3-candle reversal",
                pattern_bars=3,
            )

        first_bull = c[-3] > o[-3] and abs(c[-3] - o[-3]) > atr * 0.3
        third_bear = c[-1] < o[-1] and c[-1] < (o[-3] + c[-3]) / 2
        if first_bull and small_body and third_bear:
            sl = max(h[-3], h[-2], h[-1]) + atr * 0.3
            tp = close - (sl - close) * 2.5
            sl, tp = _sl_tp_from_pattern(close, "SELL", sl, tp, atr)
            return PatternResult(
                name="evening_star",
                signal="SELL",
                score=68,
                confidence=0.68,
                sl_price=sl,
                tp_price=tp,
                reasoning="Evening star 3-candle reversal",
                pattern_bars=3,
            )

    return None


# ── Public API ────────────────────────────────────────────────────────────────

ALL_PATTERN_DETECTORS = [
    detect_double_top_bottom,
    detect_head_shoulders,
    detect_triangles,
    detect_flags,
    detect_wedges,
    detect_sr_breakout,
    detect_candlestick_patterns,
]

PATTERN_NAMES = [
    "double_top", "double_bottom",
    "head_shoulders", "inverse_head_shoulders",
    "ascending_triangle", "descending_triangle", "symmetrical_triangle",
    "bull_flag", "bear_flag",
    "rising_wedge", "falling_wedge",
    "resistance_breakout", "support_breakdown",
    "hammer", "shooting_star",
    "bullish_engulfing", "bearish_engulfing",
    "morning_star", "evening_star",
]


def detect_all_patterns(df: pd.DataFrame) -> List[PatternResult]:
    """
    Run every chart pattern detector on the given OHLCV frame.
    Intended for 4h swing timeframe with 100+ bars.
    Returns all detected patterns (may be empty).
    """
    if df is None or len(df) < 30:
        return []

    atr = _atr(df)
    results: List[PatternResult] = []

    for detector in ALL_PATTERN_DETECTORS:
        try:
            hit = detector(df, atr)
            if hit is not None and hit.signal != "HOLD":
                results.append(hit)
        except Exception:
            continue

    return results


def select_best_pattern(df: pd.DataFrame, min_confidence: float = 0.55) -> Optional[PatternResult]:
    """Return highest-scoring pattern above confidence threshold."""
    patterns = detect_all_patterns(df)
    if not patterns:
        return None

    best = max(patterns, key=lambda p: p.score)
    if best.confidence < min_confidence:
        return None
    return best


def patterns_summary(patterns: List[PatternResult]) -> Dict:
    """Compact summary for logging / LLM prompts."""
    return {
        "count": len(patterns),
        "patterns": [
            {
                "name": p.name,
                "signal": p.signal,
                "score": round(p.score, 1),
                "confidence": round(p.confidence, 2),
                "reasoning": p.reasoning,
            }
            for p in sorted(patterns, key=lambda x: -x.score)
        ],
    }
