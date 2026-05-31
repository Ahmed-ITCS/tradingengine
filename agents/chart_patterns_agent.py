"""
EvoTrade AI - Chart Patterns Agent
Rule-based detection of classical swing chart patterns on OHLCV data.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.state import trading_state


Pattern = Dict[str, Any]

KNOWN_PATTERNS = (
    "double_top",
    "double_bottom",
    "head_and_shoulders",
    "inverse_head_and_shoulders",
    "ascending_triangle",
    "descending_triangle",
    "symmetric_triangle",
    "bull_flag",
    "bear_flag",
)


def _pct_diff(a: float, b: float) -> float:
    if b == 0:
        return 1.0
    return abs(a - b) / abs(b)


def find_pivots(
    df: pd.DataFrame,
    order: int = 5,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Local swing highs and lows using a symmetric window."""
    highs = df["high"].values
    lows = df["low"].values
    pivot_highs: List[Tuple[int, float]] = []
    pivot_lows: List[Tuple[int, float]] = []

    for i in range(order, len(df) - order):
        window_h = highs[i - order : i + order + 1]
        window_l = lows[i - order : i + order + 1]
        if highs[i] >= window_h.max():
            pivot_highs.append((i, float(highs[i])))
        if lows[i] <= window_l.min():
            pivot_lows.append((i, float(lows[i])))

    return pivot_highs, pivot_lows


def _cluster_levels(prices: List[float], tolerance: float = 0.015) -> List[float]:
    """Merge nearby price levels into representative zones."""
    if not prices:
        return []
    sorted_prices = sorted(prices)
    clusters: List[List[float]] = [[sorted_prices[0]]]
    for price in sorted_prices[1:]:
        if _pct_diff(price, clusters[-1][-1]) <= tolerance:
            clusters[-1].append(price)
        else:
            clusters.append([price])
    return [float(np.mean(c)) for c in clusters]


def detect_double_top(
    df: pd.DataFrame,
    pivot_highs: List[Tuple[int, float]],
    pivot_lows: List[Tuple[int, float]],
) -> List[Pattern]:
    patterns: List[Pattern] = []
    if len(pivot_highs) < 2:
        return patterns

    recent = pivot_highs[-4:]
    for i in range(len(recent) - 1):
        idx1, peak1 = recent[i]
        idx2, peak2 = recent[i + 1]
        if idx2 - idx1 < 8:
            continue
        if _pct_diff(peak1, peak2) > 0.02:
            continue

        between_lows = [p for idx, p in pivot_lows if idx1 < idx < idx2]
        if not between_lows:
            continue
        neckline = min(between_lows)
        if neckline >= min(peak1, peak2) * 0.97:
            continue

        height = ((peak1 + peak2) / 2) - neckline
        target = neckline - height
        bars_ago = len(df) - 1 - idx2
        conf = max(0.55, min(0.85, 0.75 - _pct_diff(peak1, peak2)))

        patterns.append({
            "name": "Double Top",
            "key": "double_top",
            "type": "reversal",
            "bias": "BEARISH",
            "confidence": round(conf, 2),
            "status": "confirmed" if df["close"].iloc[-1] < neckline else "forming",
            "neckline": round(neckline, 4),
            "target": round(target, 4),
            "stop_reference": round(max(peak1, peak2), 4),
            "bars_ago": bars_ago,
            "description": (
                f"Two peaks near ${peak2:,.2f} with neckline ${neckline:,.2f}; "
                f"bearish reversal if neckline breaks."
            ),
        })
        break

    return patterns


def detect_double_bottom(
    df: pd.DataFrame,
    pivot_highs: List[Tuple[int, float]],
    pivot_lows: List[Tuple[int, float]],
) -> List[Pattern]:
    patterns: List[Pattern] = []
    if len(pivot_lows) < 2:
        return patterns

    recent = pivot_lows[-4:]
    for i in range(len(recent) - 1):
        idx1, trough1 = recent[i]
        idx2, trough2 = recent[i + 1]
        if idx2 - idx1 < 8:
            continue
        if _pct_diff(trough1, trough2) > 0.02:
            continue

        between_highs = [p for idx, p in pivot_highs if idx1 < idx < idx2]
        if not between_highs:
            continue
        neckline = max(between_highs)
        if neckline <= max(trough1, trough2) * 1.03:
            continue

        depth = neckline - ((trough1 + trough2) / 2)
        target = neckline + depth
        bars_ago = len(df) - 1 - idx2
        conf = max(0.55, min(0.85, 0.75 - _pct_diff(trough1, trough2)))

        patterns.append({
            "name": "Double Bottom",
            "key": "double_bottom",
            "type": "reversal",
            "bias": "BULLISH",
            "confidence": round(conf, 2),
            "status": "confirmed" if df["close"].iloc[-1] > neckline else "forming",
            "neckline": round(neckline, 4),
            "target": round(target, 4),
            "stop_reference": round(min(trough1, trough2), 4),
            "bars_ago": bars_ago,
            "description": (
                f"Two troughs near ${trough2:,.2f} with neckline ${neckline:,.2f}; "
                f"bullish reversal if neckline breaks."
            ),
        })
        break

    return patterns


def detect_head_and_shoulders(
    df: pd.DataFrame,
    pivot_highs: List[Tuple[int, float]],
) -> List[Pattern]:
    patterns: List[Pattern] = []
    if len(pivot_highs) < 3:
        return patterns

    recent = pivot_highs[-5:]
    for i in range(len(recent) - 2):
        (idx_l, left), (idx_h, head), (idx_r, right) = recent[i], recent[i + 1], recent[i + 2]
        if not (idx_l < idx_h < idx_r):
            continue
        if head <= left * 1.015 or head <= right * 1.015:
            continue
        if _pct_diff(left, right) > 0.03:
            continue

        neckline = float(df["low"].iloc[idx_l:idx_r + 1].min())
        height = head - neckline
        target = neckline - height
        bars_ago = len(df) - 1 - idx_r

        patterns.append({
            "name": "Head and Shoulders",
            "key": "head_and_shoulders",
            "type": "reversal",
            "bias": "BEARISH",
            "confidence": 0.72,
            "status": "confirmed" if df["close"].iloc[-1] < neckline else "forming",
            "neckline": round(neckline, 4),
            "target": round(target, 4),
            "stop_reference": round(head, 4),
            "bars_ago": bars_ago,
            "description": (
                f"H&S with head ${head:,.2f}, shoulders ${left:,.2f}/${right:,.2f}, "
                f"neckline ${neckline:,.2f}."
            ),
        })
        break

    return patterns


def detect_inverse_head_and_shoulders(
    df: pd.DataFrame,
    pivot_lows: List[Tuple[int, float]],
) -> List[Pattern]:
    patterns: List[Pattern] = []
    if len(pivot_lows) < 3:
        return patterns

    recent = pivot_lows[-5:]
    for i in range(len(recent) - 2):
        (idx_l, left), (idx_h, head), (idx_r, right) = recent[i], recent[i + 1], recent[i + 2]
        if not (idx_l < idx_h < idx_r):
            continue
        if head >= left * 0.985 or head >= right * 0.985:
            continue
        if _pct_diff(left, right) > 0.03:
            continue

        neckline = float(df["high"].iloc[idx_l:idx_r + 1].max())
        depth = neckline - head
        target = neckline + depth
        bars_ago = len(df) - 1 - idx_r

        patterns.append({
            "name": "Inverse Head and Shoulders",
            "key": "inverse_head_and_shoulders",
            "type": "reversal",
            "bias": "BULLISH",
            "confidence": 0.72,
            "status": "confirmed" if df["close"].iloc[-1] > neckline else "forming",
            "neckline": round(neckline, 4),
            "target": round(target, 4),
            "stop_reference": round(head, 4),
            "bars_ago": bars_ago,
            "description": (
                f"Inverse H&S with head ${head:,.2f}, shoulders ${left:,.2f}/${right:,.2f}, "
                f"neckline ${neckline:,.2f}."
            ),
        })
        break

    return patterns


def detect_triangles(df: pd.DataFrame, lookback: int = 50) -> List[Pattern]:
    """Detect ascending, descending, and symmetric triangles."""
    patterns: List[Pattern] = []
    window = df.tail(lookback)
    if len(window) < 20:
        return patterns

    highs = window["high"].values
    lows = window["low"].values
    x = np.arange(len(window))

    high_slope = float(np.polyfit(x, highs, 1)[0])
    low_slope = float(np.polyfit(x, lows, 1)[0])
    high_range = (highs.max() - highs.min()) / max(highs.mean(), 1e-9)
    low_range = (lows.max() - lows.min()) / max(lows.mean(), 1e-9)
    close = float(window["close"].iloc[-1])

    flat_highs = high_range < 0.04 and abs(high_slope) < close * 0.0005
    flat_lows = low_range < 0.04 and abs(low_slope) < close * 0.0005
    rising_lows = low_slope > close * 0.0008
    falling_highs = high_slope < -close * 0.0008
    converging = high_slope < 0 and low_slope > 0

    if flat_highs and rising_lows:
        support = float(window["low"].iloc[-1])
        resistance = float(window["high"].max())
        patterns.append({
            "name": "Ascending Triangle",
            "key": "ascending_triangle",
            "type": "continuation",
            "bias": "BULLISH",
            "confidence": 0.68,
            "status": "forming",
            "neckline": round(resistance, 4),
            "target": round(resistance + (resistance - support), 4),
            "stop_reference": round(support, 4),
            "bars_ago": 0,
            "description": (
                f"Flat resistance near ${resistance:,.2f} with rising support; "
                f"bullish breakout watch."
            ),
        })
    elif flat_lows and falling_highs:
        support = float(window["low"].min())
        resistance = float(window["high"].iloc[-1])
        patterns.append({
            "name": "Descending Triangle",
            "key": "descending_triangle",
            "type": "continuation",
            "bias": "BEARISH",
            "confidence": 0.68,
            "status": "forming",
            "neckline": round(support, 4),
            "target": round(support - (resistance - support), 4),
            "stop_reference": round(resistance, 4),
            "bars_ago": 0,
            "description": (
                f"Flat support near ${support:,.2f} with falling highs; "
                f"bearish breakdown watch."
            ),
        })
    elif converging and abs(high_slope) > close * 0.0003 and abs(low_slope) > close * 0.0003:
        apex_high = float(window["high"].iloc[-1])
        apex_low = float(window["low"].iloc[-1])
        height = float(window["high"].max() - window["low"].min())
        patterns.append({
            "name": "Symmetric Triangle",
            "key": "symmetric_triangle",
            "type": "continuation",
            "bias": "NEUTRAL",
            "confidence": 0.62,
            "status": "forming",
            "neckline": round((apex_high + apex_low) / 2, 4),
            "target": round(close + height * 0.5, 4),
            "stop_reference": round(apex_low, 4),
            "bars_ago": 0,
            "description": "Converging highs and lows — wait for directional breakout.",
        })

    return patterns


def detect_flags(df: pd.DataFrame, lookback: int = 40) -> List[Pattern]:
    """Detect bull/bear flags after a sharp impulse move."""
    patterns: List[Pattern] = []
    if len(df) < lookback + 10:
        return patterns

    impulse = df.iloc[-(lookback + 10):-10]
    flag = df.iloc[-10:]
    if len(impulse) < 10 or len(flag) < 5:
        return patterns

    impulse_move = (impulse["close"].iloc[-1] - impulse["close"].iloc[0]) / impulse["close"].iloc[0]
    flag_range = (flag["high"].max() - flag["low"].min()) / flag["close"].mean()
    flag_slope = float(np.polyfit(np.arange(len(flag)), flag["close"].values, 1)[0])
    close = float(df["close"].iloc[-1])

    if abs(impulse_move) < 0.03 or flag_range > 0.035:
        return patterns

    if impulse_move > 0.03 and flag_slope < 0:
        pole = impulse_move * close
        patterns.append({
            "name": "Bull Flag",
            "key": "bull_flag",
            "type": "continuation",
            "bias": "BULLISH",
            "confidence": 0.65,
            "status": "forming",
            "neckline": round(float(flag["high"].max()), 4),
            "target": round(close + pole, 4),
            "stop_reference": round(float(flag["low"].min()), 4),
            "bars_ago": 0,
            "description": "Sharp rally followed by downward-sloping consolidation — bull continuation setup.",
        })
    elif impulse_move < -0.03 and flag_slope > 0:
        pole = abs(impulse_move) * close
        patterns.append({
            "name": "Bear Flag",
            "key": "bear_flag",
            "type": "continuation",
            "bias": "BEARISH",
            "confidence": 0.65,
            "status": "forming",
            "neckline": round(float(flag["low"].min()), 4),
            "target": round(close - pole, 4),
            "stop_reference": round(float(flag["high"].max()), 4),
            "bars_ago": 0,
            "description": "Sharp selloff followed by upward-sloping consolidation — bear continuation setup.",
        })

    return patterns


def analyze_chart_patterns(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> Dict[str, Any]:
    """Run all known pattern detectors on OHLCV data."""
    pivot_highs, pivot_lows = find_pivots(df, order=5)

    detected: List[Pattern] = []
    detected.extend(detect_double_top(df, pivot_highs, pivot_lows))
    detected.extend(detect_double_bottom(df, pivot_highs, pivot_lows))
    detected.extend(detect_head_and_shoulders(df, pivot_highs))
    detected.extend(detect_inverse_head_and_shoulders(df, pivot_lows))
    detected.extend(detect_triangles(df))
    detected.extend(detect_flags(df))

    # De-duplicate by pattern key, keep highest confidence
    by_key: Dict[str, Pattern] = {}
    for p in detected:
        key = p["key"]
        if key not in by_key or p["confidence"] > by_key[key]["confidence"]:
            by_key[key] = p
    patterns = sorted(by_key.values(), key=lambda p: p["confidence"], reverse=True)

    support_levels = _cluster_levels([p for _, p in pivot_lows[-8:]])
    resistance_levels = _cluster_levels([p for _, p in pivot_highs[-8:]])

    bullish = sum(1 for p in patterns if p["bias"] == "BULLISH")
    bearish = sum(1 for p in patterns if p["bias"] == "BEARISH")
    if bullish > bearish:
        structure = "BULLISH"
    elif bearish > bullish:
        structure = "BEARISH"
    elif patterns:
        structure = "MIXED"
    else:
        structure = "NEUTRAL"

    if patterns:
        top = patterns[0]
        summary = (
            f"{len(patterns)} pattern(s) on {timeframe}: "
            f"primary {top['name']} ({top['bias']}, {top['confidence']:.0%} conf, {top['status']})."
        )
    else:
        summary = f"No classical chart patterns detected on {timeframe} window."

    return {
        "patterns": patterns,
        "support_levels": [round(x, 4) for x in support_levels[:4]],
        "resistance_levels": [round(x, 4) for x in resistance_levels[:4]],
        "structure": structure,
        "summary": summary,
        "pattern_count": len(patterns),
        "known_patterns_tested": list(KNOWN_PATTERNS),
        "symbol": symbol,
        "timeframe": timeframe,
        "candles_analyzed": len(df),
        "timestamp": datetime.utcnow().isoformat(),
    }


def format_patterns_for_prompt(analysis: Dict[str, Any]) -> str:
    """Human-readable block for the decision agent."""
    lines = [
        f"Market Structure: {analysis.get('structure', 'NEUTRAL')}",
        f"Summary: {analysis.get('summary', 'N/A')}",
    ]

    supports = analysis.get("support_levels") or []
    resistances = analysis.get("resistance_levels") or []
    if supports:
        lines.append("Support: " + ", ".join(f"${x:,.2f}" for x in supports))
    if resistances:
        lines.append("Resistance: " + ", ".join(f"${x:,.2f}" for x in resistances))

    patterns = analysis.get("patterns") or []
    if not patterns:
        lines.append("Detected Patterns: none")
        return "\n".join(lines)

    lines.append("Detected Patterns:")
    for p in patterns[:5]:
        lines.append(
            f"- {p['name']} | {p['bias']} | conf={p['confidence']:.0%} | "
            f"{p['status']} | neckline=${p.get('neckline', 0):,.2f} | "
            f"target=${p.get('target', 0):,.2f} | {p['description']}"
        )
    return "\n".join(lines)


def run_chart_patterns_agent(
    symbol: str,
    timeframe: str,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Main entry point — scans OHLCV for known chart patterns."""
    trading_state.add_log("ChartPatternsAgent", f"Scanning {symbol} [{timeframe}] for chart patterns...")

    if df is None:
        df = trading_state.last_ohlcv
    if df is None or len(df) < 50:
        raise RuntimeError("Insufficient OHLCV data for chart pattern analysis")

    analysis = analyze_chart_patterns(df, symbol, timeframe)
    trading_state.last_chart_patterns = analysis

    if analysis["pattern_count"]:
        names = ", ".join(p["name"] for p in analysis["patterns"][:3])
        trading_state.add_log(
            "ChartPatternsAgent",
            f"{analysis['pattern_count']} pattern(s): {names} | structure={analysis['structure']}",
            data=analysis,
            level="success",
        )
    else:
        trading_state.add_log(
            "ChartPatternsAgent",
            f"No patterns matched ({len(KNOWN_PATTERNS)} rules tested)",
            data=analysis,
            level="info",
        )

    return analysis
