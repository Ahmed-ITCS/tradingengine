"""
EvoTrade AI - Decision Agent
Deterministic swing signal from TA + chart patterns + news sentiment scores.
No LLM — every rule and weight is explicit code.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from datetime import datetime

from core.state import trading_state, TradeDecision, TradeSignal
from core.database import db
from config import settings


@dataclass
class ModuleScore:
    bull: float = 0.0
    bear: float = 0.0
    reasons: List[str] = field(default_factory=list)

    @property
    def lean(self) -> str:
        if self.bull - self.bear >= 0.08:
            return "BULLISH"
        if self.bear - self.bull >= 0.08:
            return "BEARISH"
        return "NEUTRAL"


@dataclass
class DecisionScore:
    ta: ModuleScore
    patterns: ModuleScore
    sentiment: ModuleScore
    weighted_bull: float
    weighted_bear: float
    net: float
    alignment_votes: int
    confidence: float
    signal: TradeSignal
    reasoning: str
    ta_summary: str
    pattern_summary: str
    sentiment_summary: str
    risk_assessment: str


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def score_technical(indicators: Dict[str, Any]) -> ModuleScore:
    """Rule-based TA score from indicator thresholds."""
    score = ModuleScore()
    close = float(indicators.get("close") or 0)
    ema20 = float(indicators.get("ema_20") or close)
    ema50 = float(indicators.get("ema_50") or close)
    rsi = float(indicators.get("rsi") or 50)
    macd_hist = float(indicators.get("macd_hist") or 0)
    adx = float(indicators.get("adx") or 0)
    vwap = float(indicators.get("vwap") or close)
    mfi = float(indicators.get("mfi") or 50)
    trend = str(indicators.get("trend") or "NEUTRAL").upper()

    if trend == "BULLISH":
        score.bull += 0.28
        score.reasons.append("EMA stack bullish (price > EMA20 > EMA50)")
    elif trend == "BEARISH":
        score.bear += 0.28
        score.reasons.append("EMA stack bearish (price < EMA20 < EMA50)")
    else:
        score.reasons.append("EMA trend neutral")

    if macd_hist > 0:
        score.bull += 0.18
        score.reasons.append(f"MACD histogram positive ({macd_hist:.6f})")
    elif macd_hist < 0:
        score.bear += 0.18
        score.reasons.append(f"MACD histogram negative ({macd_hist:.6f})")

    if rsi < 32:
        score.bull += 0.16
        score.reasons.append(f"RSI oversold ({rsi:.1f})")
    elif rsi < 45:
        score.bull += 0.08
        score.reasons.append(f"RSI leaning low ({rsi:.1f})")
    elif rsi > 68:
        score.bear += 0.16
        score.reasons.append(f"RSI overbought ({rsi:.1f})")
    elif rsi > 55:
        score.bear += 0.08
        score.reasons.append(f"RSI leaning high ({rsi:.1f})")

    if close > vwap:
        score.bull += 0.10
        score.reasons.append("Price above VWAP")
    elif close < vwap:
        score.bear += 0.10
        score.reasons.append("Price below VWAP")

    if mfi < 25:
        score.bull += 0.08
        score.reasons.append(f"MFI oversold ({mfi:.1f})")
    elif mfi > 75:
        score.bear += 0.08
        score.reasons.append(f"MFI overbought ({mfi:.1f})")

    if adx >= 25:
        boost = 1.12
        score.bull = _clamp(score.bull * boost)
        score.bear = _clamp(score.bear * boost)
        score.reasons.append(f"ADX trend strength ({adx:.1f})")

    if close > ema20 and ema20 > ema50 and macd_hist > 0:
        score.bull += 0.06
    elif close < ema20 and ema20 < ema50 and macd_hist < 0:
        score.bear += 0.06

    score.bull = _clamp(score.bull)
    score.bear = _clamp(score.bear)
    return score


def score_patterns(chart_patterns: Dict[str, Any]) -> ModuleScore:
    """Score from ChartPatternsAgent detections only — no invented patterns."""
    score = ModuleScore()
    patterns = chart_patterns.get("patterns") or []
    structure = str(chart_patterns.get("structure") or "NEUTRAL").upper()

    for p in patterns:
        bias = str(p.get("bias") or "NEUTRAL").upper()
        conf = float(p.get("confidence") or 0)
        status = str(p.get("status") or "forming").lower()
        weight = conf * (0.28 if status == "confirmed" else 0.20)
        name = p.get("name") or p.get("key") or "pattern"

        if bias == "BULLISH":
            score.bull += weight
            score.reasons.append(f"{name} bullish ({status}, {conf:.0%})")
        elif bias == "BEARISH":
            score.bear += weight
            score.reasons.append(f"{name} bearish ({status}, {conf:.0%})")

    if structure == "BULLISH":
        score.bull += 0.12
        score.reasons.append("Pattern structure bullish")
    elif structure == "BEARISH":
        score.bear += 0.12
        score.reasons.append("Pattern structure bearish")
    elif structure == "MIXED":
        score.reasons.append("Mixed pattern structure — reduced pattern edge")
        score.bull *= 0.85
        score.bear *= 0.85
    elif not patterns:
        score.reasons.append("No classical patterns matched")

    score.bull = _clamp(score.bull)
    score.bear = _clamp(score.bear)
    return score


def score_sentiment(sentiment: Dict[str, Any]) -> ModuleScore:
    """Map news sentiment score (0–1) to directional bull/bear components."""
    score = ModuleScore()
    raw = float(sentiment.get("sentiment_score") or 0.5)
    label = str(sentiment.get("sentiment_label") or "NEUTRAL").upper()

    if raw >= 0.5:
        score.bull = _clamp((raw - 0.5) * 2.0)
    else:
        score.bear = _clamp((0.5 - raw) * 2.0)

    if "VERY_BULLISH" in label:
        score.bull = _clamp(score.bull + 0.12)
    elif "VERY_BEARISH" in label:
        score.bear = _clamp(score.bear + 0.12)
    elif "BULLISH" in label:
        score.bull = _clamp(score.bull + 0.06)
    elif "BEARISH" in label:
        score.bear = _clamp(score.bear + 0.06)

    score.reasons.append(f"News {label} (score={raw:.2f})")
    return score


def _count_alignment(ta: ModuleScore, patterns: ModuleScore, sentiment: ModuleScore) -> Tuple[int, int]:
    """Return (bullish_votes, bearish_votes) among the three modules."""
    bulls = sum(1 for m in (ta, patterns, sentiment) if m.lean == "BULLISH")
    bears = sum(1 for m in (ta, patterns, sentiment) if m.lean == "BEARISH")
    return bulls, bears


def compute_decision_score(
    indicators: Dict[str, Any],
    chart_patterns: Dict[str, Any],
    sentiment: Dict[str, Any],
    portfolio: Dict[str, Any],
) -> DecisionScore:
    """Combine module scores into signal + confidence."""
    ta = score_technical(indicators)
    patterns = score_patterns(chart_patterns)
    sentiment_mod = score_sentiment(sentiment)

    w_ta = settings.DECISION_WEIGHT_TA
    w_pat = settings.DECISION_WEIGHT_PATTERNS
    w_sent = settings.DECISION_WEIGHT_SENTIMENT

    weighted_bull = _clamp(ta.bull * w_ta + patterns.bull * w_pat + sentiment_mod.bull * w_sent)
    weighted_bear = _clamp(ta.bear * w_ta + patterns.bear * w_pat + sentiment_mod.bear * w_sent)
    net = weighted_bull - weighted_bear

    bull_votes, bear_votes = _count_alignment(ta, patterns, sentiment_mod)
    dominant_votes = bull_votes if net >= 0 else bear_votes
    opposing = bear_votes if net >= 0 else bull_votes

    side_strength = weighted_bull if net >= 0 else weighted_bear
    edge = abs(net)

    alignment_bonus = 0.0
    if dominant_votes >= 3:
        alignment_bonus = 0.14
    elif dominant_votes >= 2:
        alignment_bonus = 0.08

    conflict_penalty = 0.0
    if opposing >= 2 and edge < 0.15:
        conflict_penalty = 0.12

    drawdown = float(portfolio.get("drawdown") or 0)
    dd_penalty = 0.0
    if drawdown >= settings.MAX_DRAWDOWN_KILL:
        dd_penalty = 1.0
    elif drawdown >= settings.MAX_DRAWDOWN_KILL * 0.6:
        dd_penalty = 0.08 * (drawdown / settings.MAX_DRAWDOWN_KILL)

    confidence = _clamp(side_strength * 0.55 + edge * 0.85 + alignment_bonus - conflict_penalty - dd_penalty)

    min_edge = settings.DECISION_MIN_NET_EDGE
    min_votes = settings.DECISION_MIN_MODULE_AGREEMENT
    signal = TradeSignal.HOLD

    if drawdown >= settings.MAX_DRAWDOWN_KILL:
        signal = TradeSignal.HOLD
        hold_reason = f"Drawdown kill switch ({drawdown*100:.1f}% >= {settings.MAX_DRAWDOWN_KILL*100:.0f}%)"
    elif edge < min_edge:
        signal = TradeSignal.HOLD
        hold_reason = f"Net edge too weak ({edge:.2f} < {min_edge:.2f})"
    elif net >= min_edge and bull_votes >= min_votes:
        signal = TradeSignal.BUY
        hold_reason = ""
    elif net <= -min_edge and bear_votes >= min_votes:
        signal = TradeSignal.SELL
        hold_reason = ""
    else:
        signal = TradeSignal.HOLD
        hold_reason = f"Insufficient module agreement (need {min_votes}+ modules aligned)"

    ta_summary = (
        f"TA {ta.lean}: bull={ta.bull:.2f} bear={ta.bear:.2f} — "
        + ("; ".join(ta.reasons[:3]) if ta.reasons else "no signals")
    )
    pattern_summary = (
        f"Patterns {patterns.lean}: bull={patterns.bull:.2f} bear={patterns.bear:.2f} — "
        + ("; ".join(patterns.reasons[:3]) if patterns.reasons else chart_patterns.get("summary", "none"))
    )
    sentiment_summary = (
        f"Sentiment {sentiment_mod.lean}: bull={sentiment_mod.bull:.2f} bear={sentiment_mod.bear:.2f} — "
        + sentiment.get("news_summary", "N/A")[:120]
    )

    if signal == TradeSignal.HOLD:
        risk = hold_reason or "No actionable edge"
    else:
        risk = (
            f"Confidence {confidence:.0%} | votes bull={bull_votes} bear={bear_votes} | "
            f"net={net:+.2f} | size capped at {settings.MAX_RISK_PER_TRADE*100:.1f}% equity"
        )

    reasoning = (
        f"Deterministic score — TA {ta.lean}, Patterns {patterns.lean}, Sentiment {sentiment_mod.lean}. "
        f"Weighted bull={weighted_bull:.2f}, bear={weighted_bear:.2f}, net={net:+.2f}, "
        f"confidence={confidence:.0%}. "
        f"{' → ' + signal.value if signal != TradeSignal.HOLD else ' → HOLD: ' + (hold_reason or 'no edge')}."
    )

    return DecisionScore(
        ta=ta,
        patterns=patterns,
        sentiment=sentiment_mod,
        weighted_bull=weighted_bull,
        weighted_bear=weighted_bear,
        net=net,
        alignment_votes=dominant_votes,
        confidence=confidence,
        signal=signal,
        reasoning=reasoning,
        ta_summary=ta_summary,
        pattern_summary=pattern_summary,
        sentiment_summary=sentiment_summary,
        risk_assessment=risk,
    )


def build_trade_decision(
    symbol: str,
    indicators: Dict[str, Any],
    portfolio: Dict[str, Any],
    result: DecisionScore,
) -> TradeDecision:
    price = float(indicators.get("close") or 1)
    equity = float(portfolio.get("equity") or settings.INITIAL_CAPITAL)

    if result.signal == TradeSignal.HOLD:
        size_pct = 0.0
    else:
        conf_ratio = result.confidence / max(settings.MIN_TRADE_CONFIDENCE, 0.01)
        size_pct = min(settings.MAX_RISK_PER_TRADE, settings.MAX_RISK_PER_TRADE * conf_ratio * 0.85)

    size_usdt = equity * size_pct
    sl_pct = settings.DEFAULT_STOP_LOSS_PCT
    tp_pct = settings.DEFAULT_TAKE_PROFIT_PCT

    atr = float(indicators.get("atr") or 0)
    if atr > 0 and price > 0:
        atr_sl = (atr * 1.5) / price
        sl_pct = _clamp(max(sl_pct, min(atr_sl, 0.05)), 0.015, 0.05)
        tp_pct = sl_pct * 3.0

    if result.signal == TradeSignal.BUY:
        sl = price * (1 - sl_pct)
        tp = price * (1 + tp_pct)
    elif result.signal == TradeSignal.SELL:
        sl = price * (1 + sl_pct)
        tp = price * (1 - tp_pct)
    else:
        sl = tp = None

    return TradeDecision(
        id=str(uuid.uuid4())[:8],
        timestamp=datetime.utcnow().isoformat(),
        symbol=symbol,
        signal=result.signal,
        confidence=result.confidence,
        size_usdt=size_usdt,
        entry_price=price,
        stop_loss=sl,
        take_profit=tp,
        reasoning=result.reasoning,
        agent_contributions={
            "technical": result.ta_summary,
            "patterns": result.pattern_summary,
            "sentiment": result.sentiment_summary,
            "risk": result.risk_assessment,
            "scores": (
                f"bull={result.weighted_bull:.2f} bear={result.weighted_bear:.2f} "
                f"net={result.net:+.2f} votes={result.alignment_votes}"
            ),
        },
    )


def run_decision_agent(
    symbol: str,
    indicators: Dict,
    sentiment: Optional[Dict],
    portfolio_dict: Dict,
    chart_patterns: Optional[Dict] = None,
    timeframe: str = "4h",
) -> TradeDecision:
    if not sentiment:
        raise ValueError("News sentiment required for decision")
    if not chart_patterns:
        raise ValueError("Chart pattern analysis required for decision")
    if not indicators:
        raise ValueError("Indicators required for decision")

    trading_state.add_log(
        "DecisionAgent",
        f"Scoring swing setup for {symbol} [{timeframe}] (deterministic rules)...",
    )

    result = compute_decision_score(indicators, chart_patterns, sentiment, portfolio_dict)
    decision = build_trade_decision(symbol, indicators, portfolio_dict, result)

    db.save_decision(decision)
    trading_state.add_decision(decision)

    emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(decision.signal.value, "⚪")
    trading_state.add_log(
        "DecisionAgent",
        f"{emoji} {decision.signal.value} | Confidence={decision.confidence:.0%} | "
        f"net={result.net:+.2f} | bull={result.weighted_bull:.2f} bear={result.weighted_bear:.2f} | "
        f"Size=${decision.size_usdt:.0f}",
        data={
            "signal": decision.signal.value,
            "confidence": decision.confidence,
            "net": result.net,
            "weighted_bull": result.weighted_bull,
            "weighted_bear": result.weighted_bear,
            "ta_lean": result.ta.lean,
            "pattern_lean": result.patterns.lean,
            "sentiment_lean": result.sentiment.lean,
            "alignment_votes": result.alignment_votes,
            "reasoning": decision.reasoning,
        },
        level="success",
    )

    return decision
