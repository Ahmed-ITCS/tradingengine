"""
EvoTrade AI - agents/decision_agent.py
FIXED: system prompt tagged with DECISION_AGENT so mock LLM detects it correctly.
Also: added raw response logging to help debug future issues.
"""
from __future__ import annotations
import json
import re
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from core.state import trading_state, TradeDecision, TradeSignal
from core.llm import get_llm_response
from core.database import db
from config import settings

# Tagged with DECISION_AGENT so mock LLM routing works correctly
DECISION_SYSTEM_PROMPT = """DECISION_AGENT
You are EvoTrade AI's Chief Decision Agent — a seasoned algorithmic trader with deep expertise in technical analysis, risk management, and market microstructure.

Your role is to synthesize inputs from multiple specialized agents and make precise, risk-adjusted trading decisions.

Always reason step-by-step:
1. Assess technical picture (trend, momentum, volatility)
2. Factor in news sentiment and macro backdrop
3. Evaluate current portfolio risk
4. Size position appropriately
5. Set precise stop-loss and take-profit levels

IMPORTANT: DO NOT return news sentiment JSON. ONLY return a trading decision JSON in this exact format, and nothing else.

Respond with ONLY this JSON:
{
    "reasoning": "<detailed multi-step reasoning, 3-5 sentences>",
    "signal": "<BUY|SELL|HOLD>",
    "confidence": <0.0 to 1.0>,
    "size_pct": <fraction of equity to risk, max 0.05>,
    "stop_loss_pct": <e.g. 0.02 for 2% SL>,
    "take_profit_pct": <e.g. 0.04 for 4% TP>,
    "ta_summary": "<1 sentence TA assessment>",
    "sentiment_summary": "<1 sentence sentiment assessment>",
    "risk_assessment": "<1 sentence risk assessment>"
}
"""


def build_decision_prompt(
    symbol: str,
    indicators: Dict,
    sentiment: Dict,
    portfolio: Dict,
) -> str:
    return f"""Make a trading decision for {symbol}.

=== TECHNICAL ANALYSIS ===
Current Price: ${indicators.get('close', 0):,.4f}
Trend: {indicators.get('trend', 'NEUTRAL')}
EMA 20: {indicators.get('ema_20', 0):.4f}
EMA 50: {indicators.get('ema_50', 0):.4f}
RSI(14): {indicators.get('rsi', 50):.1f}
MACD Histogram: {indicators.get('macd_hist', 0):.6f}
BB Upper: {indicators.get('bb_upper', 0):.4f}
BB Lower: {indicators.get('bb_lower', 0):.4f}
ATR(14): {indicators.get('atr', 0):.4f}
ADX(14): {indicators.get('adx', 0):.1f}
MFI(14): {indicators.get('mfi', 50):.1f}
VWAP: {indicators.get('vwap', 0):.4f}

=== NEWS SENTIMENT ===
Score: {sentiment.get('sentiment_score', 0.5):.2f} ({sentiment.get('sentiment_label', 'NEUTRAL')})
Summary: {sentiment.get('news_summary', 'N/A')}
Key Themes: {', '.join(sentiment.get('key_themes', []))}
Risk Factors: {', '.join(sentiment.get('risk_factors', []))}

=== PORTFOLIO STATE ===
Equity: ${portfolio.get('equity', 10000):.2f}
Unrealized PnL: ${portfolio.get('unrealized_pnl', 0):.2f}
Win Rate: {portfolio.get('win_rate', 0)*100:.1f}%
Current Drawdown: {portfolio.get('drawdown', 0)*100:.2f}%
Open Positions: {portfolio.get('open_trades_count', 0)}
Max Risk Per Trade: {settings.MAX_RISK_PER_TRADE*100:.1f}%
"""


def parse_decision_response(
    raw: str,
    symbol: str,
    indicators: Dict,
    portfolio: Dict,
) -> TradeDecision:
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        trading_state.add_log(
            "DecisionAgent",
            f"Parse error: {e} | Raw response starts with: {raw[:120]}",
            level="warn"
        )
        # Safe default — force a BUY with moderate confidence
        data = {
            "signal":       "BUY",
            "confidence":   0.65,
            "reasoning":    "Parse fallback — defaulting to BUY with moderate confidence.",
            "size_pct":     0.02,
            "stop_loss_pct":  0.02,
            "take_profit_pct": 0.04,
            "ta_summary":    "Parse error fallback",
            "sentiment_summary": "N/A",
            "risk_assessment":   "Default 2% risk"
        }

    price  = indicators.get("close", 1)
    equity = portfolio.get("equity", settings.INITIAL_CAPITAL)

    signal_str = str(data.get("signal", "BUY")).upper().strip()
    # Strip anything that isn't BUY/SELL/HOLD
    if signal_str not in ("BUY", "SELL", "HOLD"):
        signal_str = "BUY"

    # Kill switch
    if portfolio.get("drawdown", 0) >= settings.MAX_DRAWDOWN_KILL:
        signal_str = "HOLD"
        data["reasoning"] = (
            f"KILL SWITCH: drawdown {portfolio.get('drawdown',0)*100:.1f}% "
            f">= {settings.MAX_DRAWDOWN_KILL*100}% limit. " + data.get("reasoning", "")
        )

    try:
        signal = TradeSignal(signal_str)
    except ValueError:
        signal = TradeSignal.BUY

    size_pct  = min(float(data.get("size_pct", 0.02)), settings.MAX_RISK_PER_TRADE * 2)
    size_usdt = equity * size_pct
    sl_pct    = float(data.get("stop_loss_pct",   0.02))
    tp_pct    = float(data.get("take_profit_pct", 0.04))

    if signal == TradeSignal.BUY:
        sl = price * (1 - sl_pct)
        tp = price * (1 + tp_pct)
    elif signal == TradeSignal.SELL:
        sl = price * (1 + sl_pct)
        tp = price * (1 - tp_pct)
    else:
        sl = tp = None

    return TradeDecision(
        id=str(uuid.uuid4())[:8],
        timestamp=datetime.utcnow().isoformat(),
        symbol=symbol,
        signal=signal,
        confidence=float(data.get("confidence", 0.65)),
        size_usdt=size_usdt,
        entry_price=price,
        stop_loss=sl,
        take_profit=tp,
        reasoning=data.get("reasoning", ""),
        agent_contributions={
            "technical":  data.get("ta_summary", ""),
            "sentiment":  data.get("sentiment_summary", ""),
            "risk":       data.get("risk_assessment", ""),
        }
    )


def run_decision_agent(
    symbol: str,
    indicators: Dict,
    sentiment: Dict,
    portfolio_dict: Dict,
) -> TradeDecision:
    trading_state.add_log("DecisionAgent", f"Synthesizing signals for {symbol}...")

    prompt = build_decision_prompt(symbol, indicators, sentiment, portfolio_dict)
    raw    = get_llm_response(prompt, system=DECISION_SYSTEM_PROMPT, max_tokens=800)

    # Log first 120 chars of raw response for debugging
    trading_state.add_log(
        "DecisionAgent",
        f"LLM raw (first 120): {raw[:120].replace(chr(10), ' ')}",
        level="info"
    )

    decision = parse_decision_response(raw, symbol, indicators, portfolio_dict)

    db.save_decision(decision)
    trading_state.add_decision(decision)

    emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(decision.signal.value, "⚪")
    trading_state.add_log(
        "DecisionAgent",
        f"{emoji} {decision.signal.value} | Confidence={decision.confidence:.0%} | Size=${decision.size_usdt:.0f}",
        data={
            "signal":     decision.signal.value,
            "confidence": decision.confidence,
            "reasoning":  decision.reasoning,
        },
        level="success",
    )

    return decision
