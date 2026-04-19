"""
EvoTrade AI - LLM Provider Abstraction
Supports Anthropic, OpenAI, and Ollama (mock fallback).
"""
from __future__ import annotations
import json
import re
from typing import Optional, Dict, Any
from config import settings


def get_llm_response(prompt: str, system: str = "", max_tokens: int = 1500) -> str:
    """
    Universal LLM call. Returns plain text response.
    Falls back to mock response if no API key configured.
    """
    provider = settings.LLM_PROVIDER.lower()

    try:
        if provider == "anthropic" and settings.ANTHROPIC_API_KEY:
            return _call_anthropic(prompt, system, max_tokens)
        elif provider == "openai" and settings.OPENAI_API_KEY:
            return _call_openai(prompt, system, max_tokens)
        elif provider == "ollama":
            return _call_ollama(prompt, system, max_tokens)
        else:
            return _mock_response(prompt)
    except Exception as e:
        return _mock_response(prompt, error=str(e))


def _call_anthropic(prompt: str, system: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=max_tokens,
        system=system or "You are EvoTrade AI, an expert algorithmic trading assistant.",
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text


def _call_openai(prompt: str, system: str, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system or "You are EvoTrade AI."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content


def _call_ollama(prompt: str, system: str, max_tokens: int) -> str:
    import requests
    payload = {
        "model": "llama3",
        "prompt": f"{system}\n\n{prompt}" if system else prompt,
        "stream": False
    }
    resp = requests.post(f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload, timeout=60)
    return resp.json().get("response", "")


def _mock_response(prompt: str, error: str = "") -> str:
    """
    Deterministic mock responses for demo/testing when no API key is set.
    """
    prompt_lower = prompt.lower()

    if "news" in prompt_lower or "sentiment" in prompt_lower:
        return json.dumps({
            "sentiment_score": 0.65,
            "sentiment_label": "BULLISH",
            "key_themes": ["institutional adoption", "ETF inflows", "Fed rate outlook"],
            "news_summary": "Market sentiment is cautiously bullish with strong institutional interest.",
            "risk_factors": ["regulatory uncertainty", "macro headwinds"]
        })
    elif "decision" in prompt_lower or "signal" in prompt_lower or "buy" in prompt_lower:
        return json.dumps({
            "signal": "BUY",
            "confidence": 0.72,
            "reasoning": "RSI showing oversold conditions with bullish divergence. MACD crossover imminent. News sentiment positive. Risk/reward favorable at current levels.",
            "size_pct": 0.03,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05
        })
    elif "strateg" in prompt_lower or "evolut" in prompt_lower:
        return json.dumps({
            "strategies": [
                {
                    "name": "Momentum Breakout v2",
                    "description": "Enhanced momentum strategy with volume confirmation",
                    "indicators": ["EMA_20", "EMA_50", "RSI_14", "Volume_MA", "ATR_14"],
                    "rules": "Enter long when EMA_20 > EMA_50 AND RSI_14 > 55 AND volume > 1.5x avg AND price breaks above 20-period high. SL at 1.5x ATR. TP at 3x ATR."
                },
                {
                    "name": "Mean Reversion Alpha",
                    "description": "Bollinger Band mean reversion with RSI filter",
                    "indicators": ["BB_20_2", "RSI_7", "VWAP", "MFI_14"],
                    "rules": "Enter long when price touches lower BB AND RSI_7 < 30 AND price below VWAP AND MFI < 25. TP at middle BB. SL at 2% below entry."
                }
            ]
        })
    else:
        return json.dumps({
            "analysis": "Technical analysis complete. Market showing mixed signals.",
            "recommendation": "HOLD",
            "confidence": 0.55
        })
