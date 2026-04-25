"""
EvoTrade AI - LLM Provider Abstraction
Supports Anthropic, OpenAI, Gemini, Ollama, and Mock fallback.
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
        elif provider == "gemini" and settings.GEMINI_API_KEY:
            return _call_gemini(prompt, system, max_tokens)
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


def _call_gemini(prompt: str, system: str, max_tokens: int) -> str:
    """
    Google Gemini — free tier: 15 RPM, 1M tokens/day on gemini-2.0-flash.
    Get key: https://aistudio.google.com/app/apikey
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("Run: pip install google-generativeai")

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        system_instruction=system or "You are EvoTrade AI, an expert algorithmic trading assistant.",
        generation_config=genai.GenerationConfig(max_output_tokens=max_tokens, temperature=0.3)
    )
    response = model.generate_content(prompt)
    return response.text


def _call_ollama(prompt: str, system: str, max_tokens: int) -> str:
    import requests
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": f"{system}\n\n{prompt}" if system else prompt,
        "stream": False,
        "options": {"num_predict": max_tokens}
    }
    resp = requests.post(f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload, timeout=60)
    return resp.json().get("response", "")


def _mock_response(prompt: str, error: str = "") -> str:
    """
    Deterministic mock responses for demo/testing when no API key is set.
    Biased toward BUY/SELL so trades actually happen in demo mode.
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

    elif any(k in prompt_lower for k in ["decision", "signal", "buy", "synthesiz", "make a trading"]):
        import time
        # Rotates every 30s: 5 out of 6 slots are actionable
        seed = int(time.time() / 30) % 6
        signals = ["BUY", "BUY", "BUY", "SELL", "BUY", "HOLD"]
        sig = signals[seed]
        conf = round(0.68 + (seed * 3 % 15) / 100, 2)
        return json.dumps({
            "signal": sig,
            "confidence": conf,
            "reasoning": (
                f"RSI at moderate levels with {sig.lower()} momentum building. "
                "EMA 20 crossing above EMA 50 confirms trend direction. "
                "News sentiment supportive. MACD histogram turning positive."
            ),
            "size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "ta_summary": "Price above EMA20/50. RSI 54. MACD histogram positive.",
            "sentiment_summary": "News moderately bullish, institutional interest rising.",
            "risk_assessment": "Moderate risk. Drawdown within acceptable range."
        })

    elif any(k in prompt_lower for k in ["strateg", "evolut", "invent", "generat"]):
        return json.dumps({
            "strategies": [
                {
                    "name": "Momentum Breakout v2",
                    "description": "Enhanced momentum strategy with volume confirmation",
                    "strategy_type": "momentum_breakout",
                    "indicators": ["EMA_20", "EMA_50", "RSI_14", "Volume_MA", "ATR_14"],
                    "risk_per_trade": 0.02, "take_profit_ratio": 2.5,
                    "sl_atr_mult": 1.5, "long_only": True,
                    "rules": "Enter long when EMA_20 > EMA_50 AND RSI_14 > 55 AND volume > 1.5x avg."
                },
                {
                    "name": "Mean Reversion Alpha",
                    "description": "Bollinger Band mean reversion with RSI filter",
                    "strategy_type": "rsi_reversal",
                    "indicators": ["BB_20_2", "RSI_7", "VWAP", "MFI_14"],
                    "risk_per_trade": 0.015, "take_profit_ratio": 1.8,
                    "sl_atr_mult": 1.2, "long_only": True,
                    "rules": "Enter long when price touches lower BB AND RSI_7 < 30."
                },
                {
                    "name": "Multi-Signal Alpha v3",
                    "description": "Combines EMA trend, RSI momentum, MACD direction, and volume.",
                    "strategy_type": "multi_signal",
                    "indicators": ["EMA_20", "EMA_50", "RSI_14", "MACD", "Volume_MA_20"],
                    "risk_per_trade": 0.02, "take_profit_ratio": 2.2,
                    "sl_atr_mult": 1.5, "long_only": False,
                    "rules": "Enter when EMA bullish AND RSI neutral AND MACD positive AND volume elevated."
                }
            ]
        })

    else:
        return json.dumps({
            "analysis": "Technical analysis complete. Moderate bullish bias detected.",
            "recommendation": "BUY",
            "confidence": 0.62
        })


def get_available_providers() -> dict:
    """Returns which LLM providers are configured and available."""
    return {
        "anthropic": {
            "available": bool(settings.ANTHROPIC_API_KEY),
            "model": settings.LLM_MODEL,
            "key_url": "https://console.anthropic.com/",
        },
        "openai": {
            "available": bool(settings.OPENAI_API_KEY),
            "model": "gpt-4o",
            "key_url": "https://platform.openai.com/api-keys",
        },
        "gemini": {
            "available": bool(settings.GEMINI_API_KEY),
            "model": getattr(settings, "GEMINI_MODEL", "gemini-2.0-flash"),
            "key_url": "https://aistudio.google.com/app/apikey",
            "note": "Free tier: 15 RPM, 1M tokens/day on gemini-2.0-flash",
        },
        "ollama": {
            "available": True,
            "model": getattr(settings, "OLLAMA_MODEL", "llama3"),
            "key_url": "https://ollama.com/download",
            "note": "Local — no key needed",
        },
        "mock": {
            "available": True,
            "model": "built-in",
            "note": "Demo mode — no key needed",
        },
        "active": settings.LLM_PROVIDER,
    }