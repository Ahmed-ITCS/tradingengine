"""
EvoTrade AI - LLM Provider Abstraction
Supports: Anthropic Claude, OpenAI GPT-4o, Google Gemini, Ollama (local), Mock (demo).

HOW TO GET EACH API KEY:
  Anthropic  → https://console.anthropic.com/  (Settings → API Keys)
  OpenAI     → https://platform.openai.com/api-keys
  Gemini     → https://aistudio.google.com/app/apikey  (free tier available)
  Ollama     → No key needed, runs locally: https://ollama.com/download
"""
from __future__ import annotations
import json
import re
from typing import Optional
from config import settings


def get_llm_response(prompt: str, system: str = "", max_tokens: int = 1500) -> str:
    """
    Universal LLM call dispatcher.
    Auto-falls-back to mock if the selected provider has no API key.
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
            # No API key configured → demo mock
            return _mock_response(prompt)
    except Exception as e:
        # Graceful fallback so engine never crashes on LLM errors
        return _mock_response(prompt, error=str(e))


# ── Provider implementations ──────────────────────────────────────────────────

def _call_anthropic(prompt: str, system: str, max_tokens: int) -> str:
    """Anthropic Claude (claude-sonnet-4-20250514 by default)."""
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
    """OpenAI GPT-4o."""
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
    Google Gemini via google-generativeai SDK.
    Models: gemini-2.0-flash (fast/cheap), gemini-1.5-pro (capable).
    Free tier: 15 RPM / 1M TPD on gemini-2.0-flash.
    Get key: https://aistudio.google.com/app/apikey
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError(
            "google-generativeai not installed. Run: pip install google-generativeai"
        )

    genai.configure(api_key=settings.GEMINI_API_KEY)

    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        system_instruction=system or "You are EvoTrade AI, an expert algorithmic trading assistant.",
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.3,
        )
    )
    response = model.generate_content(prompt)
    return response.text


def _call_ollama(prompt: str, system: str, max_tokens: int) -> str:
    """
    Ollama local LLM — no API key needed.
    Install: https://ollama.com/download
    Then pull a model: ollama pull llama3
    """
    import requests
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": f"{system}\n\n{prompt}" if system else prompt,
        "stream": False,
        "options": {"num_predict": max_tokens}
    }
    resp = requests.post(
        f"{settings.OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


# ── Mock / Demo fallback ──────────────────────────────────────────────────────

def _mock_response(prompt: str, error: str = "") -> str:
    """
    Deterministic mock responses used when no API key is configured.
    The engine runs fully in demo mode with these — no API key needed.
    """
    p = prompt.lower()

    if "news" in p or "sentiment" in p:
        return json.dumps({
            "sentiment_score": 0.65,
            "sentiment_label": "BULLISH",
            "key_themes": ["institutional adoption", "ETF inflows", "Fed rate outlook"],
            "news_summary": "Market sentiment is cautiously bullish with strong institutional inflows and positive macro signals.",
            "risk_factors": ["regulatory uncertainty", "macro headwinds", "liquidity risks"]
        })

    elif "decision" in p or "signal" in p or "buy" in p or "synthesiz" in p:
        import random, hashlib
        # Make it slightly random so not every decision is BUY
        seed = int(hashlib.md5(prompt[:50].encode()).hexdigest(), 16) % 100
        signals = ["BUY", "BUY", "BUY", "HOLD", "HOLD", "SELL"]
        sig = signals[seed % len(signals)]
        return json.dumps({
            "signal": sig,
            "confidence": 0.62 + (seed % 20) / 100,
            "reasoning": (
                f"RSI at moderate levels with {sig.lower()} pressure. "
                "MACD histogram turning positive with EMA alignment. "
                "News sentiment supportive. Risk/reward ratio favorable at current levels. "
                "Position sizing set conservatively given current drawdown."
            ),
            "size_pct": 0.02,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "ta_summary": "Price above EMA20/50. RSI neutral. MACD trending up.",
            "sentiment_summary": "News moderately bullish with institutional interest.",
            "risk_assessment": "Moderate risk. Drawdown within acceptable range."
        })

    elif "strateg" in p or "evolut" in p or "invent" in p or "generat" in p:
        return json.dumps({
            "strategies": [
                {
                    "name": "Momentum Breakout Pro",
                    "description": "Combines EMA trend filter with volume-confirmed breakouts above 20-period highs. Uses ATR-based stops for adaptive risk management.",
                    "strategy_type": "momentum_breakout",
                    "indicators": ["EMA_20", "EMA_50", "ATR_14", "Volume_MA_20", "ADX_14"],
                    "risk_per_trade": 0.02,
                    "take_profit_ratio": 2.5,
                    "sl_atr_mult": 1.5,
                    "timeframe": "1h",
                    "long_only": True
                },
                {
                    "name": "RSI Mean Reversion Elite",
                    "description": "Enters at RSI extremes with VWAP confirmation. Targets return to mean with tight risk controls.",
                    "strategy_type": "rsi_reversal",
                    "indicators": ["RSI_7", "RSI_14", "VWAP", "BB_20_2", "MFI_14"],
                    "risk_per_trade": 0.015,
                    "take_profit_ratio": 1.8,
                    "sl_atr_mult": 1.2,
                    "timeframe": "1h",
                    "long_only": True
                },
                {
                    "name": "Multi-Signal Alpha v3",
                    "description": "Synthesizes EMA trend, RSI momentum, MACD direction, and volume filters for high-confidence entries only.",
                    "strategy_type": "multi_signal",
                    "indicators": ["EMA_20", "EMA_50", "RSI_14", "MACD", "Volume_MA_20", "ADX_14"],
                    "risk_per_trade": 0.02,
                    "take_profit_ratio": 2.2,
                    "sl_atr_mult": 1.5,
                    "timeframe": "1h",
                    "long_only": False
                }
            ]
        })

    else:
        return json.dumps({
            "analysis": "Technical analysis complete. Market showing mixed signals with moderate volatility.",
            "recommendation": "HOLD",
            "confidence": 0.55
        })


def get_available_providers() -> dict:
    """Return which providers are configured and ready."""
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
            "model": settings.GEMINI_MODEL,
            "key_url": "https://aistudio.google.com/app/apikey",
            "note": "Free tier: 15 RPM, 1M tokens/day on gemini-2.0-flash",
        },
        "ollama": {
            "available": True,  # always try; fails gracefully if not running
            "model": settings.OLLAMA_MODEL,
            "key_url": "https://ollama.com/download",
            "note": "Run locally: ollama pull llama3",
        },
        "mock": {
            "available": True,
            "model": "built-in",
            "note": "Demo mode — no API key needed",
        },
        "active": settings.LLM_PROVIDER,
    }
