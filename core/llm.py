"""
EvoTrade AI - core/llm.py
LLM provider routing — requires a real provider (no mock/fake responses).
"""
from __future__ import annotations
from typing import Optional, Dict, Any
from config import settings


def get_llm_response(prompt: str, system: str = "", max_tokens: int = 1500) -> str:
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
        raise RuntimeError(
            f"No LLM configured for provider '{provider}'. "
            "Set a valid API key (GEMINI_API_KEY, OPENAI_API_KEY, etc.) or use LLM_PROVIDER=ollama."
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"LLM call failed ({provider}): {e}") from e


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
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("Run: pip install google-generativeai")
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=getattr(settings, "GEMINI_MODEL", "gemini-2.0-flash"),
        system_instruction=system or "You are EvoTrade AI.",
        generation_config=genai.GenerationConfig(
            max_output_tokens=max_tokens, temperature=0.3
        )
    )
    return model.generate_content(prompt).text


def _call_ollama(prompt: str, system: str, max_tokens: int) -> str:
    import requests
    payload = {
        "model": getattr(settings, "OLLAMA_MODEL", "llama3"),
        "prompt": f"{system}\n\n{prompt}" if system else prompt,
        "stream": False,
        "options": {"num_predict": max_tokens}
    }
    resp = requests.post(
        f"{getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')}/api/generate",
        json=payload, timeout=120
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def get_available_providers() -> dict:
    return {
        "anthropic": {
            "available": bool(settings.ANTHROPIC_API_KEY),
            "model":     settings.LLM_MODEL,
            "key_url":   "https://console.anthropic.com/",
        },
        "openai": {
            "available": bool(settings.OPENAI_API_KEY),
            "model":     "gpt-4o",
            "key_url":   "https://platform.openai.com/api-keys",
        },
        "gemini": {
            "available": bool(settings.GEMINI_API_KEY),
            "model":     getattr(settings, "GEMINI_MODEL", "gemini-2.0-flash"),
            "key_url":   "https://aistudio.google.com/app/apikey",
            "note":      "Free tier: 15 RPM, 1M tokens/day",
        },
        "ollama": {
            "available": True,
            "model":     getattr(settings, "OLLAMA_MODEL", "llama3"),
            "key_url":   "https://ollama.com/download",
            "note":      "Local — no key needed",
        },
        "active": settings.LLM_PROVIDER,
    }
