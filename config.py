"""
EvoTrade AI - Central Configuration
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "EvoTrade AI"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # LLM Provider: "openai" | "anthropic" | "gemini" | "ollama" | "mock"
    LLM_PROVIDER: str = "gemini"
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None          # Google Gemini
    GEMINI_MODEL: str = "gemini-2.0-flash"        # or gemini-1.5-pro
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    LLM_MODEL: str = "claude-sonnet-4-20250514"   # used by Anthropic

    # Exchange
    EXCHANGE: str = "binance"
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_SECRET: Optional[str] = None
    USE_TESTNET: bool = True
    PAPER_TRADING: bool = True

    # Trading Defaults
    DEFAULT_SYMBOL: str = "BTC/USDT"
    DEFAULT_TIMEFRAME: str = "1h"
    MAX_RISK_PER_TRADE: float = 0.02      # 2% per trade
    MAX_DRAWDOWN_KILL: float = 0.10       # 10% global drawdown kill
    INITIAL_CAPITAL: float = 100.0

    # Execution gate (graph + execution agent). Override via .env without code edits.
    MIN_TRADE_CONFIDENCE: float = 0.60
    MAX_OPEN_TRADES_TOTAL: int = 3
    MAX_OPEN_TRADES_PER_SYMBOL: int = 1
    TRADE_COOLDOWN_SECONDS: int = 300
    # Non-paper: human_review auto-approves at or above this confidence.
    LIVE_AUTO_APPROVE_CONFIDENCE: float = 0.80

    # Adaptive floor: each cycle nudges effective min confidence from recent closed trades.
    ADAPTIVE_TRADE_CONFIDENCE: bool = False
    ADAPTIVE_CONFIDENCE_WINDOW: int = 15
    ADAPTIVE_CONFIDENCE_STEP: float = 0.03
    ADAPTIVE_CONFIDENCE_MIN_FLOOR: float = 0.45
    ADAPTIVE_CONFIDENCE_MAX_CEIL: float = 0.75

    # Engine
    ENGINE_INTERVAL_SECONDS: int = 30
    EVOLUTION_INTERVAL_HOURS: int = 6
    # If true, start the trading scheduler as soon as FastAPI boots (for headless servers).
    # Use a single uvicorn worker (--workers 1); multiple workers would run duplicate engines.
    AUTO_START_ENGINE: bool = True

    # ── Scalping Mode ──────────────────────────────────────────────────────────
    # Enable/disable intraday scalping engine (operates alongside the main LangGraph cycle).
    SCALPING_MODE: bool = False
    # Short timeframe for scalp data fetches: "1m" | "3m" | "5m"
    SCALPING_TIMEFRAME: str = "5m"
    # Comma-separated symbols to scalp (cycles through them in order)
    SCALPING_SYMBOLS: str = "BTC/USDT"
    # Account size used for all risk calculations (USDT)
    SCALPING_ACCOUNT_SIZE: float = 1000.0
    # Hard daily loss limit as fraction of account (0.03 = 3%)
    SCALPING_MAX_DAILY_LOSS_PCT: float = 0.03
    # Target risk per trade as fraction of account (0.005 = 0.5%)
    SCALPING_RISK_PER_TRADE_PCT: float = 0.005
    # Maximum fraction of account in any single scalp position (0.25 = 25%)
    SCALPING_MAX_POSITION_PCT: float = 0.25
    # Maximum trades allowed per UTC day (anti-overtrading guard)
    SCALPING_MAX_TRADES_PER_DAY: int = 20
    # How often the scalping cycle runs (seconds); keep >= 15 to avoid rate limits
    SCALPING_INTERVAL_SECONDS: int = 30
    # Minimum SL distance as % of price — floor used when ATR is abnormally small (0.002 = 0.2%)
    SCALPING_SL_PCT: float = 0.002
    # Take-profit = actual SL distance * this multiplier (1.5 = 1:1.5 RR)
    SCALPING_TP_MULTIPLIER: float = 1.5
    # ATR multiplier for dynamic SL sizing: SL_distance = max(ATR * mult, price * SL_PCT)
    # Per-setup tuning: breakout=0.8x, pullback=1.0x, vwap_reversion=1.2x, rsi_extreme=1.5x
    SCALPING_ATR_SL_MULT: float = 0.8
    # If true, scalping final decision is LLM-driven (Gemini/OpenAI/etc), with rule fallback.
    SCALPING_USE_LLM: bool = True

    # ── Auto timeframe selection ───────────────────────────────────────────────
    # When True, the engine evaluates all SCALPING_TF_OPTIONS each cycle and
    # picks the timeframe that produces the highest-scoring setup.
    SCALPING_TF_AUTO: bool = True
    # Comma-separated timeframes to evaluate in each cycle (fastest → slowest)
    SCALPING_TF_OPTIONS: str = "1m,3m,5m"
    # Noise filter: if ATR/price ratio on a given TF exceeds this, that TF is skipped
    # (0.015 = 1.5% ATR/price — typical threshold above which 1m becomes too choppy)
    SCALPING_TF_NOISE_FILTER: float = 0.015
    # Minimum setup confidence (0–1) required before a scalp trade fires
    SCALPING_MIN_CONFIDENCE: float = 0.55
    # If True, force-close all scalp positions at SCALPING_EOD_HOUR_UTC each day
    SCALPING_EOD_CLOSE_ALL: bool = True
    # UTC hour at which end-of-day close-all triggers (0–23); 23 = 11 PM UTC
    SCALPING_EOD_HOUR_UTC: int = 23

    # Database
    DB_PATH: str = "data/evotrade.duckdb"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "data/evotrade.log"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
