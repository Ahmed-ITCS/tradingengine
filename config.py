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

    # Database
    DB_PATH: str = "data/evotrade.duckdb"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "data/evotrade.log"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
