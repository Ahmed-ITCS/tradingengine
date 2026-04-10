"""
EvoTrade AI - FastAPI Backend
Real-time multi-agent trading engine with WebSocket streaming,
LangGraph pipeline, and self-evolving strategy evolution.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from core.state import trading_state, EngineStatus
from core.database import db
from core.engine import engine
from core.analytics import compute_portfolio_metrics
from core.llm import get_available_providers
from evolution.evolution_agent import promote_strategy, run_evolution_cycle
from agents.graph import approve_pending_trade, reject_pending_trade

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.LOG_FILE, encoding="utf-8"),
    ]
)
logger = logging.getLogger("evotrade.main")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EvoTrade AI",
    version=settings.VERSION,
    description="Self-Evolving Multi-Agent Trading Engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# ── WebSocket Manager ─────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self.active.append(ws)
        logger.debug(f"WS connected. Total: {len(self.active)}")

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            if ws in self.active:
                self.active.remove(ws)

    async def broadcast(self, data: Dict):
        msg = json.dumps(data, default=str)
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)


manager = ConnectionManager()


async def broadcast_loop():
    """Drain the state broadcast queue and push to all WS clients."""
    while True:
        try:
            drained = 0
            while not trading_state.broadcast_queue.empty() and drained < 20:
                item = trading_state.broadcast_queue.get_nowait()
                await manager.broadcast(item)
                drained += 1
        except Exception:
            pass
        await asyncio.sleep(0.3)


@app.on_event("startup")
async def startup():
    asyncio.create_task(broadcast_loop())
    logger.info("EvoTrade AI FastAPI started ✅")


# ── Root / Frontend ───────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    index = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"name": "EvoTrade AI", "version": settings.VERSION, "docs": "/docs"}


# ── Status & Dashboard ────────────────────────────────────────────────────────
@app.get("/api/status")
def get_status():
    return trading_state.to_dashboard_dict()


@app.get("/api/indicators")
def get_indicators():
    return {
        "indicators": trading_state.last_indicators or {},
        "sentiment": trading_state.last_news_sentiment or {},
        "price": trading_state.current_price,
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": trading_state.symbol,
        "timeframe": trading_state.timeframe,
    }


@app.get("/api/chart")
def get_chart():
    """OHLCV candlestick + indicator overlay data."""
    df = trading_state.last_ohlcv
    if df is None:
        return {"candles": [], "ema_20": [], "ema_50": [], "bb_upper": [], "bb_lower": [], "rsi": [], "macd_hist": []}

    import numpy as np

    def series(pattern: str):
        for col in df.columns:
            if pattern.upper() in col.upper():
                s = df[col].tail(150)
                return [
                    {"time": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                     "value": round(float(v), 6)}
                    for ts, v in zip(s.index, s.values)
                    if not (isinstance(v, float) and np.isnan(v))
                ]
        return []

    candles = []
    for ts, row in df.tail(150).iterrows():
        candles.append({
            "time": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "open": round(float(row["open"]), 4),
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "close": round(float(row["close"]), 4),
            "volume": round(float(row["volume"]), 2),
        })

    return {
        "candles": candles,
        "ema_20": series("EMA_20"),
        "ema_50": series("EMA_50"),
        "bb_upper": series("BBU_20_2"),
        "bb_lower": series("BBL_20_2"),
        "rsi": series("RSI_14"),
        "macd_hist": series("MACDh_12_26_9"),
    }


@app.get("/api/trades")
def get_trades():
    return {
        "open": [t.__dict__ for t in trading_state.open_trades],
        "closed": db.get_all_trades(),
        "equity_curve": trading_state.equity_curve[-500:],
    }


@app.get("/api/decisions")
def get_decisions(limit: int = Query(default=50, le=200)):
    return db.get_all_decisions()[:limit]


@app.get("/api/logs")
def get_logs(limit: int = Query(default=100, le=500)):
    with trading_state._lock:
        logs = list(reversed(trading_state.agent_logs))[:limit]
    return [{"agent": l.agent, "timestamp": l.timestamp, "message": l.message, "level": l.level} for l in logs]


@app.get("/api/strategies")
def get_strategies():
    db_strats = db.get_all_strategies()
    if db_strats:
        return db_strats
    return [s.__dict__ for s in trading_state.strategies]


@app.get("/api/performance")
def get_performance():
    return db.get_performance_stats()


@app.get("/api/analytics")
def get_analytics():
    """Full portfolio analytics — Sharpe, Sortino, Calmar, monthly PnL, etc."""
    closed = db.get_all_trades()
    return compute_portfolio_metrics(
        equity_curve=trading_state.equity_curve,
        closed_trades=closed,
        initial_capital=settings.INITIAL_CAPITAL,
    )


@app.get("/api/backtest/builtin")
def run_builtin_backtest(
    symbol: str = Query(default="BTC/USDT"),
    timeframe: str = Query(default="1h"),
):
    """Run all built-in strategies and return ranked results."""
    from agents.data_agent import fetch_ohlcv
    from evolution.backtester import run_strategy_comparison
    try:
        df = fetch_ohlcv(symbol, timeframe, limit=500)
        if df is None or len(df) < 100:
            raise HTTPException(status_code=400, detail="Insufficient data")
        results = run_strategy_comparison(df, symbol, timeframe)
        return [r.to_dict() for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Engine Control ────────────────────────────────────────────────────────────
class EngineConfig(BaseModel):
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    paper_trading: Optional[bool] = None
    llm_provider: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    max_risk_per_trade: Optional[float] = None
    engine_interval_seconds: Optional[int] = None


@app.post("/api/engine/start")
def start_engine(config: EngineConfig):
    if config.symbol:
        trading_state.symbol = config.symbol
    if config.timeframe:
        trading_state.timeframe = config.timeframe
    if config.paper_trading is not None:
        settings.PAPER_TRADING = config.paper_trading
        trading_state.paper_trading = config.paper_trading
    if config.llm_provider:
        settings.LLM_PROVIDER = config.llm_provider
    if config.anthropic_api_key:
        settings.ANTHROPIC_API_KEY = config.anthropic_api_key
        settings.LLM_PROVIDER = "anthropic"
    if config.openai_api_key:
        settings.OPENAI_API_KEY = config.openai_api_key
        settings.LLM_PROVIDER = "openai"
    if config.gemini_api_key:
        settings.GEMINI_API_KEY = config.gemini_api_key
        settings.LLM_PROVIDER = "gemini"
    if config.max_risk_per_trade:
        settings.MAX_RISK_PER_TRADE = min(config.max_risk_per_trade, 0.05)
    if config.engine_interval_seconds:
        settings.ENGINE_INTERVAL_SECONDS = max(config.engine_interval_seconds, 10)

    engine.start()
    return {
        "status": "started",
        "symbol": trading_state.symbol,
        "timeframe": trading_state.timeframe,
        "paper_trading": settings.PAPER_TRADING,
        "llm_provider": settings.LLM_PROVIDER,
    }


@app.post("/api/engine/stop")
def stop_engine():
    engine.stop()
    return {"status": "stopped"}


@app.post("/api/engine/evolve")
def trigger_evolution(background_tasks: BackgroundTasks):
    if trading_state.status == EngineStatus.EVOLVING:
        raise HTTPException(status_code=409, detail="Evolution already running")
    background_tasks.add_task(_evolution_bg)
    return {"status": "evolution_started"}


def _evolution_bg():
    prev = trading_state.status
    trading_state.status = EngineStatus.EVOLVING
    try:
        run_evolution_cycle()
    except Exception as e:
        trading_state.add_log("Engine", f"Evolution bg error: {e}", level="error")
    finally:
        trading_state.status = prev if prev != EngineStatus.EVOLVING else EngineStatus.STOPPED


# ── Strategy Control ──────────────────────────────────────────────────────────
@app.post("/api/strategies/{strategy_id}/promote")
def promote_strat(strategy_id: str):
    ok = promote_strategy(strategy_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {"status": "promoted", "id": strategy_id}


@app.post("/api/strategies/{strategy_id}/retire")
def retire_strat(strategy_id: str):
    strats = db.get_all_strategies()
    for s_dict in strats:
        if s_dict["id"] == strategy_id:
            from core.state import Strategy
            s = Strategy(**{k: v for k, v in s_dict.items() if k in Strategy.__dataclass_fields__})
            s.status = "retired"
            db.save_strategy(s)
            return {"status": "retired"}
    raise HTTPException(status_code=404, detail="Strategy not found")


# ── Trade Control ─────────────────────────────────────────────────────────────
@app.post("/api/trades/{trade_id}/close")
def manual_close(trade_id: str):
    price = trading_state.current_price
    if price <= 0:
        raise HTTPException(status_code=400, detail="No current price available")
    trading_state.close_trade(trade_id, price)
    return {"status": "closed", "trade_id": trade_id, "close_price": price}


# ── Human-in-the-Loop ─────────────────────────────────────────────────────────
@app.post("/api/trades/approve/{cycle_id}")
def approve_trade(cycle_id: str):
    ok = approve_pending_trade(cycle_id)
    return {"status": "approved" if ok else "not_found"}


@app.post("/api/trades/reject/{cycle_id}")
def reject_trade(cycle_id: str):
    ok = reject_pending_trade(cycle_id)
    return {"status": "rejected" if ok else "not_found"}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.get("/api/llm/providers")
def llm_providers():
    """Returns which LLM providers are configured and available."""
    return get_available_providers()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send full snapshot on connect
        await ws.send_text(json.dumps({
            "type": "snapshot",
            "payload": trading_state.to_dashboard_dict(),
            "ts": datetime.utcnow().isoformat(),
        }, default=str))

        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=25)
                if data == "ping":
                    await ws.send_text(json.dumps({"type": "pong", "ts": datetime.utcnow().isoformat()}))
                elif data.startswith("{"):
                    # Handle JSON commands from frontend
                    cmd = json.loads(data)
                    if cmd.get("type") == "subscribe":
                        pass  # Future: per-topic subscriptions
            except asyncio.TimeoutError:
                # Heartbeat with fresh state
                await ws.send_text(json.dumps({
                    "type": "heartbeat",
                    "payload": trading_state.to_dashboard_dict(),
                    "ts": datetime.utcnow().isoformat(),
                }, default=str))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WS error: {e}")
    finally:
        await manager.disconnect(ws)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": trading_state.status.value,
        "symbol": trading_state.symbol,
        "uptime_trades": trading_state.portfolio.trade_count,
        "ws_connections": len(manager.active),
    }


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  EvoTrade AI — Self-Evolving Multi-Agent Trading Engine")
    print(f"  Version: {settings.VERSION}")
    print(f"  Mode: {'PAPER' if settings.PAPER_TRADING else 'LIVE'}")
    print(f"  LLM: {settings.LLM_PROVIDER.upper()}")
    print("=" * 60)
    print("  Open: http://localhost:8000")
    print("  API:  http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

