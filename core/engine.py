"""
EvoTrade AI - Trading Engine (LangGraph-powered)
Single swing-trading cycle: data → news → LLM decision → execute on 4h.
"""
from __future__ import annotations
import threading
import time
import uuid
import logging
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from core.state import trading_state, EngineStatus
from core.database import db
from agents.graph import run_langgraph_cycle
from evolution.evolution_agent import run_evolution_cycle
from config import settings

logger = logging.getLogger("evotrade.engine")


class TradingEngine:
    """Drives the LangGraph pipeline on a configurable schedule."""

    def __init__(self):
        self._scheduler: BackgroundScheduler | None = None
        self._running = False
        self._cycle_lock = threading.Lock()

    def _new_scheduler(self) -> BackgroundScheduler:
        """Create a fresh scheduler (APScheduler cannot restart after shutdown)."""
        return BackgroundScheduler(daemon=False)

    def _ensure_scheduler(self) -> BackgroundScheduler:
        if self._scheduler is None:
            self._scheduler = self._new_scheduler()
            return self._scheduler
        if not self._scheduler.running:
            # APScheduler cannot reuse an executor after shutdown().
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:
                pass
            self._scheduler = self._new_scheduler()
        return self._scheduler

    def _register_jobs(self, scheduler: BackgroundScheduler) -> None:
        scheduler.add_job(
            self._trading_cycle,
            trigger=IntervalTrigger(seconds=settings.ENGINE_INTERVAL_SECONDS),
            id="trading_cycle",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        scheduler.add_job(
            self._evolution_cycle,
            trigger=IntervalTrigger(hours=settings.EVOLUTION_INTERVAL_HOURS),
            id="evolution_cycle",
            replace_existing=True,
            max_instances=1,
        )

    def start(self):
        if trading_state.status == EngineStatus.RUNNING and self._running:
            trading_state.add_log("Engine", "Already running", level="warn")
            return

        scheduler = self._ensure_scheduler()

        trading_state.status = EngineStatus.RUNNING
        trading_state.paper_trading = settings.PAPER_TRADING
        self._running = True

        self._register_jobs(scheduler)

        if not scheduler.running:
            scheduler.start()

        mode = "PAPER" if settings.PAPER_TRADING else ("TESTNET" if settings.USE_TESTNET else "LIVE")
        trading_state.add_log(
            "Engine",
            f"🚀 EvoTrade started | {trading_state.symbol} [{trading_state.timeframe}] swing | "
            f"{mode} | LLM={settings.LLM_PROVIDER.upper()} | cycle={settings.ENGINE_INTERVAL_SECONDS}s",
            level="success",
        )

        threading.Thread(target=self._trading_cycle, daemon=True, name="initial_cycle").start()

    def stop(self, *, final: bool = False):
        """Pause trading. Use final=True only on process shutdown."""
        self._running = False
        trading_state.status = EngineStatus.STOPPED

        scheduler = self._scheduler
        if scheduler is None:
            trading_state.add_log("Engine", "⏹️ Engine stopped", level="warn")
            return

        try:
            scheduler.remove_all_jobs()
        except Exception as e:
            logger.debug("remove_all_jobs: %s", e)

        if final:
            try:
                if scheduler.running:
                    scheduler.shutdown(wait=True)
            except Exception as e:
                logger.debug("scheduler shutdown: %s", e)
            self._scheduler = None

        trading_state.add_log("Engine", "⏹️ Engine stopped", level="warn")

    def trigger_evolution(self):
        threading.Thread(target=self._evolution_cycle, daemon=True, name="manual_evolution").start()

    def _trading_cycle(self):
        if not self._running:
            return
        if not self._cycle_lock.acquire(blocking=False):
            return
        cycle_id = str(uuid.uuid4())[:8]
        try:
            trading_state.add_log(
                "Engine",
                f"⚡ Cycle #{cycle_id} | {trading_state.symbol} [{trading_state.timeframe}]",
            )
            t0 = time.time()
            result = run_langgraph_cycle(
                symbol=trading_state.symbol,
                timeframe=trading_state.timeframe,
                cycle_id=cycle_id,
            )
            elapsed = time.time() - t0
            err = result.get("error")
            trading_state.add_log(
                "Engine",
                f"✅ Cycle #{cycle_id} done in {elapsed:.1f}s | {'❌ ' + err[:60] if err else 'OK'}",
                level="success" if not err else "warn",
            )
            db.log_message("Engine", "info", f"Cycle {cycle_id} in {elapsed:.1f}s")
        except Exception as e:
            logger.exception(f"Cycle {cycle_id} crashed: {e}")
            trading_state.add_log("Engine", f"💥 Cycle crashed: {str(e)[:150]}", level="error")
        finally:
            self._cycle_lock.release()

    def _evolution_cycle(self):
        if not self._running:
            return
        prev = trading_state.status
        trading_state.status = EngineStatus.EVOLVING
        try:
            run_evolution_cycle()
        except Exception as e:
            trading_state.add_log("Engine", f"Evolution error: {e}", level="error")
        finally:
            trading_state.status = prev


engine = TradingEngine()
