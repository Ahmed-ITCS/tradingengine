"""
EvoTrade AI - Trading Engine (LangGraph-powered)
Uses the StateGraph pipeline for each trading cycle.
"""
from __future__ import annotations
import threading
import time
import uuid
import logging
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from core.state import trading_state, EngineStatus
from core.database import db
from agents.graph import run_langgraph_cycle
from evolution.evolution_agent import run_evolution_cycle
from config import settings

logger = logging.getLogger("evotrade.engine")


class TradingEngine:
    """
    Main engine that drives the LangGraph multi-agent pipeline
    on a configurable schedule via APScheduler.
    """

    def __init__(self):
        # daemon=False keeps the scheduler thread non-daemon so the process stays predictable
        # under process managers (systemd, Docker) until engine.stop() runs on shutdown.
        self._scheduler = BackgroundScheduler(daemon=False)
        self._running = False
        self._cycle_lock = threading.Lock()

    def start(self):
        if trading_state.status == EngineStatus.RUNNING:
            trading_state.add_log("Engine", "Already running", level="warn")
            return

        trading_state.status = EngineStatus.RUNNING
        trading_state.paper_trading = settings.PAPER_TRADING
        self._running = True

        self._scheduler.add_job(
            self._trading_cycle,
            trigger=IntervalTrigger(seconds=settings.ENGINE_INTERVAL_SECONDS),
            id="trading_cycle", replace_existing=True, max_instances=1, coalesce=True,
        )
        self._scheduler.add_job(
            self._evolution_cycle,
            trigger=IntervalTrigger(hours=settings.EVOLUTION_INTERVAL_HOURS),
            id="evolution_cycle", replace_existing=True, max_instances=1,
        )

        if not self._scheduler.running:
            self._scheduler.start()

        mode = "PAPER" if settings.PAPER_TRADING else ("TESTNET" if settings.USE_TESTNET else "LIVE")
        trading_state.add_log("Engine",
            f"🚀 EvoTrade AI started | {trading_state.symbol} [{trading_state.timeframe}] | "
            f"{mode} | LLM={settings.LLM_PROVIDER.upper()} | Interval={settings.ENGINE_INTERVAL_SECONDS}s",
            level="success")

        threading.Thread(target=self._trading_cycle, daemon=True, name="initial_cycle").start()

    def stop(self):
        self._running = False
        trading_state.status = EngineStatus.STOPPED
        if self._scheduler.running:
            try:
                self._scheduler.remove_all_jobs()
                self._scheduler.shutdown(wait=False)
            except Exception:
                pass
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
            trading_state.add_log("Engine", f"⚡ Cycle #{cycle_id} | {trading_state.symbol}")
            t0 = time.time()
            result = run_langgraph_cycle(
                symbol=trading_state.symbol,
                timeframe=trading_state.timeframe,
                cycle_id=cycle_id,
            )
            elapsed = time.time() - t0
            err = result.get("error")
            trading_state.add_log("Engine",
                f"✅ Cycle #{cycle_id} done in {elapsed:.1f}s | {'❌ ' + err[:60] if err else 'OK'}",
                level="success" if not err else "warn")
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
