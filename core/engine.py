"""
EvoTrade AI - Trading Engine (LangGraph-powered)
Uses the StateGraph pipeline for each trading cycle.
Includes optional scalping (1m–5m) and swing (4h chart patterns) cycles.
"""
from __future__ import annotations
import threading
import time
import uuid
import logging
from datetime import datetime, timezone
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

        # Start scalping / swing cycles if enabled
        if settings.SCALPING_MODE:
            self._start_scalping_jobs()
        if settings.SWING_MODE:
            self._start_swing_jobs()

        if not self._scheduler.running:
            self._scheduler.start()

        mode = "PAPER" if settings.PAPER_TRADING else ("TESTNET" if settings.USE_TESTNET else "LIVE")
        scalp_tag = " | ⚡ SCALPING ON" if settings.SCALPING_MODE else ""
        swing_tag = " | 📈 SWING ON (4h)" if settings.SWING_MODE else ""
        trading_state.add_log("Engine",
            f"🚀 EvoTrade AI started | {trading_state.symbol} [{trading_state.timeframe}] | "
            f"{mode} | LLM={settings.LLM_PROVIDER.upper()} | Interval={settings.ENGINE_INTERVAL_SECONDS}s"
            f"{scalp_tag}{swing_tag}",
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

    # ── Scalping control ───────────────────────────────────────────────────────

    def enable_scalping(self):
        """Activate scalping mode and start scalping scheduler jobs."""
        settings.SCALPING_MODE = True
        if not self._scheduler.running:
            self._scheduler.start()
        self._start_scalping_jobs()
        tf_desc = (f"AUTO({settings.SCALPING_TF_OPTIONS})"
                   if settings.SCALPING_TF_AUTO else settings.SCALPING_TIMEFRAME)
        trading_state.add_log(
            "Engine",
            f"⚡ Scalping mode ENABLED | {settings.SCALPING_SYMBOLS} [{tf_desc}] "
            f"| max_loss={settings.SCALPING_MAX_DAILY_LOSS_PCT*100:.0f}% "
            f"| risk/trade={settings.SCALPING_RISK_PER_TRADE_PCT*100:.1f}% "
            f"| cycle={settings.SCALPING_INTERVAL_SECONDS}s",
            level="success",
        )

    def disable_scalping(self):
        """Deactivate scalping mode and remove scheduler jobs."""
        settings.SCALPING_MODE = False
        for job_id in ("scalping_cycle", "scalping_eod"):
            try:
                self._scheduler.remove_job(job_id)
            except Exception:
                pass
        trading_state.add_log("Engine", "⚡ Scalping mode DISABLED", level="warn")

    def _start_scalping_jobs(self):
        """Register APScheduler jobs for scalping cycle and EOD close-all."""
        self._scheduler.add_job(
            self._scalping_cycle,
            trigger=IntervalTrigger(seconds=max(settings.SCALPING_INTERVAL_SECONDS, 15)),
            id="scalping_cycle", replace_existing=True, max_instances=1, coalesce=True,
        )
        if settings.SCALPING_EOD_CLOSE_ALL:
            from apscheduler.triggers.cron import CronTrigger
            self._scheduler.add_job(
                self._scalping_eod,
                trigger=CronTrigger(hour=settings.SCALPING_EOD_HOUR_UTC, minute=0, timezone="UTC"),
                id="scalping_eod", replace_existing=True, max_instances=1,
            )

    def _scalping_cycle(self):
        """High-frequency scalp cycle: fetch short-TF data → rule-based decision → execute."""
        if not self._running or not settings.SCALPING_MODE:
            return
        if not self._cycle_lock.acquire(blocking=False):
            return

        cycle_id = str(uuid.uuid4())[:8]
        try:
            from agents.scalping_agent import (
                run_scalping_agent, run_scalping_agent_auto, close_all_scalp_positions
            )
            from agents.data_agent import fetch_ohlcv, compute_indicators
            from agents.execution_agent import run_execution_agent, monitor_open_trades

            symbols    = [s.strip() for s in settings.SCALPING_SYMBOLS.split(",") if s.strip()]
            auto_tf    = settings.SCALPING_TF_AUTO
            tf_options = settings.SCALPING_TF_OPTIONS
            fixed_tf   = settings.SCALPING_TIMEFRAME

            for symbol in symbols:
                tf_label = f"AUTO({tf_options})" if auto_tf else fixed_tf
                trading_state.add_log("ScalpingEngine", f"⚡ Scalp cycle #{cycle_id} | {symbol} [{tf_label}]")
                t0 = time.time()

                portfolio = trading_state.to_dashboard_dict()
                portfolio["open_trades_count"] = len(trading_state.open_trades)

                if auto_tf:
                    # Auto-TF: agent handles its own multi-TF data fetching
                    decision = run_scalping_agent_auto(symbol, portfolio)
                else:
                    # Fixed TF: fetch here as before
                    df = fetch_ohlcv(symbol, fixed_tf, limit=150)
                    if df is None or len(df) < 30:
                        trading_state.add_log("ScalpingEngine", f"Insufficient data for {symbol}", level="warn")
                        continue
                    df, indicators = compute_indicators(df)
                    trading_state.current_price = indicators.get("close", trading_state.current_price)
                    trading_state.last_ohlcv = df
                    trading_state.last_indicators = indicators
                    trading_state.update_price(indicators["close"])
                    trading_state.update_unrealized_pnl()
                    decision = run_scalping_agent(symbol, indicators, df, portfolio)

                run_execution_agent(decision)
                monitor_open_trades()

                trading_state.equity_curve.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "equity": trading_state.portfolio.equity + trading_state.portfolio.unrealized_pnl,
                })

                elapsed = time.time() - t0
                # Extract which TF was actually selected (stored in agent_contributions)
                chosen_tf = (decision.agent_contributions or {}).get("timeframe", fixed_tf)
                trading_state.add_log(
                    "ScalpingEngine",
                    f"✅ Scalp cycle #{cycle_id} done in {elapsed:.1f}s | {symbol} [{chosen_tf}] → {decision.signal.value}",
                    level="success",
                )
                db.log_message("ScalpingEngine", "info", f"Scalp cycle {cycle_id} | {symbol} [{chosen_tf}] | {elapsed:.1f}s")

        except Exception as e:
            logger.exception(f"Scalping cycle {cycle_id} crashed: {e}")
            trading_state.add_log("ScalpingEngine", f"💥 Scalp cycle crashed: {str(e)[:150]}", level="error")
        finally:
            self._cycle_lock.release()

    # ── Swing control ────────────────────────────────────────────────────────

    def enable_swing(self):
        """Activate swing mode and start 4h pattern scheduler jobs."""
        settings.SWING_MODE = True
        if not self._scheduler.running:
            self._scheduler.start()
        self._start_swing_jobs()
        trading_state.add_log(
            "Engine",
            f"📈 Swing mode ENABLED | {settings.SWING_SYMBOLS} [4h chart patterns] "
            f"| max_weekly_loss={settings.SWING_MAX_WEEKLY_LOSS_PCT*100:.0f}% "
            f"| risk/trade={settings.SWING_RISK_PER_TRADE_PCT*100:.1f}% "
            f"| cycle={settings.SWING_INTERVAL_SECONDS}s",
            level="success",
        )

    def disable_swing(self):
        settings.SWING_MODE = False
        try:
            self._scheduler.remove_job("swing_cycle")
        except Exception:
            pass
        trading_state.add_log("Engine", "📈 Swing mode DISABLED", level="warn")

    def _start_swing_jobs(self):
        self._scheduler.add_job(
            self._swing_cycle,
            trigger=IntervalTrigger(seconds=max(settings.SWING_INTERVAL_SECONDS, 300)),
            id="swing_cycle", replace_existing=True, max_instances=1, coalesce=True,
        )

    def _swing_cycle(self):
        """Swing cycle: fetch 4h OHLCV → detect all chart patterns → execute."""
        if not self._running or not settings.SWING_MODE:
            return
        if not self._cycle_lock.acquire(blocking=False):
            return

        cycle_id = str(uuid.uuid4())[:8]
        try:
            from agents.swing_agent import run_swing_agent
            from agents.data_agent import fetch_ohlcv, compute_indicators
            from agents.execution_agent import run_execution_agent, monitor_open_trades

            symbols = [s.strip() for s in settings.SWING_SYMBOLS.split(",") if s.strip()]
            tf = settings.SWING_TIMEFRAME

            for symbol in symbols:
                trading_state.add_log(
                    "SwingEngine",
                    f"📈 Swing cycle #{cycle_id} | {symbol} [{tf}] — scanning chart patterns",
                )
                t0 = time.time()

                df = fetch_ohlcv(symbol, tf, limit=settings.SWING_OHLCV_LIMIT)
                if df is None or len(df) < 50:
                    trading_state.add_log("SwingEngine", f"Insufficient 4h data for {symbol}", level="warn")
                    continue

                df, indicators = compute_indicators(df)
                trading_state.current_price = indicators.get("close", trading_state.current_price)
                trading_state.last_ohlcv = df
                trading_state.last_indicators = indicators
                trading_state.update_price(indicators["close"])
                trading_state.update_unrealized_pnl()

                portfolio = trading_state.to_dashboard_dict()
                portfolio["open_trades_count"] = len(trading_state.open_trades)

                decision = run_swing_agent(symbol, indicators, df, portfolio)
                run_execution_agent(decision)
                monitor_open_trades()

                trading_state.equity_curve.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "equity": trading_state.portfolio.equity + trading_state.portfolio.unrealized_pnl,
                })

                elapsed = time.time() - t0
                patterns_n = (decision.agent_contributions or {}).get("patterns_detected", "")
                trading_state.add_log(
                    "SwingEngine",
                    f"✅ Swing cycle #{cycle_id} done in {elapsed:.1f}s | {symbol} [4h] → "
                    f"{decision.signal.value} | patterns: {patterns_n or 'none'}",
                    level="success",
                )
                db.log_message("SwingEngine", "info", f"Swing cycle {cycle_id} | {symbol} [4h] | {elapsed:.1f}s")

        except Exception as e:
            logger.exception(f"Swing cycle {cycle_id} crashed: {e}")
            trading_state.add_log("SwingEngine", f"💥 Swing cycle crashed: {str(e)[:150]}", level="error")
        finally:
            self._cycle_lock.release()

    def _scalping_eod(self):
        """End-of-day routine: close all scalp positions and log the daily summary."""
        if not settings.SCALPING_MODE:
            return
        try:
            from agents.scalping_agent import close_all_scalp_positions
            from core.scalping_risk import scalping_risk

            close_all_scalp_positions("Scheduled EOD close-all")
            summary = scalping_risk.end_of_day_checklist()
            trading_state.add_log(
                "ScalpingEngine",
                f"📋 EOD Summary | {summary['verdict']} | "
                f"PnL={summary['summary']['daily_pnl_usdt']:.2f} USDT | "
                f"Trades={summary['summary']['closed_trades_today']} | "
                f"WR={summary['summary']['win_rate_today']:.0f}%",
                data=summary,
                level="success" if "GREEN" in summary["verdict"] else "warn",
            )
        except Exception as e:
            trading_state.add_log("ScalpingEngine", f"EOD routine error: {e}", level="error")

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
