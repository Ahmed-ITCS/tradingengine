"""
EvoTrade AI - Scalping Risk Manager
Tracks daily P&L, enforces daily loss kill switch, sizes positions for small accounts.

Designed for: Crypto (Binance), 24/7 operation, <$1000 account, 3% max daily loss.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from config import settings


class ScalpingRiskManager:
    """
    Daily risk guard for the scalping engine.

    Responsibilities:
    - Track realized P&L per UTC day
    - Kill trading when daily loss >= SCALPING_MAX_DAILY_LOSS_PCT of account
    - Size positions so each trade risks exactly SCALPING_RISK_PER_TRADE_PCT
    - Count daily trades, enforce SCALPING_MAX_TRADES_PER_DAY
    - Generate daily performance summary / checklist
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._current_day: str = self._today_utc()

        # Rolling daily stats (reset at UTC midnight)
        self._daily_realized_pnl: float = 0.0
        self._daily_trades: List[Dict] = []        # {side, entry, exit, pnl, setup, symbol}
        self._daily_start_equity: float = settings.SCALPING_ACCOUNT_SIZE

        # Whether trading is allowed today
        self._daily_kill_active: bool = False

        # Session-level stats (never reset within a session)
        self._session_start: str = datetime.now(timezone.utc).isoformat()
        self._session_realized_pnl: float = 0.0
        self._session_trades: int = 0

    # ── Day boundary ──────────────────────────────────────────────────────────

    @staticmethod
    def _today_utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _check_day_rollover(self) -> bool:
        """Returns True if we just crossed midnight UTC and reset state."""
        today = self._today_utc()
        if today != self._current_day:
            self._current_day = today
            self._daily_realized_pnl = 0.0
            self._daily_trades = []
            self._daily_kill_active = False
            self._daily_start_equity = settings.SCALPING_ACCOUNT_SIZE
            return True
        return False

    # ── Kill switch ───────────────────────────────────────────────────────────

    @property
    def kill_active(self) -> bool:
        with self._lock:
            self._check_day_rollover()
            return self._daily_kill_active

    def is_trading_allowed(self) -> Tuple[bool, str]:
        """
        Main gate checked before every scalp entry.
        Returns (allowed: bool, reason: str).
        """
        with self._lock:
            self._check_day_rollover()

            if self._daily_kill_active:
                return False, f"Daily loss kill active — {self._daily_realized_pnl:.2f} USDT lost today"

            max_loss = settings.SCALPING_ACCOUNT_SIZE * settings.SCALPING_MAX_DAILY_LOSS_PCT
            if self._daily_realized_pnl <= -max_loss:
                self._daily_kill_active = True
                return False, (
                    f"Daily loss limit hit: {self._daily_realized_pnl:.2f} USDT "
                    f"(limit={-max_loss:.2f} USDT / {settings.SCALPING_MAX_DAILY_LOSS_PCT*100:.0f}%)"
                )

            if len(self._daily_trades) >= settings.SCALPING_MAX_TRADES_PER_DAY:
                return False, (
                    f"Max daily trades reached: {len(self._daily_trades)}"
                    f"/{settings.SCALPING_MAX_TRADES_PER_DAY}"
                )

            return True, "ok"

    # ── Position sizing ───────────────────────────────────────────────────────

    def calculate_position_size(
        self,
        entry_price: float,
        sl_price: float,
        current_equity: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        ATR-aware position sizing for small accounts.

        Risk per trade = SCALPING_RISK_PER_TRADE_PCT × account_size
        Position size  = risk / |entry - sl|_pct
        Capped at       SCALPING_MAX_POSITION_PCT × account_size

        Returns dict with size_usdt, risk_usdt, sl_pct, max_loss_today_remaining.
        """
        account = current_equity or settings.SCALPING_ACCOUNT_SIZE
        target_risk_usdt = account * settings.SCALPING_RISK_PER_TRADE_PCT
        max_position_usdt = account * settings.SCALPING_MAX_POSITION_PCT

        if entry_price <= 0 or sl_price <= 0:
            return {
                "size_usdt": max_position_usdt * 0.1,
                "risk_usdt": target_risk_usdt,
                "sl_pct": settings.SCALPING_SL_PCT,
                "max_loss_today_remaining": self._remaining_daily_loss_budget(),
            }

        sl_pct = abs(entry_price - sl_price) / entry_price
        if sl_pct < 0.0005:
            sl_pct = settings.SCALPING_SL_PCT          # floor: 0.05% min SL

        ideal_size = target_risk_usdt / sl_pct
        size_usdt = min(ideal_size, max_position_usdt)
        actual_risk = size_usdt * sl_pct

        return {
            "size_usdt": round(size_usdt, 2),
            "risk_usdt": round(actual_risk, 4),
            "sl_pct": round(sl_pct, 6),
            "max_loss_today_remaining": self._remaining_daily_loss_budget(),
        }

    def _remaining_daily_loss_budget(self) -> float:
        max_loss = settings.SCALPING_ACCOUNT_SIZE * settings.SCALPING_MAX_DAILY_LOSS_PCT
        return round(max(-max_loss - self._daily_realized_pnl, 0.0), 4)

    # ── Trade recording ───────────────────────────────────────────────────────

    def record_trade_open(self, trade_id: str, symbol: str, side: str, setup: str, entry_price: float, size_usdt: float):
        with self._lock:
            self._check_day_rollover()
            self._daily_trades.append({
                "trade_id": trade_id,
                "symbol": symbol,
                "side": side,
                "setup": setup,
                "entry_price": entry_price,
                "size_usdt": size_usdt,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "exit_price": None,
                "pnl": None,
                "status": "open",
            })

    def record_trade_close(self, trade_id: str, exit_price: float, pnl: float):
        with self._lock:
            self._check_day_rollover()
            for t in self._daily_trades:
                if t["trade_id"] == trade_id:
                    t["exit_price"] = exit_price
                    t["pnl"] = round(pnl, 4)
                    t["status"] = "closed"
                    t["exit_time"] = datetime.now(timezone.utc).isoformat()
                    break

            self._daily_realized_pnl += pnl
            self._session_realized_pnl += pnl
            self._session_trades += 1

            # Re-check kill switch after each close
            max_loss = settings.SCALPING_ACCOUNT_SIZE * settings.SCALPING_MAX_DAILY_LOSS_PCT
            if self._daily_realized_pnl <= -max_loss:
                self._daily_kill_active = True

    # ── Status & summary ──────────────────────────────────────────────────────

    def daily_status(self) -> Dict:
        with self._lock:
            self._check_day_rollover()
            closed = [t for t in self._daily_trades if t["status"] == "closed"]
            wins = [t for t in closed if (t["pnl"] or 0) > 0]
            losses = [t for t in closed if (t["pnl"] or 0) <= 0]
            max_loss = settings.SCALPING_ACCOUNT_SIZE * settings.SCALPING_MAX_DAILY_LOSS_PCT
            pnl_pct = (self._daily_realized_pnl / settings.SCALPING_ACCOUNT_SIZE) * 100

            return {
                "date_utc": self._current_day,
                "trading_allowed": not self._daily_kill_active,
                "kill_active": self._daily_kill_active,
                "daily_pnl_usdt": round(self._daily_realized_pnl, 4),
                "daily_pnl_pct": round(pnl_pct, 3),
                "daily_loss_limit_usdt": round(-max_loss, 4),
                "daily_loss_budget_remaining_usdt": self._remaining_daily_loss_budget(),
                "daily_loss_used_pct": round(
                    min(abs(min(self._daily_realized_pnl, 0)) / max_loss * 100, 100), 1
                ),
                "total_trades_today": len(self._daily_trades),
                "closed_trades_today": len(closed),
                "open_trades_today": len(self._daily_trades) - len(closed),
                "wins_today": len(wins),
                "losses_today": len(losses),
                "win_rate_today": round(len(wins) / len(closed) * 100, 1) if closed else 0.0,
                "max_trades_per_day": settings.SCALPING_MAX_TRADES_PER_DAY,
                "trades_remaining_today": max(settings.SCALPING_MAX_TRADES_PER_DAY - len(self._daily_trades), 0),
                "account_size": settings.SCALPING_ACCOUNT_SIZE,
                "risk_per_trade_pct": settings.SCALPING_RISK_PER_TRADE_PCT * 100,
                "max_daily_loss_pct": settings.SCALPING_MAX_DAILY_LOSS_PCT * 100,
                "session_pnl_usdt": round(self._session_realized_pnl, 4),
                "session_trades": self._session_trades,
                "session_start": self._session_start,
            }

    def end_of_day_checklist(self) -> Dict:
        """
        End-of-day review summary — call at SCALPING_EOD_HOUR_UTC or on demand.
        """
        with self._lock:
            status = self.daily_status()
            closed = [t for t in self._daily_trades if t["status"] == "closed"]

            setups_used: Dict[str, Dict] = {}
            for t in closed:
                s = t.get("setup", "unknown")
                if s not in setups_used:
                    setups_used[s] = {"count": 0, "pnl": 0.0, "wins": 0}
                setups_used[s]["count"] += 1
                setups_used[s]["pnl"] += t.get("pnl") or 0.0
                if (t.get("pnl") or 0) > 0:
                    setups_used[s]["wins"] += 1

            best_trade = max(closed, key=lambda t: t.get("pnl") or 0.0, default=None)
            worst_trade = min(closed, key=lambda t: t.get("pnl") or 0.0, default=None)

            goal_met = status["daily_pnl_usdt"] > 0
            limit_hit = status["kill_active"]

            verdict = "GREEN DAY" if goal_met else ("STOP-OUT" if limit_hit else "RED DAY — review setups")

            improvements = []
            if status["win_rate_today"] < 40 and closed:
                improvements.append("Win rate below 40% — reduce trade frequency, be more selective")
            if limit_hit:
                improvements.append("Daily loss limit was hit — stick to the plan tomorrow")
            if status["total_trades_today"] > settings.SCALPING_MAX_TRADES_PER_DAY * 0.8:
                improvements.append("Near max daily trades — watch for overtrading patterns")
            if not improvements:
                improvements.append("Execution looked disciplined — maintain consistency")

            return {
                "verdict": verdict,
                "date_utc": status["date_utc"],
                "summary": status,
                "setups_breakdown": setups_used,
                "best_trade": best_trade,
                "worst_trade": worst_trade,
                "improvements": improvements,
                "checklist": {
                    "all_positions_closed": len([t for t in self._daily_trades if t["status"] == "open"]) == 0,
                    "daily_loss_within_limit": not limit_hit,
                    "trade_count_within_limit": status["total_trades_today"] <= settings.SCALPING_MAX_TRADES_PER_DAY,
                    "positive_day": goal_met,
                },
            }

    def reset_kill_switch(self):
        """Manual override — only call after deliberate review."""
        with self._lock:
            self._daily_kill_active = False

    def force_day_reset(self):
        """Force a new trading day (for testing or manual resets)."""
        with self._lock:
            self._daily_realized_pnl = 0.0
            self._daily_trades = []
            self._daily_kill_active = False
            self._current_day = self._today_utc()


# Global singleton shared between engine and API layer
scalping_risk = ScalpingRiskManager()
