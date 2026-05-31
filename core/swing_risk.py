"""
EvoTrade AI - Swing Trading Risk Manager
Weekly P&L tracking, wider position sizing, lower trade frequency than scalping.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from config import settings


class SwingRiskManager:
    """
    Weekly risk guard for the swing engine (4h chart patterns).

    - Track realized P&L per ISO week
    - Kill trading when weekly loss >= SWING_MAX_WEEKLY_LOSS_PCT
    - Size positions for wider ATR stops
    - Enforce SWING_MAX_TRADES_PER_WEEK
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._current_week: str = self._week_utc()
        self._weekly_realized_pnl: float = 0.0
        self._weekly_trades: List[Dict] = []
        self._weekly_kill_active: bool = False
        self._weekly_start_equity: float = settings.SWING_ACCOUNT_SIZE
        self._session_start: str = datetime.now(timezone.utc).isoformat()
        self._session_realized_pnl: float = 0.0
        self._session_trades: int = 0

    @staticmethod
    def _week_utc() -> str:
        now = datetime.now(timezone.utc)
        iso = now.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"

    def _check_week_rollover(self) -> bool:
        week = self._week_utc()
        if week != self._current_week:
            self._current_week = week
            self._weekly_realized_pnl = 0.0
            self._weekly_trades = []
            self._weekly_kill_active = False
            self._weekly_start_equity = settings.SWING_ACCOUNT_SIZE
            return True
        return False

    @property
    def kill_active(self) -> bool:
        with self._lock:
            self._check_week_rollover()
            return self._weekly_kill_active

    def is_trading_allowed(self) -> Tuple[bool, str]:
        with self._lock:
            self._check_week_rollover()

            if self._weekly_kill_active:
                return False, f"Weekly loss kill active — {self._weekly_realized_pnl:.2f} USDT lost this week"

            max_loss = settings.SWING_ACCOUNT_SIZE * settings.SWING_MAX_WEEKLY_LOSS_PCT
            if self._weekly_realized_pnl <= -max_loss:
                self._weekly_kill_active = True
                return False, (
                    f"Weekly loss limit hit: {self._weekly_realized_pnl:.2f} USDT "
                    f"(limit={-max_loss:.2f} USDT / {settings.SWING_MAX_WEEKLY_LOSS_PCT*100:.0f}%)"
                )

            if len(self._weekly_trades) >= settings.SWING_MAX_TRADES_PER_WEEK:
                return False, (
                    f"Max weekly trades reached: {len(self._weekly_trades)}"
                    f"/{settings.SWING_MAX_TRADES_PER_WEEK}"
                )

            return True, "ok"

    def calculate_position_size(
        self,
        entry_price: float,
        sl_price: float,
        current_equity: Optional[float] = None,
    ) -> Dict[str, float]:
        account = current_equity or settings.SWING_ACCOUNT_SIZE
        target_risk_usdt = account * settings.SWING_RISK_PER_TRADE_PCT
        max_position_usdt = account * settings.SWING_MAX_POSITION_PCT

        if entry_price <= 0 or sl_price <= 0:
            return {
                "size_usdt": max_position_usdt * 0.15,
                "risk_usdt": target_risk_usdt,
                "sl_pct": settings.SWING_SL_PCT,
                "max_loss_week_remaining": self._remaining_weekly_loss_budget(),
            }

        sl_pct = abs(entry_price - sl_price) / entry_price
        if sl_pct < 0.005:
            sl_pct = settings.SWING_SL_PCT

        ideal_size = target_risk_usdt / sl_pct
        size_usdt = min(ideal_size, max_position_usdt)
        actual_risk = size_usdt * sl_pct

        return {
            "size_usdt": round(size_usdt, 2),
            "risk_usdt": round(actual_risk, 4),
            "sl_pct": round(sl_pct, 6),
            "max_loss_week_remaining": self._remaining_weekly_loss_budget(),
        }

    def _remaining_weekly_loss_budget(self) -> float:
        max_loss = settings.SWING_ACCOUNT_SIZE * settings.SWING_MAX_WEEKLY_LOSS_PCT
        return round(max(-max_loss - self._weekly_realized_pnl, 0.0), 4)

    def record_trade_open(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        pattern: str,
        entry_price: float,
        size_usdt: float,
    ):
        with self._lock:
            self._check_week_rollover()
            self._weekly_trades.append({
                "trade_id": trade_id,
                "symbol": symbol,
                "side": side,
                "pattern": pattern,
                "entry_price": entry_price,
                "size_usdt": size_usdt,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "exit_price": None,
                "pnl": None,
                "status": "open",
            })

    def record_trade_close(self, trade_id: str, exit_price: float, pnl: float):
        with self._lock:
            self._check_week_rollover()
            for t in self._weekly_trades:
                if t["trade_id"] == trade_id:
                    t["exit_price"] = exit_price
                    t["pnl"] = round(pnl, 4)
                    t["status"] = "closed"
                    t["exit_time"] = datetime.now(timezone.utc).isoformat()
                    break

            self._weekly_realized_pnl += pnl
            self._session_realized_pnl += pnl
            self._session_trades += 1

            max_loss = settings.SWING_ACCOUNT_SIZE * settings.SWING_MAX_WEEKLY_LOSS_PCT
            if self._weekly_realized_pnl <= -max_loss:
                self._weekly_kill_active = True

    def weekly_status(self) -> Dict:
        with self._lock:
            self._check_week_rollover()
            closed = [t for t in self._weekly_trades if t["status"] == "closed"]
            wins = [t for t in closed if (t["pnl"] or 0) > 0]
            losses = [t for t in closed if (t["pnl"] or 0) <= 0]
            max_loss = settings.SWING_ACCOUNT_SIZE * settings.SWING_MAX_WEEKLY_LOSS_PCT
            pnl_pct = (self._weekly_realized_pnl / settings.SWING_ACCOUNT_SIZE) * 100

            return {
                "week_utc": self._current_week,
                "trading_allowed": not self._weekly_kill_active,
                "kill_active": self._weekly_kill_active,
                "weekly_pnl_usdt": round(self._weekly_realized_pnl, 4),
                "weekly_pnl_pct": round(pnl_pct, 3),
                "weekly_loss_limit_usdt": round(-max_loss, 4),
                "weekly_loss_budget_remaining_usdt": self._remaining_weekly_loss_budget(),
                "weekly_loss_used_pct": round(
                    min(abs(min(self._weekly_realized_pnl, 0)) / max_loss * 100, 100), 1
                ) if max_loss > 0 else 0.0,
                "total_trades_week": len(self._weekly_trades),
                "closed_trades_week": len(closed),
                "open_trades_week": len(self._weekly_trades) - len(closed),
                "wins_week": len(wins),
                "losses_week": len(losses),
                "win_rate_week": round(len(wins) / len(closed) * 100, 1) if closed else 0.0,
                "max_trades_per_week": settings.SWING_MAX_TRADES_PER_WEEK,
                "trades_remaining_week": max(settings.SWING_MAX_TRADES_PER_WEEK - len(self._weekly_trades), 0),
                "account_size": settings.SWING_ACCOUNT_SIZE,
                "risk_per_trade_pct": settings.SWING_RISK_PER_TRADE_PCT * 100,
                "max_weekly_loss_pct": settings.SWING_MAX_WEEKLY_LOSS_PCT * 100,
                "session_pnl_usdt": round(self._session_realized_pnl, 4),
                "session_trades": self._session_trades,
                "session_start": self._session_start,
                "weekly_trades": self._weekly_trades,
            }

    def weekly_checklist(self) -> Dict:
        with self._lock:
            status = self.weekly_status()
            closed = [t for t in self._weekly_trades if t["status"] == "closed"]

            patterns_used: Dict[str, Dict] = {}
            for t in closed:
                p = t.get("pattern", "unknown")
                if p not in patterns_used:
                    patterns_used[p] = {"count": 0, "pnl": 0.0, "wins": 0}
                patterns_used[p]["count"] += 1
                patterns_used[p]["pnl"] += t.get("pnl") or 0.0
                if (t.get("pnl") or 0) > 0:
                    patterns_used[p]["wins"] += 1

            goal_met = status["weekly_pnl_usdt"] > 0
            limit_hit = status["kill_active"]
            verdict = "GREEN WEEK" if goal_met else ("STOP-OUT" if limit_hit else "RED WEEK — review patterns")

            improvements = []
            if status["win_rate_week"] < 40 and closed:
                improvements.append("Win rate below 40% — wait for higher-confidence 4h patterns")
            if limit_hit:
                improvements.append("Weekly loss limit hit — reduce size or pause until next week")
            if status["total_trades_week"] > settings.SWING_MAX_TRADES_PER_WEEK * 0.8:
                improvements.append("Near max weekly trades — swing trading needs patience")
            if not improvements:
                improvements.append("Disciplined swing execution — maintain selectivity on 4h setups")

            return {
                "verdict": verdict,
                "week_utc": status["week_utc"],
                "summary": status,
                "patterns_breakdown": patterns_used,
                "improvements": improvements,
                "checklist": {
                    "weekly_loss_within_limit": not limit_hit,
                    "trade_count_within_limit": status["total_trades_week"] <= settings.SWING_MAX_TRADES_PER_WEEK,
                    "positive_week": goal_met,
                },
            }

    def reset_kill_switch(self):
        with self._lock:
            self._weekly_kill_active = False


swing_risk = SwingRiskManager()
