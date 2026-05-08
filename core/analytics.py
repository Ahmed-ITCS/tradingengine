"""
EvoTrade AI - Analytics Module
Computes rolling performance metrics, Sharpe, Sortino,
and generates report-ready data structures.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


def _json_safe_float(x: float, *, default: float = 0.0, pos_cap: float = 999.0) -> float:
    """JSON cannot represent nan/inf; normalize for API responses."""
    try:
        v = float(np.asarray(x).item()) if isinstance(x, (np.floating, np.ndarray)) else float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(v):
        return default
    if math.isinf(v):
        return pos_cap if v > 0 else -pos_cap
    return v


def _json_safe_tree(obj: Any) -> Any:
    """Recursively replace non-finite floats so Starlette JSONResponse succeeds."""
    if isinstance(obj, dict):
        return {k: _json_safe_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_tree(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _json_safe_float(obj)
    return obj


def compute_portfolio_metrics(
    equity_curve: List[Dict],
    closed_trades: List[Dict],
    initial_capital: float = 10000.0,
) -> Dict[str, Any]:
    """
    Compute comprehensive portfolio analytics from equity curve and trades.
    Returns a dict ready to be served via /api/analytics.
    """
    if not equity_curve and not closed_trades:
        return _empty_metrics(initial_capital)

    # ── Equity curve metrics ───────────────────────────────────────────────────
    if equity_curve:
        eq_vals = np.array([e["equity"] for e in equity_curve], dtype=float)
        eq_times = [e["time"] for e in equity_curve]
    else:
        eq_vals = np.array([initial_capital])
        eq_times = [datetime.utcnow().isoformat()]

    current_equity = float(eq_vals[-1])
    total_return_pct = (current_equity / initial_capital - 1) * 100

    # Bar returns for Sharpe/Sortino
    if len(eq_vals) > 1:
        bar_returns = np.diff(eq_vals) / np.where(eq_vals[:-1] > 0, eq_vals[:-1], 1)
        sharpe = _sharpe(bar_returns, periods_per_year=8760)  # assume hourly
        sortino = _sortino(bar_returns, periods_per_year=8760)
        volatility = float(bar_returns.std() * np.sqrt(8760) * 100)
    else:
        sharpe = sortino = volatility = 0.0

    # ── Trade metrics ──────────────────────────────────────────────────────────
    if closed_trades:
        pnls = [t.get("pnl", 0) or 0 for t in closed_trades]
        rets = [t.get("pnl_pct", 0) or 0 for t in closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(pnls) * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_trade = np.mean(pnls)
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        expectancy = win_rate / 100 * avg_win + (1 - win_rate / 100) * abs(avg_loss) * -1
        total_pnl = sum(pnls)

        # Consecutive metrics
        max_consec_wins, max_consec_losses = _consecutive_stats(pnls)

        # Monthly PnL grouping
        monthly_pnl = _monthly_pnl(closed_trades)

        # Streak
        current_streak = _current_streak(pnls)
    else:
        win_rate = avg_win = avg_loss = avg_trade = best_trade = worst_trade = profit_factor = expectancy = total_pnl = 0.0
        max_consec_wins = max_consec_losses = 0
        monthly_pnl = {}
        current_streak = {"type": "none", "count": 0}

    calmar = 0.0

    # ── Equity sparkline (downsampled to 50 points) ────────────────────────────
    if len(eq_vals) > 50:
        step = len(eq_vals) // 50
        sparkline = eq_vals[::step].tolist()
    else:
        sparkline = eq_vals.tolist()

    payload = {
        # Capital
        "initial_capital": initial_capital,
        "current_equity": round(_json_safe_float(current_equity), 2),
        "total_pnl": round(_json_safe_float(total_pnl), 2),
        "total_return_pct": round(_json_safe_float(total_return_pct), 2),

        # Risk-adjusted
        "sharpe_ratio": round(_json_safe_float(sharpe), 3),
        "sortino_ratio": round(_json_safe_float(sortino), 3),
        "calmar_ratio": round(_json_safe_float(calmar), 3),
        "volatility_ann_pct": round(_json_safe_float(volatility), 2),

        # Trade stats
        "total_trades": len(closed_trades),
        "win_rate_pct": round(_json_safe_float(win_rate), 1),
        "profit_factor": round(_json_safe_float(min(profit_factor, 999)), 2),
        "expectancy": round(_json_safe_float(expectancy), 2),
        "avg_win": round(_json_safe_float(avg_win), 2),
        "avg_loss": round(_json_safe_float(avg_loss), 2),
        "avg_trade": round(_json_safe_float(avg_trade), 2),
        "best_trade": round(_json_safe_float(best_trade), 2),
        "worst_trade": round(_json_safe_float(worst_trade), 2),
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "current_streak": current_streak,

        # Charts
        "equity_sparkline": [round(_json_safe_float(v, default=initial_capital), 2) for v in sparkline],
        "monthly_pnl": monthly_pnl,
    }
    return _json_safe_tree(payload)


def _sharpe(returns: np.ndarray, periods_per_year: int = 8760, risk_free: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def _sortino(returns: np.ndarray, periods_per_year: int = 8760, mar: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < mar]
    if len(downside) == 0:
        return 99.0
    downside_std = downside.std()
    if downside_std == 0:
        return 0.0
    return float(returns.mean() / downside_std * np.sqrt(periods_per_year))


def _consecutive_stats(pnls: List[float]):
    """Find max consecutive wins and losses."""
    if not pnls:
        return 0, 0
    max_wins = max_losses = cur_wins = cur_losses = 0
    for p in pnls:
        if p > 0:
            cur_wins += 1
            cur_losses = 0
            max_wins = max(max_wins, cur_wins)
        else:
            cur_losses += 1
            cur_wins = 0
            max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses


def _monthly_pnl(trades: List[Dict]) -> Dict[str, float]:
    """Group trade PnL by YYYY-MM."""
    monthly = {}
    for t in trades:
        ts = t.get("exit_time") or t.get("entry_time") or ""
        if len(ts) >= 7:
            month = ts[:7]
            monthly[month] = round(monthly.get(month, 0) + (t.get("pnl") or 0), 2)
    return dict(sorted(monthly.items()))


def _current_streak(pnls: List[float]) -> Dict[str, Any]:
    if not pnls:
        return {"type": "none", "count": 0}
    last = pnls[-1]
    kind = "win" if last > 0 else "loss"
    count = 0
    for p in reversed(pnls):
        if (p > 0) == (last > 0):
            count += 1
        else:
            break
    return {"type": kind, "count": count}


def _empty_metrics(initial_capital: float) -> Dict[str, Any]:
    return {
        "initial_capital": initial_capital,
        "current_equity": initial_capital,
        "total_pnl": 0.0,
        "total_return_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "volatility_ann_pct": 0.0,
        "total_trades": 0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "avg_trade": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "current_streak": {"type": "none", "count": 0},
        "equity_sparkline": [initial_capital],
        "monthly_pnl": {},
    }
