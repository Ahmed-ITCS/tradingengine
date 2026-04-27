"""
EvoTrade AI - Core State Management
Shared state between all agents and the API layer.
"""
from __future__ import annotations
import threading
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import queue

from config import settings


class TradeSignal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class EngineStatus(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    EVOLVING = "evolving"
    ERROR = "error"


@dataclass
class AgentLog:
    agent: str
    timestamp: str
    message: str
    data: Optional[Dict] = None
    level: str = "info"   # info | warn | error | success


@dataclass
class TradeDecision:
    id: str
    timestamp: str
    symbol: str
    signal: TradeSignal
    confidence: float          # 0-1
    size_usdt: float
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    agent_contributions: Dict[str, str] = field(default_factory=dict)
    executed: bool = False


@dataclass
class Trade:
    id: str
    decision_id: str
    symbol: str
    side: str
    entry_price: float
    size: float                # in base currency
    entry_time: str
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: str = "open"       # open | closed
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class Strategy:
    id: str
    name: str
    description: str
    indicators: List[str]
    rules: str
    created_at: str
    backtest_sharpe: Optional[float] = None
    backtest_return: Optional[float] = None
    backtest_drawdown: Optional[float] = None
    backtest_winrate: Optional[float] = None
    status: str = "candidate"  # candidate | promoted | retired
    score: Optional[float] = None


@dataclass
class PortfolioState:
    capital: float = settings.INITIAL_CAPITAL
    equity: float = settings.INITIAL_CAPITAL
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    drawdown: float = 0.0
    peak_equity: float = settings.INITIAL_CAPITAL
    win_count: int = 0
    loss_count: int = 0
    trade_count: int = 0

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0

    @property
    def sharpe(self) -> float:
        # Simplified: will be replaced by actual calc
        return 0.0


class TradingState:
    """Thread-safe shared state for the entire trading engine."""

    def __init__(self):
        self._lock = threading.RLock()
        self.status: EngineStatus = EngineStatus.STOPPED
        self.symbol: str = "BTC/USDT"
        self.timeframe: str = "1h"
        self.paper_trading: bool = True

        self.portfolio: PortfolioState = PortfolioState()
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.decisions: List[TradeDecision] = []
        self.strategies: List[Strategy] = []
        self.agent_logs: List[AgentLog] = []

        self.current_price: float = 0.0
        self.last_ohlcv: Optional[List] = None
        self.last_indicators: Optional[Dict] = None
        self.last_news_sentiment: Optional[Dict] = None
        self.last_decision: Optional[TradeDecision] = None

        self.equity_curve: List[Dict] = []
        self.price_history: List[Dict] = []

        # Adaptive min confidence (updated when ADAPTIVE_TRADE_CONFIDENCE is enabled)
        self._dynamic_min_confidence: Optional[float] = None

        # WebSocket broadcast queue
        self.broadcast_queue: queue.Queue = queue.Queue(maxsize=500)

    def effective_min_trade_confidence(self) -> float:
        """
        Minimum confidence required to route to execution.
        Uses MIN_TRADE_CONFIDENCE from settings; optionally nudges each cycle from recent PnL.
        """
        from config import settings

        base = float(settings.MIN_TRADE_CONFIDENCE)
        if not settings.ADAPTIVE_TRADE_CONFIDENCE:
            return base

        with self._lock:
            if self._dynamic_min_confidence is None:
                self._dynamic_min_confidence = base
            window = max(3, int(settings.ADAPTIVE_CONFIDENCE_WINDOW))
            recent = list(self.closed_trades[-window:])
            if len(recent) >= 3:
                wins = sum(1 for t in recent if (t.pnl or 0) > 0)
                wr = wins / len(recent)
                step = float(settings.ADAPTIVE_CONFIDENCE_STEP)
                cur = float(self._dynamic_min_confidence)
                if wr < 0.35:
                    cur = min(float(settings.ADAPTIVE_CONFIDENCE_MAX_CEIL), cur + step)
                elif wr > 0.55:
                    cur = max(float(settings.ADAPTIVE_CONFIDENCE_MIN_FLOOR), cur - step)
                self._dynamic_min_confidence = cur
            out = float(self._dynamic_min_confidence)
            lo = float(settings.ADAPTIVE_CONFIDENCE_MIN_FLOOR)
            hi = float(settings.ADAPTIVE_CONFIDENCE_MAX_CEIL)
            return max(lo, min(hi, out))

    def add_log(self, agent: str, message: str, data: Optional[Dict] = None, level: str = "info"):
        with self._lock:
            log = AgentLog(
                agent=agent,
                timestamp=datetime.utcnow().isoformat(),
                message=message,
                data=data,
                level=level
            )
            self.agent_logs.append(log)
            # Keep last 500 logs
            if len(self.agent_logs) > 500:
                self.agent_logs = self.agent_logs[-500:]
            self._broadcast({"type": "log", "payload": {
                "agent": log.agent,
                "timestamp": log.timestamp,
                "message": log.message,
                "level": log.level
            }})

    def update_price(self, price: float):
        with self._lock:
            self.current_price = price
            self.price_history.append({
                "time": datetime.utcnow().isoformat(),
                "price": price
            })
            if len(self.price_history) > 1000:
                self.price_history = self.price_history[-1000:]

    def add_decision(self, decision: TradeDecision):
        with self._lock:
            self.decisions.append(decision)
            self.last_decision = decision
            self._broadcast({"type": "decision", "payload": {
                "id": decision.id,
                "signal": decision.signal.value,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning[:200],
                "timestamp": decision.timestamp
            }})

    def add_trade(self, trade: Trade):
        with self._lock:
            self.open_trades.append(trade)
            self.portfolio.trade_count += 1
            self._broadcast({"type": "trade_open", "payload": {
                "id": trade.id,
                "symbol": trade.symbol,
                "side": trade.side,
                "entry_price": trade.entry_price,
                "size": trade.size
            }})

    def close_trade(self, trade_id: str, exit_price: float):
        with self._lock:
            for i, t in enumerate(self.open_trades):
                if t.id == trade_id:
                    t.exit_price = exit_price
                    t.exit_time = datetime.utcnow().isoformat()
                    t.status = "closed"
                    if t.side == "buy":
                        t.pnl = (exit_price - t.entry_price) * t.size
                    else:
                        t.pnl = (t.entry_price - exit_price) * t.size
                    t.pnl_pct = t.pnl / (t.entry_price * t.size) * 100

                    self.portfolio.realized_pnl += t.pnl
                    self.portfolio.equity += t.pnl
                    if t.pnl > 0:
                        self.portfolio.win_count += 1
                    else:
                        self.portfolio.loss_count += 1

                    if self.portfolio.equity > self.portfolio.peak_equity:
                        self.portfolio.peak_equity = self.portfolio.equity
                    self.portfolio.drawdown = (
                        (self.portfolio.peak_equity - self.portfolio.equity)
                        / self.portfolio.peak_equity
                    )

                    self.closed_trades.append(t)
                    self.open_trades.pop(i)

                    self.equity_curve.append({
                        "time": t.exit_time,
                        "equity": self.portfolio.equity
                    })
                    self._broadcast({"type": "trade_closed", "payload": {
                        "id": t.id,
                        "pnl": t.pnl,
                        "pnl_pct": t.pnl_pct
                    }})
                    break

    def update_unrealized_pnl(self):
        with self._lock:
            total_unrealized = 0.0
            for t in self.open_trades:
                if self.current_price > 0:
                    if t.side == "buy":
                        t.pnl = (self.current_price - t.entry_price) * t.size
                    else:
                        t.pnl = (t.entry_price - self.current_price) * t.size
                    total_unrealized += t.pnl
            self.portfolio.unrealized_pnl = total_unrealized

    def _broadcast(self, message: Dict):
        try:
            self.broadcast_queue.put_nowait(message)
        except queue.Full:
            pass

    def to_dashboard_dict(self) -> Dict:
        with self._lock:
            return {
                "status": self.status.value,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "paper_trading": self.paper_trading,
                "current_price": self.current_price,
                "portfolio": {
                    "capital": self.portfolio.capital,
                    "equity": self.portfolio.equity,
                    "unrealized_pnl": self.portfolio.unrealized_pnl,
                    "realized_pnl": self.portfolio.realized_pnl,
                    "drawdown": self.portfolio.drawdown,
                    "win_rate": self.portfolio.win_rate,
                    "trade_count": self.portfolio.trade_count,
                },
                "open_trades_count": len(self.open_trades),
                "closed_trades_count": len(self.closed_trades),
                "last_signal": self.last_decision.signal.value if self.last_decision else "NONE",
                "last_confidence": self.last_decision.confidence if self.last_decision else 0.0,
                "last_reasoning": (self.last_decision.reasoning or "")[:800] if self.last_decision else "",
                "min_trade_confidence": self.effective_min_trade_confidence(),
            }


# Global singleton
trading_state = TradingState()
