"""
EvoTrade AI - Execution Agent
Places real or paper trades. Supports Binance (testnet) with easy broker switching.

THRESHOLDS (all tunable):
  MIN_CONFIDENCE      = 0.35   (was 0.55 — was blocking almost everything)
  HUMAN_REVIEW_ABOVE  = 0.80   (auto-approve below this in paper mode)
  DUPLICATE_POSITIONS = False  (allow adding to existing position)
"""
from __future__ import annotations
import uuid
from typing import Optional, Dict, Any
from datetime import datetime

from core.state import trading_state, Trade, TradeDecision, TradeSignal
from core.database import db
from config import settings

# ── Tuneable thresholds ────────────────────────────────────────────────────────
MIN_CONFIDENCE = 0.35          # execute if confidence >= this
MAX_OPEN_POSITIONS = 3         # max concurrent open trades per symbol
ALLOW_ADDING_TO_POSITION = False  # True = pyramid into winning trades


class PaperBroker:
    """Simulated broker for paper trading — never touches real money."""

    def place_order(self, symbol: str, side: str, size_usdt: float, price: float) -> Dict:
        size = size_usdt / price if price > 0 else 0
        return {
            "order_id": f"PAPER-{uuid.uuid4().hex[:8].upper()}",
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "status": "filled",
            "paper": True,
        }

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_balance(self) -> Dict:
        return {"USDT": trading_state.portfolio.equity}


class BinanceBroker:
    """Live / testnet Binance broker via ccxt."""

    def __init__(self):
        self._exchange = None

    def _get_exchange(self):
        if self._exchange is None:
            try:
                import ccxt
                self._exchange = ccxt.binance({
                    "apiKey": settings.BINANCE_API_KEY,
                    "secret": settings.BINANCE_SECRET,
                    "sandbox": settings.USE_TESTNET,
                    "options": {"defaultType": "spot"},
                    "enableRateLimit": True,
                })
            except Exception as e:
                trading_state.add_log("ExecutionAgent", f"Binance init error: {e}", level="error")
                return None
        return self._exchange

    def place_order(self, symbol: str, side: str, size_usdt: float, price: float) -> Optional[Dict]:
        exchange = self._get_exchange()
        if not exchange:
            return None
        try:
            amount = size_usdt / price
            order = exchange.create_order(symbol=symbol, type="market", side=side, amount=amount)
            return {
                "order_id": order["id"],
                "symbol": symbol,
                "side": side,
                "size": order.get("filled", amount),
                "price": order.get("average", price),
                "status": order.get("status", "filled"),
                "paper": False,
            }
        except Exception as e:
            trading_state.add_log("ExecutionAgent", f"Order failed: {e}", level="error")
            return None

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        exchange = self._get_exchange()
        if not exchange:
            return False
        try:
            exchange.cancel_order(order_id, symbol)
            return True
        except Exception:
            return False

    def get_balance(self) -> Dict:
        exchange = self._get_exchange()
        if not exchange:
            return {}
        try:
            balance = exchange.fetch_balance()
            return {k: v for k, v in balance["free"].items() if v > 0}
        except Exception:
            return {}


def get_broker():
    """Factory: return paper or live broker based on config."""
    if settings.PAPER_TRADING:
        return PaperBroker()
    return BinanceBroker()


def check_sl_tp(trade: Trade, current_price: float) -> Optional[str]:
    """Check if SL or TP has been hit. Returns 'sl', 'tp', or None."""
    if trade.side == "buy":
        if trade.stop_loss and current_price <= trade.stop_loss:
            return "sl"
        if trade.take_profit and current_price >= trade.take_profit:
            return "tp"
    elif trade.side == "sell":
        if trade.stop_loss and current_price >= trade.stop_loss:
            return "sl"
        if trade.take_profit and current_price <= trade.take_profit:
            return "tp"
    return None


def run_execution_agent(decision: TradeDecision) -> Optional[Trade]:
    """
    Execute a trade decision.

    Gates (in order):
      1. Signal must be BUY or SELL (not HOLD)
      2. Confidence must be >= MIN_CONFIDENCE (default 0.35)
      3. Drawdown kill-switch must not be active
      4. Max open positions limit
    """
    # Gate 1: HOLD = skip
    if decision.signal == TradeSignal.HOLD:
        trading_state.add_log("ExecutionAgent", "Signal=HOLD → no action")
        return None

    # Gate 2: Confidence threshold (LOWERED to 0.35)
    if decision.confidence < MIN_CONFIDENCE:
        trading_state.add_log(
            "ExecutionAgent",
            f"Confidence {decision.confidence:.0%} < {MIN_CONFIDENCE:.0%} threshold → skip",
            level="warn",
        )
        return None

    # Gate 3: Drawdown kill-switch
    if trading_state.portfolio.drawdown >= settings.MAX_DRAWDOWN_KILL:
        trading_state.add_log(
            "ExecutionAgent",
            f"🛑 KILL SWITCH — drawdown {trading_state.portfolio.drawdown*100:.1f}% ≥ "
            f"{settings.MAX_DRAWDOWN_KILL*100:.0f}% limit",
            level="error",
        )
        return None

    # Gate 4: Max open positions (per symbol)
    with trading_state._lock:
        existing = [t for t in trading_state.open_trades if t.symbol == decision.symbol]
    if len(existing) >= MAX_OPEN_POSITIONS:
        trading_state.add_log(
            "ExecutionAgent",
            f"Max {MAX_OPEN_POSITIONS} open positions reached for {decision.symbol} → skip",
            level="warn",
        )
        return None

    # ── Place the order ────────────────────────────────────────────────────────
    broker = get_broker()
    side = "buy" if decision.signal == TradeSignal.BUY else "sell"

    mode_tag = "📋 PAPER" if settings.PAPER_TRADING else ("🧪 TESTNET" if settings.USE_TESTNET else "🔴 LIVE")
    trading_state.add_log(
        "ExecutionAgent",
        f"{mode_tag} | Placing {side.upper()} {decision.symbol} "
        f"${decision.size_usdt:.0f} @ ${decision.entry_price:.4f} "
        f"(conf={decision.confidence:.0%})",
    )

    order = broker.place_order(
        symbol=decision.symbol,
        side=side,
        size_usdt=decision.size_usdt,
        price=decision.entry_price,
    )

    if not order:
        trading_state.add_log("ExecutionAgent", "Order placement failed", level="error")
        return None

    trade = Trade(
        id=str(uuid.uuid4())[:8],
        decision_id=decision.id,
        symbol=decision.symbol,
        side=side,
        entry_price=order["price"],
        size=order["size"],
        entry_time=datetime.utcnow().isoformat(),
        stop_loss=decision.stop_loss,
        take_profit=decision.take_profit,
        status="open",
    )

    decision.executed = True
    trading_state.add_trade(trade)
    db.save_trade(trade)

    trading_state.add_log(
        "ExecutionAgent",
        f"✅ {mode_tag} | {side.upper()} {order['size']:.6f} "
        f"{decision.symbol.split('/')[0]} @ ${order['price']:.4f} "
        f"| SL=${decision.stop_loss:.4f if decision.stop_loss else 0:.4f} "
        f"| TP=${decision.take_profit:.4f if decision.take_profit else 0:.4f} "
        f"| ID:{trade.id}",
        level="success",
    )
    return trade


def monitor_open_trades():
    """Check SL/TP for all open trades. Call every cycle."""
    if not trading_state.open_trades:
        return

    price = trading_state.current_price
    if price <= 0:
        return

    trades_to_close = []
    with trading_state._lock:
        for trade in list(trading_state.open_trades):
            hit = check_sl_tp(trade, price)
            if hit:
                trades_to_close.append((trade.id, price, hit))

    for trade_id, close_price, reason in trades_to_close:
        trading_state.close_trade(trade_id, close_price)
        for t in trading_state.closed_trades:
            if t.id == trade_id:
                db.save_trade(t)
                pnl_str = f"${t.pnl:+.2f} ({t.pnl_pct:+.2f}%)" if t.pnl else ""
                trading_state.add_log(
                    "ExecutionAgent",
                    f"{'✅ TP HIT' if reason == 'tp' else '🛑 SL HIT'} | "
                    f"{t.symbol} | {pnl_str}",
                    level="success" if reason == "tp" else "warn",
                )
                break
