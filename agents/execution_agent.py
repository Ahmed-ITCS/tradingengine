"""
EvoTrade AI - Execution Agent
Places real or paper trades. Supports Binance (testnet) with easy broker switching.
"""
from __future__ import annotations
import uuid
from typing import Optional, Dict, Any
from datetime import datetime

from core.state import trading_state, Trade, TradeDecision, TradeSignal
from core.database import db
from config import settings


class PaperBroker:
    """Simulated broker for paper trading."""

    def place_order(self, symbol: str, side: str, size_usdt: float, price: float) -> Dict:
        size = size_usdt / price
        return {
            "order_id": f"PAPER-{uuid.uuid4().hex[:8].upper()}",
            "symbol": symbol,
            "side": side,
            "size": size,
            "price": price,
            "status": "filled",
            "paper": True
        }

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_balance(self) -> Dict:
        return {"USDT": trading_state.portfolio.equity}


class BinanceBroker:
    """Real Binance broker via ccxt."""

    def __init__(self):
        self._exchange = None

    def _get_exchange(self):
        if self._exchange is None:
            try:
                import ccxt
                self._exchange = ccxt.binance({
                    'apiKey': settings.BINANCE_API_KEY,
                    'secret': settings.BINANCE_SECRET,
                    'sandbox': settings.USE_TESTNET,
                    'options': {'defaultType': 'spot'},
                    'enableRateLimit': True,
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
            order = exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )
            return {
                "order_id": order['id'],
                "symbol": symbol,
                "side": side,
                "size": order.get('filled', amount),
                "price": order.get('average', price),
                "status": order.get('status', 'filled'),
                "paper": False
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
            return {k: v for k, v in balance['free'].items() if v > 0}
        except Exception:
            return {}


def get_broker():
    """Factory: return the appropriate broker based on config."""
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
    Execute a trade decision (no extra gates: graph already chose to act).
    Returns the Trade object if executed, None if HOLD or broker failure.
    """
    if decision.signal == TradeSignal.HOLD:
        trading_state.add_log("ExecutionAgent", "Signal is HOLD — no action taken")
        return None

    broker = get_broker()
    side = "buy" if decision.signal == TradeSignal.BUY else "sell"

    trading_state.add_log(
        "ExecutionAgent",
        f"Placing {'PAPER ' if settings.PAPER_TRADING else ''}{'TESTNET ' if settings.USE_TESTNET else ''}"
        f"{side.upper()} order: {decision.symbol} ${decision.size_usdt:.0f} @ ${decision.entry_price:.4f}"
    )

    order = broker.place_order(
        symbol=decision.symbol,
        side=side,
        size_usdt=decision.size_usdt,
        price=decision.entry_price
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
        status="open"
    )

    # Mark decision as executed
    decision.executed = True

    # Update state
    trading_state.add_trade(trade)
    db.save_trade(trade)

    mode_tag = "📋 PAPER" if settings.PAPER_TRADING else ("🧪 TESTNET" if settings.USE_TESTNET else "🔴 LIVE")
    trading_state.add_log(
        "ExecutionAgent",
        f"{mode_tag} | {side.upper()} {order['size']:.6f} {decision.symbol.split('/')[0]} "
        f"@ ${order['price']:.4f} | Order: {order['order_id']}",
        level="success"
    )

    return trade


def monitor_open_trades():
    """Check SL/TP for all open trades. Call periodically."""
    if not trading_state.open_trades:
        return

    price = trading_state.current_price
    if price <= 0:
        return

    broker = get_broker()
    trades_to_close = []

    with trading_state._lock:
        for trade in list(trading_state.open_trades):
            hit = check_sl_tp(trade, price)
            if hit:
                trades_to_close.append((trade.id, price, hit))

    for trade_id, close_price, reason in trades_to_close:
        trading_state.close_trade(trade_id, close_price)
        # Persist
        for t in trading_state.closed_trades:
            if t.id == trade_id:
                db.save_trade(t)
                trading_state.add_log(
                    "ExecutionAgent",
                    f"{'🛑 SL' if reason == 'sl' else '✅ TP'} hit for {t.symbol} | "
                    f"PnL: ${t.pnl:.2f} ({t.pnl_pct:.2f}%)",
                    level="success" if reason == "tp" else "warn"
                )
                break
