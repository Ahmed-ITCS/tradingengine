"""
EvoTrade AI - LangGraph Multi-Agent Workflow
Routes to execution when signal is BUY/SELL and confidence >= effective floor
(MIN_TRADE_CONFIDENCE, optionally nudged by ADAPTIVE_TRADE_CONFIDENCE).
"""
from __future__ import annotations
import json
from typing import TypedDict, Optional, Dict, Any, List
from datetime import datetime

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from core.state import trading_state, TradeDecision, TradeSignal
from agents.news_agent import run_news_agent
from agents.data_agent import run_data_agent
from agents.decision_agent import run_decision_agent
from agents.execution_agent import run_execution_agent, monitor_open_trades

class TradingGraphState(TypedDict):
    symbol: str
    timeframe: str
    cycle_id: str
    market_data: Optional[Dict[str, Any]]
    indicators: Optional[Dict[str, Any]]
    news_sentiment: Optional[Dict[str, Any]]
    decision: Optional[Dict[str, Any]]
    trade_result: Optional[Dict[str, Any]]
    should_execute: bool
    error: Optional[str]
    completed_nodes: List[str]
    awaiting_approval: bool
    human_approved: Optional[bool]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_fetch_market_data(state: TradingGraphState) -> TradingGraphState:
    try:
        trading_state.add_log("LangGraph", f"[Node: DataAgent] Fetching market data for {state['symbol']}")
        result = run_data_agent(state["symbol"], state["timeframe"])
        return {
            **state,
            "market_data": result,
            "indicators": result.get("indicators", {}),
            "completed_nodes": state.get("completed_nodes", []) + ["data_agent"],
            "error": None,
        }
    except Exception as e:
        return {**state, "error": f"DataAgent failed: {e}",
                "completed_nodes": state.get("completed_nodes", []) + ["data_agent"]}


def node_fetch_news_sentiment(state: TradingGraphState) -> TradingGraphState:
    try:
        trading_state.add_log("LangGraph", f"[Node: NewsAgent] Analyzing sentiment for {state['symbol']}")
        sentiment = run_news_agent(state["symbol"])
        return {
            **state,
            "news_sentiment": sentiment,
            "completed_nodes": state.get("completed_nodes", []) + ["news_agent"],
        }
    except Exception as e:
        return {
            **state,
            "news_sentiment": {"sentiment_score": 0.5, "sentiment_label": "NEUTRAL"},
            "completed_nodes": state.get("completed_nodes", []) + ["news_agent"],
        }


def node_make_decision(state: TradingGraphState) -> TradingGraphState:
    try:
        indicators = state.get("indicators", {})
        sentiment  = state.get("news_sentiment", {})

        if not indicators:
            return {**state, "error": "No indicator data", "should_execute": False}

        trading_state.add_log("LangGraph", "[Node: DecisionAgent] Synthesizing multi-agent signals...")

        portfolio_dict = trading_state.to_dashboard_dict()
        portfolio_dict["open_trades_count"] = len(trading_state.open_trades)

        decision = run_decision_agent(
            symbol=state["symbol"],
            indicators=indicators,
            sentiment=sentiment,
            portfolio_dict=portfolio_dict,
        )

        min_confidence = trading_state.effective_min_trade_confidence()
        signal_ok      = decision.signal != TradeSignal.HOLD
        confidence_ok  = decision.confidence >= min_confidence
        should_exec    = signal_ok and confidence_ok

        trading_state.add_log(
            "LangGraph",
            f"[Node: DecisionAgent] signal={decision.signal.value} "
            f"conf={decision.confidence:.0%} min(effective)={min_confidence:.0%} "
            f"→ should_execute={should_exec}",
        )

        return {
            **state,
            "decision": {
                "id":           decision.id,
                "signal":       decision.signal.value,
                "confidence":   decision.confidence,
                "size_usdt":    decision.size_usdt,
                "entry_price":  decision.entry_price,
                "stop_loss":    decision.stop_loss,
                "take_profit":  decision.take_profit,
                "reasoning":    decision.reasoning,
            },
            "should_execute":  should_exec,
            "awaiting_approval": False,
            "completed_nodes": state.get("completed_nodes", []) + ["decision_agent"],
        }
    except Exception as e:
        trading_state.add_log("LangGraph", f"[Node: DecisionAgent] Error: {e}", level="error")
        return {**state, "error": f"DecisionAgent failed: {e}", "should_execute": False}


def node_execute_trade(state: TradingGraphState) -> TradingGraphState:
    try:
        decision_data = state.get("decision", {})
        if not decision_data:
            return {**state, "trade_result": {"status": "no_decision"}}

        decision = TradeDecision(
            id=decision_data.get("id", ""),
            timestamp=datetime.utcnow().isoformat(),
            symbol=state["symbol"],
            signal=TradeSignal(decision_data.get("signal", "HOLD")),
            confidence=decision_data.get("confidence", 0),
            size_usdt=decision_data.get("size_usdt", 0),
            entry_price=decision_data.get("entry_price", 0),
            stop_loss=decision_data.get("stop_loss"),
            take_profit=decision_data.get("take_profit"),
            reasoning=decision_data.get("reasoning", ""),
        )

        trading_state.add_log("LangGraph", f"[Node: ExecutionAgent] Placing {decision.signal.value} order...")
        trade = run_execution_agent(decision)

        return {
            **state,
            "trade_result": {"trade_id": trade.id, "status": "executed"} if trade else {"status": "skipped"},
            "completed_nodes": state.get("completed_nodes", []) + ["execute"],
        }
    except Exception as e:
        trading_state.add_log("LangGraph", f"[Node: ExecutionAgent] Error: {e}", level="error")
        return {**state, "error": f"ExecutionAgent failed: {e}", "trade_result": {"status": "failed"}}


def node_monitor_positions(state: TradingGraphState) -> TradingGraphState:
    try:
        monitor_open_trades()
        n = len(trading_state.open_trades)
        trading_state.add_log("LangGraph", f"[Node: Monitor] Checked {n} open positions")
    except Exception as e:
        trading_state.add_log("LangGraph", f"[Node: Monitor] Error: {e}", level="warn")
    return {**state, "completed_nodes": state.get("completed_nodes", []) + ["monitor"]}


def node_finalize(state: TradingGraphState) -> TradingGraphState:
    trading_state.equity_curve.append({
        "time":   datetime.utcnow().isoformat(),
        "equity": trading_state.portfolio.equity + trading_state.portfolio.unrealized_pnl,
    })
    nodes = " → ".join(state.get("completed_nodes", []))
    err   = state.get("error")
    trading_state.add_log(
        "LangGraph",
        f"[Cycle Complete] Nodes: {nodes} | Status: {'ERROR: ' + err if err else 'OK'}",
        level="success" if not err else "warn",
    )
    return {**state, "completed_nodes": state.get("completed_nodes", []) + ["finalize"]}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_decision(state: TradingGraphState) -> str:
    if state.get("error"):
        return "monitor"
    if state.get("should_execute"):
        return "execute"   # paper trading: skip human review entirely
    return "monitor"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_trading_graph():
    if not LANGGRAPH_AVAILABLE:
        return None
    g = StateGraph(TradingGraphState)
    g.add_node("data_agent",  node_fetch_market_data)
    g.add_node("news_agent",  node_fetch_news_sentiment)
    g.add_node("decision_agent", node_make_decision)
    g.add_node("execute",     node_execute_trade)
    g.add_node("monitor",     node_monitor_positions)
    g.add_node("finalize",    node_finalize)
    g.set_entry_point("data_agent")
    g.add_edge("data_agent",  "news_agent")
    g.add_edge("news_agent",  "decision_agent")
    g.add_conditional_edges(
        "decision_agent",
        route_after_decision,
        {"execute": "execute", "monitor": "monitor"},
    )
    g.add_edge("execute",  "monitor")
    g.add_edge("monitor",  "finalize")
    g.add_edge("finalize", END)
    return g.compile()


_compiled_graph = None


def run_langgraph_cycle(symbol: str, timeframe: str, cycle_id: str) -> Dict[str, Any]:
    global _compiled_graph

    if not LANGGRAPH_AVAILABLE:
        return _fallback_cycle(symbol, timeframe, cycle_id)

    if _compiled_graph is None:
        _compiled_graph = build_trading_graph()
    if _compiled_graph is None:
        return _fallback_cycle(symbol, timeframe, cycle_id)

    initial: TradingGraphState = {
        "symbol":            symbol,
        "timeframe":         timeframe,
        "cycle_id":          cycle_id,
        "market_data":       None,
        "indicators":        None,
        "news_sentiment":    None,
        "decision":          None,
        "trade_result":      None,
        "should_execute":    False,
        "error":             None,
        "completed_nodes":   [],
        "awaiting_approval": False,
        "human_approved":    None,
    }

    try:
        trading_state.add_log("LangGraph", f"Starting graph execution: {symbol} [{timeframe}]")
        return _compiled_graph.invoke(initial)
    except Exception as e:
        trading_state.add_log("LangGraph", f"Graph error: {e} — falling back", level="warn")
        return _fallback_cycle(symbol, timeframe, cycle_id)


def _fallback_cycle(symbol: str, timeframe: str, cycle_id: str) -> Dict[str, Any]:
    """Direct execution when LangGraph not available."""
    trading_state.add_log("Engine", "Running direct-agent mode (LangGraph unavailable)")
    data = run_data_agent(symbol, timeframe)
    if not data:
        return {"error": "Data fetch failed", "completed_nodes": ["data_agent"]}
    sentiment    = run_news_agent(symbol)
    portfolio    = trading_state.to_dashboard_dict()
    portfolio["open_trades_count"] = len(trading_state.open_trades)
    decision     = run_decision_agent(symbol, data.get("indicators", {}), sentiment, portfolio)
    trade        = run_execution_agent(decision)
    monitor_open_trades()
    trading_state.equity_curve.append({
        "time":   datetime.utcnow().isoformat(),
        "equity": trading_state.portfolio.equity + trading_state.portfolio.unrealized_pnl,
    })
    return {
        "symbol":          symbol,
        "completed_nodes": ["data_agent", "news_agent", "decision_agent", "execute", "monitor", "finalize"],
        "trade_result":    {"trade_id": trade.id if trade else None},
    }


def approve_pending_trade(cycle_id: str) -> bool:
    trading_state.add_log("LangGraph", f"Human approved trade for cycle {cycle_id}", level="success")
    return True


def reject_pending_trade(cycle_id: str) -> bool:
    trading_state.add_log("LangGraph", f"Human rejected trade for cycle {cycle_id}", level="warn")
    return True
