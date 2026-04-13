"""
EvoTrade AI - LangGraph Multi-Agent Workflow
Implements a proper StateGraph with nodes, edges, conditional routing,
and human-in-the-loop capability.
"""
from __future__ import annotations
import json
from typing import TypedDict, Annotated, Optional, Dict, Any, List
from datetime import datetime

# LangGraph imports (with graceful fallback if not installed)
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from core.state import trading_state, TradeDecision, TradeSignal
from agents.news_agent import run_news_agent
from agents.data_agent import run_data_agent
from agents.decision_agent import run_decision_agent
from agents.execution_agent import run_execution_agent, monitor_open_trades
from config import settings


# ── LangGraph State Schema ────────────────────────────────────────────────────

class TradingGraphState(TypedDict):
    """State that flows through the LangGraph agent pipeline."""
    # Inputs
    symbol: str
    timeframe: str
    cycle_id: str

    # Agent outputs
    market_data: Optional[Dict[str, Any]]
    indicators: Optional[Dict[str, Any]]
    news_sentiment: Optional[Dict[str, Any]]
    decision: Optional[Dict[str, Any]]
    trade_result: Optional[Dict[str, Any]]

    # Control flow
    should_execute: bool
    error: Optional[str]
    completed_nodes: List[str]

    # Human-in-the-loop
    awaiting_approval: bool
    human_approved: Optional[bool]


# ── Node Functions ────────────────────────────────────────────────────────────

def node_fetch_market_data(state: TradingGraphState) -> TradingGraphState:
    """Node 1: Fetch OHLCV and compute technical indicators."""
    try:
        trading_state.add_log("LangGraph", f"[Node: DataAgent] Fetching market data for {state['symbol']}")
        result = run_data_agent(state["symbol"], state["timeframe"])

        return {
            **state,
            "market_data": result,
            "indicators": result.get("indicators", {}),
            "completed_nodes": state.get("completed_nodes", []) + ["data_agent"],
            "error": None
        }
    except Exception as e:
        return {**state, "error": f"DataAgent failed: {str(e)}", "completed_nodes": state.get("completed_nodes", []) + ["data_agent"]}


def node_fetch_news_sentiment(state: TradingGraphState) -> TradingGraphState:
    """Node 2: Fetch news and analyze sentiment."""
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
    """Node 3: Main decision orchestrator — synthesizes all signals."""
    try:
        indicators = state.get("indicators", {})
        sentiment = state.get("news_sentiment", {})

        if not indicators:
            return {**state, "error": "No indicator data available", "should_execute": False}

        portfolio_dict = trading_state.to_dashboard_dict()
        portfolio_dict["open_trades_count"] = len(trading_state.open_trades)

        trading_state.add_log("LangGraph", "[Node: DecisionAgent] Synthesizing multi-agent signals...")
        decision = run_decision_agent(
            symbol=state["symbol"],
            indicators=indicators,
            sentiment=sentiment,
            portfolio_dict=portfolio_dict
        )

        floor = trading_state.effective_min_trade_confidence()
        should_exec = (
            decision.signal != TradeSignal.HOLD
            and decision.confidence >= floor
            and trading_state.portfolio.drawdown < settings.MAX_DRAWDOWN_KILL
        )

        trading_state.add_log(
            "LangGraph",
            f"[Node: DecisionAgent] Execute gate: min_conf={floor:.0%} "
            f"(adaptive={settings.ADAPTIVE_TRADE_CONFIDENCE}) → should_execute={should_exec}",
        )

        return {
            **state,
            "decision": {
                "id": decision.id,
                "signal": decision.signal.value,
                "confidence": decision.confidence,
                "size_usdt": decision.size_usdt,
                "entry_price": decision.entry_price,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit,
                "reasoning": decision.reasoning,
            },
            "should_execute": should_exec,
            "awaiting_approval": False,  # Set True for human-in-loop mode
            "completed_nodes": state.get("completed_nodes", []) + ["decision_agent"],
        }
    except Exception as e:
        return {**state, "error": f"DecisionAgent failed: {str(e)}", "should_execute": False}


def node_human_review(state: TradingGraphState) -> TradingGraphState:
    """
    Node 4 (optional): Human-in-the-loop checkpoint.
    In async/UI mode, this would pause and wait for approval.
    For now, auto-approves high-confidence signals.
    """
    decision = state.get("decision", {})
    confidence = decision.get("confidence", 0)
    signal = decision.get("signal", "HOLD")

    # Auto-approve at LIVE_AUTO_APPROVE_CONFIDENCE+ (non-paper path)
    auto_approve = confidence >= settings.LIVE_AUTO_APPROVE_CONFIDENCE or signal == "HOLD"

    trading_state.add_log(
        "LangGraph",
        f"[Node: HumanReview] Signal={signal} Conf={confidence:.0%} → "
        f"{'AUTO-APPROVED' if auto_approve else 'AWAITING HUMAN'}",
        level="info" if auto_approve else "warn"
    )

    return {
        **state,
        "awaiting_approval": not auto_approve,
        "human_approved": auto_approve if auto_approve else None,
        "completed_nodes": state.get("completed_nodes", []) + ["human_review"],
    }


def node_execute_trade(state: TradingGraphState) -> TradingGraphState:
    """Node 5: Execute the trade decision."""
    try:
        decision_data = state.get("decision", {})
        if not decision_data:
            return {**state, "trade_result": None}

        # Reconstruct TradeDecision object for executor
        from core.state import TradeDecision, TradeSignal
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

        trading_state.add_log("LangGraph", f"[Node: ExecutionAgent] Executing {decision.signal.value} order...")
        trade = run_execution_agent(decision)

        return {
            **state,
            "trade_result": {"trade_id": trade.id, "status": "executed"} if trade else {"status": "skipped"},
            "completed_nodes": state.get("completed_nodes", []) + ["execution_agent"],
        }
    except Exception as e:
        return {**state, "error": f"ExecutionAgent failed: {str(e)}", "trade_result": {"status": "failed"}}


def node_monitor_positions(state: TradingGraphState) -> TradingGraphState:
    """Node 6: Monitor open positions for SL/TP hits."""
    try:
        monitor_open_trades()
        trading_state.add_log("LangGraph", f"[Node: Monitor] Checked {len(trading_state.open_trades)} open positions")
    except Exception as e:
        trading_state.add_log("LangGraph", f"[Node: Monitor] Error: {e}", level="warn")
    return {
        **state,
        "completed_nodes": state.get("completed_nodes", []) + ["monitor"],
    }


def node_finalize(state: TradingGraphState) -> TradingGraphState:
    """Node 7: Finalize cycle, snapshot equity."""
    from datetime import datetime
    trading_state.equity_curve.append({
        "time": datetime.utcnow().isoformat(),
        "equity": trading_state.portfolio.equity + trading_state.portfolio.unrealized_pnl
    })
    nodes_run = state.get("completed_nodes", [])
    err = state.get("error")
    if err:
        msg = f"[Cycle Complete] Nodes: {' → '.join(nodes_run)} | Error: {err}"
        lvl = "warn"
    else:
        msg = f"[Cycle Complete] Nodes: {' → '.join(nodes_run)} | Status: OK"
        lvl = "success"
    trading_state.add_log("LangGraph", msg, level=lvl)
    return {**state, "completed_nodes": nodes_run + ["finalize"]}


# ── Conditional Edge Functions ────────────────────────────────────────────────

def route_after_decision(state: TradingGraphState) -> str:
    """Route: after decision, go to human_review or skip to monitor."""
    if state.get("error"):
        return "monitor"
    if state.get("should_execute") and settings.PAPER_TRADING:
        # Paper trading: skip human review
        return "execute"
    if state.get("should_execute"):
        return "human_review"
    return "monitor"


def route_after_human_review(state: TradingGraphState) -> str:
    """Route: after human review, execute or skip."""
    if state.get("awaiting_approval") and not state.get("human_approved"):
        trading_state.add_log("LangGraph", "[HumanReview] Trade requires manual approval", level="warn")
        return "monitor"
    return "execute"


def route_after_execution(state: TradingGraphState) -> str:
    """Route: always go to monitor after execution."""
    return "monitor"


# ── Build the Graph ───────────────────────────────────────────────────────────

def build_trading_graph():
    """Build and compile the LangGraph StateGraph."""
    if not LANGGRAPH_AVAILABLE:
        return None

    graph = StateGraph(TradingGraphState)

    # Add nodes
    graph.add_node("data_agent", node_fetch_market_data)
    graph.add_node("news_agent", node_fetch_news_sentiment)
    graph.add_node("decision_agent", node_make_decision)
    graph.add_node("human_review", node_human_review)
    graph.add_node("execute", node_execute_trade)
    graph.add_node("monitor", node_monitor_positions)
    graph.add_node("finalize", node_finalize)

    # Entry point
    graph.set_entry_point("data_agent")

    # Parallel-ish: data → news → decision
    graph.add_edge("data_agent", "news_agent")
    graph.add_edge("news_agent", "decision_agent")

    # Conditional routing from decision
    graph.add_conditional_edges(
        "decision_agent",
        route_after_decision,
        {
            "human_review": "human_review",
            "execute": "execute",
            "monitor": "monitor",
        }
    )

    # Conditional routing from human review
    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "execute": "execute",
            "monitor": "monitor",
        }
    )

    # After execution → monitor
    graph.add_edge("execute", "monitor")
    graph.add_edge("monitor", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# ── Run One Cycle ─────────────────────────────────────────────────────────────

_compiled_graph = None

def run_langgraph_cycle(symbol: str, timeframe: str, cycle_id: str) -> Dict[str, Any]:
    """
    Run one complete trading cycle through the LangGraph.
    Falls back to direct agent calls if LangGraph is not available.
    """
    global _compiled_graph

    if not LANGGRAPH_AVAILABLE:
        return _fallback_cycle(symbol, timeframe, cycle_id)

    if _compiled_graph is None:
        _compiled_graph = build_trading_graph()

    if _compiled_graph is None:
        return _fallback_cycle(symbol, timeframe, cycle_id)

    initial_state: TradingGraphState = {
        "symbol": symbol,
        "timeframe": timeframe,
        "cycle_id": cycle_id,
        "market_data": None,
        "indicators": None,
        "news_sentiment": None,
        "decision": None,
        "trade_result": None,
        "should_execute": False,
        "error": None,
        "completed_nodes": [],
        "awaiting_approval": False,
        "human_approved": None,
    }

    try:
        trading_state.add_log("LangGraph", f"Starting graph execution: {symbol} [{timeframe}]")
        final_state = _compiled_graph.invoke(initial_state)
        return final_state
    except Exception as e:
        trading_state.add_log("LangGraph", f"Graph execution error: {e}", level="error")
        return _fallback_cycle(symbol, timeframe, cycle_id)


def _fallback_cycle(symbol: str, timeframe: str, cycle_id: str) -> Dict[str, Any]:
    """Direct agent execution when LangGraph is unavailable."""
    trading_state.add_log("Engine", "Running in direct-agent mode (LangGraph not available)")

    data = run_data_agent(symbol, timeframe)
    if not data:
        return {"error": "Data fetch failed", "completed_nodes": ["data_agent"]}

    sentiment = run_news_agent(symbol)
    portfolio_dict = trading_state.to_dashboard_dict()
    portfolio_dict["open_trades_count"] = len(trading_state.open_trades)

    decision = run_decision_agent(symbol, data.get("indicators", {}), sentiment, portfolio_dict)
    trade = run_execution_agent(decision)
    monitor_open_trades()

    trading_state.equity_curve.append({
        "time": datetime.utcnow().isoformat(),
        "equity": trading_state.portfolio.equity + trading_state.portfolio.unrealized_pnl
    })

    return {
        "symbol": symbol,
        "completed_nodes": ["data_agent", "news_agent", "decision_agent", "execution_agent", "monitor", "finalize"],
        "trade_result": {"trade_id": trade.id if trade else None}
    }


def approve_pending_trade(cycle_id: str) -> bool:
    """Human approval endpoint for pending trades (human-in-loop)."""
    # In a full implementation, this would update a pending state store
    trading_state.add_log("LangGraph", f"Human approved trade for cycle {cycle_id}", level="success")
    return True


def reject_pending_trade(cycle_id: str) -> bool:
    """Human rejection endpoint."""
    trading_state.add_log("LangGraph", f"Human rejected trade for cycle {cycle_id}", level="warn")
    return True
