"""
EvoTrade AI - Evolution / Self-Improvement Agent
Generates new strategy variations via LLM, backtests with the vectorized
engine, ranks them, and auto-promotes the best performers.
"""
from __future__ import annotations
import json
import re
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.state import trading_state, Strategy
from core.llm import get_llm_response
from core.database import db
from agents.data_agent import fetch_ohlcv, _compute_manual_indicators
from evolution.backtester import VectorizedBacktester, run_strategy_comparison
from config import settings


EVOLUTION_SYSTEM = """You are EvoTrade AI's Strategy Evolution Engine — a quantitative researcher who designs algorithmic trading strategies.
Analyze past performance and invent improved, diverse strategy variations.
Always respond with valid JSON only. Be creative but disciplined."""


def get_performance_context() -> str:
    stats = db.get_performance_stats()
    strategies = db.get_all_strategies()
    promoted = [s for s in strategies if s.get("status") == "promoted"]
    return f"""
LIVE TRADING PERFORMANCE:
- Total Trades: {stats.get('total_trades', 0)}
- Win Rate: {stats.get('win_rate', 0):.1f}%
- Total PnL: ${stats.get('total_pnl', 0):.2f}
- Avg PnL/Trade: {stats.get('avg_pnl_pct', 0):.2f}%
- Best: ${stats.get('best_trade', 0):.2f} | Worst: ${stats.get('worst_trade', 0):.2f}

PROMOTED STRATEGIES: {len(promoted)}
{f"Top: {promoted[0]['name']} Sharpe={promoted[0].get('backtest_sharpe', 0):.2f}" if promoted else "None yet."}
"""


def generate_strategy_ideas(n: int = 3) -> List[Dict]:
    perf = get_performance_context()
    symbol = trading_state.symbol
    prompt = f"""Based on this trading data, generate {n} distinct algorithmic strategies for {symbol}.
{perf}
Design diverse strategies: trending, ranging, and volatile regimes.

Respond ONLY with valid JSON:
{{
    "strategies": [
        {{
            "name": "<creative name>",
            "description": "<2-3 sentences>",
            "strategy_type": "<ema_cross|rsi_reversal|bb_squeeze|macd_momentum|multi_signal|vwap_reversion|momentum_breakout>",
            "indicators": ["ind1", "ind2"],
            "risk_per_trade": <0.01-0.03>,
            "take_profit_ratio": <1.5-4.0>,
            "sl_atr_mult": <1.0-3.0>,
            "timeframe": "<1h|4h|1d>",
            "long_only": <true|false>
        }}
    ]
}}"""
    raw = get_llm_response(prompt, system=EVOLUTION_SYSTEM, max_tokens=2000)
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group()).get("strategies", [])
    except Exception:
        pass
    # Fallback hardcoded ideas
    return [
        {"name": "EMA Crossover Alpha", "description": "Classic trend following with EMA 20/50 cross.",
         "strategy_type": "ema_cross", "indicators": ["EMA_20", "EMA_50"],
         "risk_per_trade": 0.02, "take_profit_ratio": 2.0, "sl_atr_mult": 1.5,
         "timeframe": "1h", "long_only": True},
        {"name": "RSI Bounce Pro", "description": "RSI mean reversion at extremes with ATR SL.",
         "strategy_type": "rsi_reversal", "indicators": ["RSI_14", "ATR_14"],
         "risk_per_trade": 0.015, "take_profit_ratio": 1.8, "sl_atr_mult": 1.2,
         "timeframe": "1h", "long_only": True},
        {"name": "Multi-Signal Fusion", "description": "Combines EMA trend + RSI + MACD + Volume.",
         "strategy_type": "multi_signal", "indicators": ["EMA_20", "EMA_50", "RSI_14", "MACD", "Volume"],
         "risk_per_trade": 0.02, "take_profit_ratio": 2.5, "sl_atr_mult": 1.5,
         "timeframe": "1h", "long_only": True},
    ]


def backtest_strategy_idea(idea: Dict, symbol: str, timeframe: str) -> Dict[str, float]:
    """Fetch data and run vectorized backtest. Requires live OHLCV."""
    empty = _empty_metrics()
    try:
        df = fetch_ohlcv(symbol, timeframe, limit=500)
        if df is None or len(df) < 100:
            trading_state.add_log("EvolutionAgent", "Backtest skipped — insufficient live OHLCV data", level="warn")
            return empty

        if "close" not in df.columns:
            return empty

        bt = VectorizedBacktester(initial_capital=settings.INITIAL_CAPITAL)
        config = {
            "strategy_type": idea.get("strategy_type", "ema_cross"),
            "name": idea.get("name", "Unknown"),
            "risk_per_trade": idea.get("risk_per_trade", 0.02),
            "take_profit_ratio": idea.get("take_profit_ratio", 2.0),
            "sl_atr_mult": idea.get("sl_atr_mult", 1.5),
            "long_only": idea.get("long_only", True),
        }
        result = bt.run(df, config, symbol, timeframe)
        return {
            "total_return": result.total_return_pct,
            "win_rate": result.win_rate_pct,
            "sharpe": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown_pct,
            "trade_count": result.total_trades,
            "profit_factor": result.profit_factor,
            "composite_score": result.composite_score,
        }
    except Exception as e:
        trading_state.add_log("EvolutionAgent", f"Backtest error: {e}", level="warn")
        return empty


def _empty_metrics() -> Dict[str, float]:
    """Returned when backtest cannot run on real data."""
    return {
        "total_return": 0.0,
        "win_rate": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "trade_count": 0,
        "profit_factor": 0.0,
        "composite_score": 0.0,
    }


def run_evolution_cycle() -> List[Strategy]:
    """Generate → backtest → rank → promote."""
    trading_state.add_log("EvolutionAgent", "🧬 Evolution cycle starting...", level="success")

    symbol = trading_state.symbol
    timeframe = trading_state.timeframe

    # Phase 1: Built-in strategy sweep
    trading_state.add_log("EvolutionAgent", "📊 Running built-in strategy comparison...")
    try:
        df = fetch_ohlcv(symbol, timeframe, limit=500)
        if df is not None and len(df) >= 100:
            builtin_results = run_strategy_comparison(df, symbol, timeframe)
            trading_state.add_log("EvolutionAgent",
                f"Compared {len(builtin_results)} built-in strategies. "
                f"Best: {builtin_results[0].strategy_name if builtin_results else 'N/A'}")
        else:
            builtin_results = []
    except Exception as e:
        builtin_results = []
        trading_state.add_log("EvolutionAgent", f"Built-in sweep error: {e}", level="warn")

    # Phase 2: LLM-generated ideas
    trading_state.add_log("EvolutionAgent", "🤖 Generating LLM strategy ideas...")
    ideas = generate_strategy_ideas(n=3)
    trading_state.add_log("EvolutionAgent", f"Generated {len(ideas)} ideas. Backtesting...")

    evaluated: List[Strategy] = []

    # Evaluate LLM ideas
    for idea in ideas:
        name = idea.get("name", "Unknown")
        trading_state.add_log("EvolutionAgent", f"⏳ Backtesting: {name}...")

        metrics = backtest_strategy_idea(idea, symbol, idea.get("timeframe", timeframe))
        score = metrics.get("composite_score", 0)

        rules = (f"Type: {idea.get('strategy_type')} | "
                 f"SL: {idea.get('sl_atr_mult', 1.5)}x ATR | "
                 f"TP: {idea.get('take_profit_ratio', 2.0)}x Risk | "
                 f"TF: {idea.get('timeframe', timeframe)}")

        status = "candidate"
        if (score > 35 and metrics.get("sharpe", 0) > 0.8
                and metrics.get("max_drawdown", -100) > -25
                and metrics.get("trade_count", 0) >= 10):
            status = "promoted"

        strategy = Strategy(
            id=str(uuid.uuid4())[:8],
            name=name,
            description=idea.get("description", ""),
            indicators=idea.get("indicators", []),
            rules=rules,
            created_at=datetime.utcnow().isoformat(),
            backtest_sharpe=metrics.get("sharpe"),
            backtest_return=metrics.get("total_return"),
            backtest_drawdown=metrics.get("max_drawdown"),
            backtest_winrate=metrics.get("win_rate"),
            score=score,
            status=status,
        )

        level = "success" if status == "promoted" else "info"
        prefix = "✅ AUTO-PROMOTED" if status == "promoted" else "📊"
        trading_state.add_log("EvolutionAgent",
            f"{prefix}: {name} | Score={score:.1f} | Return={metrics['total_return']:.1f}% | "
            f"Sharpe={metrics['sharpe']:.2f} | WR={metrics['win_rate']:.1f}% | "
            f"DD={metrics['max_drawdown']:.1f}%",
            level=level)

        evaluated.append(strategy)
        db.save_strategy(strategy)
        with trading_state._lock:
            trading_state.strategies.append(strategy)

    # Also save top built-in strategies
    for r in builtin_results[:3]:
        s = Strategy(
            id=str(uuid.uuid4())[:8],
            name=f"{r.strategy_name} [builtin]",
            description=f"Built-in vectorized strategy. {r.total_trades} trades, {r.win_rate_pct:.1f}% WR.",
            indicators=[],
            rules=f"Type: builtin | TF: {r.timeframe}",
            created_at=datetime.utcnow().isoformat(),
            backtest_sharpe=r.sharpe_ratio,
            backtest_return=r.total_return_pct,
            backtest_drawdown=r.max_drawdown_pct,
            backtest_winrate=r.win_rate_pct,
            score=r.composite_score,
            status="promoted" if r.composite_score > 40 else "candidate",
        )
        evaluated.append(s)
        db.save_strategy(s)
        with trading_state._lock:
            trading_state.strategies.append(s)

    evaluated.sort(key=lambda s: s.score or 0, reverse=True)
    best = evaluated[0] if evaluated else None
    trading_state.add_log("EvolutionAgent",
        f"🧬 Evolution complete! {len(evaluated)} strategies evaluated. "
        f"Best: {best.name if best else 'N/A'} (Score={best.score:.1f} if best else 0)",
        level="success")
    return evaluated


def promote_strategy(strategy_id: str) -> bool:
    with trading_state._lock:
        for s in trading_state.strategies:
            if s.id == strategy_id:
                s.status = "promoted"
                db.save_strategy(s)
                trading_state.add_log("EvolutionAgent", f"✅ '{s.name}' promoted", level="success")
                return True
    # Try DB
    strats = db.get_all_strategies()
    for s_dict in strats:
        if s_dict["id"] == strategy_id:
            from core.state import Strategy
            s = Strategy(**{k: v for k, v in s_dict.items()
                           if k in Strategy.__dataclass_fields__})
            s.status = "promoted"
            db.save_strategy(s)
            return True
    return False
