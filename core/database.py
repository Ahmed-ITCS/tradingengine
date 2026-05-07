"""
EvoTrade AI - DuckDB Persistence Layer
"""
import os
import json
import duckdb
import threading
from typing import List, Optional, Dict, Any
from datetime import datetime
from config import settings


class Database:
    """Thread-safe DuckDB wrapper for persistent storage."""

    def __init__(self):
        os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = duckdb.connect(settings.DB_PATH)
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id VARCHAR PRIMARY KEY,
                    decision_id VARCHAR,
                    symbol VARCHAR,
                    side VARCHAR,
                    entry_price DOUBLE,
                    exit_price DOUBLE,
                    size DOUBLE,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    pnl DOUBLE,
                    pnl_pct DOUBLE,
                    status VARCHAR,
                    stop_loss DOUBLE,
                    take_profit DOUBLE
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id VARCHAR PRIMARY KEY,
                    timestamp TIMESTAMP,
                    symbol VARCHAR,
                    signal VARCHAR,
                    confidence DOUBLE,
                    size_usdt DOUBLE,
                    entry_price DOUBLE,
                    stop_loss DOUBLE,
                    take_profit DOUBLE,
                    reasoning TEXT,
                    agent_contributions JSON,
                    executed BOOLEAN
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    description TEXT,
                    indicators JSON,
                    rules TEXT,
                    created_at TIMESTAMP,
                    backtest_sharpe DOUBLE,
                    backtest_return DOUBLE,
                    backtest_drawdown DOUBLE,
                    backtest_winrate DOUBLE,
                    status VARCHAR,
                    score DOUBLE
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS equity_curve (
                    ts TIMESTAMP,
                    equity DOUBLE,
                    drawdown DOUBLE
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    ts TIMESTAMP,
                    agent VARCHAR,
                    level VARCHAR,
                    message TEXT
                )
            """)

    def save_trade(self, trade) -> None:
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, [
                trade.id, trade.decision_id, trade.symbol, trade.side,
                trade.entry_price, trade.exit_price, trade.size,
                trade.entry_time, trade.exit_time,
                trade.pnl, trade.pnl_pct, trade.status,
                trade.stop_loss, trade.take_profit
            ])

    def save_decision(self, decision) -> None:
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, [
                decision.id, decision.timestamp, decision.symbol,
                decision.signal.value, decision.confidence,
                decision.size_usdt, decision.entry_price,
                decision.stop_loss, decision.take_profit,
                decision.reasoning,
                json.dumps(decision.agent_contributions),
                decision.executed
            ])

    def save_strategy(self, strategy) -> None:
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO strategies VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, [
                strategy.id, strategy.name, strategy.description,
                json.dumps(strategy.indicators), strategy.rules,
                strategy.created_at, strategy.backtest_sharpe,
                strategy.backtest_return, strategy.backtest_drawdown,
                strategy.backtest_winrate, strategy.status, strategy.score
            ])

    def get_all_trades(self) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM trades ORDER BY entry_time DESC LIMIT 500"
            ).fetchall()
            cols = ["id","decision_id","symbol","side","entry_price","exit_price",
                    "size","entry_time","exit_time","pnl","pnl_pct","status",
                    "stop_loss","take_profit"]
            return [dict(zip(cols, r)) for r in rows]

    def get_all_decisions(self) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT 200"
            ).fetchall()
            cols = ["id","timestamp","symbol","signal","confidence","size_usdt",
                    "entry_price","stop_loss","take_profit","reasoning",
                    "agent_contributions","executed"]
            result = []
            for r in rows:
                d = dict(zip(cols, r))
                d["agent_contributions"] = json.loads(d["agent_contributions"] or "{}")
                result.append(d)
            return result

    def get_all_strategies(self) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM strategies ORDER BY score DESC NULLS LAST"
            ).fetchall()
            cols = ["id","name","description","indicators","rules","created_at",
                    "backtest_sharpe","backtest_return","backtest_drawdown",
                    "backtest_winrate","status","score"]
            result = []
            for r in rows:
                d = dict(zip(cols, r))
                d["indicators"] = json.loads(d["indicators"] or "[]")
                result.append(d)
            return result

    def log_message(self, agent: str, level: str, message: str):
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO system_logs VALUES (?,?,?,?)",
                    [datetime.utcnow().isoformat(), agent, level, message]
                )
            except Exception:
                pass

    def get_logs(self, limit: int = 200) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM system_logs ORDER BY ts DESC LIMIT ?", [limit]
            ).fetchall()
            return [{"ts": r[0], "agent": r[1], "level": r[2], "message": r[3]} for r in rows]

    def get_performance_stats(self) -> Dict:
        with self._lock:
            row = self._conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(pnl) as total_pnl,
                    AVG(pnl_pct) as avg_pnl_pct,
                    MIN(pnl) as worst_trade,
                    MAX(pnl) as best_trade
                FROM trades WHERE status='closed'
            """).fetchone()
            if row:
                total = row[0] or 0
                wins = row[1] or 0
                return {
                    "total_trades": total,
                    "win_rate": (wins / total * 100) if total > 0 else 0,
                    "total_pnl": row[2] or 0,
                    "avg_pnl_pct": row[3] or 0,
                    "worst_trade": row[4] or 0,
                    "best_trade": row[5] or 0,
                }
            return {}

    def clear_all_data(self) -> None:
        """Delete all persisted runtime data while keeping schema intact."""
        with self._lock:
            self._conn.execute("DELETE FROM trades")
            self._conn.execute("DELETE FROM decisions")
            self._conn.execute("DELETE FROM strategies")
            self._conn.execute("DELETE FROM equity_curve")
            self._conn.execute("DELETE FROM system_logs")


# Global singleton
db = Database()
