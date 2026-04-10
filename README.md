# EvoTrade AI — Self-Evolving Multi-Agent Trading Engine

A professional-grade algorithmic trading system with real-time WebSocket UI, LangGraph multi-agent architecture, and self-improving strategy evolution.

## 🏗 Architecture

```
EvoTrade AI
├── FastAPI Backend (main.py)          ← REST + WebSocket server
├── agents/
│   ├── news_agent.py                  ← Fetches & analyzes news sentiment
│   ├── data_agent.py                  ← OHLCV + pandas-ta indicators
│   ├── decision_agent.py              ← LLM orchestrator (BUY/SELL/HOLD)
│   └── execution_agent.py             ← Paper/live order execution
├── evolution/
│   └── evolution_agent.py             ← LLM strategy generation + backtesting
├── core/
│   ├── engine.py                      ← APScheduler trading loop
│   ├── state.py                       ← Thread-safe shared state
│   ├── database.py                    ← DuckDB persistence
│   └── llm.py                        ← Anthropic/OpenAI/Ollama abstraction
├── frontend/
│   └── index.html                     ← Bloomberg-style trading terminal UI
└── config.py                          ← Pydantic settings
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd evotrade
pip install -r requirements.txt
```

### 2. Configure (optional)

Create `.env` file:

```env
# LLM (at least one)
ANTHROPIC_API_KEY=sk-ant-...
# or
OPENAI_API_KEY=sk-...

# Exchange (optional — synthetic data works without keys)
BINANCE_API_KEY=your_key
BINANCE_SECRET=your_secret
USE_TESTNET=true

# Trading
PAPER_TRADING=true
INITIAL_CAPITAL=100
MAX_RISK_PER_TRADE=0.02
```

### 3. Run

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

## 🎛 Features

### Multi-Agent System

| Agent                     | Role                                                                                     |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| **Data Agent**      | Fetches OHLCV via ccxt (Binance), computes 15+ indicators with pandas-ta                 |
| **News Agent**      | RSS news aggregation + LLM sentiment analysis                                            |
| **Decision Agent**  | Synthesizes all signals, reasons step-by-step, outputs BUY/SELL/HOLD + sizing            |
| **Execution Agent** | Paper or live order placement, SL/TP monitoring                                          |
| **Evolution Agent** | Generates strategy variants, backtests with vectorbt-style engine, auto-promotes winners |

### UI Tabs

- **Live Dashboard** — Price chart with TA overlays, metrics grid, sentiment meter
- **Agent Minds** — Real-time scrolling agent thought stream with filtering
- **AI Decisions** — Full history with confidence bars, reasoning, and agent contributions
- **Trades & Portfolio** — Equity curve, open positions, trade history, CSV export
- **Evolution Lab** — Trigger evolution, view strategy library, promote candidates
- **Logs** — Full system log with export

### Risk Management

- Configurable max risk per trade (default 2%)
- Global drawdown kill-switch (default 10%)
- Confidence threshold (only executes ≥55% confidence)
- No duplicate positions per symbol

## 🔧 Configuration

| Setting                      | Default       | Description                        |
| ---------------------------- | ------------- | ---------------------------------- |
| `LLM_PROVIDER`             | `anthropic` | LLM backend                        |
| `PAPER_TRADING`            | `true`      | Never touches real money when true |
| `USE_TESTNET`              | `true`      | Binance testnet                    |
| `ENGINE_INTERVAL_SECONDS`  | `30`        | How often the trading cycle runs   |
| `EVOLUTION_INTERVAL_HOURS` | `6`         | Auto-evolution frequency           |
| `MAX_RISK_PER_TRADE`       | `0.02`      | 2% equity per trade                |
| `MAX_DRAWDOWN_KILL`        | `0.10`      | Kill switch at 10% drawdown        |

## 📡 API Reference

```
GET  /api/status           Engine state + portfolio snapshot
GET  /api/chart            OHLCV + indicator series for charts
GET  /api/indicators       Latest indicator values + sentiment
GET  /api/decisions        All trading decisions
GET  /api/trades           Open + closed trades + equity curve
GET  /api/strategies       Strategy library
GET  /api/logs             System logs
GET  /api/performance      Aggregate stats

POST /api/engine/start     Start engine (with config body)
POST /api/engine/stop      Stop engine
POST /api/engine/evolve    Trigger evolution cycle
POST /api/strategies/{id}/promote
POST /api/trades/{id}/close

WS   /ws                   Real-time event stream
```

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It is not financial advice. Always test with paper trading before any real capital. Past backtest performance does not guarantee future results.
