# EvoTrade AI — Data Sources & API Reference

## How the Engine Gets Its Data

EvoTrade AI gets data from **3 distinct sources**. Here is exactly what each one
does, whether it's free, and where to get any required keys.

---

## 1. Market Data (OHLCV — Price Candles)

**What it is:** Open/High/Low/Close/Volume candlestick data used for all
technical indicators and charting.

**Library used:** `ccxt` (CryptoCurrency eXchange Trading library)

**Source:** Binance public REST API

| Detail | Info |
|--------|------|
| Endpoint | `https://api.binance.com/api/v3/klines` |
| API Key needed? | **NO** — market data is completely public |
| Rate limit | 1200 requests/minute (very generous) |
| Data available | 1m, 5m, 15m, 1h, 4h, 1d, 1w candles |
| History depth | Up to 1000 candles per request (~41 days on 1h) |
| Symbols | BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, and 300+ more |

**What happens without a Binance key:**
The engine automatically falls back to **synthetic (simulated) OHLCV data**
generated with a realistic random walk. All indicators, charts, and backtests
work perfectly in this mode — ideal for development and testing.

**To use real data without a key:**
```python
# In agents/data_agent.py, ccxt fetches public data automatically:
import ccxt
exchange = ccxt.binance()  # no keys needed for public endpoints
ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1h", limit=200)
```

**To place live/testnet orders (keys needed):**
1. Go to https://www.binance.com/en/my/settings/api-management
2. Create an API key (enable "Spot & Margin Trading")
3. For testnet: https://testnet.binance.vision/ → Log in with GitHub → Create key
4. Add to `.env`: `BINANCE_API_KEY=...` and `BINANCE_SECRET=...`

---

## 2. News & Sentiment Data

**What it is:** Recent news headlines analyzed by the LLM to gauge market
sentiment (bullish/bearish score 0–1).

**Library used:** `feedparser` (RSS parsing)

**Sources used (free, no key needed):**

| Feed | URL | Coverage |
|------|-----|----------|
| CoinTelegraph | https://cointelegraph.com/rss | Crypto news |
| Decrypt | https://decrypt.co/feed | Crypto/Web3 |
| CoinDesk | https://www.coindesk.com/arc/outboundfeeds/rss/ | Crypto markets |
| Reuters Business | https://feeds.reuters.com/reuters/businessNews | Macro/Finance |

**How it works:**
1. RSS feeds are fetched (no API key, completely free)
2. Articles are filtered by relevance to the trading symbol
3. The LLM analyzes headlines and summaries to produce a sentiment score
4. Fallback: if all RSS feeds fail, mock headlines are used

**Optional upgrade — Tavily AI Search (better news quality):**
- Sign up: https://tavily.com → free tier: 1000 searches/month
- Add to `.env`: `TAVILY_API_KEY=tvly-...`
- The news agent will automatically use it if the key is present

---

## 3. AI / LLM — Decision Making & Strategy Generation

**What it is:** The "brain" of the system. The LLM reads market data + news
and decides BUY/SELL/HOLD, generates new trading strategies, and explains
its reasoning in plain English.

### Option A: Google Gemini ← **Recommended for free usage**

| Detail | Info |
|--------|------|
| Get key | https://aistudio.google.com/app/apikey |
| Free tier | **15 requests/min, 1 million tokens/day** |
| Best model | `gemini-2.0-flash` (fast + free) |
| Paid model | `gemini-1.5-pro` (more capable) |
| Cost (paid) | ~$0.075 per 1M input tokens |

```env
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-2.0-flash
LLM_PROVIDER=gemini
```

### Option B: Anthropic Claude

| Detail | Info |
|--------|------|
| Get key | https://console.anthropic.com/ → Settings → API Keys |
| Free tier | No free tier — pay-as-you-go |
| Best model | `claude-sonnet-4-20250514` |
| Cost | ~$3 per 1M input tokens |

```env
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-sonnet-4-20250514
LLM_PROVIDER=anthropic
```

### Option C: OpenAI GPT-4o

| Detail | Info |
|--------|------|
| Get key | https://platform.openai.com/api-keys |
| Free tier | $5 free credits for new accounts |
| Model used | `gpt-4o` |
| Cost | ~$5 per 1M input tokens |

```env
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
```

### Option D: Ollama (100% Local, 100% Free)

Run a local LLM — no internet required after setup, completely free, private.

| Detail | Info |
|--------|------|
| Download | https://ollama.com/download |
| Setup | `ollama pull llama3` (downloads ~4GB model) |
| Cost | Free forever |
| GPU recommended | Yes, but CPU works (slower) |

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
LLM_PROVIDER=ollama
```

### Option E: Demo / Mock Mode (No key, no internet)

The engine runs with pre-programmed responses. Perfect for UI testing
and development. Set `LLM_PROVIDER=mock` or just leave all keys empty.

---

## Summary Table

| Data | Source | Key Needed? | Cost | Where to get key |
|------|--------|-------------|------|-----------------|
| Price candles (OHLCV) | Binance public API | ❌ No | Free | — |
| News headlines | RSS feeds | ❌ No | Free | — |
| LLM decisions | Google Gemini | ✅ Yes | **Free** (15 RPM) | aistudio.google.com/app/apikey |
| LLM decisions | Anthropic Claude | ✅ Yes | Paid | console.anthropic.com |
| LLM decisions | OpenAI GPT-4o | ✅ Yes | ~$5 free credits | platform.openai.com/api-keys |
| LLM decisions | Ollama (local) | ❌ No | Free | ollama.com/download |
| Live order placement | Binance API | ✅ Yes | Free | binance.com/en/my/settings/api-management |
| Testnet trading | Binance Testnet | ✅ Yes | Free | testnet.binance.vision |

---

## Quickstart: Fully Free Setup

```bash
# 1. Install
pip install -r requirements.txt

# 2. Get a free Gemini key at: https://aistudio.google.com/app/apikey

# 3. Create .env
echo "GEMINI_API_KEY=AIza_your_key_here" > .env
echo "GEMINI_MODEL=gemini-2.0-flash" >> .env
echo "LLM_PROVIDER=gemini" >> .env
echo "PAPER_TRADING=true" >> .env

# 4. Run
python main.py
# Open http://localhost:8000
```

Total cost: **$0** — Gemini free tier is more than enough for personal trading research.

---

## No Key At All — Demo Mode

Just run without any `.env` file:
```bash
python main.py
```
Everything works — synthetic price data, mock news, mock LLM decisions.
Great for exploring the UI and understanding the system.
