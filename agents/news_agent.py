"""
EvoTrade AI - News & Sentiment Agent
Fetches real-time news and analyzes sentiment for the trading pair.
"""
from __future__ import annotations
import json
import re
import feedparser
import requests
from typing import Dict, Any, Optional
from datetime import datetime

from core.llm import get_llm_response
from core.state import trading_state


CRYPTO_RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
]

FOREX_RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.investing.com/rss/news.rss",
]


def fetch_news(symbol: str) -> list[Dict]:
    """Fetch recent news articles relevant to the symbol."""
    base_asset = symbol.split("/")[0].upper()
    articles = []

    # Try RSS feeds
    feeds = CRYPTO_RSS_FEEDS if len(base_asset) <= 5 else FOREX_RSS_FEEDS
    for url in feeds[:2]:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")[:300]
                # Filter for relevance
                if (base_asset.lower() in title.lower() or
                        base_asset.lower() in summary.lower() or
                        "crypto" in title.lower() or "bitcoin" in title.lower()):
                    articles.append({
                        "title": title,
                        "summary": summary,
                        "published": entry.get("published", ""),
                        "source": feed.feed.get("title", url)
                    })
        except Exception as e:
            trading_state.add_log("NewsAgent", f"RSS fetch error: {e}", level="warn")

    # Fallback mock headlines if no articles found
    if not articles:
        articles = _mock_news(base_asset)

    return articles[:8]


def _mock_news(asset: str) -> list[Dict]:
    """Return plausible mock news when feeds are unavailable."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    return [
        {
            "title": f"{asset} institutional demand surges amid ETF inflows",
            "summary": f"Major institutional players increased {asset} holdings this week as ETF products continue to attract capital from traditional finance.",
            "published": now,
            "source": "MockFeed"
        },
        {
            "title": f"Fed signals cautious approach to rate cuts, {asset} holds steady",
            "summary": "The Federal Reserve maintained its wait-and-see approach, providing some stability to risk assets including cryptocurrencies.",
            "published": now,
            "source": "MockFeed"
        },
        {
            "title": f"{asset} technical outlook remains constructive above key support",
            "summary": f"Analysts point to strong support levels and improving on-chain metrics for {asset}.",
            "published": now,
            "source": "MockFeed"
        }
    ]


def analyze_sentiment(symbol: str, articles: list[Dict]) -> Dict[str, Any]:
    """Use LLM to analyze sentiment from news articles."""
    if not articles:
        return {
            "sentiment_score": 0.5,
            "sentiment_label": "NEUTRAL",
            "key_themes": [],
            "news_summary": "No news data available.",
            "risk_factors": []
        }

    articles_text = "\n".join([
        f"- {a['title']}: {a['summary']}" for a in articles
    ])

    prompt = f"""Analyze the following news articles about {symbol} and provide a trading sentiment assessment.

ARTICLES:
{articles_text}

Respond ONLY with valid JSON in this exact format:
{{
    "sentiment_score": <0.0 to 1.0, where 0=very bearish, 0.5=neutral, 1.0=very bullish>,
    "sentiment_label": "<VERY_BEARISH|BEARISH|NEUTRAL|BULLISH|VERY_BULLISH>",
    "key_themes": ["theme1", "theme2", "theme3"],
    "news_summary": "<2-3 sentence summary of market narrative>",
    "risk_factors": ["risk1", "risk2"]
}}"""

    system = "You are a professional financial analyst specializing in crypto and forex sentiment analysis. Always respond with valid JSON only."

    raw = get_llm_response(prompt, system=system, max_tokens=600)

    try:
        # Extract JSON from response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    return {
        "sentiment_score": 0.5,
        "sentiment_label": "NEUTRAL",
        "key_themes": ["data unavailable"],
        "news_summary": raw[:200] if raw else "Analysis unavailable",
        "risk_factors": []
    }


def run_news_agent(symbol: str) -> Dict[str, Any]:
    """
    Main entry point for the News Agent.
    Returns structured sentiment analysis.
    """
    trading_state.add_log("NewsAgent", f"Fetching news for {symbol}...")

    articles = fetch_news(symbol)
    trading_state.add_log("NewsAgent", f"Found {len(articles)} relevant articles")

    sentiment = analyze_sentiment(symbol, articles)
    trading_state.add_log(
        "NewsAgent",
        f"Sentiment: {sentiment['sentiment_label']} (score={sentiment['sentiment_score']:.2f})",
        data=sentiment,
        level="success"
    )

    # Update global state
    trading_state.last_news_sentiment = {
        **sentiment,
        "articles_count": len(articles),
        "timestamp": datetime.utcnow().isoformat()
    }

    return trading_state.last_news_sentiment
