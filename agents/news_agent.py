"""
EvoTrade AI - News & Sentiment Agent
Fetches real-time news and analyzes sentiment for the trading pair.
"""
from __future__ import annotations
import json
import re
import feedparser
from typing import Dict, Any
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

    # Try all RSS feeds before giving up (no mock fallback).
    feeds = CRYPTO_RSS_FEEDS if len(base_asset) <= 5 else FOREX_RSS_FEEDS
    for url in feeds:
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

    if not articles:
        raise RuntimeError(
            f"No live news articles found for {base_asset} — cannot compute sentiment"
        )

    return articles[:8]


def analyze_sentiment(symbol: str, articles: list[Dict]) -> Dict[str, Any]:
    """Use LLM to analyze sentiment from news articles. Requires real articles."""
    if not articles:
        raise RuntimeError(f"No news articles for {symbol} — sentiment unavailable")

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
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            if "sentiment_score" not in result or "sentiment_label" not in result:
                raise ValueError("Missing sentiment_score or sentiment_label")
            return result
    except Exception as e:
        raise RuntimeError(f"Failed to parse sentiment from LLM: {e}") from e

    raise RuntimeError("LLM returned no valid sentiment JSON")


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
