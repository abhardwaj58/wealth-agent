import base64
import json
import math
import re
import time
from datetime import datetime, time as dt_time
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import schedule
import streamlit as st
import yfinance as yf
from anthropic import Anthropic, APIStatusError
from supabase_memory import (
    fetch_recent_runs,
    get_or_create_user_id,
    get_supabase_client_singleton,
    load_saved_portfolio,
    save_portfolio_run,
)


HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-20250514"

# Static Sonnet instructions + JSON schemas (prompt caching: same bytes → cache hit across holdings in one run).
SONNET_STOCK_ANALYSIS_SYSTEM = """You are an equity research assistant analyzing ONE symbol per user message.

INPUT JSON keys you may rely on:
- scoring_weights: decimal weights summing to 1.0 for:
  analyst_consensus, institutional_positioning, forward_revenue_visibility,
  valuation_vs_historical_pe, macro_and_sector_factors, options_and_short_interest
- holding_quantity, data (normalized market/analyst/options fields), institutional_positioning (from yfinance),
- web_research_summaries (list of Haiku summaries with title, url, summary, search_query; trust higher source_weight),

You cannot use web_search or browse; only those JSON keys exist.

Produce strict JSON only (no markdown fences).

OUTPUT SCHEMA:
{
  "bull_case": {
    "reasoning": "<string>",
    "confidence_pct": <0-100>,
    "supporting_institutions": ["<firm or fund names from 13F/analyst/web context>", ...]
  },
  "bear_case": {
    "reasoning": "<string>",
    "confidence_pct": <0-100>,
    "supporting_institutions": [<string>, ...]
  },
  "hold_case": {
    "reasoning": "<string>",
    "confidence_pct": <0-100>,
    "supporting_institutions": [<string>, ...]
  },
  "analyst_score": <0-10>,
  "institutional_score": <0-10>,
  "revenue_score": <0-10>,
  "valuation_score": <0-10>,
  "macro_score": <0-10>,
  "options_score": <0-10>,
  "composite_score_10": <number>,
  "action": "Strong Buy"|"Buy"|"Hold"|"Reduce"|"Sell",
  "thesis": "<optional one-line synthesis>"
}

SCORING:
- Score each *_score from 0–10 independently using data + web_research_summaries (weight credibility via source_weight in summaries).
- compound composite_score_10 EXACTLY as weighted average:
  analyst_consensus*analyst_score + institutional_positioning*institutional_score + forward_revenue_visibility*revenue_score + valuation_vs_historical_pe*valuation_score + macro_and_sector_factors*macro_score + options_and_short_interest*options_score
  (divide implied sum is already weighted since weights sum to 1 — compute sum of weight_i * score_i).

ACTION mapping from composite_score_10:
- Strong Buy: >= 8.5
- Buy: >= 7
- Hold: >= 5
- Reduce: >= 3
- Sell: < 3"""

SONNET_PORTFOLIO_SYSTEM = """You are a portfolio strategist. Each user message contains JSON with a holdings_analysis array (already analyzed positions) and web_research_summaries (Haiku-compressed web digests).

You cannot use web_search or browse; only the JSON keys in each user message are available.

Respond with strict JSON only (no markdown code fences). Use exactly this shape:
{
  "concentration_risks": ["<string>", ...],
  "rebalancing_recommendation": "<string>",
  "new_stock_ideas": [{"ticker": "<string>", "thesis": "<string>"}, ...]
}

Suggest 3 to 5 new stock ideas not already held; ground themes in web_research_summaries when present."""

SONNET_SYSTEM_CACHE_CONTROL = {"type": "ephemeral"}


def _sonnet_system_with_ephemeral_cache(instruction_text: str) -> List[Dict[str, Any]]:
    """Build Messages API ``system`` with prompt caching on static instructions only.

    ``cache_control`` is attached to blocks in the top-level ``system`` parameter—not inside
    ``messages``—so the user JSON payload does not steal the cache breakpoint.
    """
    return [
        {
            "type": "text",
            "text": instruction_text,
            "cache_control": SONNET_SYSTEM_CACHE_CONTROL,
        }
    ]


# On HTTP 429: exponential backoff (60s, 120s, 240s), up to MAX_RETRIES retries per request.
# Applies to every Messages API call (vision extract, Haiku research/summarize, Sonnet calls).
ANTHROPIC_429_RETRY_WAIT_SEC = 60.0
ANTHROPIC_429_MAX_RETRIES = 3  # retries after a 429; total attempts = MAX_RETRIES + 1

# Pause after each per-stock Sonnet analysis before the next holding (not after the last).
PER_STOCK_ANALYSIS_GAP_SEC = 10.0

# Scoring weights keys (decimals sum to 1.0 for Sonnet).
DEFAULT_SCORING_WEIGHTS: Dict[str, float] = {
    "analyst_consensus": 0.25,
    "institutional_positioning": 0.20,
    "forward_revenue_visibility": 0.20,
    "valuation_vs_historical_pe": 0.15,
    "macro_and_sector_factors": 0.10,
    "options_and_short_interest": 0.10,
}

WEIGHT_SLIDER_DEFINITIONS: List[tuple[str, str]] = [
    ("analyst_consensus", "Analyst consensus"),
    ("institutional_positioning", "Institutional positioning"),
    ("forward_revenue_visibility", "Forward revenue visibility"),
    ("valuation_vs_historical_pe", "Valuation vs historical P/E"),
    ("macro_and_sector_factors", "Macro and sector factors"),
    ("options_and_short_interest", "Options and short interest"),
]

HAIKU_PORTFOLIO_WEB_RESEARCH_MAX_TOKENS = 8192
HAIKU_SINGLE_WEB_SEARCH_MAX_TOKENS = 6144
HAIKU_SEARCH_HIT_SUMMARY_MAX_TOKENS = 900
MAX_WEB_SEARCH_HITS_PER_ROUND = 20
MAX_SUMMARIES_PER_TARGETED_QUERY = 5
MAX_PORTFOLIO_SEARCH_HITS_TO_SUMMARIZE = 15

HAIKU_PORTFOLIO_WEB_RESEARCH_USER = """The portfolio currently includes these tickers: {tickers}

Use the web_search tool to run one or more searches for: (1) current market themes useful for rebalancing, (2) diversification and risk, (3) quality companies or sectors to consider. Focus on recent, actionable context.

You do not need to write a long answer—tool results are recorded for the next step."""


def _normalized_hostname(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        if "@" in netloc:
            netloc = netloc.split("@")[-1]
        netloc = netloc.split(":")[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def classify_search_source(url: str) -> tuple[str, float]:
    """Return (tier_label, weight). Blocked sources get weight 0."""
    h = _normalized_hostname(url)
    if not h:
        return "blocked", 0.0
    path_l = (urlparse(url).path or "").lower()

    blocked_hosts = (
        "reddit.com",
        "twitter.com",
        "x.com",
        "t.co",
        "stocktwits.com",
        "discord.com",
        "discord.gg",
        "facebook.com",
        "tiktok.com",
    )
    if any(h == b or h.endswith("." + b) for b in blocked_hosts):
        return "blocked", 0.0
    if "forum." in h or h.startswith("forum.") or "blogspot." in h or ".blog." in h:
        return "blocked", 0.0
    if "/forum/" in path_l or path_l.rstrip("/").endswith("/blog") or "/blogs/" in path_l:
        return "blocked", 0.0

    tier_1_hosts = frozenset({"sec.gov", "edgar.sec.gov", "seekingalpha.com"})
    tier_2_hosts = frozenset({"bloomberg.com", "wsj.com", "ft.com", "reuters.com"})
    tier_3_hosts = frozenset({"cnbc.com", "marketwatch.com", "finance.yahoo.com", "yahoo.com"})

    def _ends(host: str, root: str) -> bool:
        return host == root or host.endswith("." + root)

    if _ends(h, "sec.gov") or _ends(h, "edgar.sec.gov") or h.endswith("seekingalpha.com"):
        return "tier_1", 1.0
    if h in tier_1_hosts:
        return "tier_1", 1.0
    if any(h == x or h.endswith("." + x) for x in tier_2_hosts):
        return "tier_2", 0.8
    if any(h == x or h.endswith("." + x) for x in tier_3_hosts):
        return "tier_3", 0.6

    return "tier_other", 0.55


def stock_targeted_search_queries(ticker: str, sector: str | None) -> List[str]:
    sect = sector or "Unknown sector"
    t = ticker.strip().upper()
    return [
        f"{t} SEC EDGAR 10-K 10-Q latest filing",
        f"{t} earnings call transcript Q1 2026",
        f"{t} 8-K material events 2026",
        f"{t} 13F institutional holdings BlackRock Vanguard Berkshire 2026",
        f"{t} Goldman Sachs Morgan Stanley analyst rating 2026",
        f"{t} JPMorgan BofA Barclays price target 2026",
        f"{t} Bloomberg revenue outlook 2026",
        f"{t} Wall Street Journal business strategy 2026",
        f"{t} Financial Times competitive landscape 2026",
        f"{t} Reuters earnings guidance 2026",
        f"{t} insider buying selling SEC Form 4 2026",
        f"{t} sector macro outlook {sect} 2026",
    ]


def extract_web_search_hits_from_response(response: Any) -> List[Dict[str, Any]]:
    """Pull individual web_search_result records from a completed Messages response."""
    hits: List[Dict[str, Any]] = []
    for block in response.content:
        if getattr(block, "type", None) != "web_search_tool_result":
            continue
        inner = getattr(block, "content", None)
        if inner is not None and getattr(inner, "type", None) == "web_search_tool_result_error":
            continue
        if not isinstance(inner, list):
            continue
        for item in inner:
            if getattr(item, "type", None) != "web_search_result":
                continue
            hits.append(
                {
                    "title": getattr(item, "title", "") or "",
                    "url": getattr(item, "url", "") or "",
                    "page_age": getattr(item, "page_age", None),
                    "encrypted_content": getattr(item, "encrypted_content", "") or "",
                }
            )
    return hits


def _optional_text_from_encrypted(encrypted: str, max_len: int = 2000) -> str:
    """If provider payload looks like base64 text, decode for summarizer context."""
    if not encrypted or len(encrypted) < 24:
        return ""
    try:
        raw = base64.b64decode(encrypted, validate=False)
        t = raw.decode("utf-8", errors="ignore")
        if len(t) > 40 and any(c.isalnum() for c in t):
            return t[:max_len]
    except Exception:
        pass
    return (encrypted[:max_len] + "…") if len(encrypted) > max_len else encrypted


def _format_search_hit_for_summarizer(hit: Dict[str, Any]) -> str:
    lines = [
        f"Title: {hit.get('title', '')}",
        f"URL: {hit.get('url', '')}",
    ]
    pa = hit.get("page_age")
    if pa:
        lines.append(f"Page age: {pa}")
    dec = _optional_text_from_encrypted(hit.get("encrypted_content") or "")
    if dec:
        lines.append(f"Page text (excerpt if available): {dec}")
    return "\n".join(lines)


def haiku_summarize_search_hit(
    client: Anthropic,
    *,
    research_context: str,
    hit: Dict[str, Any],
    source_weight: float,
    sentences_min: int = 8,
    sentences_max: int = 10,
) -> str:
    """Summarize one search hit (Haiku, no tools), with tier-weight dependent prompting."""
    # Tier-focused guidance is driven by the classifier's `source_weight`.
    # NOTE: sentence length is controlled by sentences_min/sentences_max supplied by the caller.
    if source_weight >= 0.95:
        tier_focus = (
            "Tier 1 (highest credibility). Extract specific numbers, dates, filed metrics, and any forward guidance "
            "verbatim where present. If page text is scarce, rely on the title/URL/domain but do not invent details."
        )
    elif source_weight >= 0.75:
        tier_focus = (
            "Tier 2 (high credibility). Focus on market signals, analyst commentary, and business developments. "
            "Extract key takeaways and any cited figures/dates if explicitly present."
        )
    elif source_weight >= 0.59:
        tier_focus = (
            "Tier 3 (moderate credibility). Keep it high level. Use only what is clearly supported by the result text/title; "
            "avoid heavy quoting or detailed filings."
        )
    else:
        tier_focus = (
            "Other/unclassified source. Keep the summary concise and high level, focusing on general themes rather than specifics."
        )
    user_msg = (
        f"Research context: {research_context}\n\n"
        f"Single web search result to summarize:\n{_format_search_hit_for_summarizer(hit)}\n\n"
        f"Source credibility weight: {source_weight}\n"
        f"Target summary length: {sentences_min} to {sentences_max} complete sentences\n\n"
        f"{tier_focus}\n\n"
        f"Write between {sentences_min} and {sentences_max} complete sentences in plain prose "
        "(no bullets, no JSON). Explain what the source is about, why it may matter to investors, "
        "and any limitations if page text was scarce (infer carefully from title/URL/domain when needed)."
    )
    r = anthropic_messages_create(
        client,
        model=HAIKU_MODEL,
        max_tokens=HAIKU_SEARCH_HIT_SUMMARY_MAX_TOKENS,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(c.text for c in r.content if c.type == "text").strip()


def haiku_web_search_single_query(client: Anthropic, query: str) -> List[Dict[str, Any]]:
    """One Haiku call with web_search for a fixed query."""
    research = anthropic_messages_create(
        client,
        model=HAIKU_MODEL,
        max_tokens=HAIKU_SINGLE_WEB_SEARCH_MAX_TOKENS,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[
            {
                "role": "user",
                "content": (
                    "Use the web_search tool to run searches for exactly this topic. "
                    "When you have retrieval results in context, reply with one short acknowledging sentence.\n\n"
                    f"{query}"
                ),
            }
        ],
    )
    hits = extract_web_search_hits_from_response(research)
    return hits[:MAX_WEB_SEARCH_HITS_PER_ROUND]


def sentence_targets_for_source_weight(source_weight: float) -> tuple[int, int]:
    """Map classifier weight → (sentences_min, sentences_max)."""
    if source_weight >= 0.95:
        return (10, 12)
    if source_weight >= 0.75:
        return (7, 8)
    if source_weight >= 0.59:
        return (4, 5)
    return (3, 4)


def build_web_research_summaries_for_stock(
    client: Anthropic, ticker: str, sector: str | None
) -> List[Dict[str, Any]]:
    """12 sequential targeted Haiku+web searches; tier-weighted per-hit summarization."""
    out: List[Dict[str, Any]] = []
    ctx = f"Targeted diligence for {ticker.upper()}"
    for query in stock_targeted_search_queries(ticker, sector):
        hits = haiku_web_search_single_query(client, query)
        tier3_count = 0
        other_count = 0
        for h in hits:
            url = h.get("url") or ""
            tier, weight = classify_search_source(url)
            if tier == "blocked" or weight <= 0:
                continue

            # Per search-round policy:
            # - Tier 1/2: summarize all hits found (no cap).
            # - Tier 3 and "other": summarize only the first 3 hits in that tier.
            if tier == "tier_3":
                if tier3_count >= 3:
                    continue
            elif tier == "tier_other":
                if other_count >= 3:
                    continue
            # tier_1 and tier_2 have no cap.

            sentences_min, sentences_max = sentence_targets_for_source_weight(weight)
            summary = haiku_summarize_search_hit(
                client,
                research_context=ctx,
                hit=h,
                source_weight=weight,
                sentences_min=sentences_min,
                sentences_max=sentences_max,
            )
            out.append(
                {
                    "search_query": query,
                    "title": h.get("title"),
                    "url": url,
                    "page_age": h.get("page_age"),
                    "summary": summary,
                    "source_tier": tier,
                    "source_weight": weight,
                }
            )
            if tier == "tier_3":
                tier3_count += 1
            elif tier == "tier_other":
                other_count += 1
    return out


def build_web_research_summaries_for_portfolio(
    client: Anthropic, current_tickers: List[str]
) -> List[Dict[str, Any]]:
    tickers_str = ", ".join(current_tickers) if current_tickers else "(none)"
    research = anthropic_messages_create(
        client,
        model=HAIKU_MODEL,
        max_tokens=HAIKU_PORTFOLIO_WEB_RESEARCH_MAX_TOKENS,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[
            {
                "role": "user",
                "content": HAIKU_PORTFOLIO_WEB_RESEARCH_USER.format(tickers=tickers_str),
            }
        ],
    )
    hits = extract_web_search_hits_from_response(research)[:MAX_PORTFOLIO_SEARCH_HITS_TO_SUMMARIZE]
    out: List[Dict[str, Any]] = []
    ctx = f"Portfolio-level research; current holdings: {tickers_str}"
    for h in hits:
        url = h.get("url") or ""
        tier, weight = classify_search_source(url)
        if tier == "blocked" or weight <= 0:
            continue
        sentences_min, sentences_max = sentence_targets_for_source_weight(weight)
        summary = haiku_summarize_search_hit(
            client,
            research_context=ctx,
            hit=h,
            source_weight=weight,
            sentences_min=sentences_min,
            sentences_max=sentences_max,
        )
        out.append(
            {
                "title": h.get("title"),
                "url": h.get("url"),
                "summary": summary,
            }
        )
    return out


def _claude_debug_payload_enabled() -> bool:
    try:
        return bool(st.session_state.get("debug_claude_payload"))
    except Exception:
        return False


def anthropic_messages_create(client: Anthropic, **kwargs: Any) -> Any:
    """Call Messages API. On 429: exponential backoff from RETRY_WAIT_SEC, up to MAX_RETRIES retries."""
    if _claude_debug_payload_enabled():
        payload_repr = json.dumps(kwargs, default=str, sort_keys=True)
        model = kwargs.get("model", "?")
        print(
            f"[Claude API debug] model={model} payload_character_count={len(payload_repr)}",
            flush=True,
        )
    max_attempts = ANTHROPIC_429_MAX_RETRIES + 1
    for attempt in range(max_attempts):
        try:
            return client.messages.create(**kwargs)
        except APIStatusError as e:
            if getattr(e, "status_code", None) != 429:
                raise
            if attempt == max_attempts - 1:
                raise
            time.sleep(ANTHROPIC_429_RETRY_WAIT_SEC * (2**attempt))
    assert False, "anthropic_messages_create exhausted without return"


SCORE_WEIGHT_PAIRS: List[Tuple[str, str]] = [
    ("analyst_consensus", "analyst_score"),
    ("institutional_positioning", "institutional_score"),
    ("forward_revenue_visibility", "revenue_score"),
    ("valuation_vs_historical_pe", "valuation_score"),
    ("macro_and_sector_factors", "macro_score"),
    ("options_and_short_interest", "options_score"),
]


def composite_to_action(score: float) -> str:
    s = float(score)
    if s >= 8.5:
        return "Strong Buy"
    if s >= 7.0:
        return "Buy"
    if s >= 5.0:
        return "Hold"
    if s >= 3.0:
        return "Reduce"
    return "Sell"


def reconcile_stock_analysis(parsed: Dict[str, Any], scoring_weights: Dict[str, float]) -> None:
    """Clamp factor scores, recompute composite from session weights, set action."""
    mu = 5.0
    sub = 0.0
    for w_key, sk in SCORE_WEIGHT_PAIRS:
        try:
            x = float(parsed.get(sk, mu))
        except (TypeError, ValueError):
            x = mu
        x = max(0.0, min(10.0, x))
        parsed[sk] = x
        wf = float(scoring_weights.get(w_key, 0.0))
        sub += wf * x
    comp = max(0.0, min(10.0, sub))
    parsed["composite_score_10"] = round(comp, 4)
    parsed["action"] = composite_to_action(comp)


def grouped_research_by_tier(
    summaries: List[Dict[str, Any]],
) -> List[tuple[str, List[Dict[str, Any]]]]:
    """Order: tier 1 → 2 → 3 → other; blocked omitted."""
    order = ("tier_1", "tier_2", "tier_3", "tier_other")
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in summaries or []:
        tier = str(s.get("source_tier") or "tier_other")
        if tier == "blocked":
            continue
        buckets[tier].append(s)
    return [(t, buckets[t]) for t in order if buckets.get(t)]

CRYPTO_BASE_SYMBOLS_USD = frozenset({
    "1INCH", "AAVE", "ADA", "ALGO", "APE", "APT", "ARB", "ATOM", "AVAX", "AXS",
    "BCH", "BLUR", "BNB", "BONK", "BTC", "CELO", "CHZ", "COMP", "CRV", "CVX",
    "DAI", "DOGE", "DOT", "DYDX", "EGLD", "ENJ", "EOS", "ETC", "ETH", "FET",
    "FIL", "FLOW", "FTM", "GALA", "GMT", "GRT", "HBAR", "ICP", "IMX", "INJ",
    "JUP", "KAVA", "LDO", "LINK", "LRC", "LTC", "MANA", "MASK", "MINA", "MKR",
    "NEAR", "OP", "PEPE", "POL", "MATIC", "QNT", "RENDER", "RNDR", "RUNE", "SAND",
    "SEI", "SHIB", "SNX", "SOL", "STX", "SUI", "THETA", "TIA", "TON", "TRX",
    "UNI", "VET", "WAVES", "WLD", "XLM", "XMR", "XRP", "XTZ", "YFI", "ZEC", "ZIL",
})


def yfinance_lookup_symbol(display_ticker: str) -> str:
    """Map UI ticker to Yahoo Finance symbol; append -USD for known crypto bases without a suffix."""
    t = display_ticker.strip().upper()
    if not t or "-" in t:
        return t
    if t in CRYPTO_BASE_SYMBOLS_USD:
        return f"{t}-USD"
    return t


def extract_json_from_text(text: str) -> Any:
    # This function attempts to extract and parse JSON from a text string, even if the JSON is wrapped in Markdown code blocks.

    # First, remove any leading/trailing whitespace.
    text = text.strip()
    
    # If the text starts with triple backticks (optional "json" annotation), strip off any leading "```json" or "```", and any trailing "```".
    if text.startswith("```"):
        # Remove the leading code block marker (with or without 'json').
        text = re.sub(r"^```(?:json)?", "", text).strip()
        # Remove the trailing code block marker.
        text = re.sub(r"```$", "", text).strip()
    
    # Now, try to parse the result directly as JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If that fails, try to find the first JSON object or array in the text using a regular expression.
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if not match:
            # If no JSON fragment is found, re-raise the original exception.
            raise
        # If a JSON object/array is found, parse and return that.
        return json.loads(match.group(1))


def get_claude_client() -> Anthropic:
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or st.session_state.get("anthropic_key")
    if not api_key:
        api_key = st.text_input("Enter your Anthropic API key", type="password")
        if api_key:
            st.session_state["anthropic_key"] = api_key
    if not api_key:
        st.stop()
    return Anthropic(api_key=api_key)


def image_to_base64(uploaded_file) -> str:
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")


def extract_holdings_from_image(client: Anthropic, image_b64: str, media_type: str) -> List[Dict[str, Any]]:
    prompt = (
        "You are reading a brokerage portfolio screenshot. Extract only stock/ETF holdings. "
        "Return strict JSON array where each item is "
        '{"ticker":"AAPL","quantity":10}. '
        "Rules: uppercase ticker, numeric quantity, ignore cash/headers/rows without ticker."
    )
    response = anthropic_messages_create(
        client,
        model=HAIKU_MODEL,
        max_tokens=1200,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                ],
            }
        ],
    )
    text = "".join([c.text for c in response.content if c.type == "text"])
    parsed = extract_json_from_text(text)
    if not isinstance(parsed, list):
        return []
    cleaned = []
    for row in parsed:
        ticker = str(row.get("ticker", "")).upper().strip()
        qty = row.get("quantity", 0)
        if ticker and isinstance(qty, (int, float)):
            cleaned.append({"ticker": ticker, "quantity": float(qty)})
    return cleaned


def parse_manual_holdings(raw: str) -> List[Dict[str, Any]]:
    holdings = []
    pairs = [p.strip() for p in raw.split(",") if p.strip()]
    for pair in pairs:
        if ":" in pair:
            ticker, qty = pair.split(":", 1)
            ticker = ticker.strip().upper()
            try:
                quantity = float(qty.strip())
            except ValueError:
                continue
            if ticker and quantity > 0:
                holdings.append({"ticker": ticker, "quantity": quantity})
    return holdings


def format_holdings_as_manual_input(holdings: List[Dict[str, Any]]) -> str:
    """Serialize holdings to TICKER:QTY, ... for the manual input field."""
    parts: List[str] = []
    for h in holdings:
        t = str(h.get("ticker", "")).strip().upper()
        q = h.get("quantity", 0)
        if not t or not isinstance(q, (int, float)) or float(q) <= 0:
            continue
        qf = float(q)
        parts.append(f"{t}:{int(qf)}" if qf == int(qf) else f"{t}:{qf}")
    return ", ".join(parts)


def extract_institutional_snapshot(tk: yf.Ticker) -> Dict[str, Any]:
    """Top institutional holders (yfinance) + major_holders table; no API cost."""
    top: List[Dict[str, Any]] = []
    major: List[Dict[str, Any]] = []
    try:
        ih = tk.institutional_holders
    except Exception:
        ih = None
    try:
        mh = tk.major_holders
    except Exception:
        mh = None

    if isinstance(ih, pd.DataFrame) and not ih.empty:
        # yfinance returns columns like: Holder, Shares, Date Reported, % Out, Value
        # Earlier logic guessed columns and could accidentally pick the reset index
        # (showing row numbers as "holder" names). Here we read the intended columns.
        df = ih.head(10).copy()

        def _norm_col(x: Any) -> str:
            return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())

        cols_norm = {_norm_col(c): c for c in df.columns}
        holder_col = cols_norm.get(_norm_col("Holder"))
        shares_col = cols_norm.get(_norm_col("Shares"))

        # "% Out" normalizes to something like "out"
        pct_col = None
        for c in df.columns:
            cn = _norm_col(c)
            if cn in {"out", "pctout", "percentout"} or cn.endswith("out"):
                pct_col = c
                break

        for _, row in df.iterrows():
            holder_val = row[holder_col] if holder_col in df.columns else None
            shares_val = row[shares_col] if shares_col in df.columns else None
            pct_val = row[pct_col] if pct_col in df.columns else None

            name = "Unknown"
            if holder_val is not None and not (isinstance(holder_val, float) and pd.isna(holder_val)):
                name = str(holder_val).strip() or "Unknown"

            shares: int | None = None
            if shares_val is not None and not (isinstance(shares_val, float) and pd.isna(shares_val)):
                try:
                    sf = float(shares_val)
                    shares = int(sf) if sf == int(sf) else int(round(sf))
                except (TypeError, ValueError):
                    shares = None

            pct: float | None = None
            if pct_val is not None and not (isinstance(pct_val, float) and pd.isna(pct_val)):
                try:
                    pf = float(pct_val)
                    pct = pf * 100.0 if pf <= 1.0 else pf
                    pct = round(pct, 4)
                except (TypeError, ValueError):
                    pct = None

            top.append({"name": name, "shares": shares, "pct": pct})

    if isinstance(mh, pd.DataFrame) and not mh.empty:
        mr = mh.reset_index(drop=False)
        for _, row in mr.iterrows():
            major.append({str(c): ("" if pd.isna(row[c]) else str(row[c])) for c in mr.columns})

    return {"top_institutional_holders": top, "major_holders": major}


def get_stock_raw_data(yf_symbol: str, *, display_ticker: str | None = None) -> Dict[str, Any]:
    """
    Retrieves comprehensive raw data for a given Yahoo Finance instrument.

    Use ``yf_symbol`` for yfinance (e.g. ``BTC-USD``); ``display_ticker`` is what the UI shows (e.g. ``BTC``).

    All information gathered in this function is sourced from the `yfinance` (Yahoo Finance) Python package,
    which acts as a web scraper and data aggregator for Yahoo Finance (finance.yahoo.com).

    Information sources:
        - Stock metadata, historical pricing, and financials: Yahoo Finance via yfinance's Ticker.info, Ticker.history(), and related attributes.
        - Analyst recommendations and upgrades/downgrades: Yahoo Finance analyst data via yfinance's Ticker.recommendations and Ticker.upgrades_downgrades attributes.
        - Options statistics (open interest, implied volatility): Yahoo Finance options chain data scraped through Ticker.options and Ticker.option_chain().
        - All data ultimately comes from finance.yahoo.com, as provided by the public yfinance library.

    Args:
        yf_symbol: Symbol passed to ``yf.Ticker`` (e.g. ``BTC-USD``, ``AAPL``).
        display_ticker: Label stored in payloads and shown in the UI; defaults to ``yf_symbol``.

    Returns:
        Dict[str, Any]: Raw data for ``normalize_stock_data``; ``ticker`` field is ``display_ticker``.
    """
    label = (display_ticker.strip().upper() if display_ticker else None) or yf_symbol
    # Instantiates the Yahoo Finance scraper/client object for this instrument
    tk = yf.Ticker(yf_symbol)

    # General metadata pulled from Yahoo Finance for this ticker
    info = tk.info or {}

    # Past 1 year and 5 years daily pricing (from Yahoo Finance historical quotes)
    hist_1y = tk.history(period="1y", auto_adjust=False)
    hist_5y = tk.history(period="5y", auto_adjust=False)

    # Analyst recommendations and upgrade/downgrade history (from Yahoo Finance's analysis tab)
    recommendations = tk.recommendations
    upgrades = tk.upgrades_downgrades
    analyst_price_targets = getattr(tk, "analyst_price_targets", None)

    # Option expiration dates (public Yahoo Finance options listing)
    options_dates = tk.options

    # Pull selected statistics from nearest options expiration chain (Yahoo Finance option chain)
    opt_snapshot = {}
    if options_dates:
        nearest_expiry = options_dates[0]
        chain = tk.option_chain(nearest_expiry)
        call_oi = float(chain.calls["openInterest"].fillna(0).sum()) if not chain.calls.empty else 0.0
        put_oi = float(chain.puts["openInterest"].fillna(0).sum()) if not chain.puts.empty else 0.0
        call_put_ratio = (call_oi / put_oi) if put_oi > 0 else None
        call_iv = float(chain.calls["impliedVolatility"].dropna().mean()) if not chain.calls.empty else None
        put_iv = float(chain.puts["impliedVolatility"].dropna().mean()) if not chain.puts.empty else None
        opt_snapshot = {
            "nearest_expiry": nearest_expiry,
            "call_open_interest": call_oi,
            "put_open_interest": put_oi,
            "call_put_oi_ratio": call_put_ratio,
            "avg_call_iv": call_iv,
            "avg_put_iv": put_iv,
        }

    # Summarize analyst recommendations (Yahoo Finance analyst data via yfinance)
    rec_summary = {"buy": 0, "hold": 0, "sell": 0, "sources": []}
    if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
        latest = recommendations.tail(60).copy()
        for _, row in latest.iterrows():
            # yfinance schema varies by version: "To Grade" vs "toGrade"
            raw_grade = row.get("To Grade")
            if raw_grade is None or (isinstance(raw_grade, float) and pd.isna(raw_grade)):
                raw_grade = row.get("toGrade")
            to_grade = str(raw_grade or "").lower()
            firm = str(row.get("Firm", "")).strip()
            if any(x in to_grade for x in ["buy", "overweight", "outperform", "strong buy"]):
                rec_summary["buy"] += 1
            elif any(x in to_grade for x in ["hold", "neutral", "market perform", "equal weight"]):
                rec_summary["hold"] += 1
            elif any(x in to_grade for x in ["sell", "underperform", "underweight"]):
                rec_summary["sell"] += 1
            if firm:
                rec_summary["sources"].append(firm)
    if isinstance(upgrades, pd.DataFrame) and not upgrades.empty:
        if "Firm" in upgrades.columns:
            rec_summary["sources"].extend(upgrades["Firm"].dropna().astype(str).tolist())

    # Fallback: use analyst_price_targets when recommendation grades are unavailable/empty.
    # Some yfinance versions expose sentiment-style counts here (buy/hold/sell, strongBuy/strongSell).
    if (rec_summary["buy"] + rec_summary["hold"] + rec_summary["sell"]) == 0 and isinstance(
        analyst_price_targets, dict
    ):
        def _int_from_targets(key: str) -> int:
            try:
                v = analyst_price_targets.get(key)
                return int(float(v)) if v is not None else 0
            except (TypeError, ValueError):
                return 0

        buy_fallback = _int_from_targets("buy") + _int_from_targets("strongBuy")
        hold_fallback = _int_from_targets("hold")
        sell_fallback = _int_from_targets("sell") + _int_from_targets("strongSell")
        if buy_fallback or hold_fallback or sell_fallback:
            rec_summary["buy"] = buy_fallback
            rec_summary["hold"] = hold_fallback
            rec_summary["sell"] = sell_fallback
            rec_summary["sources"].append("analyst_price_targets")

    rec_summary["sources"] = sorted(list(set(rec_summary["sources"])))[:12]

    income_stmt = tk.income_stmt
    yearly_pe_points = []
    if isinstance(income_stmt, pd.DataFrame) and not income_stmt.empty and not hist_5y.empty:
        eps_row_name = None
        for candidate in ["Diluted EPS", "Basic EPS"]:
            if candidate in income_stmt.index:
                eps_row_name = candidate
                break
        if eps_row_name:
            eps_series = income_stmt.loc[eps_row_name].dropna()
            for col, eps in eps_series.items():
                year = pd.Timestamp(col).year
                year_prices = hist_5y[hist_5y.index.year == year]
                if year_prices.empty:
                    continue
                avg_close = float(year_prices["Close"].mean())
                if eps and not math.isclose(float(eps), 0.0):
                    yearly_pe_points.append({"year": year, "pe": round(avg_close / float(eps), 2)})
    yearly_pe_points = sorted(yearly_pe_points, key=lambda x: x["year"])

    return {
        "ticker": label,
        "info": {
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "short_percent_of_float": info.get("shortPercentOfFloat"),
            "float_shares": info.get("floatShares"),
        },
        "history_1y_close": (
            hist_1y["Close"].dropna().tail(252).reset_index().rename(columns={"Date": "date", "Close": "close"}).to_dict("records")
            if not hist_1y.empty
            else []
        ),
        "yearly_pe_points": yearly_pe_points,
        "recommendation_summary_raw": rec_summary,
        "options_snapshot_raw": opt_snapshot,
        "institutional_snapshot_raw": extract_institutional_snapshot(tk),
    }


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fmt_two_dec(value: Any, *, prefix: str = "", suffix: str = "") -> str:
    """Format numeric value with at most 2 decimal places; non-numeric becomes em dash."""
    if value is None:
        return "—"
    try:
        x = float(value)
        if math.isnan(x):
            return "—"
        return f"{prefix}{x:.2f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def _iv_as_decimal(iv: float | None) -> float | None:
    """Normalize IV to 0–1 style for threshold labels (yfinance usually uses decimals)."""
    if iv is None:
        return None
    if iv > 1.0 and iv <= 100.0:
        return iv / 100.0
    return iv


def interpret_call_put_oi_ratio(ratio: float | None) -> str:
    if ratio is None or (isinstance(ratio, float) and math.isnan(ratio)):
        return "No options open-interest ratio available (missing chain or data)."
    if ratio > 1.5:
        return "Bullish options positioning"
    if ratio < 0.7:
        return "Bearish options positioning"
    return "Neutral options positioning"


def interpret_implied_volatility(iv: float | None) -> str:
    v = _iv_as_decimal(iv)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "No implied volatility data available."
    if v < 0.3:
        return "Low volatility, calm market"
    if v <= 0.8:
        return "Moderate volatility"
    return "High volatility, large price swings expected"


def interpret_short_interest_pct(pct: float | None) -> str:
    if pct is None or (isinstance(pct, float) and math.isnan(pct)):
        return "No short interest data available."
    if pct < 5.0:
        return "Low shorting activity"
    if pct <= 15.0:
        return "Moderate short interest"
    return "Heavily shorted"


def normalize_stock_data(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Map yfinance raw dict to the structured shape used by UI and Sonnet (no LLM)."""
    ticker = str(raw_payload.get("ticker") or "")
    info = raw_payload.get("info") or {}
    rec = raw_payload.get("recommendation_summary_raw") or {}
    opt = raw_payload.get("options_snapshot_raw") or {}
    yearly_pe = raw_payload.get("yearly_pe_points") or []

    trail_pe = _as_float(info.get("trailing_pe"))
    fwd_pe = _as_float(info.get("forward_pe"))
    current_pe = trail_pe if trail_pe is not None else fwd_pe

    call_iv = _as_float(opt.get("avg_call_iv"))
    put_iv = _as_float(opt.get("avg_put_iv"))
    implied_volatility: float | None = None
    if call_iv is not None and put_iv is not None:
        implied_volatility = (call_iv + put_iv) / 2.0
    elif call_iv is not None:
        implied_volatility = call_iv
    elif put_iv is not None:
        implied_volatility = put_iv

    short_raw = _as_float(info.get("short_percent_of_float"))
    short_interest_pct_float: float | None = None
    if short_raw is not None:
        short_interest_pct_float = short_raw * 100.0 if short_raw <= 1.0 else short_raw

    pe_series_5y: List[Dict[str, Any]] = []
    for p in yearly_pe:
        try:
            pe_series_5y.append({"year": int(p["year"]), "pe": float(p["pe"])})
        except (KeyError, TypeError, ValueError):
            continue

    history_1y_close = raw_payload.get("history_1y_close") or []

    inst = raw_payload.get("institutional_snapshot_raw") or {
        "top_institutional_holders": [],
        "major_holders": [],
    }

    return {
        "ticker": ticker,
        "price_snapshot": {
            "current_price": _as_float(info.get("current_price")),
            "market_cap": _as_float(info.get("market_cap")),
            "sector": info.get("sector"),
            "fifty_two_week_high": _as_float(info.get("fifty_two_week_high")),
            "fifty_two_week_low": _as_float(info.get("fifty_two_week_low")),
            "current_pe": current_pe,
            "pe_series_5y": pe_series_5y,
        },
        "analyst_consensus": {
            "buy_count": int(rec.get("buy") or 0),
            "hold_count": int(rec.get("hold") or 0),
            "sell_count": int(rec.get("sell") or 0),
            "sources": list(rec.get("sources") or []),
        },
        "options_sentiment": {
            "call_put_open_interest_ratio": _as_float(opt.get("call_put_oi_ratio")),
            "implied_volatility": implied_volatility,
            "nearest_expiry": opt.get("nearest_expiry"),
        },
        "short_interest_pct_float": short_interest_pct_float,
        "history_1y_close": history_1y_close,
        "institutional_positioning": dict(inst),
    }


def normalized_payload_for_sonnet(normalized: Dict[str, Any]) -> Dict[str, Any]:
    """Drop bulky series and institutional blob duplicate (passed at user_payload root for stock)."""
    drop = frozenset({"history_1y_close", "institutional_positioning"})
    return {k: v for k, v in normalized.items() if k not in drop}


def analyze_with_sonnet(
    client: Anthropic,
    normalized_payload: Dict[str, Any],
    quantity: float,
    scoring_weights: Dict[str, float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    data = normalized_payload_for_sonnet(normalized_payload)
    ticker = str(data.get("ticker") or normalized_payload.get("ticker") or "")
    sector = (normalized_payload.get("price_snapshot") or {}).get("sector")
    inst = normalized_payload.get("institutional_positioning") or {
        "top_institutional_holders": [],
        "major_holders": [],
    }
    summaries = build_web_research_summaries_for_stock(client, ticker, sector)

    user_payload = {
        "holding_quantity": quantity,
        "scoring_weights": scoring_weights,
        "data": data,
        "institutional_positioning": inst,
        "web_research_summaries": summaries,
    }

    kwargs: Dict[str, Any] = {
        "model": SONNET_MODEL,
        "max_tokens": 4096,
        "system": _sonnet_system_with_ephemeral_cache(SONNET_STOCK_ANALYSIS_SYSTEM),
        "messages": [
            {
                "role": "user",
                "content": json.dumps(user_payload, default=str),
            }
        ],
    }
    # No ``tools``: web_search runs only during Haiku research; Sonnet uses web_research_summaries only.

    response = anthropic_messages_create(client, **kwargs)
    text = "".join([c.text for c in response.content if c.type == "text"])
    parsed = extract_json_from_text(text)
    reconcile_stock_analysis(parsed, scoring_weights)
    return parsed, summaries


def portfolio_recommendations_with_sonnet(client: Anthropic, holdings_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
    trimmed = [
        {
            "ticker": rec.get("ticker"),
            "quantity": rec.get("quantity"),
            "analysis": rec.get("analysis"),
            "normalized": normalized_payload_for_sonnet(rec.get("normalized") or {}),
        }
        for rec in holdings_analysis
    ]
    tickers = []
    seen: set[str] = set()
    for rec in holdings_analysis:
        t = str(rec.get("ticker") or "").strip()
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)
    summaries = build_web_research_summaries_for_portfolio(client, tickers)

    user_payload = {
        "holdings_analysis": trimmed,
        "web_research_summaries": summaries,
    }
    # Sonnet receives no ``tools``: web_search runs only in Haiku research + summarize steps above.
    response = anthropic_messages_create(
        client,
        model=SONNET_MODEL,
        max_tokens=2000,
        system=_sonnet_system_with_ephemeral_cache(SONNET_PORTFOLIO_SYSTEM),
        messages=[
            {"role": "user", "content": json.dumps(user_payload, default=str)}
        ],
    )
    text = "".join([c.text for c in response.content if c.type == "text"])
    return extract_json_from_text(text)


def compute_allocation(holdings: List[Dict[str, Any]], normalized_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    total_value = 0.0
    for h in holdings:
        ticker = h["ticker"]
        qty = float(h["quantity"])
        price = (
            normalized_map.get(ticker, {})
            .get("price_snapshot", {})
            .get("current_price")
        )
        value = (price or 0.0) * qty
        total_value += value
        rows.append({"Ticker": ticker, "Shares": qty, "Price": price, "Position Value": value})

    df = pd.DataFrame(rows)
    if total_value > 0 and not df.empty:
        df["Allocation %"] = (df["Position Value"] / total_value * 100).round(2)
        df["Concentration Flag"] = df["Allocation %"].apply(
            lambda x: "High" if x >= 25 else ("Medium" if x >= 15 else "Low")
        )
    else:
        df["Allocation %"] = 0.0
        df["Concentration Flag"] = "Low"
    return df.sort_values(by="Allocation %", ascending=False)


def derive_top_action(analysis_map: Dict[str, Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for _, a in analysis_map.items():
        action = str((a or {}).get("action") or "").strip()
        if not action:
            continue
        counts[action] = counts.get(action, 0) + 1
    if not counts:
        return "N/A"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def build_rebalancing_comparison(
    allocation_df: pd.DataFrame,
    analysis_map: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    if allocation_df.empty:
        return pd.DataFrame()
    # Simple action-weighted tilt model for "after" allocation guidance.
    multipliers = {
        "Strong Buy": 1.20,
        "Buy": 1.10,
        "Hold": 1.00,
        "Reduce": 0.85,
        "Sell": 0.70,
    }
    rows: List[Dict[str, Any]] = []
    weighted_total = 0.0
    for _, row in allocation_df.iterrows():
        t = str(row.get("Ticker", "")).strip()
        before = _as_float(row.get("Allocation %")) or 0.0
        action = str((analysis_map.get(t) or {}).get("action") or "Hold")
        mult = multipliers.get(action, 1.0)
        w = before * mult
        weighted_total += w
        rows.append(
            {
                "Ticker": t,
                "Before Allocation %": round(before, 2),
                "_weighted": w,
                "Suggested Action": action,
            }
        )
    if weighted_total <= 0:
        weighted_total = 1.0
    for r in rows:
        r["After Allocation %"] = round((float(r["_weighted"]) / weighted_total) * 100.0, 2)
        r.pop("_weighted", None)
    return pd.DataFrame(rows)


def run_full_analysis_pipeline(
    client: Anthropic,
    holdings: List[Dict[str, Any]],
    scoring_weights_run: Dict[str, float],
    *,
    show_progress: bool,
    status_placeholder: Any | None = None,
) -> tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, List[Dict[str, Any]]],
    List[Dict[str, Any]],
    Dict[str, Any],
    pd.DataFrame,
]:
    normalized_map: Dict[str, Dict[str, Any]] = {}
    analysis_map: Dict[str, Dict[str, Any]] = {}
    research_by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    all_stock_records: List[Dict[str, Any]] = []
    prog = st.progress(0.0) if show_progress else None

    for idx, h in enumerate(holdings):
        ticker = h["ticker"]
        if show_progress and status_placeholder is not None:
            status_placeholder.info(f"Analyzing {ticker} ({idx + 1}/{len(holdings)})...")
        raw = get_stock_raw_data(
            yfinance_lookup_symbol(ticker),
            display_ticker=ticker,
        )
        normalized = normalize_stock_data(raw)
        analysis, web_summaries = analyze_with_sonnet(
            client,
            normalized,
            h["quantity"],
            scoring_weights_run,
        )
        if idx < len(holdings) - 1:
            time.sleep(PER_STOCK_ANALYSIS_GAP_SEC)
        normalized_map[ticker] = normalized
        analysis_map[ticker] = analysis
        research_by_ticker[ticker] = web_summaries
        all_stock_records.append(
            {
                "ticker": ticker,
                "quantity": h["quantity"],
                "normalized": normalized,
                "analysis": analysis,
            }
        )
        if prog is not None:
            prog.progress((idx + 1) / len(holdings))

    portfolio_ai = portfolio_recommendations_with_sonnet(client, all_stock_records)
    if show_progress and status_placeholder is not None:
        status_placeholder.info("Generating portfolio-level recommendation...")
    allocation_df = compute_allocation(holdings, normalized_map)
    if show_progress and status_placeholder is not None:
        status_placeholder.success("Analysis complete.")
    return (
        normalized_map,
        analysis_map,
        research_by_ticker,
        all_stock_records,
        portfolio_ai,
        allocation_df,
    )


def build_results_summary_rows(
    holdings: List[Dict[str, Any]],
    analysis_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for h in holdings:
        t = h["ticker"]
        a = analysis_map.get(t) or {}
        out.append(
            {
                "ticker": t,
                "composite_score": a.get("composite_score_10"),
                "action": a.get("action"),
            }
        )
    return out


def run_app():
    st.set_page_config(page_title="Wealth Manager Agent", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.1rem; padding-bottom: 1.5rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] { background: #f6f7fb; border-radius: 8px; padding: 8px 14px; font-weight: 600; }
        .app-hero { padding: 12px 2px 16px 2px; margin-bottom: 8px; }
        .app-title { font-size: 2rem; font-weight: 800; line-height: 1.1; margin: 0; }
        .app-tagline { font-size: 1.02rem; color: #546071; margin-top: 8px; margin-bottom: 0; }
        .summary-banner {
            border: 1px solid #e8ebf2; border-radius: 10px; padding: 12px 14px;
            background: linear-gradient(180deg, #fbfcfe 0%, #f6f8fc 100%); margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="app-hero">
          <p class="app-title">AI Wealth Manager Agent</p>
          <p class="app-tagline">Institutional-grade portfolio analysis, powered by AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    user_id = get_or_create_user_id()
    sb_client = get_supabase_client_singleton()

    with st.sidebar:
        st.checkbox(
            "Debug mode (print Claude payload character counts to server terminal)",
            key="debug_claude_payload",
            help="Before each Messages API request, prints len(JSON-serialized kwargs) including model id.",
        )
        if sb_client is not None:
            st.success("Supabase connected — memory + run history enabled.")
        else:
            st.warning("Supabase not configured — provide SUPABASE_URL / SUPABASE_KEY in `st.secrets`.")
        st.divider()
        st.subheader("Batch Scheduling")
        overnight_enabled = st.toggle(
            "Schedule overnight analysis",
            key="overnight_batch_enabled",
        )
        if "overnight_batch_time" not in st.session_state:
            st.session_state["overnight_batch_time"] = dt_time(hour=7, minute=0)
        overnight_time = st.time_input(
            "Batch run time (Pacific)",
            key="overnight_batch_time",
            disabled=not overnight_enabled,
        )
        if overnight_enabled:
            st.caption("Keep this Streamlit process running for scheduled jobs to execute.")
        last_batch = st.session_state.get("last_batch_run")
        st.caption(
            "Last batch run: "
            + (str(last_batch) if last_batch else "Never")
        )
        st.divider()
        st.subheader("Scoring Weights")
        for weight_key, label in WEIGHT_SLIDER_DEFINITIONS:
            sk = f"wrel_{weight_key}"
            if sk not in st.session_state:
                st.session_state[sk] = int(round(DEFAULT_SCORING_WEIGHTS[weight_key] * 100))
            st.slider(label, min_value=0, max_value=100, step=1, key=sk)
        rel_vals = [
            float(st.session_state.get(f"wrel_{k}", 0.0)) for k, _ in WEIGHT_SLIDER_DEFINITIONS
        ]
        rel_sum = sum(rel_vals)
        if rel_sum <= 0:
            scoring_weights_live = DEFAULT_SCORING_WEIGHTS.copy()
            st.caption("Sliders sum to zero — using default weights until at least one is above 0.")
        else:
            scoring_weights_live = {
                k: rel_vals[i] / rel_sum
                for i, (k, _) in enumerate(WEIGHT_SLIDER_DEFINITIONS)
            }
        st.session_state["scoring_weights"] = scoring_weights_live
        st.caption(f"Normalized total: **{sum(scoring_weights_live.values()) * 100:.2f}%**")

    client = get_claude_client()

    def _run_scheduled_batch() -> None:
        holdings_batch = load_saved_portfolio(sb_client, user_id)
        if not holdings_batch:
            return
        scoring_weights_run = dict(
            st.session_state.get("scoring_weights") or DEFAULT_SCORING_WEIGHTS
        )
        (
            _normalized_map,
            analysis_map,
            _research_by_ticker,
            _all_stock_records,
            _portfolio_ai,
            _allocation_df,
        ) = run_full_analysis_pipeline(
            client,
            holdings_batch,
            scoring_weights_run,
            show_progress=False,
            status_placeholder=None,
        )
        results_summary = build_results_summary_rows(holdings_batch, analysis_map)
        save_portfolio_run(
            sb_client,
            user_id=user_id,
            holdings=holdings_batch,
            results_summary=results_summary,
            weights_used={
                **scoring_weights_run,
                "_run_mode": "batch",
            },
        )
        st.session_state["last_batch_run"] = datetime.now(
            ZoneInfo("America/Los_Angeles")
        ).strftime("%Y-%m-%d %H:%M:%S %Z")

    scheduled_at_str = overnight_time.strftime("%H:%M")
    if overnight_enabled:
        if st.session_state.get("_scheduled_job_time") != scheduled_at_str:
            schedule.clear("overnight_analysis")
            schedule.every().day.at(scheduled_at_str).do(_run_scheduled_batch).tag("overnight_analysis")
            st.session_state["_scheduled_job_time"] = scheduled_at_str
    else:
        schedule.clear("overnight_analysis")
        st.session_state["_scheduled_job_time"] = None
    schedule.run_pending()

    if "manual_holdings_input" not in st.session_state:
        loaded_holdings = load_saved_portfolio(sb_client, user_id)
        st.session_state["manual_holdings_input"] = format_holdings_as_manual_input(loaded_holdings)

    st.subheader("Portfolio Input")
    st.text_area(
        "Manual input (format: TICKER:QTY, TICKER:QTY)",
        placeholder="AAPL:12, MSFT:8, NVDA:5",
        key="manual_holdings_input",
    )
    parsed_live = parse_manual_holdings(st.session_state.get("manual_holdings_input", ""))
    uploaded = st.file_uploader("Or upload brokerage screenshot", type=["png", "jpg", "jpeg", "webp"])

    if st.button("Extract holdings from screenshot"):
        if not uploaded:
            st.warning("Upload an image first.")
        else:
            with st.spinner("Extracting holdings with Claude Haiku Vision..."):
                image_b64 = image_to_base64(uploaded)
                extracted = extract_holdings_from_image(client, image_b64, uploaded.type or "image/png")
                st.session_state["manual_holdings_input"] = format_holdings_as_manual_input(extracted)

    st.write("Confirm or edit holdings before analysis (updates from the text field as you type):")
    edit_df = pd.DataFrame(parsed_live if parsed_live else [{"ticker": "", "quantity": 0}])
    edited = st.data_editor(edit_df, num_rows="dynamic", use_container_width=True, key="holdings_editor")
    holdings = []
    for _, row in edited.iterrows():
        ticker = str(row.get("ticker", "")).upper().strip()
        qty = row.get("quantity", 0)
        if ticker and isinstance(qty, (int, float)) and float(qty) > 0:
            holdings.append({"ticker": ticker, "quantity": float(qty)})

    if st.button("Run Full Analysis", type="primary"):
        if not holdings:
            st.error("Please add at least one valid holding.")
            st.stop()
        with st.spinner("Pulling market data and running AI analysis..."):
            scoring_weights_run = dict(
                st.session_state.get("scoring_weights") or DEFAULT_SCORING_WEIGHTS
            )
            status_box = st.empty()
            (
                normalized_map,
                analysis_map,
                research_by_ticker,
                _all_stock_records,
                portfolio_ai,
                allocation_df,
            ) = run_full_analysis_pipeline(
                client,
                holdings,
                scoring_weights_run,
                show_progress=True,
                status_placeholder=status_box,
            )
            results_summary = build_results_summary_rows(holdings, analysis_map)
            save_portfolio_run(
                sb_client,
                user_id=user_id,
                holdings=holdings,
                results_summary=results_summary,
                weights_used={**scoring_weights_run, "_run_mode": "manual"},
            )

            st.session_state["results"] = {
                "holdings": holdings,
                "normalized_map": normalized_map,
                "analysis_map": analysis_map,
                "research_by_ticker": research_by_ticker,
                "allocation_df": allocation_df,
                "portfolio_ai": portfolio_ai,
            }
    has_results = "results" in st.session_state
    if not has_results:
        st.info("Run the analysis to view insights.")
    results = st.session_state.get(
        "results",
        {
            "holdings": [],
            "normalized_map": {},
            "analysis_map": {},
            "research_by_ticker": {},
            "allocation_df": pd.DataFrame(),
            "portfolio_ai": {},
        },
    )
    run_history = fetch_recent_runs(sb_client, user_id, limit=10)

    if has_results:
        alloc = results.get("allocation_df", pd.DataFrame())
        analysis_map = results.get("analysis_map", {}) or {}
        total_value = 0.0
        if isinstance(alloc, pd.DataFrame) and not alloc.empty and "Position Value" in alloc.columns:
            try:
                total_value = float(pd.to_numeric(alloc["Position Value"], errors="coerce").fillna(0).sum())
            except Exception:
                total_value = 0.0
        best_ticker = "N/A"
        best_score = None
        for t, a in analysis_map.items():
            try:
                sc = float((a or {}).get("composite_score_10"))
            except Exception:
                continue
            if best_score is None or sc > best_score:
                best_score = sc
                best_ticker = str(t)
        top_action = derive_top_action(analysis_map)
        st.markdown('<div class="summary-banner">', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        b1.metric("Portfolio Total Value", fmt_two_dec(total_value, prefix="$"))
        b2.metric(
            "Best Performer",
            f"{best_ticker} ({best_score:.2f}/10)" if best_score is not None else "N/A",
        )
        b3.metric("Top Action", top_action)
        st.markdown("</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Portfolio Overview & Allocation",
            "Stock Deep Dives",
            "Rebalancing & New Ideas",
            "Run History",
        ]
    )

    with tab1:
        if not has_results:
            st.caption("No run results yet.")
        st.subheader("Allocation")
        alloc = results["allocation_df"]
        st.dataframe(alloc, use_container_width=True)
        if not alloc.empty:
            st.bar_chart(alloc.set_index("Ticker")["Allocation %"])
            high_flags = alloc[alloc["Concentration Flag"] == "High"]
            if not high_flags.empty:
                st.warning(
                    "Concentration risk detected: "
                    + ", ".join(high_flags["Ticker"].tolist())
                )

    with tab2:
        if not has_results:
            st.caption("No run results yet.")
        st.caption("Per-stock deep dives with AI stances, factor scoring, and source credibility.")
        score_labels = [
            ("analyst_score", "Analyst consensus"),
            ("institutional_score", "Institutional positioning"),
            ("revenue_score", "Forward revenue visibility"),
            ("valuation_score", "Valuation vs historical P/E"),
            ("macro_score", "Macro and sector"),
            ("options_score", "Options and short interest"),
        ]
        for h in results["holdings"]:
            ticker = h["ticker"]
            normalized = results["normalized_map"].get(ticker, {})
            analysis = results["analysis_map"].get(ticker, {})
            web_sum = results.get("research_by_ticker", {}).get(ticker, [])

            st.markdown(f"### {ticker}")
            px = normalized.get("price_snapshot", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", fmt_two_dec(px.get("current_price"), prefix="$"))
            c2.metric("Market Cap", fmt_two_dec(px.get("market_cap"), prefix="$"))
            c3.metric("Sector", px.get("sector") or "N/A")

            c4, c5, c6 = st.columns(3)
            c4.metric("52W High", fmt_two_dec(px.get("fifty_two_week_high"), prefix="$"))
            c5.metric("52W Low", fmt_two_dec(px.get("fifty_two_week_low"), prefix="$"))
            c6.metric("Current P/E", fmt_two_dec(px.get("current_pe")))

            score_v = analysis.get("composite_score_10")
            try:
                score_f = float(score_v) if score_v is not None else None
            except (TypeError, ValueError):
                score_f = None
            act = analysis.get("action") or "Hold"
            sc1, sc2 = st.columns([1, 2])
            with sc1:
                st.metric(
                    "Composite score (0–10)",
                    f"{score_f:.2f}" if score_f is not None else "N/A",
                )
            with sc2:
                st.markdown(f"### Action: **{act}**")
            st.markdown("**Factor scores (0–10)**")
            fcols = st.columns(6)
            for i, (sk, human) in enumerate(score_labels):
                try:
                    sv = float(analysis.get(sk, 0))
                except (TypeError, ValueError):
                    sv = 0.0
                sv = max(0.0, min(10.0, sv))
                with fcols[i]:
                    st.caption(human)
                    st.progress(sv / 10.0)
                    st.caption(f"{sv:.1f}")

            def _stance_block(title: str, key: str) -> None:
                stance = analysis.get(key)
                if not isinstance(stance, dict):
                    stance = {}
                with st.expander(title, expanded=False):
                    st.write(stance.get("reasoning") or "—")
                    cp = stance.get("confidence_pct")
                    try:
                        cp_s = f"{int(float(cp))}%" if cp is not None else "—"
                    except (TypeError, ValueError):
                        cp_s = "—"
                    st.metric("Confidence", cp_s)
                    sup = stance.get("supporting_institutions") or []
                    if isinstance(sup, list) and sup:
                        st.caption("Supporting institutions / firms: " + ", ".join(map(str, sup)))

            _stance_block("Bull Case", "bull_case")
            _stance_block("Bear Case", "bear_case")
            _stance_block("Hold Case", "hold_case")

            inst = normalized.get("institutional_positioning") or {}
            holders = inst.get("top_institutional_holders") or []
            if holders:
                st.markdown("**Top institutional holders**")
                h_rows = []
                for row in holders:
                    pct = row.get("pct")
                    pct_disp = ""
                    if pct is not None:
                        try:
                            pct_disp = f"{float(pct):.2f}%"
                        except (TypeError, ValueError):
                            pct_disp = str(pct)
                    sh = row.get("shares")
                    h_rows.append(
                        {
                            "Holder": row.get("name") or "—",
                            "Shares": sh if sh is not None else "—",
                            "Pct (est.)": pct_disp or "—",
                        }
                    )
                st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)

            st.markdown("**Source credibility (web research)**")
            grouped = grouped_research_by_tier(web_sum)
            if grouped:
                for tier_key, items in grouped:
                    tier_label = {
                        "tier_1": "Tier 1 — weight 1.0 — sec.gov / EDGAR / Seeking Alpha",
                        "tier_2": "Tier 2 — weight 0.8 — Bloomberg / WSJ / FT / Reuters",
                        "tier_3": "Tier 3 — weight 0.6 — CNBC / MarketWatch / Yahoo Finance",
                        "tier_other": "Other — weight 0.55 — remaining domains",
                    }.get(tier_key, tier_key)
                    st.markdown(f"**{tier_label}**")
                    for it in items:
                        ttl = str(it.get("title") or "Untitled")
                        url = str(it.get("url") or "")
                        if url:
                            st.markdown(f"- [{ttl}]({url})")
                        else:
                            st.markdown(f"- {ttl}")
            else:
                st.caption("No tier-tagged web summaries (all sources may have been filtered).")

            pe_series = px.get("pe_series_5y") or []
            if pe_series:
                st.caption("Historical P/E (5y)")
                pe_df = pd.DataFrame(pe_series).sort_values("year")
                pe_df["pe"] = pe_df["pe"].round(2)
                st.line_chart(pe_df.set_index("year")["pe"])

            hist = normalized.get("history_1y_close") or []
            if hist:
                st.caption("1 year daily close")
                hdf = pd.DataFrame(hist)
                if "date" in hdf.columns and "close" in hdf.columns:
                    hdf["date"] = pd.to_datetime(hdf["date"], errors="coerce")
                    hdf = hdf.dropna(subset=["date", "close"]).set_index("date").sort_index()
                    hdf["close"] = hdf["close"].astype(float).round(2)
                    st.line_chart(hdf["close"])

            consensus = normalized.get("analyst_consensus", {})
            st.write(
                f"**Analyst Consensus** — Buy: {consensus.get('buy_count', 0)} | "
                f"Hold: {consensus.get('hold_count', 0)} | "
                f"Sell: {consensus.get('sell_count', 0)}"
            )
            sources = consensus.get("sources", [])
            if sources:
                st.caption("Yahoo / broker sources: " + ", ".join(sources))

            opt = normalized.get("options_sentiment", {})
            cp_ratio = _as_float(opt.get("call_put_open_interest_ratio"))
            iv_raw = _as_float(opt.get("implied_volatility"))
            nearest = opt.get("nearest_expiry")
            short_pct = _as_float(normalized.get("short_interest_pct_float"))

            st.markdown("**Call / put open interest ratio**")
            st.markdown(interpret_call_put_oi_ratio(cp_ratio))

            st.markdown("**Implied volatility (nearest expiry chain)**")
            st.markdown(interpret_implied_volatility(iv_raw))

            st.markdown("**Nearest options expiry**")
            if nearest:
                st.markdown(str(nearest))
            else:
                st.markdown("Not available")

            st.markdown("**Short interest (% of float)**")
            st.markdown(interpret_short_interest_pct(short_pct))
            st.caption(
                f"Raw metrics — OI ratio: {fmt_two_dec(cp_ratio)} | "
                f"IV: {fmt_two_dec(iv_raw)} | "
                f"Short: {fmt_two_dec(short_pct, suffix=' %')}"
            )

            st.divider()

    with tab3:
        if not has_results:
            st.caption("No run results yet.")
        portfolio_ai = results["portfolio_ai"]
        st.subheader("Rebalancing Recommendation")
        st.write(portfolio_ai.get("rebalancing_recommendation", "N/A"))

        alloc_before = results.get("allocation_df", pd.DataFrame())
        alloc_after = build_rebalancing_comparison(
            alloc_before if isinstance(alloc_before, pd.DataFrame) else pd.DataFrame(),
            results.get("analysis_map", {}) or {},
        )
        if not alloc_after.empty:
            st.markdown("**Before / After Allocation Comparison**")
            st.dataframe(alloc_after, use_container_width=True, hide_index=True)

        risks = portfolio_ai.get("concentration_risks", [])
        if risks:
            st.write("**Concentration Risk Flags**")
            for risk in risks:
                st.write(f"- {risk}")

        st.subheader("New Stock Ideas (3-5)")
        ideas = portfolio_ai.get("new_stock_ideas", [])
        for idea in ideas:
            st.write(f"**{idea.get('ticker', 'N/A')}**: {idea.get('thesis', '')}")

    with tab4:
        st.subheader("Run History (Last 10)")
        if not run_history:
            st.caption("No historical runs found for this session user.")
        for i, row in enumerate(run_history):
            ts = str(row.get("run_timestamp") or "Unknown time")
            hs = row.get("holdings") or []
            tickers = [str(h.get("ticker", "")).upper().strip() for h in hs if isinstance(h, dict)]
            tickers_disp = ", ".join([t for t in tickers if t]) or "(none)"
            summary = row.get("results_summary") or []
            score_bits: List[str] = []
            for s in summary:
                if not isinstance(s, dict):
                    continue
                t = str(s.get("ticker", "")).upper().strip()
                sc = s.get("composite_score")
                try:
                    scs = f"{float(sc):.2f}"
                except Exception:
                    scs = "N/A"
                if t:
                    score_bits.append(f"{t}:{scs}")
            score_disp = " | ".join(score_bits) if score_bits else "No scores"
            with st.expander(f"{ts} — {tickers_disp} — {score_disp}", expanded=False):
                if isinstance(summary, list) and summary:
                    rows = []
                    for s in summary:
                        if not isinstance(s, dict):
                            continue
                        rows.append(
                            {
                                "Ticker": s.get("ticker"),
                                "Composite Score": s.get("composite_score"),
                                "Action": s.get("action"),
                            }
                        )
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption("Holdings")
                if isinstance(hs, list) and hs:
                    hrows = []
                    for h in hs:
                        if not isinstance(h, dict):
                            continue
                        hrows.append({"Ticker": h.get("ticker"), "Quantity": h.get("quantity")})
                    if hrows:
                        st.dataframe(pd.DataFrame(hrows), use_container_width=True, hide_index=True)
                w = row.get("weights_used")
                if isinstance(w, dict) and w:
                    st.caption("Weights used")
                    st.json(w)


if __name__ == "__main__":
    run_app()
