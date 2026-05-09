import base64
import json
import math
import re
import time
from uuid import uuid4
from pathlib import Path
from datetime import datetime, time as dt_time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import schedule
import streamlit as st
import yfinance as yf
from anthropic import Anthropic, APIStatusError
from supabase_memory import (
    fetch_cached_stock_analysis,
    fetch_recent_runs,
    get_or_create_user_by_email,
    get_supabase_client_singleton,
    load_saved_portfolio,
    save_stock_analysis_cache,
    upsert_portfolio_run_snapshot,
)


HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-20250514"

# Estimated token count: ~1500 (intentionally >1024 for Anthropic prompt cache threshold).
SONNET_STOCK_ANALYSIS_SYSTEM = """You are a senior equity research and portfolio recommendation assistant.
You analyze exactly one symbol per request, using only structured JSON user input.

PRIMARY OBJECTIVE
Produce a transparent institutional-style assessment with balanced bull/bear/hold framing,
factor-by-factor scoring on a 0-10 scale, and an action recommendation derived from weighted
composite scoring.

DATA CONTRACT
User input includes:
- holding_quantity
- scoring_weights with six decimal weights that sum to 1.0:
  analyst_consensus, institutional_positioning, forward_revenue_visibility,
  valuation_vs_historical_pe, macro_and_sector_factors, options_and_short_interest
- data object with normalized market, fundamentals, and analyst fields
- institutional_positioning data
- web_research_summaries list of source summaries. Each summary may include source_tier and source_weight.

HARD CONSTRAINTS
- You cannot browse or use tools.
- You cannot invent data not inferable from user payload.
- If data is missing, acknowledge uncertainty and degrade confidence.
- Output must be strict JSON only (no markdown, no prose wrapper).

SOURCE CREDIBILITY WEIGHTING
- tier_1 / source_weight near 1.0: filings and highest-credibility context; prioritize strongly.
- tier_2 / source_weight near 0.8: high-quality business press and institutional commentary; use heavily.
- tier_3 / source_weight near 0.6: mainstream financial media; use moderately.
- tier_other / source_weight near 0.55: supportive context only; do not over-weight.
- Conflicting claims: prefer higher credibility and more recent context.
- Do not overfit to any single source. Combine with structured market/fundamental signals.

SCORING RUBRIC OVERVIEW
Score every factor from 0.0 to 10.0.
Anchor interpretation:
- 0-2: materially negative
- 3-4: weak
- 5-6: neutral/mixed
- 7-8: constructive
- 9-10: exceptional
Avoid extreme scores unless evidence is strong and multi-sourced.

FACTOR 1: analyst_score (analyst_consensus)
- Inputs: buy/hold/sell counts, trend indications, reputable research commentary.
- Higher when analyst stance is consistently constructive with coherent rationale.
- Lower when downgrades dominate, revisions trend negative, or outlook uncertainty rises.

FACTOR 2: institutional_score (institutional_positioning)
- Inputs: top holders, ownership stability, concentration among sophisticated funds.
- Higher with stable/high-quality institutional sponsorship and positive ownership trends.
- Lower with sharp ownership deterioration, weak sponsorship, or churn concerns.

FACTOR 3: revenue_score (forward_revenue_visibility)
- Inputs: guidance quality, demand durability, backlog/order visibility, catalyst clarity.
- Higher when forward growth path has credible, specific support.
- Lower when revenue outlook is opaque, cyclically fragile, or heavily assumption-driven.

FACTOR 4: valuation_score (valuation_vs_historical_pe)
- Inputs: current valuation versus own history/sector context, growth quality, risk premium.
- Higher when valuation is attractive relative to quality and growth durability.
- Lower when valuation is stretched without commensurate quality/visibility.

FACTOR 5: macro_score (macro_and_sector_factors)
- Inputs: rate backdrop, inflation sensitivity, commodity/fx effects, sector cycle.
- Higher when macro and sector regime support earnings resilience.
- Lower when top-down regime likely compresses multiples or margins.

FACTOR 6: options_score (options_and_short_interest)
- Inputs: call/put open-interest ratio, implied volatility context, short-interest pressure.
- Higher when positioning and sentiment imply constructive skew without complacency.
- Lower when derivatives/short-interest indicate stress, crowding risk, or downside skew.

COMPOSITE CALCULATION REQUIREMENT
Compute exactly:
composite_score_10 =
  analyst_consensus*analyst_score +
  institutional_positioning*institutional_score +
  forward_revenue_visibility*revenue_score +
  valuation_vs_historical_pe*valuation_score +
  macro_and_sector_factors*macro_score +
  options_and_short_interest*options_score

ACTION MAPPING REQUIREMENT
- Strong Buy: composite_score_10 >= 8.5
- Buy: 7.0 <= composite_score_10 < 8.5
- Hold: 5.0 <= composite_score_10 < 7.0
- Reduce: 3.0 <= composite_score_10 < 5.0
- Sell: composite_score_10 < 3.0

CASE-WRITING GUIDELINES
Provide bull_case, bear_case, hold_case:
- reasoning: concise but specific, evidence-led narrative
- confidence_pct: 0-100 reflecting evidence quality + consistency
- supporting_institutions: known relevant firms/funds if present, else empty array
Reasoning should be distinct across the three cases.

QUALITY CONTROL CHECKLIST
- Ensure all required keys exist.
- Ensure factor scores are numeric and bounded 0-10.
- Ensure confidence_pct values are 0-100.
- Ensure composite_score_10 is numeric.
- Ensure action aligns to computed composite.
- Ensure no extra top-level keys beyond schema.

OUTPUT SCHEMA (STRICT)
{
  "bull_case": {
    "reasoning": "<string>",
    "confidence_pct": <number 0-100>,
    "supporting_institutions": ["<string>", "..."]
  },
  "bear_case": {
    "reasoning": "<string>",
    "confidence_pct": <number 0-100>,
    "supporting_institutions": ["<string>", "..."]
  },
  "hold_case": {
    "reasoning": "<string>",
    "confidence_pct": <number 0-100>,
    "supporting_institutions": ["<string>", "..."]
  },
  "analyst_score": <number 0-10>,
  "institutional_score": <number 0-10>,
  "revenue_score": <number 0-10>,
  "valuation_score": <number 0-10>,
  "macro_score": <number 0-10>,
  "options_score": <number 0-10>,
  "composite_score_10": <number 0-10>,
  "action": "Strong Buy" | "Buy" | "Hold" | "Reduce" | "Sell",
  "thesis": "<short synthesis string>"
}"""

# Estimated token count: ~1350 (intentionally >1024 for Anthropic prompt cache threshold).
SONNET_PORTFOLIO_SYSTEM = """You are a portfolio strategist producing concise, actionable guidance.
Each request contains holdings_analysis and web_research_summaries.

OBJECTIVE
Identify concentration risks, provide a practical rebalancing recommendation, and suggest
3-5 new ideas not currently held.

CONSTRAINTS
- Use only the provided JSON.
- No browsing or tool use.
- Prefer high-credibility web evidence (tier_1 then tier_2, etc.).
- If evidence conflicts, disclose uncertainty and avoid overconfidence.
- Output strict JSON only.

PORTFOLIO RISK FRAMEWORK
Evaluate:
- single-name concentration
- sector/style/factor crowding
- valuation concentration (multiple compression sensitivity)
- macro sensitivity concentration (rates/fx/commodity)
- liquidity/event risk clustering
Flag concrete risks in plain language with why they matter.

REBALANCING GUIDANCE
Recommendations should:
- prioritize risk-adjusted return and diversification
- align to quality/visibility over narrative momentum
- indicate what to reduce/add and why
- avoid unnecessary churn if portfolio is already balanced
- mention sequencing when useful (e.g., trim first, then stage adds)

NEW IDEA GENERATION
Return 3 to 5 tickers not already held.
For each idea thesis:
- 1-3 sentences
- include catalyst, risk lens, and fit versus existing exposures
- avoid duplicating existing concentration unless justified

SOURCE-WEIGHT POLICY
- tier_1: strongest grounding (regulatory/primary-source style)
- tier_2: high-quality secondary confirmation
- tier_3 and other: contextual support, lower conviction weight
Never let low-tier evidence dominate portfolio decisions.

OUTPUT SCHEMA (STRICT)
{
  "concentration_risks": ["<string>", "..."],
  "rebalancing_recommendation": "<string>",
  "new_stock_ideas": [
    {"ticker": "<string>", "thesis": "<string>"},
    {"ticker": "<string>", "thesis": "<string>"}
  ]
}

QUALITY CHECKS
- concentration_risks must be an array of strings.
- new_stock_ideas length must be 3..5 unless insufficient evidence.
- Tickers should be uppercase symbols.
- Do not include held symbols in new_stock_ideas.
- No extra top-level keys.

Extended rubric notes for consistent evaluations:
- Prefer businesses with resilient cash-flow and durable demand.
- Penalize highly levered balance sheets in deteriorating macro setups.
- Favor diversified earnings engines over single-product dependency.
- Use valuation discipline even for high-growth narratives.
- Evaluate regime dependence: rates up/down, inflation persistence, credit spread widening.
- Translate risks into actionable portfolio moves, not generic commentary.
- Keep recommendations implementable by an investor without institutional infrastructure.
- Focus on durable risk controls first, alpha ideas second.
- Minimize style whiplash unless clear evidence supports rotation.
- Explain tradeoffs succinctly to support decision quality.

Additional implementation guidance:
- If holdings already include multiple correlated semis/software names, flag clustering.
- If macro-sensitive cyclicals dominate, discuss economic slowdown drawdown risk.
- If ideas are added in same sector, justify with non-overlapping drivers.
- If data quality is low, return conservative and explicit caveats in recommendation text.
- Preserve JSON validity under all circumstances.
- Keep wording direct and evidence-linked.

Repeatable decision principles:
- evidence > narrative
- diversification > concentration drift
- durability > short-term noise
- asymmetric upside with bounded downside > uncertain upside with unbounded downside
- clear catalysts + quality balance sheets + reasonable valuation preferred

These principles should shape every response consistently."""

# Estimated token count: ~1160 (intentionally >1024 for Anthropic prompt cache threshold).
HAIKU_STOCK_WEB_RESEARCH_SYSTEM = """You are a web-research retriever for a single stock.
Use the web_search tool exactly once per API call.

INPUTS PROVIDED IN USER MESSAGE
- ticker
- sector
- query

STRICT EXECUTION RULES
- Run exactly one web_search call using the provided query text.
- Do not generate additional queries.
- Do not run multiple searches.
- Do not broaden scope beyond the provided query.
- Do not call web_search more than once.

OUTPUT BEHAVIOR
- Keep final assistant text minimal (one short acknowledgement).
- Retrieved tool results are what downstream pipeline consumes.
- Do not provide long narrative analysis in this step.

SOURCE PRIORITY
Prioritize domains typically mapped to:
- tier_1: sec.gov/edgar/seekingalpha-style transcript/filing sources
- tier_2: Bloomberg/WSJ/FT/Reuters class sources
- tier_3: CNBC/MarketWatch/Yahoo Finance class sources
Avoid social chatter, forums, and unverified rumor sources.

QUALITY BAR
- Favor recency and specificity.
- Prefer concrete signals (figures, guidance, filings) to vague opinions.
- Keep retrieval relevant to investor decision making.
- Never hallucinate search results; rely on tool output only.

Implementation discipline:
- Use the exact query string from user input.
- Keep behavior deterministic across runs for consistent downstream processing.

This instruction block is intentionally verbose for prompt caching and consistency."""

# Estimated token count: ~1280 (intentionally >1024 for Anthropic prompt cache threshold).
HAIKU_PORTFOLIO_WEB_RESEARCH_SYSTEM = """You are a web-research retriever for portfolio-level strategy.
Use web_search to collect broad market and allocation-relevant context for current holdings.

INPUT IN USER MESSAGE
- tickers list/string representing currently held names

RETRIEVAL OBJECTIVES
1) cross-sector macro themes likely to affect rebalancing
2) concentration and diversification considerations
3) quality opportunities adjacent to current holdings
4) risk management context (rates, liquidity, policy, cycle)
5) candidate sectors/companies with improving forward setups

SOURCE POLICY
- Prefer high-credibility financial journalism and primary disclosures.
- Keep a mix of top-tier and corroborative sources.
- Avoid social media/forum rumor channels.

WORKFLOW
- Execute multiple web_search queries.
- Gather diverse but relevant result sets.
- Keep final natural-language reply short; downstream code ingests tool results.

RISK-FIRST LENS
- Search for evidence of regime shifts, earnings dispersion, valuation extremes,
  and crowded positioning.
- Seek data points that can justify trimming/adds in a rebalancing plan.

QUERY DESIGN PRINCIPLES
- Include tickers when helpful, but also search at portfolio/theme level.
- Cover both upside opportunities and downside risks.
- Favor recent information unless historical framing is needed.
- Use terms such as "allocation", "diversification", "sector rotation",
  "earnings revisions", "valuation dispersion", "risk premium".

QUALITY CONTROLS
- Avoid duplicate queries with only superficial wording changes.
- Prefer sources with author/date/transparency.
- Capture a balanced set of perspectives.
- Keep retrieval tuned to practical investing actions.
- Avoid noisy or promotional content.

Extended guidance:
- If holdings are concentrated in one sector, prioritize offsetting-theme research.
- If growth-heavy, include rates and profitability regime context.
- If value-heavy, include cyclicality and earnings sensitivity context.
- Include global macro linkages where they materially impact US equities.
- Keep retrieval broad enough to produce diverse new idea candidates.

This instruction block is intentionally verbose for prompt caching and consistency."""

# Estimated token count: ~1260 (intentionally >1024 for Anthropic prompt cache threshold).
HAIKU_BATCH_SUMMARIZATION_SYSTEM = """You summarize multiple web results in batch for one source tier.
Each request includes structured hit content; produce one summary per hit.

GOAL
Create investor-useful summaries that are faithful, specific when supported, and uncertainty-aware.

INPUT EXPECTATIONS
- A source_tier (tier_1, tier_2, tier_3, tier_other)
- A sentence range target for each summary
- A JSON array of hits with hit_id, title, url, page_age, and optional text excerpts

OUTPUT REQUIREMENTS
- Return strict JSON only:
  {"summaries":[{"hit_id":0,"summary":"..."}]}
- Include exactly one summary per input hit_id.
- No markdown, no prose outside JSON.

SUMMARY STYLE
- Plain prose paragraphs.
- Respect sentence range exactly for each hit.
- Mention limitations when content is sparse.
- Do not fabricate facts beyond title/url/excerpt evidence.

TIER-SENSITIVE EMPHASIS
- tier_1: prioritize concrete figures, dates, guidance, and filing-like specifics.
- tier_2: emphasize credible market/analyst/business developments with context.
- tier_3: keep to moderate-confidence takeaways and clearly supported points.
- tier_other: concise, theme-level interpretation with explicit caution.

QUALITY CONTROLS
- Preserve attribution context implied by source metadata.
- Prefer factual grounding to stylistic flourish.
- Avoid repetitive phrasing across summaries.
- Keep investor relevance explicit: potential impact on risk, earnings, valuation, or sentiment.
- Avoid unsupported certainty language.

ROBUSTNESS RULES
- If a hit lacks enough text, infer cautiously from title/url/domain cues.
- Never invent precise numbers that are not present.
- If uncertain, state uncertainty in the summary text.
- Keep each summary coherent and self-contained.

FORMAT ENFORCEMENT
- Ensure JSON parses cleanly.
- hit_id must remain integer.
- summary must be non-empty string.
- output array length must match input array length.

This instruction block is intentionally verbose for prompt caching and consistency."""

# Estimated token count: ~1250 (intentionally >1024 for Anthropic prompt cache threshold).
HAIKU_IMAGE_EXTRACTION_SYSTEM = """You extract holdings from brokerage screenshots.
Output only valid JSON array of objects with ticker and quantity.

PRIMARY TASK
Identify equity/ETF rows and convert them to:
[{"ticker":"AAPL","quantity":10}]

STRICT RULES
- No markdown, no prose, no explanations.
- Ticker must be uppercase symbol text.
- Quantity must be numeric.
- Exclude cash, totals, headers, footers, account labels.
- Exclude rows where ticker or quantity is missing/ambiguous.

OCR ROBUSTNESS GUIDANCE
- Handle commas, decimals, and common OCR artifacts.
- Distinguish quantity from market value/price columns.
- Ignore percentage-only rows unless quantity is clearly present.
- Prefer precision but avoid false positives.

NORMALIZATION RULES
- Trim whitespace.
- Remove currency symbols from quantities.
- Preserve decimal shares when shown.
- Skip duplicate rows unless they represent distinct holdings lines.

QUALITY CHECKS
- Return empty array if no reliable holdings are visible.
- Never include malformed objects.
- Keep schema exact on every element.

This instruction block is intentionally verbose for prompt caching and consistency."""

# Shared web_search tool definition (same object for every Haiku search request).
HAIKU_WEB_SEARCH_TOOLS: List[Dict[str, str]] = [{"type": "web_search_20250305", "name": "web_search"}]

# Identical suffix on every cached system route: keeps cache_control on block 1 only; no dynamic text here.
_SYSTEM_CACHE_ANCHOR_SUFFIX = (
    "Static policy anchor: all ticker symbols, quantities, prices, fetched JSON payloads, "
    "run identifiers, and timestamps appear only in user messages, never in these system instructions."
)


def _cached_instruction_system_blocks(instruction_text: str) -> List[Dict[str, Any]]:
    """Block 1: cacheable static instructions. Block 2: short static text without cache_control."""
    return [
        {"type": "text", "text": instruction_text, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": _SYSTEM_CACHE_ANCHOR_SUFFIX},
    ]


_STATIC_CACHED_SYSTEM_BLOCKS: Dict[str, List[Dict[str, Any]]] = {
    "sonnet_stock": _cached_instruction_system_blocks(SONNET_STOCK_ANALYSIS_SYSTEM),
    "sonnet_portfolio": _cached_instruction_system_blocks(SONNET_PORTFOLIO_SYSTEM),
    "haiku_stock_research": _cached_instruction_system_blocks(HAIKU_STOCK_WEB_RESEARCH_SYSTEM),
    "haiku_portfolio_research": _cached_instruction_system_blocks(HAIKU_PORTFOLIO_WEB_RESEARCH_SYSTEM),
    "haiku_batch_summarization": _cached_instruction_system_blocks(HAIKU_BATCH_SUMMARIZATION_SYSTEM),
    "haiku_image_extraction": _cached_instruction_system_blocks(HAIKU_IMAGE_EXTRACTION_SYSTEM),
}


# On HTTP 429: exponential backoff (60s, 120s, 240s), up to MAX_RETRIES retries per request.
# Applies to every Messages API call (vision extract, Haiku research/summarize, Sonnet calls).
ANTHROPIC_429_RETRY_WAIT_SEC = 60.0
ANTHROPIC_429_MAX_RETRIES = 3  # retries after a 429; total attempts = MAX_RETRIES + 1
ANTHROPIC_500_RETRY_WAIT_SEC = 10.0
ANTHROPIC_500_MAX_RETRIES = 3  # retries after a 500; total attempts = MAX_RETRIES + 1

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
HAIKU_SINGLE_WEB_SEARCH_MAX_TOKENS = 2048
HAIKU_SEARCH_HIT_SUMMARY_MAX_TOKENS = 4000
MAX_WEB_SEARCH_HITS_PER_ROUND = 20
MAX_SUMMARIES_PER_TARGETED_QUERY = 5
MAX_PORTFOLIO_SEARCH_HITS_TO_SUMMARIZE = 15
MAX_HAIKU_WEB_SEARCH_CALLS_PER_STOCK = 8
DEBUG_LOG_FILE = Path(__file__).resolve().parent / "debug_log.txt"



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


def stock_targeted_search_queries(
    ticker: str,
    sector: str | None,
    market_cap: float | None,
) -> List[str]:
    _ = sector or "Unknown sector"
    t = ticker.strip().upper()
    cap = _as_float(market_cap)

    large_cap_queries = [
        f"{t} SEC EDGAR 10-K latest filing",
        f"{t} SEC EDGAR latest 10-Q filing",
        f"{t} SEC EDGAR latest 8-K material events",
        f"{t} earnings call transcript latest quarter",
        f"{t} earnings guidance management commentary latest",
        f"{t} Goldman Sachs Morgan Stanley analyst rating latest",
        f"{t} JPMorgan BofA Barclays price target latest",
        f"{t} 13F institutional holdings latest quarter",
        f"{t} BlackRock Vanguard 13F position latest",
        f"{t} insider buying selling SEC Form 4 latest",
        f"{t} Bloomberg company news latest",
        f"{t} Reuters company news latest",
    ]

    mid_cap_queries = [
        f"{t} SEC EDGAR 10-K latest filing",
        f"{t} SEC EDGAR latest 10-Q filing",
        f"{t} earnings call transcript latest quarter",
        f"{t} earnings guidance management commentary latest",
        f"{t} Goldman Sachs Morgan Stanley analyst rating latest",
        f"{t} JPMorgan BofA Barclays price target latest",
        f"{t} insider buying selling SEC Form 4 latest",
        f"{t} Reuters company news latest",
    ]

    small_cap_queries = [
        f"{t} SEC EDGAR 10-K latest filing",
        f"{t} SEC EDGAR latest 10-Q filing",
        f"{t} earnings call transcript latest quarter",
        f"{t} earnings guidance management commentary latest",
        f"{t} analyst rating latest",
        f"{t} analyst price target latest",
    ]

    if cap is not None and cap < 5_000_000_000:
        return small_cap_queries
    if cap is not None and cap < 10_000_000_000:
        return mid_cap_queries
    return large_cap_queries


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


def _tier_sentence_bounds_and_focus(tier: str) -> tuple[int, int, str]:
    if tier == "tier_1":
        return (
            10,
            12,
            "Tier 1 (highest credibility). Extract specific numbers, dates, filed metrics, and forward guidance "
            "verbatim when available. If page text is scarce, rely on title/URL/domain and do not invent details.",
        )
    if tier == "tier_2":
        return (
            7,
            8,
            "Tier 2 (high credibility). Focus on market signals, analyst commentary, and business developments. "
            "Extract key takeaways and any explicitly cited figures/dates.",
        )
    if tier == "tier_3":
        return (
            4,
            5,
            "Tier 3 (moderate credibility). Keep summaries high level and grounded only in clearly supported details.",
        )
    return (
        3,
        4,
        "Other/unclassified source. Keep summaries concise and thematic rather than overly specific.",
    )


def haiku_summarize_search_hits_for_tier(
    client: Anthropic,
    *,
    research_context: str,
    source_tier: str,
    hits: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Summarize all hits in a tier with one Haiku call; returns [{hit_id, summary}]."""
    if not hits:
        return []
    sentences_min, sentences_max, _tier_focus = _tier_sentence_bounds_and_focus(source_tier)
    hits_payload: List[Dict[str, Any]] = []
    for i, hit in enumerate(hits):
        hits_payload.append(
            {
                "hit_id": i,
                "title": hit.get("title", ""),
                "url": hit.get("url", ""),
                "page_age": hit.get("page_age"),
                "page_text_excerpt": _optional_text_from_encrypted(hit.get("encrypted_content") or ""),
            }
        )
    user_msg = (
        f"research_context={research_context}\n"
        f"source_tier={source_tier}\n"
        f"sentences_min={sentences_min}\n"
        f"sentences_max={sentences_max}\n"
        f"hits_json={json.dumps(hits_payload, default=str)}"
    )
    r = anthropic_messages_create(
        client,
        model=HAIKU_MODEL,
        max_tokens=HAIKU_SEARCH_HIT_SUMMARY_MAX_TOKENS,
        system=_STATIC_CACHED_SYSTEM_BLOCKS["haiku_batch_summarization"],
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(c.text for c in r.content if c.type == "text").strip()
    try:
        parsed = extract_json_from_text(text)
    except Exception as e:
        print(
            f"[WARN] Failed to parse Haiku tier batch summaries for tier={source_tier}: {e}",
            flush=True,
        )
        return [{"hit_id": i, "summary": ""} for i, _ in enumerate(hits)]
    if isinstance(parsed, dict):
        summaries = parsed.get("summaries")
        if isinstance(summaries, list):
            out: List[Dict[str, Any]] = []
            for item in summaries:
                if not isinstance(item, dict):
                    continue
                try:
                    hit_id = int(item.get("hit_id"))
                except Exception:
                    continue
                summary_text = str(item.get("summary", "")).strip()
                if summary_text:
                    out.append({"hit_id": hit_id, "summary": summary_text})
            if out:
                return out
    # Fallback: keep the run resilient even if model output is malformed.
    print(
        f"[WARN] Haiku tier batch summaries malformed for tier={source_tier}; returning empty summaries.",
        flush=True,
    )
    return [{"hit_id": i, "summary": ""} for i, _ in enumerate(hits)]


def haiku_web_search_for_stock(
    client: Anthropic,
    ticker: str,
    sector: str | None,
    query: str,
) -> List[Dict[str, Any]]:
    """One Haiku+web_search retrieval call for a single targeted stock query."""
    sector_text = (sector or "Unknown").strip() or "Unknown"
    research = anthropic_messages_create(
        client,
        model=HAIKU_MODEL,
        max_tokens=HAIKU_SINGLE_WEB_SEARCH_MAX_TOKENS,
        system=_STATIC_CACHED_SYSTEM_BLOCKS["haiku_stock_research"],
        tools=HAIKU_WEB_SEARCH_TOOLS,
        messages=[
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "ticker": ticker.strip().upper(),
                        "sector": sector_text,
                        "query": query,
                    },
                    default=str,
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
    client: Anthropic, ticker: str, sector: str | None, market_cap: float | None
) -> List[Dict[str, Any]]:
    """Stock web research retrieval + tier-batched summarization."""
    collected_hits: List[Dict[str, Any]] = []
    ctx = f"Targeted diligence for {ticker.upper()}"
    search_calls_made = 0
    for query in stock_targeted_search_queries(ticker, sector, market_cap):
        if search_calls_made >= MAX_HAIKU_WEB_SEARCH_CALLS_PER_STOCK:
            break
        hits = haiku_web_search_for_stock(client, ticker, sector, query)
        search_calls_made += 1
        tier3_count = 0
        other_count = 0
        for h in hits:
            url = h.get("url") or ""
            tier, weight = classify_search_source(url)
            if tier == "blocked" or weight <= 0:
                continue

            if tier == "tier_3":
                if tier3_count >= 3:
                    continue
            elif tier == "tier_other":
                if other_count >= 3:
                    continue

            collected_hits.append(
                {
                    "search_query": query,
                    "title": h.get("title"),
                    "url": url,
                    "page_age": h.get("page_age"),
                    "encrypted_content": h.get("encrypted_content"),
                    "source_tier": tier,
                    "source_weight": weight,
                }
            )
            if tier == "tier_3":
                tier3_count += 1
            elif tier == "tier_other":
                other_count += 1
    print(
        f"[Haiku Search] {ticker.upper()} completed {search_calls_made} web_search calls "
        f"(cap={MAX_HAIKU_WEB_SEARCH_CALLS_PER_STOCK}).",
        flush=True,
    )
    out: List[Dict[str, Any]] = []
    for tier in ["tier_1", "tier_2", "tier_3", "tier_other"]:
        tier_hits = [h for h in collected_hits if h.get("source_tier") == tier]
        if not tier_hits:
            continue
        summaries = haiku_summarize_search_hits_for_tier(
            client,
            research_context=ctx,
            source_tier=tier,
            hits=tier_hits,
        )
        summary_by_id = {
            int(s.get("hit_id")): str(s.get("summary", "")).strip()
            for s in summaries
            if isinstance(s, dict)
        }
        for i, h in enumerate(tier_hits):
            out.append(
                {
                    "search_query": h.get("search_query"),
                    "title": h.get("title"),
                    "url": h.get("url"),
                    "page_age": h.get("page_age"),
                    "summary": summary_by_id.get(i, "Summary unavailable for this source."),
                    "source_tier": h.get("source_tier"),
                    "source_weight": h.get("source_weight"),
                }
            )
    return out


def build_web_research_summaries_for_portfolio(
    client: Anthropic, current_tickers: List[str]
) -> List[Dict[str, Any]]:
    tickers_str = ", ".join(current_tickers) if current_tickers else "(none)"
    research = anthropic_messages_create(
        client,
        model=HAIKU_MODEL,
        max_tokens=HAIKU_PORTFOLIO_WEB_RESEARCH_MAX_TOKENS,
        system=_STATIC_CACHED_SYSTEM_BLOCKS["haiku_portfolio_research"],
        tools=HAIKU_WEB_SEARCH_TOOLS,
        messages=[
            {
                "role": "user",
                "content": json.dumps({"tickers": tickers_str}, default=str),
            }
        ],
    )
    hits = extract_web_search_hits_from_response(research)[:MAX_PORTFOLIO_SEARCH_HITS_TO_SUMMARIZE]
    collected_hits: List[Dict[str, Any]] = []
    ctx = f"Portfolio-level research; current holdings: {tickers_str}"
    for h in hits:
        url = h.get("url") or ""
        tier, weight = classify_search_source(url)
        if tier == "blocked" or weight <= 0:
            continue
        collected_hits.append(
            {
                "title": h.get("title"),
                "url": h.get("url"),
                "page_age": h.get("page_age"),
                "encrypted_content": h.get("encrypted_content"),
                "source_tier": tier,
                "source_weight": weight,
            }
        )
    out: List[Dict[str, Any]] = []
    for tier in ["tier_1", "tier_2", "tier_3", "tier_other"]:
        tier_hits = [h for h in collected_hits if h.get("source_tier") == tier]
        if not tier_hits:
            continue
        summaries = haiku_summarize_search_hits_for_tier(
            client,
            research_context=ctx,
            source_tier=tier,
            hits=tier_hits,
        )
        summary_by_id = {
            int(s.get("hit_id")): str(s.get("summary", "")).strip()
            for s in summaries
            if isinstance(s, dict)
        }
        for i, h in enumerate(tier_hits):
            out.append(
                {
                    "title": h.get("title"),
                    "url": h.get("url"),
                    "summary": summary_by_id.get(i, "Summary unavailable for this source."),
                    "source_tier": h.get("source_tier"),
                    "source_weight": h.get("source_weight"),
                }
            )
    return out


def _claude_debug_payload_enabled() -> bool:
    try:
        return bool(st.session_state.get("debug_claude_payload"))
    except Exception:
        return False


def _append_debug_log_line(line: str) -> None:
    try:
        with DEBUG_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        return


def _ensure_debug_run_header() -> None:
    key = "_debug_log_run_header_written"
    try:
        already_written = bool(st.session_state.get(key))
    except Exception:
        already_written = False
    if already_written:
        return
    ts = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")
    _append_debug_log_line("")
    _append_debug_log_line(f"=== Claude debug run started: {ts} ===")
    try:
        st.session_state[key] = True
    except Exception:
        return


def _count_web_search_results(response: Any) -> int:
    count = 0
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) != "web_search_tool_result":
            continue
        inner = getattr(block, "content", None)
        if not isinstance(inner, list):
            continue
        for item in inner:
            if getattr(item, "type", None) == "web_search_result":
                count += 1
    return count


def anthropic_messages_create(client: Anthropic, **kwargs: Any) -> Any:
    """Call Messages API with retries for throttling/server faults."""
    debug_enabled = _claude_debug_payload_enabled()
    payload_repr = ""
    model = kwargs.get("model", "?")
    if debug_enabled:
        _ensure_debug_run_header()
        payload_repr = json.dumps(kwargs, default=str, sort_keys=True)
    max_attempts = max(ANTHROPIC_429_MAX_RETRIES, ANTHROPIC_500_MAX_RETRIES) + 1
    for attempt in range(max_attempts):
        try:
            response = client.messages.create(**kwargs)
            if debug_enabled:
                search_count = _count_web_search_results(response)
                debug_line = (
                    f"[Claude API debug] model={model} "
                    f"payload_character_count={len(payload_repr)} "
                    f"search_count={search_count}"
                )
                print(debug_line, flush=True)
                _append_debug_log_line(debug_line)
            return response
        except APIStatusError as e:
            status_code = getattr(e, "status_code", None)
            if status_code == 429:
                if attempt >= ANTHROPIC_429_MAX_RETRIES:
                    raise
                time.sleep(ANTHROPIC_429_RETRY_WAIT_SEC * (2**attempt))
                continue
            if status_code == 500:
                if attempt >= ANTHROPIC_500_MAX_RETRIES:
                    raise
                time.sleep(ANTHROPIC_500_RETRY_WAIT_SEC * (2**attempt))
                continue
            if status_code is None:
                raise
            raise
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
    response = anthropic_messages_create(
        client,
        model=HAIKU_MODEL,
        max_tokens=1200,
        system=_STATIC_CACHED_SYSTEM_BLOCKS["haiku_image_extraction"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract holdings from this brokerage screenshot."},
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
    opt_snapshot: Dict[str, Any] = {}
    if not options_dates:
        opt_snapshot = {"options_chain_unavailable": True}
    else:
        try:
            nearest_expiry = options_dates[0]
            chain = tk.option_chain(nearest_expiry)
            calls_empty = chain.calls.empty
            puts_empty = chain.puts.empty
            if calls_empty and puts_empty:
                opt_snapshot = {"options_chain_unavailable": True}
            else:
                call_oi = float(chain.calls["openInterest"].fillna(0).sum()) if not calls_empty else 0.0
                put_oi = float(chain.puts["openInterest"].fillna(0).sum()) if not puts_empty else 0.0
                if call_oi + put_oi <= 0.0:
                    opt_snapshot = {"options_chain_unavailable": True}
                else:
                    call_put_ratio = (call_oi / put_oi) if put_oi > 0 else None
                    call_iv = (
                        float(chain.calls["impliedVolatility"].dropna().mean()) if not calls_empty else None
                    )
                    put_iv = (
                        float(chain.puts["impliedVolatility"].dropna().mean()) if not puts_empty else None
                    )
                    opt_snapshot = {
                        "nearest_expiry": nearest_expiry,
                        "call_open_interest": call_oi,
                        "put_open_interest": put_oi,
                        "call_put_oi_ratio": call_put_ratio,
                        "avg_call_iv": call_iv,
                        "avg_put_iv": put_iv,
                        "options_chain_unavailable": False,
                    }
        except Exception:
            opt_snapshot = {"options_chain_unavailable": True}

    # Summarize analyst recommendations (Yahoo Finance analyst data via yfinance)
    rec_summary = {"buy": 0, "hold": 0, "sell": 0, "sources": []}
    if _claude_debug_payload_enabled():
        rec_cols = (
            list(recommendations.columns)
            if isinstance(recommendations, pd.DataFrame)
            else []
        )
        up_cols = list(upgrades.columns) if isinstance(upgrades, pd.DataFrame) else []
        print(f"[DEBUG] {label} tk.recommendations columns: {rec_cols}", flush=True)
        print(f"[DEBUG] {label} tk.upgrades_downgrades columns: {up_cols}", flush=True)

    def _extract_grade_text(row: pd.Series) -> str:
        # yfinance schema varies by endpoint/version.
        for key in ("To Grade", "toGrade", "Action"):
            raw = row.get(key)
            if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
                return str(raw).lower().strip()
        return ""

    def _summarize_analyst_df(df: pd.DataFrame, *, max_rows: int = 60) -> Dict[str, Any]:
        out = {"buy": 0, "hold": 0, "sell": 0, "sources": []}
        if not isinstance(df, pd.DataFrame) or df.empty:
            return out
        latest = df.tail(max_rows).copy()
        for _, row in latest.iterrows():
            grade_text = _extract_grade_text(row)
            firm = str(row.get("Firm", "")).strip()
            if any(x in grade_text for x in ["buy", "overweight", "outperform", "strong buy", "upgrade", "up"]):
                out["buy"] += 1
            elif any(x in grade_text for x in ["hold", "neutral", "market perform", "equal weight", "maintain"]):
                out["hold"] += 1
            elif any(x in grade_text for x in ["sell", "underperform", "underweight", "downgrade", "down"]):
                out["sell"] += 1
            if firm:
                out["sources"].append(firm)
        return out

    rec_from_recommendations = _summarize_analyst_df(recommendations, max_rows=60)
    rec_from_upgrades = _summarize_analyst_df(upgrades, max_rows=120)

    total_recs = (
        rec_from_recommendations["buy"]
        + rec_from_recommendations["hold"]
        + rec_from_recommendations["sell"]
    )
    total_upgrades = (
        rec_from_upgrades["buy"]
        + rec_from_upgrades["hold"]
        + rec_from_upgrades["sell"]
    )

    chosen = rec_from_recommendations if total_recs > 0 else rec_from_upgrades
    rec_summary["buy"] = int(chosen["buy"])
    rec_summary["hold"] = int(chosen["hold"])
    rec_summary["sell"] = int(chosen["sell"])
    rec_summary["sources"].extend(chosen["sources"])
    if isinstance(upgrades, pd.DataFrame) and not upgrades.empty and "Firm" in upgrades.columns:
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


def get_live_price_refresh_snapshot(yf_symbol: str) -> Dict[str, float | None]:
    """Single lightweight yfinance info fetch for live price/52W refresh on cache hits."""
    try:
        tk = yf.Ticker(yf_symbol)
        info = tk.info or {}
    except Exception:
        info = {}
    return {
        "current_price": _as_float(info.get("currentPrice") or info.get("regularMarketPrice")),
        "fifty_two_week_high": _as_float(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low": _as_float(info.get("fiftyTwoWeekLow")),
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


def fmt_human_money(value: Any) -> str:
    """Format large currency values using T/B/M suffixes."""
    if value is None:
        return "—"
    try:
        x = float(value)
        if math.isnan(x):
            return "—"
    except (TypeError, ValueError):
        return "—"
    sign = "-" if x < 0 else ""
    ax = abs(x)
    if ax >= 1_000_000_000_000:
        return f"{sign}${ax / 1_000_000_000_000:.1f}T"
    if ax >= 1_000_000_000:
        return f"{sign}${ax / 1_000_000_000:.1f}B"
    if ax >= 1_000_000:
        return f"{sign}${ax / 1_000_000:.1f}M"
    return f"{sign}${ax:.2f}"


def _iv_as_decimal(iv: float | None) -> float | None:
    """Normalize IV to 0–1 style for threshold labels (yfinance usually uses decimals)."""
    if iv is None:
        return None
    if iv > 1.0 and iv <= 100.0:
        return iv / 100.0
    return iv


def interpret_call_put_oi_ratio(ratio: float | None) -> str:
    if ratio is None or (isinstance(ratio, float) and math.isnan(ratio)):
        return "Options data unavailable for this ticker."
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

    chain_unavailable = bool(opt.get("options_chain_unavailable"))

    call_iv = None if chain_unavailable else _as_float(opt.get("avg_call_iv"))
    put_iv = None if chain_unavailable else _as_float(opt.get("avg_put_iv"))
    implied_volatility: float | None = None
    if not chain_unavailable:
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
            "call_put_open_interest_ratio": (
                None if chain_unavailable else _as_float(opt.get("call_put_oi_ratio"))
            ),
            "implied_volatility": implied_volatility,
            "nearest_expiry": None if chain_unavailable else opt.get("nearest_expiry"),
            "options_chain_unavailable": chain_unavailable,
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
    market_cap = (normalized_payload.get("price_snapshot") or {}).get("market_cap")
    inst = normalized_payload.get("institutional_positioning") or {
        "top_institutional_holders": [],
        "major_holders": [],
    }
    summaries = build_web_research_summaries_for_stock(client, ticker, sector, market_cap)

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
        "system": _STATIC_CACHED_SYSTEM_BLOCKS["sonnet_stock"],
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
        system=_STATIC_CACHED_SYSTEM_BLOCKS["sonnet_portfolio"],
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
    sb_client: Any | None,
    holdings: List[Dict[str, Any]],
    scoring_weights_run: Dict[str, float],
    *,
    show_progress: bool,
    status_placeholder: Any | None = None,
    on_stock_complete: Callable[[int, str, Dict[str, Any]], None] | None = None,
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
        try:
            ticker = h["ticker"]
            if show_progress and status_placeholder is not None:
                status_placeholder.info(f"Analyzing {ticker} ({idx + 1}/{len(holdings)})...")
            yf_symbol = yfinance_lookup_symbol(ticker)
            cached_row = fetch_cached_stock_analysis(sb_client, ticker, max_age_days=14)
            if isinstance(cached_row, dict) and isinstance(cached_row.get("full_result"), dict):
                cached_full = cached_row.get("full_result") or {}
                normalized = dict(cached_full.get("normalized") or {})
                analysis = dict(cached_full.get("analysis") or {})
                web_summaries = list(cached_full.get("web_summaries") or [])
                live_px = get_live_price_refresh_snapshot(yf_symbol)
                px = dict(normalized.get("price_snapshot") or {})
                px["current_price"] = live_px.get("current_price")
                px["fifty_two_week_high"] = live_px.get("fifty_two_week_high")
                px["fifty_two_week_low"] = live_px.get("fifty_two_week_low")
                normalized["price_snapshot"] = px
                normalized["_cache_meta"] = {
                    "from_cache": True,
                    "analyzed_at": str(cached_row.get("analyzed_at") or ""),
                }
            else:
                raw = get_stock_raw_data(
                    yf_symbol,
                    display_ticker=ticker,
                )
                normalized = normalize_stock_data(raw)
                analysis, web_summaries = analyze_with_sonnet(
                    client,
                    normalized,
                    h["quantity"],
                    scoring_weights_run,
                )
                normalized["_cache_meta"] = {"from_cache": False}
                print(f"DEBUG CACHE SAVE ATTEMPT: ticker={ticker}, sb_client is None: {sb_client is None}", flush=True)
                try:
                    save_stock_analysis_cache(
                        sb_client,
                        ticker=ticker,
                        full_result={
                            "normalized": normalized,
                            "analysis": analysis,
                            "web_summaries": web_summaries,
                        },
                        composite_score=_as_float(analysis.get("composite_score_10")),
                        action=str(analysis.get("action") or ""),
                    )
                    print(f"DEBUG CACHE SAVE SUCCESS: {ticker}", flush=True)
                    print(f"DEBUG CACHE SAVE DONE: {ticker}", flush=True)
                    time.sleep(3)
                except Exception as e:
                    print(f"DEBUG CACHE SAVE FAILED: {ticker} error: {e}", flush=True)
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
            if on_stock_complete is not None:
                on_stock_complete(idx + 1, ticker, analysis)
            if prog is not None:
                prog.progress((idx + 1) / len(holdings))
        except Exception as e:
            print(f"Stock analysis failed: ticker={h.get('ticker', '?')} error: {e}", flush=True)
            ft = str(h.get("ticker", "")).upper().strip() or "?"
            failed_analysis: Dict[str, Any] = {
                "composite_score_10": 5.0,
                "action": "Hold",
                "_analysis_failed": True,
                "failure_reason": str(e),
            }
            failed_normalized: Dict[str, Any] = {
                "ticker": ft,
                "price_snapshot": {},
                "analyst_consensus": {
                    "buy_count": 0,
                    "hold_count": 0,
                    "sell_count": 0,
                    "sources": [],
                },
                "options_sentiment": {},
            }
            normalized_map[ft] = failed_normalized
            analysis_map[ft] = failed_analysis
            research_by_ticker[ft] = []
            all_stock_records.append(
                {
                    "ticker": ft,
                    "quantity": h.get("quantity", 0),
                    "normalized": failed_normalized,
                    "analysis": failed_analysis,
                }
            )
            if on_stock_complete is not None:
                on_stock_complete(idx + 1, ft, failed_analysis)
            if prog is not None:
                prog.progress((idx + 1) / len(holdings))
            continue

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
    sb_client = get_supabase_client_singleton()
    user_id = st.session_state.get("user_id")

    if not user_id:
        st.subheader("Sign in")
        login_email = st.text_input(
            "Email",
            key="login_email",
            placeholder="you@example.com",
        ).strip()
        if st.button("Continue", key="login_continue"):
            if not login_email or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", login_email):
                st.error("Enter a valid email address.")
            elif sb_client is None:
                st.error("Supabase is required for email login. Configure SUPABASE_URL and SUPABASE_KEY.")
            else:
                resolved_user_id = get_or_create_user_by_email(sb_client, login_email)
                if not resolved_user_id:
                    st.error("Could not sign in with Supabase. Check users table access.")
                else:
                    st.session_state["user_id"] = resolved_user_id
                    st.session_state["user_email"] = login_email.lower()
                    st.rerun()
        st.stop()

    user_id = str(st.session_state.get("user_id"))

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
        batch_run_id = str(uuid4())
        scoring_weights_run = dict(
            st.session_state.get("scoring_weights") or DEFAULT_SCORING_WEIGHTS
        )
        batch_weights_used = {**scoring_weights_run, "_run_mode": "batch"}
        batch_analysis_map: Dict[str, Dict[str, Any]] = {}

        def _save_batch_partial(completed_count: int, _ticker: str, analysis: Dict[str, Any]) -> None:
            batch_analysis_map[_ticker] = analysis
            partial_holdings = holdings_batch[:completed_count]
            upsert_portfolio_run_snapshot(
                sb_client,
                run_id=batch_run_id,
                user_id=user_id,
                holdings=partial_holdings,
                results_summary=build_results_summary_rows(partial_holdings, batch_analysis_map),
                weights_used=batch_weights_used,
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
            sb_client,
            holdings_batch,
            scoring_weights_run,
            show_progress=False,
            status_placeholder=None,
            on_stock_complete=_save_batch_partial,
        )
        results_summary = build_results_summary_rows(holdings_batch, analysis_map)
        upsert_portfolio_run_snapshot(
            sb_client,
            run_id=batch_run_id,
            user_id=user_id,
            holdings=holdings_batch,
            results_summary=results_summary,
            weights_used=batch_weights_used,
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

    def _sync_holdings_editor_from_manual_input() -> None:
        parsed = parse_manual_holdings(st.session_state.get("manual_holdings_input", ""))
        st.session_state["manual_holdings_parsed_rows"] = (
            parsed if parsed else [{"ticker": "", "quantity": 0.0}]
        )
        st.session_state["holdings_editor_version"] = int(
            st.session_state.get("holdings_editor_version", 0)
        ) + 1

    staged_manual_holdings = st.session_state.pop("manual_holdings_input_staged", None)
    if staged_manual_holdings is not None:
        st.session_state["manual_holdings_input"] = staged_manual_holdings
        _sync_holdings_editor_from_manual_input()

    if "manual_holdings_input" not in st.session_state:
        loaded_holdings = load_saved_portfolio(sb_client, user_id)
        st.session_state["manual_holdings_input"] = format_holdings_as_manual_input(loaded_holdings)
    if "manual_holdings_parsed_rows" not in st.session_state:
        _sync_holdings_editor_from_manual_input()

    st.subheader("Portfolio Input")
    st.text_area(
        "Manual input (format: TICKER:QTY, TICKER:QTY)",
        placeholder="AAPL:12, MSFT:8, NVDA:5",
        key="manual_holdings_input",
        on_change=_sync_holdings_editor_from_manual_input,
    )
    apply_col, _ = st.columns([1, 6])
    with apply_col:
        if st.button("✔ Apply", key="apply_manual_holdings_input"):
            _sync_holdings_editor_from_manual_input()
            st.rerun()
    uploaded = st.file_uploader("Or upload brokerage screenshot", type=["png", "jpg", "jpeg", "webp"])

    if st.button("Extract holdings from screenshot"):
        if not uploaded:
            st.warning("Upload an image first.")
        else:
            with st.spinner("Extracting holdings with Claude Haiku Vision..."):
                image_b64 = image_to_base64(uploaded)
                extracted = extract_holdings_from_image(client, image_b64, uploaded.type or "image/png")
                st.session_state["manual_holdings_input_staged"] = format_holdings_as_manual_input(extracted)
                st.rerun()

    st.write("Confirm or edit holdings before analysis (updates from the text field as you type):")
    edit_df = pd.DataFrame(st.session_state.get("manual_holdings_parsed_rows") or [{"ticker": "", "quantity": 0.0}])
    editor_key = f"holdings_editor_v{int(st.session_state.get('holdings_editor_version', 0))}"
    edited = st.data_editor(edit_df, num_rows="dynamic", use_container_width=True, key=editor_key)
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
        manual_run_id = str(uuid4())
        live_weights_used = {
            **(st.session_state.get("scoring_weights") or DEFAULT_SCORING_WEIGHTS),
            "_run_mode": "manual",
        }
        live_progress_box = st.empty()
        progressive_rows: List[Dict[str, Any]] = []
        live_analysis_map: Dict[str, Dict[str, Any]] = {}

        def _save_and_render_partial(completed_count: int, ticker: str, analysis: Dict[str, Any]) -> None:
            live_analysis_map[ticker] = analysis
            score_v = analysis.get("composite_score_10")
            try:
                score_disp = f"{float(score_v):.2f}"
            except (TypeError, ValueError):
                score_disp = "N/A"
            progressive_rows.append(
                {
                    "Ticker": ticker,
                    "Composite Score (0-10)": score_disp,
                    "Action": analysis.get("action") or "Hold",
                }
            )
            partial_holdings = holdings[:completed_count]
            upsert_portfolio_run_snapshot(
                sb_client,
                run_id=manual_run_id,
                user_id=user_id,
                holdings=partial_holdings,
                results_summary=build_results_summary_rows(partial_holdings, live_analysis_map),
                weights_used=live_weights_used,
            )
            with live_progress_box.container():
                st.info(f"Completed {completed_count}/{len(holdings)} stocks")
                st.dataframe(pd.DataFrame(progressive_rows), use_container_width=True, hide_index=True)

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
                sb_client,
                holdings,
                scoring_weights_run,
                show_progress=True,
                status_placeholder=status_box,
                on_stock_complete=_save_and_render_partial,
            )
            results_summary = build_results_summary_rows(holdings, analysis_map)
            upsert_portfolio_run_snapshot(
                sb_client,
                run_id=manual_run_id,
                user_id=user_id,
                holdings=holdings,
                results_summary=results_summary,
                weights_used=live_weights_used,
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
            cache_meta = normalized.get("_cache_meta") if isinstance(normalized, dict) else {}
            if isinstance(cache_meta, dict) and cache_meta.get("from_cache"):
                cached_at = str(cache_meta.get("analyzed_at") or "").strip()
                stamp = cached_at if cached_at else "recent date"
                st.caption(f"AI analysis cached from {stamp}. Price refreshed live.")
            px = normalized.get("price_snapshot", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", fmt_two_dec(px.get("current_price"), prefix="$"))
            c2.metric("Market Cap", fmt_human_money(px.get("market_cap")))
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
                    st.caption(
                        "Confidence reflects strength of available evidence supporting this stance, not a price prediction."
                    )
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
            chain_unavail = bool(opt.get("options_chain_unavailable"))
            cp_ratio = _as_float(opt.get("call_put_open_interest_ratio"))
            iv_raw = _as_float(opt.get("implied_volatility"))
            nearest = opt.get("nearest_expiry")
            short_pct = _as_float(normalized.get("short_interest_pct_float"))

            st.markdown("**Call / put open interest ratio**")
            if chain_unavail:
                st.markdown("Options data unavailable for this ticker")
            else:
                st.markdown(interpret_call_put_oi_ratio(cp_ratio))

            st.markdown("**Implied volatility (nearest expiry chain)**")
            st.caption(
                "IV measures the market's expectation of future price swings. Higher IV means larger expected moves in either direction."
            )
            st.markdown(interpret_implied_volatility(iv_raw))

            st.markdown("**Nearest options expiry**")
            if nearest:
                st.markdown(str(nearest))
            else:
                st.markdown("Not available")

            st.markdown("**Short interest (% of float)**")
            st.markdown(interpret_short_interest_pct(short_pct))
            oi_ratio_disp = "N/A" if chain_unavail else fmt_two_dec(cp_ratio)
            iv_disp = "N/A" if chain_unavail else fmt_two_dec(iv_raw)
            st.caption(
                f"Raw metrics — OI ratio: {oi_ratio_disp} | "
                f"IV: {iv_disp} | "
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
