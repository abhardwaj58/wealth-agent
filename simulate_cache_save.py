"""
Standalone cache-save smoke test (no Streamlit UI, no Anthropic).

Uses the same get_supabase_client_singleton + save_stock_analysis_cache as app.py.

Run from the project root so Streamlit can load .streamlit/secrets.toml:
  python simulate_cache_save.py

You may see Streamlit warnings about missing ScriptRunContext; they are harmless here.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when run as: python simulate_cache_save.py
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from supabase_memory import (  # noqa: E402
    fetch_cached_stock_analysis,
    get_supabase_client_singleton,
    save_stock_analysis_cache,
)

# Distinct ticker so you can spot / delete this row in Supabase if needed.
TEST_TICKER = "STANDALONE_CACHE_SIM"


def main() -> int:
    client = get_supabase_client_singleton()
    if client is None:
        print(
            "FAIL: get_supabase_client_singleton() returned None. "
            "Set SUPABASE_URL and SUPABASE_KEY in .streamlit/secrets.toml.",
            flush=True,
        )
        return 1

    dummy_full_result = {
        "normalized": {"ticker": TEST_TICKER, "price_snapshot": {"current_price": 123.45}},
        "analysis": {
            "composite_score_10": 7.25,
            "action": "Buy",
            "thesis": "standalone simulation only",
        },
        "web_summaries": [],
    }

    print(f"Saving cache row for {TEST_TICKER} (same path as app)...", flush=True)
    err = save_stock_analysis_cache(
        client,
        ticker=TEST_TICKER,
        full_result=dummy_full_result,
        composite_score=7.25,
        action="Buy",
    )
    if err:
        print(f"FAIL: save_stock_analysis_cache: {err}", flush=True)
        return 4

    row = fetch_cached_stock_analysis(client, TEST_TICKER, max_age_days=14)
    if not row:
        print(
            "FAIL: save returned without raising, but fetch_cached_stock_analysis "
            "found no fresh row. Check RLS, table name, or PostgREST errors above.",
            flush=True,
        )
        return 2

    fr = row.get("full_result") or {}
    ok = (
        isinstance(fr, dict)
        and (fr.get("analysis") or {}).get("action") == "Buy"
        and float(row.get("composite_score") or 0) == 7.25
    )
    if not ok:
        print(f"FAIL: unexpected row payload: {row!r}", flush=True)
        return 3

    print("OK: row saved and readable via fetch_cached_stock_analysis.", flush=True)
    print(f"  ticker={row.get('ticker')} analyzed_at={row.get('analyzed_at')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
