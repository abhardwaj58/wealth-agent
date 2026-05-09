#!/usr/bin/env python3
"""Standalone test: insert then delete a row in Supabase stock_cache via REST."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import tomllib

ROOT = Path(__file__).resolve().parent
SECRETS_PATH = ROOT / ".streamlit" / "secrets.toml"

TEST_TICKER = "TEST2"


def main() -> int:
    if not SECRETS_PATH.is_file():
        print(f"ERROR: secrets file not found: {SECRETS_PATH}", file=sys.stderr)
        return 1

    with SECRETS_PATH.open("rb") as f:
        secrets = tomllib.load(f)

    supabase_url = str(secrets.get("SUPABASE_URL", "")).strip().rstrip("/")
    supabase_key = str(secrets.get("SUPABASE_KEY", "")).strip()
    if not supabase_url or not supabase_key:
        print("ERROR: SUPABASE_URL or SUPABASE_KEY missing in secrets.toml", file=sys.stderr)
        return 1

    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json",
    }
    base_url = f"{supabase_url}/rest/v1"

    payload = {
        "ticker": TEST_TICKER,
        "full_result": {"test": True, "note": "dummy cache row from test_cache_save.py"},
        "composite_score": 5.0,
        "action": "Hold",
    }

    with httpx.Client(base_url=base_url, headers=headers, timeout=30.0) as client:
        try:
            resp = client.post(
                "/stock_cache",
                json=payload,
                headers={"Prefer": "return=minimal"},
            )
            resp.raise_for_status()
            print("SUCCESS: inserted row into stock_cache", flush=True)
        except httpx.HTTPStatusError as e:
            print(f"INSERT FAILED (HTTP): {e}", flush=True)
            print(f"Response status: {e.response.status_code}", flush=True)
            print(f"Response body: {e.response.text}", flush=True)
            return 1
        except Exception as e:
            print(f"INSERT FAILED: {type(e).__name__}: {e}", flush=True)
            return 1

        try:
            del_resp = client.delete(
                "/stock_cache",
                params={"ticker": f"eq.{TEST_TICKER}"},
                headers={"Prefer": "return=minimal"},
            )
            del_resp.raise_for_status()
            print(f"SUCCESS: deleted stock_cache rows where ticker={TEST_TICKER}", flush=True)
        except httpx.HTTPStatusError as e:
            print(f"DELETE FAILED (HTTP): {e}", flush=True)
            print(f"Response status: {e.response.status_code}", flush=True)
            print(f"Response body: {e.response.text}", flush=True)
            return 1
        except Exception as e:
            print(f"DELETE FAILED: {type(e).__name__}: {e}", flush=True)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
