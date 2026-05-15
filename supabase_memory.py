from __future__ import annotations

# SQL for Supabase SQL editor (create cache table if missing):
# ------------------------------------------------------------
# create table if not exists public.stock_cache (
#   id uuid primary key default gen_random_uuid(),
#   ticker text not null,
#   analyzed_at timestamptz not null default now(),
#   full_result jsonb not null,
#   composite_score double precision,
#   action text
# );
# create index if not exists stock_cache_ticker_analyzed_idx
#   on public.stock_cache (ticker, analyzed_at desc);
#
# If inserts fail or rows never appear for the app but work in the SQL editor, check RLS:
# the role in SUPABASE_KEY must be allowed INSERT and SELECT on rows you need to read back.
#
# create table if not exists public.saved_portfolio (
#   user_id text primary key,
#   tickers_json jsonb not null default '[]'::jsonb,
#   updated_at timestamptz not null default now()
# );
#
# create table if not exists public.portfolio_runs (
#   id uuid primary key default gen_random_uuid(),
#   user_id text not null,
#   run_timestamp timestamptz not null default now(),
#   holdings jsonb not null,
#   results_summary jsonb not null,
#   weights_used jsonb not null
# );
#
# create index if not exists portfolio_runs_user_ts_idx
#   on public.portfolio_runs (user_id, run_timestamp desc);
# ------------------------------------------------------------

from datetime import datetime, timedelta, timezone
import json
import math
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st


def get_or_create_user_by_email(client: Optional[httpx.Client], email: str) -> Optional[str]:
    if client is None:
        return None
    email_norm = str(email or "").strip().lower()
    if not email_norm:
        return None
    try:
        resp = client.get(
            "/users",
            params={
                "select": "id,email",
                "email": f"eq.{email_norm}",
                "limit": 1,
            },
        )
        resp.raise_for_status()
        data = resp.json() or []
        if isinstance(data, list) and data and isinstance(data[0], dict):
            uid = data[0].get("id")
            return str(uid) if uid else None
    except Exception:
        pass
    try:
        resp = client.post(
            "/users",
            params={"on_conflict": "email"},
            json={"email": email_norm},
            headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        )
        resp.raise_for_status()
        data = resp.json() or []
        if isinstance(data, list) and data and isinstance(data[0], dict):
            uid = data[0].get("id")
            return str(uid) if uid else None
    except Exception:
        return None
    return None


def get_supabase_client_singleton() -> Optional[httpx.Client]:
    if "supabase_client" in st.session_state and st.session_state["supabase_client"] is not None:
        return st.session_state["supabase_client"]
    try:
        supabase_url = str(st.secrets.get("SUPABASE_URL", "")).strip()
        supabase_key = str(st.secrets.get("SUPABASE_KEY", "")).strip()
    except Exception:
        supabase_url = ""
        supabase_key = ""
    if not supabase_url or not supabase_key:
        st.session_state["supabase_client"] = None
        return None
    try:
        base = supabase_url.rstrip("/")
        # Accept either project root URL or full REST base (common misconfiguration).
        if base.endswith("/rest/v1"):
            base = base[: -len("/rest/v1")].rstrip("/")
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Accept-Profile": "public",
            "Content-Profile": "public",
        }
        client = httpx.Client(
            base_url=f"{base}/rest/v1",
            headers=headers,
            timeout=20.0,
        )
        st.session_state["supabase_client"] = client
        return client
    except Exception:
        st.session_state["supabase_client"] = None
        return None


def _safe_number(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _normalize_holdings_shape(raw: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).upper().strip()
        qty = _safe_number(item.get("quantity"))
        if ticker and qty is not None and qty > 0:
            out.append({"ticker": ticker, "quantity": qty})
    return out


def load_saved_portfolio(client: Optional[httpx.Client], user_id: str) -> List[Dict[str, Any]]:
    if client is None:
        return []
    try:
        resp = client.get(
            "/saved_portfolio",
            params={
                "select": "tickers_json",
                "user_id": f"eq.{user_id}",
                "limit": 1,
            },
        )
        resp.raise_for_status()
        data = resp.json() or []
        if not data:
            return []
        row = data[0] if isinstance(data[0], dict) else {}
        return _normalize_holdings_shape(row.get("tickers_json"))
    except Exception:
        return []


def save_portfolio_run(
    client: Optional[httpx.Client],
    *,
    user_id: str,
    holdings: List[Dict[str, Any]],
    results_summary: List[Dict[str, Any]],
    weights_used: Dict[str, float],
) -> None:
    if client is None:
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    run_payload = {
        "user_id": user_id,
        "run_timestamp": now_iso,
        "holdings": holdings,
        "results_summary": results_summary,
        "weights_used": weights_used,
    }
    memory_payload = {
        "user_id": user_id,
        "tickers_json": holdings,
        "updated_at": now_iso,
    }
    try:
        client.post(
            "/portfolio_runs",
            json=run_payload,
            headers={"Prefer": "return=minimal"},
        ).raise_for_status()
        client.post(
            "/saved_portfolio",
            params={"on_conflict": "user_id"},
            json=memory_payload,
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        ).raise_for_status()
    except Exception:
        return


def upsert_portfolio_run_snapshot(
    client: Optional[httpx.Client],
    *,
    run_id: str,
    user_id: str,
    holdings: List[Dict[str, Any]],
    results_summary: List[Dict[str, Any]],
    weights_used: Dict[str, float],
) -> None:
    if client is None:
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    run_payload = {
        "id": run_id,
        "user_id": user_id,
        "run_timestamp": now_iso,
        "holdings": holdings,
        "results_summary": results_summary,
        "weights_used": weights_used,
    }
    memory_payload = {
        "user_id": user_id,
        "tickers_json": holdings,
        "updated_at": now_iso,
    }
    try:
        client.post(
            "/portfolio_runs",
            params={"on_conflict": "id"},
            json=run_payload,
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        ).raise_for_status()
        client.post(
            "/saved_portfolio",
            params={"on_conflict": "user_id"},
            json=memory_payload,
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
        ).raise_for_status()
    except Exception:
        return


def fetch_recent_runs(client: Optional[httpx.Client], user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    if client is None:
        return []
    try:
        resp = client.get(
            "/portfolio_runs",
            params={
                "select": "id,run_timestamp,holdings,results_summary,weights_used",
                "user_id": f"eq.{user_id}",
                "order": "run_timestamp.desc",
                "limit": limit,
            },
        )
        resp.raise_for_status()
        data = resp.json() or []
        return data if isinstance(data, list) else []
    except Exception:
        return []


def fetch_cached_stock_analysis(
    client: Optional[httpx.Client],
    ticker: str,
    *,
    max_age_days: int = 14,
) -> Optional[Dict[str, Any]]:
    if client is None:
        return None
    t = str(ticker or "").upper().strip()
    if not t:
        return None
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
    try:
        resp = client.get(
            "/stock_cache",
            params={
                "select": "id,ticker,analyzed_at,full_result,composite_score,action",
                "ticker": f"eq.{t}",
                "analyzed_at": f"gt.{cutoff}",
                "order": "analyzed_at.desc",
                "limit": 1,
            },
        )
        resp.raise_for_status()
        data = resp.json() or []
        if isinstance(data, list) and data and isinstance(data[0], dict):
            row = data[0]
            fr = row.get("full_result")
            if isinstance(fr, str):
                try:
                    row = {**row, "full_result": json.loads(fr)}
                except json.JSONDecodeError:
                    pass
            return row
    except Exception:
        return None
    return None


def _sanitize_non_finite(obj: Any) -> Any:
    """Replace NaN/Inf so JSON is valid for PostgreSQL jsonb (no NaN/Infinity tokens)."""
    if obj is None:
        return None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_non_finite(v) for v in obj]
    if isinstance(obj, set):
        return [_sanitize_non_finite(v) for v in obj]
    return obj


def _json_safe_for_postgrest(obj: Any) -> Any:
    """Round-trip through JSON so numpy/pandas/Timestamp values become plain Python types."""
    cleaned = _sanitize_non_finite(obj)
    try:
        return json.loads(json.dumps(cleaned, default=str, allow_nan=False))
    except (TypeError, ValueError) as e:
        raise ValueError(f"not JSON-serializable: {e}") from e


def _parse_analyzed_at_loose(val: Any) -> Optional[datetime]:
    """Parse timestamps returned by PostgREST / Postgres (ISO or 'YYYY-MM-DD HH:MM:SS+00')."""
    if val is None:
        return None
    text = str(val).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    if len(text) >= 11 and text[10:11] == " ":
        text = text[:10] + "T" + text[11:]
    if text.endswith("+00") and not text.endswith("+00:00"):
        text = text + ":00"
    if text.endswith("-00") and not text.endswith("-00:00"):
        text = text[:-3] + "-00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _verify_stock_cache_row_readable(
    client: httpx.Client,
    *,
    ticker: str,
    analyzed_at_iso: str,
) -> bool:
    """
    Confirm the row is visible with the same REST role used for INSERT.

    PostgREST can return HTTP 201 with an empty list for Prefer: return=representation
    when RLS allows INSERT but filters rows out of the RETURNING set. A follow-up GET
    matches what fetch_cached_stock_analysis will see later.
    """
    t = str(ticker or "").upper().strip()
    if not t:
        return False
    try:
        ins_dt = datetime.fromisoformat(str(analyzed_at_iso).replace("Z", "+00:00"))
    except ValueError:
        ins_dt = datetime.now(timezone.utc)
    cutoff_dt = ins_dt - timedelta(seconds=10)
    cutoff_iso = cutoff_dt.astimezone(timezone.utc).isoformat()
    try:
        resp = client.get(
            "/stock_cache",
            params={
                "select": "ticker,analyzed_at",
                "ticker": f"eq.{t}",
                "analyzed_at": f"gte.{cutoff_iso}",
                "order": "analyzed_at.desc",
                "limit": 1,
            },
        )
        resp.raise_for_status()
        rows = resp.json() or []
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            return False
        r0 = rows[0]
        if str(r0.get("ticker", "")).upper() != t:
            return False
        row_ts = _parse_analyzed_at_loose(r0.get("analyzed_at"))
        if row_ts is None:
            return True
        return row_ts >= ins_dt - timedelta(seconds=15)
    except Exception:
        return False


def save_stock_analysis_cache(
    client: Optional[httpx.Client],
    *,
    ticker: str,
    full_result: Dict[str, Any],
    composite_score: Optional[float],
    action: str,
) -> Optional[str]:
    """Insert cache row. Returns None on success, or an error string on failure."""
    if client is None:
        return "Supabase client not configured (missing SUPABASE_URL / SUPABASE_KEY)."
    t = str(ticker or "").upper().strip()
    if not t:
        return "Empty ticker."
    try:
        safe_full = _json_safe_for_postgrest(full_result)
    except ValueError as e:
        return str(e)
    cs: float | None = None
    if composite_score is not None:
        try:
            cf = float(composite_score)
            cs = cf if math.isfinite(cf) else None
        except (TypeError, ValueError):
            cs = None
    payload = {
        "ticker": t,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "full_result": safe_full,
        "composite_score": cs,
        "action": str(action or "")[:64],
    }
    try:
        resp = client.post(
            "/stock_cache",
            json=payload,
            headers={"Prefer": "return=representation"},
        )
        if resp.status_code >= 400:
            return f"HTTP {resp.status_code}: {resp.text[:800]}"
        body: Any = None
        try:
            body = resp.json()
        except Exception:
            body = None
        ok_repr = (
            isinstance(body, list)
            and len(body) > 0
            and isinstance(body[0], dict)
            and str(body[0].get("ticker", "")).upper() == t
        )
        if ok_repr:
            return None
        if resp.status_code not in (200, 201):
            return f"Unexpected HTTP {resp.status_code} after insert: {resp.text[:500]!r}"
        if _verify_stock_cache_row_readable(client, ticker=t, analyzed_at_iso=payload["analyzed_at"]):
            return None
        return (
            "Insert did not return a row and read-back found nothing for this ticker in the last ~15s. "
            "If the row appears in the SQL editor but not here, check RLS policies on public.stock_cache "
            "for the key in SUPABASE_KEY (anon must be allowed INSERT and SELECT on rows it inserts). "
            f"HTTP {resp.status_code} representation body: {resp.text[:600]!r}"
        )
    except Exception as e:
        return str(e)