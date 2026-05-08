# SQL to run in Supabase SQL Editor:
# ------------------------------------------------------------
# -- Enable gen_random_uuid() (usually already enabled on Supabase)
# create extension if not exists pgcrypto;
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

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx
import streamlit as st


def get_or_create_user_id() -> str:
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = str(uuid4())
    return str(st.session_state["user_id"])


def get_supabase_client_singleton() -> Optional[httpx.Client]:
    if "supabase_client" in st.session_state:
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
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
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
    """Upsert an in-flight run snapshot so partial progress is durable."""
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
