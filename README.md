# Wealth Agent

AI-powered portfolio analysis app built with Streamlit, yfinance, Claude (Anthropic), and Supabase-compatible REST persistence.

## Features

- Portfolio input via manual ticker/quantity entry or brokerage screenshot extraction.
- Deep per-stock analysis with:
  - Bull / Bear / Hold cases
  - Weighted composite scoring
  - Analyst + institutional positioning signals
  - Targeted web-research summaries with source-tier weighting
- Portfolio-level recommendations:
  - Concentration risk flags
  - Rebalancing guidance
  - New stock ideas
- Adjustable scoring weights from the sidebar (auto-normalized to 100%).
- Supabase-backed memory and history:
  - Auto-load last saved portfolio
  - Save each run summary
  - View last 10 runs in the Run History tab
- Optional overnight batch scheduling for automated runs.

## Project Structure

- `app.py` - Main Streamlit application
- `supabase_memory.py` - Supabase REST persistence helpers
- `.streamlit/config.toml` - Streamlit config/theme
- `.streamlit/secrets.toml` - Local secrets (not committed)
- `requirements.txt` - Python dependencies

## Prerequisites

- Python 3.10+ recommended
- Anthropic API key
- Supabase project (optional but recommended for memory/history)

## Setup

1. Clone the repo and enter the project directory.
2. Create/activate a virtual environment.
3. Install dependencies:

```bash
py -m pip install -r requirements.txt
```

## Configure `.streamlit/secrets.toml`

Create `.streamlit/secrets.toml` with:

```toml
ANTHROPIC_API_KEY = "your_anthropic_api_key"
SUPABASE_URL = "https://YOUR_PROJECT_REF.supabase.co"
SUPABASE_KEY = "your_supabase_anon_or_service_role_key"
```

Notes:
- `SUPABASE_URL` and `SUPABASE_KEY` are optional for basic usage, but required for:
  - saved portfolio memory
  - run history
  - overnight persistence

## Supabase SQL Setup

Use the SQL comment block at the top of `supabase_memory.py` in your Supabase SQL editor to create:
- `stock_cache` (14-day per-ticker analysis cache)
- `saved_portfolio` and `portfolio_runs` (memory + run history)
- supporting indexes

Per-ticker behavior: within **14 days** of `analyzed_at`, the app loads `full_result` from `stock_cache`, skips new web research and full Sonnet re-run for that ticker, refreshes **live price** and 52-week high/low via yfinance, and shows a disclaimer with the cached date (**MM.DD.YYYY** Pacific).

## Run Locally

```bash
py -m streamlit run app.py
```

Then open the URL shown in terminal (usually `http://localhost:8501`).

## Overnight Batch Notes

- Enable **Schedule overnight analysis** in the sidebar.
- Select desired Pacific time (defaults to 7:00 AM).
- Keep the Streamlit process running for scheduled jobs to execute.

## Troubleshooting

- If analyst consensus appears empty, the app already includes schema-variant handling and fallback parsing.
- If Supabase features are unavailable, verify:
  - `SUPABASE_URL`
  - `SUPABASE_KEY`
  - tables exist (run SQL from `supabase_memory.py`)

<!--
Streamlit Community Cloud deployment (commented instructions):
1. Push this repo to GitHub.
2. Go to share.streamlit.io and create a new app from your repo.
3. Set main file path to: app.py
4. In Streamlit Cloud app settings -> Secrets, paste:
   ANTHROPIC_API_KEY, SUPABASE_URL, SUPABASE_KEY
5. Deploy and verify tabs + Supabase memory/history behavior.
-->
