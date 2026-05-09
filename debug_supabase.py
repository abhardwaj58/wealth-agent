import tomllib
import httpx

with open('.streamlit/secrets.toml', 'rb') as f:
    s = tomllib.load(f)

c = httpx.Client(
    base_url=s['SUPABASE_URL'].rstrip('/') + '/rest/v1',
    headers={
        'apikey': s['SUPABASE_KEY'],
        'Authorization': 'Bearer ' + s['SUPABASE_KEY'],
        'Content-Type': 'application/json'
    }
)

r = c.post(
    '/stock_cache',
    json={
        'ticker': 'DEBUG1',
        'analyzed_at': '2026-05-09T00:00:00+00:00',
        'full_result': {'normalized': {}, 'analysis': {}, 'web_summaries': []},
        'composite_score': 5.0,
        'action': 'Hold'
    },
    headers={'Prefer': 'return=minimal'}
)

print('STATUS:', r.status_code)
print('BODY:', r.text)
