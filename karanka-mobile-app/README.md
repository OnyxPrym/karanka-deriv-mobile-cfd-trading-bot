# KARANKA MULTIVERSE ALGO AI - DERIV BOT

## Deploy on Render:

1. Push these files to GitHub
2. Go to render.com → New Web Service
3. Connect GitHub repo
4. Use:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -k eventlet -w 1 app:app`
5. Add env var: `PYTHON_VERSION = 3.11.11`
6. Deploy!

## Features:
- 49 Markets (Synthetics, Forex, Indices, Crypto)
- Market State Engine
- Swapped Logic (SELL in uptrend, BUY in downtrend)
- 2 Pip Retest
- Trade Tracking
- Black & Gold UI
