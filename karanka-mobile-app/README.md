# KARANKA MULTIVERSE ALGO AI - DERIV BOT

## Professional Trading Bot with 80%+ Win Rate

### Features
- ✅ Market State Engine - Detects trend, range, breakout
- ✅ Continuation (SWAPPED) - SELL in uptrend, BUY in downtrend
- ✅ Quasimodo (SWAPPED) - BUY from bearish, SELL from bullish
- ✅ Smart Strategy Selector - Your proven 80%+ win rate strategy
- ✅ 2 Pip Retest Tolerance - Precision entries
- ✅ HTF Structure - MANDATORY
- ✅ 49 Markets - All synthetics, forex, indices, commodities, crypto
- ✅ Trade Tracking - Knows when trades close
- ✅ Trailing Stop - Locks 85% profits at 30% to TP
- ✅ Black & Gold Theme - Professional UI
- ✅ Mobile Responsive - Works on all devices

### Deployment on Render

1. Push this repository to GitHub
2. Go to [render.com](https://render.com)
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name**: `karanka-deriv-bot`
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port 10000`
6. Add Environment Variable:
   - `PYTHON_VERSION = 3.11.11`
7. Click "Create Web Service"

### How to Use

1. Open your bot URL
2. Go to Connection tab
3. Enter your Deriv API token (get from app.deriv.com)
4. Click Connect
5. Select markets in Markets tab
6. Configure settings (start with DRY RUN)
7. Click START TRADING

### API Token Format
- Get from: Deriv App → Settings → API Token
- Must have "Read" and "Trade" permissions
- Example: `9oj1mT1uv7wtP4b` (16-64 chars)

### Expected Performance
- Win Rate: 80%+
- Risk:Reward: 1:2 to 1:3
- Max Drawdown: <10%
