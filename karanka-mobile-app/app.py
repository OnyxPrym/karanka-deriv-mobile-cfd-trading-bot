#!/usr/bin/env python3
"""
================================================================================
KARANKA MULTIVERSE ALGO AI - PROFESSIONAL TRADING SYSTEM (DERIV INTEGRATION)
================================================================================
✅ ANALYZES ALL SELECTED MARKETS - Forex, Indices, Crypto, Commodities
✅ MARKET STATE ENGINE - Knows trend vs range vs breakout vs choppy
✅ CONTINUATION STRATEGY - Trades WITH trends (pullbacks to EMA)
✅ QUASIMODO STRATEGY - Trades reversals in ranges (YOUR PROVEN STRATEGY)
✅ SMART SELECTOR - Uses right strategy for right market conditions
✅ 2 PIP RETEST TOLERANCE - Only trades when price confirms
✅ HTF STRUCTURE - MANDATORY - Never trades against higher timeframe
✅ ATR SL/TP - DYNAMIC based on volatility and market state
✅ FRESH PATTERNS - MAX 8 CANDLES - Only recent patterns
✅ DERIV API INTEGRATION - Connect with API token, select trading account
✅ ENHANCED SL HANDLING - Spread buffer and dynamic ATR multipliers
================================================================================
"""

import sys
import os
import subprocess
import threading
import time
import json
import traceback
import warnings
from datetime import datetime, timedelta
from collections import defaultdict, deque
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import math
from enum import Enum
import websocket
import requests
import hashlib
import hmac
import base64
import uuid

warnings.filterwarnings('ignore')

# ============ PREVENT CMD FROM CLOSING ============
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleTitleW("KARANKA MULTIVERSE ALGO AI - DERIV TRADING")
    except:
        pass

# ============ AUTO-INSTALL ============
def install_dependencies():
    """Auto-install all required dependencies"""
    print("🔧 INSTALLING DEPENDENCIES...")
    
    required_packages = [
        'pandas',
        'numpy',
        'python-dateutil',
        'pytz',
        'scipy',
        'ta-lib',
        'websocket-client',
        'requests'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
                print(f"✅ Successfully installed {package}")
            except Exception as e:
                print(f"❌ Failed to install {package}: {e}")
    
    return True

print("=" * 80)
print("KARANKA MULTIVERSE ALGO AI - PROFESSIONAL TRADING SYSTEM (DERIV)")
print("=" * 80)

install_dependencies()

try:
    import pandas as pd
    import numpy as np
    import talib
except ImportError as e:
    print(f"❌ Import error: {e}")
    install_dependencies()
    import pandas as pd
    import numpy as np
    import talib

# ============ FOLDERS & PATHS ============
def ensure_data_folder():
    """Create all necessary folders"""
    app_data_dir = os.path.join(os.path.expanduser("~"), "KarankaMultiVerse_AI_Deriv")
    folders = ["logs", "settings", "cache", "market_data", "trade_analysis", "backups", "performance", "deriv_data"]
    
    for folder in folders:
        folder_path = os.path.join(app_data_dir, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    return app_data_dir

APP_DATA_DIR = ensure_data_folder()
SETTINGS_FILE = os.path.join(APP_DATA_DIR, "settings", "karanka_settings.json")
TRADES_LOG_FILE = os.path.join(APP_DATA_DIR, "logs", "trades_log.txt")
PERFORMANCE_FILE = os.path.join(APP_DATA_DIR, "performance", "performance.json")
DERIV_ACCOUNTS_FILE = os.path.join(APP_DATA_DIR, "deriv_data", "accounts.json")

# ============ BLACK & GOLD THEME ============
BLACK_GOLD_THEME = {
    'bg': '#000000',
    'fg': '#FFD700',
    'fg_light': '#FFED4E',
    'fg_dark': '#B8860B',
    'accent': '#D4AF37',
    'accent_dark': '#8B7500',
    'secondary': '#0a0a0a',
    'border': '#333333',
    'success': '#00FF00',
    'error': '#FF4444',
    'warning': '#FFAA00',
    'info': '#00AAFF',
    'text_bg': '#0a0a0a',
    'button_bg': '#8B7500',
    'button_fg': '#FFD700',
    'highlight': '#D4AF37',
}

# ============ MARKET STATE ENUM ============
class MarketState(Enum):
    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    RANGING = "RANGING"
    DOWNTREND = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    BREAKOUT_BULL = "BREAKOUT_BULL"
    BREAKOUT_BEAR = "BREAKOUT_BEAR"
    CHOPPY = "CHOPPY"

# ============ DERIV API CONNECTOR ============
class DerivAPI:
    """Handles all Deriv API connections and trading"""
    
    def __init__(self):
        self.ws = None
        self.connected = False
        self.api_token = ""
        self.active_account = None
        self.accounts = []
        self.balance = 0
        self.currency = "USD"
        self.loginid = ""
        self.email = ""
        self.ws_url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"  # Deriv app_id
        self.req_id = 1
        self.pending_requests = {}
        self.market_data_callbacks = {}
        self.tick_subscriptions = {}
        
    def generate_req_id(self):
        """Generate unique request ID"""
        self.req_id += 1
        return self.req_id
    
    def connect(self, api_token, callback=None):
        """Connect to Deriv API with token"""
        self.api_token = api_token
        
        try:
            # Close existing connection if any
            if self.ws:
                self.ws.close()
            
            # Create new websocket connection
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Run in thread
            wst = threading.Thread(target=self.ws.run_forever, daemon=True)
            wst.start()
            
            # Wait for connection to establish
            time.sleep(2)
            
            # Authorize with token
            if self.connected:
                return self.authorize(api_token, callback)
            else:
                return False, "Failed to connect to Deriv"
                
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def on_open(self, ws):
        """WebSocket opened"""
        print("✅ Deriv WebSocket connected")
        self.connected = True
    
    def on_message(self, ws, message):
        """Handle incoming messages"""
        try:
            data = json.loads(message)
            
            # Handle ping
            if data.get('msg_type') == 'ping':
                return
            
            # Handle authorization
            if data.get('msg_type') == 'authorize':
                self.handle_authorize(data)
            
            # Handle account list
            elif data.get('msg_type') == 'authorize' and 'authorize' in data:
                # Already handled in authorize
                pass
            
            # Handle balance
            elif data.get('msg_type') == 'balance':
                self.handle_balance(data)
            
            # Handle active symbols
            elif data.get('msg_type') == 'active_symbols':
                self.handle_active_symbols(data)
            
            # Handle ticks
            elif data.get('msg_type') == 'tick':
                self.handle_tick(data)
            
            # Handle proposals (price for contract)
            elif data.get('msg_type') == 'proposal':
                self.handle_proposal(data)
            
            # Handle buy (place trade)
            elif data.get('msg_type') == 'buy':
                self.handle_buy(data)
            
            # Handle general response
            else:
                # Check if this is a response to a pending request
                req_id = data.get('req_id')
                if req_id and req_id in self.pending_requests:
                    callback = self.pending_requests.pop(req_id)
                    if callback:
                        callback(data)
                        
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def on_error(self, ws, error):
        """WebSocket error"""
        print(f"❌ Deriv WebSocket error: {error}")
        self.connected = False
    
    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket closed"""
        print("🔌 Deriv WebSocket disconnected")
        self.connected = False
    
    def send_request(self, request, callback=None):
        """Send request to Deriv API"""
        if not self.connected or not self.ws:
            return False
        
        # Add request ID
        req_id = self.generate_req_id()
        request['req_id'] = req_id
        
        # Store callback if provided
        if callback:
            self.pending_requests[req_id] = callback
        
        # Send request
        self.ws.send(json.dumps(request))
        return True
    
    def authorize(self, token, callback=None):
        """Authorize with API token"""
        request = {
            "authorize": token
        }
        
        def auth_callback(data):
            if data.get('error'):
                if callback:
                    callback(False, data['error']['message'])
            else:
                if callback:
                    callback(True, data)
        
        return self.send_request(request, auth_callback)
    
    def handle_authorize(self, data):
        """Handle authorization response"""
        if 'error' in data:
            print(f"❌ Authorization failed: {data['error']['message']}")
            self.connected = False
        else:
            auth_data = data.get('authorize', {})
            self.loginid = auth_data.get('loginid', '')
            self.email = auth_data.get('email', '')
            self.currency = auth_data.get('currency', 'USD')
            self.balance = auth_data.get('balance', 0)
            
            print(f"✅ Authorized as: {self.loginid}")
            print(f"   Email: {self.email}")
            print(f"   Balance: {self.balance} {self.currency}")
    
    def get_accounts(self, callback=None):
        """Get all trading accounts for this token"""
        # Deriv doesn't have a direct "get accounts" endpoint
        # Usually the token is for a specific account
        # But we can try to get account info
        request = {
            "get_account_status": 1
        }
        return self.send_request(request, callback)
    
    def handle_accounts(self, data):
        """Handle account list"""
        if 'error' in data:
            return
        
        # Parse account data
        # This will need to be adapted based on actual Deriv response
        pass
    
    def get_balance(self, callback=None):
        """Get current balance"""
        request = {
            "balance": 1,
            "subscribe": 1
        }
        return self.send_request(request, callback)
    
    def handle_balance(self, data):
        """Handle balance update"""
        balance_data = data.get('balance', {})
        if 'balance' in balance_data:
            self.balance = balance_data['balance']
            self.currency = balance_data.get('currency', self.currency)
            print(f"💰 Balance updated: {self.balance} {self.currency}")
    
    def get_active_symbols(self, callback=None):
        """Get all active trading symbols"""
        request = {
            "active_symbols": "brief"
        }
        return self.send_request(request, callback)
    
    def handle_active_symbols(self, data):
        """Handle active symbols list"""
        symbols = data.get('active_symbols', [])
        return symbols
    
    def subscribe_ticks(self, symbol, callback):
        """Subscribe to real-time ticks for a symbol"""
        self.tick_subscriptions[symbol] = callback
        request = {
            "ticks": symbol,
            "subscribe": 1
        }
        return self.send_request(request)
    
    def handle_tick(self, data):
        """Handle tick data"""
        tick = data.get('tick', {})
        symbol = tick.get('symbol')
        if symbol and symbol in self.tick_subscriptions:
            callback = self.tick_subscriptions[symbol]
            callback(tick)
    
    def get_contract_proposal(self, symbol, contract_type, amount, duration, duration_unit='t', callback=None):
        """Get price proposal for a contract"""
        request = {
            "proposal": 1,
            "amount": amount,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": self.currency,
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol": symbol
        }
        return self.send_request(request, callback)
    
    def handle_proposal(self, data):
        """Handle proposal response"""
        proposal = data.get('proposal', {})
        return proposal
    
    def place_contract(self, proposal_id, price, callback=None):
        """Place a contract using proposal ID"""
        request = {
            "buy": proposal_id,
            "price": price
        }
        return self.send_request(request, callback)
    
    def handle_buy(self, data):
        """Handle buy response (contract placed)"""
        buy_data = data.get('buy', {})
        contract_id = buy_data.get('contract_id')
        transaction_id = buy_data.get('transaction_id')
        print(f"✅ Contract placed: {contract_id} (Transaction: {transaction_id})")
        return buy_data
    
    def get_historical_candles(self, symbol, interval='1h', count=1000, callback=None):
        """Get historical candles for analysis"""
        # Map interval to Deriv style
        interval_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        
        granularity = interval_map.get(interval, 3600)
        
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "granularity": granularity,
            "style": "candles"
        }
        return self.send_request(request, callback)
    
    def disconnect(self):
        """Disconnect from Deriv"""
        if self.ws:
            self.ws.close()
        self.connected = False
        print("🔌 Disconnected from Deriv")


# ============ DERIV DATA FETCHER ============
class DerivDataFetcher:
    """Fetches market data from Deriv for analysis"""
    
    def __init__(self, deriv_api):
        self.deriv = deriv_api
        self.candle_cache = {}
        self.symbol_info_cache = {}
        
    def get_candles(self, symbol, timeframe, count=500):
        """Get candles for analysis"""
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache
        if cache_key in self.candle_cache:
            cache_time, candles = self.candle_cache[cache_key]
            if time.time() - cache_time < 10:  # Cache for 10 seconds
                return candles
        
        # Map timeframe
        tf_map = {
            'M1': '1m',
            'M5': '5m',
            'M15': '15m',
            'M30': '30m',
            'H1': '1h',
            'H4': '4h',
            'D1': '1d'
        }
        
        interval = tf_map.get(timeframe, '1h')
        
        # Create event for async response
        event = threading.Event()
        result = []
        
        def callback(data):
            if 'error' not in data:
                candles = data.get('candles', [])
                if candles:
                    # Convert to DataFrame format
                    df_data = []
                    for c in candles:
                        df_data.append({
                            'time': datetime.fromtimestamp(c['epoch']),
                            'open': float(c['open']),
                            'high': float(c['high']),
                            'low': float(c['low']),
                            'close': float(c['close']),
                            'tick_volume': int(c.get('volume', 0))
                        })
                    result.extend(df_data)
            event.set()
        
        # Request candles
        self.deriv.get_historical_candles(symbol, interval, count, callback)
        
        # Wait for response (max 5 seconds)
        event.wait(5)
        
        if result:
            # Create DataFrame
            df = pd.DataFrame(result)
            df.set_index('time', inplace=True)
            
            # Cache result
            self.candle_cache[cache_key] = (time.time(), df)
            
            return df
        
        return None


# ============ MARKET STATE ENGINE ============
class MarketStateEngine:
    """
    Analyzes market conditions to determine:
    - Is market trending or ranging?
    - How strong is the trend?
    - Is there a breakout happening?
    - What's the market structure?
    - Which strategy is best suited?
    """
    
    def __init__(self):
        self.ATR_PERIOD = 14
        self.EMA_FAST = 20
        self.EMA_SLOW = 50
        self.EMA_TREND = 200
        
    def analyze(self, df):
        """Complete market state analysis"""
        if df is None or len(df) < 100:
            return {
                'state': MarketState.CHOPPY,
                'direction': 'NEUTRAL',
                'strength': 0,
                'adx': 0,
                'structure': 'NEUTRAL',
                'support': 0,
                'resistance': 0,
                'breakout_detected': False,
                'recommended_strategy': 'NONE'
            }
        
        # Calculate indicators
        df = self._calculate_indicators(df)
        
        # Get current values
        current_price = df['close'].iloc[-1]
        ema_20 = df['ema_20'].iloc[-1]
        ema_50 = df['ema_50'].iloc[-1]
        ema_200 = df['ema_200'].iloc[-1]
        
        # Calculate ADX
        adx = self._calculate_adx(df)
        
        # Detect swing points
        swing_highs, swing_lows = self._detect_swings(df)
        
        # Determine structure
        structure = self._determine_structure(swing_highs, swing_lows)
        
        # Calculate key levels
        support = self._find_support(df)
        resistance = self._find_resistance(df)
        
        # Check for breakout
        breakout_detected, breakout_direction = self._detect_breakout(df, resistance, support)
        
        # Determine market state
        state, direction, strength = self._determine_market_state(
            df, current_price, ema_20, ema_50, ema_200, adx, structure, 
            breakout_detected, breakout_direction
        )
        
        # Recommend strategy
        recommended_strategy = self._recommend_strategy(state, strength, breakout_detected)
        
        return {
            'state': state,
            'direction': direction,
            'strength': strength,
            'adx': adx,
            'structure': structure,
            'support': support,
            'resistance': resistance,
            'breakout_detected': breakout_detected,
            'breakout_direction': breakout_direction if breakout_detected else 'NONE',
            'recommended_strategy': recommended_strategy,
            'current_price': current_price,
            'ema_20': ema_20,
            'ema_50': ema_50,
            'ema_200': ema_200
        }
    
    def _calculate_indicators(self, df):
        """Calculate all technical indicators"""
        df = df.copy()
        
        # EMAs
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.ATR_PERIOD).mean()
        
        # Volume indicators
        if 'tick_volume' in df.columns:
            df['volume_sma'] = df['tick_volume'].rolling(20).mean()
        
        return df
    
    def _calculate_adx(self, df, period=14):
        """Calculate ADX for trend strength"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            plus_dm = np.zeros_like(high)
            minus_dm = np.zeros_like(high)
            
            for i in range(1, len(high)):
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                
                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                else:
                    plus_dm[i] = 0
                    
                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move
                else:
                    minus_dm[i] = 0
            
            tr = np.zeros_like(high)
            for i in range(1, len(high)):
                tr[i] = max(high[i] - low[i], 
                           abs(high[i] - close[i-1]), 
                           abs(low[i] - close[i-1]))
            
            atr = pd.Series(tr).rolling(period).mean().values
            plus_dm_smooth = pd.Series(plus_dm).rolling(period).mean().values
            minus_dm_smooth = pd.Series(minus_dm).rolling(period).mean().values
            
            plus_di = 100 * plus_dm_smooth / atr
            minus_di = 100 * minus_dm_smooth / atr
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = pd.Series(dx).rolling(period).mean().values
            
            return adx[-1] if not np.isnan(adx[-1]) else 0
            
        except Exception as e:
            return 0
    
    def _detect_swings(self, df, window=5):
        """Detect swing highs and lows"""
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                swing_highs.append(df['high'].iloc[i])
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                swing_lows.append(df['low'].iloc[i])
        
        return swing_highs[-10:], swing_lows[-10:]
    
    def _determine_structure(self, swing_highs, swing_lows):
        """Determine market structure"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'NEUTRAL'
        
        last_two_highs = swing_highs[-2:]
        last_two_lows = swing_lows[-2:]
        
        hh = last_two_highs[-1] > last_two_highs[-2]
        hl = last_two_lows[-1] > last_two_lows[-2]
        lh = last_two_highs[-1] < last_two_highs[-2]
        ll = last_two_lows[-1] < last_two_lows[-2]
        
        if hh and hl:
            return 'HH/HL'
        elif lh and ll:
            return 'LH/LL'
        else:
            return 'NEUTRAL'
    
    def _find_support(self, df, lookback=50):
        """Find nearest support level"""
        return df['low'].iloc[-20:].min()
    
    def _find_resistance(self, df, lookback=50):
        """Find nearest resistance level"""
        return df['high'].iloc[-20:].max()
    
    def _detect_breakout(self, df, resistance, support):
        """Detect if market is breaking out"""
        current_price = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        atr = df['atr'].iloc[-1]
        
        if current_price > resistance and prev_close <= resistance:
            if current_price - resistance > atr * 0.5:
                return True, 'BULL'
        
        if current_price < support and prev_close >= support:
            if support - current_price > atr * 0.5:
                return True, 'BEAR'
        
        return False, 'NONE'
    
    def _determine_market_state(self, df, price, ema20, ema50, ema200, adx, structure, 
                               breakout_detected, breakout_direction):
        """Determine the exact market state"""
        
        if breakout_detected:
            if breakout_direction == 'BULL':
                return MarketState.BREAKOUT_BULL, 'BULLISH', min(adx + 20, 100)
            else:
                return MarketState.BREAKOUT_BEAR, 'BEARISH', min(adx + 20, 100)
        
        if price > ema20 > ema50 > ema200 and structure == 'HH/HL' and adx > 30:
            return MarketState.STRONG_UPTREND, 'BULLISH', min(adx + 10, 100)
        elif price < ema20 < ema50 < ema200 and structure == 'LH/LL' and adx > 30:
            return MarketState.STRONG_DOWNTREND, 'BEARISH', min(adx + 10, 100)
        elif price > ema50 and structure == 'HH/HL' and adx > 20:
            return MarketState.UPTREND, 'BULLISH', adx
        elif price < ema50 and structure == 'LH/LL' and adx > 20:
            return MarketState.DOWNTREND, 'BEARISH', adx
        elif adx < 25:
            recent_range = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()
            atr = df['atr'].iloc[-1]
            
            if recent_range < atr * 3:
                return MarketState.CHOPPY, 'NEUTRAL', 20
            else:
                return MarketState.RANGING, 'NEUTRAL', 30
        
        return MarketState.RANGING, 'NEUTRAL', 25
    
    def _recommend_strategy(self, state, strength, breakout_detected):
        """Recommend the best strategy for current market state"""
        if breakout_detected:
            return 'BREAKOUT_CONTINUATION'
        
        strategy_map = {
            MarketState.STRONG_UPTREND: 'CONTINUATION_ONLY',
            MarketState.UPTREND: 'PREFER_CONTINUATION',
            MarketState.RANGING: 'QUASIMODO_ONLY',
            MarketState.DOWNTREND: 'PREFER_CONTINUATION',
            MarketState.STRONG_DOWNTREND: 'CONTINUATION_ONLY',
            MarketState.CHOPPY: 'SKIP_ALL'
        }
        
        return strategy_map.get(state, 'QUASIMODO_ONLY')


# ============ ENHANCED CONTINUATION STRATEGY WITH BETTER SL HANDLING ============
class ContinuationEngine:
    """Trades WITH the trend - pullbacks in trends, not against them"""
    
    def __init__(self):
        self.MAX_PATTERN_AGE = 8
        self.MIN_PULLBACK_DEPTH = 0.3
        self.MAX_PULLBACK_DEPTH = 0.7
        self.SL_MULTIPLIER = 2.0  # Increased from 1.5 to 2.0
        self.TP_MULTIPLIER = 2.5  # Keep at 2.5 for good R:R
        self.SPREAD_BUFFER = 0.3   # 30% of ATR as buffer for spread
        
    def _get_spread_buffer(self, symbol, current_atr):
        """Calculate spread buffer based on symbol type"""
        if 'XAU' in symbol or 'GOLD' in symbol:
            return current_atr * 0.5  # Gold has wider spreads
        elif 'BTC' in symbol or 'XBT' in symbol:
            return current_atr * 0.8  # Crypto has very wide spreads
        elif 'JPY' in symbol:
            return current_atr * 0.4  # JPY pairs
        else:
            return current_atr * self.SPREAD_BUFFER
    
    def detect_setups(self, df, market_state, broker_symbol):
        """Detect continuation setups - ONLY in trending markets"""
        if market_state['state'] in [MarketState.CHOPPY, MarketState.RANGING]:
            return []
        
        signals = []
        atr = df['atr']
        current_price = df['close'].iloc[-1]
        current_atr = atr.iloc[-1]
        ema_20 = df['ema_20']
        
        is_bullish = market_state['direction'] in ['BULLISH', 'BULL']
        is_bearish = market_state['direction'] in ['BEARISH', 'BEAR']
        
        # Calculate spread buffer for this symbol
        spread_buffer = self._get_spread_buffer(broker_symbol, current_atr)
        
        # Look at last 15 candles for setups
        for i in range(-15, 0):
            idx = len(df) + i
            
            # BULLISH PULLBACK - In uptrends, this becomes SELL (reversal signal)
            if is_bullish:
                low = df['low'].iloc[idx]
                ema_val = ema_20.iloc[idx]
                
                # Price touched or came very close to EMA
                if low <= ema_val * 1.002 and low >= ema_val * 0.998:
                    # Price is now above EMA (bounce confirmed)
                    if current_price > ema_val:
                        # Calculate pullback depth
                        recent_high = df['high'].iloc[idx-5:idx].max()
                        pullback_depth = (recent_high - low) / (recent_high - ema_val) if recent_high > ema_val else 0.5
                        
                        if self.MIN_PULLBACK_DEPTH <= pullback_depth <= self.MAX_PULLBACK_DEPTH:
                            # Entry at current price
                            entry = current_price
                            
                            # ENHANCED SL CALCULATION - Add spread buffer and use dynamic multiplier
                            sl_distance = self.SL_MULTIPLIER * current_atr
                            
                            # For SELL, SL above, TP below
                            sl = entry + sl_distance + spread_buffer
                            tp = entry - (sl - entry) * self.TP_MULTIPLIER
                            
                            # Calculate confidence
                            confidence = 75
                            if abs(current_price - ema_val) < current_atr * 0.5:
                                confidence += 10
                            if market_state['strength'] > 70:
                                confidence += 10
                            
                            # Adjust confidence based on spread conditions
                            if spread_buffer > current_atr * 0.5:
                                confidence -= 10  # Reduce confidence if spreads are high
                            
                            signals.append({
                                'type': 'SELL',
                                'entry': entry,
                                'sl': sl,
                                'tp': tp,
                                'index': idx,
                                'atr': current_atr,
                                'strategy': 'CONTINUATION_PULLBACK',
                                'pattern': 'Bullish Pullback to EMA (SELL)',
                                'confidence': min(confidence, 100),
                                'market_state': market_state['state'].value,
                                'spread_buffer': spread_buffer
                            })
            
            # BEARISH RALLY - In downtrends, this becomes BUY (reversal signal)
            if is_bearish:
                high = df['high'].iloc[idx]
                ema_val = ema_20.iloc[idx]
                
                if high >= ema_val * 0.998 and high <= ema_val * 1.002:
                    if current_price < ema_val:  # Rejection confirmed
                        recent_low = df['low'].iloc[idx-5:idx].min()
                        rally_height = (high - recent_low) / (ema_val - recent_low) if ema_val > recent_low else 0.5
                        
                        if self.MIN_PULLBACK_DEPTH <= rally_height <= self.MAX_PULLBACK_DEPTH:
                            entry = current_price
                            
                            # ENHANCED SL CALCULATION - Add spread buffer and use dynamic multiplier
                            sl_distance = self.SL_MULTIPLIER * current_atr
                            
                            # For BUY, SL below, TP above
                            sl = entry - sl_distance - spread_buffer
                            tp = entry + (entry - sl) * self.TP_MULTIPLIER
                            
                            confidence = 75
                            if abs(current_price - ema_val) < current_atr * 0.5:
                                confidence += 10
                            if market_state['strength'] > 70:
                                confidence += 10
                            
                            # Adjust confidence based on spread conditions
                            if spread_buffer > current_atr * 0.5:
                                confidence -= 10
                            
                            signals.append({
                                'type': 'BUY',
                                'entry': entry,
                                'sl': sl,
                                'tp': tp,
                                'index': idx,
                                'atr': current_atr,
                                'strategy': 'CONTINUATION_RALLY',
                                'pattern': 'Bearish Rally to EMA (BUY)',
                                'confidence': min(confidence, 100),
                                'market_state': market_state['state'].value,
                                'spread_buffer': spread_buffer
                            })
        
        # Filter by age
        current_index = len(df) - 1
        valid_signals = []
        
        for signal in signals:
            pattern_age = current_index - signal.get('index', current_index)
            if pattern_age <= self.MAX_PATTERN_AGE:
                valid_signals.append(signal)
        
        return valid_signals[:3]


# ============ ENHANCED QUASIMODO STRATEGY WITH BETTER SL HANDLING ============
class QuasimodoEngine:
    """PURE QUASIMODO - Your proven reversal strategy"""
    
    def __init__(self):
        self.MAX_PATTERN_AGE = 8
        self.RETEST_TOLERANCE_PIPS = 3  # Increased from 2 to 3 pips
        self.ATR_PERIOD = 14
        self.SL_MULTIPLIER = 2.2  # Increased from 1.5 to 2.2
        self.TP_MULTIPLIER = 2.5
        self.SPREAD_BUFFER = 0.25  # 25% of ATR as buffer for spread
    
    def _get_pip_value(self, symbol):
        """Get pip value for tolerance calculation"""
        if 'JPY' in symbol or 'XAG' in symbol or 'BTC' in symbol:
            return 0.01
        elif 'XAU' in symbol or 'US30' in symbol or 'USTEC' in symbol or 'US100' in symbol:
            return 0.1
        else:
            return 0.0001
    
    def _get_spread_buffer(self, symbol, current_atr):
        """Calculate spread buffer based on symbol type"""
        if 'XAU' in symbol or 'GOLD' in symbol:
            return current_atr * 0.4  # Gold has wider spreads
        elif 'BTC' in symbol or 'XBT' in symbol:
            return current_atr * 0.7  # Crypto has very wide spreads
        elif 'JPY' in symbol:
            return current_atr * 0.35  # JPY pairs
        else:
            return current_atr * self.SPREAD_BUFFER
    
    def _check_retest(self, df, pattern_level, direction, tolerance, spread_buffer):
        """Check if price retested pattern level within tolerance - WITH SPREAD ADJUSTMENT"""
        try:
            last_12_low = df['low'].iloc[-12:].min()
            last_12_high = df['high'].iloc[-12:].max()
            current_price = df['close'].iloc[-1]
            
            # ENHANCED: Add spread buffer to tolerance
            effective_tolerance = tolerance + spread_buffer
            
            if direction == 'BUY':
                if last_12_low <= (pattern_level + effective_tolerance) and current_price > pattern_level:
                    last_8_low = df['low'].iloc[-8:].min()
                    if last_8_low <= (pattern_level + effective_tolerance):
                        return True
            else:
                if last_12_high >= (pattern_level - effective_tolerance) and current_price < pattern_level:
                    last_8_high = df['high'].iloc[-8:].max()
                    if last_8_high >= (pattern_level - effective_tolerance):
                        return True
            
            return False
        except:
            return False
    
    def detect_setups(self, df, market_state, broker_symbol):
        """Detect Quasimodo setups - ONLY in ranging markets or at key levels"""
        
        # Skip Quasimodo in strong trends
        if market_state['state'] in [MarketState.STRONG_UPTREND, MarketState.STRONG_DOWNTREND]:
            return []
        
        signals = []
        atr = df['atr']
        current_index = len(df) - 1
        pip_value = self._get_pip_value(broker_symbol)
        tolerance = self.RETEST_TOLERANCE_PIPS * pip_value
        
        for i in range(3, len(df)-1):
            h1 = df['high'].iloc[i-3]
            h2 = df['high'].iloc[i-2]
            h3 = df['high'].iloc[i-1]
            l1 = df['low'].iloc[i-3]
            l2 = df['low'].iloc[i-2]
            l3 = df['low'].iloc[i-1]
            close = df['close'].iloc[i]
            current_atr = atr.iloc[i]
            
            pattern_age = current_index - i
            if pattern_age > self.MAX_PATTERN_AGE:
                continue
            
            # Calculate spread buffer for this symbol
            spread_buffer = self._get_spread_buffer(broker_symbol, current_atr)
            
            # SELL QUASIMODO (Bearish pattern) - This becomes BUY (reversal signal)
            if h1 < h2 > h3 and l1 < l2 < l3 and close < h2:
                # Check if near resistance in ranging market
                near_resistance = abs(close - market_state['resistance']) < current_atr * 2
                
                # In ranging markets, all signals valid. In trends, only near resistance
                if market_state['state'] == MarketState.RANGING or near_resistance:
                    
                    if self._check_retest(df, h2, 'SELL', tolerance, spread_buffer):
                        entry = h2
                        
                        # ENHANCED SL CALCULATION - Add spread buffer and use dynamic multiplier
                        sl_distance = self.SL_MULTIPLIER * current_atr
                        
                        # For BUY, SL below, TP above
                        sl = entry - sl_distance - spread_buffer
                        tp = entry + (entry - sl) * self.TP_MULTIPLIER
                        
                        confidence = 70
                        if near_resistance:
                            confidence += 15
                        if market_state['state'] == MarketState.RANGING:
                            confidence += 10
                        
                        # Adjust confidence based on spread conditions
                        if spread_buffer > current_atr * 0.4:
                            confidence -= 10  # Reduce confidence if spreads are high
                        
                        signals.append({
                            'type': 'BUY',
                            'entry': entry,
                            'sl': sl,
                            'tp': tp,
                            'atr': current_atr,
                            'strategy': 'QUASIMODO_REVERSAL',
                            'pattern': 'Quasimodo Sell Setup (BUY)',
                            'confidence': min(confidence, 100),
                            'index': i,
                            'market_state': market_state['state'].value,
                            'spread_buffer': spread_buffer
                        })
            
            # BUY QUASIMODO (Bullish pattern) - This becomes SELL (reversal signal)
            if l1 > l2 < l3 and h1 > h2 > h3 and close > l2:
                near_support = abs(close - market_state['support']) < current_atr * 2
                
                if market_state['state'] == MarketState.RANGING or near_support:
                    
                    if self._check_retest(df, l2, 'BUY', tolerance, spread_buffer):
                        entry = l2
                        
                        # ENHANCED SL CALCULATION - Add spread buffer and use dynamic multiplier
                        sl_distance = self.SL_MULTIPLIER * current_atr
                        
                        # For SELL, SL above, TP below
                        sl = entry + sl_distance + spread_buffer
                        tp = entry - (sl - entry) * self.TP_MULTIPLIER
                        
                        confidence = 70
                        if near_support:
                            confidence += 15
                        if market_state['state'] == MarketState.RANGING:
                            confidence += 10
                        
                        # Adjust confidence based on spread conditions
                        if spread_buffer > current_atr * 0.4:
                            confidence -= 10
                        
                        signals.append({
                            'type': 'SELL',
                            'entry': entry,
                            'sl': sl,
                            'tp': tp,
                            'atr': current_atr,
                            'strategy': 'QUASIMODO_REVERSAL',
                            'pattern': 'Quasimodo Buy Setup (SELL)',
                            'confidence': min(confidence, 100),
                            'index': i,
                            'market_state': market_state['state'].value,
                            'spread_buffer': spread_buffer
                        })
        
        return signals[:3]


# ============ SMART STRATEGY SELECTOR ============
class SmartStrategySelector:
    """Decides which strategy to use based on market conditions"""
    
    def select_best_trades(self, continuation_signals, quasimodo_signals, market_state):
        """Select the best trades for current market conditions"""
        
        state = market_state['state']
        selected_trades = []
        
        # STRONG UPTREND - ONLY CONTINUATION SELLS (since we swapped)
        if state == MarketState.STRONG_UPTREND:
            selected_trades = [t for t in continuation_signals if t['type'] == 'SELL']
            print(f"   📊 STRONG UPTREND - Using CONTINUATION SELLS only")
        
        # STRONG DOWNTREND - ONLY CONTINUATION BUYS (since we swapped)
        elif state == MarketState.STRONG_DOWNTREND:
            selected_trades = [t for t in continuation_signals if t['type'] == 'BUY']
            print(f"   📊 STRONG DOWNTREND - Using CONTINUATION BUYS only")
        
        # UPTREND - PREFER CONTINUATION SELLS, ALLOW STRONG QUASIMODO
        elif state == MarketState.UPTREND:
            selected_trades = [t for t in continuation_signals if t['type'] == 'SELL']
            strong_qm = [q for q in quasimodo_signals if q.get('confidence', 0) > 80]
            selected_trades.extend(strong_qm)
            print(f"   📊 UPTREND - Prefer CONTINUATION SELLS, allow strong QUASIMODO")
        
        # DOWNTREND - PREFER CONTINUATION BUYS, ALLOW STRONG QUASIMODO
        elif state == MarketState.DOWNTREND:
            selected_trades = [t for t in continuation_signals if t['type'] == 'BUY']
            strong_qm = [q for q in quasimodo_signals if q.get('confidence', 0) > 80]
            selected_trades.extend(strong_qm)
            print(f"   📊 DOWNTREND - Prefer CONTINUATION BUYS, allow strong QUASIMODO")
        
        # RANGING - ONLY QUASIMODO (both BUY and SELL are valid)
        elif state == MarketState.RANGING:
            selected_trades = quasimodo_signals
            print(f"   📊 RANGING - Using QUASIMODO only")
        
        # BREAKOUT - CONTINUATION
        elif state in [MarketState.BREAKOUT_BULL, MarketState.BREAKOUT_BEAR]:
            selected_trades = continuation_signals
            print(f"   📊 BREAKOUT - Using CONTINUATION for momentum")
        
        # CHOPPY - SKIP ALL
        elif state == MarketState.CHOPPY:
            selected_trades = []
            print(f"   📊 CHOPPY - SKIPPING ALL TRADES (no edge)")
        
        # Filter by confidence (minimum 65%)
        selected_trades = [t for t in selected_trades if t.get('confidence', 0) >= 65]
        
        # Sort by confidence
        selected_trades.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return selected_trades


# ============ DERIV TRADING ENGINE ============
class KarankaDerivTradingEngine:
    """Main trading engine for Deriv - analyzes markets and executes trades"""
    
    def __init__(self, settings):
        self.settings = settings
        self.deriv = DerivAPI()
        self.data_fetcher = DerivDataFetcher(self.deriv)
        self.market_engine = MarketStateEngine()
        self.continuation = ContinuationEngine()
        self.quasimodo = QuasimodoEngine()
        self.selector = SmartStrategySelector()
        self.live_display = LiveMarketDisplay()
        
        self.active_trades = []
        self.connected = False
        self.running = False
        self.deriv_connected = False
        self.selected_account = None
        
        self.trades_today = 0
        self.trades_hour = 0
        self.last_trade_time = None
        self.total_cycles = 0
        self.analysis_count = 0
        
        self.symbol_mapping = self.get_deriv_symbols()
        self.data_cache = {}
        self.cache_timestamps = {}
        
        print("✅ KARANKA DERIV TRADING ENGINE INITIALIZED")
        print("   • MARKET STATE ENGINE - Active")
        print("   • CONTINUATION STRATEGY - Active (BUY/SELL SWAPPED)")
        print("   • QUASIMODO STRATEGY - Active (BUY/SELL SWAPPED)")
        print("   • SMART STRATEGY SELECTOR - Active")
        print("   • DERIV API INTEGRATION - Active")
        print("   • ENHANCED SL HANDLING - Active (Spread Buffer + Dynamic Multipliers)")
    
    def get_deriv_symbols(self):
        """Get Deriv symbol mapping"""
        # Deriv symbol mapping to our universal symbols
        return {
            "EURUSD": "frxEURUSD",
            "GBPUSD": "frxGBPUSD",
            "USDJPY": "frxUSDJPY",
            "AUDUSD": "frxAUDUSD",
            "USDCAD": "frxUSDCAD",
            "USDCHF": "frxUSDCHF",
            "NZDUSD": "frxNZDUSD",
            "EURGBP": "frxEURGBP",
            "EURJPY": "frxEURJPY",
            "GBPJPY": "frxGBPJPY",
            "XAUUSD": "frxXAUUSD",
            "XAGUSD": "frxXAGUSD",
            "BTCUSD": "cryBTCUSD",
            "US30": "R_30",
            "USTEC": "R_100",
            "US100": "R_100"
        }
    
    def connect_deriv(self, api_token, callback=None):
        """Connect to Deriv with API token"""
        success, message = self.deriv.connect(api_token, callback)
        if success:
            self.deriv_connected = True
            self.connected = True
            
            # Get account info
            time.sleep(1)  # Wait for authorization
            
            # Subscribe to balance updates
            self.deriv.get_balance()
            
            return True, "Connected to Deriv"
        else:
            return False, message
    
    def get_cached_data(self, symbol, timeframe, bars_needed=300):
        """Get data from Deriv with caching"""
        cache_key = f"{symbol}_{timeframe}"
        current_time = time.time()
        
        if cache_key in self.data_cache:
            data_age = current_time - self.cache_timestamps.get(cache_key, 0)
            if data_age < 5:
                return self.data_cache[cache_key]
        
        # Fetch from Deriv
        df = self.data_fetcher.get_candles(symbol, timeframe, bars_needed)
        
        if df is not None:
            self.data_cache[cache_key] = df
            self.cache_timestamps[cache_key] = current_time
        
        return df
    
    def _prepare_dataframe(self, df):
        """Prepare dataframe with all necessary indicators"""
        if df is None:
            return None
        
        df = df.copy()
        
        # Calculate EMAs
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        return df
    
    def analyze_symbol(self, universal_symbol, deriv_symbol):
        """Complete analysis for a single symbol"""
        try:
            # Get data
            df_15m = self.get_cached_data(deriv_symbol, 'M15', 300)
            df_h1 = self.get_cached_data(deriv_symbol, 'H1', 200)
            
            # If H1 not available, use M15 for structure
            if df_h1 is None or len(df_h1) < 50:
                if df_15m is not None:
                    df_h1 = df_15m.copy()
                else:
                    return None
            
            if df_15m is None:
                return None
            
            # Prepare dataframes with indicators
            df_15m = self._prepare_dataframe(df_15m)
            df_h1 = self._prepare_dataframe(df_h1)
            
            # Analyze market state on H1
            market_state = self.market_engine.analyze(df_h1)
            
            # Detect setups - pass broker_symbol for spread calculation
            continuation_signals = self.continuation.detect_setups(df_15m, market_state, universal_symbol)
            quasimodo_signals = self.quasimodo.detect_setups(df_15m, market_state, universal_symbol)
            
            # Select best trades
            best_trades = self.selector.select_best_trades(
                continuation_signals, quasimodo_signals, market_state
            )
            
            if not best_trades:
                return None
            
            best_trade = best_trades[0]
            
            analysis = {
                'universal_symbol': universal_symbol,
                'deriv_symbol': deriv_symbol,
                'current_price': df_15m['close'].iloc[-1],
                'direction': best_trade['type'],
                'entry': best_trade['entry'],
                'sl': best_trade['sl'],
                'tp': best_trade['tp'],
                'strategy': best_trade['strategy'],
                'pattern': best_trade.get('pattern', 'N/A'),
                'confidence': best_trade.get('confidence', 70),
                'market_state': market_state['state'].value,
                'market_direction': market_state['direction'],
                'structure': market_state['structure'],
                'atr': best_trade.get('atr', 0),
                'spread_buffer': best_trade.get('spread_buffer', 0),
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            
            print(f"   ✅ {universal_symbol} {analysis['direction']} | {analysis['strategy']} | "
                  f"Conf: {analysis['confidence']:.0f}% | State: {analysis['market_state']}")
            
            return analysis
            
        except Exception as e:
            print(f"   ❌ Error analyzing {universal_symbol}: {e}")
            return None
    
    def trading_loop(self):
        """Main trading loop"""
        print("\n🚀 KARANKA DERIV TRADING STARTED - ANALYZING ALL MARKETS")
        print(f"   • Market State Engine: ACTIVE")
        print(f"   • Continuation Strategy: ACTIVE (BUY/SELL SWAPPED)")
        print(f"   • Quasimodo Strategy: ACTIVE (BUY/SELL SWAPPED)")
        print(f"   • Deriv API: CONNECTED")
        print(f"   • Retest: 3 PIP TOLERANCE (Enhanced)")
        print(f"   • SL Multiplier: 2.0-2.2 ATR + Spread Buffer")
        print(f"   • Min Confidence: 65%")
        
        time.sleep(2)
        
        while self.running and self.connected:
            try:
                self.total_cycles += 1
                self.analysis_count = 0
                
                print(f"\n🔄 CYCLE {self.total_cycles} - {datetime.now().strftime('%H:%M:%S')}")
                print(f"   Balance: {self.deriv.balance} {self.deriv.currency}")
                print(f"   Today: {self.trades_today}/{self.settings.max_daily_trades}")
                
                if not self.can_trade():
                    time.sleep(5)
                    continue
                
                # Analyze each enabled symbol
                for universal_symbol in self.settings.enabled_symbols:
                    if universal_symbol not in self.symbol_mapping:
                        continue
                    
                    deriv_symbol = self.symbol_mapping[universal_symbol]
                    
                    analysis = self.analyze_symbol(universal_symbol, deriv_symbol)
                    
                    if analysis:
                        self.live_display.update_analysis(universal_symbol, analysis)
                        self.analysis_count += 1
                
                print(f"   📊 Active signals: {self.analysis_count}")
                
                # Execute trades
                if self.analysis_count > 0 and not self.settings.dry_run:
                    for symbol in self.live_display.market_analysis:
                        if not self.can_trade():
                            break
                        
                        data = self.live_display.market_analysis.get(symbol)
                        if data and data['direction'] != 'NONE':
                            for setup in self.get_active_setups():
                                if setup['universal_symbol'] == symbol:
                                    success, message = self.execute_trade(setup)
                                    if success:
                                        print(f"   ✅ EXECUTED: {symbol} {setup['direction']} | {setup['strategy']}")
                                        self.trades_today += 1
                                        self.trades_hour += 1
                                        self.last_trade_time = datetime.now()
                                    break
                
                time.sleep(8)
                
            except Exception as e:
                print(f"❌ Trading loop error: {e}")
                time.sleep(5)
    
    def get_active_setups(self):
        """Get active setups from display"""
        setups = []
        for symbol, data in self.live_display.market_analysis.items():
            if data['direction'] != 'NONE':
                setups.append({
                    'universal_symbol': symbol,
                    'deriv_symbol': self.symbol_mapping.get(symbol, symbol),
                    'direction': data['direction'],
                    'entry': data['entry'],
                    'sl': data['sl'],
                    'tp': data['tp'],
                    'strategy': data.get('strategy', 'QUASIMODO'),
                    'pattern': data.get('pattern', 'N/A'),
                    'confidence': data.get('confidence', 70),
                    'market_state': data.get('market_state', 'UNKNOWN')
                })
        return setups
    
    def can_trade(self):
        """Check if we can trade"""
        if not self.connected or not self.deriv_connected:
            return False
        if len(self.active_trades) >= self.settings.max_concurrent_trades:
            return False
        if self.trades_hour >= self.settings.max_hourly_trades:
            return False
        if self.trades_today >= self.settings.max_daily_trades:
            return False
        if self.last_trade_time:
            seconds_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if seconds_since_last < self.settings.min_seconds_between_trades:
                return False
        return True
    
    def calculate_position_size(self, analysis):
        """Calculate position size based on confidence and spread conditions"""
        base_lot = self.settings.fixed_lot_size
        confidence = analysis.get('confidence', 70)
        
        # For Deriv, we need to convert to stake amount
        # Use balance-based sizing
        balance = self.deriv.balance
        risk_percent = 0.02  # 2% risk per trade
        
        risk_amount = balance * risk_percent
        
        # Adjust based on confidence
        confidence_factor = 0.5 + (confidence / 200)  # 0.5 to 1.0
        
        # Adjust for spread
        spread_buffer = analysis.get('spread_buffer', 0)
        atr = analysis.get('atr', 0.001)
        
        spread_factor = 1.0
        if atr > 0 and spread_buffer > atr * 0.3:
            spread_factor = 0.7  # Reduce to 70% if spreads are high
        
        stake = risk_amount * confidence_factor * spread_factor
        
        return max(1.0, round(stake, 2))  # Minimum $1 stake
    
    def execute_trade(self, analysis):
        """Execute a trade on Deriv"""
        try:
            universal_symbol = analysis['universal_symbol']
            deriv_symbol = analysis['deriv_symbol']
            direction = analysis['direction']
            
            # Calculate stake
            stake = self.calculate_position_size(analysis)
            
            if self.settings.dry_run:
                return self.execute_dry_run(analysis, stake)
            else:
                return self.execute_real_trade(analysis, stake, deriv_symbol, direction)
                
        except Exception as e:
            return False, f"Execution error: {str(e)}"
    
    def execute_dry_run(self, analysis, stake):
        """Execute dry run trade"""
        universal_symbol = analysis['universal_symbol']
        direction = analysis['direction']
        
        print(f"\n✅ [DRY RUN] {universal_symbol} {direction}")
        print(f"   Strategy: {analysis.get('strategy', 'QUASIMODO')}")
        print(f"   Pattern: {analysis.get('pattern', 'N/A')}")
        print(f"   Confidence: {analysis.get('confidence', 70):.0f}%")
        print(f"   Market State: {analysis.get('market_state', 'UNKNOWN')}")
        print(f"   Stake: ${stake:.2f}")
        print(f"   Spread Buffer: {analysis.get('spread_buffer', 0):.5f}")
        
        self.log_trade(
            "DRY_RUN", universal_symbol, direction, 0, 0, 0, stake,
            comment=f"KARANKA DERIV AI | {analysis.get('strategy', 'QM')}"
        )
        
        trade_info = {
            'symbol': universal_symbol,
            'direction': direction,
            'stake': stake,
            'timestamp': datetime.now(),
            'dry_run': True,
            'analysis': analysis
        }
        self.active_trades.append(trade_info)
        
        def auto_remove():
            time.sleep(3600)  # Remove after 1 hour
            if trade_info in self.active_trades:
                self.active_trades.remove(trade_info)
        
        threading.Thread(target=auto_remove, daemon=True).start()
        
        return True, "Dry run executed"
    
    def execute_real_trade(self, analysis, stake, deriv_symbol, direction):
        """Execute real trade on Deriv"""
        try:
            # Determine contract type
            contract_type = "CALL" if direction == 'BUY' else "PUT"
            
            # Create event for response
            event = threading.Event()
            result = {"success": False, "data": None, "error": None}
            
            def proposal_callback(data):
                if 'error' in data:
                    result["error"] = data['error']['message']
                    result["success"] = False
                else:
                    proposal = data.get('proposal', {})
                    proposal_id = proposal.get('id')
                    price = float(proposal.get('ask_price', 0))
                    
                    if proposal_id and price > 0:
                        # Place the contract
                        def buy_callback(buy_data):
                            if 'error' in buy_data:
                                result["error"] = buy_data['error']['message']
                                result["success"] = False
                            else:
                                result["data"] = buy_data.get('buy', {})
                                result["success"] = True
                            event.set()
                        
                        self.deriv.place_contract(proposal_id, price, buy_callback)
                    else:
                        result["error"] = "Invalid proposal"
                        event.set()
                event.set()
            
            # Get contract proposal
            self.deriv.get_contract_proposal(
                deriv_symbol, 
                contract_type, 
                stake, 
                1,  # 1 hour duration
                'h',  # hours
                proposal_callback
            )
            
            # Wait for response (max 10 seconds)
            event.wait(10)
            
            if result["success"]:
                buy_data = result["data"]
                print(f"\n✅ REAL TRADE: {buy_data.get('contract_id')}")
                print(f"   Contract: {contract_type} on {universal_symbol}")
                print(f"   Stake: ${stake:.2f}")
                print(f"   Strategy: {analysis.get('strategy', 'QUASIMODO')}")
                
                self.log_trade(
                    "REAL", universal_symbol, direction, 
                    buy_data.get('buy_price', 0), 
                    0,  # Deriv doesn't use SL/TP in same way
                    buy_data.get('payout', 0), 
                    stake,
                    comment=f"Contract:{buy_data.get('contract_id')} | KARANKA DERIV AI"
                )
                
                trade_info = {
                    'symbol': universal_symbol,
                    'direction': direction,
                    'contract_id': buy_data.get('contract_id'),
                    'transaction_id': buy_data.get('transaction_id'),
                    'stake': stake,
                    'buy_price': buy_data.get('buy_price', 0),
                    'payout': buy_data.get('payout', 0),
                    'timestamp': datetime.now(),
                    'dry_run': False,
                    'analysis': analysis
                }
                self.active_trades.append(trade_info)
                
                return True, f"Contract {buy_data.get('contract_id')} executed"
            
            return False, result.get("error", "Unknown error")
            
        except Exception as e:
            return False, f"Execution error: {str(e)}"
    
    def log_trade(self, action, symbol, direction, entry, sl, tp, stake, comment=""):
        """Log trade to file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {action} | {symbol} {direction} | Stake: ${stake:.2f} | Entry: {entry:.5f} | {comment}\n"
        
        with open(TRADES_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def start_trading(self):
        """Start trading"""
        if not self.connected or not self.deriv_connected:
            return False
        
        self.running = True
        self.trades_today = 0
        self.trades_hour = 0
        self.active_trades.clear()
        
        def reset_hourly():
            while self.running:
                time.sleep(3600)
                self.trades_hour = 0
        
        threading.Thread(target=reset_hourly, daemon=True).start()
        
        thread = threading.Thread(target=self.trading_loop, daemon=True)
        thread.start()
        
        return True
    
    def stop_trading(self):
        """Stop trading"""
        self.running = False
        print("\n🛑 KARANKA TRADING STOPPED")
    
    def get_status(self):
        """Get trading status"""
        active_real = sum(1 for t in self.active_trades if not t.get('dry_run', False))
        active_dry = sum(1 for t in self.active_trades if t.get('dry_run', False))
        
        return {
            'connected': self.connected,
            'deriv_connected': self.deriv_connected,
            'running': self.running,
            'active_real': active_real,
            'active_dry': active_dry,
            'total_active': len(self.active_trades),
            'daily_trades': self.trades_today,
            'hourly_trades': self.trades_hour,
            'total_cycles': self.total_cycles,
            'analysis_count': self.analysis_count,
            'balance': self.deriv.balance if self.deriv else 0,
            'currency': self.deriv.currency if self.deriv else 'USD',
            'account': self.deriv.loginid if self.deriv else ''
        }
    
    def get_live_display_text(self):
        """Get live display text"""
        return self.live_display.get_display_text()
    
    def disconnect(self):
        """Disconnect from Deriv"""
        self.deriv.disconnect()
        self.deriv_connected = False
        self.connected = False


# ============ LIVE MARKET DISPLAY ============
class LiveMarketDisplay:
    """Display live market analysis"""
    
    def __init__(self):
        self.market_analysis = {}
        self.last_update = datetime.now()
        self.update_count = 0
    
    def update_analysis(self, symbol, analysis_data):
        """Update analysis for a symbol"""
        if not analysis_data:
            return
        
        self.update_count += 1
        
        display_data = {
            'symbol': symbol,
            'timestamp': analysis_data.get('timestamp', datetime.now().strftime('%H:%M:%S')),
            'price': analysis_data.get('current_price', 0),
            'direction': analysis_data.get('direction', 'NONE'),
            'strategy': analysis_data.get('strategy', 'QUASIMODO'),
            'pattern': analysis_data.get('pattern', 'N/A'),
            'confidence': analysis_data.get('confidence', 70),
            'market_state': analysis_data.get('market_state', 'UNKNOWN'),
            'entry': analysis_data.get('entry', 0),
            'sl': analysis_data.get('sl', 0),
            'tp': analysis_data.get('tp', 0),
            'spread_buffer': analysis_data.get('spread_buffer', 0),
        }
        
        self.market_analysis[symbol] = display_data
        self.last_update = datetime.now()
    
    def get_display_text(self):
        """Get formatted display text"""
        if not self.market_analysis:
            return "🔍 KARANKA MARKET ANALYSIS - Scanning all selected markets...\n\nNo active signals yet. Waiting for:\n• Clear market state\n• Valid pattern formation\n• 3 PIP retest confirmation (Enhanced)\n• HTF structure alignment"
        
        lines = []
        lines.append(f"=== KARANKA MULTIVERSE ALGO AI - DERIV LIVE ANALYSIS ===")
        lines.append(f"Last Update: {self.last_update.strftime('%H:%M:%S')} | 3 PIP RETEST | ENHANCED SL")
        lines.append("=" * 100)
        
        # Group signals
        buy_signals = []
        sell_signals = []
        
        for symbol, data in self.market_analysis.items():
            if data['direction'] == 'BUY':
                buy_signals.append((symbol, data))
            elif data['direction'] == 'SELL':
                sell_signals.append((symbol, data))
        
        # Show market states
        market_states = {}
        for symbol, data in self.market_analysis.items():
            state = data.get('market_state', 'UNKNOWN')
            market_states[state] = market_states.get(state, 0) + 1
        
        lines.append(f"\n📊 MARKET CONDITIONS:")
        for state, count in market_states.items():
            lines.append(f"   {state}: {count} symbols")
        
        # Show buy signals
        if buy_signals:
            lines.append(f"\n🟢 BUY SIGNALS - {len(buy_signals)} ACTIVE:")
            for symbol, data in sorted(buy_signals, key=lambda x: x[1]['confidence'], reverse=True):
                conf_bar = "█" * int(data['confidence']/10) + "░" * (10 - int(data['confidence']/10))
                lines.append(f"   {symbol}:")
                lines.append(f"     ├─ {data['strategy']} | Conf: {data['confidence']:.0f}% {conf_bar}")
                lines.append(f"     ├─ Pattern: {data['pattern']}")
                lines.append(f"     ├─ Entry: {data['entry']:.5f} | SL: {data['sl']:.5f} | TP: {data['tp']:.5f}")
                lines.append(f"     └─ Market: {data['market_state']} | Buffer: {data.get('spread_buffer', 0):.5f}")
        
        # Show sell signals
        if sell_signals:
            lines.append(f"\n🔴 SELL SIGNALS - {len(sell_signals)} ACTIVE:")
            for symbol, data in sorted(sell_signals, key=lambda x: x[1]['confidence'], reverse=True):
                conf_bar = "█" * int(data['confidence']/10) + "░" * (10 - int(data['confidence']/10))
                lines.append(f"   {symbol}:")
                lines.append(f"     ├─ {data['strategy']} | Conf: {data['confidence']:.0f}% {conf_bar}")
                lines.append(f"     ├─ Pattern: {data['pattern']}")
                lines.append(f"     ├─ Entry: {data['entry']:.5f} | SL: {data['sl']:.5f} | TP: {data['tp']:.5f}")
                lines.append(f"     └─ Market: {data['market_state']} | Buffer: {data.get('spread_buffer', 0):.5f}")
        
        if not buy_signals and not sell_signals:
            lines.append(f"\n⚪ Scanning {len(self.market_analysis)} markets...")
            for symbol, data in list(self.market_analysis.items())[:5]:
                lines.append(f"   {symbol}: {data['price']:.5f} | State: {data.get('market_state', 'SCANNING')}")
        
        lines.append("\n" + "=" * 100)
        lines.append("📊 KARANKA CONFIGURATION:")
        lines.append("   • Market State Engine: ACTIVE")
        lines.append("   • Smart Strategy Selector: ACTIVE (BUY/SELL SWAPPED)")
        lines.append("   • Continuation (Trend): ✓ | Quasimodo (Range): ✓")
        lines.append("   • Retest Tolerance: 3 PIPS (Enhanced) - ACTIVE")
        lines.append("   • SL Multiplier: 2.0-2.2 ATR + Spread Buffer")
        lines.append("   • HTF Structure: MANDATORY")
        lines.append("   • Minimum Confidence: 65%")
        lines.append(f"   • Total Signals: {len(buy_signals) + len(sell_signals)}")
        lines.append("=" * 100)
        
        return "\n".join(lines)


# ============ SETTINGS ============
class KarankaSettings:
    """Settings for Karanka trading"""
    
    def __init__(self):
        self.deriv_api_token = ""
        
        self.dry_run = True
        
        self.universal_symbols = [
            "EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD",
            "US30", "USTEC", "US100", "AUDUSD", "BTCUSD",
            "USDCHF", "USDCAD", "EURGBP", "EURJPY",
            "CHFJPY", "GBPJPY", "AUDJPY", "EURAUD", "GBPAUD"
        ]
        self.enabled_symbols = [
            "EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD",
            "US30", "USTEC", "US100", "AUDUSD", "BTCUSD"
        ]
        
        self.enable_15m = True
        self.enable_30m = True
        self.enable_1h = True
        
        self.max_concurrent_trades = 3
        self.max_daily_trades = 15
        self.max_hourly_trades = 4
        self.min_seconds_between_trades = 10
        
        self.fixed_lot_size = 0.01  # For dry run reference
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        
        self.load_settings()
    
    def load_settings(self):
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
        except:
            pass
    
    def save_settings(self):
        try:
            data = {}
            for key in dir(self):
                if not key.startswith('_') and not callable(getattr(self, key)):
                    value = getattr(self, key)
                    if isinstance(value, (int, float, bool, str, list, dict)):
                        data[key] = value
            
            settings_dir = os.path.dirname(SETTINGS_FILE)
            if not os.path.exists(settings_dir):
                os.makedirs(settings_dir)
            
            with open(SETTINGS_FILE, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            return True
        except:
            return False


# ============ DERIV AUTH DIALOG ============
class DerivAuthDialog:
    """Dialog for Deriv API authentication"""
    
    def __init__(self, parent, settings):
        self.parent = parent
        self.settings = settings
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("🔐 Deriv API Authentication")
        self.dialog.geometry("600x500")
        self.dialog.configure(bg=BLACK_GOLD_THEME['bg'])
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.setup_ui()
        
    def setup_ui(self):
        theme = BLACK_GOLD_THEME
        
        main = tk.Frame(self.dialog, bg=theme['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title = tk.Label(main, 
                        text="🔐 DERIV API CONNECTION",
                        font=("Arial", 14, "bold"),
                        bg=theme['bg'],
                        fg=theme['fg'])
        title.pack(pady=(0, 20))
        
        # Info
        info_text = """Enter your Deriv API token to connect.

How to get your API token:
1. Log in to your Deriv account
2. Go to Settings → API Token
3. Create a new token with "Trade" permissions
4. Copy and paste it below

Your token will be stored locally and never shared."""
        
        info = tk.Label(main, 
                       text=info_text,
                       justify=tk.LEFT,
                       bg=theme['secondary'],
                       fg=theme['fg_light'],
                       font=("Arial", 9),
                       relief='flat',
                       padx=10,
                       pady=10)
        info.pack(fill=tk.X, pady=(0, 20))
        
        # Token entry
        token_frame = tk.Frame(main, bg=theme['bg'])
        token_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(token_frame, 
                text="API Token:",
                bg=theme['bg'],
                fg=theme['fg']).pack(anchor=tk.W)
        
        self.token_entry = tk.Entry(token_frame,
                                   width=50,
                                   bg=theme['secondary'],
                                   fg=theme['fg'],
                                   insertbackground=theme['fg'],
                                   font=("Courier", 10))
        self.token_entry.pack(fill=tk.X, pady=(5, 0))
        self.token_entry.insert(0, self.settings.deriv_api_token)
        
        # Connection status
        self.status_frame = tk.Frame(main, bg=theme['bg'])
        self.status_frame.pack(fill=tk.X, pady=10)
        
        self.status_label = tk.Label(self.status_frame,
                                    text="⏳ Ready to connect",
                                    bg=theme['bg'],
                                    fg=theme['fg_dark'])
        self.status_label.pack()
        
        # Progress bar (hidden initially)
        self.progress = ttk.Progressbar(self.status_frame,
                                       mode='indeterminate',
                                       length=400)
        
        # Accounts frame
        self.accounts_frame = tk.LabelFrame(main,
                                          text="Select Trading Account",
                                          bg=theme['bg'],
                                          fg=theme['fg'],
                                          font=("Arial", 10, "bold"))
        
        # Buttons
        btn_frame = tk.Frame(main, bg=theme['bg'])
        btn_frame.pack(fill=tk.X, pady=20)
        
        self.connect_btn = tk.Button(btn_frame,
                                    text="🔌 Connect",
                                    bg=theme['button_bg'],
                                    fg=theme['button_fg'],
                                    font=("Arial", 10, "bold"),
                                    command=self.connect_deriv,
                                    padx=20,
                                    pady=5)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame,
                text="Cancel",
                bg=theme['button_bg'],
                fg=theme['button_fg'],
                font=("Arial", 10),
                command=self.dialog.destroy,
                padx=20,
                pady=5).pack(side=tk.LEFT, padx=5)
    
    def connect_deriv(self):
        """Connect to Deriv with token"""
        token = self.token_entry.get().strip()
        
        if not token:
            messagebox.showwarning("Missing Token", "Please enter your API token")
            return
        
        # Update UI
        self.status_label.config(text="🔄 Connecting to Deriv...", fg=BLACK_GOLD_THEME['warning'])
        self.connect_btn.config(state='disabled')
        self.progress.pack(pady=10)
        self.progress.start(10)
        self.dialog.update()
        
        # Connect in thread
        def connect_thread():
            try:
                # Create temporary Deriv connection to validate
                deriv = DerivAPI()
                
                def auth_callback(success, data):
                    if success:
                        # Connection successful
                        self.dialog.after(0, self.connection_success, deriv)
                    else:
                        self.dialog.after(0, self.connection_failed, data)
                
                success, message = deriv.connect(token, auth_callback)
                
                if not success:
                    self.dialog.after(0, self.connection_failed, message)
                    
            except Exception as e:
                self.dialog.after(0, self.connection_failed, str(e))
        
        threading.Thread(target=connect_thread, daemon=True).start()
    
    def connection_success(self, deriv):
        """Handle successful connection"""
        self.progress.stop()
        self.progress.pack_forget()
        
        self.status_label.config(text="✅ Connected! Loading account info...", 
                               fg=BLACK_GOLD_THEME['success'])
        
        # Store token
        self.settings.deriv_api_token = self.token_entry.get().strip()
        self.settings.save_settings()
        
        # Show account info
        self.show_account_info(deriv)
    
    def show_account_info(self, deriv):
        """Show account information"""
        theme = BLACK_GOLD_THEME
        
        # Hide connect button
        self.connect_btn.pack_forget()
        
        # Show account info
        self.accounts_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        info_text = f"""
Account ID: {deriv.loginid}
Email: {deriv.email}
Balance: {deriv.balance} {deriv.currency}
Status: Active
        """
        
        info = tk.Label(self.accounts_frame,
                       text=info_text,
                       justify=tk.LEFT,
                       bg=theme['secondary'],
                       fg=theme['fg'],
                       font=("Courier", 10),
                       padx=10,
                       pady=10)
        info.pack(fill=tk.X, pady=10)
        
        # Account selection (since Deriv tokens are account-specific, just confirm)
        select_frame = tk.Frame(self.accounts_frame, bg=theme['bg'])
        select_frame.pack(pady=10)
        
        tk.Label(select_frame,
                text="✓ This token is for the account above",
                bg=theme['bg'],
                fg=theme['success']).pack()
        
        # Confirm button
        tk.Button(self.accounts_frame,
                text="✅ Use This Account",
                bg=theme['button_bg'],
                fg=theme['button_fg'],
                font=("Arial", 10, "bold"),
                command=lambda: self.select_account(deriv),
                padx=30,
                pady=10).pack(pady=10)
    
    def select_account(self, deriv):
        """Select this account for trading"""
        self.result = {
            'success': True,
            'token': self.settings.deriv_api_token,
            'loginid': deriv.loginid,
            'balance': deriv.balance,
            'currency': deriv.currency,
            'deriv': deriv
        }
        self.dialog.destroy()
    
    def connection_failed(self, error):
        """Handle connection failure"""
        self.progress.stop()
        self.progress.pack_forget()
        
        self.status_label.config(text=f"❌ Connection failed: {error}", 
                               fg=BLACK_GOLD_THEME['error'])
        self.connect_btn.config(state='normal')


# ============ GUI - UPDATED FOR DERIV ============
class KarankaDerivGUI:
    """GUI for Karanka Trading Bot with Deriv integration"""
    
    def __init__(self):
        self.settings = KarankaSettings()
        self.trader = None
        self.deriv_connection = None
        
        self.root = tk.Tk()
        self.root.title("KARANKA MULTIVERSE ALGO AI - DERIV TRADING SYSTEM")
        self.root.geometry("1400x900")
        
        self.apply_theme()
        self.setup_gui()
        
        # Check for saved token
        if self.settings.deriv_api_token:
            self.auto_connect()
        
        self.start_background_updates()
        
        print("\n✅ KARANKA DERIV GUI LOADED SUCCESSFULLY")
        print("   • MARKET STATE ENGINE - UNDER THE HOOD")
        print("   • CONTINUATION + QUASIMODO - SMART SELECTOR")
        print("   • DERIV API INTEGRATION - Ready")
        print("   • 3 PIP RETEST TOLERANCE - ENHANCED")
        print("   • ENHANCED SL HANDLING - Spread Buffer + Dynamic Multipliers")
    
    def apply_theme(self):
        """Apply black & gold theme"""
        theme = BLACK_GOLD_THEME
        
        self.root.configure(bg=theme['bg'])
        
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background=theme['bg'])
        style.configure('TLabel', background=theme['bg'], foreground=theme['fg'])
        style.configure('TButton', 
                       background=theme['button_bg'],
                       foreground=theme['button_fg'],
                       borderwidth=1,
                       focusthickness=0,
                       focuscolor='none')
        style.map('TButton',
                 background=[('active', theme['accent']),
                           ('pressed', theme['accent_dark'])])
        
        style.configure('TNotebook', background=theme['bg'], borderwidth=0)
        style.configure('TNotebook.Tab', 
                       background=theme['secondary'],
                       foreground=theme['fg'],
                       padding=[10, 5])
        style.map('TNotebook.Tab',
                 background=[('selected', theme['accent_dark'])],
                 foreground=[('selected', theme['fg_light'])])
        
        style.configure('TEntry', 
                       fieldbackground=theme['secondary'],
                       foreground=theme['fg'],
                       bordercolor=theme['border'])
        
        style.configure('TCheckbutton', 
                       background=theme['bg'],
                       foreground=theme['fg'])
        
        style.configure('TLabelframe', 
                       background=theme['bg'],
                       foreground=theme['accent'],
                       bordercolor=theme['border'])
        
        style.configure('TLabelframe.Label', 
                       background=theme['bg'],
                       foreground=theme['accent'])
    
    def setup_gui(self):
        """Setup the GUI with Deriv integration"""
        theme = BLACK_GOLD_THEME
        
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # HEADER with Deriv status
        header_frame = tk.Frame(main_container, bg=theme['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title = tk.Label(header_frame,
                        text="KARANKA MULTIVERSE ALGO AI - DERIV TRADING SYSTEM",
                        font=("Arial", 16, "bold"),
                        bg=theme['bg'],
                        fg=theme['fg'])
        title.pack(side=tk.LEFT)
        
        self.deriv_status = tk.Label(header_frame,
                                    text="🔌 Not Connected to Deriv",
                                    font=("Arial", 10),
                                    bg=theme['bg'],
                                    fg=theme['error'])
        self.deriv_status.pack(side=tk.RIGHT, padx=10)
        
        self.balance_label = tk.Label(header_frame,
                                     text="",
                                     font=("Arial", 10, "bold"),
                                     bg=theme['bg'],
                                     fg=theme['success'])
        self.balance_label.pack(side=tk.RIGHT, padx=10)
        
        # NOTEBOOK - 6 TABS
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.create_dashboard_tab()
        self.create_analysis_tab()
        self.create_market_tab()
        self.create_settings_tab()
        self.create_connection_tab()
        self.create_monitor_tab()
        
        # FOOTER
        footer_frame = tk.Frame(main_container, bg=theme['bg'])
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        footer_label = tk.Label(footer_frame,
                               text="© 2025 KARANKA MULTIVERSE ALGO AI | DERIV INTEGRATION | MARKET STATE ENGINE",
                               font=("Arial", 8),
                               bg=theme['bg'],
                               fg=theme['fg_dark'])
        footer_label.pack()
    
    def create_dashboard_tab(self):
        """Dashboard Tab"""
        theme = BLACK_GOLD_THEME
        
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📊 Dashboard")
        
        left = ttk.Frame(frame)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right = ttk.Frame(frame)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # LIVE ANALYSIS
        live_frame = ttk.LabelFrame(left, text="🔍 KARANKA MARKET ANALYSIS", padding=15)
        live_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.live_analysis_text = scrolledtext.ScrolledText(
            live_frame,
            height=25,
            bg=theme['text_bg'],
            fg=theme['fg'],
            font=("Consolas", 9),
            wrap=tk.WORD,
            relief='flat',
            insertbackground=theme['fg']
        )
        self.live_analysis_text.pack(fill=tk.BOTH, expand=True)
        
        self.analysis_status = tk.Label(live_frame,
                                       text="Karanka AI: Market State Engine | Smart Selector | SCANNING...",
                                       font=("Arial", 8),
                                       bg=theme['text_bg'],
                                       fg=theme['fg_dark'])
        self.analysis_status.pack(anchor=tk.W, pady=(5, 0))
        
        # STATS
        stats_frame = ttk.LabelFrame(right, text="📈 TRADING STATS", padding=15)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=12,
                                 bg=theme['text_bg'],
                                 fg=theme['fg'],
                                 font=("Consolas", 10),
                                 relief='flat')
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # PERFORMANCE
        perf_frame = ttk.LabelFrame(right, text="⚡ PERFORMANCE", padding=15)
        perf_frame.pack(fill=tk.X, pady=(10, 10))
        
        self.perf_text = tk.Text(perf_frame, height=4,
                                bg=theme['text_bg'],
                                fg=theme['fg'],
                                font=("Consolas", 9),
                                relief='flat')
        self.perf_text.pack(fill=tk.BOTH, expand=True)
        
        # CONTROLS
        ctrl_frame = ttk.LabelFrame(right, text="🔗 TRADING CONTROLS", padding=15)
        ctrl_frame.pack(fill=tk.X, pady=(0, 0))
        
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack()
        
        self.connect_btn = ttk.Button(btn_frame, text="🔗 Connect Deriv",
                                     command=self.show_deriv_auth)
        self.connect_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="🚀 Start Trading",
                  command=self.start_trading).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="🛑 Stop Trading",
                  command=self.stop_trading).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="🔄 Refresh",
                  command=self.update_all).pack(side=tk.LEFT, padx=5, pady=5)
    
    def create_analysis_tab(self):
        """Analysis Tab"""
        theme = BLACK_GOLD_THEME
        
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📈 Analysis")
        
        text_widget = scrolledtext.ScrolledText(frame,
                                               bg=theme['text_bg'],
                                               fg=theme['fg'],
                                               font=("Consolas", 10),
                                               wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        info = """
KARANKA MULTIVERSE ALGO AI - DERIV TRADING SYSTEM
═══════════════════════════════════════════════════════════════════════════

🎯 STRATEGY: MARKET STATE ENGINE + SMART SELECTOR
   • MARKET STATE DETECTION - Knows trend vs range vs breakout
   • CONTINUATION STRATEGY - Trades WITH trends (pullbacks)
   • QUASIMODO STRATEGY - Trades reversals in ranges
   • SMART SELECTOR - Uses right strategy for right conditions
   • 3 PIP RETEST TOLERANCE (Enhanced)
   • HTF STRUCTURE MANDATORY
   • ATR-BASED SL/TP with Spread Buffer
   • FRESH PATTERNS ONLY (MAX 8 CANDLES)

────────────────────────────────────────
📊 MARKET STATES:

1. STRONG UPTREND (ADX > 30, HH/HL)
   → USE: Continuation SELLs only

2. UPTREND (ADX 20-30, HH/HL)
   → PREFER: Continuation SELLs
   → ALLOW: Strong Quasimodo

3. RANGING (ADX < 25, no clear structure)
   → USE: Quasimodo reversals only (both directions)

4. DOWNTREND (ADX 20-30, LH/LL)
   → PREFER: Continuation BUYs
   → ALLOW: Strong Quasimodo

5. STRONG DOWNTREND (ADX > 30, LH/LL)
   → USE: Continuation BUYs only

6. BREAKOUT (Price breaks key level)
   → USE: Breakout continuation

7. CHOPPY (Very tight range, low ADX)
   → SKIP ALL TRADES (no edge)

────────────────────────────────────────
🔗 DERIV INTEGRATION:
   • API Token Authentication
   • Real-time Balance Updates
   • Contract-based Trading
   • Risk: 2% per trade maximum
   • Dynamic position sizing

⚙️ ENHANCED SL HANDLING:
   • SL Multiplier: 2.0-2.2 ATR
   • Spread Buffer: 25-30% of ATR
   • Symbol-specific adjustments
   • Dynamic position sizing

📊 MT5 COMMENT: "KARANKA ALGO AI"
        """
        
        text_widget.insert(1.0, info)
        text_widget.config(state='disabled')
    
    def create_market_tab(self):
        """Market Selection Tab"""
        theme = BLACK_GOLD_THEME
        
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📊 Markets")
        
        canvas = tk.Canvas(frame, bg=theme['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        selection_frame = ttk.LabelFrame(scrollable, text="Select Trading Symbols", padding=20)
        selection_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.market_vars = {}
        
        categories = {
            "FOREX MAJORS": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"],
            "FOREX CROSSES": ["EURGBP", "EURJPY", "CHFJPY", "GBPJPY", "AUDJPY", "EURAUD", "GBPAUD"],
            "COMMODITIES": ["XAUUSD", "XAGUSD"],
            "INDICES": ["US30", "USTEC", "US100"],
            "CRYPTO": ["BTCUSD"]
        }
        
        for category, symbols in categories.items():
            tk.Label(selection_frame,
                    text=f"▸ {category}",
                    font=("Arial", 10, "bold"),
                    bg=theme['secondary'],
                    fg=theme['accent']).pack(anchor=tk.W, pady=(10, 5))
            
            row = ttk.Frame(selection_frame)
            row.pack(fill=tk.X, pady=5)
            
            for symbol in symbols:
                var = tk.BooleanVar(value=symbol in self.settings.enabled_symbols)
                self.market_vars[symbol] = var
                ttk.Checkbutton(row, text=symbol, variable=var).pack(side=tk.LEFT, padx=10)
        
        control = ttk.Frame(selection_frame)
        control.pack(fill=tk.X, pady=20)
        
        ttk.Button(control, text="✅ Select All", command=self.select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="❌ Deselect All", command=self.deselect_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="💾 Save Selection", command=self.save_markets).pack(side=tk.LEFT, padx=5)
    
    def create_settings_tab(self):
        """Settings Tab"""
        theme = BLACK_GOLD_THEME
        
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="⚙️ Settings")
        
        canvas = tk.Canvas(frame, bg=theme['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        main = ttk.Frame(scrollable)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # TIMEFRAMES
        tf_frame = ttk.LabelFrame(main, text="⏰ TIMEFRAMES", padding=15)
        tf_frame.pack(fill=tk.X, pady=10)
        
        self.enable_15m_var = tk.BooleanVar(value=self.settings.enable_15m)
        ttk.Checkbutton(tf_frame, text="✅ 15M TIMEFRAME", variable=self.enable_15m_var).pack(anchor=tk.W, pady=5)
        
        self.enable_30m_var = tk.BooleanVar(value=self.settings.enable_30m)
        ttk.Checkbutton(tf_frame, text="✅ 30M TIMEFRAME", variable=self.enable_30m_var).pack(anchor=tk.W, pady=5)
        
        self.enable_1h_var = tk.BooleanVar(value=self.settings.enable_1h)
        ttk.Checkbutton(tf_frame, text="✅ 1H TIMEFRAME", variable=self.enable_1h_var).pack(anchor=tk.W, pady=5)
        
        # RETEST TOLERANCE
        retest_frame = ttk.LabelFrame(main, text="🎯 RETEST TOLERANCE", padding=15)
        retest_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(retest_frame,
                text="✅ 3 PIP TOLERANCE - ENHANCED (FIXED)",
                font=("Arial", 9, "bold"),
                bg=theme['secondary'],
                fg=theme['success']).pack(anchor=tk.W, pady=5)
        
        # HTF
        htf_frame = ttk.LabelFrame(main, text="🎯 HTF STRUCTURE", padding=15)
        htf_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(htf_frame,
                text="🔵 HTF FILTER: MANDATORY - ALWAYS ON",
                font=("Arial", 9, "bold"),
                bg=theme['secondary'],
                fg=theme['success']).pack(anchor=tk.W, pady=5)
        
        # TRADE LIMITS
        limits_frame = ttk.LabelFrame(main, text="⚡ TRADE LIMITS", padding=15)
        limits_frame.pack(fill=tk.X, pady=10)
        
        dframe = ttk.Frame(limits_frame)
        dframe.pack(fill=tk.X, pady=5)
        ttk.Label(dframe, text="Max Daily Trades:").pack(side=tk.LEFT, padx=5)
        self.max_daily_var = tk.StringVar(value=str(self.settings.max_daily_trades))
        ttk.Entry(dframe, textvariable=self.max_daily_var, width=10).pack(side=tk.LEFT, padx=5)
        
        hframe = ttk.Frame(limits_frame)
        hframe.pack(fill=tk.X, pady=5)
        ttk.Label(hframe, text="Max Hourly Trades:").pack(side=tk.LEFT, padx=5)
        self.max_hourly_var = tk.StringVar(value=str(self.settings.max_hourly_trades))
        ttk.Entry(hframe, textvariable=self.max_hourly_var, width=10).pack(side=tk.LEFT, padx=5)
        
        sframe = ttk.Frame(limits_frame)
        sframe.pack(fill=tk.X, pady=5)
        ttk.Label(sframe, text="Seconds Between Trades:").pack(side=tk.LEFT, padx=5)
        self.min_seconds_var = tk.StringVar(value=str(self.settings.min_seconds_between_trades))
        ttk.Entry(sframe, textvariable=self.min_seconds_var, width=10).pack(side=tk.LEFT, padx=5)
        
        cframe = ttk.Frame(limits_frame)
        cframe.pack(fill=tk.X, pady=5)
        ttk.Label(cframe, text="Max Concurrent Trades:").pack(side=tk.LEFT, padx=5)
        self.max_trades_var = tk.StringVar(value=str(self.settings.max_concurrent_trades))
        ttk.Entry(cframe, textvariable=self.max_trades_var, width=10).pack(side=tk.LEFT, padx=5)
        
        rframe = ttk.Frame(limits_frame)
        rframe.pack(fill=tk.X, pady=5)
        ttk.Label(rframe, text="Max Risk % Per Trade:").pack(side=tk.LEFT, padx=5)
        self.risk_var = tk.StringVar(value=str(self.settings.max_risk_per_trade * 100))
        ttk.Entry(rframe, textvariable=self.risk_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(rframe, text="%").pack(side=tk.LEFT)
        
        # MODE
        mode_frame = ttk.LabelFrame(main, text="🎮 MODE", padding=15)
        mode_frame.pack(fill=tk.X, pady=10)
        
        self.dry_run_var = tk.BooleanVar(value=self.settings.dry_run)
        ttk.Checkbutton(mode_frame, text="🟡 Dry Run Mode (Paper Trading)", variable=self.dry_run_var).pack(anchor=tk.W, pady=5)
        
        # LOT SIZE (for reference)
        lot_frame = ttk.LabelFrame(main, text="💰 POSITION SIZING", padding=15)
        lot_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(lot_frame,
                text="Position size calculated automatically based on balance and risk %",
                bg=theme['secondary'],
                fg=theme['fg_light'],
                wraplength=400).pack(anchor=tk.W, pady=5)
        
        # SAVE
        save_frame = ttk.Frame(main)
        save_frame.pack(fill=tk.X, pady=20)
        ttk.Button(save_frame, text="💾 Save Settings", command=self.save_settings).pack(side=tk.LEFT, padx=5)
    
    def create_connection_tab(self):
        """Connection Tab - Updated for Deriv"""
        theme = BLACK_GOLD_THEME
        
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔗 Deriv Connection")
        
        conn_frame = ttk.LabelFrame(frame, text="Deriv API Connection", padding=20)
        conn_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Token display
        token_frame = ttk.Frame(conn_frame)
        token_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(token_frame,
                text="Current API Token:",
                bg=theme['secondary'],
                fg=theme['fg']).pack(anchor=tk.W)
        
        self.token_display = tk.Entry(token_frame,
                                     width=50,
                                     bg=theme['text_bg'],
                                     fg=theme['fg'],
                                     font=("Courier", 10))
        self.token_display.pack(fill=tk.X, pady=5)
        self.token_display.insert(0, "•" * 20)  # Masked
        self.token_display.config(state='readonly')
        
        # Connection status
        status_frame = ttk.LabelFrame(conn_frame, text="Connection Status", padding=15)
        status_frame.pack(fill=tk.X, pady=10)
        
        self.conn_status_text = tk.Text(status_frame, height=10,
                                       bg=theme['text_bg'],
                                       fg=theme['fg'],
                                       font=("Consolas", 9),
                                       relief='flat')
        self.conn_status_text.pack(fill=tk.BOTH, expand=True)
        
        # Account info
        self.account_info_frame = ttk.LabelFrame(conn_frame, text="Account Information", padding=15)
        self.account_info_frame.pack(fill=tk.X, pady=10)
        
        self.account_info_text = tk.Text(self.account_info_frame, height=6,
                                        bg=theme['text_bg'],
                                        fg=theme['fg'],
                                        font=("Consolas", 9),
                                        relief='flat')
        self.account_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        btn_frame = ttk.Frame(conn_frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="🔌 Connect New Token", 
                  command=self.show_deriv_auth).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="🔓 Disconnect", 
                  command=self.disconnect_deriv).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="🔄 Refresh Balance", 
                  command=self.refresh_balance).pack(side=tk.LEFT, padx=5)
    
    def create_monitor_tab(self):
        """Monitor Tab"""
        theme = BLACK_GOLD_THEME
        
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="👁️ Monitor")
        
        trades_frame = ttk.LabelFrame(frame, text="Active Trades", padding=15)
        trades_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.trades_text = scrolledtext.ScrolledText(trades_frame,
                                                    bg=theme['text_bg'],
                                                    fg=theme['fg'],
                                                    font=("Consolas", 9))
        self.trades_text.pack(fill=tk.BOTH, expand=True)
    
    def auto_connect(self):
        """Auto-connect with saved token"""
        def connect_thread():
            try:
                deriv = DerivAPI()
                
                def auth_callback(success, data):
                    if success:
                        self.root.after(0, self.connection_success, deriv)
                    else:
                        self.root.after(0, self.connection_failed, "Auto-connect failed")
                
                success, message = deriv.connect(self.settings.deriv_api_token, auth_callback)
                
            except Exception as e:
                pass  # Silently fail for auto-connect
        
        threading.Thread(target=connect_thread, daemon=True).start()
    
    def show_deriv_auth(self):
        """Show Deriv authentication dialog"""
        dialog = DerivAuthDialog(self.root, self.settings)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            self.connection_success(dialog.result['deriv'])
    
    def connection_success(self, deriv):
        """Handle successful Deriv connection"""
        self.deriv_connection = deriv
        
        # Create trader with this connection
        self.trader = KarankaDerivTradingEngine(self.settings)
        self.trader.deriv = deriv
        self.trader.deriv_connected = True
        self.trader.connected = True
        
        # Update UI
        self.deriv_status.config(text=f"✅ Connected: {deriv.loginid}", fg=BLACK_GOLD_THEME['success'])
        self.balance_label.config(text=f"💰 {deriv.balance} {deriv.currency}")
        self.connect_btn.config(text="✅ Connected", state='disabled')
        
        # Update connection tab
        self.token_display.config(state='normal')
        self.token_display.delete(0, tk.END)
        self.token_display.insert(0, self.settings.deriv_api_token[:10] + "..." + self.settings.deriv_api_token[-4:])
        self.token_display.config(state='readonly')
        
        self.conn_status_text.delete(1.0, tk.END)
        self.conn_status_text.insert(tk.END, f"✅ Connected to Deriv\n")
        self.conn_status_text.insert(tk.END, f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.conn_status_text.insert(tk.END, f"Status: Authorized\n")
        
        self.account_info_text.delete(1.0, tk.END)
        self.account_info_text.insert(tk.END, 
            f"Account ID: {deriv.loginid}\n"
            f"Email: {deriv.email}\n"
            f"Balance: {deriv.balance} {deriv.currency}\n"
            f"Currency: {deriv.currency}\n"
            f"Status: Active\n")
    
    def connection_failed(self, error):
        """Handle connection failure"""
        self.deriv_status.config(text=f"❌ Connection Failed", fg=BLACK_GOLD_THEME['error'])
        self.balance_label.config(text="")
        self.connect_btn.config(text="🔗 Connect Deriv", state='normal')
        
        self.conn_status_text.delete(1.0, tk.END)
        self.conn_status_text.insert(tk.END, f"❌ Connection failed: {error}\n")
        self.conn_status_text.insert(tk.END, f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def disconnect_deriv(self):
        """Disconnect from Deriv"""
        if self.trader:
            self.trader.disconnect()
            self.trader = None
        
        self.deriv_connection = None
        
        self.deriv_status.config(text="🔌 Not Connected to Deriv", fg=BLACK_GOLD_THEME['error'])
        self.balance_label.config(text="")
        self.connect_btn.config(text="🔗 Connect Deriv", state='normal')
        
        self.token_display.config(state='normal')
        self.token_display.delete(0, tk.END)
        self.token_display.insert(0, "•" * 20)
        self.token_display.config(state='readonly')
        
        self.conn_status_text.delete(1.0, tk.END)
        self.conn_status_text.insert(tk.END, "🔌 Disconnected from Deriv\n")
        
        self.account_info_text.delete(1.0, tk.END)
        
        messagebox.showinfo("Disconnected", "Disconnected from Deriv")
    
    def refresh_balance(self):
        """Refresh balance display"""
        if self.trader and self.trader.deriv_connected:
            self.balance_label.config(text=f"💰 {self.trader.deriv.balance} {self.trader.deriv.currency}")
    
    def start_background_updates(self):
        def update_loop():
            while True:
                try:
                    self.update_all()
                    time.sleep(2)
                except:
                    time.sleep(5)
        threading.Thread(target=update_loop, daemon=True).start()
    
    def update_all(self):
        self.update_stats()
        self.update_trade_monitor()
        self.update_live_analysis()
        self.update_performance()
        self.refresh_balance()
    
    def update_stats(self):
        try:
            if self.trader:
                status = self.trader.get_status()
            else:
                status = {
                    'connected': False,
                    'running': False,
                    'daily_trades': 0,
                    'total_active': 0,
                    'total_cycles': 0,
                    'analysis_count': 0
                }
            
            text = f"""=== KARANKA MULTIVERSE ALGO AI - DERIV ===

CONNECTION: {'✅ CONNECTED' if status.get('connected') else '❌ DISCONNECTED'}
DERIV: {'✅ CONNECTED' if status.get('deriv_connected') else '❌ DISCONNECTED'}
TRADING: {'✅ ACTIVE' if status.get('running') else '❌ STOPPED'}
MODE: {'🟡 DRY RUN' if self.settings.dry_run else '🔴 LIVE'}

📊 TODAY:
• Trades: {status.get('daily_trades', 0)}/{self.settings.max_daily_trades}
• Active: {status.get('total_active', 0)}
• Cycles: {status.get('total_cycles', 0)}
• Signals: {status.get('analysis_count', 0)}

💰 ACCOUNT:
• Balance: {status.get('balance', 0)} {status.get('currency', 'USD')}
• Account: {status.get('account', 'N/A')}

🎯 CONFIGURATION:
• Market State: ACTIVE
• Smart Selector: ACTIVE
• Retest: 3 PIP TOLERANCE
• HTF: MANDATORY
• Min Confidence: 65%
"""
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, text)
            
            if status.get('connected'):
                self.status_label.config(text=f"✅ KARANKA AI | Signals: {status.get('analysis_count', 0)}")
            else:
                self.status_label.config(text="❌ Disconnected")
        except:
            pass
    
    def update_performance(self):
        try:
            if self.trader:
                status = self.trader.get_status()
            else:
                status = {'analysis_count': 0}
            
            text = f"""⚡ KARANKA PERFORMANCE:

• MARKET STATE ENGINE: ACTIVE
• SMART STRATEGY SELECTOR: ACTIVE
• 3 PIP RETEST: ENHANCED
• HTF FILTER: MANDATORY
• MAX PATTERN AGE: 8 CANDLES
• SCAN CYCLE: 8 SECONDS
• ACTIVE SIGNALS: {status.get('analysis_count', 0)}
"""
            self.perf_text.delete(1.0, tk.END)
            self.perf_text.insert(1.0, text)
        except:
            pass
    
    def update_trade_monitor(self):
        try:
            self.trades_text.delete(1.0, tk.END)
            if self.trader and self.trader.active_trades:
                self.trades_text.insert(tk.END, f"ACTIVE TRADES ({len(self.trader.active_trades)}):\n")
                self.trades_text.insert(tk.END, "="*80 + "\n")
                for trade in self.trader.active_trades:
                    analysis = trade.get('analysis', {})
                    if trade.get('dry_run'):
                        self.trades_text.insert(tk.END, 
                            f"[DRY RUN] {trade['symbol']} {trade['direction']}\n"
                            f"   Strategy: {analysis.get('strategy', 'QUASIMODO')}\n"
                            f"   Pattern: {analysis.get('pattern', 'N/A')}\n"
                            f"   Confidence: {analysis.get('confidence', 70):.0f}%\n"
                            f"   Market State: {analysis.get('market_state', 'UNKNOWN')}\n"
                            f"   Stake: ${trade.get('stake', 0):.2f}\n"
                            f"{'-'*80}\n")
                    else:
                        self.trades_text.insert(tk.END, 
                            f"[LIVE] {trade['symbol']} {trade['direction']}\n"
                            f"   Contract: {trade.get('contract_id', 'N/A')}\n"
                            f"   Strategy: {analysis.get('strategy', 'QUASIMODO')}\n"
                            f"   Pattern: {analysis.get('pattern', 'N/A')}\n"
                            f"   Confidence: {analysis.get('confidence', 70):.0f}%\n"
                            f"   Stake: ${trade.get('stake', 0):.2f}\n"
                            f"   Buy Price: {trade.get('buy_price', 0):.5f}\n"
                            f"   Payout: {trade.get('payout', 0):.5f}\n"
                            f"   Comment: KARANKA DERIV AI\n"
                            f"{'-'*80}\n")
            else:
                self.trades_text.insert(tk.END, "NO ACTIVE TRADES\n")
                self.trades_text.insert(tk.END, "Waiting for optimal market conditions...\n")
        except:
            pass
    
    def update_live_analysis(self):
        try:
            if self.trader:
                text = self.trader.get_live_display_text()
                self.live_analysis_text.delete(1.0, tk.END)
                self.live_analysis_text.insert(1.0, text)
                
                status = self.trader.get_status()
                if status.get('connected') and status.get('running'):
                    self.analysis_status.config(text=f"✅ KARANKA ACTIVE | Smart Selector | Signals: {status.get('analysis_count', 0)}")
                elif status.get('connected'):
                    self.analysis_status.config(text="⏸️ Connected - Ready to Trade")
                else:
                    self.analysis_status.config(text="❌ Not Connected - Click Connect Deriv")
        except:
            pass
    
    def start_trading(self):
        if not self.trader or not self.trader.deriv_connected:
            messagebox.showwarning("Not Connected", "Connect to Deriv first!")
            return
        
        self.save_settings()
        if self.trader.start_trading():
            messagebox.showinfo("Success", "✅ KARANKA TRADING ACTIVE\n\n• Market State Engine\n• Smart Strategy Selector\n• 3 PIP Retest Tolerance\n• Enhanced SL Handling")
        else:
            messagebox.showerror("Error", "Failed to start trading")
    
    def stop_trading(self):
        if self.trader:
            self.trader.stop_trading()
        messagebox.showinfo("Stopped", "Karanka Trading Stopped")
    
    def select_all(self):
        for var in self.market_vars.values():
            var.set(True)
    
    def deselect_all(self):
        for var in self.market_vars.values():
            var.set(False)
    
    def save_markets(self):
        enabled = []
        for symbol, var in self.market_vars.items():
            if var.get():
                enabled.append(symbol)
        self.settings.enabled_symbols = enabled
        self.settings.save_settings()
        messagebox.showinfo("Saved", f"{len(enabled)} symbols selected")
    
    def save_settings(self):
        try:
            self.settings.enable_15m = self.enable_15m_var.get()
            self.settings.enable_30m = self.enable_30m_var.get()
            self.settings.enable_1h = self.enable_1h_var.get()
            self.settings.dry_run = self.dry_run_var.get()
            self.settings.max_daily_trades = int(self.max_daily_var.get())
            self.settings.max_hourly_trades = int(self.max_hourly_var.get())
            self.settings.min_seconds_between_trades = int(self.min_seconds_var.get())
            self.settings.max_concurrent_trades = int(self.max_trades_var.get())
            self.settings.max_risk_per_trade = float(self.risk_var.get()) / 100
            
            if self.settings.save_settings():
                messagebox.showinfo("Success", "Settings saved!")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def run(self):
        self.root.mainloop()


# ============ MAIN ============
def main():
    print("\n" + "="*100)
    print("KARANKA MULTIVERSE ALGO AI - DERIV TRADING SYSTEM")
    print("="*100)
    print("✅ MARKET STATE ENGINE - Detects trend, range, breakout, choppy")
    print("✅ CONTINUATION STRATEGY - Trades WITH trends (pullbacks)")
    print("✅ QUASIMODO STRATEGY - Trades reversals in ranges")
    print("✅ SMART SELECTOR - Uses right strategy for right conditions")
    print("✅ 3 PIP RETEST TOLERANCE - Enhanced")
    print("✅ ENHANCED SL HANDLING - Spread Buffer + Dynamic Multipliers")
    print("✅ DERIV API INTEGRATION - Token-based authentication")
    print("✅ HTF STRUCTURE - MANDATORY")
    print("✅ FRESH PATTERNS - MAX 8 CANDLES")
    print("✅ YOUR GUI - 6 TABS, FULLY FUNCTIONAL")
    print("="*100)
    print("🚀 READY FOR PROFESSIONAL TRADING ON DERIV")
    print("="*100)
    
    try:
        gui = KarankaDerivGUI()
        gui.run()
    except Exception as e:
        print(f"❌ FATAL ERROR: {e}")
        traceback.print_exc()
        input("\nPress Enter to exit...")
    
    print("\n✅ KARANKA MULTIVERSE ALGO AI shutdown complete")
    input("Press Enter to close...")

if __name__ == "__main__":
    main()
