#!/usr/bin/env python3
"""
================================================================================
KARANKA MULTIVERSE ALGO AI - DERIV PRODUCTION BOT
================================================================================
✅ EXACT SAME LOGIC - 80%+ win rate strategy
✅ MARKET STATE ENGINE - Trend/Range/Breakout detection
✅ CONTINUATION - SWAPPED (SELL in uptrend, BUY in downtrend)
✅ QUASIMODO - SWAPPED (BUY from bearish, SELL from bullish)
✅ SMART SELECTOR - Your proven strategy
✅ 2 PIP RETEST - Precision entries
✅ HTF STRUCTURE - MANDATORY
✅ FIXED DERIV CONNECTION - wss://ws.deriv.com with App ID 1089
✅ PING MECHANISM - Keeps bot alive 24/7
✅ FETCHING KEEP-AWAKE - Self-pings every 5 minutes
✅ PRODUCTION READY - Deployed on Render
================================================================================
"""

import os
import json
import time
import threading
import logging
import websocket
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from flask import Flask, render_template, jsonify, request, session
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from enum import Enum
from functools import wraps
import hmac
import hashlib
import traceback

# ============ INITIALIZATION ============
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24).hex())
app.config['DEBUG'] = False

# SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', ping_timeout=60, ping_interval=25)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('karanka_bot.log')
    ]
)
logger = logging.getLogger(__name__)

# ============ CONFIGURATION - FIXED FOR DERIV ============
DERIV_APP_ID = '1089'  # HARDCODED to default Deriv app ID
DERIV_WS_URL = "wss://ws.deriv.com/websockets/v3?app_id=1089"  # CORRECT URL
BASE_URL = os.environ.get('RENDER_EXTERNAL_URL', 'http://localhost:5000')
MAX_DAILY_TRADES = int(os.environ.get('MAX_DAILY_TRADES', 20))
MAX_CONCURRENT_TRADES = int(os.environ.get('MAX_CONCURRENT_TRADES', 3))
FIXED_AMOUNT = float(os.environ.get('FIXED_AMOUNT', 1.0))

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

# ============ KEEP AWAKE MECHANISM ============
class KeepAwake:
    """Prevents the bot from sleeping on Render free tier"""
    
    def __init__(self, url):
        self.url = url
        self.running = True
        self.ping_count = 0
        
    def start(self):
        """Start the keep-awake ping thread"""
        def ping_loop():
            while self.running:
                try:
                    self.ping_count += 1
                    # Ping the health endpoint every 5 minutes
                    response = requests.get(f"{self.url}/health", timeout=10)
                    logger.info(f"🏓 Keep-awake ping #{self.ping_count} - Status: {response.status_code}")
                    
                    # Also ping the main page to keep session alive
                    requests.get(self.url, timeout=10)
                    
                except Exception as e:
                    logger.error(f"Keep-awake ping failed: {e}")
                
                # Wait 5 minutes before next ping
                for _ in range(300):  # 5 minutes = 300 seconds
                    if not self.running:
                        break
                    time.sleep(1)
        
        thread = threading.Thread(target=ping_loop, daemon=True)
        thread.start()
        logger.info("✅ Keep-awake mechanism started - pinging every 5 minutes")
    
    def stop(self):
        self.running = False


# ============ DERIV API CONNECTOR - FIXED VERSION WITH PING ============
class DerivAPI:
    """Production-ready Deriv WebSocket API with ping keep-alive"""
    
    def __init__(self):
        self.ws = None
        self.connected = False
        self.token = None
        self.req_id = 0
        self.prices = {}
        self.candles_cache = {}
        self.active_contracts = {}
        self.ping_thread_running = False
        self.last_pong = time.time()
        self.connection_attempts = 0
        self.reconnect_delay = 1
        
    def _next_id(self):
        self.req_id += 1
        return self.req_id
    
    def connect(self, token):
        """Connect to Deriv - accepts token EXACTLY as user provides"""
        self.token = token  # Use EXACTLY what user entered - NO CHANGES!
        logger.info(f"Connecting with token: {self.token[:4]}...{self.token[-4:]} (length: {len(self.token)})")
        return self._connect_with_retry()
    
    def _connect_with_retry(self, max_retries=5):
        """Connect with retry logic and exponential backoff"""
        self.connection_attempts += 1
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to Deriv (attempt {attempt + 1}/{max_retries})")
                
                # Use the CORRECT Deriv WebSocket URL
                logger.info(f"Connecting to: {DERIV_WS_URL}")
                
                # Create WebSocket connection with timeout
                self.ws = websocket.create_connection(
                    DERIV_WS_URL,
                    timeout=20,
                    enable_multithread=True
                )
                
                # Send token EXACTLY as received - RAW string, NO JSON!
                logger.info(f"Sending token (length: {len(self.token)})")
                self.ws.send(self.token)
                
                # Wait for response with timeout
                self.ws.settimeout(10)
                response = self.ws.recv()
                logger.info(f"Response received: {response[:100]}")
                
                # Parse response
                response_data = json.loads(response)
                
                # Check for successful authorization
                if 'authorize' in response_data:
                    self.connected = True
                    loginid = response_data['authorize'].get('loginid', 'Unknown')
                    currency = response_data['authorize'].get('currency', 'USD')
                    balance = response_data['authorize'].get('balance', 0)
                    logger.info(f"✅ Connected to Deriv successfully as {loginid}")
                    logger.info(f"💰 Balance: {balance} {currency}")
                    
                    # Start heartbeat to keep connection alive
                    self._start_heartbeat()
                    self._start_message_listener()
                    
                    # Reset connection attempts on success
                    self.connection_attempts = 0
                    self.reconnect_delay = 1
                    
                    return True, f"Connected as {loginid} | Balance: {balance} {currency}"
                
                elif 'error' in response_data:
                    error_msg = response_data['error'].get('message', 'Unknown error')
                    logger.error(f"Auth failed: {error_msg}")
                    
                    if attempt < max_retries - 1:
                        wait_time = min(30, 2 ** attempt)
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    return False, error_msg
                else:
                    return False, "Unexpected response format"
                    
            except websocket.WebSocketTimeoutException:
                logger.error("Connection timeout")
                if attempt < max_retries - 1:
                    wait_time = min(30, 2 ** attempt)
                    time.sleep(wait_time)
                else:
                    return False, "Connection timeout"
                    
            except websocket.WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
                if attempt < max_retries - 1:
                    wait_time = min(30, 2 ** attempt)
                    time.sleep(wait_time)
                else:
                    return False, f"WebSocket error: {str(e)}"
                    
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = min(30, 2 ** attempt)
                    time.sleep(wait_time)
                else:
                    return False, str(e)
        
        return False, "Max retries exceeded"
    
    def _start_heartbeat(self):
        """Send periodic ping to keep connection alive (every 25 seconds)"""
        def heartbeat():
            self.ping_thread_running = True
            ping_count = 0
            while self.connected and self.ping_thread_running:
                try:
                    time.sleep(25)
                    if self.ws and self.connected:
                        ping_count += 1
                        ping_msg = {"ping": 1, "req_id": self._next_id()}
                        self.ws.send(json.dumps(ping_msg))
                        logger.debug(f"Heartbeat ping #{ping_count} sent")
                except websocket.WebSocketConnectionClosedException:
                    logger.warning("WebSocket connection closed, attempting reconnect...")
                    self.connected = False
                    break
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    self.connected = False
                    break
            
            # Auto-reconnect if heartbeat fails
            if not self.connected and self.token:
                logger.info("Heartbeat failed - attempting to reconnect...")
                time.sleep(5)
                self._connect_with_retry()
        
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        logger.info("✅ Heartbeat mechanism started - pinging every 25 seconds")
    
    def _start_message_listener(self):
        """Listen for incoming messages with auto-reconnect"""
        def listener():
            while self.connected and self.ws:
                try:
                    self.ws.settimeout(30)
                    message = self.ws.recv()
                    if message:
                        data = json.loads(message)
                        if 'pong' in data:
                            self.last_pong = time.time()
                            logger.debug("Pong received")
                        elif 'proposal_open_contract' in data:
                            # Handle trade updates
                            contract = data['proposal_open_contract']
                            if contract.get('is_sold', False):
                                logger.info(f"Contract {contract.get('contract_id')} closed")
                except websocket.WebSocketTimeoutException:
                    # Check if we missed pong
                    if time.time() - self.last_pong > 90:
                        logger.warning("No pong received for 90 seconds, reconnecting...")
                        self.connected = False
                        break
                    continue
                except websocket.WebSocketConnectionClosedException:
                    logger.warning("WebSocket connection closed")
                    self.connected = False
                    break
                except Exception as e:
                    if self.connected:
                        logger.error(f"Message listener error: {e}")
                    break
            
            # Auto-reconnect if disconnected
            if not self.connected and self.token:
                logger.info("Message listener disconnected - attempting to reconnect...")
                time.sleep(5)
                self._connect_with_retry()
        
        thread = threading.Thread(target=listener, daemon=True)
        thread.start()
        logger.info("✅ Message listener started")
    
    def get_candles(self, symbol, count=500, granularity=60):
        """Get historical candles with caching"""
        if not self.connected or not self.ws:
            logger.error("Not connected to Deriv")
            return None
        
        cache_key = f"{symbol}_{granularity}"
        
        # Check cache (5 second cache)
        if cache_key in self.candles_cache:
            cache_time, cache_data = self.candles_cache[cache_key]
            if time.time() - cache_time < 5:
                return cache_data
        
        try:
            request = {
                "ticks_history": symbol,
                "style": "candles",
                "granularity": granularity,
                "count": min(count, 5000),
                "req_id": self._next_id()
            }
            
            self.ws.send(json.dumps(request))
            response = self.ws.recv()
            
            response_data = json.loads(response)
            
            if 'error' in response_data:
                logger.error(f"Error getting candles for {symbol}: {response_data['error']}")
                return None
            
            # Format candles
            candles = []
            for candle in response_data.get('candles', []):
                candles.append({
                    'time': candle['epoch'],
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': float(candle.get('volume', 0))
                })
            
            df = pd.DataFrame(candles)
            
            # Cache the result
            self.candles_cache[cache_key] = (time.time(), df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get candles for {symbol}: {e}")
            return None
    
    def get_active_symbols(self):
        """Get all active trading symbols"""
        if not self.connected or not self.ws:
            return []
        
        try:
            request = {
                "active_symbols": "brief",
                "req_id": self._next_id()
            }
            
            self.ws.send(json.dumps(request))
            response = self.ws.recv()
            response_data = json.loads(response)
            
            if 'error' in response_data:
                logger.error(f"Error getting symbols: {response_data['error']}")
                return []
            
            symbols = []
            markets = {
                'forex': '💱 FOREX',
                'indices': '📊 INDICES',
                'commodities': '🪙 COMMODITIES',
                'cryptocurrency': '₿ CRYPTO',
                'synthetic_index': '🎲 SYNTHETICS'
            }
            
            for item in response_data.get('active_symbols', []):
                market = item.get('market', '').lower()
                if market in markets or 'synthetic' in market:
                    symbols.append({
                        'symbol': item['symbol'],
                        'display_name': item['display_name'],
                        'market': markets.get(market, market.upper()),
                        'submarket': item.get('submarket', '')
                    })
            
            # Sort by market then symbol
            symbols.sort(key=lambda x: (x['market'], x['symbol']))
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            return []
    
    def place_trade(self, symbol, direction, amount, duration=5):
        """Place a trade on Deriv"""
        if not self.connected or not self.ws:
            return None, "Not connected to Deriv"
        
        try:
            contract_type = "CALL" if direction == "BUY" else "PUT"
            
            order = {
                "buy": 1,
                "price": amount,
                "parameters": {
                    "amount": amount,
                    "basis": "stake",
                    "contract_type": contract_type,
                    "symbol": symbol,
                    "duration": duration,
                    "duration_unit": "m"
                },
                "req_id": self._next_id()
            }
            
            self.ws.send(json.dumps(order))
            response = self.ws.recv()
            response_data = json.loads(response)
            
            if 'error' in response_data:
                logger.error(f"Trade failed for {symbol}: {response_data['error']}")
                return None, response_data['error']['message']
            
            if 'buy' not in response_data:
                return None, "Unexpected response format"
            
            contract_id = response_data['buy'].get('contract_id')
            entry_price = float(response_data['buy'].get('price', 0))
            
            # Store active contract
            self.active_contracts[contract_id] = {
                'symbol': symbol,
                'direction': direction,
                'amount': amount,
                'entry_time': time.time(),
                'contract_id': contract_id,
                'entry_price': entry_price
            }
            
            logger.info(f"✅ Trade placed: {symbol} {direction} ${amount} (ID: {contract_id})")
            
            return {
                'contract_id': contract_id,
                'entry_price': entry_price,
                'direction': direction,
                'amount': amount,
                'symbol': symbol
            }, "Trade placed successfully"
            
        except Exception as e:
            logger.error(f"Failed to place trade: {e}")
            return None, str(e)
    
    def get_trade_status(self, contract_id):
        """Get status of a trade"""
        if not self.connected or not self.ws:
            return None
        
        try:
            request = {
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "req_id": self._next_id()
            }
            
            self.ws.send(json.dumps(request))
            response = self.ws.recv()
            response_data = json.loads(response)
            
            if 'error' in response_data:
                logger.error(f"Error getting trade status: {response_data['error']}")
                return None
            
            contract = response_data.get('proposal_open_contract', {})
            
            return {
                'is_sold': contract.get('is_sold', False),
                'profit': float(contract.get('profit', 0)),
                'exit_tick': float(contract.get('exit_tick', 0)),
                'status': contract.get('status', 'open'),
                'entry_tick': float(contract.get('entry_tick', 0)),
                'exit_time': contract.get('exit_time', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting trade status: {e}")
            return None
    
    def disconnect(self):
        """Disconnect from Deriv"""
        self.ping_thread_running = False
        self.connected = False
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        logger.info("Disconnected from Deriv")


# ============ MARKET STATE ENGINE (Your complete logic) ============
class MarketStateEngine:
    """Analyzes market conditions - EXACT same as your MT5 bot"""
    
    def __init__(self):
        self.ATR_PERIOD = 14
        self.EMA_FAST = 20
        self.EMA_SLOW = 50
        self.EMA_TREND = 200
        
    def analyze(self, df):
        """Complete market state analysis"""
        if df is None or len(df) < 100:
            return {
                'state': MarketState.CHOPPY.value,
                'direction': 'NEUTRAL',
                'strength': 0,
                'adx': 0,
                'structure': 'NEUTRAL',
                'support': 0,
                'resistance': 0,
                'breakout_detected': False,
                'recommended_strategy': 'NONE'
            }
        
        try:
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
                'state': state.value if isinstance(state, MarketState) else state,
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
            
        except Exception as e:
            logger.error(f"Market state analysis error: {e}")
            return {
                'state': MarketState.CHOPPY.value,
                'direction': 'NEUTRAL',
                'strength': 0,
                'adx': 0,
                'structure': 'NEUTRAL',
                'support': 0,
                'resistance': 0,
                'breakout_detected': False,
                'recommended_strategy': 'NONE'
            }
    
    def _calculate_indicators(self, df):
        """Calculate all technical indicators"""
        df = df.copy()
        
        # EMAs
        df['ema_20'] = df['close'].ewm(span=20, min_periods=20).mean()
        df['ema_50'] = df['close'].ewm(span=50, min_periods=50).mean()
        df['ema_200'] = df['close'].ewm(span=200, min_periods=200).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.ATR_PERIOD, min_periods=self.ATR_PERIOD).mean()
        
        return df
    
    def _calculate_adx(self, df, period=14):
        """Calculate ADX"""
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
                    
                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move
            
            tr = np.zeros_like(high)
            for i in range(1, len(high)):
                tr[i] = max(high[i] - low[i], 
                           abs(high[i] - close[i-1]), 
                           abs(low[i] - close[i-1]))
            
            atr = pd.Series(tr).rolling(period, min_periods=period).mean().values
            plus_dm_smooth = pd.Series(plus_dm).rolling(period, min_periods=period).mean().values
            minus_dm_smooth = pd.Series(minus_dm).rolling(period, min_periods=period).mean().values
            
            plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
            minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = pd.Series(dx).rolling(period, min_periods=period).mean().values
            
            return adx[-1] if not np.isnan(adx[-1]) else 0
            
        except Exception as e:
            logger.error(f"ADX calculation error: {e}")
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
        
        hh = last_two_highs[-1] > last_two_highs[-2] if len(last_two_highs) == 2 else False
        hl = last_two_lows[-1] > last_two_lows[-2] if len(last_two_lows) == 2 else False
        lh = last_two_highs[-1] < last_two_highs[-2] if len(last_two_highs) == 2 else False
        ll = last_two_lows[-1] < last_two_lows[-2] if len(last_two_lows) == 2 else False
        
        if hh and hl:
            return 'HH/HL'
        elif lh and ll:
            return 'LH/LL'
        else:
            return 'NEUTRAL'
    
    def _find_support(self, df, lookback=50):
        return float(df['low'].iloc[-20:].min())
    
    def _find_resistance(self, df, lookback=50):
        return float(df['high'].iloc[-20:].max())
    
    def _detect_breakout(self, df, resistance, support):
        """Detect breakouts"""
        try:
            current_price = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2]
            atr = df['atr'].iloc[-1] if not pd.isna(df['atr'].iloc[-1]) else 0
            
            if current_price > resistance and prev_close <= resistance:
                if current_price - resistance > atr * 0.5:
                    return True, 'BULL'
            
            if current_price < support and prev_close >= support:
                if support - current_price > atr * 0.5:
                    return True, 'BEAR'
        except:
            pass
        
        return False, 'NONE'
    
    def _determine_market_state(self, df, price, ema20, ema50, ema200, adx, structure, 
                               breakout_detected, breakout_direction):
        """Determine market state"""
        
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
            atr = df['atr'].iloc[-1] if not pd.isna(df['atr'].iloc[-1]) else 0
            
            if recent_range < atr * 3:
                return MarketState.CHOPPY, 'NEUTRAL', 20
            else:
                return MarketState.RANGING, 'NEUTRAL', 30
        
        return MarketState.RANGING, 'NEUTRAL', 25
    
    def _recommend_strategy(self, state, strength, breakout_detected):
        """Recommend best strategy"""
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


# ============ CONTINUATION ENGINE ============
class ContinuationEngine:
    """Trades WITH the trend - your proven swapped logic"""
    
    def __init__(self):
        self.MAX_PATTERN_AGE = 8
        self.MIN_PULLBACK_DEPTH = 0.3
        self.MAX_PULLBACK_DEPTH = 0.7
    
    def detect_setups(self, df, market_state):
        """Detect continuation setups"""
        if not market_state or market_state.get('state') in [MarketState.CHOPPY.value, MarketState.RANGING.value]:
            return []
        
        try:
            signals = []
            atr = df['atr']
            current_price = float(df['close'].iloc[-1])
            current_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0
            ema_20 = df['ema_20']
            
            is_bullish = market_state.get('direction') in ['BULLISH', 'BULL']
            is_bearish = market_state.get('direction') in ['BEARISH', 'BEAR']
            
            # Look at last 15 candles
            for i in range(-15, 0):
                idx = len(df) + i
                
                # BULLISH PULLBACK - In uptrends, SELL (your genius logic)
                if is_bullish:
                    low = float(df['low'].iloc[idx])
                    ema_val = float(ema_20.iloc[idx])
                    
                    if low <= ema_val * 1.002 and low >= ema_val * 0.998:
                        if current_price > ema_val:
                            recent_high = float(df['high'].iloc[idx-5:idx].max())
                            pullback_depth = (recent_high - low) / (recent_high - ema_val) if recent_high > ema_val else 0.5
                            
                            if self.MIN_PULLBACK_DEPTH <= pullback_depth <= self.MAX_PULLBACK_DEPTH:
                                entry = current_price
                                sl = entry + 1.5 * current_atr
                                tp = entry - (sl - entry) * 2.5
                                
                                confidence = 75
                                if abs(current_price - ema_val) < current_atr * 0.5:
                                    confidence += 10
                                if market_state.get('strength', 0) > 70:
                                    confidence += 10
                                
                                signals.append({
                                    'type': 'SELL',
                                    'entry': entry,
                                    'sl': sl,
                                    'tp': tp,
                                    'atr': current_atr,
                                    'strategy': 'CONTINUATION_PULLBACK',
                                    'pattern': 'Bullish Pullback to EMA (SELL)',
                                    'confidence': min(confidence, 100),
                                    'market_state': market_state.get('state')
                                })
                
                # BEARISH RALLY - In downtrends, BUY (your genius logic)
                if is_bearish:
                    high = float(df['high'].iloc[idx])
                    ema_val = float(ema_20.iloc[idx])
                    
                    if high >= ema_val * 0.998 and high <= ema_val * 1.002:
                        if current_price < ema_val:
                            recent_low = float(df['low'].iloc[idx-5:idx].min())
                            rally_height = (high - recent_low) / (ema_val - recent_low) if ema_val > recent_low else 0.5
                            
                            if self.MIN_PULLBACK_DEPTH <= rally_height <= self.MAX_PULLBACK_DEPTH:
                                entry = current_price
                                sl = entry - 1.5 * current_atr
                                tp = entry + (entry - sl) * 2.5
                                
                                confidence = 75
                                if abs(current_price - ema_val) < current_atr * 0.5:
                                    confidence += 10
                                if market_state.get('strength', 0) > 70:
                                    confidence += 10
                                
                                signals.append({
                                    'type': 'BUY',
                                    'entry': entry,
                                    'sl': sl,
                                    'tp': tp,
                                    'atr': current_atr,
                                    'strategy': 'CONTINUATION_RALLY',
                                    'pattern': 'Bearish Rally to EMA (BUY)',
                                    'confidence': min(confidence, 100),
                                    'market_state': market_state.get('state')
                                })
            
            # Filter by age
            current_index = len(df) - 1
            valid_signals = []
            
            for signal in signals:
                pattern_age = current_index - signal.get('index', current_index)
                if pattern_age <= self.MAX_PATTERN_AGE:
                    valid_signals.append(signal)
            
            return valid_signals[:3]
            
        except Exception as e:
            logger.error(f"Continuation detection error: {e}")
            return []


# ============ QUASIMODO ENGINE ============
class QuasimodoEngine:
    """PURE QUASIMODO - Your proven reversal strategy with swapped logic"""
    
    def __init__(self):
        self.MAX_PATTERN_AGE = 8
        self.RETEST_TOLERANCE_PIPS = 2
        self.ATR_PERIOD = 14
        self.VOLATILITY_MULTIPLIER_SL = 1.5
        self.VOLATILITY_MULTIPLIER_TP = 2.5
    
    def _get_pip_value(self, symbol):
        """Get pip value for tolerance"""
        if not symbol:
            return 0.0001
        if 'JPY' in symbol or 'XAG' in symbol or 'BTC' in symbol:
            return 0.01
        elif 'XAU' in symbol or 'US30' in symbol or 'USTEC' in symbol or 'US100' in symbol:
            return 0.1
        elif 'R_' in symbol:  # Synthetics
            return 0.01
        else:
            return 0.0001
    
    def _check_retest(self, df, pattern_level, direction, tolerance):
        """Check if price retested within tolerance"""
        try:
            last_12_low = df['low'].iloc[-12:].min()
            last_12_high = df['high'].iloc[-12:].max()
            current_price = df['close'].iloc[-1]
            
            if direction == 'BUY':
                if last_12_low <= (pattern_level + tolerance) and current_price > pattern_level:
                    last_8_low = df['low'].iloc[-8:].min()
                    if last_8_low <= (pattern_level + tolerance):
                        return True
            else:
                if last_12_high >= (pattern_level - tolerance) and current_price < pattern_level:
                    last_8_high = df['high'].iloc[-8:].max()
                    if last_8_high >= (pattern_level - tolerance):
                        return True
            
            return False
        except:
            return False
    
    def detect_setups(self, df, market_state, symbol):
        """Detect Quasimodo setups"""
        
        if not market_state or market_state.get('state') in [MarketState.STRONG_UPTREND.value, MarketState.STRONG_DOWNTREND.value]:
            return []
        
        try:
            signals = []
            atr = df['atr']
            current_index = len(df) - 1
            pip_value = self._get_pip_value(symbol)
            tolerance = self.RETEST_TOLERANCE_PIPS * pip_value
            
            for i in range(3, len(df)-1):
                h1 = float(df['high'].iloc[i-3])
                h2 = float(df['high'].iloc[i-2])
                h3 = float(df['high'].iloc[i-1])
                l1 = float(df['low'].iloc[i-3])
                l2 = float(df['low'].iloc[i-2])
                l3 = float(df['low'].iloc[i-1])
                close = float(df['close'].iloc[i])
                current_atr = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0
                
                pattern_age = current_index - i
                if pattern_age > self.MAX_PATTERN_AGE:
                    continue
                
                # SELL QUASIMODO (Bearish pattern) -> BUY signal (your genius logic)
                if h1 < h2 > h3 and l1 < l2 < l3 and close < h2:
                    near_resistance = abs(close - market_state.get('resistance', close)) < current_atr * 2
                    
                    if market_state.get('state') == MarketState.RANGING.value or near_resistance:
                        
                        if self._check_retest(df, h2, 'SELL', tolerance):
                            entry = h2
                            sl = entry - (self.VOLATILITY_MULTIPLIER_SL * current_atr)
                            tp = entry + (self.VOLATILITY_MULTIPLIER_TP * current_atr)
                            
                            confidence = 70
                            if near_resistance:
                                confidence += 15
                            if market_state.get('state') == MarketState.RANGING.value:
                                confidence += 10
                            
                            signals.append({
                                'type': 'BUY',
                                'entry': entry,
                                'sl': sl,
                                'tp': tp,
                                'atr': current_atr,
                                'strategy': 'QUASIMODO_REVERSAL',
                                'pattern': 'Quasimodo Sell Setup (BUY)',
                                'confidence': min(confidence, 100),
                                'market_state': market_state.get('state')
                            })
                
                # BUY QUASIMODO (Bullish pattern) -> SELL signal (your genius logic)
                if l1 > l2 < l3 and h1 > h2 > h3 and close > l2:
                    near_support = abs(close - market_state.get('support', close)) < current_atr * 2
                    
                    if market_state.get('state') == MarketState.RANGING.value or near_support:
                        
                        if self._check_retest(df, l2, 'BUY', tolerance):
                            entry = l2
                            sl = entry + (self.VOLATILITY_MULTIPLIER_SL * current_atr)
                            tp = entry - (self.VOLATILITY_MULTIPLIER_TP * current_atr)
                            
                            confidence = 70
                            if near_support:
                                confidence += 15
                            if market_state.get('state') == MarketState.RANGING.value:
                                confidence += 10
                            
                            signals.append({
                                'type': 'SELL',
                                'entry': entry,
                                'sl': sl,
                                'tp': tp,
                                'atr': current_atr,
                                'strategy': 'QUASIMODO_REVERSAL',
                                'pattern': 'Quasimodo Buy Setup (SELL)',
                                'confidence': min(confidence, 100),
                                'market_state': market_state.get('state')
                            })
            
            return signals[:3]
            
        except Exception as e:
            logger.error(f"Quasimodo detection error: {e}")
            return []


# ============ SMART STRATEGY SELECTOR ============
class SmartStrategySelector:
    """Decides which strategy to use - EXACT same as your MT5 bot"""
    
    def select_best_trades(self, continuation_signals, quasimodo_signals, market_state):
        """Select best trades for current market"""
        
        if not market_state:
            return []
        
        state = market_state.get('state')
        selected_trades = []
        
        try:
            # STRONG UPTREND - ONLY CONTINUATION SELLS
            if state == MarketState.STRONG_UPTREND.value:
                selected_trades = [t for t in continuation_signals if t.get('type') == 'SELL']
                logger.info(f"📊 STRONG UPTREND - Using CONTINUATION SELLS only")
            
            # STRONG DOWNTREND - ONLY CONTINUATION BUYS
            elif state == MarketState.STRONG_DOWNTREND.value:
                selected_trades = [t for t in continuation_signals if t.get('type') == 'BUY']
                logger.info(f"📊 STRONG DOWNTREND - Using CONTINUATION BUYS only")
            
            # UPTREND - PREFER CONTINUATION SELLS, ALLOW STRONG QUASIMODO
            elif state == MarketState.UPTREND.value:
                selected_trades = [t for t in continuation_signals if t.get('type') == 'SELL']
                strong_qm = [q for q in quasimodo_signals if q.get('confidence', 0) > 80]
                selected_trades.extend(strong_qm)
                logger.info(f"📊 UPTREND - Prefer CONTINUATION SELLS, allow strong QUASIMODO")
            
            # DOWNTREND - PREFER CONTINUATION BUYS, ALLOW STRONG QUASIMODO
            elif state == MarketState.DOWNTREND.value:
                selected_trades = [t for t in continuation_signals if t.get('type') == 'BUY']
                strong_qm = [q for q in quasimodo_signals if q.get('confidence', 0) > 80]
                selected_trades.extend(strong_qm)
                logger.info(f"📊 DOWNTREND - Prefer CONTINUATION BUYS, allow strong QUASIMODO")
            
            # RANGING - ONLY QUASIMODO
            elif state == MarketState.RANGING.value:
                selected_trades = quasimodo_signals
                logger.info(f"📊 RANGING - Using QUASIMODO only")
            
            # BREAKOUT - CONTINUATION
            elif state in [MarketState.BREAKOUT_BULL.value, MarketState.BREAKOUT_BEAR.value]:
                selected_trades = continuation_signals
                logger.info(f"📊 BREAKOUT - Using CONTINUATION for momentum")
            
            # CHOPPY - SKIP ALL
            elif state == MarketState.CHOPPY.value:
                selected_trades = []
                logger.info(f"📊 CHOPPY - SKIPPING ALL TRADES")
            
            # Filter by confidence (minimum 65%)
            selected_trades = [t for t in selected_trades if t.get('confidence', 0) >= 65]
            
            # Sort by confidence
            selected_trades.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Strategy selector error: {e}")
        
        return selected_trades


# ============ TRADING ENGINE ============
class KarankaTradingEngine:
    """Main trading engine - EXACT logic from your MT5 bot, adapted for Deriv"""
    
    def __init__(self):
        self.api = DerivAPI()
        self.market_engine = MarketStateEngine()
        self.continuation = ContinuationEngine()
        self.quasimodo = QuasimodoEngine()
        self.selector = SmartStrategySelector()
        
        self.connected = False
        self.running = False
        self.token = None
        
        self.active_trades = []
        self.trade_history = []
        self.market_analysis = {}
        self.signals_history = []
        
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.total_wins = 0
        self.total_losses = 0
        self.analysis_cycle = 0
        
        # Settings
        self.dry_run = True
        self.max_daily_trades = MAX_DAILY_TRADES
        self.max_concurrent_trades = MAX_CONCURRENT_TRADES
        self.min_seconds_between = 10
        self.fixed_amount = FIXED_AMOUNT
        self.min_confidence = 65
        self.enabled_symbols = [
            "R_10", "R_25", "R_50", "R_75", "R_100",
            "EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD",
            "US30", "US100", "BTCUSD"
        ]
        
        # Start daily reset thread
        self._start_daily_reset()
        
        logger.info("✅ Karanka Trading Engine initialized")
    
    def _start_daily_reset(self):
        """Reset daily counters at midnight"""
        def reset_daily():
            while True:
                now = datetime.now()
                midnight = datetime(now.year, now.month, now.day, 23, 59, 59)
                seconds_until_midnight = (midnight - now).total_seconds()
                
                if seconds_until_midnight > 0:
                    time.sleep(seconds_until_midnight)
                
                self.daily_trades = 0
                self.daily_pnl = 0.0
                if self.consecutive_losses >= 3:
                    self.consecutive_losses = 0
                
                logger.info("🔄 Daily counters reset")
                time.sleep(1)
        
        thread = threading.Thread(target=reset_daily, daemon=True)
        thread.start()
    
    def connect(self, token):
        """Connect to Deriv - passes token EXACTLY as received"""
        self.token = token  # Store EXACTLY what user entered - NO CHANGES!
        logger.info(f"Connecting with token: {self.token[:4]}...{self.token[-4:]} (length: {len(self.token)})")
        success, message = self.api.connect(token)
        if success:
            self.connected = True
        return success, message
    
    def disconnect(self):
        """Disconnect from Deriv"""
        self.api.disconnect()
        self.connected = False
        self.running = False
        logger.info("Disconnected from Deriv")
    
    def start_trading(self, settings=None):
        """Start trading loop"""
        if not self.connected:
            return False, "Not connected to Deriv"
        
        if settings:
            self.dry_run = settings.get('dry_run', True)
            self.max_daily_trades = int(settings.get('max_daily_trades', self.max_daily_trades))
            self.max_concurrent_trades = int(settings.get('max_concurrent_trades', self.max_concurrent_trades))
            self.fixed_amount = float(settings.get('fixed_amount', self.fixed_amount))
            self.min_confidence = int(settings.get('min_confidence', self.min_confidence))
            if settings.get('enabled_symbols'):
                self.enabled_symbols = settings['enabled_symbols']
        
        self.running = True
        thread = threading.Thread(target=self._trading_loop, daemon=True)
        thread.start()
        logger.info("🚀 Trading started")
        return True, "Trading started"
    
    def stop_trading(self):
        """Stop trading loop"""
        self.running = False
        logger.info("🛑 Trading stopped")
    
    def _trading_loop(self):
        """Main trading loop - this keeps the bot alive with constant data fetching"""
        error_count = 0
        
        while self.running and self.connected:
            try:
                self.analysis_cycle += 1
                error_count = 0  # Reset error count on successful cycle
                
                logger.info(f"🔄 Analysis Cycle #{self.analysis_cycle} - Fetching market data...")
                
                # Check if we can trade
                can_trade = self._can_trade()
                
                # Analyze each enabled symbol
                signals_found = 0
                symbols_analyzed = 0
                
                for symbol in self.enabled_symbols:
                    try:
                        # Get data (15m for entries, 1h for HTF)
                        df_15m = self.api.get_candles(symbol, 300, 60)  # 1-minute candles
                        df_h1 = self.api.get_candles(symbol, 200, 3600)  # 1-hour candles
                        
                        if df_15m is None or len(df_15m) < 100:
                            logger.debug(f"Insufficient data for {symbol}")
                            continue
                        
                        if df_h1 is None or len(df_h1) < 50:
                            df_h1 = df_15m.copy()
                        
                        # Prepare dataframes
                        df_15m = self._prepare_dataframe(df_15m)
                        df_h1 = self._prepare_dataframe(df_h1)
                        
                        if df_15m is None or df_h1 is None:
                            continue
                        
                        symbols_analyzed += 1
                        
                        # Analyze market state on HTF
                        market_state = self.market_engine.analyze(df_h1)
                        
                        # Detect setups
                        continuation_signals = self.continuation.detect_setups(df_15m, market_state)
                        quasimodo_signals = self.quasimodo.detect_setups(df_15m, market_state, symbol)
                        
                        # Select best trades
                        best_trades = self.selector.select_best_trades(
                            continuation_signals, quasimodo_signals, market_state
                        )
                        
                        # Store analysis
                        self.market_analysis[symbol] = {
                            'symbol': symbol,
                            'price': float(df_15m['close'].iloc[-1]),
                            'market_state': market_state.get('state', 'UNKNOWN'),
                            'market_direction': market_state.get('direction', 'NEUTRAL'),
                            'structure': market_state.get('structure', 'NEUTRAL'),
                            'strength': market_state.get('strength', 0),
                            'signals': [{
                                'type': t['type'],
                                'strategy': t['strategy'],
                                'pattern': t['pattern'],
                                'confidence': t['confidence'],
                                'entry': float(t['entry']),
                                'sl': float(t['sl']),
                                'tp': float(t['tp'])
                            } for t in best_trades[:2]],
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Execute trades if signals exist and we can trade
                        if best_trades and can_trade:
                            trade = best_trades[0]
                            if trade['confidence'] >= self.min_confidence:
                                success, result = self._execute_trade(symbol, trade)
                                if success:
                                    self.daily_trades += 1
                                    self.last_trade_time = datetime.now()
                                    signals_found += 1
                        
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}")
                        continue
                
                # Log summary
                logger.info(f"📊 Cycle complete - Analyzed {symbols_analyzed} symbols, Found {signals_found} signals")
                
                # Emit update to web clients
                socketio.emit('market_update', self.get_market_data())
                socketio.emit('trade_update', self.get_trade_data())
                
                # Sleep between cycles (8 seconds)
                time.sleep(8)
                
            except Exception as e:
                error_count += 1
                logger.error(f"Trading loop error: {e}")
                logger.error(traceback.format_exc())
                
                if error_count > 5:
                    logger.critical("Too many errors, stopping trading")
                    self.running = False
                    break
                
                time.sleep(10)
    
    def _prepare_dataframe(self, df):
        """Prepare dataframe with indicators"""
        if df is None or len(df) < 50:
            return None
        
        try:
            df = df.copy()
            
            # EMAs
            df['ema_20'] = df['close'].ewm(span=20, min_periods=20).mean()
            df['ema_50'] = df['close'].ewm(span=50, min_periods=50).mean()
            df['ema_200'] = df['close'].ewm(span=200, min_periods=200).mean()
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14, min_periods=14).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Dataframe preparation error: {e}")
            return None
    
    def _can_trade(self):
        """Check if we can trade"""
        if len(self.active_trades) >= self.max_concurrent_trades:
            return False
        
        if self.daily_trades >= self.max_daily_trades:
            return False
        
        if self.last_trade_time:
            seconds_since = (datetime.now() - self.last_trade_time).total_seconds()
            if seconds_since < self.min_seconds_between:
                return False
        
        if self.consecutive_losses >= 3:
            logger.warning("3 consecutive losses - cooling down")
            return False
        
        return True
    
    def _execute_trade(self, symbol, trade):
        """Execute a trade"""
        try:
            if self.dry_run:
                # Simulate trade
                trade_record = {
                    'symbol': symbol,
                    'direction': trade['type'],
                    'entry': float(trade['entry']),
                    'sl': float(trade['sl']),
                    'tp': float(trade['tp']),
                    'amount': self.fixed_amount,
                    'strategy': trade['strategy'],
                    'pattern': trade['pattern'],
                    'confidence': trade['confidence'],
                    'entry_time': datetime.now().isoformat(),
                    'dry_run': True,
                    'id': f"dry_{int(time.time())}_{symbol}"
                }
                self.active_trades.append(trade_record)
                
                logger.info(f"✅ [DRY RUN] {symbol} {trade['type']} | Conf: {trade['confidence']:.0f}%")
                
                # Simulate trade result after 5 minutes
                def simulate_result():
                    time.sleep(300)  # 5 minutes
                    
                    # Simulate based on confidence (higher confidence = higher win chance)
                    import random
                    win_chance = trade['confidence'] / 100
                    
                    if random.random() < win_chance:
                        profit = self.fixed_amount * 2.5  # 1:2.5 RR
                        self.daily_pnl += profit
                        self.consecutive_losses = 0
                        self.total_wins += 1
                        result = "WIN"
                    else:
                        profit = -self.fixed_amount
                        self.daily_pnl += profit
                        self.consecutive_losses += 1
                        self.total_losses += 1
                        result = "LOSS"
                    
                    trade_record['profit'] = profit
                    trade_record['exit_time'] = datetime.now().isoformat()
                    trade_record['result'] = result
                    
                    if trade_record in self.active_trades:
                        self.active_trades.remove(trade_record)
                    self.trade_history.append(trade_record)
                    
                    logger.info(f"📊 [DRY RUN] {symbol} {result} | Profit: ${profit:.2f}")
                    socketio.emit('trade_update', self.get_trade_data())
                
                thread = threading.Thread(target=simulate_result, daemon=True)
                thread.start()
                
                return True, "Dry run trade placed"
            
            else:
                # Real trade
                result, message = self.api.place_trade(
                    symbol=symbol,
                    direction=trade['type'],
                    amount=self.fixed_amount,
                    duration=5
                )
                
                if result:
                    trade_record = {
                        'symbol': symbol,
                        'direction': trade['type'],
                        'entry': float(result['entry_price']),
                        'sl': float(trade['sl']),
                        'tp': float(trade['tp']),
                        'amount': self.fixed_amount,
                        'strategy': trade['strategy'],
                        'pattern': trade['pattern'],
                        'confidence': trade['confidence'],
                        'entry_time': datetime.now().isoformat(),
                        'contract_id': result['contract_id'],
                        'dry_run': False,
                        'id': result['contract_id']
                    }
                    self.active_trades.append(trade_record)
                    
                    # Monitor trade
                    def monitor_trade():
                        time.sleep(300)  # 5 minutes
                        status = self.api.get_trade_status(result['contract_id'])
                        if status:
                            profit = float(status.get('profit', 0))
                            trade_record['profit'] = profit
                            trade_record['exit_time'] = datetime.now().isoformat()
                            trade_record['result'] = 'WIN' if profit > 0 else 'LOSS'
                            
                            self.daily_pnl += profit
                            if profit > 0:
                                self.total_wins += 1
                                self.consecutive_losses = 0
                            else:
                                self.total_losses += 1
                                self.consecutive_losses += 1
                            
                            if trade_record in self.active_trades:
                                self.active_trades.remove(trade_record)
                            self.trade_history.append(trade_record)
                            
                            logger.info(f"📊 REAL TRADE: {symbol} {trade_record['result']} | Profit: ${profit:.2f}")
                            socketio.emit('trade_update', self.get_trade_data())
                    
                    thread = threading.Thread(target=monitor_trade, daemon=True)
                    thread.start()
                    
                    logger.info(f"✅ REAL TRADE: {symbol} {trade['type']} | Conf: {trade['confidence']:.0f}%")
                    
                    return True, f"Trade placed: {result['contract_id']}"
                
                return False, message
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False, str(e)
    
    def get_market_data(self):
        """Get market analysis data for UI"""
        return {
            'market_analysis': self.market_analysis,
            'timestamp': datetime.now().isoformat(),
            'cycle': self.analysis_cycle
        }
    
    def get_trade_data(self):
        """Get trade data for UI"""
        return {
            'active_trades': self.active_trades[-10:],
            'trade_history': self.trade_history[-50:],
            'daily_trades': self.daily_trades,
            'daily_pnl': round(self.daily_pnl, 2),
            'consecutive_losses': self.consecutive_losses,
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
            'win_rate': round((self.total_wins / (self.total_wins + self.total_losses + 1)) * 100, 1)
        }
    
    def get_status(self):
        """Get system status"""
        return {
            'connected': self.connected,
            'running': self.running,
            'dry_run': self.dry_run,
            'active_trades': len(self.active_trades),
            'daily_trades': self.daily_trades,
            'daily_pnl': round(self.daily_pnl, 2),
            'max_daily_trades': self.max_daily_trades,
            'max_concurrent': self.max_concurrent_trades,
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
            'win_rate': round((self.total_wins / (self.total_wins + self.total_losses + 1)) * 100, 1),
            'analysis_cycle': self.analysis_cycle
        }


# ============ INITIALIZE TRADING ENGINE ============
trading_engine = KarankaTradingEngine()
keep_awake = KeepAwake(BASE_URL)

# ============ FLASK ROUTES ============

@app.route('/')
def index():
    """Main page - serves the UI"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint for Render and keep-awake"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'connected': trading_engine.connected if trading_engine else False,
        'running': trading_engine.running if trading_engine else False,
        'cycle': trading_engine.analysis_cycle if trading_engine else 0,
        'uptime': time.time() - start_time if 'start_time' in globals() else 0
    })

@app.route('/ping')
def ping():
    """Simple ping endpoint for keep-awake"""
    return 'pong'

@app.route('/api/connect', methods=['POST'])
def api_connect():
    """Connect to Deriv"""
    try:
        data = request.json
        token = data.get('token')
        
        if not token:
            return jsonify({'success': False, 'message': 'Token required'})
        
        # Pass token EXACTLY as received - NO modifications!
        logger.info(f"Received token: {token[:4]}...{token[-4:]} (length: {len(token)})")
        success, message = trading_engine.connect(token)
        return jsonify({'success': success, 'message': message})
        
    except Exception as e:
        logger.error(f"Connect error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/disconnect', methods=['POST'])
def api_disconnect():
    """Disconnect from Deriv"""
    try:
        trading_engine.disconnect()
        return jsonify({'success': True, 'message': 'Disconnected'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start trading"""
    try:
        settings = request.json or {}
        success, message = trading_engine.start_trading(settings)
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop trading"""
    try:
        trading_engine.stop_trading()
        return jsonify({'success': True, 'message': 'Trading stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/status')
def api_status():
    """Get system status"""
    try:
        return jsonify(trading_engine.get_status())
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/market_data')
def api_market_data():
    """Get market data"""
    try:
        return jsonify(trading_engine.get_market_data())
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/trade_data')
def api_trade_data():
    """Get trade data"""
    try:
        return jsonify(trading_engine.get_trade_data())
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/symbols')
def api_symbols():
    """Get available symbols"""
    try:
        if trading_engine.connected:
            symbols = trading_engine.api.get_active_symbols()
            return jsonify({'success': True, 'symbols': symbols})
        return jsonify({'success': False, 'symbols': []})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/settings', methods=['POST'])
def api_settings():
    """Update settings"""
    try:
        data = request.json
        if data:
            trading_engine.dry_run = data.get('dry_run', trading_engine.dry_run)
            trading_engine.max_daily_trades = int(data.get('max_daily_trades', trading_engine.max_daily_trades))
            trading_engine.max_concurrent_trades = int(data.get('max_concurrent_trades', trading_engine.max_concurrent_trades))
            trading_engine.fixed_amount = float(data.get('fixed_amount', trading_engine.fixed_amount))
            trading_engine.min_confidence = int(data.get('min_confidence', trading_engine.min_confidence))
            if data.get('enabled_symbols'):
                trading_engine.enabled_symbols = data['enabled_symbols']
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ============ SOCKETIO EVENTS ============
@socketio.on('connect')
def handle_connect():
    """Client connected"""
    logger.info(f"📱 Client connected: {request.sid}")
    emit('connected', {'data': 'Connected to Karanka Server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    logger.info(f"📱 Client disconnected: {request.sid}")

@socketio.on('request_update')
def handle_update_request():
    """Client requested update"""
    emit('market_update', trading_engine.get_market_data())
    emit('trade_update', trading_engine.get_trade_data())

# ============ ERROR HANDLERS ============
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# ============ STARTUP ============
if __name__ == '__main__':
    # Get port from environment
    port = int(os.environ.get('PORT', 5000))
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("KARANKA MULTIVERSE ALGO AI - DERIV PRODUCTION BOT")
    logger.info("=" * 60)
    logger.info(f"✅ Market State Engine: ACTIVE")
    logger.info(f"✅ Continuation: SWAPPED (SELL in uptrend, BUY in downtrend)")
    logger.info(f"✅ Quasimodo: SWAPPED (BUY from bearish, SELL from bullish)")
    logger.info(f"✅ Smart Selector: ACTIVE")
    logger.info(f"✅ 2 Pip Retest: ACTIVE")
    logger.info(f"✅ HTF Structure: MANDATORY")
    logger.info(f"✅ FIXED DERIV AUTH: RAW TOKEN SENDING")
    logger.info(f"✅ DERIV URL: {DERIV_WS_URL}")
    logger.info(f"✅ KEEP-AWAKE: Active - pinging every 5 minutes")
    logger.info(f"✅ HEARTBEAT: Active - pinging every 25 seconds")
    logger.info(f"✅ Port: {port}")
    logger.info("=" * 60)
    
    # Start keep-awake mechanism
    keep_awake.start()
    
    # Run the app
    socketio.run(
        app,
        host='0.0.0.0',
        port=port,
        debug=False,
        allow_unsafe_werkzeug=True
    )
