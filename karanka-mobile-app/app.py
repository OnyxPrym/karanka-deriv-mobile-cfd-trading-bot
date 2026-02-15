#!/usr/bin/env python3
"""
================================================================================
KARANKA MULTIVERSE ALGO AI - DERIV PRODUCTION BOT
================================================================================
✅ FIXED DERIV CONNECTION - PROPER JSON AUTHORIZATION
✅ SHOWS ACCOUNT BALANCE
✅ TAKES RAW API TOKEN EXACTLY AS USER INPUTS
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
import math

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


# ============ DERIV API CONNECTOR - FIXED VERSION WITH PROPER AUTH ============
class DerivAPI:
    """Production-ready Deriv WebSocket API with proper JSON authorization"""
    
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
        self.trade_callbacks = []
        self.balance = 0
        self.currency = "USD"
        self.loginid = ""
        self.account_type = ""
        
    def _next_id(self):
        self.req_id += 1
        return self.req_id
    
    def register_trade_callback(self, callback):
        """Register a callback function to be called when trades close"""
        self.trade_callbacks.append(callback)
    
    def connect(self, token):
        """Connect to Deriv using proper JSON authorization"""
        self.token = token.strip()  # Store EXACTLY what user entered, just trim whitespace
        logger.info(f"Connecting with token: {self.token[:4]}...{self.token[-4:]} (length: {len(self.token)})")
        logger.info(f"Token format: {self.token}")
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
                    timeout=30,
                    enable_multithread=True
                )
                
                # Send authorization request in CORRECT JSON format
                auth_request = {
                    "authorize": self.token,  # Send the raw token in JSON
                    "req_id": self._next_id()
                }
                
                logger.info(f"Sending authorization request (req_id: {auth_request['req_id']})")
                self.ws.send(json.dumps(auth_request))
                
                # Wait for response with timeout
                self.ws.settimeout(15)
                response = self.ws.recv()
                logger.info(f"Response received: {response[:200]}")
                
                # Parse response
                response_data = json.loads(response)
                
                # Check for successful authorization
                if 'error' in response_data:
                    error_code = response_data['error'].get('code', 'Unknown')
                    error_msg = response_data['error'].get('message', 'Unknown error')
                    logger.error(f"Auth failed: Code: {error_code}, Message: {error_msg}")
                    
                    if attempt < max_retries - 1:
                        wait_time = min(30, 2 ** attempt)
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    return False, f"Authentication failed: {error_msg} (Code: {error_code})"
                
                elif 'authorize' in response_data:
                    self.connected = True
                    auth_data = response_data['authorize']
                    self.loginid = auth_data.get('loginid', 'Unknown')
                    self.currency = auth_data.get('currency', 'USD')
                    self.balance = float(auth_data.get('balance', 0))
                    self.account_type = auth_data.get('account_type', 'Unknown')
                    self.email = auth_data.get('email', 'Unknown')
                    
                    logger.info(f"✅ Connected to Deriv successfully!")
                    logger.info(f"👤 Account: {self.loginid} ({self.account_type})")
                    logger.info(f"💰 Balance: {self.balance} {self.currency}")
                    logger.info(f"📧 Email: {self.email}")
                    
                    # Start heartbeat to keep connection alive
                    self._start_heartbeat()
                    self._start_message_listener()
                    
                    # Reset connection attempts on success
                    self.connection_attempts = 0
                    self.reconnect_delay = 1
                    
                    return True, f"Connected as {self.loginid} | Balance: {self.balance} {self.currency}"
                
                else:
                    logger.error(f"Unexpected response format: {response_data.keys()}")
                    return False, "Unexpected response format from server"
                    
            except websocket.WebSocketTimeoutException:
                logger.error("Connection timeout - server not responding")
                if attempt < max_retries - 1:
                    wait_time = min(30, 2 ** attempt)
                    time.sleep(wait_time)
                else:
                    return False, "Connection timeout - server not responding"
                    
            except websocket.WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
                if attempt < max_retries - 1:
                    wait_time = min(30, 2 ** attempt)
                    time.sleep(wait_time)
                else:
                    return False, f"WebSocket error: {str(e)}"
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse server response: {e}")
                if attempt < max_retries - 1:
                    wait_time = min(30, 2 ** attempt)
                    time.sleep(wait_time)
                else:
                    return False, "Server returned invalid JSON"
                    
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                logger.error(traceback.format_exc())
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
                        
                        # Handle different message types
                        if 'pong' in data:
                            self.last_pong = time.time()
                            logger.debug("Pong received")
                            
                        elif 'error' in data:
                            logger.error(f"API Error: {data['error']}")
                            
                        elif 'balance' in data:
                            # Balance update
                            self.balance = float(data['balance']['balance'])
                            logger.info(f"💰 Balance updated: {self.balance} {self.currency}")
                            
                        elif 'proposal_open_contract' in data:
                            # Handle trade updates
                            contract = data['proposal_open_contract']
                            contract_id = contract.get('contract_id')
                            
                            if contract.get('is_sold', False):
                                logger.info(f"📊 Contract {contract_id} closed")
                                # Notify all registered callbacks
                                for callback in self.trade_callbacks:
                                    try:
                                        callback(contract_id, contract)
                                    except Exception as e:
                                        logger.error(f"Trade callback error: {e}")
                                        
                        elif 'buy' in data:
                            # New trade placed
                            logger.info(f"✅ Trade confirmed: {data['buy']}")
                            
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
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    continue
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
    
    def get_balance(self):
        """Get current account balance"""
        return {
            'balance': self.balance,
            'currency': self.currency,
            'loginid': self.loginid,
            'account_type': self.account_type
        }
    
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
            longcode = response_data['buy'].get('longcode', '')
            
            # Store active contract
            self.active_contracts[contract_id] = {
                'symbol': symbol,
                'direction': direction,
                'amount': amount,
                'entry_time': time.time(),
                'contract_id': contract_id,
                'entry_price': entry_price,
                'longcode': longcode
            }
            
            logger.info(f"✅ Trade placed: {symbol} {direction} ${amount} (ID: {contract_id})")
            logger.info(f"📝 {longcode}")
            
            return {
                'contract_id': contract_id,
                'entry_price': entry_price,
                'direction': direction,
                'amount': amount,
                'symbol': symbol,
                'longcode': longcode
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
                'exit_time': contract.get('exit_time', 0),
                'current_spot': float(contract.get('current_spot', 0)) if contract.get('current_spot') else None
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


# [REST OF THE CODE REMAINS THE SAME - Market State Engine, Continuation Engine, 
#  Quasimodo Engine, Smart Strategy Selector, and Trading Engine]

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
        'active_trades': len(trading_engine.active_trades) if trading_engine else 0,
        'uptime': time.time() - start_time if 'start_time' in globals() else 0,
        'balance': trading_engine.api.balance if trading_engine and trading_engine.connected else 0,
        'currency': trading_engine.api.currency if trading_engine and trading_engine.connected else 'USD'
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
        logger.info(f"Full token format: {token}")
        success, message = trading_engine.connect(token)
        
        if success:
            # Get balance info
            balance_info = trading_engine.api.get_balance()
            return jsonify({
                'success': True, 
                'message': message,
                'balance': balance_info
            })
        else:
            return jsonify({'success': False, 'message': message})
        
    except Exception as e:
        logger.error(f"Connect error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/balance')
def api_balance():
    """Get current balance"""
    try:
        if trading_engine.connected:
            balance_info = trading_engine.api.get_balance()
            return jsonify({'success': True, 'balance': balance_info})
        return jsonify({'success': False, 'message': 'Not connected'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# [REST OF THE API ROUTES REMAIN THE SAME]

# ============ STARTUP ============
if __name__ == '__main__':
    # Get port from environment
    port = int(os.environ.get('PORT', 5000))
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("KARANKA MULTIVERSE ALGO AI - DERIV PRODUCTION BOT")
    logger.info("=" * 60)
    logger.info(f"✅ FIXED DERIV CONNECTION - PROPER JSON AUTH")
    logger.info(f"✅ SHOWS ACCOUNT BALANCE")
    logger.info(f"✅ TAKES RAW API TOKEN EXACTLY AS USER INPUTS")
    logger.info(f"✅ Market State Engine: ACTIVE")
    logger.info(f"✅ Continuation: SWAPPED (SELL in uptrend, BUY in downtrend)")
    logger.info(f"✅ Quasimodo: SWAPPED (BUY from bearish, SELL from bullish)")
    logger.info(f"✅ Smart Selector: ACTIVE")
    logger.info(f"✅ 2 Pip Retest: ACTIVE")
    logger.info(f"✅ HTF Structure: MANDATORY")
    logger.info(f"✅ TRADE TRACKING: ACTIVE (knows when trades close)")
    logger.info(f"✅ TRAILING STOP: ACTIVE (30% activation, 85% profit lock)")
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
