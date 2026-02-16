#!/usr/bin/env python3
"""
================================================================================
KARANKA MULTIVERSE ALGO AI - DERIV PRODUCTION BOT
================================================================================
✅ FULLY WORKING DERIV CONNECTION
✅ SUPPORTS DEMO & REAL ACCOUNTS
✅ USER SELECTS ACCOUNT
✅ NON-BLOCKING WEBSOCKET
✅ PING/PONG KEEP-ALIVE
✅ YOUR EXACT STRATEGY
✅ YOUR UI
✅ PRODUCTION READY
================================================================================
"""

import os
import json
import time
import threading
import logging
import queue
import websocket
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from enum import Enum
import traceback
import random
import sys

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
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('karanka_bot.log')
    ]
)
logger = logging.getLogger(__name__)

# ============ CONFIGURATION ============
DERIV_APP_ID = '1089'
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
    def __init__(self, url):
        self.url = url
        self.running = True
        self.ping_count = 0
        
    def start(self):
        def ping_loop():
            while self.running:
                try:
                    self.ping_count += 1
                    response = requests.get(f"{self.url}/health", timeout=10)
                    logger.info(f"🏓 Keep-awake ping #{self.ping_count}")
                except Exception as e:
                    logger.error(f"Keep-awake ping failed: {e}")
                time.sleep(300)
        
        thread = threading.Thread(target=ping_loop, daemon=True)
        thread.start()
        logger.info("✅ Keep-awake mechanism started")
    
    def stop(self):
        self.running = False

# ============ DERIV CONNECTION - NON-BLOCKING WITH THREAD SAFETY ============
class DerivConnection:
    """Deriv API WebSocket connection - Non-blocking with proper thread handling"""
    
    def __init__(self, app_id="1089"):
        self.app_id = app_id
        self.api_token = None
        self.account_type = "demo"
        
        # Connection state
        self.ws = None
        self.is_connected = False
        self.is_authorized = False
        self.should_run = True
        
        # Thread safety
        self.ws_thread = None
        self.message_queue = queue.Queue()
        self.lock = threading.Lock()
        
        # Account info
        self.balance = 0
        self.currency = "USD"
        self.login_id = ""
        self.account_list = []
        
        # Data storage
        self.candle_cache = {}
        self.candle_cache_lock = threading.Lock()
        
        self.active_contracts = {}
        self.contracts_lock = threading.Lock()
        
        self.trade_callbacks = []
        self.message_handlers = {}
        
        # Ping/Pong for keep-alive
        self.last_pong = time.time()
        self.ping_interval = 25
        
    def register_trade_callback(self, callback):
        with self.lock:
            self.trade_callbacks.append(callback)
    
    def connect(self, token, account_type="demo"):
        """Connect to Deriv WebSocket API with account type selection"""
        self.api_token = token.strip()
        self.account_type = account_type
        ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={self.app_id}"
        
        logger.info("=" * 60)
        logger.info(f"🔌 Connecting to Deriv...")
        logger.info(f"Token: {self.api_token[:4]}...{self.api_token[-4:]}")
        logger.info(f"Requested Account: {account_type}")
        logger.info(f"URL: {ws_url}")
        logger.info("=" * 60)
        
        def run_websocket():
            """Run WebSocket in separate thread - PREVENTS BLOCKING"""
            while self.should_run:
                try:
                    self.ws = websocket.WebSocketApp(
                        ws_url,
                        on_open=self._on_open,
                        on_message=self._on_message,
                        on_error=self._on_error,
                        on_close=self._on_close
                    )
                    
                    # Run with ping/pong to keep alive
                    self.ws.run_forever(ping_interval=self.ping_interval, ping_timeout=10)
                    
                except Exception as e:
                    logger.error(f"WebSocket thread error: {e}")
                
                if self.should_run:
                    logger.info("🔄 Reconnecting in 5 seconds...")
                    time.sleep(5)
        
        # Start WebSocket in background thread
        self.ws_thread = threading.Thread(target=run_websocket, daemon=True)
        self.ws_thread.start()
        
        # Wait for authorization (max 15 seconds)
        for i in range(30):
            if self.is_authorized:
                logger.info(f"✅ Authorization successful after {i*0.5:.1f} seconds")
                
                if len(self.account_list) > 1:
                    logger.info(f"🔄 Found {len(self.account_list)} accounts, switching to {account_type}...")
                    self.switch_account(account_type)
                
                time.sleep(1)
                
                return True, f"Connected to {account_type} account | Balance: {self.balance} {self.currency}"
            time.sleep(0.5)
        
        logger.error("❌ Authorization timeout")
        return False, "Authorization timeout - check your token"
    
    def _on_open(self, ws):
        """Called when WebSocket opens"""
        with self.lock:
            self.is_connected = True
        logger.info("✅ WebSocket connected to Deriv")
        
        # Send authorization - THIS VALIDATES THE TOKEN
        auth_msg = {"authorize": self.api_token}
        logger.info(f"📤 Sending authorization request")
        ws.send(json.dumps(auth_msg))
        
        # Start ping thread
        self._start_ping_thread()
    
    def _on_message(self, ws, message):
        """Handle incoming messages"""
        try:
            data = json.loads(message)
            
            if 'pong' in data:
                with self.lock:
                    self.last_pong = time.time()
                logger.debug("📥 Pong received")
                return
            
            self.handle_message(data)
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"❌ WebSocket error: {error}")
    
    def _on_close(self, ws, code, msg):
        """Handle WebSocket close"""
        with self.lock:
            self.is_connected = False
            self.is_authorized = False
        logger.warning(f"⚠️ WebSocket closed: {code} - {msg}")
    
    def _start_ping_thread(self):
        """Start a thread to send periodic pings"""
        def ping_loop():
            while self.is_connected and self.should_run:
                try:
                    time.sleep(self.ping_interval)
                    if self.ws and self.is_connected:
                        self.ws.send(json.dumps({"ping": 1}))
                        logger.debug("📤 Ping sent")
                except Exception as e:
                    logger.error(f"Ping error: {e}")
                    break
        
        thread = threading.Thread(target=ping_loop, daemon=True)
        thread.start()
    
    def handle_message(self, data):
        """Handle incoming messages from Deriv"""
        
        # Handle errors
        if 'error' in data:
            error_msg = data['error'].get('message', 'Unknown error')
            error_code = data['error'].get('code', 'Unknown')
            logger.error(f"❌ API Error [{error_code}]: {error_msg}")
            
            if error_code == 'AuthorizationRequired':
                logger.error("Token expired - please reconnect")
            elif error_code == 'InvalidToken':
                logger.error("Token is invalid - check your token")
            elif error_code == 'InsufficientPermissions':
                logger.error("Token needs 'Trade' permission")
            return
        
        # AUTHORIZATION RESPONSE - TOKEN IS VALIDATED HERE
        if 'authorize' in data:
            with self.lock:
                self.is_authorized = True
                auth = data['authorize']
                
                self.login_id = auth['loginid']
                self.balance = float(auth['balance'])
                self.currency = auth['currency']
            
            logger.info(f"✅ AUTHORIZED: {self.login_id}")
            logger.info(f"💰 Balance: {self.balance} {self.currency}")
            
            # Request ALL accounts linked to this token
            self.send({"account_list": 1})
            self.send({"balance": 1, "subscribe": 1})
        
        # ACCOUNT LIST RESPONSE - ALL ACCOUNTS RETRIEVED
        elif 'account_list' in data:
            accounts_data = data['account_list']
            account_list = []
            
            for acc in accounts_data:
                loginid = acc['loginid']
                is_demo = 'VRTC' in loginid or 'VRW' in loginid
                
                account_info = {
                    'loginid': loginid,
                    'account_type': 'demo' if is_demo else 'real',
                    'currency': acc.get('currency', 'USD'),
                    'balance': float(acc.get('balance', 0)),
                    'is_virtual': acc.get('is_virtual', 0) == 1,
                    'disabled': acc.get('is_disabled', 0) == 1
                }
                
                if not account_info['disabled']:
                    account_list.append(account_info)
            
            with self.lock:
                self.account_list = account_list
            
            logger.info(f"📋 Found {len(account_list)} account(s):")
            for acc in account_list:
                logger.info(f"   • {acc['loginid']} ({acc['account_type']}) - {acc['currency']}")
        
        # BALANCE UPDATE
        elif 'balance' in data:
            with self.lock:
                self.balance = float(data['balance']['balance'])
            logger.info(f"💰 Balance updated: {self.balance} {self.currency}")
        
        # CANDLES RESPONSE
        elif 'candles' in data:
            symbol = data['echo_req']['ticks_history']
            with self.candle_cache_lock:
                self.candle_cache[symbol] = data['candles']
            
            if symbol in self.message_handlers:
                handler = self.message_handlers.pop(symbol)
                handler(data['candles'])
        
        # TRADE EXECUTION RESPONSE
        elif 'buy' in data:
            contract = data['buy']
            contract_id = contract['contract_id']
            price = contract['buy_price']
            
            logger.info(f"✅ Trade opened: {contract_id} | Price: {price}")
            
            with self.contracts_lock:
                self.active_contracts[contract_id] = contract
            
            self.send({
                'proposal_open_contract': 1,
                'contract_id': contract_id,
                'subscribe': 1
            })
        
        # CONTRACT UPDATE
        elif 'proposal_open_contract' in data:
            contract = data['proposal_open_contract']
            contract_id = contract.get('contract_id')
            
            if contract.get('is_sold'):
                profit = float(contract.get('profit', 0))
                
                if profit > 0:
                    logger.info(f"✅ WIN: +{profit:.2f} {self.currency}")
                else:
                    logger.info(f"❌ LOSS: {profit:.2f} {self.currency}")
                
                for callback in self.trade_callbacks:
                    try:
                        callback(contract_id, contract)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                with self.contracts_lock:
                    self.active_contracts.pop(contract_id, None)
    
    def switch_account(self, account_type):
        """Switch to a different account (demo/real)"""
        with self.lock:
            if not self.account_list:
                logger.warning("⚠️ No account list available")
                return False
            
            target_account = None
            for acc in self.account_list:
                if acc['account_type'].lower() == account_type.lower():
                    target_account = acc
                    break
            
            if not target_account:
                logger.warning(f"⚠️ No {account_type} account found")
                return False
            
            logger.info(f"🔄 Switching to {account_type} account: {target_account['loginid']}")
            
        self.send({
            "account_switch": 1,
            "loginid": target_account['loginid'],
            "req_id": 1
        })
        
        with self.lock:
            self.account_type = account_type
        
        return True
    
    def send(self, request):
        """Send request to Deriv API"""
        if self.ws and self.is_connected:
            try:
                self.ws.send(json.dumps(request))
                return True
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                return False
        else:
            logger.error("❌ Not connected to Deriv")
            return False
    
    def get_candles(self, symbol, count=200, granularity=60):
        """Fetch historical candles"""
        if not self.is_authorized:
            logger.error("❌ Not authorized")
            return None
        
        cache_key = f"{symbol}_{granularity}"
        with self.candle_cache_lock:
            if cache_key in self.candle_cache:
                return self._candles_to_dataframe(self.candle_cache[cache_key])
        
        request = {
            'ticks_history': symbol,
            'adjust_start_time': 1,
            'count': count,
            'end': 'latest',
            'start': 1,
            'style': 'candles',
            'granularity': granularity
        }
        
        event = threading.Event()
        result = []
        
        def handler(candles):
            nonlocal result
            result = candles
            event.set()
        
        self.message_handlers[symbol] = handler
        self.send(request)
        
        if event.wait(timeout=10):
            with self.candle_cache_lock:
                self.candle_cache[cache_key] = result
            return self._candles_to_dataframe(result)
        
        return None
    
    def _candles_to_dataframe(self, candles):
        """Convert candles list to DataFrame"""
        if not candles:
            return None
        
        df_data = []
        for c in candles:
            df_data.append({
                'time': c['epoch'],
                'open': float(c['open']),
                'high': float(c['high']),
                'low': float(c['low']),
                'close': float(c['close'])
            })
        
        return pd.DataFrame(df_data)
    
    def execute_trade(self, symbol, direction, stake, duration=5, duration_unit='m'):
        """Execute a trade on Deriv"""
        if not self.is_authorized:
            return False, "Not authorized"
        
        contract_type = "CALL" if direction == "BUY" else "PUT"
        
        request = {
            'buy': 1,
            'price': stake,
            'parameters': {
                'contract_type': contract_type,
                'symbol': symbol,
                'duration': duration,
                'duration_unit': duration_unit,
                'basis': 'stake',
                'amount': stake,
                'currency': self.currency
            }
        }
        
        logger.info(f"📤 Executing {contract_type} on {symbol} | Stake: {stake} {self.currency}")
        self.send(request)
        return True, "Trade executed"
    
    def get_trade_status(self, contract_id):
        """Get status of a trade"""
        with self.contracts_lock:
            return self.active_contracts.get(contract_id)
    
    def get_balance(self):
        """Get current balance"""
        with self.lock:
            return {
                'balance': self.balance,
                'currency': self.currency,
                'loginid': self.login_id,
                'account_type': self.account_type,
                'available_accounts': self.account_list
            }
    
    def disconnect(self):
        """Disconnect from Deriv"""
        self.should_run = False
        if self.ws:
            self.ws.close()
        with self.lock:
            self.is_connected = False
            self.is_authorized = False
        logger.info("🔌 Disconnected from Deriv")

# ============ MARKET STATE ENGINE ============
class MarketStateEngine:
    def __init__(self):
        self.ATR_PERIOD = 14
        
    def analyze(self, df):
        if df is None or len(df) < 100:
            return {
                'state': MarketState.CHOPPY.value,
                'direction': 'NEUTRAL',
                'strength': 0,
                'support': 0,
                'resistance': 0,
                'current_price': 0
            }
        
        try:
            df = self._calculate_indicators(df)
            current_price = df['close'].iloc[-1]
            ema_20 = df['ema_20'].iloc[-1]
            ema_50 = df['ema_50'].iloc[-1]
            
            if current_price > ema_20 > ema_50:
                state = MarketState.STRONG_UPTREND.value
                direction = 'BULLISH'
                strength = 80
            elif current_price < ema_20 < ema_50:
                state = MarketState.STRONG_DOWNTREND.value
                direction = 'BEARISH'
                strength = 80
            elif current_price > ema_50:
                state = MarketState.UPTREND.value
                direction = 'BULLISH'
                strength = 60
            elif current_price < ema_50:
                state = MarketState.DOWNTREND.value
                direction = 'BEARISH'
                strength = 60
            else:
                state = MarketState.RANGING.value
                direction = 'NEUTRAL'
                strength = 40
            
            support = float(df['low'].iloc[-20:].min())
            resistance = float(df['high'].iloc[-20:].max())
            
            return {
                'state': state,
                'direction': direction,
                'strength': strength,
                'support': support,
                'resistance': resistance,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                'state': MarketState.CHOPPY.value,
                'direction': 'NEUTRAL',
                'strength': 0,
                'support': 0,
                'resistance': 0,
                'current_price': 0
            }
    
    def _calculate_indicators(self, df):
        df = df.copy()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        return df

# ============ CONTINUATION ENGINE ============
class ContinuationEngine:
    def detect_setups(self, df, market_state):
        if not market_state or market_state.get('direction') == 'NEUTRAL':
            return []
        
        try:
            signals = []
            current_price = float(df['close'].iloc[-1])
            ema_20 = float(df['ema_20'].iloc[-1])
            atr = float(df['atr'].iloc[-1]) if not pd.isna(df['atr'].iloc[-1]) else 0
            
            is_bullish = market_state.get('direction') == 'BULLISH'
            is_bearish = market_state.get('direction') == 'BEARISH'
            
            if abs(current_price - ema_20) / ema_20 < 0.002:
                if is_bullish:
                    signals.append({
                        'type': 'SELL',
                        'entry': current_price,
                        'sl': current_price * 1.01,
                        'tp': current_price * 0.975,
                        'strategy': 'CONTINUATION',
                        'pattern': 'Bullish Pullback (SELL)',
                        'confidence': 75
                    })
                elif is_bearish:
                    signals.append({
                        'type': 'BUY',
                        'entry': current_price,
                        'sl': current_price * 0.99,
                        'tp': current_price * 1.025,
                        'strategy': 'CONTINUATION',
                        'pattern': 'Bearish Rally (BUY)',
                        'confidence': 75
                    })
            
            return signals[:2]
            
        except Exception as e:
            logger.error(f"Continuation error: {e}")
            return []

# ============ QUASIMODO ENGINE ============
class QuasimodoEngine:
    def detect_setups(self, df, market_state, symbol):
        if not market_state:
            return []
        
        try:
            signals = []
            current_price = float(df['close'].iloc[-1])
            support = market_state.get('support', current_price * 0.99)
            resistance = market_state.get('resistance', current_price * 1.01)
            
            if abs(current_price - resistance) / resistance < 0.001:
                signals.append({
                    'type': 'BUY',
                    'entry': current_price,
                    'sl': current_price * 0.99,
                    'tp': current_price * 1.02,
                    'strategy': 'QUASIMODO',
                    'pattern': 'Quasimodo Sell Setup (BUY)',
                    'confidence': 70
                })
            elif abs(current_price - support) / support < 0.001:
                signals.append({
                    'type': 'SELL',
                    'entry': current_price,
                    'sl': current_price * 1.01,
                    'tp': current_price * 0.98,
                    'strategy': 'QUASIMODO',
                    'pattern': 'Quasimodo Buy Setup (SELL)',
                    'confidence': 70
                })
            
            return signals[:2]
            
        except Exception as e:
            logger.error(f"Quasimodo error: {e}")
            return []

# ============ SMART STRATEGY SELECTOR ============
class SmartStrategySelector:
    def select_best_trades(self, cont, quasi, market_state):
        if not market_state:
            return []
        
        state = market_state.get('state')
        selected = []
        
        if state in [MarketState.STRONG_UPTREND.value, MarketState.STRONG_DOWNTREND.value]:
            selected = cont
        elif state in [MarketState.UPTREND.value, MarketState.DOWNTREND.value]:
            selected = cont + [q for q in quasi if q.get('confidence', 0) > 80]
        elif state == MarketState.RANGING.value:
            selected = quasi
        
        selected = [s for s in selected if s.get('confidence', 0) >= 65]
        selected.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return selected[:3]

# ============ TRADING ENGINE ============
class KarankaTradingEngine:
    def __init__(self):
        self.api = DerivConnection(app_id="1089")
        self.market_engine = MarketStateEngine()
        self.continuation = ContinuationEngine()
        self.quasimodo = QuasimodoEngine()
        self.selector = SmartStrategySelector()
        
        self.connected = False
        self.running = False
        self.token = None
        self.account_type = "demo"
        
        self.active_trades = []
        self.trade_history = []
        self.market_analysis = {}
        
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.total_wins = 0
        self.total_losses = 0
        self.analysis_cycle = 0
        
        self.dry_run = True
        self.max_daily_trades = MAX_DAILY_TRADES
        self.max_concurrent_trades = MAX_CONCURRENT_TRADES
        self.min_seconds_between = 10
        self.fixed_amount = FIXED_AMOUNT
        self.min_confidence = 65
        self.trailing_stop_enabled = True
        self.trailing_activation = 0.30
        self.trailing_lock = 0.85
        self.enabled_symbols = [
            "R_10", "R_25", "R_50", "R_75", "R_100",
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "US30", "US100", "XAUUSD", "XAGUSD", "BTCUSD"
        ]
        
        self.api.register_trade_callback(self.on_trade_update)
        self._start_daily_reset()
        self._start_trade_monitor()
        
        logger.info("✅ Karanka Trading Engine initialized")
    
    def _start_daily_reset(self):
        def reset():
            while True:
                now = datetime.now()
                midnight = datetime(now.year, now.month, now.day, 23, 59, 59)
                seconds = (midnight - now).total_seconds()
                if seconds > 0:
                    time.sleep(seconds)
                self.daily_trades = 0
                self.daily_pnl = 0.0
                self.consecutive_losses = 0
                time.sleep(1)
        threading.Thread(target=reset, daemon=True).start()
    
    def _start_trade_monitor(self):
        def monitor():
            while True:
                try:
                    if self.connected and self.running and self.active_trades:
                        for trade in self.active_trades[:]:
                            if not trade.get('dry_run', False):
                                status = self.api.get_trade_status(trade.get('contract_id'))
                                if status:
                                    trade['current_price'] = status.get('entry_tick', trade['entry'])
                                    self._apply_trailing_stop(trade)
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
                    time.sleep(10)
        threading.Thread(target=monitor, daemon=True).start()
    
    def _apply_trailing_stop(self, trade):
        if not self.trailing_stop_enabled:
            return
        
        try:
            direction = trade['direction']
            entry = trade['entry']
            tp = trade['tp']
            current = trade.get('current_price', entry)
            
            if direction == 'BUY':
                total_range = tp - entry
                if total_range <= 0:
                    return
                progress = (current - entry) / total_range
                if progress >= self.trailing_activation:
                    profit_so_far = current - entry
                    locked_profit = profit_so_far * self.trailing_lock
                    new_stop = current - (profit_so_far - locked_profit)
                    if new_stop > trade.get('current_stop', trade['sl']):
                        trade['current_stop'] = new_stop
                        logger.info(f"🔒 Trailing stop moved to {new_stop:.5f}")
            else:
                total_range = entry - tp
                if total_range <= 0:
                    return
                progress = (entry - current) / total_range
                if progress >= self.trailing_activation:
                    profit_so_far = entry - current
                    locked_profit = profit_so_far * self.trailing_lock
                    new_stop = current + (profit_so_far - locked_profit)
                    if new_stop < trade.get('current_stop', trade['sl']):
                        trade['current_stop'] = new_stop
                        logger.info(f"🔒 Trailing stop moved to {new_stop:.5f}")
        except Exception as e:
            logger.error(f"Trailing stop error: {e}")
    
    def on_trade_update(self, contract_id, contract_data):
        for trade in self.active_trades[:]:
            if trade.get('contract_id') == contract_id or trade.get('id') == contract_id:
                profit = float(contract_data.get('profit', 0))
                trade['profit'] = profit
                trade['exit_time'] = datetime.now().isoformat()
                trade['exit_price'] = float(contract_data.get('exit_tick', 0))
                trade['result'] = 'WIN' if profit > 0 else 'LOSS'
                
                self.daily_pnl += profit
                if profit > 0:
                    self.total_wins += 1
                    self.consecutive_losses = 0
                else:
                    self.total_losses += 1
                    self.consecutive_losses += 1
                
                if trade in self.active_trades:
                    self.active_trades.remove(trade)
                self.trade_history.append(trade)
                
                logger.info(f"📊 TRADE CLOSED: {trade['symbol']} | Profit: ${profit:.2f}")
                socketio.emit('trade_update', self.get_trade_data())
                break
    
    def connect(self, token, account_type="demo"):
        self.token = token
        self.account_type = account_type
        success, message = self.api.connect(token, account_type)
        if success:
            self.connected = True
        return success, message
    
    def disconnect(self):
        self.api.disconnect()
        self.connected = False
        self.running = False
    
    def start_trading(self, settings=None):
        if not self.connected:
            return False, "Not connected"
        if settings:
            self.dry_run = settings.get('dry_run', True)
            self.max_daily_trades = int(settings.get('max_daily_trades', self.max_daily_trades))
            self.max_concurrent_trades = int(settings.get('max_concurrent_trades', self.max_concurrent_trades))
            self.fixed_amount = float(settings.get('fixed_amount', self.fixed_amount))
            self.min_confidence = int(settings.get('min_confidence', self.min_confidence))
            self.trailing_stop_enabled = settings.get('trailing_stop', True)
            if settings.get('enabled_symbols'):
                self.enabled_symbols = settings['enabled_symbols']
        self.running = True
        threading.Thread(target=self._trading_loop, daemon=True).start()
        return True, f"Trading started on {self.account_type} account"
    
    def stop_trading(self):
        self.running = False
    
    def _trading_loop(self):
        while self.running and self.connected:
            try:
                self.analysis_cycle += 1
                can_trade = self._can_trade()
                
                for symbol in self.enabled_symbols[:10]:
                    try:
                        df = self._get_market_data(symbol)
                        if df is None or len(df) < 100:
                            continue
                        
                        market_state = self.market_engine.analyze(df)
                        cont = self.continuation.detect_setups(df, market_state)
                        quasi = self.quasimodo.detect_setups(df, market_state, symbol)
                        best = self.selector.select_best_trades(cont, quasi, market_state)
                        
                        self.market_analysis[symbol] = {
                            'symbol': symbol,
                            'price': float(df['close'].iloc[-1]),
                            'market_state': market_state.get('state'),
                            'strength': market_state.get('strength'),
                            'signals': [{'type': t['type'], 'strategy': t['strategy'], 'confidence': t['confidence']} for t in best[:2]]
                        }
                        
                        if best and can_trade and len(self.active_trades) < self.max_concurrent_trades:
                            trade = best[0]
                            if trade['confidence'] >= self.min_confidence:
                                self._execute_trade(symbol, trade)
                                self.daily_trades += 1
                                can_trade = self._can_trade()
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}")
                
                socketio.emit('market_update', {'market_analysis': self.market_analysis})
                socketio.emit('trade_update', self.get_trade_data())
                time.sleep(8)
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(10)
    
    def _get_market_data(self, symbol):
        try:
            candles = self.api.get_candles(symbol, count=300, granularity=60)
            if candles is None or len(candles) < 100:
                return None
            
            df = candles.copy()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def _can_trade(self):
        if len(self.active_trades) >= self.max_concurrent_trades:
            return False
        if self.daily_trades >= self.max_daily_trades:
            return False
        if self.last_trade_time:
            seconds_since = (datetime.now() - self.last_trade_time).total_seconds()
            if seconds_since < self.min_seconds_between:
                return False
        if self.consecutive_losses >= 3:
            return False
        return True
    
    def _execute_trade(self, symbol, trade):
        try:
            if self.dry_run:
                trade_record = {
                    'id': f"dry_{int(time.time())}_{symbol}",
                    'symbol': symbol,
                    'direction': trade['type'],
                    'entry': float(trade['entry']),
                    'amount': self.fixed_amount,
                    'strategy': trade['strategy'],
                    'confidence': trade['confidence'],
                    'entry_time': datetime.now().isoformat(),
                    'dry_run': True,
                    'account_type': self.account_type
                }
                self.active_trades.append(trade_record)
                logger.info(f"✅ [DRY RUN] {symbol} {trade['type']} on {self.account_type}")
                
                def simulate():
                    time.sleep(120)
                    win = random.random() < (trade['confidence'] / 100)
                    profit = self.fixed_amount * 2.5 if win else -self.fixed_amount
                    trade_record['profit'] = profit
                    trade_record['exit_time'] = datetime.now().isoformat()
                    trade_record['result'] = 'WIN' if win else 'LOSS'
                    
                    self.daily_pnl += profit
                    if win:
                        self.total_wins += 1
                        self.consecutive_losses = 0
                    else:
                        self.total_losses += 1
                        self.consecutive_losses += 1
                    
                    if trade_record in self.active_trades:
                        self.active_trades.remove(trade_record)
                    self.trade_history.append(trade_record)
                    socketio.emit('trade_update', self.get_trade_data())
                
                threading.Thread(target=simulate, daemon=True).start()
                return True, "Dry run trade placed"
            else:
                success, msg = self.api.execute_trade(symbol, trade['type'], self.fixed_amount)
                if success:
                    trade_record = {
                        'id': f"real_{int(time.time())}_{symbol}",
                        'symbol': symbol,
                        'direction': trade['type'],
                        'entry': float(trade['entry']),
                        'amount': self.fixed_amount,
                        'strategy': trade['strategy'],
                        'confidence': trade['confidence'],
                        'entry_time': datetime.now().isoformat(),
                        'dry_run': False,
                        'account_type': self.account_type
                    }
                    self.active_trades.append(trade_record)
                    logger.info(f"✅ REAL TRADE: {symbol} {trade['type']} on {self.account_type}")
                    return True, "Trade executed"
                return False, msg
        except Exception as e:
            logger.error(f"Trade error: {e}")
            return False, str(e)
    
    def get_market_data(self):
        return {'market_analysis': self.market_analysis, 'cycle': self.analysis_cycle}
    
    def get_trade_data(self):
        return {
            'active_trades': self.active_trades[-20:],
            'trade_history': self.trade_history[-50:],
            'daily_trades': self.daily_trades,
            'daily_pnl': round(self.daily_pnl, 2),
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
            'win_rate': round((self.total_wins / (self.total_wins + self.total_losses + 1)) * 100, 1),
            'account_type': self.account_type
        }
    
    def get_status(self):
        return {
            'connected': self.connected,
            'running': self.running,
            'dry_run': self.dry_run,
            'active_trades': len(self.active_trades),
            'daily_trades': self.daily_trades,
            'daily_pnl': round(self.daily_pnl, 2),
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
            'win_rate': round((self.total_wins / (self.total_wins + self.total_losses + 1)) * 100, 1),
            'account_type': self.account_type
        }

# ============ INITIALIZE ============
trading_engine = KarankaTradingEngine()
keep_awake = KeepAwake(BASE_URL)

# ============ FLASK ROUTES ============

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'connected': trading_engine.connected,
        'running': trading_engine.running,
        'active_trades': len(trading_engine.active_trades),
        'balance': trading_engine.api.balance,
        'currency': trading_engine.api.currency,
        'account_type': trading_engine.account_type
    })

@app.route('/api/connect', methods=['POST'])
def api_connect():
    try:
        data = request.json
        token = data.get('token')
        account_type = data.get('account_type', 'demo')
        
        if not token:
            return jsonify({'success': False, 'message': 'Token required'})
        
        logger.info(f"Connecting with token: {token[:4]}...{token[-4:]}")
        success, message = trading_engine.connect(token, account_type)
        
        if success:
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
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/balance')
def api_balance():
    if trading_engine.connected:
        return jsonify({'success': True, 'balance': trading_engine.api.get_balance()})
    return jsonify({'success': False, 'message': 'Not connected'})

@app.route('/api/disconnect', methods=['POST'])
def api_disconnect():
    trading_engine.disconnect()
    return jsonify({'success': True})

@app.route('/api/start', methods=['POST'])
def api_start():
    settings = request.json or {}
    success, msg = trading_engine.start_trading(settings)
    return jsonify({'success': success, 'message': msg})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    trading_engine.stop_trading()
    return jsonify({'success': True})

@app.route('/api/status')
def api_status():
    return jsonify(trading_engine.get_status())

@app.route('/api/market_data')
def api_market_data():
    return jsonify(trading_engine.get_market_data())

@app.route('/api/trade_data')
def api_trade_data():
    return jsonify(trading_engine.get_trade_data())

@app.route('/api/settings', methods=['POST'])
def api_settings():
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

# ============ SOCKETIO ============
@socketio.on('connect')
def handle_connect():
    logger.info("📱 Client connected")

@socketio.on('request_update')
def handle_update():
    emit('market_update', trading_engine.get_market_data())
    emit('trade_update', trading_engine.get_trade_data())

# ============ START ============
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    logger.info("=" * 60)
    logger.info("KARANKA MULTIVERSE ALGO AI - DERIV PRODUCTION BOT")
    logger.info("=" * 60)
    logger.info("✅ WORKING DERIV CONNECTION")
    logger.info("✅ NON-BLOCKING WEBSOCKET")
    logger.info("✅ PING/PONG KEEP-ALIVE")
    logger.info("✅ DEMO & REAL ACCOUNT SUPPORT")
    logger.info("=" * 60)
    
    keep_awake.start()
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
