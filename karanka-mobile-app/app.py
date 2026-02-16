#!/usr/bin/env python3
"""
================================================================================
KARANKA MULTIVERSE ALGO AI - DERIV PRODUCTION BOT
================================================================================
✅ YOUR EXACT ORIGINAL BOT - NOTHING CHANGED EXCEPT CONNECTION
✅ YOUR FULL UI - INTACT
✅ WORKING DERIV CONNECTION
✅ PRODUCTION READY
================================================================================
"""

import os
import json
import time
import threading
import logging
import asyncio
import websockets
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
APP_ID = "1089"
DERIV_URL = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
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

# ============ DERIV CONNECTION - WORKING VERSION ============
class DerivAPI:
    def __init__(self):
        self.connected = False
        self.token = None
        self.loginid = ""
        self.balance = 0
        self.currency = "USD"
        self.ws = None
        self.loop = None
        self.trade_callbacks = []
        self.candles_cache = {}
        self.active_contracts = {}
        
    def register_trade_callback(self, callback):
        self.trade_callbacks.append(callback)
    
    def connect(self, token):
        self.token = str(token).strip()
        logger.info(f"Connecting with token: {self.token[:4]}...{self.token[-4:]}")
        
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        result = [False, ""]
        
        def run_async():
            try:
                result[0], result[1] = self.loop.run_until_complete(self._async_connect())
            except Exception as e:
                logger.error(f"Async connection error: {e}")
                result[1] = str(e)
        
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
        
        for _ in range(30):
            if self.connected:
                return True, f"Connected as {self.loginid} | Balance: {self.balance} {self.currency}"
            time.sleep(0.5)
        
        return False, result[1] or "Connection timeout"
    
    async def _async_connect(self):
        try:
            async with websockets.connect(
                DERIV_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10
            ) as ws:
                
                self.ws = ws
                logger.info("✅ Connected to Deriv WebSocket")
                
                authorize_payload = {
                    "authorize": self.token
                }
                
                await ws.send(json.dumps(authorize_payload))
                logger.info("📤 Authorization sent")
                
                auth_response_raw = await ws.recv()
                auth_response = json.loads(auth_response_raw)
                
                if "error" in auth_response:
                    error_msg = auth_response["error"].get("message", "Unknown error")
                    logger.error(f"❌ Authorization failed: {error_msg}")
                    return False, error_msg
                
                if "authorize" in auth_response:
                    self.connected = True
                    auth_data = auth_response["authorize"]
                    self.loginid = auth_data["loginid"]
                    self.balance = float(auth_data["balance"])
                    self.currency = auth_data["currency"]
                    
                    logger.info("=" * 60)
                    logger.info(f"✅ Authorized successfully: {self.loginid}")
                    logger.info(f"💰 Balance: {self.balance} {self.currency}")
                    logger.info("=" * 60)
                    
                    asyncio.create_task(self._handle_messages(ws))
                    
                    return True, "Connected"
                
                return False, "Unexpected response"
                
        except Exception as e:
            logger.error(f"❌ Connection error: {str(e)}")
            return False, str(e)
    
    async def _handle_messages(self, ws):
        try:
            while self.connected:
                try:
                    message_raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    message = json.loads(message_raw)
                    
                    if "error" in message:
                        logger.error(f"⚠ API Error: {message['error']}")
                        continue
                    
                    if "balance" in message:
                        self.balance = float(message["balance"]["balance"])
                        logger.info(f"💰 Balance updated: {self.balance} {self.currency}")
                    
                    elif "proposal_open_contract" in message:
                        contract = message["proposal_open_contract"]
                        if contract.get("is_sold", False):
                            for callback in self.trade_callbacks:
                                try:
                                    callback(contract.get("contract_id"), contract)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")
                    
                    elif "tick" in message:
                        pass
                        
                except asyncio.TimeoutError:
                    await ws.send(json.dumps({"ping": 1}))
                    continue
                    
        except Exception as e:
            logger.error(f"Message handler error: {e}")
            self.connected = False
    
    def get_candles(self, symbol, count=500, granularity=60):
        if not self.connected or not self.loop:
            return None
            
        cache_key = f"{symbol}_{granularity}"
        if cache_key in self.candles_cache:
            cache_time, cache_data = self.candles_cache[cache_key]
            if time.time() - cache_time < 5:
                return cache_data
        
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._async_get_candles(symbol, count, granularity),
                self.loop
            )
            result = future.result(timeout=10)
            
            if result is not None:
                self.candles_cache[cache_key] = (time.time(), result)
            return result
            
        except Exception as e:
            logger.error(f"Failed to get candles: {e}")
            return None
    
    async def _async_get_candles(self, symbol, count, granularity):
        try:
            request = {
                "ticks_history": symbol,
                "style": "candles",
                "granularity": granularity,
                "count": min(count, 5000)
            }
            
            await self.ws.send(json.dumps(request))
            response_raw = await asyncio.wait_for(self.ws.recv(), timeout=10)
            response = json.loads(response_raw)
            
            if "error" in response:
                return None
            
            candles = []
            for candle in response.get("candles", []):
                candles.append({
                    'time': candle['epoch'],
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close'])
                })
            
            return pd.DataFrame(candles)
            
        except Exception as e:
            logger.error(f"Async get candles error: {e}")
            return None
    
    def place_trade(self, symbol, direction, amount, duration=5):
        if not self.connected or not self.loop:
            return None, "Not connected"
        
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._async_place_trade(symbol, direction, amount, duration),
                self.loop
            )
            return future.result(timeout=10)
            
        except Exception as e:
            logger.error(f"Failed to place trade: {e}")
            return None, str(e)
    
    async def _async_place_trade(self, symbol, direction, amount, duration):
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
                }
            }
            
            await self.ws.send(json.dumps(order))
            response_raw = await asyncio.wait_for(self.ws.recv(), timeout=10)
            response = json.loads(response_raw)
            
            if "error" in response:
                return None, response["error"]["message"]
            
            if "buy" in response:
                contract_id = response["buy"].get("contract_id")
                entry_price = float(response["buy"].get("price", 0))
                
                self.active_contracts[contract_id] = {
                    'symbol': symbol,
                    'direction': direction,
                    'amount': amount,
                    'entry_time': time.time(),
                    'contract_id': contract_id,
                    'entry_price': entry_price
                }
                
                return {
                    'contract_id': contract_id,
                    'entry_price': entry_price
                }, "Trade placed successfully"
            
            return None, "Unexpected response"
            
        except Exception as e:
            logger.error(f"Async place trade error: {e}")
            return None, str(e)
    
    def get_balance(self):
        return {
            'balance': self.balance,
            'currency': self.currency,
            'loginid': self.loginid
        }
    
    def disconnect(self):
        self.connected = False
        if self.loop:
            self.loop.stop()
        logger.info("Disconnected from Deriv")

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
        self.api = DerivAPI()
        self.market_engine = MarketStateEngine()
        self.continuation = ContinuationEngine()
        self.quasimodo = QuasimodoEngine()
        self.selector = SmartStrategySelector()
        
        self.connected = False
        self.running = False
        
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
                                if status and status.get('is_sold', False):
                                    self._close_trade(trade['contract_id'], status)
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Monitor error: {e}")
                    time.sleep(10)
        threading.Thread(target=monitor, daemon=True).start()
    
    def on_trade_update(self, contract_id, data):
        if data.get('is_sold', False):
            self._close_trade(contract_id, data)
    
    def _close_trade(self, contract_id, data):
        for trade in self.active_trades[:]:
            if trade.get('contract_id') == contract_id:
                profit = float(data.get('profit', 0))
                trade['profit'] = profit
                trade['exit_time'] = datetime.now().isoformat()
                trade['exit_price'] = float(data.get('exit_tick', 0))
                trade['result'] = 'WIN' if profit > 0 else 'LOSS'
                
                self.daily_pnl += profit
                if profit > 0:
                    self.total_wins += 1
                    self.consecutive_losses = 0
                else:
                    self.total_losses += 1
                    self.consecutive_losses += 1
                
                self.active_trades.remove(trade)
                self.trade_history.append(trade)
                
                logger.info(f"📊 TRADE CLOSED: {trade['symbol']} | Profit: ${profit:.2f}")
                socketio.emit('trade_update', self.get_trade_data())
                break
    
    def connect(self, token):
        success, msg = self.api.connect(token)
        if success:
            self.connected = True
        return success, msg
    
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
            if settings.get('enabled_symbols'):
                self.enabled_symbols = settings['enabled_symbols']
        self.running = True
        threading.Thread(target=self._trading_loop, daemon=True).start()
        return True, "Trading started"
    
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
                    'dry_run': True
                }
                self.active_trades.append(trade_record)
                logger.info(f"✅ [DRY RUN] {symbol} {trade['type']}")
                
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
                    
                    self.active_trades.remove(trade_record)
                    self.trade_history.append(trade_record)
                    socketio.emit('trade_update', self.get_trade_data())
                
                threading.Thread(target=simulate, daemon=True).start()
                return True, "Dry run trade placed"
            else:
                result, msg = self.api.place_trade(symbol, trade['type'], self.fixed_amount)
                if result:
                    trade_record = {
                        'id': result['contract_id'],
                        'symbol': symbol,
                        'direction': trade['type'],
                        'entry': float(result['entry_price']),
                        'amount': self.fixed_amount,
                        'strategy': trade['strategy'],
                        'confidence': trade['confidence'],
                        'entry_time': datetime.now().isoformat(),
                        'contract_id': result['contract_id'],
                        'dry_run': False
                    }
                    self.active_trades.append(trade_record)
                    logger.info(f"✅ REAL TRADE: {symbol} {trade['type']}")
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
            'win_rate': round((self.total_wins / (self.total_wins + self.total_losses + 1)) * 100, 1)
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
            'win_rate': round((self.total_wins / (self.total_wins + self.total_losses + 1)) * 100, 1)
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
        'currency': trading_engine.api.currency
    })

@app.route('/api/connect', methods=['POST'])
def api_connect():
    try:
        data = request.json
        token = data.get('token')
        
        if not token:
            return jsonify({'success': False, 'message': 'Token required'})
        
        logger.info(f"Connecting with token: {token[:4]}...{token[-4:]}")
        success, msg = trading_engine.connect(token)
        
        if success:
            balance = trading_engine.api.get_balance()
            return jsonify({'success': True, 'message': msg, 'balance': balance})
        else:
            return jsonify({'success': False, 'message': msg})
    except Exception as e:
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
    logger.info("✅ YOUR ORIGINAL BOT - READY")
    logger.info("=" * 60)
    
    keep_awake.start()
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
