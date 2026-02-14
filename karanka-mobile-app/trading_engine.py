# trading_engine.py - Main Trading Engine
import threading
import time
from datetime import datetime
import pandas as pd

from deriv_connector import DerivConnector, timeframe_to_granularity
from market_state import MarketStateEngine
from quasimodo_engine import QuasimodoEngine
from pullback_engine import PullbackEngine
from breakout_engine import BreakoutEngine
from scoring_engine import ScoringEngine

class TradingEngine:
    """
    MAIN TRADING ENGINE - Coordinates all strategies
    """
    
    def __init__(self, api_token, account_type='demo', settings=None):
        self.api_token = api_token
        self.account_type = account_type
        self.settings = settings or {}
        
        # Initialize Deriv connector
        self.deriv = DerivConnector(api_token, account_type=account_type)
        
        # Initialize all engines
        self.market_state = MarketStateEngine()
        self.quasimodo = QuasimodoEngine()
        self.pullback = PullbackEngine()
        self.breakout = BreakoutEngine()
        self.scoring = ScoringEngine()
        
        # State tracking
        self.connected = False
        self.running = False
        self.active_trades = []
        self.trade_history = []
        self.trades_today = 0
        self.trades_hour = 0
        self.last_trade_time = None
        self.total_cycles = 0
        
        # Data cache
        self.candle_data = {}
        self.scan_results = {}
        self.ready_signals = {}
        
        # Settings
        self.max_daily_trades = self.settings.get('max_daily_trades', 25)
        self.max_hourly_trades = self.settings.get('max_hourly_trades', 6)
        self.min_seconds_between = self.settings.get('min_seconds_between', 8)
        self.max_concurrent = self.settings.get('max_concurrent', 5)
        self.fixed_lot_size = self.settings.get('fixed_lot_size', 0.01)
        self.enable_5m = self.settings.get('enable_5m', True)
        self.enable_15m = self.settings.get('enable_15m', True)
        self.enable_30m = self.settings.get('enable_30m', True)
        self.enabled_symbols = self.settings.get('enabled_symbols', [
            "EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD",
            "US30", "USTEC", "AUDUSD", "BTCUSD"
        ])
        
    def connect(self):
        """Connect to Deriv"""
        success, message = self.deriv.connect()
        if success:
            self.connected = True
            
            # Subscribe to all enabled symbols
            for symbol in self.enabled_symbols:
                if self.enable_5m:
                    self.deriv.subscribe_candles(symbol, 300)  # 5m
                if self.enable_15m:
                    self.deriv.subscribe_candles(symbol, 900)  # 15m
                if self.enable_30m:
                    self.deriv.subscribe_candles(symbol, 1800)  # 30m
                self.deriv.subscribe_candles(symbol, 3600)  # 1h always for HTF
            
            return True, message
        else:
            self.connected = False
            return False, message
    
    def start_trading(self):
        """Start trading loop"""
        if not self.connected:
            return False, "Not connected"
        
        self.running = True
        self.trades_today = 0
        self.trades_hour = 0
        self.active_trades = []
        
        # Start trading thread
        thread = threading.Thread(target=self._trading_loop, daemon=True)
        thread.start()
        
        # Start hourly reset
        def reset_hourly():
            while self.running:
                time.sleep(3600)
                self.trades_hour = 0
        
        threading.Thread(target=reset_hourly, daemon=True).start()
        
        return True, "Trading started"
    
    def stop_trading(self):
        """Stop trading"""
        self.running = False
        
    def _trading_loop(self):
        """Main trading loop"""
        print("🚀 Trading loop started")
        
        while self.running and self.connected:
            try:
                self.total_cycles += 1
                
                # Reset scan results for this cycle
                self.scan_results = {}
                self.ready_signals = {}
                
                # Scan each symbol
                for symbol in self.enabled_symbols:
                    if not self.running:
                        break
                    
                    # Check if symbol already has active trade
                    if any(t['symbol'] == symbol for t in self.active_trades):
                        continue
                    
                    # Analyze symbol
                    ready_signal, scan_data = self._analyze_symbol(symbol)
                    
                    if scan_data:
                        self.scan_results[symbol] = scan_data
                    
                    if ready_signal:
                        self.ready_signals[symbol] = ready_signal
                
                # Execute best signals
                self._execute_signals()
                
                # Sleep between cycles
                time.sleep(8)
                
            except Exception as e:
                print(f"Trading loop error: {e}")
                time.sleep(5)
    
    def _analyze_symbol(self, symbol):
        """Analyze a single symbol"""
        scan_data = {
            'price': 0,
            'regime': 'SCANNING',
            'trend': 'NEUTRAL',
            'volatility': 'NORMAL',
            'qm_patterns': 0,
            'pb_setups': 0,
            'breakout': False
        }
        
        ready_signal = None
        
        try:
            # Get data for all timeframes
            df_h1 = self.deriv.get_candles(symbol, 3600, 300)  # 1h
            df_m15 = self.deriv.get_candles(symbol, 900, 200)  # 15m
            df_m5 = self.deriv.get_candles(symbol, 300, 100) if self.enable_5m else None
            df_m30 = self.deriv.get_candles(symbol, 1800, 200) if self.enable_30m else None
            
            if df_h1 is None or df_m15 is None or len(df_h1) < 50:
                return None, scan_data
            
            # Get current price
            current_price_info = self.deriv.get_current_price(symbol)
            scan_data['price'] = current_price_info['price'] if current_price_info else df_m15['close'].iloc[-1]
            
            # Detect market regime
            regime = self.market_state.detect_regime(df_h1, df_m15, df_m5)
            scan_data['regime'] = regime['market_type']
            scan_data['trend'] = regime['primary_trend']
            scan_data['volatility'] = regime['volatility']
            
            all_signals = []
            
            # Run Quasimodo
            if self.enable_15m:
                qm_15m = self.quasimodo.detect_quasimodo(df_m15, symbol, '15M')
                for s in qm_15m:
                    s['timeframe'] = '15M'
                    all_signals.append(s)
                scan_data['qm_patterns'] += len(qm_15m)
            
            if self.enable_30m and df_m30 is not None:
                qm_30m = self.quasimodo.detect_quasimodo(df_m30, symbol, '30M')
                for s in qm_30m:
                    s['timeframe'] = '30M'
                    all_signals.append(s)
                scan_data['qm_patterns'] += len(qm_30m)
            
            # Run Pullback
            pb_signal = self.pullback.detect_pullback(df_m15, df_h1, regime, symbol)
            if pb_signal:
                all_signals.append(pb_signal)
                scan_data['pb_setups'] = 1
            
            # Run Breakout
            if self.enable_5m and df_m5 is not None:
                bo_signal = self.breakout.detect_breakout(df_m5, df_m15, regime, symbol)
                if bo_signal:
                    all_signals.append(bo_signal)
                    scan_data['breakout'] = True
            
            if not all_signals:
                return None, scan_data
            
            # Score all signals
            scored_signals = []
            for signal in all_signals:
                score_result = self.scoring.score_signal(signal, regime, df_h1, df_m15, df_m5)
                
                if score_result['should_trade']:
                    signal['score'] = score_result['score']
                    signal['reasons'] = score_result['reasons']
                    scored_signals.append((signal, score_result['score']))
            
            if not scored_signals:
                return None, scan_data
            
            # Pick best signal
            scored_signals.sort(key=lambda x: x[1], reverse=True)
            best_signal = scored_signals[0][0]
            
            # For Quasimodo, check retest
            if best_signal['strategy'] == 'QUASIMODO':
                df_current = self.deriv.get_candles(symbol, 900, 20)  # Recent 15m candles
                retested, retest_price = self.quasimodo.check_retest(df_current, best_signal, symbol)
                if not retested:
                    return None, scan_data
                best_signal['entry'] = retest_price
            
            # Format ready signal
            ready_signal = {
                'symbol': symbol,
                'direction': best_signal['type'],
                'strategy': best_signal['strategy'],
                'entry': best_signal['entry'],
                'sl': best_signal['sl'],
                'tp': best_signal['tp'],
                'score': best_signal['score'],
                'reasons': best_signal.get('reasons', []),
                'timeframe': best_signal.get('timeframe', 'N/A'),
                'regime': regime['market_type']
            }
            
            return ready_signal, scan_data
            
        except Exception as e:
            print(f"Analysis error for {symbol}: {e}")
            return None, scan_data
    
    def _execute_signals(self):
        """Execute the best signals"""
        if not self.ready_signals:
            return
        
        # Check if we can trade
        if not self._can_trade():
            return
        
        # Sort by score
        candidates = sorted(
            self.ready_signals.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Execute up to 2 per cycle
        executed = 0
        for signal in candidates[:2]:
            if not self._can_trade():
                break
            
            # Execute trade
            success, result = self._execute_trade(signal)
            
            if success:
                executed += 1
                self.trades_today += 1
                self.trades_hour += 1
                self.last_trade_time = datetime.now()
                
                # Add to active trades
                self.active_trades.append({
                    'symbol': signal['symbol'],
                    'direction': signal['direction'],
                    'strategy': signal['strategy'],
                    'entry': signal['entry'],
                    'sl': signal['sl'],
                    'tp': signal['tp'],
                    'volume': self.fixed_lot_size,
                    'time': datetime.now(),
                    'ticket': result.get('transaction_id', 'N/A')
                })
                
                print(f"✅ Executed: {signal['symbol']} {signal['direction']} | Score: {signal['score']}")
    
    def _execute_trade(self, signal):
        """Execute a single trade"""
        try:
            # Place order via Deriv
            success, result = self.deriv.place_order(
                symbol=signal['symbol'],
                direction=signal['direction'],
                volume=self.fixed_lot_size
            )
            
            if success:
                # Log trade
                self.trade_history.append({
                    'time': datetime.now(),
                    'symbol': signal['symbol'],
                    'direction': signal['direction'],
                    'strategy': signal['strategy'],
                    'entry': signal['entry'],
                    'sl': signal['sl'],
                    'tp': signal['tp'],
                    'volume': self.fixed_lot_size,
                    'result': result
                })
                
                return True, result
            else:
                print(f"Trade failed: {result}")
                return False, result
                
        except Exception as e:
            print(f"Execution error: {e}")
            return False, str(e)
    
    def _can_trade(self):
        """Check if we can trade"""
        if len(self.active_trades) >= self.max_concurrent:
            return False
        if self.trades_hour >= self.max_hourly_trades:
            return False
        if self.trades_today >= self.max_daily_trades:
            return False
        if self.last_trade_time:
            seconds_since = (datetime.now() - self.last_trade_time).total_seconds()
            if seconds_since < self.min_seconds_between:
                return False
        return True
    
    def get_status(self):
        """Get current status"""
        return {
            'connected': self.connected,
            'running': self.running,
            'balance': self.deriv.balance,
            'currency': self.deriv.currency,
            'account_type': self.account_type,
            'loginid': self.deriv.loginid,
            'active_trades': len(self.active_trades),
            'trades_today': self.trades_today,
            'trades_hour': self.trades_hour,
            'total_cycles': self.total_cycles,
            'scanned_markets': len(self.scan_results),
            'ready_signals': len(self.ready_signals)
        }
    
    def get_scan_results(self):
        """Get scan results for display"""
        return self.scan_results
    
    def get_ready_signals(self):
        """Get ready signals for display"""
        return self.ready_signals
    
    def get_active_trades(self):
        """Get active trades"""
        return self.active_trades
    
    def disconnect(self):
        """Disconnect from Deriv"""
        self.running = False
        self.deriv.disconnect()
        self.connected = False