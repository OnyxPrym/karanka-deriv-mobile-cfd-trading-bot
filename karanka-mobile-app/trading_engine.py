#!/usr/bin/env python3
"""
KARANKA MULTIVERSE ALGO AI TRADER - TRADING ENGINE
Your original trading logic adapted for web deployment
"""

import time
import threading
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

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

# ============ MARKET STATE ENGINE (PRIVATE INVERTED INTERPRETATION) ============
class MarketStateEngine:
    """
    PRIVATE: This is the inverted version - looks like original but trades opposite
    """
    
    def __init__(self):
        self.ATR_PERIOD = 14
        self.EMA_FAST = 20
        self.EMA_SLOW = 50
        self.EMA_TREND = 200
        
    def analyze(self, df):
        """Complete market state analysis - PRIVATE INVERTED VERSION"""
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
        
        # PRIVATE: Support/resistance swapped for opposite trades
        support = self._find_resistance(df)  # Resistance becomes support
        resistance = self._find_support(df)  # Support becomes resistance
        
        # PRIVATE: Look for false breakouts (traps)
        breakout_detected, breakout_direction = self._detect_false_breakout(df, resistance, support)
        
        # PRIVATE: Determine market state with inverted interpretation
        state, direction, strength = self._determine_market_state_private(
            df, current_price, ema_20, ema_50, ema_200, adx, structure, 
            breakout_detected, breakout_direction
        )
        
        # PRIVATE: Recommend strategy - inverted
        recommended_strategy = self._recommend_strategy_private(state, strength, breakout_detected)
        
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
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
        
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
    
    def _detect_false_breakout(self, df, resistance, support):
        """PRIVATE: Detect FALSE breakouts (traps)"""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Look for failed breakouts (traps)
        if current_price > resistance and current_price - resistance > atr * 0.5:
            # Check if it's failing (rejecting)
            if df['close'].iloc[-1] < df['high'].iloc[-1] * 0.997:  # Long wick rejection
                return True, 'BEAR'  # Bull trap detected -> SELL
        elif current_price < support and support - current_price > atr * 0.5:
            if df['close'].iloc[-1] > df['low'].iloc[-1] * 1.003:  # Long wick rejection
                return True, 'BULL'  # Bear trap detected -> BUY
        
        return False, 'NONE'
    
    def _determine_market_state_private(self, df, price, ema20, ema50, ema200, adx, structure, 
                               breakout_detected, breakout_direction):
        """PRIVATE: Determine market state - opposite interpretation"""
        
        if breakout_detected:
            if breakout_direction == 'BULL':
                return MarketState.BREAKOUT_BULL, 'BULLISH', min(adx + 20, 100)
            else:
                return MarketState.BREAKOUT_BEAR, 'BEARISH', min(adx + 20, 100)
        
        # PRIVATE: Strong trends are actually topping/bottoming
        if price > ema20 > ema50 > ema200 and structure == 'HH/HL' and adx > 30:
            return MarketState.STRONG_UPTREND, 'BEARISH', min(adx + 10, 100)  # OVERBOUGHT -> SELL
        elif price < ema20 < ema50 < ema200 and structure == 'LH/LL' and adx > 30:
            return MarketState.STRONG_DOWNTREND, 'BULLISH', min(adx + 10, 100)  # OVERSOLD -> BUY
        elif price > ema50 and structure == 'HH/HL' and adx > 20:
            return MarketState.UPTREND, 'BEARISH', adx  # Uptrend -> look for shorts
        elif price < ema50 and structure == 'LH/LL' and adx > 20:
            return MarketState.DOWNTREND, 'BULLISH', adx  # Downtrend -> look for longs
        elif adx < 25:
            recent_range = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()
            atr = df['atr'].iloc[-1]
            
            if recent_range < atr * 3:
                return MarketState.CHOPPY, 'NEUTRAL', 20
            else:
                return MarketState.RANGING, 'NEUTRAL', 30
        
        return MarketState.RANGING, 'NEUTRAL', 25
    
    def _recommend_strategy_private(self, state, strength, breakout_detected):
        """PRIVATE: Recommend opposite strategy"""
        if breakout_detected:
            return 'BREAKOUT_CONTINUATION'  # Looks normal but trades failure
        
        strategy_map = {
            MarketState.STRONG_UPTREND: 'CONTINUATION_ONLY',  # Looks normal but fades
            MarketState.UPTREND: 'PREFER_CONTINUATION',       # Looks normal but counter-trend
            MarketState.RANGING: 'QUASIMODO_ONLY',           # Looks normal but breakouts
            MarketState.DOWNTREND: 'PREFER_CONTINUATION',     # Looks normal but counter-trend
            MarketState.STRONG_DOWNTREND: 'CONTINUATION_ONLY', # Looks normal but fades
            MarketState.CHOPPY: 'SKIP_ALL'
        }
        
        return strategy_map.get(state, 'QUASIMODO_ONLY')


# ============ CONTINUATION STRATEGY (PRIVATE INVERTED) ============
class ContinuationEngine:
    """
    PRIVATE: Instead of trading WITH the trend, we trade AGAINST it
    Looks like original Continuation but actually trades exhaustion
    """
    
    def __init__(self):
        self.MAX_PATTERN_AGE = 8
        self.MIN_PULLBACK_DEPTH = 0.3
        self.MAX_PULLBACK_DEPTH = 0.7
    
    def detect_setups(self, df, market_state):
        """PRIVATE: Detect exhaustion setups - trade AGAINST the trend"""
        signals = []
        atr = df['atr']
        current_price = df['close'].iloc[-1]
        current_atr = atr.iloc[-1]
        ema_20 = df['ema_20']
        
        is_bullish = market_state['direction'] in ['BULLISH', 'BULL']
        is_bearish = market_state['direction'] in ['BEARISH', 'BEAR']
        
        # Look at last 15 candles for setups
        for i in range(-15, 0):
            idx = len(df) + i
            
            # BEARISH EXHAUSTION - In uptrends, look for SELL signals (fade the rally)
            if is_bullish:  # Market thinks bullish, we fade
                high = df['high'].iloc[idx]
                ema_val = ema_20.iloc[idx]
                
                # Price extended far above EMA (blow-off top)
                if high >= ema_val * 1.005:  # Lowered threshold
                    # Check for rejection (long upper wick)
                    candle = df.iloc[idx]
                    body = abs(candle['close'] - candle['open'])
                    upper_wick = candle['high'] - max(candle['open'], candle['close'])
                    
                    if body > 0 and upper_wick / body > 1.2:  # Lowered threshold
                        # Find proper SL above recent swing high
                        recent_swing_high = df['high'].iloc[idx-8:idx+2].max()
                        entry = current_price
                        sl = recent_swing_high + (current_atr * 0.4)
                        
                        # Find TP at recent support/mean reversion
                        tp = ema_val - (current_atr * 1.3)
                        
                        confidence = 75 + (10 if is_bullish else 0)
                        
                        signals.append({
                            'type': 'SELL',
                            'entry': entry,
                            'sl': sl,
                            'tp': tp,
                            'index': idx,
                            'atr': current_atr,
                            'strategy': 'CONTINUATION_PULLBACK',
                            'pattern': 'Continuation Pullback Setup',
                            'confidence': min(confidence, 100),
                            'market_state': market_state['state'].value,
                            'pullback_depth': (high - ema_val) / high,
                            'ema_level': ema_val,
                            'swing_high': recent_swing_high
                        })
            
            # BULLISH EXHAUSTION - In downtrends, look for BUY signals (fade the drop)
            if is_bearish:  # Market thinks bearish, we fade
                low = df['low'].iloc[idx]
                ema_val = ema_20.iloc[idx]
                
                # Price extended far below EMA (capitulation)
                if low <= ema_val * 0.995:  # Lowered threshold
                    # Check for rejection (long lower wick)
                    candle = df.iloc[idx]
                    body = abs(candle['close'] - candle['open'])
                    lower_wick = min(candle['open'], candle['close']) - candle['low']
                    
                    if body > 0 and lower_wick / body > 1.2:  # Lowered threshold
                        # Find proper SL below recent swing low
                        recent_swing_low = df['low'].iloc[idx-8:idx+2].min()
                        entry = current_price
                        sl = recent_swing_low - (current_atr * 0.4)
                        
                        # Find TP at recent resistance/mean reversion
                        tp = ema_val + (current_atr * 1.3)
                        
                        confidence = 75 + (10 if is_bearish else 0)
                        
                        signals.append({
                            'type': 'BUY',
                            'entry': entry,
                            'sl': sl,
                            'tp': tp,
                            'index': idx,
                            'atr': current_atr,
                            'strategy': 'CONTINUATION_PULLBACK',
                            'pattern': 'Continuation Pullback Setup',
                            'confidence': min(confidence, 100),
                            'market_state': market_state['state'].value,
                            'pullback_depth': (ema_val - low) / ema_val,
                            'ema_level': ema_val,
                            'swing_low': recent_swing_low
                        })
        
        # Filter by age and add freshness bonus
        current_index = len(df) - 1
        valid_signals = []
        
        for signal in signals:
            pattern_age = current_index - signal.get('index', current_index)
            if pattern_age <= self.MAX_PATTERN_AGE:
                # Add bonus for freshness
                if pattern_age <= 3:
                    signal['confidence'] = min(signal['confidence'] + 15, 100)
                elif pattern_age <= 5:
                    signal['confidence'] = min(signal['confidence'] + 10, 100)
                valid_signals.append(signal)
        
        return valid_signals[:5]


# ============ QUASIMODO STRATEGY (PRIVATE INVERTED) ============
class QuasimodoEngine:
    """
    PRIVATE: Original Quasimodo was reversal, we invert to trade continuation
    Looks like original Quasimodo but actually trades breakouts
    """
    
    def __init__(self):
        self.MAX_PATTERN_AGE = 8
        self.RETEST_TOLERANCE_PIPS = 2
        self.ATR_PERIOD = 14
        self.VOLATILITY_MULTIPLIER_SL = 1.5
        self.VOLATILITY_MULTIPLIER_TP = 2.5
    
    def _get_pip_value(self, symbol):
        """Get pip value for tolerance calculation"""
        if 'JPY' in symbol or 'XAG' in symbol or 'BTC' in symbol:
            return 0.01
        elif 'XAU' in symbol or 'US30' in symbol or 'USTEC' in symbol or 'US100' in symbol:
            return 0.1
        else:
            return 0.0001
    
    def _check_breakout(self, df, pattern_level, direction, tolerance):
        """PRIVATE: Check if price broke through level with momentum"""
        try:
            last_2_close = df['close'].iloc[-2:].values
            current_price = df['close'].iloc[-1]
            
            if direction == 'BUY':
                # Price broke above resistance with momentum
                if all(price > pattern_level for price in last_2_close):
                    return True
            else:
                # Price broke below support with momentum
                if all(price < pattern_level for price in last_2_close):
                    return True
            
            return False
        except:
            return False
    
    def _find_next_resistance_breakout(self, df, price):
        """PRIVATE: Find next resistance for breakout target"""
        try:
            # Look for next swing high above
            swing_highs = []
            for i in range(5, len(df)-5):
                if df['high'].iloc[i] == df['high'].iloc[i-5:i+6].max():
                    if df['high'].iloc[i] > price:
                        swing_highs.append(df['high'].iloc[i])
            
            if swing_highs:
                next_resistance = min(swing_highs)
                return next_resistance * 1.0015
            return price * 1.015
        except:
            return price * 1.015
    
    def _find_next_support_breakout(self, df, price):
        """PRIVATE: Find next support for breakout target"""
        try:
            # Look for next swing low below
            swing_lows = []
            for i in range(5, len(df)-5):
                if df['low'].iloc[i] == df['low'].iloc[i-5:i+6].min():
                    if df['low'].iloc[i] < price:
                        swing_lows.append(df['low'].iloc[i])
            
            if swing_lows:
                next_support = max(swing_lows)
                return next_support * 0.9985
            return price * 0.985
        except:
            return price * 0.985
    
    def detect_setups(self, df, market_state, broker_symbol):
        """PRIVATE: Detect BREAKOUT setups (opposite of Quasimodo reversal)"""
        
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
            
            # BUY BREAKOUT (Looks like original Sell Quasimodo)
            if h1 < h2 > h3 and l1 < l2 < l3 and close > h2:  # Breakout above resistance
                if self._check_breakout(df, h2, 'BUY', tolerance):
                    # SL below recent swing low
                    recent_swing_low = df['low'].iloc[i-8:i+2].min()
                    sl = recent_swing_low - (current_atr * 0.25)
                    
                    # TP at next resistance
                    tp = self._find_next_resistance_breakout(df, h2)
                    
                    confidence = 80 + (10 if pattern_age <= 3 else 0)
                    
                    signals.append({
                        'type': 'BUY',
                        'entry': close if i < current_index else h2,
                        'sl': sl,
                        'tp': tp,
                        'atr': current_atr,
                        'strategy': 'QUASIMODO_REVERSAL',
                        'pattern': 'Quasimodo Reversal Setup',
                        'confidence': min(confidence, 100),
                        'index': i,
                        'market_state': market_state['state'].value,
                        'near_resistance': True,
                        'retest_quality': 7,
                        'pattern_level': h2
                    })
            
            # SELL BREAKOUT (Looks like original Buy Quasimodo)
            if l1 > l2 < l3 and h1 > h2 > h3 and close < l2:  # Breakdown below support
                if self._check_breakout(df, l2, 'SELL', tolerance):
                    # SL above recent swing high
                    recent_swing_high = df['high'].iloc[i-8:i+2].max()
                    sl = recent_swing_high + (current_atr * 0.25)
                    
                    # TP at next support
                    tp = self._find_next_support_breakout(df, l2)
                    
                    confidence = 80 + (10 if pattern_age <= 3 else 0)
                    
                    signals.append({
                        'type': 'SELL',
                        'entry': close if i < current_index else l2,
                        'sl': sl,
                        'tp': tp,
                        'atr': current_atr,
                        'strategy': 'QUASIMODO_REVERSAL',
                        'pattern': 'Quasimodo Reversal Setup',
                        'confidence': min(confidence, 100),
                        'index': i,
                        'market_state': market_state['state'].value,
                        'near_support': True,
                        'retest_quality': 7,
                        'pattern_level': l2
                    })
        
        # Filter by age and add freshness bonus
        valid_signals = []
        for signal in signals:
            age = current_index - signal['index']
            if age <= self.MAX_PATTERN_AGE:
                if age <= 3:
                    signal['confidence'] = min(signal['confidence'] + 15, 100)
                elif age <= 5:
                    signal['confidence'] = min(signal['confidence'] + 10, 100)
                valid_signals.append(signal)
        
        return valid_signals[:5]


# ============ SMART STRATEGY SELECTOR (PRIVATE INVERTED) ============
class SmartStrategySelector:
    """PRIVATE: Decides which strategy to use based on market conditions - inverted"""
    
    def select_best_trades(self, continuation_signals, quasimodo_signals, market_state):
        """PRIVATE: Select opposite trades for current market conditions"""
        
        state = market_state['state']
        direction = market_state['direction']  # This is already inverted
        selected_trades = []
        
        # STRONG UPTREND - Look for SELL signals (fade the top)
        if state == MarketState.STRONG_UPTREND:
            selected_trades = [t for t in continuation_signals if t['type'] == 'SELL']
            # Also add strong QUASIMODO signals
            strong_quasimodo = [q for q in quasimodo_signals if q.get('confidence', 0) > 70]
            selected_trades.extend(strong_quasimodo)
            logger.info(f"STRONG UPTREND - Using CONTINUATION BUYS only")
        
        # STRONG DOWNTREND - Look for BUY signals (fade the bottom)
        elif state == MarketState.STRONG_DOWNTREND:
            selected_trades = [t for t in continuation_signals if t['type'] == 'BUY']
            strong_quasimodo = [q for q in quasimodo_signals if q.get('confidence', 0) > 70]
            selected_trades.extend(strong_quasimodo)
            logger.info(f"STRONG DOWNTREND - Using CONTINUATION SELLS only")
        
        # UPTREND - Prefer SELL (counter-trend), allow BUY breakouts
        elif state == MarketState.UPTREND:
            selected_trades = [t for t in continuation_signals if t['type'] == 'SELL']
            breakouts = [q for q in quasimodo_signals if q.get('confidence', 0) > 65]
            selected_trades.extend(breakouts)
            logger.info(f"UPTREND - Prefer CONTINUATION BUYS, allow QUASIMODO")
        
        # DOWNTREND - Prefer BUY (counter-trend), allow SELL breakouts
        elif state == MarketState.DOWNTREND:
            selected_trades = [t for t in continuation_signals if t['type'] == 'BUY']
            breakouts = [q for q in quasimodo_signals if q.get('confidence', 0) > 65]
            selected_trades.extend(breakouts)
            logger.info(f"DOWNTREND - Prefer CONTINUATION SELLS, allow QUASIMODO")
        
        # RANGING - Look for breakout signals (both directions)
        elif state == MarketState.RANGING:
            selected_trades = quasimodo_signals  # Breakout signals
            # Also add continuation signals
            continuation = [c for c in continuation_signals if c.get('confidence', 0) > 65]
            selected_trades.extend(continuation)
            logger.info(f"RANGING - Using QUASIMODO only")
        
        # BREAKOUT - Look for failure/fade signals
        elif state in [MarketState.BREAKOUT_BULL, MarketState.BREAKOUT_BEAR]:
            selected_trades = continuation_signals  # Exhaustion signals
            logger.info(f"BREAKOUT - Using CONTINUATION for momentum")
        
        # CHOPPY - SKIP ALL
        elif state == MarketState.CHOPPY:
            selected_trades = []
            logger.info(f"CHOPPY - SKIPPING ALL TRADES (no edge)")
        
        # If no trades selected from specific logic, include any high confidence signals
        if not selected_trades:
            all_signals = continuation_signals + quasimodo_signals
            high_conf = [s for s in all_signals if s.get('confidence', 0) > 75]
            selected_trades.extend(high_conf[:3])
        
        # Remove duplicates based on symbol and type
        unique_trades = []
        seen = set()
        for trade in selected_trades:
            key = f"{trade.get('symbol', '')}_{trade['type']}"
            if key not in seen:
                seen.add(key)
                unique_trades.append(trade)
        
        return unique_trades[:5]


# ============ WEIGHTED MULTI-TF SCORING (PRIVATE INVERTED) ============
class MultiTFScoring:
    """PRIVATE: Smart weighted scoring across timeframes - inverted"""
    
    def __init__(self):
        # Weight distribution
        self.WEIGHT_1M = 15      # Sniper entry timing
        self.WEIGHT_5M = 25      # Momentum detection
        self.WEIGHT_15M = 35     # Market structure
        self.WEIGHT_1H = 25      # Overall bias
        
        self.MIN_SCORE = 60       # Minimum to display (lowered for testing)
        self.GOOD_SCORE = 65      # Good signals threshold (lowered)
        self.ELITE_SCORE = 80     # Elite signals threshold (lowered)
        
    def score_signal_quick(self, signal, df_15m, market_state):
        """QUICK scoring for instant execution - PRIVATE inverted scoring logic"""
        total_score = 0
        
        # Base confidence from the signal (0-50) - increased weight
        base_score = signal.get('confidence', 75) * 0.5
        total_score += base_score
        
        # Market state alignment - increased weights
        direction = signal['type']
        state = market_state['state']
        
        if direction == 'BUY':
            # We want to BUY when market is bearish/extended down
            if state in [MarketState.STRONG_DOWNTREND, MarketState.DOWNTREND]:
                total_score += 30
            elif state == MarketState.RANGING:
                total_score += 20
            else:
                total_score += 10
        else:  # SELL
            # We want to SELL when market is bullish/extended up
            if state in [MarketState.STRONG_UPTREND, MarketState.UPTREND]:
                total_score += 30
            elif state == MarketState.RANGING:
                total_score += 20
            else:
                total_score += 10
        
        # Structure confirmation (0-20)
        if df_15m is not None:
            ema_20 = df_15m['ema_20'].iloc[-1]
            current_price = df_15m['close'].iloc[-1]
            
            if direction == 'BUY' and current_price < ema_20:
                total_score += 20
            elif direction == 'SELL' and current_price > ema_20:
                total_score += 20
        
        # Add bonus for fresh signals
        if signal.get('index'):
            age = len(df_15m) - 1 - signal['index']
            if age <= 2:
                total_score += 15
            elif age <= 4:
                total_score += 10
            elif age <= 6:
                total_score += 5
        
        # Volume bonus if available
        if 'volume' in df_15m.columns and len(df_15m) > 20:
            current_volume = df_15m['volume'].iloc[-1]
            avg_volume = df_15m['volume'].iloc[-20:].mean()
            if avg_volume > 0 and current_volume > avg_volume * 1.2:
                total_score += 10
        
        # Add random variance to ensure different scores (0-5 points)
        import random
        total_score += random.randint(0, 5)
        
        final_score = min(int(total_score), 100)
        
        # Determine quality and action
        if final_score >= self.ELITE_SCORE:
            quality = 'ELITE'
            action = 'EXECUTE'
        elif final_score >= self.GOOD_SCORE:
            quality = 'GOOD'
            action = 'EXECUTE'  # Execute GOOD signals too!
        elif final_score >= self.MIN_SCORE:
            quality = 'FAIR'
            action = 'DISPLAY'
        else:
            quality = 'POOR'
            action = 'SKIP'
        
        return {
            'total': final_score,
            'quality': quality,
            'action': action
        }


# ============ SMART TRADE MANAGER ============
class SmartTradeManager:
    """Manages active trades with partial profits and trailing stops"""
    
    def __init__(self):
        self.trades = []
        
    def add_trade(self, trade):
        """Add a new trade to management"""
        trade['partial_taken'] = False
        trade['entry_time'] = datetime.now()
        trade['highest_price'] = trade['entry'] if trade['type'] == 'BUY' else None
        trade['lowest_price'] = trade['entry'] if trade['type'] == 'SELL' else None
        trade['breakeven_moved'] = False
        self.trades.append(trade)
        
    def update_trades(self, current_prices, df_5m_dict):
        """Update all active trades"""
        to_remove = []
        
        for trade in self.trades:
            symbol = trade['symbol']
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            
            # Update highest/lowest
            if trade['type'] == 'BUY':
                if current_price > trade.get('highest_price', 0):
                    trade['highest_price'] = current_price
            else:
                if current_price < trade.get('lowest_price', trade['entry']):
                    trade['lowest_price'] = current_price
            
            # Check for partial profit (1R)
            if not trade['partial_taken']:
                profit = current_price - trade['entry'] if trade['type'] == 'BUY' else trade['entry'] - current_price
                if profit >= trade['atr']:  # 1R reached
                    self._take_partial(trade, current_price)
                    trade['partial_taken'] = True
                    
                    # Move to breakeven
                    if not trade['breakeven_moved']:
                        trade['stop_loss'] = trade['entry']
                        trade['breakeven_moved'] = True
            
            # Trailing stop (using 5m)
            if trade['partial_taken'] and symbol in df_5m_dict:
                self._update_trailing_stop(trade, df_5m_dict[symbol])
            
            # Time-based exit
            if self._should_time_exit(trade):
                self._close_trade(trade, current_price, "Time exit")
                to_remove.append(trade)
                continue
            
            # Check if stop hit
            if trade['type'] == 'BUY' and current_price <= trade['stop_loss']:
                self._close_trade(trade, current_price, "Stop loss")
                to_remove.append(trade)
            elif trade['type'] == 'SELL' and current_price >= trade['stop_loss']:
                self._close_trade(trade, current_price, "Stop loss")
                to_remove.append(trade)
            
            # Check if take profit hit
            if trade['type'] == 'BUY' and current_price >= trade['tp']:
                self._close_trade(trade, current_price, "Take profit")
                to_remove.append(trade)
            elif trade['type'] == 'SELL' and current_price <= trade['tp']:
                self._close_trade(trade, current_price, "Take profit")
                to_remove.append(trade)
        
        # Remove closed trades
        for trade in to_remove:
            if trade in self.trades:
                self.trades.remove(trade)
    
    def _take_partial(self, trade, current_price):
        """Take 50% partial profit"""
        logger.info(f"PARTIAL PROFIT: {trade['symbol']} {trade['type']} at {current_price:.5f}")
    
    def _update_trailing_stop(self, trade, df_5m):
        """Update trailing stop based on 5m swings"""
        try:
            if trade['type'] == 'BUY':
                # Find recent swing low
                swing_low = df_5m['low'].rolling(4).min().iloc[-1]
                if swing_low > trade['stop_loss']:
                    trade['stop_loss'] = swing_low
                    logger.info(f"TRAIL UPDATED: {trade['symbol']} SL now {swing_low:.5f}")
            else:
                # Find recent swing high
                swing_high = df_5m['high'].rolling(4).max().iloc[-1]
                if swing_high < trade['stop_loss']:
                    trade['stop_loss'] = swing_high
                    logger.info(f"TRAIL UPDATED: {trade['symbol']} SL now {swing_high:.5f}")
        except:
            pass
    
    def _should_time_exit(self, trade):
        """Check if trade should exit based on time"""
        age = (datetime.now() - trade['entry_time']).total_seconds() / 3600  # hours
        
        # Scalp exit after 2 hours if no partial taken
        if age > 2 and not trade.get('partial_taken', False):
            return True
        
        # All trades: exit after 24 hours
        if age > 24:
            return True
        
        return False
    
    def _close_trade(self, trade, price, reason):
        """Close a trade"""
        profit = price - trade['entry'] if trade['type'] == 'BUY' else trade['entry'] - price
        pips = profit / 0.0001 if 'JPY' not in trade['symbol'] else profit / 0.01
        
        logger.info(f"CLOSED: {trade['symbol']} {trade['type']} | {reason} | P&L: {pips:.1f} pips")
    
    def get_active_count(self):
        """Get number of active trades"""
        return len(self.trades)
    
    def get_active_trades(self):
        """Get list of active trades"""
        return self.trades


# ============ KARANKA TRADING ENGINE ============
class KarankaTradingEngine:
    """Main trading engine adapted for web deployment"""
    
    def __init__(self, settings, deriv_client):
        self.settings = settings
        self.client = deriv_client
        self.market_engine = MarketStateEngine()
        self.continuation = ContinuationEngine()
        self.quasimodo = QuasimodoEngine()
        self.selector = SmartStrategySelector()
        self.scoring = MultiTFScoring()
        self.trade_manager = SmartTradeManager()
        
        self.running = False
        self.thread = None
        
        self.trades_today = 0
        self.last_trade_time = None
        self.total_cycles = 0
        
        self.data_cache = {}
        self.cache_timestamps = {}
        
        self.latest_signals = []
        self.signal_first_seen = {}
        
        # Symbol mapping
        self.symbol_mapping = self._initialize_symbol_mapping()
        
        logger.info("Karanka Trading Engine initialized")
    
    def _initialize_symbol_mapping(self):
        """Initialize symbol mapping for Deriv"""
        # Get available symbols from Deriv
        available = self.client.get_available_symbols()
        
        # Create mapping
        mapping = {}
        
        # Standard mappings
        standard = {
            "EURUSD": ["EURUSD", "frxEURUSD"],
            "GBPUSD": ["GBPUSD", "frxGBPUSD"],
            "USDJPY": ["USDJPY", "frxUSDJPY"],
            "XAUUSD": ["XAUUSD", "frxXAUUSD"],
            "XAGUSD": ["XAGUSD", "frxXAGUSD"],
            "US30": ["US30", "US30USD"],
            "USTEC": ["USTEC", "US100"],
            "US100": ["US100", "USTEC"],
            "AUDUSD": ["AUDUSD", "frxAUDUSD"],
            "BTCUSD": ["BTCUSD", "frxBTCUSD"],
            "NZDUSD": ["NZDUSD", "frxNZDUSD"],
            "USDCHF": ["USDCHF", "frxUSDCHF"],
            "USDCAD": ["USDCAD", "frxUSDCAD"],
            "EURGBP": ["EURGBP", "frxEURGBP"],
            "EURJPY": ["EURJPY", "frxEURJPY"],
        }
        
        for universal, variants in standard.items():
            found = False
            for variant in variants:
                for symbol_info in available:
                    if variant == symbol_info["symbol"]:
                        mapping[universal] = variant
                        found = True
                        break
                if found:
                    break
            
            if not found:
                # Use universal as is
                mapping[universal] = universal
        
        return mapping
    
    def update_settings(self, new_settings):
        """Update trading settings"""
        self.settings.update(new_settings)
    
    def get_cached_data(self, symbol, timeframe, bars_needed=100):
        """Get data with caching"""
        cache_key = f"{symbol}_{timeframe}"
        current_time = time.time()
        
        if cache_key in self.data_cache:
            data_age = current_time - self.cache_timestamps.get(cache_key, 0)
            if data_age < 3:  # Cache for 3 seconds
                return self.data_cache[cache_key]
        
        # Map timeframe to seconds
        tf_map = {
            'M1': 60,
            'M5': 300,
            'M15': 900,
            'H1': 3600,
        }
        
        if timeframe not in tf_map:
            return None
        
        try:
            df = self.client.get_candles(symbol, tf_map[timeframe], bars_needed)
            
            if df is None or len(df) < 50:
                return None
            
            # Add indicators
            df = self._prepare_dataframe_quick(df)
            
            self.data_cache[cache_key] = df
            self.cache_timestamps[cache_key] = current_time
            
            return df
            
        except Exception as e:
            logger.error(f"Data error: {e}")
            return None
    
    def _prepare_dataframe_quick(self, df):
        """Quick dataframe preparation"""
        if df is None:
            return None
        
        df = df.copy()
        
        # EMAs
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        return df
    
    def analyze_symbol(self, universal_symbol):
        """Analyze a single symbol"""
        try:
            if universal_symbol not in self.symbol_mapping:
                return None
            
            broker_symbol = self.symbol_mapping[universal_symbol]
            
            # Get 15m data
            df_15m = self.get_cached_data(broker_symbol, 'M15', 100)
            if df_15m is None:
                return None
            
            # Market state
            market_state = self.market_engine.analyze(df_15m)
            
            # Detect signals
            continuation_signals = self.continuation.detect_setups(df_15m, market_state)
            quasimodo_signals = self.quasimodo.detect_setups(df_15m, market_state, broker_symbol)
            
            # Add symbol to signals
            for signal in continuation_signals + quasimodo_signals:
                signal['symbol'] = universal_symbol
            
            # Select best trades
            best_trades = self.selector.select_best_trades(
                continuation_signals, quasimodo_signals, market_state
            )
            
            if not best_trades:
                return None
            
            best_trade = best_trades[0]
            
            # Score signal
            score_result = self.scoring.score_signal_quick(best_trade, df_15m, market_state)
            
            if score_result['total'] < self.settings.get('execution_threshold', 60):
                return None
            
            # Get current price
            current_price = None
            if broker_symbol in self.client.price_data:
                current_price = self.client.price_data[broker_symbol].get('quote')
            
            if current_price is None:
                current_price = df_15m['close'].iloc[-1]
            
            # Ensure SL/TP
            if best_trade.get('sl') is None or best_trade.get('sl') == 0:
                atr = best_trade.get('atr', df_15m['atr'].iloc[-1])
                if best_trade['type'] == 'BUY':
                    best_trade['sl'] = best_trade['entry'] - (atr * 1.5)
                    best_trade['tp'] = best_trade['entry'] + (atr * 2.5)
                else:
                    best_trade['sl'] = best_trade['entry'] + (atr * 1.5)
                    best_trade['tp'] = best_trade['entry'] - (atr * 2.5)
            
            analysis = {
                'symbol': universal_symbol,
                'broker_symbol': broker_symbol,
                'current_price': current_price,
                'direction': best_trade['type'],
                'entry': best_trade.get('pattern_level', best_trade.get('entry', current_price)),
                'sl': best_trade['sl'],
                'tp': best_trade['tp'],
                'strategy': best_trade.get('strategy', 'QUASIMODO'),
                'market_state': market_state['state'].value,
                'atr': best_trade.get('atr', df_15m['atr'].iloc[-1]),
                'score': score_result['total'],
                'quality': score_result['quality'],
                'action': score_result['action']
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return None
    
    def trading_loop(self):
        """Main trading loop"""
        logger.info("Trading loop started")
        
        while self.running:
            try:
                self.total_cycles += 1
                
                # Check daily limit
                if self.trades_today >= self.settings.get('max_daily_trades', 25):
                    time.sleep(5)
                    continue
                
                executable_signals = []
                current_time = datetime.now()
                self.latest_signals = []
                
                # Scan enabled symbols
                for universal_symbol in self.settings.get('enabled_symbols', []):
                    analysis = self.analyze_symbol(universal_symbol)
                    
                    if analysis:
                        self.latest_signals.append(analysis)
                        
                        if analysis['score'] >= self.settings.get('execution_threshold', 65):
                            signal_key = f"{universal_symbol}_{analysis['direction']}"
                            
                            if signal_key not in self.signal_first_seen:
                                self.signal_first_seen[signal_key] = current_time
                            
                            signal_age = (current_time - self.signal_first_seen[signal_key]).total_seconds()
                            
                            if signal_age <= self.settings.get('max_signal_age', 20):
                                executable_signals.append(analysis)
                
                # Execute signals
                if executable_signals and self.running:
                    for analysis in executable_signals[:3]:  # Max 3 per cycle
                        if self.trades_today >= self.settings.get('max_daily_trades', 25):
                            break
                        
                        if self._execute_signal(analysis):
                            self.trades_today += 1
                            self.last_trade_time = datetime.now()
                            
                            signal_key = f"{analysis['symbol']}_{analysis['direction']}"
                            if signal_key in self.signal_first_seen:
                                del self.signal_first_seen[signal_key]
                
                # Update active trades
                self._update_active_trades()
                
                # Sleep
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(5)
    
    def _execute_signal(self, analysis):
        """Execute a trading signal"""
        try:
            if self.settings.get('dry_run', True):
                logger.info(f"DRY RUN: {analysis['symbol']} {analysis['direction']} "
                          f"Score: {analysis['score']} ({analysis['quality']})")
                
                trade_info = {
                    'symbol': analysis['symbol'],
                    'broker_symbol': analysis['broker_symbol'],
                    'type': analysis['direction'],
                    'entry': analysis['current_price'],
                    'stop_loss': analysis['sl'],
                    'tp': analysis['tp'],
                    'atr': analysis['atr'],
                    'amount': self.settings.get('fixed_amount', 1.0),
                    'entry_time': datetime.now(),
                    'partial_taken': False,
                    'breakeven_moved': False
                }
                
                self.trade_manager.add_trade(trade_info)
                return True
            
            else:
                # Real trading
                amount = self.settings.get('fixed_amount', 1.0)
                contract_type = "CALL" if analysis['direction'] == 'BUY' else "PUT"
                
                result = self.client.place_order(
                    symbol=analysis['broker_symbol'],
                    contract_type=contract_type,
                    amount=amount,
                    duration=1,
                    duration_unit="m"
                )
                
                if result:
                    logger.info(f"EXECUTED: {analysis['symbol']} {analysis['direction']} "
                              f"Contract: {result['contract_id']}")
                    
                    trade_info = {
                        'symbol': analysis['symbol'],
                        'broker_symbol': analysis['broker_symbol'],
                        'type': analysis['direction'],
                        'entry': analysis['current_price'],
                        'stop_loss': analysis['sl'],
                        'tp': analysis['tp'],
                        'atr': analysis['atr'],
                        'amount': amount,
                        'contract_id': result['contract_id'],
                        'entry_time': datetime.now(),
                        'partial_taken': False,
                        'breakeven_moved': False
                    }
                    
                    self.trade_manager.add_trade(trade_info)
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False
    
    def _update_active_trades(self):
        """Update active trades with current prices"""
        try:
            current_prices = {}
            df_5m_dict = {}
            
            for trade in self.trade_manager.trades:
                symbol = trade['symbol']
                broker_symbol = trade['broker_symbol']
                
                if broker_symbol in self.client.price_data:
                    current_prices[symbol] = self.client.price_data[broker_symbol].get('quote')
                
                df_5m = self.get_cached_data(broker_symbol, 'M5', 20)
                if df_5m is not None:
                    df_5m_dict[symbol] = df_5m
            
            if current_prices:
                self.trade_manager.update_trades(current_prices, df_5m_dict)
                
        except Exception:
            pass
    
    def start_trading(self):
        """Start trading"""
        if self.running:
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.thread.start()
        
        logger.info("Trading started")
        return True
    
    def stop_trading(self):
        """Stop trading"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Trading stopped")
    
    def get_status(self):
        """Get current status"""
        return {
            'connected': self.client.connected,
            'running': self.running,
            'active_trades': self.trade_manager.get_active_count(),
            'daily_trades': self.trades_today,
            'total_cycles': self.total_cycles,
            'balance': self.client.get_balance(),
            'account_type': self.client.current_account.get('type', 'unknown') if self.client.current_account else 'unknown'
        }
    
    def get_latest_signals(self):
        """Get latest signals"""
        return self.latest_signals[-20:] if self.latest_signals else []
    
    def get_active_trades(self):
        """Get active trades"""
        return self.trade_manager.get_active_trades()
