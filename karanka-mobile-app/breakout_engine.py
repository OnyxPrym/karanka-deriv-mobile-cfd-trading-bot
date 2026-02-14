# breakout_engine.py - Breakout Detection
import pandas as pd
import numpy as np

class BreakoutEngine:
    """
    BREAKOUT STRATEGY - For ranging/compressed markets
    """
    
    def __init__(self):
        self.range_period = 20
        self.breakout_confirmation_bars = 2
        
    def detect_breakout(self, df_5m, df_15m, regime, symbol):
        """
        Detect breakouts in ranging/compressed markets
        """
        try:
            if df_5m is None or df_15m is None or len(df_5m) < 30 or len(df_15m) < 30:
                return None
            
            # Best in ranging/compressed markets
            if regime['market_type'] not in ['RANGING', 'COMPRESSION']:
                confidence_multiplier = 0.6
            else:
                confidence_multiplier = 1.0
            
            # Calculate range on 15M
            range_high = df_15m['high'].iloc[-self.range_period:].max()
            range_low = df_15m['low'].iloc[-self.range_period:].min()
            range_size = range_high - range_low
            
            current_price = df_5m['close'].iloc[-1]
            
            # Check for breakout
            if current_price > range_high:
                # Upside breakout - need confirmation
                return self.confirm_breakout(
                    df_5m, 'BUY', range_high, range_size, 
                    confidence_multiplier, symbol
                )
                
            elif current_price < range_low:
                # Downside breakout
                return self.confirm_breakout(
                    df_5m, 'SELL', range_low, range_size,
                    confidence_multiplier, symbol
                )
            
            return None
            
        except Exception as e:
            return None
    
    def confirm_breakout(self, df_5m, direction, breakout_level, range_size, confidence_multiplier, symbol):
        """Confirm breakout with price action"""
        try:
            # Need at least 2 bars beyond the level
            if direction == 'BUY':
                bars_above = sum(df_5m['close'].iloc[-3:] > breakout_level)
                if bars_above >= 2:
                    # Calculate ATR for targets
                    atr = self.calculate_atr(df_5m).iloc[-1]
                    
                    # Entry is current price
                    entry = df_5m['close'].iloc[-1]
                    
                    # SL just below breakout level or recent low
                    recent_low = df_5m['low'].iloc[-5:].min()
                    sl = min(breakout_level - (0.2 * range_size), recent_low - (0.3 * atr))
                    
                    # TP based on range projection
                    tp = entry + (range_size * 0.8)
                    
                    return {
                        'type': 'BUY',
                        'strategy': 'BREAKOUT',
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'confidence_base': 70 * confidence_multiplier,
                        'reason': f"Breakout above {breakout_level:.5f}",
                        'breakout_level': breakout_level,
                        'symbol': symbol
                    }
            
            else:  # SELL
                bars_below = sum(df_5m['close'].iloc[-3:] < breakout_level)
                if bars_below >= 2:
                    atr = self.calculate_atr(df_5m).iloc[-1]
                    entry = df_5m['close'].iloc[-1]
                    
                    recent_high = df_5m['high'].iloc[-5:].max()
                    sl = max(breakout_level + (0.2 * range_size), recent_high + (0.3 * atr))
                    
                    tp = entry - (range_size * 0.8)
                    
                    return {
                        'type': 'SELL',
                        'strategy': 'BREAKOUT',
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'confidence_base': 70 * confidence_multiplier,
                        'reason': f"Breakout below {breakout_level:.5f}",
                        'breakout_level': breakout_level,
                        'symbol': symbol
                    }
            
            return None
            
        except:
            return None
    
    def calculate_atr(self, df, period=14):
        """Calculate ATR"""
        try:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            return atr
        except:
            return pd.Series([0] * len(df))