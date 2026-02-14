# quasimodo_engine.py - Quasimodo Pattern Detection
import pandas as pd
import numpy as np

class QuasimodoEngine:
    """
    PURE QUASIMODO - Reversal Pattern Specialist
    """
    
    def __init__(self):
        self.ATR_PERIOD = 14
        self.VOLATILITY_MULTIPLIER_SL = 1.5
        self.VOLATILITY_MULTIPLIER_TP = 2.5
        self.MAX_PATTERN_AGE = 8
        self.RETEST_TOLERANCE_PIPS = 2
        
    def calculate_atr(self, df):
        """Calculate Average True Range"""
        try:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(self.ATR_PERIOD).mean()
            return atr
        except:
            return pd.Series([0] * len(df))
    
    def get_pip_value(self, symbol):
        """Get pip value for tolerance calculation"""
        if 'JPY' in symbol or 'XAG' in symbol:
            return 0.01
        elif 'XAU' in symbol or 'US30' in symbol or 'USTEC' in symbol:
            return 0.1
        elif 'BTC' in symbol:
            return 1.0
        else:
            return 0.0001
    
    def detect_quasimodo(self, df, symbol, timeframe='15M'):
        """
        DETECT QUASIMODO PATTERNS - Returns ALL patterns found
        """
        if df is None or len(df) < 50:
            return []
        
        signals = []
        atr_series = self.calculate_atr(df)
        current_index = len(df) - 1
        
        for i in range(3, len(df)-1):
            h1 = df['high'].iloc[i-3]
            h2 = df['high'].iloc[i-2]
            h3 = df['high'].iloc[i-1]
            l1 = df['low'].iloc[i-3]
            l2 = df['low'].iloc[i-2]
            l3 = df['low'].iloc[i-1]
            close = df['close'].iloc[i]
            atr = atr_series.iloc[i] if not pd.isna(atr_series.iloc[i]) else 0
            
            pattern_age = current_index - i
            if pattern_age > self.MAX_PATTERN_AGE:
                continue
            
            # SELL QUASIMODO (Bearish)
            if h1 < h2 > h3 and l1 < l2 < l3 and close < h2:
                entry = h2
                sl = entry + (self.VOLATILITY_MULTIPLIER_SL * atr)
                tp = entry - (self.VOLATILITY_MULTIPLIER_TP * atr)
                
                signals.append({
                    'type': 'SELL',
                    'timeframe': timeframe,
                    'symbol': symbol,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'pattern_high': h2,
                    'pattern_low': l2,
                    'atr': atr,
                    'pattern_age': pattern_age,
                    'index': i,
                    'strategy': 'QUASIMODO',
                    'confidence_base': 70
                })
            
            # BUY QUASIMODO (Bullish)
            if l1 > l2 < l3 and h1 > h2 > h3 and close > l2:
                entry = l2
                sl = entry - (self.VOLATILITY_MULTIPLIER_SL * atr)
                tp = entry + (self.VOLATILITY_MULTIPLIER_TP * atr)
                
                signals.append({
                    'type': 'BUY',
                    'timeframe': timeframe,
                    'symbol': symbol,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'pattern_high': h2,
                    'pattern_low': l2,
                    'atr': atr,
                    'pattern_age': pattern_age,
                    'index': i,
                    'strategy': 'QUASIMODO',
                    'confidence_base': 70
                })
        
        return signals
    
    def check_retest(self, df_current, signal, symbol):
        """
        Check if price retested the pattern within 2 pips
        Returns: (retested, retest_price)
        """
        try:
            if df_current is None or len(df_current) < 12:
                return False, 0
            
            current_price = df_current['close'].iloc[-1]
            
            pip_value = self.get_pip_value(symbol)
            tolerance = self.RETEST_TOLERANCE_PIPS * pip_value
            
            last_12_low = df_current['low'].iloc[-12:].min()
            last_12_high = df_current['high'].iloc[-12:].max()
            
            if signal['type'] == 'BUY':
                pattern_level = signal['pattern_low']
                if last_12_low <= (pattern_level + tolerance) and current_price > pattern_level:
                    last_8_low = df_current['low'].iloc[-8:].min()
                    if last_8_low <= (pattern_level + tolerance):
                        return True, current_price
                    
            elif signal['type'] == 'SELL':
                pattern_level = signal['pattern_high']
                if last_12_high >= (pattern_level - tolerance) and current_price < pattern_level:
                    last_8_high = df_current['high'].iloc[-8:].max()
                    if last_8_high >= (pattern_level - tolerance):
                        return True, current_price
            
            return False, 0
            
        except Exception as e:
            return False, 0