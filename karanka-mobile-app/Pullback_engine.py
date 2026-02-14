# pullback_engine.py - Pullback Continuation Strategy
import pandas as pd
import numpy as np

class PullbackEngine:
    """
    PULLBACK STRATEGY - For trending markets
    """
    
    def __init__(self):
        self.ema_fast = 20
        self.ema_slow = 50
        self.rsi_period = 14
        self.rsi_oversold = 40
        self.rsi_overbought = 60
        
    def detect_pullback(self, df_m15, df_h1, regime, symbol):
        """
        Detect pullback setups in trending markets
        """
        try:
            if df_m15 is None or len(df_m15) < 50:
                return None
            
            # Only look for pullbacks in trending markets
            if regime['market_type'] != 'TRENDING' and regime['trend_strength'] != 'STRONG':
                confidence_multiplier = 0.7
            else:
                confidence_multiplier = 1.0
            
            # Calculate EMAs
            ema_fast = df_m15['close'].ewm(span=self.ema_fast).mean()
            ema_slow = df_m15['close'].ewm(span=self.ema_slow).mean()
            
            # Calculate RSI (simplified)
            close_prices = df_m15['close'].values
            rsi = self.calculate_rsi(close_prices)
            
            current_price = df_m15['close'].iloc[-1]
            current_ema_fast = ema_fast.iloc[-1]
            current_ema_slow = ema_slow.iloc[-1]
            current_rsi = rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) else 50
            
            # BULLISH PULLBACK (in uptrend)
            if regime['primary_trend'] in ['BULLISH', 'BULLISH_WEAK']:
                # Price near EMAs (pullback)
                price_to_ema = abs(current_price - current_ema_fast) / current_ema_fast if current_ema_fast > 0 else 1
                
                if price_to_ema < 0.003:  # Within 0.3% of EMA
                    # RSI showing oversold/neutral
                    if current_rsi < 50:
                        # Check for bullish reversal candle
                        if self.is_bullish_candle(df_m15):
                            atr = self.calculate_atr(df_m15).iloc[-1]
                            
                            return {
                                'type': 'BUY',
                                'strategy': 'PULLBACK',
                                'entry': current_price,
                                'sl': min(df_m15['low'].iloc[-5:]) - (0.3 * atr),
                                'tp': current_price + (2.0 * atr),
                                'confidence_base': 65 * confidence_multiplier,
                                'reason': f"Pullback to EMA{self.ema_fast}",
                                'symbol': symbol
                            }
            
            # BEARISH PULLBACK (in downtrend)
            elif regime['primary_trend'] in ['BEARISH', 'BEARISH_WEAK']:
                price_to_ema = abs(current_price - current_ema_fast) / current_ema_fast if current_ema_fast > 0 else 1
                
                if price_to_ema < 0.003:
                    if current_rsi > 50:
                        if self.is_bearish_candle(df_m15):
                            atr = self.calculate_atr(df_m15).iloc[-1]
                            
                            return {
                                'type': 'SELL',
                                'strategy': 'PULLBACK',
                                'entry': current_price,
                                'sl': max(df_m15['high'].iloc[-5:]) + (0.3 * atr),
                                'tp': current_price - (2.0 * atr),
                                'confidence_base': 65 * confidence_multiplier,
                                'reason': f"Pullback to EMA{self.ema_fast}",
                                'symbol': symbol
                            }
            
            return None
            
        except Exception as e:
            return None
    
    def is_bullish_candle(self, df):
        """Check for bullish candle patterns"""
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Bullish engulfing
            if (last['close'] > last['open'] and
                prev['close'] < prev['open'] and
                last['close'] > prev['open'] and
                last['open'] < prev['close']):
                return True
            
            # Hammer
            body = abs(last['close'] - last['open'])
            lower_wick = min(last['open'], last['close']) - last['low']
            if lower_wick > body * 1.5 and last['close'] > last['open']:
                return True
            
            # Green candle after red
            if last['close'] > last['open'] and prev['close'] < prev['open']:
                return True
            
            return False
        except:
            return False
    
    def is_bearish_candle(self, df):
        """Check for bearish candle patterns"""
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Bearish engulfing
            if (last['close'] < last['open'] and
                prev['close'] > prev['open'] and
                last['close'] < prev['open'] and
                last['open'] > prev['close']):
                return True
            
            # Shooting star
            body = abs(last['close'] - last['open'])
            upper_wick = last['high'] - max(last['open'], last['close'])
            if upper_wick > body * 1.5 and last['close'] < last['open']:
                return True
            
            # Red candle after green
            if last['close'] < last['open'] and prev['close'] > prev['open']:
                return True
            
            return False
        except:
            return False
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        try:
            deltas = np.diff(prices)
            seed = deltas[:period+1]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            rs = up / down if down != 0 else 100
            rsi = np.zeros_like(prices)
            rsi[:period] = 100 - 100 / (1 + rs)
            
            for i in range(period, len(prices)):
                delta = deltas[i-1]
                if delta > 0:
                    upval = delta
                    downval = 0
                else:
                    upval = 0
                    downval = -delta
                
                up = (up * (period-1) + upval) / period
                down = (down * (period-1) + downval) / period
                rs = up / down if down != 0 else 100
                rsi[i] = 100 - 100 / (1 + rs)
            
            return rsi
        except:
            return np.array([50] * len(prices))
    
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