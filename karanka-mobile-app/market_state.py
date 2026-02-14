# market_state.py - Market State Engine
import pandas as pd
import numpy as np

class MarketStateEngine:
    """
    Determines market regime - TELLS US WHAT KIND OF MARKET WE'RE IN
    """
    
    def __init__(self):
        self.ema_periods = [20, 50, 200]
        self.adx_period = 14
        self.atr_period = 14
        
    def detect_regime(self, df_h1, df_m15, df_m5=None):
        """
        Detect market regime
        Returns: dict with market context
        """
        regime = {
            'primary_trend': 'NEUTRAL',
            'trend_strength': 'WEAK',
            'volatility': 'NORMAL',
            'market_type': 'RANGING',
            'compression_level': 0,
            'momentum': 'NEUTRAL',
            'confidence': 50
        }
        
        try:
            if df_h1 is None or len(df_h1) < 50:
                return regime
            
            # 1. TREND DIRECTION (EMA200 on H1)
            ema200 = df_h1['close'].rolling(200).mean().iloc[-1] if len(df_h1) >= 200 else df_h1['close'].mean()
            ema50 = df_h1['close'].rolling(50).mean().iloc[-1] if len(df_h1) >= 50 else df_h1['close'].mean()
            current_price = df_h1['close'].iloc[-1]
            
            # Price relative to EMAs
            if current_price > ema200 * 1.005:
                regime['primary_trend'] = 'BULLISH'
            elif current_price < ema200 * 0.995:
                regime['primary_trend'] = 'BEARISH'
            elif current_price > ema200:
                regime['primary_trend'] = 'BULLISH_WEAK'
            elif current_price < ema200:
                regime['primary_trend'] = 'BEARISH_WEAK'
            
            # 2. TREND STRENGTH (ADX approximation)
            if df_m15 is not None and len(df_m15) > 30:
                # Simplified ADX using price movement
                highs = df_m15['high'].values[-20:]
                lows = df_m15['low'].values[-20:]
                
                # Calculate directional movement
                up_move = np.diff(highs)
                down_move = -np.diff(lows)
                
                avg_up = np.mean([max(0, x) for x in up_move[-14:]])
                avg_down = np.mean([max(0, x) for x in down_move[-14:]])
                
                if avg_up + avg_down > 0:
                    adx_value = 100 * abs(avg_up - avg_down) / (avg_up + avg_down)
                    
                    if adx_value > 30:
                        regime['trend_strength'] = 'STRONG'
                        regime['market_type'] = 'TRENDING'
                    elif adx_value > 22:
                        regime['trend_strength'] = 'MODERATE'
                        regime['market_type'] = 'TRENDING'
                    else:
                        regime['trend_strength'] = 'WEAK'
            
            # 3. VOLATILITY & COMPRESSION
            if df_m15 is not None and len(df_m15) > 50:
                # Calculate ATR manually
                high_low = df_m15['high'] - df_m15['low']
                tr = high_low.rolling(14).mean()
                current_atr = tr.iloc[-1] if not pd.isna(tr.iloc[-1]) else 0
                atr_mean = tr.iloc[-20:].mean() if len(tr) > 20 else current_atr
                
                if atr_mean > 0:
                    atr_ratio = current_atr / atr_mean
                    
                    if atr_ratio > 1.3:
                        regime['volatility'] = 'HIGH'
                    elif atr_ratio < 0.7:
                        regime['volatility'] = 'LOW'
                    else:
                        regime['volatility'] = 'NORMAL'
                
                # Compression detection
                recent_high = df_m15['high'].iloc[-20:].max()
                recent_low = df_m15['low'].iloc[-20:].min()
                recent_range = recent_high - recent_low
                
                avg_range = (df_m15['high'] - df_m15['low']).iloc[-50:].mean()
                
                if avg_range > 0:
                    range_ratio = recent_range / avg_range
                    
                    if range_ratio < 0.6:
                        regime['compression_level'] = 80
                        if regime['market_type'] == 'RANGING':
                            regime['market_type'] = 'COMPRESSION'
                    elif range_ratio < 0.8:
                        regime['compression_level'] = 50
                    else:
                        regime['compression_level'] = 20
            
            # 4. Check for structure break (potential reversal)
            if self.detect_structure_break(df_m15):
                if regime['market_type'] != 'TRENDING' or regime['trend_strength'] == 'WEAK':
                    regime['market_type'] = 'REVERSAL'
            
            # 5. Momentum
            if df_m15 is not None and len(df_m15) > 20:
                ema_fast = df_m15['close'].ewm(span=8).mean().iloc[-1]
                ema_slow = df_m15['close'].ewm(span=20).mean().iloc[-1]
                
                if ema_fast > ema_slow * 1.001:
                    regime['momentum'] = 'BULLISH'
                elif ema_fast < ema_slow * 0.999:
                    regime['momentum'] = 'BEARISH'
            
            # 6. Confidence score
            regime['confidence'] = self.calculate_confidence(regime)
            
            return regime
            
        except Exception as e:
            print(f"Regime detection error: {e}")
            return regime
    
    def detect_structure_break(self, df):
        """Detect if market structure has broken"""
        try:
            if df is None or len(df) < 30:
                return False
            
            # Find recent swing points
            highs = df['high'].values[-30:]
            lows = df['low'].values[-30:]
            
            swing_highs = []
            swing_lows = []
            
            for i in range(2, len(highs)-2):
                if highs[i] == max(highs[i-2:i+3]):
                    swing_highs.append((i, highs[i]))
                if lows[i] == min(lows[i-2:i+3]):
                    swing_lows.append((i, lows[i]))
            
            if len(swing_highs) < 2 or len(swing_lows) < 2:
                return False
            
            last_high = swing_highs[-1][1] if swing_highs else None
            prev_high = swing_highs[-2][1] if len(swing_highs) >= 2 else None
            last_low = swing_lows[-1][1] if swing_lows else None
            prev_low = swing_lows[-2][1] if len(swing_lows) >= 2 else None
            
            if last_high and prev_high and last_low and prev_low:
                # Bearish break: lower high + lower low
                if last_high < prev_high and last_low < prev_low:
                    return True
                # Bullish break: higher high + higher low
                if last_high > prev_high and last_low > prev_low:
                    return True
            
            return False
            
        except:
            return False
    
    def calculate_confidence(self, regime):
        """Calculate confidence in regime detection"""
        confidence = 50
        
        if regime['trend_strength'] == 'STRONG':
            confidence += 20
        elif regime['trend_strength'] == 'MODERATE':
            confidence += 10
        
        if regime['volatility'] == 'NORMAL':
            confidence += 10
        
        if regime['compression_level'] > 70:
            confidence += 10
        
        return min(confidence, 100)