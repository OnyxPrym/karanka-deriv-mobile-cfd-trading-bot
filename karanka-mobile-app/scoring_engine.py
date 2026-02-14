# scoring_engine.py - Signal Scoring System
import numpy as np

class ScoringEngine:
    """
    SCORES ALL SIGNALS OBJECTIVELY
    """
    
    def __init__(self):
        self.min_score_to_trade = 65
        
    def score_signal(self, signal, regime, df_h1, df_m15, df_5m=None):
        """
        Score any signal from 0-100
        Returns score and reasons
        """
        score = 0
        reasons = []
        
        try:
            # 1. Regime alignment (0-20)
            regime_score = self.score_regime_alignment(signal, regime)
            score += regime_score
            if regime_score > 15:
                reasons.append("Perfect regime fit")
            elif regime_score > 10:
                reasons.append("Good regime context")
            
            # 2. Momentum confirmation (0-20)
            momentum_score = self.score_momentum(signal, df_m15)
            score += momentum_score
            if momentum_score > 15:
                reasons.append("Strong momentum")
            
            # 3. Structure quality (0-25)
            structure_score = self.score_structure(signal, df_m15, df_h1)
            score += structure_score
            if structure_score > 20:
                reasons.append("Clean structure")
            elif structure_score > 15:
                reasons.append("Decent structure")
            
            # 4. Volatility conditions (0-15)
            vol_score = self.score_volatility(signal, df_m15, regime)
            score += vol_score
            if vol_score > 10:
                reasons.append("Good volatility")
            
            # 5. Risk/Reward quality (0-20)
            rr_score = self.score_risk_reward(signal)
            score += rr_score
            if rr_score > 15:
                reasons.append("Excellent R:R")
            
            # Add strategy-specific bonuses
            if signal['strategy'] == 'QUASIMODO':
                if signal.get('pattern_age', 10) <= 4:
                    score += 5
                    reasons.append("Fresh pattern")
            
            elif signal['strategy'] == 'BREAKOUT':
                if regime.get('compression_level', 0) > 70:
                    score += 10
                    reasons.append("Compression breakout")
            
        except Exception as e:
            print(f"Scoring error: {e}")
        
        # Cap at 100
        final_score = min(score, 100)
        
        return {
            'score': final_score,
            'reasons': reasons[:3],
            'should_trade': final_score >= self.min_score_to_trade
        }
    
    def score_regime_alignment(self, signal, regime):
        """Score how well signal fits current regime"""
        score = 0
        
        # Base alignment
        if signal['strategy'] == 'QUASIMODO':
            if regime['market_type'] in ['REVERSAL', 'RANGING']:
                score += 15
            elif regime['market_type'] == 'TRENDING' and regime['trend_strength'] == 'WEAK':
                score += 10
            else:
                score += 5
        
        elif signal['strategy'] == 'PULLBACK':
            if regime['market_type'] == 'TRENDING':
                score += 15
                if regime['trend_strength'] == 'STRONG':
                    score += 5
            elif regime['market_type'] == 'RANGING':
                score += 8
        
        elif signal['strategy'] == 'BREAKOUT':
            if regime['market_type'] in ['RANGING', 'COMPRESSION']:
                score += 18
            elif regime['volatility'] == 'LOW':
                score += 10
        
        # Direction alignment
        if signal['type'] == 'BUY' and regime['primary_trend'] in ['BULLISH', 'BULLISH_WEAK']:
            score += 5
        elif signal['type'] == 'SELL' and regime['primary_trend'] in ['BEARISH', 'BEARISH_WEAK']:
            score += 5
        
        return min(score, 20)
    
    def score_momentum(self, signal, df):
        """Score momentum alignment"""
        score = 0
        
        try:
            if df is None or len(df) < 20:
                return 10
            
            # EMA alignment
            ema20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema50 = df['close'].ewm(span=50).mean().iloc[-1]
            current = df['close'].iloc[-1]
            
            if signal['type'] == 'BUY':
                if current > ema20:
                    score += 10
                if ema20 > ema50:
                    score += 10
            else:
                if current < ema20:
                    score += 10
                if ema20 < ema50:
                    score += 10
            
        except:
            pass
        
        return min(score, 20)
    
    def score_structure(self, signal, df_m15, df_h1):
        """Score structure quality"""
        score = 0
        
        try:
            if signal['strategy'] == 'QUASIMODO':
                # Pattern age (fresher is better)
                age = signal.get('pattern_age', 10)
                if age <= 3:
                    score += 15
                elif age <= 5:
                    score += 10
                elif age <= 8:
                    score += 5
                
                # Pattern clarity
                if 'pattern_high' in signal and 'pattern_low' in signal:
                    pattern_height = abs(signal['pattern_high'] - signal['pattern_low'])
                    atr = signal.get('atr', 0)
                    if atr > 0 and pattern_height > atr * 0.5:
                        score += 10
            
            elif signal['strategy'] == 'PULLBACK':
                # Distance to EMA
                if df_m15 is not None:
                    ema20 = df_m15['close'].ewm(span=20).mean().iloc[-1]
                    current = df_m15['close'].iloc[-1]
                    distance = abs(current - ema20) / ema20 if ema20 > 0 else 1
                    
                    if distance < 0.001:
                        score += 20
                    elif distance < 0.002:
                        score += 15
                    elif distance < 0.003:
                        score += 10
            
            elif signal['strategy'] == 'BREAKOUT':
                # Breakout strength
                if 'breakout_level' in signal:
                    current = signal['entry']
                    level = signal['breakout_level']
                    distance = abs(current - level) / level if level > 0 else 0
                    
                    if distance > 0.002:
                        score += 20
                    elif distance > 0.001:
                        score += 15
                    else:
                        score += 10
            
            # HTF confirmation
            if df_h1 is not None and len(df_h1) > 200:
                ema200_h1 = df_h1['close'].rolling(200).mean().iloc[-1]
                current_h1 = df_h1['close'].iloc[-1]
                
                if signal['type'] == 'BUY' and current_h1 > ema200_h1:
                    score += 5
                elif signal['type'] == 'SELL' and current_h1 < ema200_h1:
                    score += 5
            
        except:
            pass
        
        return min(score, 25)
    
    def score_volatility(self, signal, df, regime):
        """Score volatility conditions"""
        score = 0
        
        try:
            if df is None:
                return 5
            
            # ATR conditions (simplified)
            high_low = df['high'] - df['low']
            avg_range = high_low.iloc[-20:].mean()
            current_range = high_low.iloc[-5:].mean()
            
            if avg_range > 0:
                range_ratio = current_range / avg_range
                
                if 0.8 <= range_ratio <= 1.2:
                    score += 10
                elif 0.6 <= range_ratio <= 1.4:
                    score += 5
            
            # Volatility regime
            if regime['volatility'] == 'NORMAL':
                score += 5
            elif regime['volatility'] == 'HIGH' and signal['strategy'] == 'BREAKOUT':
                score += 5
            
        except:
            pass
        
        return min(score, 15)
    
    def score_risk_reward(self, signal):
        """Score risk/reward ratio"""
        score = 0
        
        try:
            entry = signal['entry']
            sl = signal['sl']
            tp = signal['tp']
            
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            
            if risk > 0:
                rr_ratio = reward / risk
                
                if rr_ratio >= 2.5:
                    score += 20
                elif rr_ratio >= 2.0:
                    score += 15
                elif rr_ratio >= 1.5:
                    score += 10
                elif rr_ratio >= 1.0:
                    score += 5
        except:
            pass
        
        return min(score, 20)