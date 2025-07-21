"""
Filtre de tendance global pour favoriser les BUY en tendance haussière
Combine EMA longue, ADX, pente de prix et structure de marché
"""
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class TrendFilter:
    """
    Filtre de tendance pour adapter les signaux selon la direction du marché
    - Favorise les BUY en tendance haussière
    - Réduit les SELL en tendance haussière forte
    - Utilise EMA 50/100/200, ADX, pente et structure HH/HL
    """
    
    def __init__(self):
        self.ema_fast = 50   # EMA rapide pour tendance court terme
        self.ema_slow = 100  # EMA lente pour tendance moyen terme
        self.ema_long = 200  # EMA très lente pour tendance long terme
        self.adx_trend_threshold = 25  # ADX minimum pour tendance confirmée
        self.slope_threshold = 0.0001  # Pente minimum pour tendance
        
    def analyze_trend(self, df: pd.DataFrame, indicators: Dict) -> Dict[str, any]:
        """
        Analyse complète de la tendance
        
        Args:
            df: DataFrame avec données OHLCV  
            indicators: Indicateurs pré-calculés
            
        Returns:
            Dict avec trend_direction, strength, confidence, etc.
        """
        try:
            if len(df) < self.ema_long:
                return self._default_trend_analysis()
            
            current_price = df['close'].iloc[-1]
            
            # 1. Analyse des EMAs
            ema_analysis = self._analyze_ema_structure(df, current_price)
            
            # 2. Force de la tendance (ADX)
            adx_analysis = self._analyze_adx_strength(indicators)
            
            # 3. Pente de prix (momentum directionnel)
            slope_analysis = self._analyze_price_slope(df)
            
            # 4. Structure de marché (Higher Highs / Higher Lows)
            structure_analysis = self._analyze_market_structure(df)
            
            # 5. Synthèse de la tendance
            trend_synthesis = self._synthesize_trend(
                ema_analysis, adx_analysis, slope_analysis, structure_analysis
            )
            
            return trend_synthesis
            
        except Exception as e:
            logger.error(f"Erreur analyse tendance: {e}")
            return self._default_trend_analysis()
    
    def _analyze_ema_structure(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Analyse la structure des moyennes mobiles"""
        try:
            ema50 = df['close'].ewm(span=self.ema_fast).mean().iloc[-1]
            ema100 = df['close'].ewm(span=self.ema_slow).mean().iloc[-1] 
            ema200 = df['close'].ewm(span=self.ema_long).mean().iloc[-1]
            
            # Score basé sur l'alignement des EMAs
            score = 0
            details = []
            
            # Prix au-dessus des EMAs
            if current_price > ema50:
                score += 25
                details.append("Prix > EMA50")
            if current_price > ema100:
                score += 25  
                details.append("Prix > EMA100")
            if current_price > ema200:
                score += 25
                details.append("Prix > EMA200")
            
            # Alignement croissant des EMAs (bullish)
            if ema50 > ema100:
                score += 15
                details.append("EMA50 > EMA100")
            if ema100 > ema200:
                score += 10
                details.append("EMA100 > EMA200")
                
            # Direction des EMAs (pente)
            if len(df) >= self.ema_fast + 5:
                ema50_slope = (ema50 - df['close'].ewm(span=self.ema_fast).mean().iloc[-6]) / ema50
                if ema50_slope > 0.001:  # Pente haussière > 0.1%
                    score += 15
                    details.append("EMA50 pente haussière")
            
            return {
                'score': score,
                'ema50': ema50,
                'ema100': ema100, 
                'ema200': ema200,
                'details': details
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse EMAs: {e}")
            return {'score': 50, 'details': []}
    
    def _analyze_adx_strength(self, indicators: Dict) -> Dict:
        """Analyse la force de tendance via ADX"""
        try:
            adx = self._get_indicator_value(indicators, 'adx_14')
            di_plus = self._get_indicator_value(indicators, 'di_plus')
            di_minus = self._get_indicator_value(indicators, 'di_minus')
            
            score = 0
            details = []
            
            if adx is not None:
                if adx >= 50:  # Tendance très forte
                    score += 40
                    details.append(f"ADX très fort ({adx:.1f})")
                elif adx >= 30:  # Tendance forte
                    score += 30
                    details.append(f"ADX fort ({adx:.1f})")
                elif adx >= self.adx_trend_threshold:  # Tendance modérée
                    score += 20
                    details.append(f"ADX modéré ({adx:.1f})")
                else:  # Tendance faible
                    score += 5
                    details.append(f"ADX faible ({adx:.1f})")
                
                # Direction de la tendance (DI+ vs DI-)
                if di_plus is not None and di_minus is not None:
                    if di_plus > di_minus:
                        score += 15
                        details.append("DI+ > DI- (haussier)")
                    else:
                        score -= 5
                        details.append("DI+ < DI- (baissier)")
                        
            return {
                'score': score,
                'adx': adx,
                'di_plus': di_plus,
                'di_minus': di_minus,
                'details': details
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse ADX: {e}")
            return {'score': 25, 'details': []}
    
    def _analyze_price_slope(self, df: pd.DataFrame) -> Dict:
        """Analyse la pente du prix (momentum directionnel)"""
        try:
            if len(df) < 20:
                return {'score': 25, 'details': []}
            
            current_price = df['close'].iloc[-1]
            
            # Pentes sur différentes périodes
            periods = [5, 10, 20]
            total_score = 0
            details = []
            
            for period in periods:
                if len(df) >= period + 1:
                    past_price = df['close'].iloc[-(period+1)]
                    slope = (current_price - past_price) / past_price
                    
                    if slope > 0.02:  # +2% sur la période
                        period_score = 20
                        details.append(f"Pente {period}p très haussière ({slope*100:.1f}%)")
                    elif slope > 0.01:  # +1% sur la période  
                        period_score = 15
                        details.append(f"Pente {period}p haussière ({slope*100:.1f}%)")
                    elif slope > 0.005:  # +0.5% sur la période
                        period_score = 10
                        details.append(f"Pente {period}p légèrement haussière ({slope*100:.1f}%)")
                    elif slope > -0.005:  # Neutre
                        period_score = 5
                        details.append(f"Pente {period}p neutre ({slope*100:.1f}%)")
                    else:  # Baissière
                        period_score = 0
                        details.append(f"Pente {period}p baissière ({slope*100:.1f}%)")
                    
                    total_score += period_score
            
            # Moyenne des scores
            avg_score = total_score / len(periods) if periods else 25
            
            return {
                'score': avg_score,
                'details': details
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse pente: {e}")
            return {'score': 25, 'details': []}
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyse la structure de marché (HH/HL vs LH/LL)"""
        try:
            if len(df) < 30:
                return {'score': 25, 'details': []}
            
            # Identifier les pivots (hauts et bas locaux)
            window = 5  # Fenêtre pour les pivots
            highs = []
            lows = []
            
            for i in range(window, len(df) - window):
                # Haut local
                if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1)) and \
                   all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)):
                    highs.append((i, df['high'].iloc[i]))
                
                # Bas local  
                if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1)) and \
                   all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)):
                    lows.append((i, df['low'].iloc[i]))
            
            # Analyser les 4 derniers pivots de chaque type
            recent_highs = highs[-4:] if len(highs) >= 4 else highs
            recent_lows = lows[-4:] if len(lows) >= 4 else lows
            
            score = 0
            details = []
            
            # Higher Highs
            if len(recent_highs) >= 2:
                hh_count = sum(1 for i in range(1, len(recent_highs)) 
                              if recent_highs[i][1] > recent_highs[i-1][1])
                if hh_count >= len(recent_highs) - 1:
                    score += 25
                    details.append("Structure Higher Highs confirmée")
                elif hh_count >= len(recent_highs) // 2:
                    score += 15
                    details.append("Tendance Higher Highs partielle")
            
            # Higher Lows
            if len(recent_lows) >= 2:
                hl_count = sum(1 for i in range(1, len(recent_lows))
                              if recent_lows[i][1] > recent_lows[i-1][1])
                if hl_count >= len(recent_lows) - 1:
                    score += 25
                    details.append("Structure Higher Lows confirmée")
                elif hl_count >= len(recent_lows) // 2:
                    score += 15
                    details.append("Tendance Higher Lows partielle")
            
            return {
                'score': score,
                'recent_highs': len(recent_highs),
                'recent_lows': len(recent_lows),
                'details': details
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse structure: {e}")
            return {'score': 25, 'details': []}
    
    def _synthesize_trend(self, ema_analysis: Dict, adx_analysis: Dict, 
                         slope_analysis: Dict, structure_analysis: Dict) -> Dict:
        """Synthétise l'analyse de tendance"""
        try:
            # Pondération des composantes
            total_score = (
                ema_analysis['score'] * 0.3 +      # 30% - Structure EMAs
                adx_analysis['score'] * 0.25 +     # 25% - Force ADX  
                slope_analysis['score'] * 0.25 +   # 25% - Momentum prix
                structure_analysis['score'] * 0.2   # 20% - Structure marché
            )
            
            # Déterminer direction et force
            if total_score >= 75:
                trend_direction = "STRONG_BULLISH"
                confidence = 0.9
            elif total_score >= 60:
                trend_direction = "BULLISH" 
                confidence = 0.75
            elif total_score >= 45:
                trend_direction = "WEAK_BULLISH"
                confidence = 0.6
            elif total_score <= 25:
                trend_direction = "STRONG_BEARISH"
                confidence = 0.9
            elif total_score <= 40:
                trend_direction = "BEARISH"
                confidence = 0.75
            else:
                trend_direction = "NEUTRAL"
                confidence = 0.5
            
            # Compiler tous les détails
            all_details = (
                ema_analysis.get('details', []) +
                adx_analysis.get('details', []) + 
                slope_analysis.get('details', []) +
                structure_analysis.get('details', [])
            )
            
            return {
                'trend_direction': trend_direction,
                'total_score': total_score,
                'confidence': confidence,
                'components': {
                    'ema_score': ema_analysis['score'],
                    'adx_score': adx_analysis['score'], 
                    'slope_score': slope_analysis['score'],
                    'structure_score': structure_analysis['score']
                },
                'details': all_details,
                'should_favor_buys': total_score >= 55,  # Favoriser BUY si score >= 55
                'should_reduce_sells': total_score >= 65  # Réduire SELL si score >= 65
            }
            
        except Exception as e:
            logger.error(f"Erreur synthèse tendance: {e}")
            return self._default_trend_analysis()
    
    def should_boost_buy_signal(self, trend_analysis: Dict) -> Tuple[bool, float]:
        """
        Détermine si un signal BUY doit être boosté selon la tendance
        
        Returns:
            (should_boost, confidence_adjustment)  
        """
        try:
            direction = trend_analysis.get('trend_direction', 'NEUTRAL')
            score = trend_analysis.get('total_score', 50)
            
            if direction == "STRONG_BULLISH":
                return True, 0.15  # +15% confidence
            elif direction == "BULLISH":
                return True, 0.10  # +10% confidence  
            elif direction == "WEAK_BULLISH" and score >= 50:
                return True, 0.05  # +5% confidence
                
            return False, 0.0
            
        except Exception as e:
            logger.error(f"Erreur boost BUY: {e}")
            return False, 0.0
    
    def should_reduce_sell_signal(self, trend_analysis: Dict) -> Tuple[bool, float]:
        """
        Détermine si un signal SELL doit être réduit en tendance haussière
        
        Returns:
            (should_reduce, confidence_reduction)
        """
        try:
            direction = trend_analysis.get('trend_direction', 'NEUTRAL')
            score = trend_analysis.get('total_score', 50)
            
            if direction in ["STRONG_BULLISH", "BULLISH"] and score >= 65:
                return True, 0.20  # -20% confidence pour SELL en forte hausse
            elif direction == "BULLISH" and score >= 60:
                return True, 0.15  # -15% confidence
            elif score >= 55:
                return True, 0.10  # -10% confidence
                
            return False, 0.0
            
        except Exception as e:
            logger.error(f"Erreur réduction SELL: {e}")
            return False, 0.0
    
    def _get_indicator_value(self, indicators: Dict, key: str) -> Optional[float]:
        """Récupère la valeur actuelle d'un indicateur"""
        value = indicators.get(key)
        if value is None:
            return None
        
        if isinstance(value, (list, np.ndarray)) and len(value) > 0:
            return float(value[-1])
        elif isinstance(value, (int, float)):
            return float(value)
        
        return None
    
    def _default_trend_analysis(self) -> Dict:
        """Retourne une analyse de tendance par défaut"""
        return {
            'trend_direction': 'NEUTRAL',
            'total_score': 50.0,
            'confidence': 0.5,
            'components': {
                'ema_score': 50,
                'adx_score': 25,
                'slope_score': 25, 
                'structure_score': 25
            },
            'details': [],
            'should_favor_buys': False,
            'should_reduce_sells': False
        }