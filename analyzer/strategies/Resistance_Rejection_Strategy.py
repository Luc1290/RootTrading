"""
Resistance_Rejection_Strategy - Stratégie basée sur le rejet au niveau de résistance.
Détecte les échecs de cassure de résistance pour signaler des ventes (retournement baissier).
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class Resistance_Rejection_Strategy(BaseStrategy):
    """
    Stratégie détectant les rejets de prix au niveau des résistances pour des signaux SELL.
    
    Pattern de rejet de résistance :
    1. Prix approche d'une résistance forte
    2. Tentative de cassure mais échec (rejection)
    3. Volume élevé au moment du rejet
    4. Indicateurs techniques montrent essoufflement haussier
    5. Retournement baissier attendu
    
    Signaux générés:
    - SELL: Rejet confirmé au niveau de résistance avec momentum baissier
    - Pas de BUY dans cette stratégie (focus sur les rejets baissiers)
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        
        # Paramètres de proximité résistance
        self.resistance_proximity_threshold = 0.005  # 0.5% de la résistance
        self.tight_proximity_threshold = 0.002       # 0.2% = très proche
        
        # Paramètres de rejet
        self.min_rejection_distance = 0.001          # 0.1% retour minimum depuis résistance
        self.rejection_confirmation_bars = 3         # Barres pour confirmer rejet
        
        # Paramètres volume et momentum
        self.min_rejection_volume = 1.2              # Volume 20% au-dessus normal
        self.strong_rejection_volume = 2.0           # Volume 2x pour rejet fort
        self.momentum_reversal_threshold = -0.2      # Momentum devient négatif
        
        # Paramètres RSI/oscillateurs (essoufflement haussier)
        self.overbought_rsi_threshold = 70           # RSI surachat
        self.extreme_overbought_threshold = 80       # RSI extrême
        self.williams_r_overbought = -20             # Williams %R surachat
        
        # Paramètres de résistance
        self.min_resistance_strength = 0.5           # Force minimum résistance
        self.strong_resistance_threshold = 0.8       # Résistance très forte
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            # Support/Résistance (principal)
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            'resistance_strength': self.indicators.get('resistance_strength'),
            'resistance_levels': self.indicators.get('resistance_levels'),
            'break_probability': self.indicators.get('break_probability'),
            'pivot_count': self.indicators.get('pivot_count'),
            
            # Bollinger Bands (résistance dynamique)
            'bb_upper': self.indicators.get('bb_upper'),
            'bb_lower': self.indicators.get('bb_lower'),
            'bb_position': self.indicators.get('bb_position'),
            'bb_width': self.indicators.get('bb_width'),
            
            # Oscillateurs momentum (essoufflement)
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            'williams_r': self.indicators.get('williams_r'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            
            # Volume analysis (confirmation rejet)
            'volume_ratio': self.indicators.get('volume_ratio'),
            'relative_volume': self.indicators.get('relative_volume'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'volume_spike_multiplier': self.indicators.get('volume_spike_multiplier'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            
            # Momentum et trend
            'momentum_score': self.indicators.get('momentum_score'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'roc_10': self.indicators.get('roc_10'),
            'momentum_10': self.indicators.get('momentum_10'),
            
            # Pattern et confluence
            'pattern_detected': self.indicators.get('pattern_detected'),
            'pattern_confidence': self.indicators.get('pattern_confidence'),
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score')
        }
        
    def _detect_resistance_rejection(self, values: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Détecte un pattern de rejet de résistance."""
        rejection_score = 0.0
        rejection_indicators = []
        
        # Vérification résistance principale
        nearest_resistance = values.get('nearest_resistance')
        if nearest_resistance is None:
            return {'is_rejection': False, 'score': 0.0, 'indicators': []}
            
        try:
            resistance_level = float(nearest_resistance)
        except (ValueError, TypeError):
            return {'is_rejection': False, 'score': 0.0, 'indicators': []}
            
        # Distance à la résistance
        distance_to_resistance = abs(current_price - resistance_level) / resistance_level
        
        # Vérifier proximité de la résistance
        if distance_to_resistance > self.resistance_proximity_threshold:
            return {
                'is_rejection': False, 
                'score': 0.0, 
                'indicators': [f"Trop loin de résistance ({distance_to_resistance*100:.2f}%)"]
            }
            
        # Score de proximité
        if distance_to_resistance <= self.tight_proximity_threshold:
            rejection_score += 0.3
            rejection_indicators.append(f"Très proche résistance ({distance_to_resistance*100:.2f}%)")
        else:
            rejection_score += 0.15
            rejection_indicators.append(f"Proche résistance ({distance_to_resistance*100:.2f}%)")
            
        # Vérifier si le prix est en-dessous de la résistance (rejet effectif)
        if current_price >= resistance_level:
            rejection_score *= 0.5  # Pénalité si pas encore rejeté
            rejection_indicators.append("Prix encore au-dessus résistance")
        else:
            rejection_score += 0.2
            rejection_indicators.append("Prix rejeté sous résistance")
            
        # Force de la résistance
        resistance_strength = values.get('resistance_strength')
        if resistance_strength is not None:
            try:
                # Supposer que resistance_strength est un string comme "STRONG", "MODERATE", etc.
                if isinstance(resistance_strength, str):
                    strength_map = {'WEAK': 0.2, 'MODERATE': 0.5, 'STRONG': 0.8, 'MAJOR': 1.0}
                    strength_val = strength_map.get(resistance_strength.upper(), 0.5)
                else:
                    strength_val = float(resistance_strength)
                    
                if strength_val >= self.strong_resistance_threshold:
                    rejection_score += 0.25
                    rejection_indicators.append(f"Résistance très forte ({strength_val:.2f})")
                elif strength_val >= self.min_resistance_strength:
                    rejection_score += 0.15
                    rejection_indicators.append(f"Résistance forte ({strength_val:.2f})")
            except (ValueError, TypeError):
                pass
                
        # Bollinger Band résistance dynamique
        bb_upper = values.get('bb_upper')
        bb_position = values.get('bb_position')
        if bb_upper is not None and bb_position is not None:
            try:
                bb_upper_val = float(bb_upper)
                bb_pos_val = float(bb_position)
                
                # Si près de BB upper ET position élevée = résistance dynamique
                bb_distance = abs(current_price - bb_upper_val) / bb_upper_val
                if bb_distance <= 0.002 and bb_pos_val >= 0.9:
                    rejection_score += 0.2
                    rejection_indicators.append(f"Résistance Bollinger (pos={bb_pos_val:.2f})")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_rejection': rejection_score >= 0.5,
            'score': rejection_score,
            'indicators': rejection_indicators,
            'resistance_level': resistance_level,
            'distance_pct': distance_to_resistance * 100
        }
        
    def _detect_momentum_exhaustion(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte l'essoufflement du momentum haussier."""
        exhaustion_score = 0.0
        exhaustion_indicators = []
        
        # RSI en surachat (essoufflement)
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi_val = float(rsi_14)
                if rsi_val >= self.extreme_overbought_threshold:
                    exhaustion_score += 0.3
                    exhaustion_indicators.append(f"RSI surachat extrême ({rsi_val:.1f})")
                elif rsi_val >= self.overbought_rsi_threshold:
                    exhaustion_score += 0.2
                    exhaustion_indicators.append(f"RSI surachat ({rsi_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # Williams %R confirme surachat
        williams_r = values.get('williams_r')
        if williams_r is not None:
            try:
                wr_val = float(williams_r)
                if wr_val >= self.williams_r_overbought:
                    exhaustion_score += 0.15
                    exhaustion_indicators.append(f"Williams%R surachat ({wr_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # Stochastic en surachat
        stoch_k = values.get('stoch_k')
        stoch_d = values.get('stoch_d')
        if stoch_k is not None and stoch_d is not None:
            try:
                k_val = float(stoch_k)
                d_val = float(stoch_d)
                if k_val >= 80 and d_val >= 80:
                    exhaustion_score += 0.15
                    exhaustion_indicators.append(f"Stoch surachat (K={k_val:.1f}, D={d_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # Momentum score devient négatif
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum_val = float(momentum_score)
                if momentum_val <= self.momentum_reversal_threshold:
                    exhaustion_score += 0.2
                    exhaustion_indicators.append(f"Momentum reversal ({momentum_val:.2f})")
            except (ValueError, TypeError):
                pass
                
        # ROC ralentissement
        roc_10 = values.get('roc_10')
        if roc_10 is not None:
            try:
                roc_val = float(roc_10)
                if roc_val < 0:  # ROC négatif = retournement
                    exhaustion_score += 0.15
                    exhaustion_indicators.append(f"ROC négatif ({roc_val:.2f}%)")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_exhausted': exhaustion_score >= 0.4,
            'score': exhaustion_score,
            'indicators': exhaustion_indicators
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur le rejet de résistance.
        """
        if not self.validate_data():
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données insuffisantes",
                "metadata": {}
            }
            
        values = self._get_current_values()
        
        # Récupérer le prix actuel depuis les données OHLCV
        current_price = None
        if 'ohlcv' in self.data and self.data['ohlcv']:
            current_price = float(self.data['ohlcv'][-1]['close'])
        
        if current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix actuel non disponible",
                "metadata": {"strategy": self.name}
            }
            
        # Détection du rejet de résistance
        rejection_analysis = self._detect_resistance_rejection(values, current_price)
        
        if not rejection_analysis['is_rejection']:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Pas de rejet de résistance détecté: {', '.join(rejection_analysis['indicators'][:2]) if rejection_analysis['indicators'] else 'Aucune résistance proche'}",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "current_price": current_price,
                    "rejection_score": rejection_analysis['score']
                }
            }
            
        # Détection de l'essoufflement du momentum
        exhaustion_analysis = self._detect_momentum_exhaustion(values)
        
        # Signal SELL si rejet + essoufflement
        if rejection_analysis['is_rejection']:
            base_confidence = 0.5
            confidence_boost = rejection_analysis['score'] * 0.6
            
            reason = f"Rejet résistance {rejection_analysis['resistance_level']:.2f} ({rejection_analysis['distance_pct']:.2f}%)"
            
            # Bonus pour essoufflement momentum
            if exhaustion_analysis['is_exhausted']:
                confidence_boost += exhaustion_analysis['score'] * 0.4
                reason += f" + momentum épuisé"
                
            # Volume de confirmation
            volume_ratio = values.get('volume_ratio')
            if volume_ratio is not None:
                try:
                    vol_ratio = float(volume_ratio)
                    if vol_ratio >= self.strong_rejection_volume:
                        confidence_boost += 0.2
                        reason += f" + volume fort ({vol_ratio:.1f}x)"
                    elif vol_ratio >= self.min_rejection_volume:
                        confidence_boost += 0.1
                        reason += f" + volume confirmé ({vol_ratio:.1f}x)"
                except (ValueError, TypeError):
                    pass
                    
            # Confluence score
            confluence_score = values.get('confluence_score')
            if confluence_score is not None:
                try:
                    conf_val = float(confluence_score)
                    if conf_val > 0.7:
                        confidence_boost += 0.1
                        reason += " + haute confluence"
                except (ValueError, TypeError):
                    pass
                    
            # Pattern confidence
            pattern_confidence = values.get('pattern_confidence')
            if pattern_confidence is not None:
                try:
                    pat_conf = float(pattern_confidence)
                    if pat_conf > 0.8:
                        confidence_boost += 0.1
                except (ValueError, TypeError):
                    pass
                    
            confidence = self.calculate_confidence(base_confidence, 1.0 + confidence_boost)
            strength = self.get_strength_from_confidence(confidence)
            
            return {
                "side": "SELL",
                "confidence": confidence,
                "strength": strength,
                "reason": reason,
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "current_price": current_price,
                    "resistance_level": rejection_analysis['resistance_level'],
                    "resistance_distance_pct": rejection_analysis['distance_pct'],
                    "rejection_score": rejection_analysis['score'],
                    "exhaustion_score": exhaustion_analysis['score'],
                    "rejection_indicators": rejection_analysis['indicators'],
                    "exhaustion_indicators": exhaustion_analysis['indicators'],
                    "volume_ratio": volume_ratio,
                    "rsi_14": values.get('rsi_14'),
                    "williams_r": values.get('williams_r'),
                    "momentum_score": values.get('momentum_score'),
                    "confluence_score": confluence_score,
                    "pattern_confidence": pattern_confidence
                }
            }
            
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": "Conditions de rejet non remplies",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "current_price": current_price
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        required_indicators = [
            'nearest_resistance', 'rsi_14', 'volume_ratio'
        ]
        
        if not self.indicators:
            logger.warning(f"{self.name}: Aucun indicateur disponible")
            return False
            
        for indicator in required_indicators:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        # Vérifier données OHLCV pour prix actuel
        if 'ohlcv' not in self.data or not self.data['ohlcv']:
            logger.warning(f"{self.name}: Données OHLCV manquantes")
            return False
            
        return True
