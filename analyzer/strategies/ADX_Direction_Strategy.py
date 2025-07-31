"""
ADX_Direction_Strategy - Stratégie basée sur la force et direction de tendance ADX.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class ADX_Direction_Strategy(BaseStrategy):
    """
    Stratégie utilisant ADX, +DI et -DI pour identifier les tendances fortes.
    
    Signaux générés:
    - BUY: ADX > 25 avec +DI > -DI et momentum haussier
    - SELL: ADX > 25 avec -DI > +DI et momentum baissier
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Seuils ADX
        self.adx_threshold = 25.0  # Tendance forte
        self.adx_strong = 35.0     # Tendance très forte
        self.adx_extreme = 50.0    # Tendance extrême
        self.di_diff_threshold = 5.0  # Différence minimale entre DI
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs ADX."""
        return {
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            'dx': self.indicators.get('dx'),
            'adxr': self.indicators.get('adxr'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'trend_angle': self.indicators.get('trend_angle'),
            'momentum_score': self.indicators.get('momentum_score'),
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score')
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur ADX et les indicateurs directionnels.
        """
        if not self.validate_data():
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données insuffisantes",
                "metadata": {"strategy": self.name}
            }
            
        values = self._get_current_values()
        
        # Vérification des indicateurs essentiels
        try:
            adx = float(values['adx_14']) if values['adx_14'] is not None else None
            plus_di = float(values['plus_di']) if values['plus_di'] is not None else None
            minus_di = float(values['minus_di']) if values['minus_di'] is not None else None
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion ADX: {e}",
                "metadata": {"strategy": self.name}
            }
            
        if adx is None or plus_di is None or minus_di is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "ADX ou DI non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # ADX trop faible = pas de tendance claire
        if adx < self.adx_threshold:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"ADX trop faible ({adx:.1f}) - marché sans tendance",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "adx": adx,
                    "plus_di": plus_di,
                    "minus_di": minus_di
                }
            }
            
        # Calcul de la différence entre DI
        di_diff = plus_di - minus_di
        di_diff_abs = abs(di_diff)
        
        # Vérification de la différence minimale entre DI
        if di_diff_abs < self.di_diff_threshold:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"DI trop proches ({di_diff_abs:.1f}) - direction incertaine",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "adx": adx,
                    "plus_di": plus_di,
                    "minus_di": minus_di
                }
            }
            
        signal_side = None
        reason = ""
        base_confidence = 0.5
        confidence_boost = 0.0
        
        # Logique de signal basée sur la direction des DI
        if plus_di > minus_di:
            # Tendance haussière
            signal_side = "BUY"
            reason = f"ADX ({adx:.1f}) avec tendance haussière (+DI > -DI)"
        else:
            # Tendance baissière
            signal_side = "SELL"
            reason = f"ADX ({adx:.1f}) avec tendance baissière (-DI > +DI)"
            
        # Ajustement de confiance selon la force de l'ADX
        if adx >= self.adx_extreme:
            confidence_boost += 0.25
            reason += " - tendance EXTRÊME"
        elif adx >= self.adx_strong:
            confidence_boost += 0.15
            reason += " - tendance très forte"
        else:
            confidence_boost += 0.05
            reason += " - tendance forte"
            
        # Ajustement selon la différence entre DI
        if di_diff_abs >= 20:
            confidence_boost += 0.15
            reason += f" (écart DI: {di_diff_abs:.1f})"
        elif di_diff_abs >= 10:
            confidence_boost += 0.10
        else:
            confidence_boost += 0.05
            
        # Utilisation des indicateurs complémentaires
        
        # Trend strength pré-calculé
        trend_strength = values.get('trend_strength')
        if trend_strength and str(trend_strength) in ['STRONG', 'VERY_STRONG']:
            confidence_boost += 0.10
            reason += f" avec trend_strength {str(trend_strength).lower()}"
                
        # Directional bias confirmation
        directional_bias = values.get('directional_bias')
        if directional_bias:
            if (signal_side == "BUY" and directional_bias == "bullish") or \
               (signal_side == "SELL" and directional_bias == "bearish"):
                confidence_boost += 0.10
                reason += " confirmé par bias directionnel"
                
        # Momentum score
        momentum_score = values.get('momentum_score')
        if momentum_score:
            try:
                momentum_val = float(momentum_score)
                if (signal_side == "BUY" and momentum_val > 0.3) or \
                   (signal_side == "SELL" and momentum_val < -0.3):
                    confidence_boost += 0.10
                    reason += " avec momentum favorable"
            except (ValueError, TypeError):
                pass
                
        # ADXR pour confirmation de persistance
        adxr = values.get('adxr')
        if adxr:
            try:
                adxr_val = float(adxr)
                if adxr_val > self.adx_threshold:
                    confidence_boost += 0.05
                    reason += " (ADXR confirme)"
            except (ValueError, TypeError):
                pass
                
        # Signal strength et confluence
        signal_strength_calc = values.get('signal_strength')
        if signal_strength_calc:
            try:
                signal_str = float(signal_strength_calc)
                if signal_str > 0.7:
                    confidence_boost += 0.05
            except (ValueError, TypeError):
                pass
                
        confluence_score = values.get('confluence_score')
        if confluence_score:
            try:
                confluence = float(confluence_score)
                if confluence > 0.6:
                    confidence_boost += 0.10
                    reason += " avec haute confluence"
            except (ValueError, TypeError):
                pass
                
        confidence = self.calculate_confidence(base_confidence, 1 + confidence_boost)
        strength = self.get_strength_from_confidence(confidence)
        
        return {
            "side": signal_side,
            "confidence": confidence,
            "strength": strength,
            "reason": reason,
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "adx": adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "di_difference": di_diff,
                "dx": values.get('dx'),
                "adxr": adxr,
                "trend_strength": trend_strength,
                "directional_bias": directional_bias,
                "momentum_score": momentum_score,
                "signal_strength_calc": signal_strength_calc,
                "confluence_score": confluence_score
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs ADX requis sont présents."""
        if not super().validate_data():
            return False
            
        required = ['adx_14', 'plus_di', 'minus_di']
        
        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        return True
