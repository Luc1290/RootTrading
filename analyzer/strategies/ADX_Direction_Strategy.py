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
        # Seuils ADX CRYPTO STRICTS - Anti-faux signaux
        self.adx_threshold = 22.0     # Tendance minimum CRYPTO (relevé de 15 à 22)
        self.adx_strong = 30.0        # Tendance forte (relevé de 25 à 30)
        self.adx_extreme = 40.0       # Tendance très forte (relevé de 35 à 40)
        self.di_diff_threshold = 5.0  # Différence minimale DI STRICT (5.0 vs 2.0)
        
        # NOUVEAUX FILTRES CRYPTO
        self.min_confluence_required = 55    # Confluence minimum OBLIGATOIRE
        self.min_momentum_alignment = 8      # Momentum alignment minimum (échelle 0-100, écart min 8 points)
        self.required_confirmations = 2      # Confirmations minimum requises
        
        # Gestion des régimes de marché DURCIE
        self.ranging_penalty = 0.25   # Pénalité marché ranging augmentée  
        self.volatile_penalty = 0.18  # Pénalité marché volatil augmentée
        
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
            'confluence_score': self.indicators.get('confluence_score'),
            'market_regime': self.indicators.get('market_regime')
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
            
        # VALIDATION PRÉLIMINAIRE CONFLUENCE OBLIGATOIRE
        confluence_score = values.get('confluence_score', 0)
        if not confluence_score or float(confluence_score) < self.min_confluence_required:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Confluence insuffisante ({confluence_score}) < {self.min_confluence_required} - ADX rejeté",
                "metadata": {"strategy": self.name, "rejected_reason": "low_confluence"}
            }

        # ADX trop faible = pas de tendance claire (seuil relevé)
        if adx < self.adx_threshold:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"ADX insuffisant ({adx:.1f}) < {self.adx_threshold} - tendance trop faible",
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
        
        # Vérification STRICTE de la différence entre DI
        if di_diff_abs < self.di_diff_threshold:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"DI écart insuffisant ({di_diff_abs:.1f}) < {self.di_diff_threshold} - direction incertaine",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "adx": adx,
                    "plus_di": plus_di,
                    "minus_di": minus_di
                }
            }
            
        # VALIDATION MOMENTUM ALIGNMENT OBLIGATOIRE 
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum_val = float(momentum_score)
                # Vérifier alignement momentum/direction ADX
                momentum_center = 50  # Neutre à 50
                momentum_deviation = abs(momentum_val - momentum_center)
                
                # Direction déterminée par DI
                adx_direction = "bullish" if plus_di > minus_di else "bearish"
                
                # Momentum doit être aligné avec direction ADX
                if adx_direction == "bullish" and momentum_val < (momentum_center + self.min_momentum_alignment):
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Momentum non aligné bullish: {momentum_val:.1f} insuffisant pour ADX haussier",
                        "metadata": {"strategy": self.name, "rejected_reason": "momentum_misalignment"}
                    }
                elif adx_direction == "bearish" and momentum_val > (momentum_center - self.min_momentum_alignment):
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Momentum non aligné bearish: {momentum_val:.1f} insuffisant pour ADX baissier",
                        "metadata": {"strategy": self.name, "rejected_reason": "momentum_misalignment"}
                    }
            except (ValueError, TypeError):
                pass
            
        signal_side = None
        reason = ""
        base_confidence = 0.50  # Standardisé à 0.50 pour équité avec autres stratégies
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
            
        # SYSTÈME CONFIRMATIONS OBLIGATOIRES
        confirmations_count = 0
        confirmations_details = []
        
        # Ajustement confiance selon ADX (RÉDUIT pour éviter sur-confiance)
        if adx >= self.adx_extreme:
            confidence_boost += 0.15  # RÉDUIT de 0.18 à 0.15
            reason += " - tendance très forte"
            confirmations_count += 1
            confirmations_details.append("ADX_extreme")
        elif adx >= self.adx_strong:
            confidence_boost += 0.10  # RÉDUIT de 0.12 à 0.10
            reason += " - tendance forte"
            confirmations_count += 1
            confirmations_details.append("ADX_strong")
        else:
            confidence_boost += 0.04  # RÉDUIT de 0.06 à 0.04
            reason += " - tendance minimum"
            
        # Ajustement selon différence DI (PLUS STRICT)
        if di_diff_abs >= 15:  # Seuil relevé de 20 à 15
            confidence_boost += 0.12
            reason += f" (écart DI excellent: {di_diff_abs:.1f})"
            confirmations_count += 1
            confirmations_details.append("DI_gap_excellent")
        elif di_diff_abs >= 8:   # Seuil relevé de 10 à 8
            confidence_boost += 0.08
            reason += f" (écart DI bon: {di_diff_abs:.1f})"
            confirmations_count += 1
            confirmations_details.append("DI_gap_good")
        else:
            confidence_boost += 0.02  # RÉDUIT de 0.04 à 0.02
            reason += f" (écart DI minimum: {di_diff_abs:.1f})"
            
        # Utilisation des indicateurs complémentaires
        
        # Trend strength pré-calculé (COMPTAGE CONFIRMATIONS)
        trend_strength = values.get('trend_strength')
        if trend_strength:
            trend_str = str(trend_strength).lower()
            if trend_str in ['extreme', 'very_strong']:
                confidence_boost += 0.12
                confirmations_count += 1
                confirmations_details.append("trend_very_strong")
                reason += f" avec trend_strength {trend_str}"
            elif trend_str == 'strong':
                confidence_boost += 0.08
                confirmations_count += 1
                confirmations_details.append("trend_strong")
                reason += f" avec trend_strength {trend_str}"
                
        # Directional bias confirmation (COMPTAGE CONFIRMATIONS)
        directional_bias = values.get('directional_bias')
        if directional_bias:
            if (signal_side == "BUY" and str(directional_bias).upper() == "BULLISH") or \
               (signal_side == "SELL" and str(directional_bias).upper() == "BEARISH"):
                confidence_boost += 0.10
                confirmations_count += 1
                confirmations_details.append("directional_bias_aligned")
                reason += " confirmé par bias directionnel"
                
        # Momentum score DÉJÀ VALIDÉ plus haut - juste bonus ici
        if momentum_score is not None:
            try:
                momentum_val = float(momentum_score)
                # Bonus plus strict pour momentum fort
                if (signal_side == "BUY" and momentum_val > 65) or \
                   (signal_side == "SELL" and momentum_val < 35):
                    confidence_boost += 0.12
                    confirmations_count += 1
                    confirmations_details.append("momentum_strong")
                    reason += f" avec momentum FORT ({momentum_val:.1f})"
                elif (signal_side == "BUY" and momentum_val > 58) or \
                     (signal_side == "SELL" and momentum_val < 42):
                    confidence_boost += 0.08
                    reason += f" avec momentum favorable ({momentum_val:.1f})"
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
                
        # Signal strength (varchar: WEAK/MODERATE/STRONG)
        signal_strength_calc = values.get('signal_strength')
        if signal_strength_calc:
            signal_str = str(signal_strength_calc).upper()
            if signal_str == 'STRONG':
                confidence_boost += 0.10
                reason += " + signal fort"
            elif signal_str == 'MODERATE':
                confidence_boost += 0.05
                reason += " + signal modéré"
                
        # VALIDATION FINALE CONFIRMATIONS OBLIGATOIRES
        if confirmations_count < self.required_confirmations:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Confirmations insuffisantes ADX ({confirmations_count}/{self.required_confirmations}) - rejeté",
                "metadata": {
                    "strategy": self.name,
                    "rejected_reason": "insufficient_confirmations",
                    "confirmations_count": confirmations_count,
                    "confirmations_details": confirmations_details
                }
            }
        
        # Confluence score DÉJÀ VALIDÉ comme obligatoire - juste bonus ici  
        try:
            confluence = float(confluence_score)
            # Bonus confluence adapté (RÉDUIT)
            if confluence > 80:
                confidence_boost += 0.12  # RÉDUIT de 0.15
                reason += f" + confluence excellente ({confluence:.0f})"
            elif confluence > 70:
                confidence_boost += 0.08  # RÉDUIT de 0.10  
                reason += f" + confluence élevée ({confluence:.0f})"
            elif confluence > 60:
                confidence_boost += 0.04  # RÉDUIT de 0.06
                reason += f" + confluence correcte ({confluence:.0f})"
        except (ValueError, TypeError):
            pass
        
        # Gestion des régimes de marché
        market_regime = values.get('market_regime')
        if market_regime:
            regime_str = str(market_regime).upper()
            if regime_str == 'RANGING':
                confidence_boost -= self.ranging_penalty
                reason += " (marché ranging)"
            elif regime_str == 'VOLATILE':
                confidence_boost -= self.volatile_penalty  
                reason += " (marché volatil)"
            elif regime_str in ['TRENDING_BULL', 'TRENDING_BEAR']:
                # Bonus si aligné avec la tendance
                if (signal_side == "BUY" and regime_str == 'TRENDING_BULL') or \
                   (signal_side == "SELL" and regime_str == 'TRENDING_BEAR'):
                    confidence_boost += 0.12
                    reason += f" (aligné {regime_str.lower()})"
                else:
                    confidence_boost -= 0.08
                    reason += f" (contre-tendance {regime_str.lower()})"
                
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
                "confluence_score": confluence_score,
                "market_regime": values.get('market_regime'),
                "confirmations_count": confirmations_count,
                "confirmations_details": confirmations_details
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
