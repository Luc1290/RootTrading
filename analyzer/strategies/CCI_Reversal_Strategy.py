"""
CCI_Reversal_Strategy - Stratégie basée sur le CCI et les indicateurs pré-calculés.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class CCI_Reversal_Strategy(BaseStrategy):
    """
    Stratégie utilisant le CCI et les indicateurs pré-calculés pour détecter les retournements.
    
    Signaux générés:
    - BUY: CCI en zone de survente avec conditions favorables
    - SELL: CCI en zone de surachat avec conditions favorables
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres CCI optimisés pour crypto 3m
        self.oversold_level = -50   # Zone survente adaptée crypto
        self.overbought_level = 50   # Zone surachat adaptée crypto
        self.extreme_oversold = -80  # Extrême accessible
        self.extreme_overbought = 80  # Extrême accessible
        
        # Paramètres de validation temporelle - SIMPLIFIÉS
        self.min_cci_persistence = 0  # Pas de persistance requise (crypto 3m rapide)
        self.cci_history = []  # Historique pour validation
        self.max_history_size = 3  # Historique réduit
        
        # Seuils adaptatifs selon volatilité - ASSOUPLIS
        self.volatility_adjustment = {
            'low': 0.9,    # Seuils légèrement réduits
            'normal': 1.0,   # Seuils standards
            'high': 1.1,     # Seuils légèrement augmentés
            'extreme': 1.15  # Nouveau : haute volatilité
        }
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            'cci_20': self.indicators.get('cci_20'),
            'momentum_score': self.indicators.get('momentum_score'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'confluence_score': self.indicators.get('confluence_score'),
            'signal_strength': self.indicators.get('signal_strength'),
            'pattern_detected': self.indicators.get('pattern_detected'),
            'pattern_confidence': self.indicators.get('pattern_confidence'),
            'market_regime': self.indicators.get('market_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            'volume_ratio': self.indicators.get('volume_ratio'),  # Ajout volume
            'rsi_14': self.indicators.get('rsi_14')  # Ajout RSI pour confirmation
        }
    
    def _update_cci_history(self, cci_value: float) -> None:
        """Met à jour l'historique CCI pour validation temporelle."""
        self.cci_history.append(cci_value)
        if len(self.cci_history) > self.max_history_size:
            self.cci_history.pop(0)
    
    def _check_cci_persistence(self, threshold: float, direction: str) -> bool:
        """Vérifie la persistance du CCI dans une zone - SIMPLIFIÉ."""
        # En crypto 3m, pas de persistance requise - réactivité importante
        if self.min_cci_persistence == 0:
            return True
            
        if len(self.cci_history) < self.min_cci_persistence:
            return True  # Accepte si pas assez d'historique
        
        # Logique assouplie : au moins 50% des valeurs dans la zone
        recent_values = self.cci_history[-self.min_cci_persistence:]
        if direction == 'oversold':
            count = sum(1 for v in recent_values if v <= threshold)
        elif direction == 'overbought':
            count = sum(1 for v in recent_values if v >= threshold)
        else:
            return False
            
        return count >= len(recent_values) * 0.5  # Au moins 50%
    
    def _get_adjusted_thresholds(self, volatility_regime: str) -> Dict[str, float]:
        """Ajuste les seuils selon le régime de volatilité."""
        adjustment = self.volatility_adjustment.get(volatility_regime, 1.0)
        return {
            'oversold': self.oversold_level * adjustment,
            'overbought': self.overbought_level * adjustment,
            'extreme_oversold': self.extreme_oversold * adjustment,
            'extreme_overbought': self.extreme_overbought * adjustment
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur le CCI et les indicateurs pré-calculés.
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
        
        # Vérification des données essentielles
        cci_20_raw = values['cci_20']
        if cci_20_raw is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "CCI non disponible",
                "metadata": {"strategy": self.name}
            }
            
        # Conversion robuste en float
        try:
            cci_20 = float(cci_20_raw)
        except (ValueError, TypeError):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"CCI invalide: {cci_20_raw}",
                "metadata": {"strategy": self.name}
            }
            
        signal_side = None
        reason = ""
        confidence_boost = 0.0
        
        # Mise à jour de l'historique CCI
        self._update_cci_history(cci_20)
        
        # Ajustement des seuils selon volatilité
        volatility_regime = values.get('volatility_regime', 'normal')
        thresholds = self._get_adjusted_thresholds(str(volatility_regime))
        
        # Logique de signal directe (crypto 3m réactif)
        if cci_20 <= thresholds['oversold']:
            signal_side = "BUY"
            if cci_20 <= thresholds['extreme_oversold']:
                zone = "survente extrême"
                confidence_boost += 0.25  # Bonus généreux
            else:
                zone = "survente"
                confidence_boost += 0.15  # Bonus amélioré
            reason = f"CCI ({cci_20:.1f}) en zone de {zone}"
            
        elif cci_20 >= thresholds['overbought']:
            signal_side = "SELL"
            if cci_20 >= thresholds['extreme_overbought']:
                zone = "surachat extrême"
                confidence_boost += 0.25  # Bonus généreux
            else:
                zone = "surachat"
                confidence_boost += 0.15  # Bonus amélioré
            reason = f"CCI ({cci_20:.1f}) en zone de {zone}"
            
        if signal_side:
            base_confidence = 0.55  # Base plus élevée pour compenser
            
            # Utilisation du momentum_score avec logique améliorée
            momentum_score_raw = values.get('momentum_score')
            momentum_score = 0
            if momentum_score_raw is not None:
                try:
                    momentum_score = float(momentum_score_raw)
                except (ValueError, TypeError):
                    momentum_score = 0
            
            if momentum_score != 0:
                # Momentum score format 0-100, 50 = neutre
                if (signal_side == "BUY" and momentum_score > 60) or \
                   (signal_side == "SELL" and momentum_score < 40):
                    confidence_boost += 0.18  # Fort momentum aligné (généreux)
                    reason += " avec momentum fort"
                elif (signal_side == "BUY" and momentum_score > 52) or \
                     (signal_side == "SELL" and momentum_score < 48):
                    confidence_boost += 0.10  # Momentum modéré aligné
                    reason += " avec momentum favorable"
                elif 48 <= momentum_score <= 52:
                    confidence_boost -= 0.05  # Pénalité réduite
                elif (signal_side == "BUY" and momentum_score < 35) or \
                     (signal_side == "SELL" and momentum_score > 65):
                    confidence_boost -= 0.08  # Pénalité réduite
                    
            # Utilisation du trend_strength
            trend_strength_raw = values.get('trend_strength')
            if trend_strength_raw and str(trend_strength_raw).lower() in ['strong']:
                confidence_boost += 0.1
                reason += f" et tendance {str(trend_strength_raw).lower()}"
                
            # Utilisation du directional_bias
            directional_bias = values.get('directional_bias')
            if directional_bias:
                bias_upper = str(directional_bias).upper()
                if (signal_side == "BUY" and bias_upper == "BULLISH") or \
                   (signal_side == "SELL" and bias_upper == "BEARISH"):
                    confidence_boost += 0.1
                    reason += " confirmé par bias directionnel"
                elif (signal_side == "BUY" and bias_upper == "BEARISH") or \
                     (signal_side == "SELL" and bias_upper == "BULLISH"):
                    confidence_boost -= 0.1  # Contradictoire
                    
            # Utilisation du confluence_score avec niveaux multiples
            confluence_score_raw = values.get('confluence_score')
            confluence_score = 0
            if confluence_score_raw is not None:
                try:
                    confluence_score = float(confluence_score_raw)
                except (ValueError, TypeError):
                    confluence_score = 0
                    
            if confluence_score > 80:
                confidence_boost += 0.15
                reason += " avec très haute confluence"
            elif confluence_score > 65:
                confidence_boost += 0.08
                reason += " avec confluence solide"
            elif confluence_score < 30:
                confidence_boost -= 0.10  # Faible confluence = risqué
                
            # Utilisation du pattern_detected et pattern_confidence avec conversion sécurisée
            pattern_detected = values.get('pattern_detected')
            pattern_confidence_raw = values.get('pattern_confidence')
            pattern_confidence = 0
            if pattern_confidence_raw is not None:
                try:
                    pattern_confidence = float(pattern_confidence_raw)
                except (ValueError, TypeError):
                    pattern_confidence = 0
                    
            if pattern_detected and pattern_confidence > 60:
                confidence_boost += 0.1
                reason += f" avec pattern {pattern_detected}"
                
            # Utilisation du market_regime
            market_regime = values.get('market_regime')
            regime_strength_raw = values.get('regime_strength')
            
            if market_regime and regime_strength_raw and str(regime_strength_raw).upper() in ['STRONG']:
                if (signal_side == "BUY" and market_regime in ["TRENDING_BULL", "BREAKOUT_BULL"]) or \
                   (signal_side == "SELL" and market_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]):
                    confidence_boost += 0.1
                    reason += f" en régime {market_regime}"
                    
            # Validation avec RSI si disponible
            rsi_raw = values.get('rsi_14')
            if rsi_raw is not None:
                try:
                    rsi = float(rsi_raw)
                    if (signal_side == "BUY" and rsi < 35) or \
                       (signal_side == "SELL" and rsi > 65):
                        confidence_boost += 0.10
                        reason += f" confirmé par RSI ({rsi:.1f})"
                    elif (signal_side == "BUY" and rsi > 65) or \
                         (signal_side == "SELL" and rsi < 35):
                        confidence_boost -= 0.08  # Pénalité réduite RSI
                except (ValueError, TypeError):
                    pass
            
            # Validation avec volume
            volume_ratio_raw = values.get('volume_ratio')
            if volume_ratio_raw is not None:
                try:
                    volume_ratio = float(volume_ratio_raw)
                    if volume_ratio > 1.5:
                        confidence_boost += 0.08
                        reason += " avec volume élevé"
                    elif volume_ratio < 0.5:
                        confidence_boost -= 0.05  # Pénalité réduite volume
                except (ValueError, TypeError):
                    pass
            
            # Ajustement final selon volatilité - ASSOUPLI
            if volatility_regime == "low":
                confidence_boost += 0.08  # Excellent pour retournements
            elif volatility_regime == "normal":
                confidence_boost += 0.03  # Léger bonus
            elif volatility_regime == "high":
                confidence_boost -= 0.03  # Pénalité réduite
            elif volatility_regime == "extreme":
                confidence_boost -= 0.06  # Pénalité assouplie
                    
            # Utilisation du signal_strength pré-calculé
            signal_strength_calc_raw = values.get('signal_strength')
            if signal_strength_calc_raw:
                strength_upper = str(signal_strength_calc_raw).upper()
                if strength_upper == 'STRONG':
                    confidence_boost += 0.1
                    reason += " + signal fort"
                elif strength_upper == 'MODERATE':
                    confidence_boost += 0.05
                    reason += " + signal modéré"
                
            # Calcul final avec plafond de confidence
            total_boost = min(confidence_boost, 0.35)  # Limite le boost total
            confidence = self.calculate_confidence(base_confidence, 1 + total_boost)
            
            # Filtre final : confidence minimum assoupli
            if confidence < 0.45:  # Seuil plus accessible
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Signal CCI {signal_side} détecté mais confidence insuffisante ({confidence:.2f} < 0.45)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "cci_20": cci_20,
                        "rejected_confidence": confidence
                    }
                }
            
            strength = self.get_strength_from_confidence(confidence)
            
            return {
                "side": signal_side,
                "confidence": confidence,
                "strength": strength,
                "reason": reason,
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "cci_20": cci_20,
                    "zone": zone,
                    "momentum_score": momentum_score,
                    "trend_strength": trend_strength_raw,
                    "directional_bias": directional_bias,
                    "confluence_score": confluence_score,
                    "pattern_detected": pattern_detected,
                    "pattern_confidence": pattern_confidence,
                    "market_regime": market_regime,
                    "volatility_regime": volatility_regime
                }
            }
            
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": f"CCI neutre ({cci_20:.1f}) - pas de zone extrême",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "cci_20": cci_20
            }
        }
