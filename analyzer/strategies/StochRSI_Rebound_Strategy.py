"""
StochRSI_Rebound_Strategy - Stratégie basée sur les signaux StochRSI pré-calculés.
OPTIMISÉE POUR CRYPTO SPOT INTRADAY
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class StochRSI_Rebound_Strategy(BaseStrategy):
    """
    Stratégie utilisant les signaux StochRSI et indicateurs pré-calculés.
    
    Signaux générés:
    - BUY: StochRSI en zone de survente avec signaux favorables
    - SELL: StochRSI en zone de surachat avec signaux favorables
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Seuils StochRSI - OPTIMISÉS CRYPTO
        self.oversold_zone = 20       # Plus strict pour crypto
        self.overbought_zone = 80     # Plus strict pour crypto
        self.extreme_oversold = 8     # Zone extrême survente
        self.extreme_overbought = 92  # Zone extrême surachat
        self.neutral_low = 40         # Zone neutre basse élargie
        self.neutral_high = 60        # Zone neutre haute élargie
        
        # Seuils RSI adaptés crypto
        self.rsi_oversold = 25        # RSI survente crypto
        self.rsi_overbought = 75      # RSI surachat crypto
        self.rsi_oversold_strong = 20 # RSI survente forte
        self.rsi_overbought_strong = 80 # RSI surachat fort
        
        # Seuils momentum adaptés
        self.momentum_bullish = 58    # Momentum haussier
        self.momentum_bearish = 42    # Momentum baissier
        self.momentum_strong_bull = 65 # Momentum très haussier
        self.momentum_strong_bear = 35 # Momentum très baissier
        
        # Volume et confluence
        self.min_volume_ratio = 0.7   # Volume minimum requis
        self.min_confluence = 45      # Confluence minimum
        self.strong_confluence = 65   # Confluence forte
        
    def _safe_float(self, value) -> Optional[float]:
        """Convertit en float de manière sécurisée."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _get_current_values(self) -> Dict[str, Any]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            'stoch_rsi': self._safe_float(self.indicators.get('stoch_rsi')),
            'stoch_k': self._safe_float(self.indicators.get('stoch_k')),
            'stoch_d': self._safe_float(self.indicators.get('stoch_d')),
            'stoch_signal': self.indicators.get('stoch_signal'),
            'stoch_divergence': self.indicators.get('stoch_divergence'),
            'rsi_14': self._safe_float(self.indicators.get('rsi_14')),
            'momentum_score': self._safe_float(self.indicators.get('momentum_score')),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'confluence_score': self._safe_float(self.indicators.get('confluence_score')),
            'signal_strength': self.indicators.get('signal_strength'),
            'volume_ratio': self._safe_float(self.indicators.get('volume_ratio')),
            'volume_quality_score': self._safe_float(self.indicators.get('volume_quality_score')),
            'market_regime': self.indicators.get('market_regime'),
            'volatility_regime': self.indicators.get('volatility_regime')
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les indicateurs StochRSI pré-calculés.
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
        stoch_rsi = values['stoch_rsi']
        if stoch_rsi is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "StochRSI non disponible",
                "metadata": {"strategy": self.name}
            }
            
        # Vérification du volume minimum (crucial en crypto)
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None and volume_ratio < self.min_volume_ratio:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Volume insuffisant ({volume_ratio:.2f}x < {self.min_volume_ratio}x)",
                "metadata": {"strategy": self.name, "volume_ratio": volume_ratio}
            }
            
        # Vérification confluence minimum
        confluence_score = values.get('confluence_score', 0)
        if confluence_score < self.min_confluence:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Confluence insuffisante ({confluence_score:.1f} < {self.min_confluence})",
                "metadata": {"strategy": self.name, "confluence_score": confluence_score}
            }
            
        # Vérification volatilité extrême
        volatility_regime = values.get('volatility_regime')
        if volatility_regime == 'extreme':
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Volatilité extrême - éviter les signaux StochRSI",
                "metadata": {"strategy": self.name, "volatility_regime": volatility_regime}
            }
            
        signal_side = None
        reason = ""
        confidence_boost = 0.0
        
        # Utilisation du signal StochRSI pré-calculé si disponible
        stoch_signal = values.get('stoch_signal')
        if stoch_signal:
            # Conversion des signaux DB vers signaux stratégie
            if stoch_signal == 'OVERSOLD':
                signal_side = "BUY"
                reason = f"Signal StochRSI pré-calculé: {stoch_signal}"
                confidence_boost += 0.18  # Signal pré-calculé fiable
            elif stoch_signal == 'OVERBOUGHT':
                signal_side = "SELL"
                reason = f"Signal StochRSI pré-calculé: {stoch_signal}"
                confidence_boost += 0.18  # Signal pré-calculé fiable
        
        # Si pas de signal pré-calculé, analyse manuelle avec seuils plus stricts
        if not signal_side:
            if stoch_rsi <= self.oversold_zone:
                signal_side = "BUY"
                if stoch_rsi <= self.extreme_oversold:
                    reason = f"StochRSI extrême ({stoch_rsi:.1f}) - rebond attendu"
                    confidence_boost += 0.22
                else:
                    reason = f"StochRSI survente ({stoch_rsi:.1f}) - opportunité rebond"
                    confidence_boost += 0.15
                    
            elif stoch_rsi >= self.overbought_zone:
                signal_side = "SELL"
                if stoch_rsi >= self.extreme_overbought:
                    reason = f"StochRSI extrême ({stoch_rsi:.1f}) - correction attendue"
                    confidence_boost += 0.22
                else:
                    reason = f"StochRSI surachat ({stoch_rsi:.1f}) - opportunité correction"
                    confidence_boost += 0.15
                    
        if signal_side:
            base_confidence = 0.42  # Base conservative pour StochRSI
            
            # Bonus MAJEUR pour divergence détectée
            stoch_divergence = values.get('stoch_divergence')
            if stoch_divergence is True:
                confidence_boost += 0.28  # Divergence = signal très puissant
                reason += " avec DIVERGENCE détectée"
                
            # Confirmation avec croisement K/D - PLUS STRICT
            stoch_k = values.get('stoch_k')
            stoch_d = values.get('stoch_d')
            if stoch_k is not None and stoch_d is not None:
                k_d_diff = abs(stoch_k - stoch_d)
                
                # Croisement favorable avec écart minimum
                if (signal_side == "BUY" and stoch_k > stoch_d and k_d_diff >= 3) or \
                   (signal_side == "SELL" and stoch_k < stoch_d and k_d_diff >= 3):
                    confidence_boost += 0.15
                    reason += f" + croisement K/D fort ({k_d_diff:.1f})"
                elif (signal_side == "BUY" and stoch_k > stoch_d and k_d_diff >= 1) or \
                     (signal_side == "SELL" and stoch_k < stoch_d and k_d_diff >= 1):
                    confidence_boost += 0.08
                    reason += f" + croisement K/D"
                # Pénalité si croisement défavorable
                elif (signal_side == "BUY" and stoch_k <= stoch_d) or \
                     (signal_side == "SELL" and stoch_k >= stoch_d):
                    confidence_boost -= 0.15
                    reason += " MAIS K/D défavorable"
                    
            # Confirmation avec RSI - SEUILS CRYPTO
            rsi_14 = values.get('rsi_14')
            if rsi_14:
                if signal_side == "BUY":
                    if rsi_14 <= self.rsi_oversold_strong:
                        confidence_boost += 0.20
                        reason += f" + RSI survente FORTE ({rsi_14:.1f})"
                    elif rsi_14 <= self.rsi_oversold:
                        confidence_boost += 0.12
                        reason += f" + RSI survente ({rsi_14:.1f})"
                    elif rsi_14 <= 35:
                        confidence_boost += 0.06
                        reason += f" + RSI favorable ({rsi_14:.1f})"
                    elif rsi_14 > 65:
                        confidence_boost -= 0.20  # Pénalité forte si RSI contradictoire
                        reason += f" MAIS RSI élevé ({rsi_14:.1f})"
                        
                elif signal_side == "SELL":
                    if rsi_14 >= self.rsi_overbought_strong:
                        confidence_boost += 0.20
                        reason += f" + RSI surachat FORT ({rsi_14:.1f})"
                    elif rsi_14 >= self.rsi_overbought:
                        confidence_boost += 0.12
                        reason += f" + RSI surachat ({rsi_14:.1f})"
                    elif rsi_14 >= 65:
                        confidence_boost += 0.06
                        reason += f" + RSI favorable ({rsi_14:.1f})"
                    elif rsi_14 < 35:
                        confidence_boost -= 0.20  # Pénalité forte si RSI contradictoire
                        reason += f" MAIS RSI faible ({rsi_14:.1f})"
                    
            # Utilisation du momentum_score - SEUILS CRYPTO
            momentum_score = values.get('momentum_score', 50)
            if momentum_score:
                if signal_side == "BUY":
                    if momentum_score >= self.momentum_strong_bull:
                        confidence_boost += 0.18
                        reason += f" + momentum TRÈS haussier ({momentum_score:.0f})"
                    elif momentum_score >= self.momentum_bullish:
                        confidence_boost += 0.10
                        reason += f" + momentum haussier ({momentum_score:.0f})"
                    elif momentum_score < self.momentum_bearish:
                        confidence_boost -= 0.18  # Pénalité momentum contraire
                        reason += f" MAIS momentum baissier ({momentum_score:.0f})"
                        
                elif signal_side == "SELL":
                    if momentum_score <= self.momentum_strong_bear:
                        confidence_boost += 0.18
                        reason += f" + momentum TRÈS baissier ({momentum_score:.0f})"
                    elif momentum_score <= self.momentum_bearish:
                        confidence_boost += 0.10
                        reason += f" + momentum baissier ({momentum_score:.0f})"
                    elif momentum_score > self.momentum_bullish:
                        confidence_boost -= 0.18  # Pénalité momentum contraire
                        reason += f" MAIS momentum haussier ({momentum_score:.0f})"
                    
            # Utilisation du trend_strength - BONUS AJUSTÉS
            trend_strength = values.get('trend_strength')
            if trend_strength:
                trend_str = str(trend_strength).lower()
                if trend_str in ['extreme', 'very_strong']:
                    confidence_boost += 0.25  # Bonus majeur pour tendance extrême
                    reason += f" + tendance {trend_str}"
                elif trend_str == 'strong':
                    confidence_boost += 0.15
                    reason += f" + tendance forte"
                elif trend_str == 'moderate':
                    confidence_boost += 0.08
                    reason += f" + tendance modérée"
                elif trend_str in ['weak', 'absent']:
                    confidence_boost -= 0.12  # Pénalité tendance faible
                    reason += f" MAIS tendance {trend_str}"
                
            # Utilisation du directional_bias - CRITIQUE EN CRYPTO
            directional_bias = values.get('directional_bias')
            if directional_bias:
                bias_upper = directional_bias.upper()
                if (signal_side == "BUY" and bias_upper == "BULLISH") or \
                   (signal_side == "SELL" and bias_upper == "BEARISH"):
                    confidence_boost += 0.15  # Bonus important si aligné
                    reason += f" + bias {bias_upper} aligné"
                elif (signal_side == "BUY" and bias_upper == "BEARISH") or \
                     (signal_side == "SELL" and bias_upper == "BULLISH"):
                    confidence_boost -= 0.18  # Forte pénalité si contraire
                    reason += f" MAIS bias CONTRAIRE ({bias_upper})"
                    
            # Utilisation du confluence_score - SEUILS STRICTS
            if confluence_score:
                if confluence_score >= 75:
                    confidence_boost += 0.25
                    reason += f" + confluence EXCELLENTE ({confluence_score:.0f})"
                elif confluence_score >= self.strong_confluence:
                    confidence_boost += 0.15
                    reason += f" + confluence forte ({confluence_score:.0f})"
                elif confluence_score >= 55:
                    confidence_boost += 0.08
                    reason += f" + confluence correcte ({confluence_score:.0f})"
                elif confluence_score < 50:
                    confidence_boost -= 0.12  # Pénalité confluence faible
                    reason += f" mais confluence faible ({confluence_score:.0f})"
                
            # Utilisation du signal_strength pré-calculé
            signal_strength_calc = values.get('signal_strength')
            if signal_strength_calc:
                sig_str = str(signal_strength_calc).upper()
                if sig_str == 'STRONG':
                    confidence_boost += 0.12
                    reason += " + signal FORT"
                elif sig_str == 'MODERATE':
                    confidence_boost += 0.06
                    reason += " + signal modéré"
                elif sig_str == 'WEAK':
                    confidence_boost -= 0.08  # Pénalité signal faible
                    reason += " mais signal faible"
                    
            # Volume confirmation - CRITIQUE EN CRYPTO
            if volume_ratio is not None:
                if volume_ratio >= 2.0:  # Volume exceptionnel
                    confidence_boost += 0.20
                    reason += f" + volume EXCEPTIONNEL ({volume_ratio:.1f}x)"
                elif volume_ratio >= 1.5:  # Volume élevé
                    confidence_boost += 0.12
                    reason += f" + volume élevé ({volume_ratio:.1f}x)"
                elif volume_ratio >= 1.0:  # Volume normal
                    confidence_boost += 0.05
                    reason += f" + volume normal ({volume_ratio:.1f}x)"
                    
            # Volume quality
            volume_quality_score = values.get('volume_quality_score')
            if volume_quality_score is not None and volume_quality_score >= 70:
                confidence_boost += 0.08
                reason += " + volume HQ"
                
            # Régime de marché
            market_regime = values.get('market_regime')
            if market_regime:
                if signal_side == "BUY" and market_regime in ["TRENDING_BULL", "BREAKOUT_BULL"]:
                    confidence_boost += 0.12
                    reason += f" ({market_regime})"
                elif signal_side == "SELL" and market_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]:
                    confidence_boost += 0.12
                    reason += f" ({market_regime})"
                elif market_regime == "VOLATILE":
                    confidence_boost -= 0.10  # Pénalité marché volatile
                    reason += " (marché volatile)"
                elif market_regime == "RANGING" and stoch_rsi not in [self.extreme_oversold, self.extreme_overbought]:
                    confidence_boost -= 0.05  # Légère pénalité en range sauf extrêmes
                    
            # Filtre final - seuil plus strict pour StochRSI
            raw_confidence = base_confidence * (1 + confidence_boost)
            if raw_confidence < 0.45:  # Seuil minimum plus élevé
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Signal StochRSI rejeté - confiance insuffisante ({raw_confidence:.2f} < 0.45)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "rejected_signal": signal_side,
                        "raw_confidence": raw_confidence,
                        "stoch_rsi": stoch_rsi
                    }
                }
            
            confidence = self.calculate_confidence(base_confidence, 1.0 + confidence_boost)
            strength = self.get_strength_from_confidence(confidence)
            
            return {
                "side": signal_side,
                "confidence": confidence,
                "strength": strength,
                "reason": reason,
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "stoch_rsi": stoch_rsi,
                    "stoch_k": stoch_k,
                    "stoch_d": stoch_d,
                    "stoch_signal": stoch_signal,
                    "stoch_divergence": stoch_divergence,
                    "rsi_14": rsi_14,
                    "momentum_score": momentum_score,
                    "trend_strength": trend_strength,
                    "confluence_score": confluence_score,
                    "volume_ratio": volume_ratio,
                    "directional_bias": directional_bias,
                    "market_regime": market_regime
                }
            }
            
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": f"StochRSI en zone neutre ({stoch_rsi:.1f}) - seuils: {self.oversold_zone}/{self.overbought_zone}",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "stoch_rsi": stoch_rsi,
                "oversold_zone": self.oversold_zone,
                "overbought_zone": self.overbought_zone
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que les indicateurs StochRSI requis sont présents."""
        if not super().validate_data():
            return False
            
        # Vérifier StochRSI obligatoire
        if 'stoch_rsi' not in self.indicators or self.indicators['stoch_rsi'] is None:
            logger.warning(f"{self.name}: StochRSI manquant")
            return False
            
        return True