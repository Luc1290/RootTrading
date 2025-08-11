"""
OBV_Crossover_Strategy - Stratégie basée sur les croisements OBV avec sa moyenne mobile.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class OBV_Crossover_Strategy(BaseStrategy):
    """
    Stratégie utilisant les croisements de l'On-Balance Volume (OBV) avec sa moyenne mobile.
    
    L'OBV accumule le volume selon la direction du prix :
    - Volume ajouté si clôture > clôture précédente
    - Volume soustrait si clôture < clôture précédente
    - Volume neutre si clôture = clôture précédente
    
    Signaux générés:
    - BUY: OBV croise au-dessus de sa MA + confirmations haussières
    - SELL: OBV croise en-dessous de sa MA + confirmations baissières
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres OBV optimisés pour éviter surtrading
        self.min_obv_ma_distance = 0.01  # Distance minimum OBV/MA (réduit pour plus de signaux)
        self.volume_confirmation_threshold = 1.2  # Volume minimum requis (assoupli)
        self.trend_alignment_bonus = 0.12  # Bonus réduit
        # Nouveaux filtres anti-bruit
        self.min_confidence_threshold = 0.45  # Confidence minimum pour valider
        self.strong_separation_threshold = 0.05  # Séparation forte OBV/MA
        self.require_volume_confirmation = True  # Exiger confirmation volume
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs OBV et volume."""
        return {
            # OBV et sa moyenne mobile
            'obv': self.indicators.get('obv'),
            'obv_ma_10': self.indicators.get('obv_ma_10'),
            'obv_oscillator': self.indicators.get('obv_oscillator'),
            'ad_line': self.indicators.get('ad_line'),  # Accumulation/Distribution Line
            # Volume pour confirmation
            'volume_ratio': self.indicators.get('volume_ratio'),
            'avg_volume_20': self.indicators.get('avg_volume_20'),
            'quote_volume_ratio': self.indicators.get('quote_volume_ratio'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            'relative_volume': self.indicators.get('relative_volume'),
            # Contexte prix pour confirmation tendance
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            # VWAP pour contexte institutional
            'vwap_10': self.indicators.get('vwap_10'),
            'anchored_vwap': self.indicators.get('anchored_vwap'),
            # Momentum pour confluence
            'rsi_14': self.indicators.get('rsi_14'),
            'momentum_score': self.indicators.get('momentum_score'),
            # Structure de marché
            'market_regime': self.indicators.get('market_regime'),
            'support_levels': self.indicators.get('support_levels'),
            'resistance_levels': self.indicators.get('resistance_levels'),
            # Confluence
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score')
        }
        
    def _get_current_price(self) -> Optional[float]:
        """Récupère le prix actuel depuis les données OHLCV."""
        try:
            if self.data and 'close' in self.data and self.data['close']:
                return float(self.data['close'][-1])
        except (IndexError, ValueError, TypeError):
            pass
        return None
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les croisements OBV/MA.
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
        current_price = self._get_current_price()
        
        # Vérification des indicateurs OBV essentiels
        try:
            obv = float(values['obv']) if values['obv'] is not None else None
            obv_ma = float(values['obv_ma_10']) if values['obv_ma_10'] is not None else None
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion OBV: {e}",
                "metadata": {"strategy": self.name}
            }
            
        if obv is None or obv_ma is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "OBV ou OBV MA non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # Analyse du croisement OBV/MA
        obv_above_ma = obv > obv_ma
        obv_distance = abs(obv - obv_ma) / abs(obv_ma) if obv_ma != 0 else 0
        
        # Vérification que les lignes ne sont pas trop proches (éviter faux signaux)
        if obv_distance < self.min_obv_ma_distance:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"OBV trop proche MA ({obv_distance:.4f}) - pas de signal clair",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "obv": obv,
                    "obv_ma": obv_ma,
                    "distance": obv_distance
                }
            }
            
        signal_side = None
        reason = ""
        base_confidence = 0.50  # Standardisé à 0.50 pour équité avec autres stratégies
        confidence_boost = 0.0
        cross_type = None
        
        # NOUVEAU: Filtres anti-surtrading préliminaires
        market_regime = values.get('market_regime')
        if market_regime in ['TRANSITION', 'VOLATILE']:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"OBV ignoré en marché {market_regime}",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "market_regime": market_regime,
                    "filter": "regime_filter"
                }
            }
            
        # Filtre volume obligatoire si activé
        if self.require_volume_confirmation:
            volume_ratio = values.get('volume_ratio')
            if volume_ratio is None or float(volume_ratio) < self.volume_confirmation_threshold:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Volume insuffisant pour OBV ({volume_ratio:.1f}x < {self.volume_confirmation_threshold}x)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "volume_ratio": volume_ratio,
                        "filter": "volume_filter"
                    }
                }
        
        # NOUVEAU: Logique OBV avec conditions strictes
        strong_signal_conditions = 0
        weak_signal_penalty = 0
        
        # Vérification de la tendance globale AVANT signal
        trend_strength = values.get('trend_strength')
        if trend_strength and str(trend_strength).lower() in ['absent', 'weak']:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Tendance trop faible ({trend_strength}) pour OBV",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "trend_strength": trend_strength,
                    "filter": "trend_filter"
                }
            }
        
        # Logique de croisement OBV avec validation
        if obv_above_ma:
            # Potentiel BUY - mais vérifier conditions
            directional_bias = values.get('directional_bias')
            if directional_bias == 'BEARISH':
                # Rejeté : OBV haussier mais bias baissier
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "OBV BUY rejeté - bias baissier dominant",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "directional_bias": directional_bias,
                        "filter": "bias_filter"
                    }
                }
            
            signal_side = "BUY"
            cross_type = "bullish_cross"
            reason = f"OBV ({obv:.0f}) > MA ({obv_ma:.0f})"
            confidence_boost += 0.10  # Réduit de 0.15
            
            # Compter conditions favorables
            if directional_bias == 'BULLISH':
                strong_signal_conditions += 1
                confidence_boost += 0.12
                reason += " + bias haussier"
            else:
                weak_signal_penalty += 1
                
        else:
            # Potentiel SELL - mais vérifier conditions  
            directional_bias = values.get('directional_bias')
            if directional_bias == 'BULLISH':
                # Rejeté : OBV baissier mais bias haussier
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "OBV SELL rejeté - bias haussier dominant",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "directional_bias": directional_bias,
                        "filter": "bias_filter"
                    }
                }
            
            signal_side = "SELL"
            cross_type = "bearish_cross"
            reason = f"OBV ({obv:.0f}) < MA ({obv_ma:.0f})"
            confidence_boost += 0.10  # Réduit de 0.15
            
            # Compter conditions favorables
            if directional_bias == 'BEARISH':
                strong_signal_conditions += 1
                confidence_boost += 0.12
                reason += " + bias baissier"
            else:
                weak_signal_penalty += 1
        
        # NOUVEAU: Exiger au moins 1 condition forte
        if strong_signal_conditions == 0:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal OBV {signal_side} trop faible - manque confirmations",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "strong_conditions": strong_signal_conditions,
                    "weak_penalties": weak_signal_penalty,
                    "filter": "insufficient_confirmation"
                }
            }
            
        # Bonus selon la force de la séparation - SEUILS PLUS STRICTS
        if obv_distance >= self.strong_separation_threshold:  # 0.05
            confidence_boost += 0.18
            reason += f" - séparation TRÈS forte ({obv_distance:.3f})"
        elif obv_distance >= 0.03:
            confidence_boost += 0.12
            reason += f" - séparation forte ({obv_distance:.3f})"
        elif obv_distance >= 0.025:
            confidence_boost += 0.06
            reason += f" - séparation modérée ({obv_distance:.3f})"
        else:
            # Séparation trop faible - pénalité
            confidence_boost -= 0.05
            reason += f" ATTENTION: séparation faible ({obv_distance:.3f})"
            
        # Confirmation avec OBV Oscillator
        obv_oscillator = values.get('obv_oscillator')
        if obv_oscillator is not None:
            try:
                obv_osc = float(obv_oscillator)
                if signal_side == "BUY" and obv_osc > 0:
                    confidence_boost += 0.15
                    reason += f" + OBV osc positif ({obv_osc:.3f})"
                elif signal_side == "SELL" and obv_osc < 0:
                    confidence_boost += 0.15
                    reason += f" + OBV osc négatif ({obv_osc:.3f})"
                else:
                    confidence_boost -= 0.05
                    reason += f" mais OBV osc défavorable ({obv_osc:.3f})"
            except (ValueError, TypeError):
                pass
                
        # Volume Ratio - SEUILS PLUS STRICTS
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.volume_confirmation_threshold * 1.5:  # 2.25x
                    confidence_boost += 0.20
                    reason += f" + volume EXCEPTIONNEL ({vol_ratio:.1f}x)"
                elif vol_ratio >= self.volume_confirmation_threshold:  # 1.5x
                    confidence_boost += 0.15
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.2:
                    confidence_boost += 0.08
                    reason += f" + volume modéré ({vol_ratio:.1f}x)"
                else:
                    confidence_boost -= 0.10  # Pénalité augmentée
                    reason += f" MAIS volume trop faible ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec A/D Line (Accumulation/Distribution)
        ad_line = values.get('ad_line')
        if ad_line is not None and obv is not None:
            try:
                ad_val = float(ad_line)
                # Vérifier si OBV et A/D Line s'alignent
                obv_direction = 1 if obv > 0 else -1
                ad_direction = 1 if ad_val > 0 else -1
                
                if (signal_side == "BUY" and obv_direction == ad_direction == 1) or \
                   (signal_side == "SELL" and obv_direction == ad_direction == -1):
                    confidence_boost += 0.12
                    reason += " + A/D Line confirme"
                elif obv_direction != ad_direction:
                    confidence_boost -= 0.08
                    reason += " mais A/D Line diverge"
            except (ValueError, TypeError):
                pass
                
        # Alignement avec tendance prix pour confirmation
        ema_12 = values.get('ema_12')
        ema_26 = values.get('ema_26')
        if ema_12 is not None and ema_26 is not None and current_price is not None:
            try:
                ema12_val = float(ema_12)
                ema26_val = float(ema_26)
                price_trend_bullish = current_price > ema12_val > ema26_val
                price_trend_bearish = current_price < ema12_val < ema26_val
                
                if (signal_side == "BUY" and price_trend_bullish) or \
                   (signal_side == "SELL" and price_trend_bearish):
                    confidence_boost += self.trend_alignment_bonus
                    reason += " + tendance prix alignée"
                elif (signal_side == "BUY" and price_trend_bearish) or \
                     (signal_side == "SELL" and price_trend_bullish):
                    confidence_boost -= 0.10
                    reason += " mais tendance prix diverge"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec EMA 50 pour filtre de tendance
        ema_50 = values.get('ema_50')
        if ema_50 is not None and current_price is not None:
            try:
                ema50_val = float(ema_50)
                if signal_side == "BUY" and current_price > ema50_val:
                    confidence_boost += 0.08
                    reason += " + prix > EMA50"
                elif signal_side == "SELL" and current_price < ema50_val:
                    confidence_boost += 0.08
                    reason += " + prix < EMA50"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec VWAP (institutional level)
        vwap_10 = values.get('vwap_10')
        if vwap_10 is not None and current_price is not None:
            try:
                vwap_val = float(vwap_10)
                if signal_side == "BUY" and current_price > vwap_val:
                    confidence_boost += 0.10
                    reason += " + prix > VWAP"
                elif signal_side == "SELL" and current_price < vwap_val:
                    confidence_boost += 0.10
                    reason += " + prix < VWAP"
            except (ValueError, TypeError):
                pass
                
        # Trend strength déjà vérifié plus haut, bonus supplémentaire
        if trend_strength is not None:
            trend_str = str(trend_strength).lower()
            if trend_str in ['strong', 'very_strong']:
                confidence_boost += 0.10  # Réduit de 0.12
                reason += f" + tendance {trend_str}"
            elif trend_str == 'moderate':
                confidence_boost += 0.06  # Réduit de 0.08
                reason += f" + tendance {trend_str}"
                
        # Directional bias déjà vérifié plus haut, pas de double bonus
                
        # Confirmation avec qualité du volume (format 0-100)
        volume_quality_score = values.get('volume_quality_score')
        if volume_quality_score is not None:
            try:
                vol_quality = float(volume_quality_score)
                if vol_quality > 70:
                    confidence_boost += 0.10
                    reason += f" + volume qualité ({vol_quality:.0f})"
                elif vol_quality < 30:
                    confidence_boost -= 0.05
                    reason += f" mais volume faible qualité ({vol_quality:.0f})"
            except (ValueError, TypeError):
                pass
                
        # Trade intensity pour confirmation
        trade_intensity = values.get('trade_intensity')
        if trade_intensity is not None:
            try:
                intensity = float(trade_intensity)
                if intensity > 1.5:
                    confidence_boost += 0.08
                    reason += " + intensité élevée"
                elif intensity < 0.8:
                    confidence_boost -= 0.05
                    reason += " mais intensité faible"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec RSI (éviter zones extrêmes)
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if signal_side == "BUY" and 30 <= rsi <= 70:
                    confidence_boost += 0.08
                    reason += " + RSI favorable"
                elif signal_side == "SELL" and 30 <= rsi <= 70:
                    confidence_boost += 0.08
                    reason += " + RSI favorable"
                elif signal_side == "BUY" and rsi >= 80:
                    confidence_boost -= 0.10
                    reason += " mais RSI surachat"
                elif signal_side == "SELL" and rsi <= 20:
                    confidence_boost -= 0.10
                    reason += " mais RSI survente"
            except (ValueError, TypeError):
                pass
                
        # Market regime - FILTRES PLUS STRICTS
        if market_regime:
            regime_str = str(market_regime).upper()
            # Bonus si aligné avec la tendance
            if (signal_side == "BUY" and regime_str in ["TRENDING_BULL", "BREAKOUT_BULL"]) or \
               (signal_side == "SELL" and regime_str in ["TRENDING_BEAR", "BREAKOUT_BEAR"]):
                confidence_boost += 0.15
                reason += f" (aligné {regime_str.lower()})"
            elif regime_str == "RANGING":
                confidence_boost -= 0.15  # Pénalité augmentée
                reason += " (marché ranging défavorable)"
            elif (signal_side == "BUY" and regime_str in ["TRENDING_BEAR", "BREAKOUT_BEAR"]) or \
                 (signal_side == "SELL" and regime_str in ["TRENDING_BULL", "BREAKOUT_BULL"]):
                confidence_boost -= 0.12
                reason += f" MAIS contre-tendance ({regime_str.lower()})"
            
        # Support/Resistance pour contexte
        if signal_side == "BUY":
            support_levels = values.get('support_levels')
            if support_levels is not None and current_price is not None:
                try:
                    # Supposer que support_levels est une liste ou valeur proche
                    confidence_boost += 0.05
                    reason += " + près support"
                except (ValueError, TypeError):
                    pass
        elif signal_side == "SELL":
            resistance_levels = values.get('resistance_levels')
            if resistance_levels is not None and current_price is not None:
                try:
                    # Supposer que resistance_levels est une liste ou valeur proche
                    confidence_boost += 0.05
                    reason += " + près résistance"
                except (ValueError, TypeError):
                    pass
                    
        # Signal strength (VARCHAR: WEAK/MODERATE/STRONG)
        signal_strength_calc = values.get('signal_strength')
        if signal_strength_calc is not None:
            sig_str = str(signal_strength_calc).upper()
            if sig_str == 'STRONG':
                confidence_boost += 0.10
                reason += " + signal fort"
            elif sig_str == 'MODERATE':
                confidence_boost += 0.05
                reason += " + signal modéré"
                
        # Confluence - SEUILS PLUS STRICTS
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if confluence > 70:  # Augmenté de 60
                    confidence_boost += 0.15
                    reason += f" + confluence EXCELLENTE ({confluence:.0f})"
                elif confluence > 55:  # Augmenté de 45
                    confidence_boost += 0.10
                    reason += f" + confluence élevée ({confluence:.0f})"
                elif confluence > 40:
                    confidence_boost += 0.05
                    reason += f" + confluence correcte ({confluence:.0f})"
                elif confluence < 35:  # NOUVEAU: Pénalité si faible
                    confidence_boost -= 0.08
                    reason += f" mais confluence FAIBLE ({confluence:.0f})"
            except (ValueError, TypeError):
                pass
        
        # NOUVEAU: Filtre final OBLIGATOIRE - confidence minimum
        raw_confidence = base_confidence * (1 + confidence_boost)
        if raw_confidence < self.min_confidence_threshold:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal OBV {signal_side} rejeté - confidence insuffisante ({raw_confidence:.2f} < {self.min_confidence_threshold})",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "rejected_signal": signal_side,
                    "raw_confidence": raw_confidence,
                    "min_required": self.min_confidence_threshold,
                    "obv_distance": obv_distance
                }
            }
                
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
                "current_price": current_price,
                "obv": obv,
                "obv_ma_10": obv_ma,
                "obv_distance": obv_distance,
                "cross_type": cross_type,
                "obv_oscillator": obv_oscillator,
                "ad_line": ad_line,
                "volume_ratio": volume_ratio,
                "volume_quality_score": volume_quality_score,
                "trade_intensity": trade_intensity,
                "ema_12": ema_12,
                "ema_26": ema_26,
                "ema_50": ema_50,
                "vwap_10": vwap_10,
                "trend_strength": trend_strength,
                "directional_bias": directional_bias,
                "rsi_14": rsi_14,
                "market_regime": market_regime,
                "confluence_score": confluence_score
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs OBV requis sont présents."""
        if not super().validate_data():
            return False
            
        required = ['obv', 'obv_ma_10']
        
        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        return True
