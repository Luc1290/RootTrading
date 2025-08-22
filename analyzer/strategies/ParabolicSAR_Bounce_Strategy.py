"""
ParabolicSAR_Bounce_Strategy - Stratégie basée sur les rebonds du prix sur le Parabolic SAR.
"""

from typing import Dict, Any, Optional, Tuple
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class ParabolicSAR_Bounce_Strategy(BaseStrategy):
    """
    Stratégie utilisant les rebonds du prix sur le Parabolic SAR pour détecter des retournements.
    
    Le Parabolic SAR (Stop and Reverse) suit la tendance et s'inverse quand le prix croise le SAR :
    - SAR sous le prix = tendance haussière
    - SAR au-dessus du prix = tendance baissière
    - Rebond = prix s'approche du SAR puis repart dans la direction de la tendance
    
    Signaux générés:
    - BUY: Prix rebondit sur SAR en tendance haussière (SAR < prix) + confirmations
    - SELL: Prix rebondit sur SAR en tendance baissière (SAR > prix) + confirmations
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres Parabolic SAR ANTI-SPAM - ULTRA STRICTS
        self.min_sar_distance = 0.003  # Distance minimum resserrée crypto (0.3%)
        self.max_sar_distance = 0.02   # Distance maximum STRICTE (2% max pour crypto 3m)
        self.trend_confirmation_bonus = 0.10  # Bonus réduit (éviter sur-confiance)
        self.volume_confirmation_threshold = 1.1  # Seuil volume assoupli (était 1.25)
        
        # FILTRES REBONDS AJUSTÉS AUX CONDITIONS RÉELLES (CORRIGÉ)
        self.min_confluence_required = 45    # Confluence minimum réaliste (était 65)
        self.min_adx_required = 18           # ADX minimum assoupli (était 22)
        self.min_pattern_confidence = 35     # Pattern confidence adapté (était 60)
        self.required_confirmations = 2      # Confirmations atteignables (était 3)
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs SAR et contexte."""
        return {
            # Parabolic SAR (à implémenter si disponible)
            # Note: Le SAR n'est pas listé dans les 108 indicateurs, utilisons d'autres
            # On utilisera les niveaux de support/résistance comme proxy pour rebonds
            'support_levels': self.indicators.get('support_levels'),
            'resistance_levels': self.indicators.get('resistance_levels'),
            'nearest_support': self.indicators.get('nearest_support'),
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            'support_strength': self.indicators.get('support_strength'),
            'resistance_strength': self.indicators.get('resistance_strength'),
            'break_probability': self.indicators.get('break_probability'),
            # Tendance pour déterminer la direction attendue du rebond
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            'trend_angle': self.indicators.get('trend_angle'),
            # EMA pour contexte de tendance
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            # ADX pour force de tendance
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            # Volume pour confirmation rebond
            'volume_ratio': self.indicators.get('volume_ratio'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            # Oscillateurs pour timing
            'rsi_14': self.indicators.get('rsi_14'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            'williams_r': self.indicators.get('williams_r'),
            # ATR pour volatilité et stops
            'atr_14': self.indicators.get('atr_14'),
            'atr_stop_long': self.indicators.get('atr_stop_long'),
            'atr_stop_short': self.indicators.get('atr_stop_short'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            # VWAP institutional levels
            'vwap_10': self.indicators.get('vwap_10'),
            'anchored_vwap': self.indicators.get('anchored_vwap'),
            # Market structure
            'market_regime': self.indicators.get('market_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
            'momentum_score': self.indicators.get('momentum_score'),
            # Pattern et confluence
            'pattern_detected': self.indicators.get('pattern_detected'),
            'pattern_confidence': self.indicators.get('pattern_confidence'),
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
        
    def _get_recent_prices(self) -> Optional[list]:
        """Récupère les derniers prix pour analyser le mouvement."""
        try:
            if self.data and 'close' in self.data and len(self.data['close']) >= 3:
                return [float(p) for p in self.data['close'][-3:]]
        except (IndexError, ValueError, TypeError):
            pass
        return None
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les rebonds sur niveaux SAR/Support/Résistance.
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
        recent_prices = self._get_recent_prices()
        
        if current_price is None or recent_prices is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # VALIDATION ANTI-SPAM PRÉLIMINAIRE
        
        # 1. Vérifier confluence OBLIGATOIRE
        confluence_score = values.get('confluence_score', 0)
        if not confluence_score or float(confluence_score) < self.min_confluence_required:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Confluence insuffisante ({confluence_score}) < {self.min_confluence_required} - rebond rejeté",
                "metadata": {"strategy": self.name, "rejected_reason": "low_confluence"}
            }
            
        # 2. Vérifier ADX OBLIGATOIRE (tendance claire requise pour rebonds fiables)
        adx_14 = values.get('adx_14')
        if not adx_14 or float(adx_14) < self.min_adx_required:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"ADX insuffisant ({adx_14}) < {self.min_adx_required} - pas de tendance claire",
                "metadata": {"strategy": self.name, "rejected_reason": "weak_trend"}
            }
            
        # 3. Déterminer la tendance principale
        trend_direction = self._get_trend_direction(values, current_price)
        if trend_direction is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Tendance non claire après validation",
                "metadata": {"strategy": self.name}
            }
            
        # Analyser les rebonds selon la tendance
        if trend_direction == "bullish":
            return self._analyze_bullish_bounce(values, current_price, recent_prices)
        else:
            return self._analyze_bearish_bounce(values, current_price, recent_prices)
            
    def _get_trend_direction(self, values: Dict[str, Any], current_price: float) -> Optional[str]:
        """Détermine la direction de la tendance principale."""
        try:
            # Vérifier directional_bias d'abord
            directional_bias = values.get('directional_bias')
            if directional_bias in ["BULLISH", "BEARISH"]:
                return directional_bias.lower()
                
            # Utiliser les EMA comme fallback
            ema_12 = values.get('ema_12')
            ema_26 = values.get('ema_26')
            ema_50 = values.get('ema_50')
            
            if all(x is not None for x in [ema_12, ema_26, ema_50]):
                ema12_val = float(ema_12) if ema_12 is not None else 0.0
                ema26_val = float(ema_26) if ema_26 is not None else 0.0
                ema50_val = float(ema_50) if ema_50 is not None else 0.0
                
                # Tendance haussière: prix > EMA12 > EMA26 > EMA50
                if current_price > ema12_val > ema26_val > ema50_val:
                    return "bullish"
                # Tendance baissière: prix < EMA12 < EMA26 < EMA50
                elif current_price < ema12_val < ema26_val < ema50_val:
                    return "bearish"
                    
            # Utiliser ADX/DI comme dernier recours
            plus_di = values.get('plus_di')
            minus_di = values.get('minus_di')
            if plus_di is not None and minus_di is not None:
                plus_val = float(plus_di)
                minus_val = float(minus_di)
                if plus_val > minus_val:
                    return "bullish"
                elif minus_val > plus_val:
                    return "bearish"
                    
        except (ValueError, TypeError):
            pass
            
        return None
        
    def _analyze_bullish_bounce(self, values: Dict[str, Any], current_price: float, recent_prices: list) -> Dict[str, Any]:
        """Analyse un rebond haussier sur support."""
        signal_side = "BUY"
        reason = ""
        base_confidence = 0.50  # Standardisé à 0.50 pour équité avec autres stratégies
        confidence_boost = 0.0
        bounce_level = None
        
        # Chercher le niveau de rebond (support ou EMA)
        nearest_support = values.get('nearest_support')
        ema_50 = values.get('ema_50')
        
        if nearest_support is not None:
            try:
                support_val = float(nearest_support)
                distance_to_support = abs(current_price - support_val) / current_price
                
                if distance_to_support <= self.max_sar_distance:
                    bounce_level = support_val
                    level_type = "support"
                    reason = f"Rebond sur support {support_val:.4f}"
                    confidence_boost += 0.15
                    
                    # Vérifier que le prix a effectivement rebondi
                    if self._detect_bounce_pattern(recent_prices, support_val, "bullish"):
                        confidence_boost += 0.18  # RÉDUIT de 0.25 à 0.18 (moins généreux)
                        reason += " - rebond CONFIRMÉ"
                    else:
                        # NOUVEAU: Rejeter si pas de rebond confirmé
                        return {
                            "side": None,
                            "confidence": 0.0,
                            "strength": "weak",
                            "reason": f"Proche support mais rebond NON confirmé - rejeté",
                            "metadata": {"strategy": self.name, "rejected_reason": "no_bounce_pattern"}
                        }
            except (ValueError, TypeError):
                pass
                
        # Si pas de support clair, utiliser EMA50 comme niveau SAR
        if bounce_level is None and ema_50 is not None:
            try:
                ema50_val = float(ema_50)
                if current_price > ema50_val:  # Prix au-dessus de EMA50 (tendance haussière)
                    distance_to_ema = abs(current_price - ema50_val) / current_price
                    
                    if distance_to_ema <= self.max_sar_distance:
                        bounce_level = ema50_val
                        level_type = "ema50"
                        reason = f"Rebond sur EMA50 {ema50_val:.4f}"
                        confidence_boost += 0.12
                        
                        if self._detect_bounce_pattern(recent_prices, ema50_val, "bullish"):
                            confidence_boost += 0.14  # RÉDUIT de 0.18 à 0.14
                            reason += " - rebond CONFIRMÉ"
                        else:
                            # NOUVEAU: Rejeter rebond EMA non confirmé aussi
                            return {
                                "side": None,
                                "confidence": 0.0,
                                "strength": "weak",
                                "reason": f"Proche EMA50 mais rebond NON confirmé - rejeté",
                                "metadata": {"strategy": self.name, "rejected_reason": "no_ema_bounce_pattern"}
                            }
            except (ValueError, TypeError):
                pass
                
        if bounce_level is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Aucun niveau de rebond détecté",
                "metadata": {"strategy": self.name}
            }
            
        # S'assurer que reason est défini (au cas où aucune condition ci-dessus ne l'a défini)
        if 'reason' not in locals() or not reason:
            reason = f"Rebond potentiel détecté niveau {bounce_level:.4f}"
            
        # Confirmations additionnelles
        additional_boost, reason_additions = self._add_confirmations(values, signal_side, current_price)
        confidence_boost += additional_boost
        reason += reason_additions
        
        # SYSTÈME CONFIRMATIONS OBLIGATOIRES ANTI-SPAM
        confirmations_count = self._count_confirmations(values, signal_side, current_price)
        if confirmations_count < self.required_confirmations:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Confirmations insuffisantes rebond ({confirmations_count}/{self.required_confirmations}) - rejeté",
                "metadata": {
                    "strategy": self.name,
                    "rejected_reason": "insufficient_confirmations",
                    "confirmations_count": confirmations_count
                }
            }
        
        # FILTRE FINAL CONFIANCE ASSOUPLI - rejeter si confidence trop faible
        raw_confidence = base_confidence * (1 + confidence_boost)
        if raw_confidence < 0.45:  # ASSOUPLI: Seuil 45% (était 60%) pour rebonds BUY
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal rebond rejeté - confiance insuffisante ({raw_confidence:.2f} < 0.45)",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "rejected_signal": signal_side,
                    "raw_confidence": raw_confidence,
                    "bounce_level": bounce_level,
                    "trend_direction": "bullish",
                    "confirmations_count": confirmations_count
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
                "bounce_level": bounce_level,
                "level_type": level_type,
                "distance_to_level": abs(current_price - bounce_level) / current_price,
                "trend_direction": "bullish",
                "recent_prices": recent_prices,
                **self._get_metadata_values(values)
            }
        }
        
    def _analyze_bearish_bounce(self, values: Dict[str, Any], current_price: float, recent_prices: list) -> Dict[str, Any]:
        """Analyse un rebond baissier sur résistance."""
        signal_side = "SELL"
        reason = ""
        base_confidence = 0.50  # Standardisé à 0.50 pour équité avec autres stratégies
        confidence_boost = 0.0
        bounce_level = None
        
        # Chercher le niveau de rebond (résistance ou EMA)
        nearest_resistance = values.get('nearest_resistance')
        ema_50 = values.get('ema_50')
        
        if nearest_resistance is not None:
            try:
                resistance_val = float(nearest_resistance)
                distance_to_resistance = abs(current_price - resistance_val) / current_price
                
                if distance_to_resistance <= self.max_sar_distance:
                    bounce_level = resistance_val
                    level_type = "resistance"
                    reason = f"Rebond sur résistance {resistance_val:.4f}"
                    confidence_boost += 0.15
                    
                    if self._detect_bounce_pattern(recent_prices, resistance_val, "bearish"):
                        confidence_boost += 0.18  # RÉDUIT de 0.25 à 0.18
                        reason += " - rebond CONFIRMÉ"
                    else:
                        # NOUVEAU: Rejeter si pas de rebond confirmé (même logique que BUY)
                        return {
                            "side": None,
                            "confidence": 0.0,
                            "strength": "weak",
                            "reason": f"Proche résistance mais rebond NON confirmé - rejeté",
                            "metadata": {"strategy": self.name, "rejected_reason": "no_bounce_pattern"}
                        }
            except (ValueError, TypeError):
                pass
                
        # Si pas de résistance claire, utiliser EMA50 comme niveau SAR
        if bounce_level is None and ema_50 is not None:
            try:
                ema50_val = float(ema_50)
                if current_price < ema50_val:  # Prix en-dessous de EMA50 (tendance baissière)
                    distance_to_ema = abs(current_price - ema50_val) / current_price
                    
                    if distance_to_ema <= self.max_sar_distance:
                        bounce_level = ema50_val
                        level_type = "ema50"
                        reason = f"Rebond sur EMA50 {ema50_val:.4f}"
                        confidence_boost += 0.12
                        
                        if self._detect_bounce_pattern(recent_prices, ema50_val, "bearish"):
                            confidence_boost += 0.14  # RÉDUIT de 0.18 à 0.14 (cohérent avec BUY)
                            reason += " - rebond CONFIRMÉ"
                        else:
                            # NOUVEAU: Rejeter rebond EMA non confirmé (cohérent avec BUY)
                            return {
                                "side": None,
                                "confidence": 0.0,
                                "strength": "weak",
                                "reason": f"Proche EMA50 mais rebond NON confirmé - rejeté",
                                "metadata": {"strategy": self.name, "rejected_reason": "no_ema_bounce_pattern"}
                            }
            except (ValueError, TypeError):
                pass
                
        if bounce_level is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Aucun niveau de rebond détecté",
                "metadata": {"strategy": self.name}
            }
            
        # S'assurer que reason est défini (au cas où aucune condition ci-dessus ne l'a défini)
        if 'reason' not in locals() or not reason:
            reason = f"Rebond potentiel détecté niveau {bounce_level:.4f}"
            
        # Confirmations additionnelles  
        additional_boost, reason_additions = self._add_confirmations(values, signal_side, current_price)
        confidence_boost += additional_boost
        reason += reason_additions
        
        # FILTRE FINAL CONFIANCE ASSOUPLI SELL - rejeter si confidence trop faible
        raw_confidence = base_confidence * (1 + confidence_boost)
        if raw_confidence < 0.40:  # ASSOUPLI: Seuil 40% (était 48%) pour rebonds SELL
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal rebond rejeté - confiance insuffisante ({raw_confidence:.2f} < 0.40)",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "rejected_signal": signal_side,
                    "raw_confidence": raw_confidence,
                    "bounce_level": bounce_level,
                    "trend_direction": "bearish"
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
                "bounce_level": bounce_level,
                "level_type": level_type,
                "distance_to_level": abs(current_price - bounce_level) / current_price,
                "trend_direction": "bearish",
                "recent_prices": recent_prices,
                **self._get_metadata_values(values)
            }
        }
        
    def _detect_bounce_pattern(self, recent_prices: list, level: float, direction: str) -> bool:
        """Détecte si les prix récents montrent un pattern de rebond."""
        if len(recent_prices) < 3:
            return False
            
        try:
            price_1, price_2, price_3 = recent_prices[-3:]
            
            if direction == "bullish":
                # Pour rebond haussier CRYPTO: détection ultra-stricte
                proximity_threshold = 0.008  # ULTRA STRICT: 0.8% (vs 1.5% avant)
                rebound_strength = (price_3 - price_2) / price_2  # Force du rebond
                return (price_1 > level and 
                        abs(price_2 - level) / level < proximity_threshold and  # Ultra strict
                        price_3 > price_2 and  # Prix remonte
                        rebound_strength > 0.005)  # Rebond minimum relevé à 0.5%
            else:  # bearish
                # Pour rebond baissier CRYPTO: détection ultra-stricte
                proximity_threshold = 0.008  # ULTRA STRICT: 0.8% (vs 1.5% avant)
                rebound_strength = (price_2 - price_3) / price_2  # Force du rebond
                return (price_1 < level and
                        abs(price_2 - level) / level < proximity_threshold and  # Ultra strict
                        price_3 < price_2 and  # Prix redescend
                        rebound_strength > 0.005)  # Rebond minimum relevé à 0.5%
        except (ValueError, TypeError, ZeroDivisionError):
            return False
            
    def _add_confirmations(self, values: Dict[str, Any], signal_side: str, current_price: float) -> Tuple[float, str]:
        """Ajoute les confirmations pour renforcer le signal."""
        boost = 0.0
        reason_additions = ""
        
        # Confirmation avec trend_strength (VARCHAR: weak/moderate/strong/very_strong/extreme)
        trend_strength = values.get('trend_strength')
        if trend_strength is not None:
            trend_str = str(trend_strength).lower()
            if trend_str in ['extreme', 'very_strong']:
                boost += 0.20  # AUGMENTÉ - trend fort = rebonds fiables
                reason_additions += f" + trend {trend_str}"
            elif trend_str == 'strong':
                boost += 0.15  # AUGMENTÉ
                reason_additions += f" + trend {trend_str}"
            elif trend_str == 'moderate':
                boost += 0.10  # AUGMENTÉ
                reason_additions += f" + trend {trend_str}"
            elif trend_str in ['weak', 'absent']:  # NOUVEAU: pénalité
                boost -= 0.10
                reason_additions += f" MAIS trend {trend_str}"
                
        # Confirmation avec ADX (force de tendance)
        adx_14 = values.get('adx_14')
        if adx_14 is not None:
            try:
                adx = float(adx_14)
                # ADX PLUS STRICT pour rebonds
                if adx > 30:  # Tendance très forte
                    boost += 0.18  # AUGMENTÉ
                    reason_additions += f" + ADX fort ({adx:.0f})"
                elif adx > 25:  # Tendance forte
                    boost += 0.12
                    reason_additions += f" + ADX correct ({adx:.0f})"
                elif adx > 20:  # Tendance modérée
                    boost += 0.06
                    reason_additions += f" + ADX faible ({adx:.0f})"
                else:  # ADX faible = pas de tendance
                    boost -= 0.08
                    reason_additions += f" MAIS ADX faible ({adx:.0f})"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec volume
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                # VOLUME PLUS STRICT - rebonds nécessitent conviction
                if vol_ratio >= self.volume_confirmation_threshold * 1.5:  # Volume très élevé
                    boost += 0.20
                    reason_additions += f" + volume TRÈS élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= self.volume_confirmation_threshold:
                    boost += 0.15
                    reason_additions += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.1:
                    boost += 0.08
                    reason_additions += f" + volume modéré ({vol_ratio:.1f}x)"
                else:
                    boost -= 0.12  # Pénalité augmentée - volume crucial pour rebonds
                    reason_additions += f" ATTENTION: volume FAIBLE ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec RSI (éviter zones extrêmes)
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                # RSI OPTIMISÉ pour rebonds
                if signal_side == "BUY" and 25 <= rsi <= 60:  # Zone idéale rebond haussier
                    boost += 0.12
                    reason_additions += f" + RSI idéal rebond ({rsi:.1f})"
                elif signal_side == "SELL" and 40 <= rsi <= 75:  # Zone idéale rebond baissier
                    boost += 0.12
                    reason_additions += f" + RSI idéal rebond ({rsi:.1f})"
                elif (signal_side == "BUY" and rsi >= 75) or (signal_side == "SELL" and rsi <= 25):
                    boost -= 0.18  # Pénalité augmentée - zones extrêmes
                    reason_additions += f" ATTENTION: RSI zone extrême ({rsi:.1f})"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec Williams %R
        williams_r = values.get('williams_r')
        if williams_r is not None:
            try:
                wr = float(williams_r)
                if signal_side == "BUY" and -80 <= wr <= -20:
                    boost += 0.08
                elif signal_side == "SELL" and -80 <= wr <= -20:
                    boost += 0.08
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec VWAP
        vwap_10 = values.get('vwap_10')
        if vwap_10 is not None:
            try:
                vwap = float(vwap_10)
                if signal_side == "BUY" and current_price > vwap:
                    boost += 0.08
                elif signal_side == "SELL" and current_price < vwap:
                    boost += 0.08
            except (ValueError, TypeError):
                pass
                
        # CORRECTION MAGISTRALE: Market regime avec psychologie des rebonds
        market_regime = values.get('market_regime')
        regime_strength = values.get('regime_strength')
        trend_strength = values.get('trend_strength')
        
        if market_regime is not None:
            regime_str = str(regime_strength).upper() if regime_strength else "WEAK"
            trend_str = str(trend_strength).lower() if trend_strength else "weak"
                
            if signal_side == "BUY":
                # Rebond haussier : trend fort > ranging > trend faible
                if market_regime in ["TRENDING_BULL", "TRENDING_BEAR"]:
                    if regime_str == "EXTREME" and trend_str in ["extreme", "very_strong"]:
                        boost += 0.20
                        reason_additions += f" + trend très fort rebond ({regime_str})"
                    elif regime_str == "STRONG":
                        boost += 0.15
                        reason_additions += f" + trend fort rebond haussier ({regime_str})"
                    elif regime_str == "MODERATE":
                        boost += 0.10
                        reason_additions += f" + trend modéré rebond ({regime_str})"
                    else:  # WEAK
                        boost += 0.05
                        reason_additions += f" + trend faible rebond ({regime_str})"
                elif market_regime == "RANGING":
                    if regime_str in ["EXTREME", "STRONG"]:
                        boost += 0.12
                        reason_additions += f" + range fort rebond support ({regime_str})"
                    else:
                        boost += 0.08
                        reason_additions += f" + range rebond ({regime_str})"
                elif market_regime == "VOLATILE":
                    boost -= 0.05
                    reason_additions += " mais marché chaotique"
                    
            else:  # SELL
                # Rebond baissier : ranging > trend baissier > trend haussier
                if market_regime == "RANGING":
                    if regime_str == "EXTREME":
                        boost += 0.18
                        reason_additions += f" + range parfait résistance ({regime_str})"
                    elif regime_str == "STRONG":
                        boost += 0.14
                        reason_additions += f" + range fort rebond résistance ({regime_str})"
                    else:
                        boost += 0.10
                        reason_additions += f" + range rebond résistance ({regime_str})"
                elif market_regime in ["TRENDING_BULL", "TRENDING_BEAR"]:
                    if regime_str in ["EXTREME", "STRONG"] and market_regime == "TRENDING_BEAR":
                        boost += 0.15
                        reason_additions += f" + trend baissier fort rebond ({regime_str})"
                    elif regime_str in ["MODERATE", "STRONG"]:
                        boost += 0.10
                        reason_additions += f" + trend rebond baissier ({regime_str})"
                    else:
                        boost += 0.06
                        reason_additions += f" + trend contre-rebond ({regime_str})"
                elif market_regime == "VOLATILE":
                    boost += 0.08
                    reason_additions += " + chaos neutre rebond baissier"
            
        # CORRECTION MAGISTRALE: Volatilité avec timing des rebonds
        volatility_regime = values.get('volatility_regime')
        atr_percentile = values.get('atr_percentile')
        
        if volatility_regime is not None:
            try:
                atr_percentile = float(atr_percentile) if atr_percentile is not None else 50
                
                if signal_side == "BUY":
                    # Rebonds haussiers : volatilité faible à normale idéale
                    if volatility_regime == "low" and atr_percentile < 30:
                        boost += 0.12
                        reason_additions += f" + volatilité faible rebond parfait ({atr_percentile:.0f}%)"
                    elif volatility_regime == "normal" and 30 <= atr_percentile <= 60:
                        boost += 0.10
                        reason_additions += f" + volatilité idéale rebond ({atr_percentile:.0f}%)"
                    elif volatility_regime == "high":
                        if atr_percentile > 85:  # Volatilité extrême = faux rebonds
                            boost -= 0.12
                            reason_additions += f" mais volatilité extrême faux rebonds ({atr_percentile:.0f}%)"
                        else:  # Volatilité modérément élevée
                            boost -= 0.06
                            reason_additions += f" mais volatilité élevée rebonds ({atr_percentile:.0f}%)"
                    elif volatility_regime == "extreme":
                        boost -= 0.08  # Volatilité extrême = instabilité défavorable rebonds
                        reason_additions += " mais volatilité extrême défavorable rebonds"
                        
                else:  # SELL
                    # Rebonds baissiers : volatilité normale à élevée acceptable
                    if volatility_regime == "low" and atr_percentile < 25:
                        boost += 0.08
                        reason_additions += f" + volatilité faible rebond contrôlé ({atr_percentile:.0f}%)"
                    elif volatility_regime == "normal" and 25 <= atr_percentile <= 70:
                        boost += 0.10
                        reason_additions += f" + volatilité normale rebond ({atr_percentile:.0f}%)"
                    elif volatility_regime == "high":
                        if atr_percentile > 80:  # Volatilité élevée = continuation baissière après rebond
                            boost += 0.12
                            reason_additions += f" + volatilité élevée continuation ({atr_percentile:.0f}%)"
                        else:
                            boost += 0.08
                            reason_additions += f" + volatilité rebond baissier ({atr_percentile:.0f}%)"
                    elif volatility_regime == "extreme":
                        boost += 0.06  # Volatilité extrême moins défavorable aux rebonds baissiers
                        reason_additions += " + volatilité extrême neutre rebond baissier"
                        
            except (ValueError, TypeError):
                pass
            
        # Pattern detected (format 0-100)
        pattern_detected = values.get('pattern_detected')
        pattern_confidence = values.get('pattern_confidence')
        if pattern_detected and pattern_confidence is not None:
            try:
                confidence = float(pattern_confidence)
                if confidence > 70:
                    boost += 0.08
                    reason_additions += f" + pattern {pattern_detected} ({confidence:.0f}%)"
                elif confidence > 50:
                    boost += 0.05
                    reason_additions += f" + pattern faible ({confidence:.0f}%)"
            except (ValueError, TypeError):
                pass
                
        # Confluence score (format 0-100)
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                # CONFLUENCE ASSOUPLIE pour rebonds (CORRIGÉ)
                if confluence > 65:  # Seuil assoupli (était 75)
                    boost += 0.18
                    reason_additions += f" + confluence EXCELLENTE ({confluence:.0f})"
                elif confluence > 55:  # Seuil assoupli (était 65)
                    boost += 0.14
                    reason_additions += f" + confluence élevée ({confluence:.0f})"
                elif confluence > 48:  # Seuil assoupli (était 55)
                    boost += 0.08
                    reason_additions += f" + confluence correcte ({confluence:.0f})"
                elif confluence < 40:  # Pénalité assouplie (était 45)
                    boost -= 0.08
                    reason_additions += f" mais confluence FAIBLE ({confluence:.0f})"
            except (ValueError, TypeError):
                pass
                
        return boost, reason_additions
        
    def _count_confirmations(self, values: Dict[str, Any], signal_side: str, current_price: float) -> int:
        """Compte les confirmations techniques OBLIGATOIRES pour rebonds."""
        count = 0
        
        # 1. Volume OBLIGATOIRE (déjà vérifié au niveau stratégie mais recompté)
        volume_ratio = values.get('volume_ratio')
        if volume_ratio and float(volume_ratio) >= self.volume_confirmation_threshold:
            count += 1
            
        # 2. RSI dans zone favorable OBLIGATOIRE
        rsi_14 = values.get('rsi_14')
        if rsi_14:
            rsi_val = float(rsi_14)
            if signal_side == "BUY" and 25 <= rsi_val <= 60:
                count += 1
            elif signal_side == "SELL" and 40 <= rsi_val <= 75:
                count += 1
                
        # 3. Trend strength minimum OBLIGATOIRE (ASSOUPLI)
        trend_strength = values.get('trend_strength')
        if trend_strength:
            trend_str = str(trend_strength).lower()
            if trend_str in ['weak', 'moderate', 'strong', 'very_strong', 'extreme']:  # Ajout de 'weak'
                count += 1
                
        # 4. Market regime favorable OBLIGATOIRE
        market_regime = values.get('market_regime')
        regime_strength = values.get('regime_strength')
        if market_regime and regime_strength:
            regime_str = str(regime_strength).upper()
            if signal_side == "BUY":
                if (market_regime in ["TRENDING_BULL", "TRENDING_BEAR"] and regime_str in ["WEAK", "MODERATE", "STRONG", "EXTREME"]) or \
                   (market_regime == "RANGING" and regime_str in ["MODERATE", "STRONG", "EXTREME"]):  # Ajout de WEAK
                    count += 1
            else:  # SELL
                if (market_regime == "RANGING" and regime_str in ["WEAK", "MODERATE", "STRONG", "EXTREME"]) or \
                   (market_regime in ["TRENDING_BULL", "TRENDING_BEAR"] and regime_str in ["MODERATE", "STRONG", "EXTREME"]):  # Ajout de WEAK
                    count += 1
                    
        # 5. Pattern confidence si disponible (OPTIONNEL si pas de patterns)
        pattern_confidence = values.get('pattern_confidence')
        if pattern_confidence and float(pattern_confidence) >= self.min_pattern_confidence:
            count += 1
        elif pattern_confidence is None or float(pattern_confidence) == 0.0:
            # Pas de pénalité si aucun pattern détecté (fréquent)
            pass
            
        return count
        
    def _get_metadata_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Récupère les valeurs importantes pour les métadonnées."""
        metadata = {}
        important_keys = [
            'trend_strength', 'directional_bias', 'adx_14', 'volume_ratio',
            'rsi_14', 'williams_r', 'vwap_10', 'market_regime', 'volatility_regime',
            'confluence_score', 'pattern_detected', 'pattern_confidence'
        ]
        
        for key in important_keys:
            if values.get(key) is not None:
                metadata[key] = values[key]
                
        return metadata
        
    def validate_data(self) -> bool:
        """Valide que les données nécessaires pour les rebonds sont présentes - ANTI-SPAM RENFORCÉ."""
        if not super().validate_data():
            return False
            
        # INDICATEURS CRITIQUES OBLIGATOIRES pour rebonds fiables
        critical_indicators = ['confluence_score', 'adx_14', 'volume_ratio', 'trend_strength']
        level_indicators = ['nearest_support', 'nearest_resistance', 'ema_50']
        
        # Vérifier indicateurs critiques
        missing_critical = 0
        for indicator in critical_indicators:
            if indicator not in self.indicators or self.indicators[indicator] is None:
                missing_critical += 1
                logger.debug(f"{self.name}: Indicateur critique manquant: {indicator}")
                
        # Maximum 1 indicateur critique peut manquer
        if missing_critical > 1:
            logger.warning(f"{self.name}: Trop d'indicateurs critiques manquants ({missing_critical}/4)")
            return False
            
        # Vérifier au moins un indicateur de niveau
        has_level_indicator = any(
            indicator in self.indicators and self.indicators[indicator] is not None 
            for indicator in level_indicators
        )
        
        if not has_level_indicator:
            logger.warning(f"{self.name}: Aucun indicateur de niveau disponible")
            return False
            
        # Validation qualité données critiques
        if 'confluence_score' in self.indicators and self.indicators['confluence_score']:
            try:
                conf_val = float(self.indicators['confluence_score'])
                if conf_val < 40:  # Confluence trop faible = données douteuses
                    logger.debug(f"{self.name}: Confluence trop faible pour validation: {conf_val}")
                    return False
            except (ValueError, TypeError):
                pass
                
        return True
