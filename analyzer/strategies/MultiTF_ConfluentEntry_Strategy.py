"""
MultiTF_ConfluentEntry_Strategy - Stratégie basée sur la confluence multi-timeframes.
OPTIMISÉE POUR CRYPTO SPOT INTRADAY
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class MultiTF_ConfluentEntry_Strategy(BaseStrategy):
    """
    Stratégie de confluence multi-timeframes pour des entrées précises.
    
    Utilise la confluence de plusieurs éléments techniques sur différents timeframes :
    - Trend alignment (toutes les moyennes mobiles alignées)
    - Signal strength élevé
    - Confluence score élevé
    - Support/résistance respectés
    - Volume et momentum favorables
    
    Signaux générés:
    - BUY: Confluence haussière sur multiple timeframes + confirmations
    - SELL: Confluence baissière sur multiple timeframes + confirmations
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres OPTIMISÉS WINRATE - Plus sélectifs
        self.min_confluence_score = 60      # Relevé pour qualité (35 -> 60)
        self.strong_confluence_score = 75   # Relevé pour boost fort (55 -> 75)
        self.min_trend_alignment = 0.25     # Plus strict (0.15 -> 0.25)
        self.strong_trend_alignment = 0.55  # Plus strict (0.45 -> 0.55)
        self.volume_confirmation_min = 1.2  # Volume minimum relevé (0.8 -> 1.2)
        self.volume_strong = 2.0            # Volume fort relevé (1.5 -> 2.0)
        
        # NOUVEAUX FILTRES ANTI-FAUX SIGNAUX
        self.min_ma_count_required = 4      # Au moins 4 MAs pour signal
        self.min_oscillator_count = 3       # Au moins 3 oscillateurs pour confluence
        self.require_adx_confirmation = True # ADX obligatoire pour tous les signaux
        
        # Seuils oscillateurs STRICTS winrate
        self.rsi_oversold = 25              # Plus strict (28 -> 25)
        self.rsi_overbought = 75            # Plus strict (72 -> 75)
        self.stoch_oversold = 15            # Plus strict (18 -> 15)
        self.stoch_overbought = 85          # Plus strict (82 -> 85)
        self.cci_oversold = -150            # Plus strict (-120 -> -150)
        self.cci_overbought = 150           # Plus strict (120 -> 150)
        self.williams_oversold = -90        # Plus strict (-85 -> -90)
        self.williams_overbought = -10      # Plus strict (-15 -> -10)
        
        # ADX STRICT pour éviter ranging markets
        self.min_adx_trend = 20             # Plus strict (15 -> 20)
        self.strong_adx_trend = 30          # Plus strict (25 -> 30)
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs multi-TF."""
        return {
            # Confluence et force du signal
            'confluence_score': self.indicators.get('confluence_score'),
            'signal_strength': self.indicators.get('signal_strength'),
            'pattern_confidence': self.indicators.get('pattern_confidence'),
            'pattern_detected': self.indicators.get('pattern_detected'),
            # Alignement des timeframes
            'trend_alignment': self.indicators.get('trend_alignment'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'trend_angle': self.indicators.get('trend_angle'),
            # Régimes de marché
            'market_regime': self.indicators.get('market_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
            'regime_confidence': self.indicators.get('regime_confidence'),
            'regime_duration': self.indicators.get('regime_duration'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            # Support/Résistance (multi-TF)
            'nearest_support': self.indicators.get('nearest_support'),
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            'support_strength': self.indicators.get('support_strength'),
            'resistance_strength': self.indicators.get('resistance_strength'),
            'break_probability': self.indicators.get('break_probability'),
            # Moyennes mobiles (alignement)
            'ema_7': self.indicators.get('ema_7'),
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            'ema_99': self.indicators.get('ema_99'),
            'sma_20': self.indicators.get('sma_20'),
            'sma_50': self.indicators.get('sma_50'),
            'hull_20': self.indicators.get('hull_20'),
            # MACD multi-TF
            'macd_line': self.indicators.get('macd_line'),
            'macd_signal': self.indicators.get('macd_signal'),
            'macd_histogram': self.indicators.get('macd_histogram'),
            'macd_trend': self.indicators.get('macd_trend'),
            # Oscillateurs convergents
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            'cci_20': self.indicators.get('cci_20'),
            'williams_r': self.indicators.get('williams_r'),
            # Volume multi-TF
            'volume_ratio': self.indicators.get('volume_ratio'),
            'relative_volume': self.indicators.get('relative_volume'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'volume_context': self.indicators.get('volume_context'),
            # ADX pour confirmation tendance
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            # Momentum général
            'momentum_score': self.indicators.get('momentum_score'),
            'momentum_10': self.indicators.get('momentum_10')
        }
        
    def _get_current_price(self) -> Optional[float]:
        """Récupère le prix actuel depuis les données OHLCV."""
        try:
            if self.data and 'close' in self.data and self.data['close']:
                return float(self.data['close'][-1])
        except (IndexError, ValueError, TypeError):
            pass
        return None
        
    def _analyze_ma_alignment(self, values: Dict[str, Optional[float]], 
                             current_price: float) -> Dict[str, Any]:
        """Analyse l'alignement des moyennes mobiles pour détecter la tendance."""
        mas = {}
        ma_keys = ['ema_7', 'ema_12', 'ema_26', 'ema_50', 'ema_99', 'sma_20', 'sma_50', 'hull_20']
        
        # Récupération des MAs disponibles
        for key in ma_keys:
            value = values.get(key)
            if value is not None:
                try:
                    mas[key] = float(value)
                except (ValueError, TypeError):
                    continue
                    
        if len(mas) < self.min_ma_count_required:
            return {'alignment_score': 0.0, 'direction': None, 'reason': f'Pas assez de MAs ({len(mas)}/{self.min_ma_count_required})'}
            
        # Analyse spécifique pour crypto : importance des EMA courtes
        ema_7 = mas.get('ema_7')
        ema_12 = mas.get('ema_12')
        ema_26 = mas.get('ema_26')
        ema_50 = mas.get('ema_50')
        
        # Score d'alignement plus sophistiqué
        alignment_score = 0.0
        direction = None
        
        # Position du prix par rapport aux MAs
        price_above_count = sum(1 for _, ma_val in mas.items() if current_price > ma_val)
        price_ratio = price_above_count / len(mas)
        
        # Analyse hiérarchique des EMAs (plus important en crypto)
        if ema_7 and ema_12 and ema_26:
            # Configuration haussière parfaite : Prix > EMA7 > EMA12 > EMA26
            if current_price > ema_7 > ema_12 > ema_26:
                alignment_score += 0.4
                direction = "bullish"
                
                # Bonus si EMA50 aussi alignée
                if ema_50 and ema_26 > ema_50:
                    alignment_score += 0.2
                    
            # Configuration baissière parfaite : Prix < EMA7 < EMA12 < EMA26
            elif current_price < ema_7 < ema_12 < ema_26:
                alignment_score += 0.4
                direction = "bearish"
                
                # Bonus si EMA50 aussi alignée
                if ema_50 and ema_26 < ema_50:
                    alignment_score += 0.2
                    
            # Configurations partielles
            elif current_price > ema_7 and ema_7 > ema_26:
                alignment_score += 0.2
                direction = "bullish_weak"
            elif current_price < ema_7 and ema_7 < ema_26:
                alignment_score += 0.2
                direction = "bearish_weak"
                
        # Ajustement selon position globale du prix
        if price_ratio >= 0.75:
            if direction in ["bullish", "bullish_weak"] or direction is None:
                direction = "bullish"
                alignment_score = max(alignment_score + 0.2, 0.6)
        elif price_ratio <= 0.25:
            if direction in ["bearish", "bearish_weak"] or direction is None:
                direction = "bearish"
                alignment_score = max(alignment_score + 0.2, 0.6)
        else:
            if direction is None:
                direction = "neutral"
                alignment_score = 0.3
                
        # Hull MA pour confirmation (très réactive)
        hull_20 = mas.get('hull_20')
        if hull_20:
            if direction == "bullish" and current_price > hull_20:
                alignment_score = min(alignment_score + 0.1, 0.95)
            elif direction == "bearish" and current_price < hull_20:
                alignment_score = min(alignment_score + 0.1, 0.95)
                
        return {
            'alignment_score': alignment_score,
            'direction': direction,
            'price_above_ratio': price_ratio,
            'ma_count': len(mas),
            'ema_aligned': ema_7 is not None and ema_12 is not None and ema_26 is not None
        }
        
    def _analyze_oscillator_confluence(self, values: Dict[str, Optional[float]]) -> Dict[str, Any]:
        """Analyse la confluence des oscillateurs avec seuils crypto."""
        oscillators = {}
        
        # RSI avec seuils crypto
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if rsi <= self.rsi_oversold:
                    oscillators['rsi_14'] = 'oversold'
                elif rsi >= self.rsi_overbought:
                    oscillators['rsi_14'] = 'overbought'
                else:
                    oscillators['rsi_14'] = 'neutral'
            except (ValueError, TypeError):
                pass
                
        # RSI 21 pour confirmation
        rsi_21 = values.get('rsi_21')
        if rsi_21 is not None:
            try:
                rsi = float(rsi_21)
                if rsi <= self.rsi_oversold + 2:  # Légèrement plus tolérant
                    oscillators['rsi_21'] = 'oversold'
                elif rsi >= self.rsi_overbought - 2:
                    oscillators['rsi_21'] = 'overbought'
                else:
                    oscillators['rsi_21'] = 'neutral'
            except (ValueError, TypeError):
                pass
                
        # Stochastic avec seuils crypto
        stoch_k = values.get('stoch_k')
        stoch_d = values.get('stoch_d')
        if stoch_k is not None and stoch_d is not None:
            try:
                k = float(stoch_k)
                d = float(stoch_d)
                if k <= self.stoch_oversold and d <= self.stoch_oversold:
                    oscillators['stoch'] = 'oversold'
                elif k >= self.stoch_overbought and d >= self.stoch_overbought:
                    oscillators['stoch'] = 'overbought'
                else:
                    oscillators['stoch'] = 'neutral'
            except (ValueError, TypeError):
                pass
                
        # CCI avec seuils crypto
        cci_20 = values.get('cci_20')
        if cci_20 is not None:
            try:
                cci = float(cci_20)
                if cci <= self.cci_oversold:
                    oscillators['cci'] = 'oversold'
                elif cci >= self.cci_overbought:
                    oscillators['cci'] = 'overbought'
                else:
                    oscillators['cci'] = 'neutral'
            except (ValueError, TypeError):
                pass
                
        # Williams %R avec seuils crypto
        williams_r = values.get('williams_r')
        if williams_r is not None:
            try:
                wr = float(williams_r)
                if wr <= self.williams_oversold:
                    oscillators['williams'] = 'oversold'
                elif wr >= self.williams_overbought:
                    oscillators['williams'] = 'overbought'
                else:
                    oscillators['williams'] = 'neutral'
            except (ValueError, TypeError):
                pass
                
        if len(oscillators) < self.min_oscillator_count:
            return {'confluence': 'insufficient', 'strength': 0.0, 'count': len(oscillators)}
            
        # Calcul de la confluence avec pondération
        oversold_count = sum(1 for v in oscillators.values() if v == 'oversold')
        overbought_count = sum(1 for v in oscillators.values() if v == 'overbought')
        total_count = len(oscillators)
        
        # Seuils ULTRA-STRICTS winrate - confluence parfaite requise
        if oversold_count >= total_count * 0.85:  # 85% des oscillateurs (ultra-strict)
            return {'confluence': 'oversold', 'strength': oversold_count / total_count, 'count': total_count}
        elif overbought_count >= total_count * 0.85:
            return {'confluence': 'overbought', 'strength': overbought_count / total_count, 'count': total_count}
        else:
            # Rejet direct si pas de consensus ultra-majoritaire
            return {'confluence': 'insufficient', 'strength': 0.0, 'count': total_count}
            
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur la confluence multi-timeframes.
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
        
        if current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix non disponible",
                "metadata": {"strategy": self.name}
            }
            
        # Vérification des scores de confluence principaux
        try:
            confluence_score = float(values['confluence_score']) if values['confluence_score'] is not None else None
            signal_strength = values['signal_strength']  # STRING: WEAK/MODERATE/STRONG/VERY_STRONG
            trend_alignment = float(values['trend_alignment']) if values['trend_alignment'] is not None else None
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion scores: {e}",
                "metadata": {"strategy": self.name}
            }
            
        # Vérification des scores minimums
        if confluence_score is None or confluence_score < self.min_confluence_score:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Confluence insuffisante ({confluence_score:.1f} < {self.min_confluence_score})",
                "metadata": {"strategy": self.name, "confluence_score": confluence_score}
            }
            
        # Vérification signal_strength valide
        valid_strengths = ['WEAK', 'MODERATE', 'STRONG', 'VERY_STRONG']
        if signal_strength not in valid_strengths:
            signal_strength = 'WEAK'  # Default si invalide
            
        # Vérification ADX pour tendance suffisante (déplacée ici)
        adx_14 = values.get('adx_14')
        adx_value = None
        if adx_14 is not None:
            try:
                adx_value = float(adx_14)
            except (ValueError, TypeError):
                pass
        
        # Filtre signal_strength STRICT - Rejet direct WEAK/MODERATE
        if signal_strength in ['WEAK', 'MODERATE']:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal strength insuffisant ({signal_strength}) - MultiTF exige STRONG/VERY_STRONG",
                "metadata": {"strategy": self.name, "signal_strength": signal_strength}
            }
            
        # Vérification trend_alignment
        if trend_alignment and trend_alignment < self.min_trend_alignment:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Alignement insuffisant ({trend_alignment:.1f} < {self.min_trend_alignment})",
                "metadata": {"strategy": self.name, "trend_alignment": trend_alignment}
            }
            
        # ADX OBLIGATOIRE - Rejet direct si absent ou faible
        if adx_value is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "ADX non disponible - requis pour MultiTF",
                "metadata": {"strategy": self.name}
            }
        
        if adx_value < self.min_adx_trend:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"ADX insuffisant ({adx_value:.1f} < {self.min_adx_trend}) - marché ranging",
                "metadata": {"strategy": self.name, "adx": adx_value}
            }
                
        # Analyse de l'alignement des moyennes mobiles
        ma_analysis = self._analyze_ma_alignment(values, current_price)
        
        # Analyse des oscillateurs avec validation stricte
        osc_analysis = self._analyze_oscillator_confluence(values)
        
        # REJET si oscillateurs insuffisants
        if osc_analysis['confluence'] == 'insufficient':
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Oscillateurs insuffisants ({osc_analysis['count']}/{self.min_oscillator_count}) ou consensus faible",
                "metadata": {"strategy": self.name, "osc_analysis": osc_analysis}
            }
        
        signal_side = None
        reason = ""
        base_confidence = 0.50  # Harmonisé avec autres stratégies
        confidence_boost = 0.0
        
        # LOGIQUE SIMPLIFIÉE ET SÉLECTIVE - Seulement setups parfaits
        if ma_analysis['direction'] == "bullish" and ma_analysis['alignment_score'] >= 0.7:
            # Setup haussier STRICT : MA parfaitement alignées + oscillateurs parfaits
            if osc_analysis['confluence'] == 'oversold':
                signal_side = "BUY"
                reason = f"Setup PARFAIT BUY: MA alignées ({ma_analysis['alignment_score']:.2f}) + oscillateurs consensus ({osc_analysis['strength']:.2f})"
                confidence_boost += 0.30  # Boost unique pour setup parfait
            
        elif ma_analysis['direction'] == "bearish" and ma_analysis['alignment_score'] >= 0.7:
            # Setup baissier STRICT : MA parfaitement alignées + oscillateurs parfaits  
            if osc_analysis['confluence'] == 'overbought':
                signal_side = "SELL"
                reason = f"Setup PARFAIT SELL: MA alignées ({ma_analysis['alignment_score']:.2f}) + oscillateurs consensus ({osc_analysis['strength']:.2f})"
                confidence_boost += 0.30  # Boost unique pour setup parfait
                
        # Pas d'alignement clair ou contradictoire
        if signal_side is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Pas de confluence claire - MA: {ma_analysis['direction']}, Osc: {osc_analysis['confluence']}",
                "metadata": {
                    "strategy": self.name,
                    "ma_analysis": ma_analysis,
                    "osc_analysis": osc_analysis
                }
            }
            
        # === BOOSTS SIMPLIFIÉS - 4 catégories principales ===
        
        # 1. Confluence score (déjà validé comme élevé)
        if confluence_score >= 85:
            confidence_boost += 0.15
            reason += f" [confluence PARFAITE: {confluence_score:.0f}]"
        elif confluence_score >= 75:
            confidence_boost += 0.10
            reason += f" [confluence excellente: {confluence_score:.0f}]"
        
        # 2. Signal strength (déjà validé comme STRONG+)
        if signal_strength == 'VERY_STRONG':
            confidence_boost += 0.12
            reason += " + VERY_STRONG"
        elif signal_strength == 'STRONG':
            confidence_boost += 0.08
            reason += " + STRONG"
        
        # 3. Trend alignment exceptionnel (déjà validé > 0.25)
        if trend_alignment and trend_alignment >= 0.7:
            confidence_boost += 0.15
            reason += f" + trend PARFAIT ({trend_alignment:.2f})"
        elif trend_alignment and trend_alignment >= self.strong_trend_alignment:
            confidence_boost += 0.08
            reason += f" + trend fort ({trend_alignment:.2f})"
        
        # 4. ADX exceptionnel (déjà validé > 20)
        if adx_value >= 40:
            confidence_boost += 0.12
            reason += f" + ADX EXTRÊEME ({adx_value:.1f})"
        elif adx_value >= self.strong_adx_trend:
            confidence_boost += 0.08
            reason += f" + ADX fort ({adx_value:.1f})"
            
        # VALIDATION DIRECTIONAL BIAS - REJET si contradictoire
        directional_bias = values.get('directional_bias')
        if directional_bias:
            if (signal_side == "BUY" and directional_bias == "BEARISH") or \
               (signal_side == "SELL" and directional_bias == "BULLISH"):
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Setup MultiTF rejeté - bias contradictoire ({directional_bias} vs {signal_side})",
                    "metadata": {"strategy": self.name, "directional_bias": directional_bias, "signal_side": signal_side}
                }
            elif (signal_side == "BUY" and directional_bias == "BULLISH") or \
                 (signal_side == "SELL" and directional_bias == "BEARISH"):
                confidence_boost += 0.10
                reason += f" + bias confirmé"
        
        # VALIDATION RÉGIME MARCHÉ - Rejet si volatile/ranging
        market_regime = values.get('market_regime')
        if market_regime:
            if market_regime in ["VOLATILE", "RANGING", "TRANSITION"]:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"MultiTF désactivé en régime {market_regime} - trop de faux signaux",
                    "metadata": {"strategy": self.name, "market_regime": market_regime}
                }
            elif (signal_side == "BUY" and market_regime in ["TRENDING_BULL", "BREAKOUT_BULL"]) or \
                 (signal_side == "SELL" and market_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]):
                confidence_boost += 0.12
                reason += f" ({market_regime})"
                
        # VALIDATION VOLUME STRICT - Rejet direct si insuffisant
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Volume non disponible - requis pour MultiTF",
                "metadata": {"strategy": self.name}
            }
        
        try:
            vol_ratio = float(volume_ratio)
            if vol_ratio < self.volume_confirmation_min:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Volume insuffisant ({vol_ratio:.1f}x < {self.volume_confirmation_min}x) - MultiTF exige volume élevé",
                    "metadata": {"strategy": self.name, "volume_ratio": vol_ratio}
                }
            elif vol_ratio >= self.volume_strong * 1.5:  # Volume exceptionnel
                confidence_boost += 0.15
                reason += f" + volume EXCEPTIONNEL ({vol_ratio:.1f}x)"
            elif vol_ratio >= self.volume_strong:
                confidence_boost += 0.10
                reason += f" + volume fort ({vol_ratio:.1f}x)"
        except (ValueError, TypeError):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Volume invalide",
                "metadata": {"strategy": self.name}
            }
                
        # VALIDATION FINALE - Tous les autres indicateurs déjà validés
        # Plus de micros-ajustements - logique simplifiée focus winrate
                
        # Calcul final
        raw_confidence = base_confidence * (1 + confidence_boost)
        
        # Calcul final optimisé sans double calcul
        confidence = min(base_confidence * (1 + confidence_boost), 0.90)
        
        # Filtre final STRICT pour MultiTF
        if confidence < 0.65:  # Seuil élevé pour qualité MultiTF
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Setup MultiTF rejeté - confiance insuffisante ({confidence:.2f} < 0.65)",
                "metadata": {
                    "strategy": self.name,
                    "rejected_signal": signal_side,
                    "rejected_confidence": confidence,
                    "confluence_score": confluence_score,
                    "signal_strength": signal_strength,
                    "trend_alignment": trend_alignment
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
                "current_price": current_price,
                "confluence_score": confluence_score,
                "signal_strength": signal_strength,
                "trend_alignment": trend_alignment,
                "ma_analysis": ma_analysis,
                "osc_analysis": osc_analysis,
                "directional_bias": directional_bias,
                "market_regime": market_regime,
                "regime_strength": regime_strength,
                "volume_ratio": volume_ratio,
                "volume_quality_score": volume_quality_score,
                "adx_14": adx_value,
                "pattern_detected": pattern_detected,
                "pattern_confidence": pattern_confidence
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs de confluence requis sont présents."""
        if not super().validate_data():
            return False
            
        required = ['confluence_score', 'signal_strength']
        
        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        # Vérifier qu'on a au moins quelques moyennes mobiles
        ma_indicators = ['ema_7', 'ema_12', 'ema_26', 'ema_50', 'sma_20', 'sma_50']
        ma_available = sum(1 for ma in ma_indicators if ma in self.indicators and self.indicators[ma] is not None)
        
        if ma_available < 3:
            logger.warning(f"{self.name}: Pas assez de moyennes mobiles ({ma_available}/6)")
            return False
            
        return True