"""
Supertrend_Reversal_Strategy - Stratégie basée sur les reversals de Supertrend utilisant ATR.
Le Supertrend est un indicateur de suivi de tendance basé sur l'ATR qui génère des signaux de reversal.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class Supertrend_Reversal_Strategy(BaseStrategy):
    """
    Stratégie détectant les reversals de tendance avec logique Supertrend simulée.
    
    Principe Supertrend :
    - Supertrend = HL2 ± (Multiplier × ATR)
    - Bullish: prix > Supertrend (signal BUY sur reversal)
    - Bearish: prix < Supertrend (signal SELL sur reversal)
    
    Comme Supertrend n'est pas directement disponible, nous le simulons avec :
    - ATR stops (atr_stop_long/short) comme proxy Supertrend
    - Trend strength + directional bias pour direction
    - EMA cross patterns pour confirmation reversal
    
    Signaux générés:
    - BUY: Reversal haussier détecté (prix > ATR stop + confirmations)
    - SELL: Reversal baissier détecté (prix < ATR stop + confirmations)
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        
        # Paramètres Supertrend simulé
        self.supertrend_multiplier = 2.0         # Multiplier ATR classique
        self.min_atr_distance = 0.002            # Distance minimum prix/ATR stop
        self.max_atr_distance = 0.02             # Distance maximum pour reversal
        
        # Paramètres de reversal
        self.min_trend_strength_change = 0.3     # Changement minimum trend strength
        self.momentum_reversal_threshold = 0.2   # Momentum change pour reversal
        self.directional_bias_flip_required = True  # Bias doit changer
        
        # Paramètres EMA confirmation
        self.ema_cross_confirmation = True       # EMA cross requis
        self.min_ema_separation = 0.001         # Séparation minimum EMA12/26
        
        # Paramètres volume et volatilité
        self.min_volume_confirmation = 1.2       # Volume minimum pour reversal
        self.min_volatility_regime = 0.4         # Volatilité minimum requise
        self.max_volatility_regime = 1.2         # Volatilité maximum (pas trop chaotique)
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            # ATR et stops (proxy Supertrend)
            'atr_14': self.indicators.get('atr_14'),
            'atr_stop_long': self.indicators.get('atr_stop_long'),
            'atr_stop_short': self.indicators.get('atr_stop_short'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            
            # Tendance et direction (cœur reversal)
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            'trend_angle': self.indicators.get('trend_angle'),
            
            # EMA cross patterns (confirmation reversal)
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            
            # ADX (force tendance)
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            
            # Momentum (détection changement)
            'momentum_score': self.indicators.get('momentum_score'),
            'momentum_10': self.indicators.get('momentum_10'),
            'roc_10': self.indicators.get('roc_10'),
            'roc_20': self.indicators.get('roc_20'),
            
            # Oscillateurs (timing reversal)
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            'williams_r': self.indicators.get('williams_r'),
            
            # Volume (confirmation)
            'volume_ratio': self.indicators.get('volume_ratio'),
            'relative_volume': self.indicators.get('relative_volume'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            
            # Support/Résistance (niveaux clés)
            'nearest_support': self.indicators.get('nearest_support'),
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            'support_strength': self.indicators.get('support_strength'),
            'resistance_strength': self.indicators.get('resistance_strength'),
            
            # Contexte marché
            'market_regime': self.indicators.get('market_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
            'confluence_score': self.indicators.get('confluence_score'),
            'signal_strength': self.indicators.get('signal_strength')
        }
        
    def _calculate_supertrend_proxy(self, values: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Calcule une approximation du Supertrend basée sur ATR stops."""
        atr_14 = values.get('atr_14')
        atr_stop_long = values.get('atr_stop_long')
        atr_stop_short = values.get('atr_stop_short')
        
        if atr_14 is None:
            return {'supertrend_bullish': None, 'supertrend_bearish': None, 'distance': None}
            
        try:
            atr_val = float(atr_14)
            
            # Calcul Supertrend approximatif
            # HL2 = (high + low) / 2, approximé par current_price
            supertrend_distance = self.supertrend_multiplier * atr_val
            
            # Utiliser ATR stops s'ils sont disponibles, sinon calculer
            if atr_stop_long is not None:
                supertrend_bullish_level = float(atr_stop_long)
            else:
                supertrend_bullish_level = current_price - supertrend_distance
                
            if atr_stop_short is not None:
                supertrend_bearish_level = float(atr_stop_short)  
            else:
                supertrend_bearish_level = current_price + supertrend_distance
                
            # Distance relative au Supertrend
            distance_to_bull_st = abs(current_price - supertrend_bullish_level) / current_price
            distance_to_bear_st = abs(current_price - supertrend_bearish_level) / current_price
            
            return {
                'supertrend_bullish_level': supertrend_bullish_level,
                'supertrend_bearish_level': supertrend_bearish_level,
                'distance_to_bullish': distance_to_bull_st,
                'distance_to_bearish': distance_to_bear_st,
                'atr_value': atr_val
            }
            
        except (ValueError, TypeError):
            return {'supertrend_bullish': None, 'supertrend_bearish': None, 'distance': None}
            
    def _detect_trend_reversal(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte un potentiel reversal de tendance."""
        reversal_score = 0.0
        reversal_indicators = []
        reversal_direction = None  # 'bullish' ou 'bearish'
        
        # Analyse directional bias (changement de direction)
        directional_bias = values.get('directional_bias')
        trend_strength = values.get('trend_strength')
        
        if directional_bias and trend_strength is not None:
            try:
                trend_val = float(trend_strength)
                
                # Trend strength faible = potentiel reversal
                if trend_val < 0.3:
                    reversal_score += 0.2
                    reversal_indicators.append(f"Trend faible ({trend_val:.2f})")
                    
                # Directional bias donne la direction attendue du reversal
                if directional_bias == 'bullish':
                    reversal_direction = 'bullish'
                    reversal_indicators.append("Bias haussier")
                elif directional_bias == 'bearish':
                    reversal_direction = 'bearish'
                    reversal_indicators.append("Bias baissier")
                    
            except (ValueError, TypeError):
                pass
                
        # ADX pour confirmer changement de tendance
        adx_14 = values.get('adx_14')
        plus_di = values.get('plus_di')
        minus_di = values.get('minus_di')
        
        if adx_14 is not None and plus_di is not None and minus_di is not None:
            try:
                adx_val = float(adx_14)
                plus_di_val = float(plus_di)
                minus_di_val = float(minus_di)
                
                # ADX décroissant = affaiblissement tendance
                if adx_val < 25:  # ADX faible
                    reversal_score += 0.15
                    reversal_indicators.append(f"ADX faible ({adx_val:.1f})")
                    
                # Cross DI pour direction reversal
                di_diff = plus_di_val - minus_di_val
                if reversal_direction == 'bullish' and di_diff > 0:
                    reversal_score += 0.15
                    reversal_indicators.append("DI+ > DI- (haussier)")
                elif reversal_direction == 'bearish' and di_diff < 0:
                    reversal_score += 0.15
                    reversal_indicators.append("DI- > DI+ (baissier)")
                    
            except (ValueError, TypeError):
                pass
                
        # Momentum reversal confirmation
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum_val = float(momentum_score)
                
                # Momentum change significatif
                if reversal_direction == 'bullish' and momentum_val > self.momentum_reversal_threshold:
                    reversal_score += 0.2
                    reversal_indicators.append(f"Momentum haussier ({momentum_val:.2f})")
                elif reversal_direction == 'bearish' and momentum_val < -self.momentum_reversal_threshold:
                    reversal_score += 0.2
                    reversal_indicators.append(f"Momentum baissier ({momentum_val:.2f})")
                    
            except (ValueError, TypeError):
                pass
                
        return {
            'is_reversal': reversal_score >= 0.4,
            'direction': reversal_direction,
            'score': reversal_score,
            'indicators': reversal_indicators
        }
        
    def _detect_ema_cross_confirmation(self, values: Dict[str, Any], reversal_direction: str) -> Dict[str, Any]:
        """Détecte confirmation EMA cross pour le reversal."""
        cross_score = 0.0
        cross_indicators = []
        
        ema_12 = values.get('ema_12')
        ema_26 = values.get('ema_26')
        ema_50 = values.get('ema_50')
        
        if ema_12 is not None and ema_26 is not None:
            try:
                ema12_val = float(ema_12)
                ema26_val = float(ema_26)
                
                ema_diff = (ema12_val - ema26_val) / ema12_val
                
                # Cross confirmation selon direction reversal
                if reversal_direction == 'bullish' and ema_diff > self.min_ema_separation:
                    cross_score += 0.3
                    cross_indicators.append(f"EMA12 > EMA26 ({ema_diff*100:.2f}%)")
                elif reversal_direction == 'bearish' and ema_diff < -self.min_ema_separation:
                    cross_score += 0.3
                    cross_indicators.append(f"EMA12 < EMA26 ({ema_diff*100:.2f}%)")
                    
                # EMA50 pour confirmation long terme
                if ema_50 is not None:
                    ema50_val = float(ema_50)
                    if reversal_direction == 'bullish' and ema12_val > ema50_val:
                        cross_score += 0.1
                        cross_indicators.append("EMA12 > EMA50")
                    elif reversal_direction == 'bearish' and ema12_val < ema50_val:
                        cross_score += 0.1
                        cross_indicators.append("EMA12 < EMA50")
                        
            except (ValueError, TypeError):
                pass
                
        return {
            'is_cross_confirmed': cross_score >= 0.2,
            'score': cross_score,
            'indicators': cross_indicators
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les reversals Supertrend.
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
        
        # Récupérer prix actuel
        current_price = None
        if 'close' in self.data and self.data['close']:
            try:
                current_price = float(self.data['close'][-1])
            except (IndexError, ValueError, TypeError):
                pass
            
        if current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix actuel non disponible",
                "metadata": {"strategy": self.name}
            }
            
        # Vérifier volatilité appropriée
        volatility_regime = values.get('volatility_regime')
        if volatility_regime is not None:
            try:
                vol_regime = self._convert_volatility_to_score(str(volatility_regime))  
                if vol_regime < self.min_volatility_regime or vol_regime > self.max_volatility_regime:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Volatilité inappropriée ({vol_regime:.2f}) pour Supertrend",
                        "metadata": {"strategy": self.name, "volatility_regime": vol_regime}
                    }
            except (ValueError, TypeError):
                pass
                
        # Calculer Supertrend proxy
        supertrend_data = self._calculate_supertrend_proxy(values, current_price)
        if supertrend_data.get('supertrend_bullish_level') is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "ATR indisponible pour calcul Supertrend",
                "metadata": {"strategy": self.name}
            }
            
        # Détecter reversal de tendance
        reversal_analysis = self._detect_trend_reversal(values)
        
        if not reversal_analysis['is_reversal']:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Pas de reversal détecté (score: {reversal_analysis['score']:.2f})",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "reversal_score": reversal_analysis['score'],
                    "current_price": current_price
                }
            }
            
        reversal_direction = reversal_analysis['direction']
        if reversal_direction is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Direction de reversal indéterminée",
                "metadata": {"strategy": self.name}
            }
            
        # Vérifier position par rapport au Supertrend
        supertrend_bullish = supertrend_data['supertrend_bullish_level']
        supertrend_bearish = supertrend_data['supertrend_bearish_level']
        
        signal_side = None
        supertrend_condition_met = False
        
        if reversal_direction == 'bullish':
            # Signal BUY: prix au-dessus Supertrend bullish level
            if current_price > supertrend_bullish:
                distance = supertrend_data['distance_to_bullish']
                if self.min_atr_distance <= distance <= self.max_atr_distance:
                    signal_side = "BUY"
                    supertrend_condition_met = True
                    
        elif reversal_direction == 'bearish':
            # Signal SELL: prix en-dessous Supertrend bearish level
            if current_price < supertrend_bearish:
                distance = supertrend_data['distance_to_bearish']
                if self.min_atr_distance <= distance <= self.max_atr_distance:
                    signal_side = "SELL"
                    supertrend_condition_met = True
                    
        if not supertrend_condition_met:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Prix pas en position Supertrend pour reversal {reversal_direction}",
                "metadata": {
                    "strategy": self.name,
                    "current_price": current_price,
                    "supertrend_bullish": supertrend_bullish,
                    "supertrend_bearish": supertrend_bearish
                }
            }
            
        # Confirmation EMA cross
        ema_analysis = self._detect_ema_cross_confirmation(values, reversal_direction)
        
        if self.ema_cross_confirmation and not ema_analysis['is_cross_confirmed']:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"EMA cross non confirmé pour reversal {reversal_direction}",
                "metadata": {"strategy": self.name}
            }
            
        # Générer signal final
        base_confidence = 0.5
        confidence_boost = 0.0
        
        # Score reversal
        confidence_boost += reversal_analysis['score'] * 0.4
        
        # Score EMA confirmation  
        confidence_boost += ema_analysis['score'] * 0.3
        
        reason = f"Supertrend reversal {reversal_direction}: {', '.join(reversal_analysis['indicators'][:2])}"
        
        if ema_analysis['indicators']:
            reason += f" + {ema_analysis['indicators'][0]}"
            
        # CORRECTION: Volume confirmation - adaptation selon direction reversal
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                
                # Volume requis plus élevé pour confirmations selon direction
                if signal_side == "BUY":
                    # Reversal haussier : volume fort pour confirmer changement de tendance
                    if vol_ratio >= self.min_volume_confirmation * 1.3:  # 1.56x minimum
                        confidence_boost += 0.15  # Bonus élevé pour BUY avec fort volume
                        reason += f" + volume reversal haussier ({vol_ratio:.1f}x)"
                    elif vol_ratio >= self.min_volume_confirmation:
                        confidence_boost += 0.08  # Bonus modéré
                        reason += f" + volume modéré ({vol_ratio:.1f}x)"
                elif signal_side == "SELL":
                    # Reversal baissier : volume modéré acceptable (selling plus naturel)
                    if vol_ratio >= self.min_volume_confirmation * 1.2:  # 1.44x minimum
                        confidence_boost += 0.15  # Bonus élevé pour SELL avec volume
                        reason += f" + volume reversal baissier ({vol_ratio:.1f}x)"
                    elif vol_ratio >= self.min_volume_confirmation * 0.9:  # Plus permissif pour SELL
                        confidence_boost += 0.10  # Bonus bon pour SELL
                        reason += f" + volume confirmé ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # CORRECTION: Confluence score - vérification cohérence directionnelle
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                conf_val = float(confluence_score)
                
                # Vérifier cohérence confluence avec direction reversal
                trend_alignment = values.get('trend_alignment')
                directional_bias = values.get('directional_bias')
                
                confluence_coherent = True
                if trend_alignment is not None and directional_bias is not None:
                    try:
                        trend_align_val = float(trend_alignment)
                        # Vérifier cohérence entre confluence, trend alignment et direction
                        if signal_side == "BUY" and directional_bias == "bullish":
                            confluence_coherent = trend_align_val > 0.5  # Alignement haussier requis
                        elif signal_side == "SELL" and directional_bias == "bearish":
                            confluence_coherent = trend_align_val < 0.5  # Alignement baissier requis
                        else:
                            confluence_coherent = False  # Incohérence direction
                    except (ValueError, TypeError):
                        pass
                
                # Bonus confluence adapté selon cohérence
                if conf_val > 0.8 and confluence_coherent:
                    confidence_boost += 0.12  # Confluence très élevée et cohérente
                    reason += " + très haute confluence cohérente"
                elif conf_val > 0.7 and confluence_coherent:
                    confidence_boost += 0.10  # Confluence élevée et cohérente
                    reason += " + haute confluence cohérente"
                elif conf_val > 0.6 and confluence_coherent:
                    confidence_boost += 0.06  # Confluence modérée et cohérente
                    reason += " + confluence modérée"
                elif conf_val > 0.7 and not confluence_coherent:
                    confidence_boost += 0.05  # Confluence élevée mais incohérente
                    reason += " + confluence élevée (partielle)"
                # Pas de bonus si confluence faible ou très incohérente
                    
            except (ValueError, TypeError):
                pass
        
        # CORRECTION: Confirmations directionnelles supplémentaires
        
        # RSI confirmation pour reversal
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi_val = float(rsi_14)
                # BUY reversal : RSI sort d'oversold ou en zone neutre-haussière
                if signal_side == "BUY":
                    if 35 <= rsi_val <= 60:  # Zone optimale pour reversal haussier
                        confidence_boost += 0.08
                        reason += f" + RSI reversal optimal ({rsi_val:.1f})"
                    elif 25 <= rsi_val <= 34:  # Sortie d'oversold
                        confidence_boost += 0.12
                        reason += f" + RSI sortie oversold ({rsi_val:.1f})"
                    elif rsi_val >= 70:  # Déjà overbought = reversal difficile
                        confidence_boost -= 0.05
                        reason += f" mais RSI overbought ({rsi_val:.1f})"
                        
                # SELL reversal : RSI sort d'overbought ou en zone neutre-baissière
                elif signal_side == "SELL":
                    if 40 <= rsi_val <= 65:  # Zone optimale pour reversal baissier
                        confidence_boost += 0.08
                        reason += f" + RSI reversal optimal ({rsi_val:.1f})"
                    elif 66 <= rsi_val <= 75:  # Sortie d'overbought
                        confidence_boost += 0.12
                        reason += f" + RSI sortie overbought ({rsi_val:.1f})"
                    elif rsi_val <= 30:  # Déjà oversold = reversal difficile
                        confidence_boost -= 0.05
                        reason += f" mais RSI oversold ({rsi_val:.1f})"
            except (ValueError, TypeError):
                pass
        
        # Support/Résistance confluence pour reversal
        if signal_side == "BUY":
            nearest_support = values.get('nearest_support')
            if nearest_support is not None:
                try:
                    support_level = float(nearest_support)
                    distance_to_support = abs(current_price - support_level) / current_price
                    if distance_to_support <= 0.015:  # 1.5% du support
                        confidence_boost += 0.10
                        reason += f" + proche support ({distance_to_support*100:.1f}%)"
                except (ValueError, TypeError):
                    pass
                    
        elif signal_side == "SELL":
            nearest_resistance = values.get('nearest_resistance')
            if nearest_resistance is not None:
                try:
                    resistance_level = float(nearest_resistance)
                    distance_to_resistance = abs(current_price - resistance_level) / current_price
                    if distance_to_resistance <= 0.015:  # 1.5% de la résistance
                        confidence_boost += 0.10
                        reason += f" + proche résistance ({distance_to_resistance*100:.1f}%)"
                except (ValueError, TypeError):
                    pass
        
        # ROC confirmation pour momentum reversal
        roc_10 = values.get('roc_10')
        if roc_10 is not None:
            try:
                roc_val = float(roc_10)
                # BUY reversal : ROC commence à devenir positif
                if signal_side == "BUY" and roc_val > 0.5:
                    confidence_boost += 0.08
                    reason += f" + ROC positif ({roc_val:.1f}%)"
                # SELL reversal : ROC commence à devenir négatif  
                elif signal_side == "SELL" and roc_val < -0.5:
                    confidence_boost += 0.08
                    reason += f" + ROC négatif ({roc_val:.1f}%)"
            except (ValueError, TypeError):
                pass
                
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
                "current_price": current_price,
                "reversal_direction": reversal_direction,
                "reversal_score": reversal_analysis['score'],
                "ema_cross_score": ema_analysis['score'],
                "supertrend_bullish_level": supertrend_bullish,
                "supertrend_bearish_level": supertrend_bearish,
                "atr_value": supertrend_data.get('atr_value'),
                "reversal_indicators": reversal_analysis['indicators'],
                "ema_indicators": ema_analysis['indicators'],
                "volume_ratio": volume_ratio,
                "volatility_regime": volatility_regime,
                "confluence_score": confluence_score
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        required_indicators = [
            'atr_14', 'trend_strength', 'directional_bias', 
            'ema_12', 'ema_26', 'adx_14'
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
                
        # Vérifier données OHLCV
        if 'close' not in self.data or not self.data['close']:
            logger.warning(f"{self.name}: Données close manquantes")
            return False
            
        return True
    
    def _convert_volatility_to_score(self, volatility_regime: str) -> float:
        """Convertit un régime de volatilité en score numérique."""
        try:
            if not volatility_regime:
                return 1.0
                
            vol_lower = volatility_regime.lower()
            
            if vol_lower in ['high', 'very_high', 'extreme']:
                return 3.0  # Haute volatilité
            elif vol_lower in ['normal', 'moderate', 'average']:
                return 2.0  # Volatilité normale
            elif vol_lower in ['low', 'very_low', 'minimal']:
                return 1.0  # Faible volatilité
            else:
                # Essayer de convertir directement en float
                try:
                    return float(volatility_regime)
                except (ValueError, TypeError):
                    return 2.0  # Valeur par défaut
                    
        except Exception:
            return 2.0
