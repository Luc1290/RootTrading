"""
Donchian_Breakout_Strategy - Stratégie basée sur les breakouts des canaux de Donchian.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging
import math

logger = logging.getLogger(__name__)


class Donchian_Breakout_Strategy(BaseStrategy):
    """
    Stratégie utilisant les canaux de Donchian pour détecter les breakouts de tendance.
    
    Signaux générés:
    - BUY: Prix casse au-dessus du canal haut + confirmation tendance haussière
    - SELL: Prix casse en-dessous du canal bas + confirmation tendance baissière
    
    Note: Les canaux de Donchian sont construits avec les plus hauts/plus bas sur N périodes.
    Cette stratégie utilise les niveaux de support/résistance comme proxy des canaux.
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres Donchian ASSOUPLIS crypto (signaux réalistes)
        self.breakout_threshold = 0.006   # 0.6% pour breakout confirmé (réduit)
        self.min_distance_threshold = 0.001  # Distance minimum 0.1% (assoupli)
        self.volume_confirmation = 1.0    # Pas de filtre strict initial
        self.min_adx_strength = 12        # ADX minimum seuil de rejet (assoupli)
        self.momentum_threshold = 40      # Seuil momentum
        
        # Paramètres breakout plus réalistes crypto
        self.early_breakout_threshold = 0.003  # 0.3% pour signaux précoces (assoupli)
        self.strong_breakout_threshold = 0.015  # 1.5% pour breakouts forts (ajusté)
        self.min_volume_breakout = 1.10         # Volume minimum unifié (assoupli)
        self.extreme_volume_threshold = 2.5    # Volume extrême pour gros breakouts
        self.donchian_period = 20              # Période pour canal Donchian pur
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs."""
        return {
            # Support/Résistance (proxy pour canaux Donchian)
            'nearest_support': self.indicators.get('nearest_support'),
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            'support_strength': self.indicators.get('support_strength'),
            'resistance_strength': self.indicators.get('resistance_strength'),
            'break_probability': self.indicators.get('break_probability'),
            'pivot_count': self.indicators.get('pivot_count'),
            # Tendance et momentum
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'momentum_score': self.indicators.get('momentum_score'),
            # Volume
            'volume_ratio': self.indicators.get('volume_ratio'),
            'relative_volume': self.indicators.get('relative_volume'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            # ATR pour contexte volatilité
            'atr_14': self.indicators.get('atr_14'),
            'atr_percentile': self.indicators.get('atr_percentile'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            # RSI pour timing des breakouts
            'rsi_14': self.indicators.get('rsi_14'),
            # Confluence et régime de marché
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score'),
            'market_regime': self.indicators.get('market_regime')
        }
        
    def _get_current_price_data(self) -> Dict[str, Optional[float]]:
        """Récupère les données de prix actuelles et précédentes."""
        try:
            if self.data and 'close' in self.data and self.data['close'] and \
               'high' in self.data and self.data['high'] and \
               'low' in self.data and self.data['low'] and \
               'volume' in self.data and self.data['volume']:
                return {
                    'current_price': float(self.data['close'][-1]),
                    'previous_price': float(self.data['close'][-2]) if len(self.data['close']) >= 2 else None,
                    'current_high': float(self.data['high'][-1]),
                    'current_low': float(self.data['low'][-1]),
                    'current_volume': float(self.data['volume'][-1])
                }
        except (IndexError, ValueError, TypeError):
            pass
        return {'current_price': None, 'previous_price': None, 'current_high': None, 'current_low': None, 'current_volume': None}
        
    def _calculate_donchian_levels(self, period: int = None) -> Dict[str, Optional[float]]:
        """Calcule les niveaux Donchian purs depuis les données OHLC."""
        if period is None:
            period = self.donchian_period
            
        try:
            highs = self.data.get('high', [])
            lows = self.data.get('low', [])
            
            if highs and lows and len(highs) >= period and len(lows) >= period:
                # Prendre les N dernières valeurs
                recent_highs = [float(h) for h in highs[-period:]]
                recent_lows = [float(l) for l in lows[-period:]]
                
                return {
                    'donchian_high': max(recent_highs),
                    'donchian_low': min(recent_lows)
                }
        except (ValueError, TypeError):
            pass
            
        return {'donchian_high': None, 'donchian_low': None}
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les breakouts de canaux de Donchian.
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
        price_data = self._get_current_price_data()
        
        # Helper pour valider les nombres (anti-NaN)
        def _is_valid(x):
            try:
                x = float(x) if x is not None else None
                return x is not None and not math.isnan(x)
            except (TypeError, ValueError):
                return False
        
        # Vérification des données essentielles
        current_price = price_data['current_price']
        previous_price = price_data['previous_price']
        current_high = price_data['current_high']
        current_low = price_data['current_low']
        current_volume = price_data['current_volume']
        
        if current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix non disponible",
                "metadata": {"strategy": self.name}
            }
            
        # Récupération Donchian avec fallback pur
        try:
            nearest_resistance = float(values['nearest_resistance']) if _is_valid(values['nearest_resistance']) else None
            nearest_support = float(values['nearest_support']) if _is_valid(values['nearest_support']) else None
            resistance_strength = values.get('resistance_strength')
            support_strength = values.get('support_strength')

            # Valider que les valeurs ne sont pas à 0 (invalides)
            if nearest_resistance is not None and nearest_resistance == 0:
                nearest_resistance = None
            if nearest_support is not None and nearest_support == 0:
                nearest_support = None

            # Fallback Donchian pur si S/R manquent ou invalides
            if nearest_resistance is None or nearest_support is None:
                donchian_levels = self._calculate_donchian_levels()
                if donchian_levels['donchian_high'] is not None and donchian_levels['donchian_low'] is not None:
                    # Utiliser Donchian pour compléter les valeurs manquantes
                    if nearest_resistance is None:
                        nearest_resistance = donchian_levels['donchian_high']
                        resistance_strength = 'MODERATE'  # Par défaut pour Donchian pur
                    if nearest_support is None:
                        nearest_support = donchian_levels['donchian_low']
                        support_strength = 'MODERATE'
            
            # Si toujours pas de niveaux après fallback, on peut continuer avec un seul niveau
            # La stratégie peut fonctionner avec seulement support OU résistance
                
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion niveaux: {e}",
                "metadata": {"strategy": self.name}
            }
            
        signal_side = None
        reason = ""
        base_confidence = 0.65  # Standardisé à 0.65 pour équité avec autres stratégies
        confidence_boost = 0.0
        breakout_type = None
        
        # Analyse breakout au-dessus de la résistance (canal haut)
        if nearest_resistance is not None and nearest_resistance > 0:
            breakout_distance = (current_price - nearest_resistance) / nearest_resistance
            
            # Détection crossing (franchissement)
            bull_cross = (previous_price is not None and 
                         previous_price <= nearest_resistance < current_price)
            
            # Breakout haussier avec niveaux progressifs + crossing
            if breakout_distance >= self.breakout_threshold or bull_cross:
                signal_side = "BUY"
                breakout_type = "resistance_cross" if bull_cross else "resistance_breakout"
                
                # Bonus selon type et force du breakout
                if bull_cross:
                    reason = f"CROSS résistance {nearest_resistance:.2f} (Prix: {current_price:.2f})"
                    confidence_boost += 0.18  # Cross détecté
                elif breakout_distance >= self.strong_breakout_threshold:  # >1.5%
                    reason = f"Breakout fort résistance {nearest_resistance:.2f} (+{breakout_distance*100:.1f}%)"
                    confidence_boost += 0.25  # Breakout vraiment fort
                elif breakout_distance >= self.breakout_threshold:  # >0.6%
                    reason = f"Breakout résistance {nearest_resistance:.2f} (+{breakout_distance*100:.1f}%)"
                    confidence_boost += 0.20  # Breakout confirmé
                    
            # Détection précoce ASSOUPLIE (volume + ADX)
            elif breakout_distance >= self.early_breakout_threshold:  # 0.3%
                volume_ratio = values.get('volume_ratio')
                adx_check = values.get('adx_14')
                if (volume_ratio is not None and _is_valid(volume_ratio) and float(volume_ratio) >= 1.3) and \
                   (adx_check is not None and _is_valid(adx_check) and float(adx_check) >= 18):
                    signal_side = "BUY"
                    breakout_type = "early_resistance_breakout"
                    reason = f"Breakout précoce {nearest_resistance:.2f} (+{breakout_distance*100:.1f}%) + volume+ADX"
                    confidence_boost += 0.15  # Augmenté car assoupli
                
                # Bonus si résistance était forte = breakout plus significatif
                if resistance_strength is not None:
                    res_str = str(resistance_strength).upper()
                    if res_str == 'MAJOR':
                        confidence_boost += 0.20
                        reason += " - résistance majeure cassée"
                    elif res_str == 'STRONG':
                        confidence_boost += 0.15
                        reason += " - résistance forte cassée"
                    elif res_str == 'MODERATE':
                        confidence_boost += 0.10
                        reason += " - résistance modérée cassée"
                    
        # Analyse breakdown en-dessous du support (canal bas)
        if signal_side is None and nearest_support is not None and nearest_support > 0:
            breakdown_distance = (nearest_support - current_price) / nearest_support
            
            # Détection crossing (franchissement)
            bear_cross = (previous_price is not None and 
                         previous_price >= nearest_support > current_price)
            
            # Breakdown baissier avec niveaux progressifs + crossing
            if breakdown_distance >= self.breakout_threshold or bear_cross:
                signal_side = "SELL"
                breakout_type = "support_cross" if bear_cross else "support_breakdown"
                
                # Bonus selon type et force du breakdown
                if bear_cross:
                    reason = f"CROSS support {nearest_support:.2f} (Prix: {current_price:.2f})"
                    confidence_boost += 0.18  # Cross détecté
                elif breakdown_distance >= self.strong_breakout_threshold:  # >1.5%
                    reason = f"Breakdown fort support {nearest_support:.2f} (-{breakdown_distance*100:.1f}%)"
                    confidence_boost += 0.25  # Breakdown vraiment fort
                elif breakdown_distance >= self.breakout_threshold:  # >0.6%
                    reason = f"Breakdown support {nearest_support:.2f} (-{breakdown_distance*100:.1f}%)"
                    confidence_boost += 0.20  # Breakdown confirmé
                    
            # Détection précoce ASSOUPLIE (volume + ADX)
            elif breakdown_distance >= self.early_breakout_threshold:  # 0.3%
                volume_ratio = values.get('volume_ratio')
                adx_check = values.get('adx_14')
                if (volume_ratio is not None and _is_valid(volume_ratio) and float(volume_ratio) >= 1.3) and \
                   (adx_check is not None and _is_valid(adx_check) and float(adx_check) >= 18):
                    signal_side = "SELL"
                    breakout_type = "early_support_breakdown"
                    reason = f"Breakdown précoce {nearest_support:.2f} (-{breakdown_distance*100:.1f}%) + volume+ADX"
                    confidence_boost += 0.15  # Augmenté car assoupli
                
                # Bonus si support était fort = breakdown plus significatif
                if support_strength is not None:
                    sup_str = str(support_strength).upper()
                    if sup_str == 'MAJOR':
                        confidence_boost += 0.20
                        reason += " - support majeur cassé"
                    elif sup_str == 'STRONG':
                        confidence_boost += 0.15
                        reason += " - support fort cassé"
                    elif sup_str == 'MODERATE':
                        confidence_boost += 0.10
                        reason += " - support modéré cassé"
                    
        # Filtre régime de marché - Autoriser TRANSITION sous conditions
        if signal_side is not None:
            market_regime = values.get('market_regime')
            if market_regime == 'RANGING':
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Marché en range - pas de breakout Donchian",
                    "metadata": {"strategy": self.name}
                }
            elif market_regime == 'TRANSITION':
                # Autoriser TRANSITION si ADX≥18 ou volume≥1.3
                adx_check = values.get('adx_14')
                volume_ratio = values.get('volume_ratio')
                transition_ok = False
                if adx_check is not None and _is_valid(adx_check) and float(adx_check) >= 18:
                    transition_ok = True
                if volume_ratio is not None and _is_valid(volume_ratio) and float(volume_ratio) >= 1.3:
                    transition_ok = True
                if not transition_ok:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": "Transition sans signes de poussée (ADX/volume) - on attend",
                        "metadata": {"strategy": self.name}
                    }
        
        # FILTRAGE VOLUME EN AMONT (avant calculs de confiance)
        if signal_side is not None:
            volume_ratio = values.get('volume_ratio')
            if volume_ratio is not None and _is_valid(volume_ratio):
                vol_ratio = float(volume_ratio)
                if vol_ratio < self.min_volume_breakout:  # Rejet unifié si volume insuffisant
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Breakout rejeté : volume insuffisant ({vol_ratio:.2f}x < {self.min_volume_breakout}x)",
                        "metadata": {"strategy": self.name, "volume_ratio": vol_ratio}
                    }
                
        # Pas de breakout détecté ou filtré
        if signal_side is None:
            proximity_info = ""
            if nearest_resistance is not None and nearest_resistance > 0:
                dist_res = abs(current_price - nearest_resistance) / nearest_resistance
                proximity_info += f"rés: {dist_res*100:.1f}%"
            if nearest_support is not None and nearest_support > 0:
                dist_sup = abs(current_price - nearest_support) / nearest_support
                if proximity_info:
                    proximity_info += ", "
                proximity_info += f"sup: {dist_sup*100:.1f}%"
                
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Pas de breakout Donchian détecté ({proximity_info})",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "current_price": current_price,
                    "nearest_resistance": nearest_resistance,
                    "nearest_support": nearest_support
                }
            }
            
        # Confirmation avec tendance (ADX et DI) - REJET si trop faible
        adx = values.get('adx_14')
        plus_di = values.get('plus_di') 
        minus_di = values.get('minus_di')
        
        if adx is not None and _is_valid(adx):
            try:
                adx_val = float(adx)
                # ADX trop faible = rejet direct (pas de breakout en marché plat)
                if adx_val < self.min_adx_strength:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet breakout: ADX trop faible ({adx_val:.1f}) < {self.min_adx_strength}",
                        "metadata": {"strategy": self.name}
                    }
                
                # Bonus progressif selon force ADX
                if adx_val > 30:  # Tendance très forte
                    confidence_boost += 0.12
                    reason += f" avec ADX très fort ({adx_val:.1f})"
                elif adx_val > 25:
                    confidence_boost += 0.08
                    reason += f" avec ADX fort ({adx_val:.1f})"
                elif adx_val > 20:
                    confidence_boost += 0.05
                    reason += f" avec ADX suffisant ({adx_val:.1f})"  # Plus de malus
                    
                    # Confirmation directionnelle
                    if plus_di is not None and minus_di is not None:
                        try:
                            plus_di_val = float(plus_di)
                            minus_di_val = float(minus_di)
                            
                            if signal_side == "BUY" and plus_di_val > minus_di_val:
                                confidence_boost += 0.10
                                reason += " + DI haussier"
                            elif signal_side == "SELL" and minus_di_val > plus_di_val:
                                confidence_boost += 0.10
                                reason += " + DI baissier"
                        except (ValueError, TypeError):
                            pass
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec trend_strength (VARCHAR: weak/absent/strong/very_strong/extreme)
        trend_strength = values.get('trend_strength')
        if trend_strength is not None:
            trend_str = str(trend_strength).lower()
            if trend_str in ['extreme', 'very_strong']:
                confidence_boost += 0.15
                reason += f" + trend {trend_str}"
            elif trend_str == 'strong':
                confidence_boost += 0.10
                reason += f" + trend {trend_str}"
            elif trend_str in ['moderate', 'present']:
                confidence_boost += 0.05
                reason += f" + trend {trend_str}"
                
        directional_bias = values.get('directional_bias')
        if directional_bias:
            bias_str = str(directional_bias).upper()
            if (signal_side == "BUY" and bias_str == "BULLISH") or \
               (signal_side == "SELL" and bias_str == "BEARISH"):
                confidence_boost += 0.10
                reason += f" + bias {bias_str.lower()}"
                
        # Confirmation avec momentum et RSI
        momentum_score = values.get('momentum_score')
        rsi_14 = values.get('rsi_14')
        
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                # Format 0-100, 50=neutre
                if (signal_side == "BUY" and momentum > 60) or \
                   (signal_side == "SELL" and momentum < 40):
                    confidence_boost += 0.10
                    reason += " + momentum favorable"
            except (ValueError, TypeError):
                pass
                
        # RSI timing - REJET si extrêmes assoupli (pas de breakout en euphorie/panique)
        if rsi_14 is not None and _is_valid(rsi_14):
            try:
                rsi = float(rsi_14)
                if signal_side == "BUY":
                    if rsi > 88:  # Trop suracheté = rejet net (assoupli)
                        return {
                            "side": None,
                            "confidence": 0.0,
                            "strength": "weak",
                            "reason": f"Rejet breakout: RSI trop haut ({rsi:.1f}) - éviter euphorie",
                            "metadata": {"strategy": self.name}
                        }
                    elif 55 <= rsi <= 85:  # Zone momentum optimale (élargie)
                        confidence_boost += 0.08
                        reason += f" + RSI momentum ({rsi:.1f})"
                elif signal_side == "SELL":
                    if rsi < 12:  # Trop survendu = rejet net (assoupli)
                        return {
                            "side": None,
                            "confidence": 0.0,
                            "strength": "weak",
                            "reason": f"Rejet breakdown: RSI trop bas ({rsi:.1f}) - éviter capitulation",
                            "metadata": {"strategy": self.name}
                        }
                    elif 15 <= rsi <= 45:  # Zone momentum optimale (élargie)
                        confidence_boost += 0.08
                        reason += f" + RSI momentum ({rsi:.1f})"
            except (ValueError, TypeError):
                pass
                
        # Volume confirmation UNIFIÉE (pas d'asymétrie)
        volume_ratio = values.get('volume_ratio')
        relative_volume = values.get('relative_volume')
        
        if volume_ratio is not None and _is_valid(volume_ratio):
            try:
                vol_ratio = float(volume_ratio)
                
                # Seuils unifiés pour BUY et SELL
                if vol_ratio >= 2.0:
                    confidence_boost += 0.15  # Volume exceptionnel
                    reason += f" + volume exceptionnel ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.6:
                    confidence_boost += 0.12  # Volume fort
                    reason += f" + volume fort ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.3:
                    confidence_boost += 0.08  # Volume confirmé
                    reason += f" + volume confirmé ({vol_ratio:.1f}x)"
                # Pas de bonus si < 1.3x (déjà filtré à 1.2x)
                        
            except (ValueError, TypeError):
                pass
                
        # Volume quality score avec validation NaN
        volume_quality_score = values.get('volume_quality_score')
        if volume_quality_score is not None and _is_valid(volume_quality_score):
            try:
                volume_quality = float(volume_quality_score)
                if volume_quality > 70:  # Format 0-100
                    confidence_boost += 0.08
                    reason += " + volume qualité"
            except (ValueError, TypeError):
                pass
                
        # Break probability avec validation NaN
        break_probability = values.get('break_probability')
        if break_probability is not None and _is_valid(break_probability):
            try:
                break_prob = float(break_probability)
                if break_prob > 0.6:
                    confidence_boost += 0.10
                    reason += f" (prob: {break_prob:.2f})"
            except (ValueError, TypeError):
                pass
                
        # Contexte volatilité (format 0-100) avec validation NaN
        atr_percentile = values.get('atr_percentile')
        if atr_percentile is not None and _is_valid(atr_percentile):
            try:
                atr_perc = float(atr_percentile)
                if atr_perc > 70:  # Volatilité élevée favorable aux breakouts
                    confidence_boost += 0.08
                    reason += f" + volatilité élevée ({atr_perc:.0f}%)"
                elif atr_perc > 50:
                    confidence_boost += 0.05
                    reason += f" + volatilité modérée ({atr_perc:.0f}%)"
            except (ValueError, TypeError):
                pass
                
        volatility_regime = values.get('volatility_regime')
        if volatility_regime == "extreme":
            confidence_boost += 0.10
            reason += " (vol. extreme)"
        elif volatility_regime == "high":
            confidence_boost += 0.05
            reason += " (vol. high)"
            
        # Signal strength (VARCHAR: WEAK/MODERATE/STRONG/VERY_WEAK)
        signal_strength_calc = values.get('signal_strength')
        if signal_strength_calc is not None:
            sig_str = str(signal_strength_calc).upper()
            if sig_str == 'STRONG':
                confidence_boost += 0.10
                reason += " + signal fort"
            elif sig_str == 'MODERATE':
                confidence_boost += 0.05
                reason += " + signal modéré"
                
        # Confluence score UNIFIÉE (pas d'asymétrie BUY/SELL)
        confluence_score = values.get('confluence_score')
        if confluence_score is not None and _is_valid(confluence_score):
            try:
                confluence = float(confluence_score)
                # Seuils unifiés pour BUY et SELL
                if confluence > 80:
                    confidence_boost += 0.15
                    reason += f" + confluence excellente ({confluence:.0f})"
                elif confluence > 70:
                    confidence_boost += 0.12
                    reason += f" + confluence élevée ({confluence:.0f})"
                elif confluence > 60:
                    confidence_boost += 0.08
                    reason += f" + confluence modérée ({confluence:.0f})"
            except (ValueError, TypeError):
                pass
                
        # Calcul confiance MULTIPLICATIF (cohérent avec autres stratégies)
        confidence = max(0.0, min(1.0, self.calculate_confidence(base_confidence, 1 + confidence_boost)))
        
        # Ajustement final - Gros breakouts avec validation volume extrême
        if breakout_type == "resistance_breakout" and nearest_resistance is not None:
            breakout_strength = (current_price - nearest_resistance) / nearest_resistance
            if breakout_strength > 0.03:  # Breakout > 3%
                # Vérifier volume extrême pour valider le gros breakout
                vol_ratio = values.get('volume_ratio')
                if vol_ratio is not None and _is_valid(vol_ratio) and float(vol_ratio) >= self.extreme_volume_threshold:
                    confidence = min(confidence * 1.15, 1.0)
                    reason += f" (gros breakout validé vol {float(vol_ratio):.1f}x)"
        elif breakout_type == "support_breakdown" and nearest_support is not None:
            breakdown_strength = (nearest_support - current_price) / nearest_support  
            if breakdown_strength > 0.03:  # Breakdown > 3%
                # Vérifier volume extrême pour valider le gros breakdown
                vol_ratio = values.get('volume_ratio')
                if vol_ratio is not None and _is_valid(vol_ratio) and float(vol_ratio) >= self.extreme_volume_threshold:
                    confidence = min(confidence * 1.15, 1.0)
                    reason += f" (gros breakdown validé vol {float(vol_ratio):.1f}x)"
                
        # Pas d'ajustement ATR supplémentaire (déjà traité plus haut)
        
        # Vérification finale : seuil minimum de confiance
        if confidence < 0.35:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal trop faible (conf: {confidence:.2f} < 0.35)",
                "metadata": {"strategy": self.name, "symbol": self.symbol}
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
                "current_high": current_high,
                "current_low": current_low,
                "breakout_type": breakout_type,
                "nearest_resistance": nearest_resistance,
                "nearest_support": nearest_support,
                "resistance_strength": resistance_strength,
                "support_strength": support_strength,
                "adx_14": adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "trend_strength": trend_strength,
                "directional_bias": directional_bias,
                "momentum_score": momentum_score,
                "volume_ratio": volume_ratio,
                "relative_volume": relative_volume,
                "break_probability": break_probability,
                "atr_percentile": atr_percentile,
                "confluence_score": confluence_score
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que toutes les données requises sont présentes."""
        if not super().validate_data():
            return False

        # Vérifier qu'on a des données de prix et au moins des high/low pour Donchian pur
        if not self.data or 'close' not in self.data or not self.data['close']:
            logger.warning(f"{self.name}: Données de prix manquantes")
            return False

        if 'high' not in self.data or not self.data['high'] or 'low' not in self.data or not self.data['low']:
            logger.warning(f"{self.name}: Données high/low manquantes pour Donchian")
            return False

        return True
