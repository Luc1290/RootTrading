"""
ATR_Breakout_Strategy - Stratégie basée sur les breakouts avec volatilité ATR.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class ATR_Breakout_Strategy(BaseStrategy):
    """
    Stratégie utilisant ATR et volatilité pour identifier les breakouts.
    
    Signaux générés:
    - BUY: Prix près des résistances avec volatilité élevée + breakout potentiel
    - SELL: Prix près des supports avec volatilité élevée + breakdown potentiel
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres ATR et volatilité
        self.atr_multiplier = 1.5      # Multiplicateur ATR pour breakout  
        self.volatility_threshold = 0.6 # Seuil volatilité (percentile)
        self.resistance_proximity = 0.02 # 2% de proximité aux niveaux
        self.support_proximity = 0.02   # 2% de proximité aux niveaux
        # Paramètres de tendance
        self.trend_filter_enabled = True  # Activer le filtre de tendance
        self.min_trend_strength = 0.3    # Force minimum de tendance ADX
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs ATR."""
        return {
            'atr_14': self.indicators.get('atr_14'),
            'atr_percentile': self.indicators.get('atr_percentile'),
            'natr': self.indicators.get('natr'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            'atr_stop_long': self.indicators.get('atr_stop_long'),
            'atr_stop_short': self.indicators.get('atr_stop_short'),
            'nearest_support': self.indicators.get('nearest_support'),
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            'support_strength': self.indicators.get('support_strength'),
            'resistance_strength': self.indicators.get('resistance_strength'),
            'break_probability': self.indicators.get('break_probability'),
            'bb_upper': self.indicators.get('bb_upper'),
            'bb_lower': self.indicators.get('bb_lower'),
            'bb_width': self.indicators.get('bb_width'),
            'bb_squeeze': self.indicators.get('bb_squeeze'),
            'bb_expansion': self.indicators.get('bb_expansion'),
            'momentum_score': self.indicators.get('momentum_score'),
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score'),
            # Indicateurs de tendance
            'market_regime': self.indicators.get('market_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
            'trend_strength': self.indicators.get('trend_strength'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            'adx_14': self.indicators.get('adx_14'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            'macd_trend': self.indicators.get('macd_trend')
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
        Génère un signal basé sur ATR et les breakouts de volatilité.
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
        
        # Vérification des indicateurs essentiels
        try:
            atr = float(values['atr_14']) if values['atr_14'] is not None else None
            atr_percentile = float(values['atr_percentile']) if values['atr_percentile'] is not None else None
            volatility_regime = values.get('volatility_regime')
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion ATR: {e}",
                "metadata": {"strategy": self.name}
            }
            
        if atr is None or current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "ATR ou prix non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # Vérification de la volatilité - on veut une volatilité élevée pour les breakouts
        if atr_percentile is not None and atr_percentile < self.volatility_threshold:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Volatilité trop faible ({atr_percentile:.2f}) - pas de setup breakout",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "atr": atr,
                    "atr_percentile": atr_percentile,
                    "current_price": current_price
                }
            }
            
        # Récupération des niveaux de support/résistance
        nearest_resistance = values.get('nearest_resistance')
        nearest_support = values.get('nearest_support')
        resistance_strength = values.get('resistance_strength')
        support_strength = values.get('support_strength')
        
        signal_side = None
        reason = ""
        base_confidence = 0.5
        confidence_boost = 0.0
        proximity_type = None
        
        # Filtre de tendance global
        market_regime = values.get('market_regime')
        trend_strength = values.get('trend_strength')
        trend_alignment = values.get('trend_alignment')
        adx_value = values.get('adx_14')
        
        # Déterminer la tendance principale
        is_uptrend = False
        is_downtrend = False
        trend_confirmed = False
        
        if self.trend_filter_enabled and market_regime:
            if market_regime == 'BULLISH':
                is_uptrend = True
                trend_confirmed = True
            elif market_regime == 'BEARISH':
                is_downtrend = True
                trend_confirmed = True
            elif market_regime in ['RANGING', 'TRANSITION']:
                # En range, on peut trader les deux côtés des niveaux
                trend_confirmed = False
        
        # Vérification supplémentaire avec trend_alignment (format décimal)
        if trend_alignment is not None:
            try:
                align_val = float(trend_alignment)
                if align_val > 0.2:  # Tendance haussière (format décimal)
                    is_uptrend = True
                elif align_val < -0.2:  # Tendance baissière (format décimal)
                    is_downtrend = True
            except (ValueError, TypeError):
                pass
        
        # Analyse de proximité aux niveaux clés avec logique adaptée à la tendance
        if nearest_resistance is not None:
            try:
                resistance_level = float(nearest_resistance)
                distance_to_resistance = abs(current_price - resistance_level) / current_price
                
                if distance_to_resistance <= self.resistance_proximity:
                    # Logique adaptée selon la tendance
                    if is_downtrend or (not trend_confirmed and trend_alignment and trend_alignment < 0):
                        # En tendance baissière : résistance = opportunité de VENTE
                        signal_side = "SELL"
                        proximity_type = "resistance"
                        reason = f"Rejet de résistance {resistance_level:.2f} en tendance baissière (ATR: {atr:.4f})"
                        confidence_boost += 0.20  # Plus de confiance car aligné avec la tendance
                    elif is_uptrend and (adx_value is None or adx_value >= self.min_trend_strength):
                        # En tendance haussière forte : possibilité de breakout
                        signal_side = "BUY"
                        proximity_type = "resistance"
                        reason = f"Breakout potentiel de résistance {resistance_level:.2f} en tendance haussière (ATR: {atr:.4f})"
                        confidence_boost += 0.10  # Moins de confiance car contre le niveau
                    elif not trend_confirmed:
                        # En range : privilégier le rejet
                        signal_side = "SELL"
                        proximity_type = "resistance"
                        reason = f"Rejet de résistance {resistance_level:.2f} en marché en range (ATR: {atr:.4f})"
                        confidence_boost += 0.15
                    
                    # Bonus si résistance forte
                    if resistance_strength is not None and signal_side:
                        res_str = str(resistance_strength).upper()
                        if res_str == 'MAJOR':
                            confidence_boost += 0.15
                            reason += " - résistance majeure"
                        elif res_str == 'STRONG':
                            confidence_boost += 0.10
                            reason += " - résistance forte"
            except (ValueError, TypeError):
                pass
                
        if signal_side is None and nearest_support is not None:
            try:
                support_level = float(nearest_support)
                distance_to_support = abs(current_price - support_level) / current_price
                
                if distance_to_support <= self.support_proximity:
                    # Logique adaptée selon la tendance
                    if is_uptrend or (not trend_confirmed and trend_alignment and trend_alignment > 0):
                        # En tendance haussière : support = opportunité d'ACHAT
                        signal_side = "BUY"
                        proximity_type = "support"
                        reason = f"Rebond sur support {support_level:.2f} en tendance haussière (ATR: {atr:.4f})"
                        confidence_boost += 0.20  # Plus de confiance car aligné avec la tendance
                    elif is_downtrend and (adx_value is None or adx_value >= self.min_trend_strength):
                        # En tendance baissière forte : possibilité de breakdown
                        signal_side = "SELL"
                        proximity_type = "support"
                        reason = f"Breakdown potentiel de support {support_level:.2f} en tendance baissière (ATR: {atr:.4f})"
                        confidence_boost += 0.10  # Moins de confiance car contre le niveau
                    elif not trend_confirmed:
                        # En range : privilégier le rebond
                        signal_side = "BUY"
                        proximity_type = "support"
                        reason = f"Rebond sur support {support_level:.2f} en marché en range (ATR: {atr:.4f})"
                        confidence_boost += 0.15
                    
                    # Bonus si support fort
                    if support_strength is not None and signal_side:
                        sup_str = str(support_strength).upper()
                        if sup_str == 'MAJOR':
                            confidence_boost += 0.15
                            reason += " - support majeur"
                        elif sup_str == 'STRONG':
                            confidence_boost += 0.10
                            reason += " - support fort"
            except (ValueError, TypeError):
                pass
                
        # Pas de proximité détectée
        if signal_side is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix pas proche des niveaux clés pour breakout",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "current_price": current_price,
                    "resistance": nearest_resistance,
                    "support": nearest_support,
                    "atr": atr
                }
            }
            
        # Ajustements de confiance selon l'ATR
        if atr_percentile is not None:
            if atr_percentile >= 80:
                confidence_boost += 0.2
                reason += " - volatilité extrême"
            elif atr_percentile >= 70:
                confidence_boost += 0.15
                reason += " - volatilité élevée"
            else:
                confidence_boost += 0.1
                reason += " - volatilité modérée"
                
        # CORRECTION MAGISTRALE: Régime volatilité ATR avec seuils adaptatifs
        if volatility_regime is not None and atr_percentile is not None:
            try:
                atr_percentile = float(atr_percentile)
                
                if signal_side == "BUY":
                    # BUY breakout ATR : volatilité expansion + proximité résistance = explosion haussière
                    if volatility_regime == "expanding":
                        if atr_percentile > 85:  # Expansion extrême + résistance
                            confidence_boost += 0.25
                            reason += f" + expansion extrême résistance ({atr_percentile:.0f}%)"
                        elif atr_percentile > 70:  # Expansion forte
                            confidence_boost += 0.20
                            reason += f" + expansion forte breakout ({atr_percentile:.0f}%)"
                        else:  # Expansion modérée
                            confidence_boost += 0.15
                            reason += f" + expansion modérée ({atr_percentile:.0f}%)"
                    elif volatility_regime == "high":
                        if atr_percentile > 90:  # Volatilité extrême = continuation explosive
                            confidence_boost += 0.18
                            reason += f" + volatilité extrême explosion ({atr_percentile:.0f}%)"
                        elif atr_percentile > 75:  # Volatilité élevée favorable
                            confidence_boost += 0.15
                            reason += f" + volatilité élevée breakout ({atr_percentile:.0f}%)"
                        else:  # Volatilité modérément élevée
                            confidence_boost += 0.10
                            reason += f" + volatilité favorable ({atr_percentile:.0f}%)"
                    elif volatility_regime == "normal":
                        if 40 <= atr_percentile <= 60:  # Volatilité idéale pour breakout contrôlé
                            confidence_boost += 0.12
                            reason += f" + volatilité idéale contrôlée ({atr_percentile:.0f}%)"
                        else:
                            confidence_boost += 0.08
                            reason += f" + volatilité normale ({atr_percentile:.0f}%)"
                    elif volatility_regime == "low":
                        confidence_boost += 0.05  # Faible volatilité = breakout moins puissant
                        reason += f" + volatilité faible limitée ({atr_percentile:.0f}%)"
                        
                else:  # SELL
                    # SELL breakdown ATR : volatilité élevée + proximité support = chute violente
                    if volatility_regime == "expanding":
                        if atr_percentile > 80:  # Expansion + support = cascade baissière
                            confidence_boost += 0.22
                            reason += f" + expansion cascade support ({atr_percentile:.0f}%)"
                        else:  # Expansion modérée
                            confidence_boost += 0.18
                            reason += f" + expansion breakdown ({atr_percentile:.0f}%)"
                    elif volatility_regime == "high":
                        if atr_percentile > 85:  # Volatilité extrême = panique/liquidation
                            confidence_boost += 0.20
                            reason += f" + volatilité extrême panique ({atr_percentile:.0f}%)"
                        else:  # Volatilité élevée
                            confidence_boost += 0.15
                            reason += f" + volatilité élevée breakdown ({atr_percentile:.0f}%)"
                    elif volatility_regime == "normal":
                        confidence_boost += 0.10
                        reason += f" + volatilité normale breakdown ({atr_percentile:.0f}%)"
                    elif volatility_regime == "low":
                        confidence_boost += 0.08  # Volatilité faible = breakdown contrôlé
                        reason += f" + volatilité faible contrôlée ({atr_percentile:.0f}%)"
                        
            except (ValueError, TypeError):
                pass
            
        # Bollinger Bands pour confirmation
        bb_upper = values.get('bb_upper')
        bb_lower = values.get('bb_lower')
        bb_squeeze = values.get('bb_squeeze')
        bb_expansion = values.get('bb_expansion')
        
        if signal_side == "BUY" and bb_upper is not None:
            try:
                bb_up = float(bb_upper)
                if current_price >= bb_up * 0.98:  # Proche de la bande haute
                    confidence_boost += 0.1
                    reason += " près BB haute"
            except (ValueError, TypeError):
                pass
                
        if signal_side == "SELL" and bb_lower is not None:
            try:
                bb_low = float(bb_lower)
                if current_price <= bb_low * 1.02:  # Proche de la bande basse
                    confidence_boost += 0.1
                    reason += " près BB basse"
            except (ValueError, TypeError):
                pass
                
        # Bollinger Squeeze = compression avant expansion
        if bb_squeeze is not None:
            try:
                squeeze = float(bb_squeeze)
                if squeeze > 0.7:  # BB compressés
                    confidence_boost += 0.15
                    reason += " avec BB squeeze"
            except (ValueError, TypeError):
                pass
                
        # Break probability
        break_probability = values.get('break_probability')
        if break_probability is not None:
            try:
                break_prob = float(break_probability)
                if break_prob > 0.6:
                    confidence_boost += 0.1
                    reason += f" (prob break: {break_prob:.2f})"
            except (ValueError, TypeError):
                pass
                
        # Momentum pour confirmation de direction (format 0-100, 50=neutre)
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                if (signal_side == "BUY" and momentum > 55) or \
                   (signal_side == "SELL" and momentum < 45):
                    confidence_boost += 0.1
                    reason += " avec momentum favorable"
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
                
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if confluence > 60:
                    confidence_boost += 0.15
                    reason += f" + confluence élevée ({confluence:.0f})"
                elif confluence > 45:
                    confidence_boost += 0.10
                    reason += f" + confluence modérée ({confluence:.0f})"
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
                "current_price": current_price,
                "atr": atr,
                "atr_percentile": atr_percentile,
                "volatility_regime": volatility_regime,
                "proximity_type": proximity_type,
                "resistance": nearest_resistance,
                "support": nearest_support,
                "resistance_strength": resistance_strength,
                "support_strength": support_strength,
                "break_probability": break_probability,
                "bb_squeeze": bb_squeeze,
                "momentum_score": momentum_score,
                "confluence_score": confluence_score,
                "market_regime": market_regime,
                "trend_alignment": trend_alignment,
                "trend_strength": trend_strength
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs ATR requis sont présents."""
        if not super().validate_data():
            return False
            
        required = ['atr_14']
        
        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        # Vérifier aussi qu'on a des données de prix
        if not self.data or 'close' not in self.data or not self.data['close']:
            logger.warning(f"{self.name}: Données de prix manquantes")
            return False
            
        return True
