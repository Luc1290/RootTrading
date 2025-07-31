"""
ROC_Threshold_Strategy - Stratégie basée sur les seuils du Rate of Change (ROC).
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class ROC_Threshold_Strategy(BaseStrategy):
    """
    Stratégie utilisant les seuils du Rate of Change (ROC) pour détecter les momentum significatifs.
    
    Le ROC mesure le changement de prix en pourcentage sur une période donnée :
    - ROC = ((Prix_actuel - Prix_n_périodes) / Prix_n_périodes) * 100
    - Valeurs positives = momentum haussier
    - Valeurs négatives = momentum baissier
    
    Signaux générés:
    - BUY: ROC dépasse seuil haussier + confirmations momentum
    - SELL: ROC dépasse seuil baissier + confirmations momentum
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres ROC
        self.bullish_threshold = 2.0  # ROC > +2% pour signal haussier
        self.bearish_threshold = -2.0  # ROC < -2% pour signal baissier
        self.extreme_bullish_threshold = 5.0  # ROC > +5% = momentum extrême
        self.extreme_bearish_threshold = -5.0  # ROC < -5% = momentum extrême
        self.momentum_confirmation_threshold = 0.3  # Seuil momentum_score pour confirmation
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs ROC et momentum."""
        return {
            # ROC pour différentes périodes
            'roc_10': self.indicators.get('roc_10'),
            'roc_20': self.indicators.get('roc_20'),
            'momentum_10': self.indicators.get('momentum_10'),  # Momentum classique
            'momentum_score': self.indicators.get('momentum_score'),  # Score momentum global
            # RSI pour confluence momentum
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            # Stochastic pour momentum court terme
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            'stoch_rsi': self.indicators.get('stoch_rsi'),
            # Williams %R pour momentum
            'williams_r': self.indicators.get('williams_r'),
            # CCI pour momentum
            'cci_20': self.indicators.get('cci_20'),
            # MACD pour confirmation momentum
            'macd_line': self.indicators.get('macd_line'),
            'macd_signal': self.indicators.get('macd_signal'),
            'macd_histogram': self.indicators.get('macd_histogram'),
            'macd_trend': self.indicators.get('macd_trend'),
            # Moyennes mobiles pour contexte tendance
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            'hull_20': self.indicators.get('hull_20'),
            # ADX pour force de tendance
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            # Volume pour confirmation
            'volume_ratio': self.indicators.get('volume_ratio'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            'relative_volume': self.indicators.get('relative_volume'),
            # ATR pour contexte volatilité
            'atr_14': self.indicators.get('atr_14'),
            'atr_percentile': self.indicators.get('atr_percentile'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            # VWAP pour niveaux institutionnels
            'vwap_10': self.indicators.get('vwap_10'),
            'anchored_vwap': self.indicators.get('anchored_vwap'),
            # Market structure
            'market_regime': self.indicators.get('market_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
            'support_levels': self.indicators.get('support_levels'),
            'resistance_levels': self.indicators.get('resistance_levels'),
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
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les seuils ROC.
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
        
        # Analyser les valeurs ROC disponibles
        roc_analysis = self._analyze_roc_values(values)
        if roc_analysis is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Aucune valeur ROC disponible",
                "metadata": {"strategy": self.name}
            }
            
        # Vérifier si les seuils sont dépassés
        threshold_result = self._check_thresholds(roc_analysis)
        if threshold_result is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Aucun seuil ROC dépassé",
                "metadata": {"strategy": self.name}
            }
            
        # Créer le signal avec confirmations
        return self._create_roc_signal(values, current_price or 0.0, roc_analysis, threshold_result)
        
    def _analyze_roc_values(self, values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyse les valeurs ROC disponibles et choisit la principale."""
        roc_data = {}
        
        # Collecter les valeurs ROC disponibles
        # Utiliser roc comme indicateur principal (existe dans analyzer_data)
        roc_value = values.get('roc_10')  # Chercher d'abord roc_10 mais utiliser 'roc' 
        if roc_value is None:
            roc_value = self.indicators.get('roc')  # Fallback vers 'roc' standard
        
        if roc_value is not None:
            try:
                roc_data['10'] = float(roc_value)
            except (ValueError, TypeError):
                pass
                    
        # Ajouter momentum_score s'il est disponible (existe dans analyzer_data)
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                # momentum_score est déjà en format 0-100, convertir en ROC-like (-100 à +100)
                # Normaliser de 0-100 vers -100 à +100 (50 = neutre = 0)
                mom_val = (float(momentum_score) - 50) * 2
                roc_data['momentum_score'] = mom_val
            except (ValueError, TypeError):
                pass
                
        if not roc_data:
            return None
            
        # Choisir le ROC principal (priorité ROC, puis momentum_score)
        primary_roc = None
        primary_period = None
        
        if '10' in roc_data:
            primary_roc = roc_data['10']
            primary_period = 'roc'
        elif 'momentum_score' in roc_data:
            primary_roc = roc_data['momentum_score']
            primary_period = 'momentum_score'
            
        if primary_roc is None:
            return None
            
        return {
            'primary_roc': primary_roc,
            'primary_period': primary_period,
            'all_roc': roc_data,
            'roc_strength': abs(primary_roc)
        }
        
    def _check_thresholds(self, roc_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Vérifie si les seuils ROC sont dépassés."""
        primary_roc = roc_analysis['primary_roc']
        
        signal_type = None
        threshold_level = None
        is_extreme = False
        
        # Vérifier seuils haussiers
        if primary_roc >= self.extreme_bullish_threshold:
            signal_type = "bullish"
            threshold_level = "extreme"
            is_extreme = True
        elif primary_roc >= self.bullish_threshold:
            signal_type = "bullish"
            threshold_level = "normal"
            
        # Vérifier seuils baissiers
        elif primary_roc <= self.extreme_bearish_threshold:
            signal_type = "bearish"
            threshold_level = "extreme"
            is_extreme = True
        elif primary_roc <= self.bearish_threshold:
            signal_type = "bearish"
            threshold_level = "normal"
            
        if signal_type is None:
            return None
            
        return {
            'signal_type': signal_type,
            'threshold_level': threshold_level,
            'is_extreme': is_extreme,
            'roc_value': primary_roc,
            'exceeded_by': abs(primary_roc - (self.bullish_threshold if signal_type == "bullish" else self.bearish_threshold))
        }
        
    def _create_roc_signal(self, values: Dict[str, Any], current_price: float,
                          roc_analysis: Dict[str, Any], threshold_result: Dict[str, Any]) -> Dict[str, Any]:
        """Crée le signal ROC avec confirmations."""
        signal_type = threshold_result['signal_type']
        threshold_level = threshold_result['threshold_level']
        roc_value = threshold_result['roc_value']
        
        signal_side = "BUY" if signal_type == "bullish" else "SELL"
        base_confidence = 0.55  # Base modérée pour momentum
        confidence_boost = 0.0
        
        # Construction de la raison
        direction = "haussier" if signal_type == "bullish" else "baissier"
        level_text = "extrême" if threshold_result['is_extreme'] else "normal"
        reason = f"ROC {direction} {level_text}: {roc_value:.2f}%"
        
        # Bonus selon le niveau de seuil
        if threshold_result['is_extreme']:
            confidence_boost += 0.20  # Momentum extrême = signal fort
            reason += " (momentum extrême)"
        else:
            confidence_boost += 0.15  # Momentum normal
            reason += " (momentum significatif)"
            
        # Bonus selon l'excès par rapport au seuil
        exceeded_by = threshold_result['exceeded_by']
        if exceeded_by > 2.0:  # Dépasse largement le seuil
            confidence_boost += 0.15
            reason += f" + excès important ({exceeded_by:.1f}%)"
        elif exceeded_by > 1.0:
            confidence_boost += 0.10
            reason += f" + excès modéré ({exceeded_by:.1f}%)"
        else:
            confidence_boost += 0.05
            reason += f" + excès faible ({exceeded_by:.1f}%)"
            
        # Confirmation avec momentum_score global
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                momentum_aligned = (signal_side == "BUY" and momentum > self.momentum_confirmation_threshold) or \
                                 (signal_side == "SELL" and momentum < -self.momentum_confirmation_threshold)
                
                if momentum_aligned:
                    confidence_boost += 0.15
                    reason += " + momentum score confirmé"
                elif abs(momentum) > 0.1:  # Momentum faible mais dans la bonne direction
                    confidence_boost += 0.08
                    reason += " + momentum score aligné"
                else:
                    confidence_boost -= 0.05
                    reason += " mais momentum score faible"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec RSI
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if signal_side == "BUY":
                    if rsi > 60:  # RSI confirme momentum haussier
                        confidence_boost += 0.12
                        reason += " + RSI confirme"
                    elif rsi > 50:
                        confidence_boost += 0.08
                        reason += " + RSI favorable"
                    elif rsi < 30:  # Oversold = momentum peut continuer
                        confidence_boost += 0.05
                        reason += " + RSI oversold"
                else:  # SELL
                    if rsi < 40:  # RSI confirme momentum baissier
                        confidence_boost += 0.12
                        reason += " + RSI confirme"
                    elif rsi < 50:
                        confidence_boost += 0.08
                        reason += " + RSI favorable"
                    elif rsi > 70:  # Overbought = momentum peut continuer
                        confidence_boost += 0.05
                        reason += " + RSI overbought"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec MACD
        macd_line = values.get('macd_line')
        macd_signal = values.get('macd_signal')
        if macd_line is not None and macd_signal is not None:
            try:
                macd_val = float(macd_line)
                macd_sig = float(macd_signal)
                macd_bullish = macd_val > macd_sig
                
                if (signal_side == "BUY" and macd_bullish) or (signal_side == "SELL" and not macd_bullish):
                    confidence_boost += 0.10
                    reason += " + MACD confirme"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec CCI (Commodity Channel Index)
        cci_20 = values.get('cci_20')
        if cci_20 is not None:
            try:
                cci = float(cci_20)
                if signal_side == "BUY" and cci > 100:  # CCI overbought = momentum fort
                    confidence_boost += 0.10
                    reason += " + CCI momentum fort"
                elif signal_side == "SELL" and cci < -100:  # CCI oversold = momentum fort
                    confidence_boost += 0.10
                    reason += " + CCI momentum fort"
                elif signal_side == "BUY" and cci > 0:
                    confidence_boost += 0.05
                    reason += " + CCI favorable"
                elif signal_side == "SELL" and cci < 0:
                    confidence_boost += 0.05
                    reason += " + CCI favorable"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec Williams %R
        williams_r = values.get('williams_r')
        if williams_r is not None:
            try:
                wr = float(williams_r)
                if signal_side == "BUY" and wr > -50:  # Williams R indique momentum haussier
                    confidence_boost += 0.08
                    reason += " + Williams R confirme"
                elif signal_side == "SELL" and wr < -50:  # Williams R indique momentum baissier
                    confidence_boost += 0.08
                    reason += " + Williams R confirme"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec volume
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= 1.5:  # Volume élevé confirme le momentum
                    confidence_boost += 0.15
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.2:
                    confidence_boost += 0.10
                    reason += f" + volume modéré ({vol_ratio:.1f}x)"
                else:
                    confidence_boost -= 0.05
                    reason += f" mais volume faible ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec ADX (force de tendance)
        adx_14 = values.get('adx_14')
        if adx_14 is not None:
            try:
                adx = float(adx_14)
                if adx > 25:  # Tendance forte = momentum plus fiable
                    confidence_boost += 0.12
                    reason += " + tendance forte"
                elif adx > 20:
                    confidence_boost += 0.08
                    reason += " + tendance modérée"
            except (ValueError, TypeError):
                pass
                
        # Context avec VWAP
        vwap_10 = values.get('vwap_10')
        if vwap_10 is not None and current_price is not None:
            try:
                vwap = float(vwap_10)
                if (signal_side == "BUY" and current_price > vwap) or \
                   (signal_side == "SELL" and current_price < vwap):
                    confidence_boost += 0.08
                    reason += " + VWAP aligné"
            except (ValueError, TypeError):
                pass
                
        # Market regime
        market_regime = values.get('market_regime')
        if market_regime == "trending":
            confidence_boost += 0.10
            reason += " (marché trending)"
        elif market_regime == "ranging":
            confidence_boost -= 0.05  # Momentum moins fiable en ranging
            reason += " (marché ranging)"
            
        # Volatility context
        volatility_regime = values.get('volatility_regime')
        if volatility_regime == "high":
            if threshold_result['is_extreme']:
                confidence_boost += 0.05  # Momentum extrême en haute volatilité = signal fort
                reason += " + volatilité élevée favorable"
            else:
                confidence_boost -= 0.05  # Momentum normal en haute volatilité = moins fiable
                reason += " mais volatilité élevée"
        elif volatility_regime == "normal":
            confidence_boost += 0.05
            reason += " + volatilité normale"
            
        # Confluence score
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if confluence > 0.7:
                    confidence_boost += 0.10
                    reason += " + confluence élevée"
                elif confluence > 0.5:
                    confidence_boost += 0.05
                    reason += " + confluence modérée"
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
                "roc_value": roc_value,
                "roc_period": roc_analysis['primary_period'],
                "threshold_level": threshold_level,
                "is_extreme": threshold_result['is_extreme'],
                "exceeded_by": threshold_result['exceeded_by'],
                "signal_type": signal_type,
                "all_roc_values": roc_analysis['all_roc'],
                "momentum_score": values.get('momentum_score'),
                "rsi_14": values.get('rsi_14'),
                "macd_line": values.get('macd_line'),
                "macd_signal": values.get('macd_signal'),
                "cci_20": values.get('cci_20'),
                "williams_r": values.get('williams_r'),
                "volume_ratio": values.get('volume_ratio'),
                "adx_14": values.get('adx_14'),
                "vwap_10": values.get('vwap_10'),
                "market_regime": values.get('market_regime'),
                "volatility_regime": values.get('volatility_regime'),
                "confluence_score": values.get('confluence_score')
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que les données ROC nécessaires sont présentes."""
        if not super().validate_data():
            return False
            
        # Au minimum, il faut un indicateur ROC ou momentum
        required_any = ['roc_10', 'roc_20', 'momentum_10']
        
        for indicator in required_any:
            if indicator in self.indicators and self.indicators[indicator] is not None:
                return True
                
        logger.warning(f"{self.name}: Aucun indicateur ROC/momentum disponible")
        return False
