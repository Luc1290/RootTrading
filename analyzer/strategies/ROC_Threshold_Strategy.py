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
        # Paramètres ROC - Valeurs déjà en pourcentage
        self.bullish_threshold = 0.8   # ROC > +0.8% pour signal haussier
        self.bearish_threshold = -0.8  # ROC < -0.8% pour signal baissier 
        self.extreme_bullish_threshold = 3.5  # ROC > +3.5% = momentum extrême
        self.extreme_bearish_threshold = -3.5  # ROC < -3.5% = momentum extrême
        self.momentum_confirmation_threshold = 50  # Seuil momentum_score
        # FILTRES RÉALISTES
        self.min_confidence_threshold = 0.40  # Confidence minimum alignée PPO
        self.max_boost_multiplier = 0.50  # Boosts plus généreux
        # Cap ROC - mettre à None pour désactiver
        self.enable_roc_cap = True  # Active/désactive le cap ROC
        
    def _cap_roc_by_timeframe(self, roc: float) -> float:
        """
        Limite les valeurs ROC extrêmes selon le timeframe.
        Pour éviter les valeurs aberrantes en scalping intraday.
        Adapté au marché crypto qui peut avoir des mouvements violents.
        """
        # Récupérer le timeframe depuis les données
        timeframe = self.data.get('timeframe', '3m')
        
        # Caps par timeframe - adaptés pour la crypto (mouvements plus violents possibles)
        caps = {
            '1m': 10.0,   # Max ±10% en 1 minute (flash crash/pump possible)
            '3m': 15.0,   # Max ±15% en 3 minutes  
            '5m': 20.0,   # Max ±20% en 5 minutes
            '15m': 30.0,  # Max ±30% en 15 minutes
            '30m': 40.0,  # Max ±40% en 30 minutes
            '1h': 50.0,   # Max ±50% en 1 heure
            '4h': 75.0,   # Max ±75% en 4 heures
            '1d': 100.0,  # Max ±100% en 1 jour
        }
        
        max_cap = caps.get(timeframe, 25.0)  # Par défaut 25%
        
        # Appliquer le cap dans les deux directions
        return max(-max_cap, min(max_cap, roc))
        
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
        # ROC_10 est le principal indicateur disponible dans analyzer_data
        roc_value = values.get('roc_10')  # ROC sur 10 périodes
        if roc_value is None:
            roc_value = values.get('roc_20')  # Fallback vers ROC 20 périodes
        
        if roc_value is not None:
            try:
                roc_data['10'] = float(roc_value)
            except (ValueError, TypeError):
                pass
                    
        # Ajouter momentum_score s'il est disponible (existe dans analyzer_data)
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                # momentum_score est en format 0-100, convertir en format ROC décimal équivalent
                # Normaliser de 0-100 vers -1 à +1 (50 = neutre = 0)
                mom_val = (float(momentum_score) - 50) / 50
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
        
        # Garder la valeur originale et la valeur cappée
        roc_value_original = threshold_result['roc_value']
        
        # Appliquer le cap seulement si activé
        if self.enable_roc_cap:
            roc_value = self._cap_roc_by_timeframe(roc_value_original)
            was_capped = roc_value != roc_value_original
        else:
            roc_value = roc_value_original
            was_capped = False
        
        signal_side = "BUY" if signal_type == "bullish" else "SELL"
        base_confidence = 0.65  # Réduit pour être plus sélectif
        confidence_boost = 0.0
        
        # Construction de la raison
        direction = "haussier" if signal_type == "bullish" else "baissier"
        level_text = "extrême" if threshold_result['is_extreme'] else "normal"
        
        # Afficher la valeur originale si cappée
        if was_capped:
            reason = f"ROC {direction} {level_text}: {roc_value_original:.2f}% (cappé à {roc_value:.2f}%)"
        else:
            reason = f"ROC {direction} {level_text}: {roc_value:.2f}%"
        
        # Bonus selon le niveau de seuil - ENCORE RÉDUIT pour winrate
        if threshold_result['is_extreme']:
            confidence_boost += 0.10  # Réduit pour être plus sélectif
            reason += " (momentum extrême)"
        else:
            confidence_boost += 0.05  # Réduit significativement
            reason += " (momentum significatif)"
            
        # Bonus selon l'excès par rapport au seuil - BOOSTÉS pour vrais moves
        exceeded_by = threshold_result['exceeded_by']
        if exceeded_by > 5:  # Dépasse LARGEMENT le seuil (>5%) - vrai move marché
            confidence_boost += 0.25  # Doublé pour signaux exceptionnels
            reason += f" + excès MAJEUR ({exceeded_by:.1f}%)"
        elif exceeded_by > 3:  # Excès important (>3%)
            confidence_boost += 0.18  # Augmenté
            reason += f" + excès important ({exceeded_by:.1f}%)"
        elif exceeded_by > 1:  # Excès modéré (>1%)
            confidence_boost += 0.08  # Doublé
            reason += f" + excès modéré ({exceeded_by:.1f}%)"
        else:
            # Excès trop faible - pénalité
            confidence_boost -= 0.05
            reason += f" mais excès FAIBLE ({exceeded_by:.1f}%)"
            
        # CORRECTION: Momentum confirmation directionnelle stricte
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                
                # MOMENTUM VALIDATION ULTRA STRICTE pour winrate
                # IMPORTANT: momentum_score est en échelle 0-100, PAS normalisé ici!
                if signal_side == "BUY":
                    # Pour BUY, on veut un momentum > 50 (haussier)
                    if momentum > 80:  # Momentum EXCEPTIONNEL requis
                        confidence_boost += 0.10  # Bonus réduit
                        reason += f" + momentum exceptionnel ({momentum:.1f})"
                    elif momentum > 70:  # Momentum très positif requis
                        confidence_boost += 0.06  # Bonus réduit
                        reason += f" + momentum fort ({momentum:.1f})"
                    elif momentum > 65:  # Momentum minimum acceptable
                        confidence_boost += 0.03  # Bonus minimal
                        reason += f" + momentum correct ({momentum:.1f})"
                    elif momentum < 40:  # Momentum contradictoire = rejet
                        return {
                            "side": None,
                            "confidence": 0.0,
                            "strength": "weak",
                            "reason": f"Rejet ROC BUY: momentum trop faible ({momentum:.1f})",
                            "metadata": {"strategy": self.name, "momentum_score": momentum}
                        }
                    elif momentum < 50:  # Momentum défavorable
                        confidence_boost -= 0.15
                        reason += f" avec momentum défavorable ({momentum:.1f})"
                        
                # SELL : validation momentum plus stricte  
                elif signal_side == "SELL":
                    # Pour SELL, on veut un momentum < 50 (baissier)
                    if momentum < 20:  # Momentum EXCEPTIONNEL requis
                        confidence_boost += 0.10  # Bonus réduit
                        reason += f" + momentum exceptionnel ({momentum:.1f})"
                    elif momentum < 30:  # Momentum très négatif requis
                        confidence_boost += 0.06  # Bonus réduit
                        reason += f" + momentum fort ({momentum:.1f})"
                    elif momentum < 35:  # Momentum minimum acceptable
                        confidence_boost += 0.03  # Bonus minimal
                        reason += f" + momentum correct ({momentum:.1f})"
                    elif momentum > 60:  # Momentum contradictoire = rejet
                        return {
                            "side": None,
                            "confidence": 0.0,
                            "strength": "weak",
                            "reason": f"Rejet ROC SELL: momentum trop fort ({momentum:.1f})",
                            "metadata": {"strategy": self.name, "momentum_score": momentum}
                        }
                    elif momentum > 50:  # Momentum défavorable
                        confidence_boost -= 0.15
                        reason += f" avec momentum défavorable ({momentum:.1f})"
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
                    elif rsi > 55:  # RSI favorable durci
                        confidence_boost += 0.08
                        reason += " + RSI favorable"
                    elif rsi < 30:  # Oversold = momentum peut continuer
                        confidence_boost += 0.05
                        reason += " + RSI oversold"
                else:  # SELL
                    if rsi < 40:  # RSI confirme momentum baissier
                        confidence_boost += 0.12
                        reason += " + RSI confirme"
                    elif rsi < 45:  # RSI favorable durci
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
                
        # CORRECTION: CCI avec seuils directionnels adaptatifs
        cci_20 = values.get('cci_20')
        if cci_20 is not None:
            try:
                cci = float(cci_20)
                # BUY : CCI favorise momentum haussier avec seuils adaptatifs
                if signal_side == "BUY":
                    if cci > 150:  # CCI très élevé = momentum exceptionnel
                        confidence_boost += 0.15
                        reason += f" + CCI momentum exceptionnel ({cci:.0f})"
                    elif cci > 100:  # CCI overbought = momentum fort haussier
                        confidence_boost += 0.12
                        reason += f" + CCI momentum fort ({cci:.0f})"
                    elif cci > 50:  # CCI positif modéré
                        confidence_boost += 0.08
                        reason += f" + CCI favorable ({cci:.0f})"
                    elif cci > 0:  # CCI légèrement positif
                        confidence_boost += 0.04
                        reason += f" + CCI neutre positif ({cci:.0f})"
                    elif cci < -50:  # CCI négatif = contradictoire avec ROC haussier
                        confidence_boost -= 0.05
                        reason += f" mais CCI négatif ({cci:.0f})"
                        
                # SELL : CCI favorise momentum baissier avec seuils adaptatifs
                elif signal_side == "SELL":
                    if cci < -150:  # CCI très bas = momentum exceptionnel
                        confidence_boost += 0.15
                        reason += f" + CCI momentum exceptionnel ({cci:.0f})"
                    elif cci < -100:  # CCI oversold = momentum fort baissier
                        confidence_boost += 0.12
                        reason += f" + CCI momentum fort ({cci:.0f})"
                    elif cci < -50:  # CCI négatif modéré
                        confidence_boost += 0.08
                        reason += f" + CCI favorable ({cci:.0f})"
                    elif cci < 0:  # CCI légèrement négatif
                        confidence_boost += 0.04
                        reason += f" + CCI neutre négatif ({cci:.0f})"
                    elif cci > 50:  # CCI positif = contradictoire avec ROC baissier
                        confidence_boost -= 0.05
                        reason += f" mais CCI positif ({cci:.0f})"
            except (ValueError, TypeError):
                pass
                
        # CORRECTION: Williams %R avec zones directionnelles spécifiques
        williams_r = values.get('williams_r')
        if williams_r is not None:
            try:
                wr = float(williams_r)
                # BUY : Williams %R favorise sortie d'oversold et momentum haussier
                if signal_side == "BUY":
                    if wr > -20:  # Williams R overbought = momentum haussier fort
                        confidence_boost += 0.12
                        reason += f" + Williams R momentum fort ({wr:.0f})"
                    elif wr > -40:  # Williams R sortie d'oversold
                        confidence_boost += 0.10
                        reason += f" + Williams R confirme ({wr:.0f})"
                    elif wr > -60:  # Williams R neutre haussier
                        confidence_boost += 0.06
                        reason += f" + Williams R favorable ({wr:.0f})"
                    elif wr < -80:  # Williams R oversold profond = potentiel mais risqué
                        confidence_boost += 0.03
                        reason += f" + Williams R oversold ({wr:.0f})"
                        
                # SELL : Williams %R favorise entrée en oversold et momentum baissier
                elif signal_side == "SELL":
                    if wr < -80:  # Williams R oversold = momentum baissier fort
                        confidence_boost += 0.12
                        reason += f" + Williams R momentum fort ({wr:.0f})"
                    elif wr < -60:  # Williams R entrée en oversold
                        confidence_boost += 0.10
                        reason += f" + Williams R confirme ({wr:.0f})"
                    elif wr < -40:  # Williams R neutre baissier
                        confidence_boost += 0.06
                        reason += f" + Williams R favorable ({wr:.0f})"
                    elif wr > -20:  # Williams R overbought = potentiel mais risqué
                        confidence_boost += 0.03
                        reason += f" + Williams R overbought ({wr:.0f})"
            except (ValueError, TypeError):
                pass
                
        # CORRECTION: Volume confirmation directionnelle - momentum BUY vs SELL
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                
                # Volume DURCI avec rejet volume insuffisant
                if vol_ratio < 1.0:  # Volume insuffisant = rejet direct
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet ROC: volume insuffisant ({vol_ratio:.1f}x)",
                        "metadata": {"strategy": self.name, "volume_ratio": vol_ratio}
                    }
                
                # BUY : momentum haussier EXIGE volume fort
                if signal_side == "BUY":
                    if vol_ratio >= 2.5:  # Volume EXCEPTIONNEL
                        confidence_boost += 0.15
                        reason += f" + volume exceptionnel BUY ({vol_ratio:.1f}x)"
                    elif vol_ratio >= 2.0:  # Volume très élevé
                        confidence_boost += 0.12
                        reason += f" + volume fort BUY ({vol_ratio:.1f}x)"
                    elif vol_ratio >= 1.5:  # Volume correct
                        confidence_boost += 0.06
                        reason += f" + volume correct ({vol_ratio:.1f}x)"
                    elif vol_ratio < 1.2:  # Volume faible
                        confidence_boost -= 0.08
                        reason += f" volume modéré ({vol_ratio:.1f}x)"
                        
                # SELL : momentum baissier EXIGE aussi du volume
                elif signal_side == "SELL":
                    if vol_ratio >= 2.0:  # Volume très élevé = panic selling
                        confidence_boost += 0.15
                        reason += f" + volume panic SELL ({vol_ratio:.1f}x)"
                    elif vol_ratio >= 1.5:  # Volume élevé
                        confidence_boost += 0.12
                        reason += f" + volume élevé SELL ({vol_ratio:.1f}x)"
                    elif vol_ratio >= 1.2:  # Volume correct
                        confidence_boost += 0.06
                        reason += f" + volume correct ({vol_ratio:.1f}x)"
                    else:  # Volume faible
                        confidence_boost -= 0.06
                        reason += f" volume faible ({vol_ratio:.1f}x)"
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
                
        # Market regime avec REJET contradictions
        market_regime = values.get('market_regime')
        if (signal_side == "BUY" and market_regime == "TRENDING_BEAR") or \
           (signal_side == "SELL" and market_regime == "TRENDING_BULL"):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Rejet ROC {signal_side}: régime contradictoire ({market_regime})",
                "metadata": {"strategy": self.name, "market_regime": market_regime}
            }
        elif market_regime in ["TRENDING_BULL", "TRENDING_BEAR"]:
            confidence_boost += 0.10
            reason += " (marché trending)"
        elif market_regime == "RANGING":
            confidence_boost += 0.02  # ROC peut être utile en ranging pour oscillations
            reason += " (marché ranging - oscillations ROC)"
            
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
            
        # Confluence score DURCIE avec rejet
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if confluence < 40:  # Confluence insuffisante = rejet
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet ROC: confluence insuffisante ({confluence})",
                        "metadata": {"strategy": self.name, "confluence_score": confluence}
                    }
                elif confluence > 70:
                    confidence_boost += 0.12
                    reason += " + confluence élevée"
                elif confluence > 60:
                    confidence_boost += 0.08
                    reason += " + confluence bonne"
            except (ValueError, TypeError):
                pass
                
        # NOUVEAU: LIMITATION des boosts et filtre final
        # Plafonner les boosts pour éviter les confidences excessives
        confidence_boost = min(confidence_boost, self.max_boost_multiplier)
        
        # Calculer confidence provisoire
        raw_confidence = base_confidence * (1 + confidence_boost)
        
        # Filtre final - rejeter si confidence insuffisante
        if raw_confidence < self.min_confidence_threshold:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal ROC {signal_side} rejeté - confidence insuffisante ({raw_confidence:.2f} < {self.min_confidence_threshold})",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "rejected_signal": signal_side,
                    "raw_confidence": raw_confidence,
                    "roc_value": roc_value,
                    "threshold_level": threshold_level,
                    "min_required": self.min_confidence_threshold
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
                "roc_value": roc_value_original,  # Valeur ROC originale
                "roc_value_capped": roc_value if was_capped else None,  # Valeur cappée si applicable
                "was_capped": was_capped,
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
