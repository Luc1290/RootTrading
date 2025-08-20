"""
Spike_Reaction_Buy_Strategy - Stratégie basée sur la réaction après spike baissier.
Détecte les spikes de vente (crash/dump) suivis d'une stabilisation pour signaler des achats opportunistes.
OPTIMISÉE POUR CRYPTO SPOT INTRADAY
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class Spike_Reaction_Buy_Strategy(BaseStrategy):
    """
    Stratégie détectant les opportunités d'achat après spikes baissiers stabilisés.
    
    Pattern de réaction après spike :
    1. Spike baissier détecté (price/volume anormal)
    2. Volume élevé confirme la panique/liquidation
    3. Prix se stabilise et arrête de chuter
    4. Indicateurs techniques montrent survente extrême  
    5. Rebond technique attendu (opportunité BUY)
    
    Types de spikes ciblés :
    - Dump/crash avec volume exceptionnel
    - Liquidation cascade
    - Survente extrême avec reversal
    - Anomalie détectée par le système
    
    Signaux générés:
    - BUY: Après spike baissier stabilisé avec confluence techniques
    - Pas de SELL (focus sur opportunités post-crash)
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        
        # Paramètres de détection spike baissier - OPTIMISÉS CRYPTO
        self.min_price_drop = -0.018             # 1.8% chute minimum (plus strict)
        self.severe_price_drop = -0.035          # 3.5% chute sévère (plus sélectif)
        self.extreme_price_drop = -0.06          # 6% chute extrême (réaliste crypto)
        
        # Paramètres volume (confirmation spike) - PLUS STRICTS
        self.min_spike_volume = 1.8              # Volume 1.8x normal minimum
        self.strong_spike_volume = 3.0           # Volume 3x pour spike fort
        self.extreme_spike_volume = 5.0          # Volume 5x pour spike extrême
        
        # Paramètres RSI (survente extrême crypto)
        self.oversold_rsi_threshold = 28         # RSI survente crypto
        self.extreme_oversold_threshold = 18     # RSI survente extrême crypto
        self.williams_r_oversold = -82           # Williams %R survente strict
        
        # Paramètres stabilisation - RENFORCÉS
        self.stabilization_bars = 3              # 3 barres minimum pour stabilisation
        self.max_continued_drop = -0.005         # Chute max autorisée (-0.5%)
        self.min_volatility_ratio = 0.15         # Volatilité réduite pour stabilisation
        
        # Paramètres momentum reversal - PLUS STRICTS
        self.momentum_reversal_threshold = 52    # Momentum reversal minimum
        self.strong_momentum_reversal = 60       # Momentum reversal fort
        self.min_roc_improvement = -0.008        # ROC amélioration (-0.8% max)
        self.min_volume_quality = 45             # Volume quality minimum
        
        # Nouveaux paramètres pour timing et protection
        self.min_confluence_score = 40           # Confluence minimum
        self.max_atr_spike = 0.08                # ATR maximum pour éviter volatilité excessive
        self.support_proximity_threshold = 0.015 # 1.5% du support pour bonus
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            # Momentum et ROC (détection spike prix)
            'roc_10': self.indicators.get('roc_10'),
            'roc_20': self.indicators.get('roc_20'),
            'momentum_10': self.indicators.get('momentum_10'),
            'momentum_score': self.indicators.get('momentum_score'),
            
            # Volume analysis (confirmation spike)
            'volume_spike_multiplier': self.indicators.get('volume_spike_multiplier'),
            'relative_volume': self.indicators.get('relative_volume'),
            'volume_ratio': self.indicators.get('volume_ratio'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            'volume_pattern': self.indicators.get('volume_pattern'),
            
            # Oscillateurs (survente extrême)
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            'williams_r': self.indicators.get('williams_r'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            'cci_20': self.indicators.get('cci_20'),
            
            # Volatilité et contexte
            'atr_14': self.indicators.get('atr_14'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            'bb_position': self.indicators.get('bb_position'),
            'bb_lower': self.indicators.get('bb_lower'),
            'bb_width': self.indicators.get('bb_width'),
            
            # Pattern et anomalies (détection système)
            'pattern_detected': self.indicators.get('pattern_detected'),
            'pattern_confidence': self.indicators.get('pattern_confidence'),
            'anomaly_detected': self.indicators.get('anomaly_detected'),
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score'),
            
            # Support (potentiel rebond)
            'nearest_support': self.indicators.get('nearest_support'),
            'support_strength': self.indicators.get('support_strength'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias')
        }
        
    def _detect_price_spike_down(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte un spike baissier de prix avec seuils crypto."""
        spike_score = 0.0
        spike_indicators = []
        
        # ROC 10 périodes (mouvement récent) - PLUS STRICT
        roc_10 = values.get('roc_10')
        if roc_10 is not None:
            try:
                roc_val = float(roc_10)
                if roc_val <= self.extreme_price_drop:
                    spike_score += 0.5  # Score augmenté pour spike extrême
                    spike_indicators.append(f"ROC10 chute EXTRÊME ({roc_val*100:.1f}%)")
                elif roc_val <= self.severe_price_drop:
                    spike_score += 0.35
                    spike_indicators.append(f"ROC10 chute SÉVÈRE ({roc_val*100:.1f}%)")
                elif roc_val <= self.min_price_drop:
                    spike_score += 0.25
                    spike_indicators.append(f"ROC10 chute significative ({roc_val*100:.1f}%)")
            except (ValueError, TypeError):
                pass
                
        # ROC 20 périodes (confirmation tendance baissière)
        roc_20 = values.get('roc_20')
        if roc_20 is not None:
            try:
                roc_20_val = float(roc_20)
                if roc_20_val <= self.min_price_drop:
                    spike_score += 0.15
                    spike_indicators.append(f"ROC20 confirme ({roc_20_val*100:.1f}%)")
                # Bonus si ROC20 moins négatif que ROC10 (début stabilisation)
                if roc_10 is not None and roc_20_val > float(roc_10) * 0.7:
                    spike_score += 0.1
                    spike_indicators.append("Début stabilisation ROC")
            except (ValueError, TypeError):
                pass
                
        # Momentum négatif fort - CRITÈRE RENFORCÉ
        momentum_10 = values.get('momentum_10')
        if momentum_10 is not None:
            try:
                momentum_val = float(momentum_10)
                if momentum_val < -0.02:  # Momentum très négatif
                    spike_score += 0.2
                    spike_indicators.append(f"Momentum très négatif ({momentum_val:.3f})")
                elif momentum_val < 0:
                    spike_score += 0.12
                    spike_indicators.append(f"Momentum négatif ({momentum_val:.3f})")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_price_spike': spike_score >= 0.4,  # Seuil plus strict
            'score': spike_score,
            'indicators': spike_indicators
        }
        
    def _detect_volume_spike(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte un spike de volume avec seuils crypto stricts."""
        volume_score = 0.0
        volume_indicators = []
        
        # Volume spike multiplier (principal) - PLUS STRICT
        volume_spike_mult = values.get('volume_spike_multiplier')
        if volume_spike_mult is not None:
            try:
                spike_mult = float(volume_spike_mult)
                if spike_mult >= self.extreme_spike_volume:
                    volume_score += 0.5  # Score augmenté
                    volume_indicators.append(f"Volume spike EXTRÊME ({spike_mult:.1f}x)")
                elif spike_mult >= self.strong_spike_volume:
                    volume_score += 0.35
                    volume_indicators.append(f"Volume spike FORT ({spike_mult:.1f}x)")
                elif spike_mult >= self.min_spike_volume:
                    volume_score += 0.25
                    volume_indicators.append(f"Volume spike ({spike_mult:.1f}x)")
            except (ValueError, TypeError):
                pass
                
        # Volume ratio (confirmation)
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= 2.5:  # Seuil plus strict
                    volume_score += 0.2
                    volume_indicators.append(f"Volume ratio élevé ({vol_ratio:.1f}x)")
                elif vol_ratio >= 1.5:
                    volume_score += 0.1
            except (ValueError, TypeError):
                pass
                
        # Trade intensity (activité anormale)
        trade_intensity = values.get('trade_intensity')
        if trade_intensity is not None:
            try:
                intensity = float(trade_intensity)
                if intensity >= 3.0:  # Seuil plus strict pour crypto
                    volume_score += 0.2
                    volume_indicators.append(f"Intensité TRÈS élevée ({intensity:.1f}x)")
                elif intensity >= 2.0:
                    volume_score += 0.1
                    volume_indicators.append(f"Intensité élevée ({intensity:.1f}x)")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_volume_spike': volume_score >= 0.35,  # Plus strict
            'score': volume_score,
            'indicators': volume_indicators
        }
        
    def _detect_oversold_extreme(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte survente extrême avec seuils crypto."""
        oversold_score = 0.0
        oversold_indicators = []
        
        # RSI en survente - SEUILS CRYPTO
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi_val = float(rsi_14)
                if rsi_val <= self.extreme_oversold_threshold:
                    oversold_score += 0.4  # Score augmenté
                    oversold_indicators.append(f"RSI survente EXTRÊME ({rsi_val:.1f})")
                elif rsi_val <= self.oversold_rsi_threshold:
                    oversold_score += 0.28
                    oversold_indicators.append(f"RSI survente crypto ({rsi_val:.1f})")
                elif rsi_val <= 35:  # Zone intermédiaire
                    oversold_score += 0.15
                    oversold_indicators.append(f"RSI favorable ({rsi_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # RSI 21 pour confirmation
        rsi_21 = values.get('rsi_21')
        if rsi_21 is not None:
            try:
                rsi_21_val = float(rsi_21)
                if rsi_21_val <= self.oversold_rsi_threshold + 2:
                    oversold_score += 0.15
                    oversold_indicators.append(f"RSI21 confirme ({rsi_21_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # Williams %R confirme survente - PLUS STRICT
        williams_r = values.get('williams_r')
        if williams_r is not None:
            try:
                wr_val = float(williams_r)
                if wr_val <= self.williams_r_oversold:
                    oversold_score += 0.22
                    oversold_indicators.append(f"Williams%R survente ({wr_val:.1f})")
                elif wr_val <= -75:
                    oversold_score += 0.12
            except (ValueError, TypeError):
                pass
                
        # Stochastic en survente - PLUS STRICT
        stoch_k = values.get('stoch_k')
        stoch_d = values.get('stoch_d')
        if stoch_k is not None and stoch_d is not None:
            try:
                k_val = float(stoch_k)
                d_val = float(stoch_d)
                if k_val <= 15 and d_val <= 15:  # Plus strict
                    oversold_score += 0.25
                    oversold_indicators.append(f"Stoch survente EXTRÊME (K={k_val:.1f}, D={d_val:.1f})")
                elif k_val <= 20 and d_val <= 20:
                    oversold_score += 0.15
                    oversold_indicators.append(f"Stoch survente (K={k_val:.1f}, D={d_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # CCI extreme - SEUIL CRYPTO
        cci_20 = values.get('cci_20')
        if cci_20 is not None:
            try:
                cci_val = float(cci_20)
                if cci_val <= -150:  # Plus strict pour crypto
                    oversold_score += 0.2
                    oversold_indicators.append(f"CCI survente EXTRÊME ({cci_val:.1f})")
                elif cci_val <= -100:
                    oversold_score += 0.12
                    oversold_indicators.append(f"CCI survente ({cci_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # Bollinger position basse
        bb_position = values.get('bb_position')
        if bb_position is not None:
            try:
                bb_pos = float(bb_position)
                if bb_pos <= 0.05:  # Très proche BB lower
                    oversold_score += 0.15
                    oversold_indicators.append(f"BB position TRÈS basse ({bb_pos:.3f})")
                elif bb_pos <= 0.1:
                    oversold_score += 0.08
                    oversold_indicators.append(f"BB position basse ({bb_pos:.2f})")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_oversold': oversold_score >= 0.5,  # Seuil plus strict
            'score': oversold_score,
            'indicators': oversold_indicators
        }
        
    def _detect_stabilization(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte la stabilisation après spike avec critères renforcés."""
        stabilization_score = 0.0
        stabilization_indicators = []
        
        # Momentum score amélioration (format 0-100, 50=neutre)
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum_val = float(momentum_score)
                if momentum_val >= self.strong_momentum_reversal:
                    stabilization_score += 0.35  # Score augmenté
                    stabilization_indicators.append(f"Momentum reversal FORT ({momentum_val:.1f})")
                elif momentum_val >= self.momentum_reversal_threshold:
                    stabilization_score += 0.25
                    stabilization_indicators.append(f"Momentum reversal ({momentum_val:.1f})")
                elif momentum_val >= 50:  # Au moins neutre
                    stabilization_score += 0.12
                    stabilization_indicators.append(f"Momentum neutre ({momentum_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # Volume pattern (décroissance après spike) - PLUS IMPORTANT
        volume_pattern = values.get('volume_pattern')
        if volume_pattern in ['DECLINING']:  # Volume décroissant = stabilisation
            stabilization_score += 0.25  # Score augmenté
            stabilization_indicators.append(f"Volume pattern: {volume_pattern}")
        elif volume_pattern in ['BUILDUP']:  # Accumulation
            stabilization_score += 0.15
            stabilization_indicators.append(f"Volume pattern: {volume_pattern}")
            
        # ROC amélioration (moins négatif) - CRITÈRE RENFORCÉ
        roc_10 = values.get('roc_10')
        if roc_10 is not None:
            try:
                roc_val = float(roc_10)
                if roc_val >= self.min_roc_improvement:
                    stabilization_score += 0.25
                    stabilization_indicators.append(f"ROC amélioration ({roc_val*100:.1f}%)")
                elif roc_val >= -0.015:  # Amélioration partielle
                    stabilization_score += 0.15
                    stabilization_indicators.append(f"ROC stabilisation ({roc_val*100:.1f}%)")
            except (ValueError, TypeError):
                pass
                
        # Pattern detected (reversal pattern) - BONUS FORT
        pattern_detected = values.get('pattern_detected')
        pattern_confidence = values.get('pattern_confidence')
        if pattern_detected and 'reversal' in str(pattern_detected).lower():
            stabilization_score += 0.2
            stabilization_indicators.append(f"Pattern reversal détecté")
            
            # Bonus si pattern confidence élevée
            if pattern_confidence is not None:
                try:
                    conf_val = float(pattern_confidence)
                    if conf_val >= 70:
                        stabilization_score += 0.15
                        stabilization_indicators.append(f"Pattern confidence {conf_val:.0f}%")
                except (ValueError, TypeError):
                    pass
                    
        # Volatilité en baisse (stabilisation)
        volatility_regime = values.get('volatility_regime')
        if volatility_regime in ['normal', 'low']:
            stabilization_score += 0.1
            stabilization_indicators.append(f"Volatilité {volatility_regime}")
            
        return {
            'is_stabilized': stabilization_score >= 0.45,  # Seuil plus strict
            'score': stabilization_score,
            'indicators': stabilization_indicators
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal BUY basé sur la réaction après spike baissier.
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
        
        # Filtre préliminaire : vérifier confluence minimum
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                conf_val = float(confluence_score)
                if conf_val < self.min_confluence_score:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Confluence insuffisante ({conf_val:.1f} < {self.min_confluence_score})",
                        "metadata": {"strategy": self.name, "confluence_score": conf_val}
                    }
            except (ValueError, TypeError):
                pass
                
        # Filtre préliminaire : éviter volatilité extrême
        atr_14 = values.get('atr_14')
        if atr_14 is not None:
            try:
                atr_val = float(atr_14)
                # Récupérer prix actuel pour calculer ATR relatif
                current_price = None
                if 'close' in self.data and self.data['close']:
                    try:
                        current_price = float(self.data['close'][-1])
                        atr_relative = atr_val / current_price
                        if atr_relative > self.max_atr_spike:
                            return {
                                "side": None,
                                "confidence": 0.0,
                                "strength": "weak",
                                "reason": f"Volatilité excessive (ATR: {atr_relative*100:.1f}% > {self.max_atr_spike*100:.1f}%)",
                                "metadata": {"strategy": self.name, "atr_relative": atr_relative}
                            }
                    except (IndexError, ValueError, TypeError):
                        pass
            except (ValueError, TypeError):
                pass
                
        # Filtre préliminaire : éviter signaux en tendance baissière forte
        directional_bias = values.get('directional_bias')
        trend_strength = values.get('trend_strength')
        if directional_bias and str(directional_bias).upper() == 'BEARISH':
            if trend_strength and str(trend_strength).lower() in ['strong', 'very_strong', 'extreme']:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Tendance baissière forte ({trend_strength}) - éviter falling knife",
                    "metadata": {"strategy": self.name, "directional_bias": directional_bias, "trend_strength": trend_strength}
                }
                
        # Étape 1: Détecter spike baissier de prix
        price_spike_analysis = self._detect_price_spike_down(values)
        
        # Étape 2: Confirmer avec spike de volume
        volume_spike_analysis = self._detect_volume_spike(values)
        
        # Étape 3: Vérifier survente extrême
        oversold_analysis = self._detect_oversold_extreme(values)
        
        # Étape 4: Confirmer stabilisation
        stabilization_analysis = self._detect_stabilization(values)
        
        # Vérifier volume quality pour éviter bottom fishing de mauvaise qualité
        volume_quality_score = values.get('volume_quality_score')
        volume_quality_ok = True
        if volume_quality_score is not None:
            try:
                vol_quality = float(volume_quality_score)
                volume_quality_ok = vol_quality >= self.min_volume_quality
            except (ValueError, TypeError):
                pass
                
        # Signal BUY si toutes conditions remplies + volume spike obligatoire
        if (price_spike_analysis['is_price_spike'] and 
            volume_spike_analysis['is_volume_spike'] and  # OBLIGATOIRE
            oversold_analysis['is_oversold'] and
            stabilization_analysis['is_stabilized'] and
            volume_quality_ok):

            base_confidence = 0.45  # Base conservative pour stratégie spike
            confidence_boost = 0.0
            
            # Score cumulé des analyses avec pondération optimisée
            confidence_boost += price_spike_analysis['score'] * 0.25
            confidence_boost += volume_spike_analysis['score'] * 0.30  # Volume plus important
            confidence_boost += oversold_analysis['score'] * 0.35
            confidence_boost += stabilization_analysis['score'] * 0.25
            
            reason = f"Spike reaction BUY: "
            reason += f"{price_spike_analysis['indicators'][0]}"
            reason += f" + {volume_spike_analysis['indicators'][0]}"
            reason += f" + {oversold_analysis['indicators'][0]}"
            reason += f" + {stabilization_analysis['indicators'][0]}"
            
            # Anomaly detected (confirmation système) - BONUS FORT
            anomaly_detected = values.get('anomaly_detected')
            if anomaly_detected and str(anomaly_detected).lower() == 'true':
                confidence_boost += 0.20  # Bonus augmenté
                reason += " + ANOMALIE détectée"
                
            # Support proche (rebond probable) - CRITÈRE RENFORCÉ
            nearest_support = values.get('nearest_support')
            support_strength = values.get('support_strength')
            if nearest_support is not None:
                try:
                    current_price = None
                    if 'close' in self.data and self.data['close']:
                        try:
                            current_price = float(self.data['close'][-1])
                        except (IndexError, ValueError, TypeError):
                            pass
                        
                    if current_price is not None:
                        support_val = float(nearest_support)
                        distance_to_support = abs(current_price - support_val) / support_val
                        if distance_to_support <= self.support_proximity_threshold:
                            if support_strength and str(support_strength).lower() in ['strong', 'very_strong']:
                                confidence_boost += 0.20  # Bonus fort support solide
                                reason += f" + support FORT à {support_val:.2f}"
                            else:
                                confidence_boost += 0.12
                                reason += f" + près support {support_val:.2f}"
                except (ValueError, TypeError):
                    pass
                    
            # Signal strength système
            signal_strength = values.get('signal_strength')
            if signal_strength and str(signal_strength).upper() == 'STRONG':
                confidence_boost += 0.15
                reason += " + signal FORT"
            elif signal_strength and str(signal_strength).upper() == 'MODERATE':
                confidence_boost += 0.08
                reason += " + signal modéré"
                
            # Confluence score bonus
            if confluence_score is not None:
                try:
                    conf_val = float(confluence_score)
                    if conf_val >= 70:
                        confidence_boost += 0.18
                        reason += f" + confluence EXCELLENTE ({conf_val:.0f})"
                    elif conf_val >= 60:
                        confidence_boost += 0.12
                        reason += f" + confluence élevée ({conf_val:.0f})"
                    elif conf_val >= 50:
                        confidence_boost += 0.06
                        reason += f" + confluence ({conf_val:.0f})"
                except (ValueError, TypeError):
                    pass
                    
            # Volume quality bonus
            if volume_quality_score is not None:
                try:
                    vol_quality = float(volume_quality_score)
                    if vol_quality >= 70:
                        confidence_boost += 0.12
                        reason += " + volume HQ"
                    elif vol_quality >= 60:
                        confidence_boost += 0.06
                except (ValueError, TypeError):
                    pass
                    
            # Filtre final - seuil minimum pour éviter faux signaux
            raw_confidence = base_confidence * (1 + confidence_boost)
            if raw_confidence < 0.48:  # Seuil strict pour spike strategy
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Signal spike rejeté - confiance insuffisante ({raw_confidence:.2f} < 0.48)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "rejected_signal": "BUY",
                        "raw_confidence": raw_confidence
                    }
                }
                    
            confidence = self.calculate_confidence(base_confidence, 1.0 + confidence_boost)
            strength = self.get_strength_from_confidence(confidence)
            
            return {
                "side": "BUY",
                "confidence": confidence,
                "strength": strength,
                "reason": reason,
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "price_spike_score": price_spike_analysis['score'],
                    "volume_spike_score": volume_spike_analysis['score'],
                    "oversold_score": oversold_analysis['score'],
                    "stabilization_score": stabilization_analysis['score'],
                    "price_indicators": price_spike_analysis['indicators'],
                    "volume_indicators": volume_spike_analysis['indicators'],
                    "oversold_indicators": oversold_analysis['indicators'],
                    "stabilization_indicators": stabilization_analysis['indicators'],
                    "roc_10": values.get('roc_10'),
                    "rsi_14": values.get('rsi_14'),
                    "volume_spike_multiplier": values.get('volume_spike_multiplier'),
                    "momentum_score": values.get('momentum_score'),
                    "anomaly_detected": anomaly_detected,
                    "confluence_score": confluence_score,
                    "volume_quality_score": volume_quality_score
                }
            }
            
        # Diagnostic détaillé si pas de signal
        missing_conditions = []
        if not price_spike_analysis['is_price_spike']:
            missing_conditions.append(f"Pas de spike prix (score: {price_spike_analysis['score']:.2f}/0.4)")
        if not volume_spike_analysis['is_volume_spike']:
            missing_conditions.append(f"Pas de spike volume (score: {volume_spike_analysis['score']:.2f}/0.35)")
        if not oversold_analysis['is_oversold']:
            missing_conditions.append(f"Pas survente extrême (score: {oversold_analysis['score']:.2f}/0.5)")
        if not stabilization_analysis['is_stabilized']:
            missing_conditions.append(f"Pas stabilisé (score: {stabilization_analysis['score']:.2f}/0.45)")
        if not volume_quality_ok:
            missing_conditions.append(f"Volume quality insuffisant ({volume_quality_score:.1f} < {self.min_volume_quality})")
            
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": f"Conditions spike incomplètes: {'; '.join(missing_conditions)}",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "price_spike_score": price_spike_analysis['score'],
                "volume_spike_score": volume_spike_analysis['score'],
                "oversold_score": oversold_analysis['score'],
                "stabilization_score": stabilization_analysis['score'],
                "missing_conditions": missing_conditions
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        required_indicators = [
            'roc_10', 'rsi_14', 'volume_spike_multiplier', 
            'momentum_score', 'relative_volume', 'volume_quality_score'
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
                
        return True