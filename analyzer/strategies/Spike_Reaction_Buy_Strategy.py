"""
Spike_Reaction_Buy_Strategy - Stratégie basée sur la réaction après spike baissier.
Détecte les spikes de vente (crash/dump) suivis d'une stabilisation pour signaler des achats opportunistes.
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
        
        # Paramètres de détection spike baissier
        self.min_price_drop = -0.02              # 2% chute minimum
        self.severe_price_drop = -0.04           # 4% chute sévère  
        self.extreme_price_drop = -0.08          # 8% chute extrême
        
        # Paramètres volume (confirmation spike)
        self.min_spike_volume = 2.0              # Volume 2x normal minimum
        self.strong_spike_volume = 3.5           # Volume 3.5x pour spike fort
        self.extreme_spike_volume = 5.0          # Volume 5x pour spike extrême
        
        # Paramètres RSI (survente extrême)
        self.oversold_rsi_threshold = 30         # RSI survente
        self.extreme_oversold_threshold = 20     # RSI survente extrême
        self.williams_r_oversold = -80           # Williams %R survente
        
        # Paramètres stabilisation
        self.stabilization_bars = 3              # Barres pour confirmer stabilisation
        self.max_continued_drop = -0.005         # Chute max après spike (0.5%)
        self.min_volatility_ratio = 0.3          # Volatilité réduite après spike
        
        # Paramètres momentum reversal
        self.momentum_reversal_threshold = 0.1   # Momentum redevient positif
        self.min_roc_improvement = -1.0          # ROC amélioration minimum
        
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
        """Détecte un spike baissier de prix."""
        spike_score = 0.0
        spike_indicators = []
        
        # ROC 10 périodes (mouvement récent)
        roc_10 = values.get('roc_10')
        if roc_10 is not None:
            try:
                roc_val = float(roc_10)
                if roc_val <= self.extreme_price_drop * 100:  # ROC en %
                    spike_score += 0.4
                    spike_indicators.append(f"ROC10 chute extrême ({roc_val:.1f}%)")
                elif roc_val <= self.severe_price_drop * 100:
                    spike_score += 0.3
                    spike_indicators.append(f"ROC10 chute sévère ({roc_val:.1f}%)")
                elif roc_val <= self.min_price_drop * 100:
                    spike_score += 0.2
                    spike_indicators.append(f"ROC10 chute ({roc_val:.1f}%)")
            except (ValueError, TypeError):
                pass
                
        # ROC 20 périodes (confirmation tendance)
        roc_20 = values.get('roc_20')
        if roc_20 is not None:
            try:
                roc_20_val = float(roc_20)
                if roc_20_val <= self.min_price_drop * 100:
                    spike_score += 0.1
                    spike_indicators.append(f"ROC20 confirme chute ({roc_20_val:.1f}%)")
            except (ValueError, TypeError):
                pass
                
        # Momentum négatif fort
        momentum_10 = values.get('momentum_10')
        if momentum_10 is not None:
            try:
                momentum_val = float(momentum_10)
                if momentum_val < 0:  # Momentum négatif
                    spike_score += 0.15
                    spike_indicators.append(f"Momentum négatif ({momentum_val:.2f})")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_price_spike': spike_score >= 0.3,
            'score': spike_score,
            'indicators': spike_indicators
        }
        
    def _detect_volume_spike(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte un spike de volume (confirmation panique)."""
        volume_score = 0.0
        volume_indicators = []
        
        # Volume spike multiplier (principal)
        volume_spike_mult = values.get('volume_spike_multiplier')
        if volume_spike_mult is not None:
            try:
                spike_mult = float(volume_spike_mult)
                if spike_mult >= self.extreme_spike_volume:
                    volume_score += 0.4
                    volume_indicators.append(f"Volume spike extrême ({spike_mult:.1f}x)")
                elif spike_mult >= self.strong_spike_volume:
                    volume_score += 0.3
                    volume_indicators.append(f"Volume spike fort ({spike_mult:.1f}x)")
                elif spike_mult >= self.min_spike_volume:
                    volume_score += 0.2
                    volume_indicators.append(f"Volume spike ({spike_mult:.1f}x)")
            except (ValueError, TypeError):
                pass
                
        # Volume relatif (confirmation)
        relative_volume = values.get('relative_volume')
        if relative_volume is not None:
            try:
                rel_vol = float(relative_volume)
                if rel_vol >= 3.0:
                    volume_score += 0.2
                    volume_indicators.append(f"Volume relatif élevé ({rel_vol:.1f}x)")
            except (ValueError, TypeError):
                pass
                
        # Trade intensity (activité anormale)
        trade_intensity = values.get('trade_intensity')
        if trade_intensity is not None:
            try:
                intensity = float(trade_intensity)
                if intensity >= 2.0:
                    volume_score += 0.15
                    volume_indicators.append(f"Intensité élevée ({intensity:.1f}x)")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_volume_spike': volume_score >= 0.25,
            'score': volume_score,
            'indicators': volume_indicators
        }
        
    def _detect_oversold_extreme(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte survente extrême (opportunité rebond)."""
        oversold_score = 0.0
        oversold_indicators = []
        
        # RSI en survente
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi_val = float(rsi_14)
                if rsi_val <= self.extreme_oversold_threshold:
                    oversold_score += 0.35
                    oversold_indicators.append(f"RSI survente extrême ({rsi_val:.1f})")
                elif rsi_val <= self.oversold_rsi_threshold:
                    oversold_score += 0.25
                    oversold_indicators.append(f"RSI survente ({rsi_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # Williams %R confirme survente
        williams_r = values.get('williams_r')
        if williams_r is not None:
            try:
                wr_val = float(williams_r)
                if wr_val <= self.williams_r_oversold:
                    oversold_score += 0.2
                    oversold_indicators.append(f"Williams%R survente ({wr_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # Stochastic en survente
        stoch_k = values.get('stoch_k')
        stoch_d = values.get('stoch_d')
        if stoch_k is not None and stoch_d is not None:
            try:
                k_val = float(stoch_k)
                d_val = float(stoch_d)
                if k_val <= 20 and d_val <= 20:
                    oversold_score += 0.2
                    oversold_indicators.append(f"Stoch survente (K={k_val:.1f}, D={d_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # CCI extreme
        cci_20 = values.get('cci_20')
        if cci_20 is not None:
            try:
                cci_val = float(cci_20)
                if cci_val <= -100:  # CCI survente extrême
                    oversold_score += 0.15
                    oversold_indicators.append(f"CCI survente ({cci_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # Bollinger position basse (près de BB lower)
        bb_position = values.get('bb_position')
        if bb_position is not None:
            try:
                bb_pos = float(bb_position)
                if bb_pos <= 0.1:  # Très proche BB lower
                    oversold_score += 0.1
                    oversold_indicators.append(f"BB position basse ({bb_pos:.2f})")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_oversold': oversold_score >= 0.4,
            'score': oversold_score,
            'indicators': oversold_indicators
        }
        
    def _detect_stabilization(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte la stabilisation après spike (fin de panique)."""
        stabilization_score = 0.0
        stabilization_indicators = []
        
        # Momentum score amélioration
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum_val = float(momentum_score)
                if momentum_val >= self.momentum_reversal_threshold:
                    stabilization_score += 0.3
                    stabilization_indicators.append(f"Momentum reversal ({momentum_val:.2f})")
                elif momentum_val >= 0:  # Au moins neutre
                    stabilization_score += 0.15
                    stabilization_indicators.append(f"Momentum neutre ({momentum_val:.2f})")
            except (ValueError, TypeError):
                pass
                
        # Volume pattern (décroissance après spike)
        volume_pattern = values.get('volume_pattern')
        if volume_pattern == 'decreasing' or volume_pattern == 'normal':
            stabilization_score += 0.2
            stabilization_indicators.append(f"Volume pattern: {volume_pattern}")
            
        # ROC amélioration (moins négatif)
        roc_10 = values.get('roc_10')
        if roc_10 is not None:
            try:
                roc_val = float(roc_10)
                if roc_val >= self.min_roc_improvement:  # ROC > -1%
                    stabilization_score += 0.2
                    stabilization_indicators.append(f"ROC amélioration ({roc_val:.1f}%)")
            except (ValueError, TypeError):
                pass
                
        # Pattern detected (reversal pattern)
        pattern_detected = values.get('pattern_detected')
        if pattern_detected and 'reversal' in str(pattern_detected).lower():
            stabilization_score += 0.2
            stabilization_indicators.append(f"Pattern reversal détecté")
            
        return {
            'is_stabilized': stabilization_score >= 0.3,
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
        
        # Étape 1: Détecter spike baissier de prix
        price_spike_analysis = self._detect_price_spike_down(values)
        
        # Étape 2: Confirmer avec spike de volume
        volume_spike_analysis = self._detect_volume_spike(values)
        
        # Étape 3: Vérifier survente extrême
        oversold_analysis = self._detect_oversold_extreme(values)
        
        # Étape 4: Confirmer stabilisation
        stabilization_analysis = self._detect_stabilization(values)
        
        # Signal BUY si: spike détecté + survente + stabilisation
        if (price_spike_analysis['is_price_spike'] and 
            oversold_analysis['is_oversold'] and
            stabilization_analysis['is_stabilized']):
            
            base_confidence = 0.4
            confidence_boost = 0.0
            
            # Score cumulé des analyses
            confidence_boost += price_spike_analysis['score'] * 0.3
            confidence_boost += oversold_analysis['score'] * 0.4
            confidence_boost += stabilization_analysis['score'] * 0.3
            
            reason = f"Réaction post-spike: "
            reason += f"{', '.join(price_spike_analysis['indicators'][:1])}"
            reason += f" + {', '.join(oversold_analysis['indicators'][:1])}"
            reason += f" + {', '.join(stabilization_analysis['indicators'][:1])}"
            
            # Bonus volume spike (confirmation panique)
            if volume_spike_analysis['is_volume_spike']:
                confidence_boost += volume_spike_analysis['score'] * 0.2
                reason += f" + {volume_spike_analysis['indicators'][0]}"
                
            # Anomaly detected (confirmation système)
            anomaly_detected = values.get('anomaly_detected')
            if anomaly_detected and str(anomaly_detected).lower() == 'true':
                confidence_boost += 0.15
                reason += " + anomalie détectée"
                
            # Support proche (rebond probable)
            nearest_support = values.get('nearest_support')
            if nearest_support is not None:
                try:
                    # Récupérer prix actuel depuis OHLCV si disponible
                    current_price = None
                    if 'ohlcv' in self.data and self.data['ohlcv']:
                        current_price = float(self.data['ohlcv'][-1]['close'])
                        
                    if current_price is not None:
                        support_val = float(nearest_support)
                        distance_to_support = abs(current_price - support_val) / support_val
                        if distance_to_support <= 0.01:  # 1% du support
                            confidence_boost += 0.1
                            reason += f" + près support ({support_val:.2f})"
                except (ValueError, TypeError):
                    pass
                    
            # Confluence score
            confluence_score = values.get('confluence_score')
            if confluence_score is not None:
                try:
                    conf_val = float(confluence_score)
                    if conf_val > 0.6:
                        confidence_boost += 0.1
                        reason += " + haute confluence"
                except (ValueError, TypeError):
                    pass
                    
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
                    "confluence_score": confluence_score
                }
            }
            
        # Diagnostic détaillé si pas de signal
        missing_conditions = []
        if not price_spike_analysis['is_price_spike']:
            missing_conditions.append(f"Pas de spike prix (score: {price_spike_analysis['score']:.2f})")
        if not oversold_analysis['is_oversold']:
            missing_conditions.append(f"Pas survente extrême (score: {oversold_analysis['score']:.2f})")
        if not stabilization_analysis['is_stabilized']:
            missing_conditions.append(f"Pas stabilisé (score: {stabilization_analysis['score']:.2f})")
            
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": f"Conditions incomplètes: {'; '.join(missing_conditions)}",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "price_spike_score": price_spike_analysis['score'],
                "oversold_score": oversold_analysis['score'],
                "stabilization_score": stabilization_analysis['score'],
                "missing_conditions": missing_conditions
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        required_indicators = [
            'roc_10', 'rsi_14', 'volume_spike_multiplier', 
            'momentum_score', 'relative_volume'
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
