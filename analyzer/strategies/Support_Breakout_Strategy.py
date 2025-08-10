"""
Support_Breakout_Strategy - Stratégie basée sur les cassures de support vers le bas.
Détecte les breakdowns de support pour signaler des opportunités de vente (breakdown baissier).
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class Support_Breakout_Strategy(BaseStrategy):
    """
    Stratégie détectant les cassures (breakouts) de niveaux de support vers le bas.
    
    Pattern de support breakout :
    1. Prix s'approche d'un niveau de support établi
    2. Volume augmente (pression vendeuse)
    3. Prix casse sous le support avec conviction
    4. Confirmation par momentum baissier
    5. Continuation baissière attendue (signal SELL)
    
    Note: Cette stratégie se concentre sur les breakdowns baissiers de support.
    Pour les breakouts haussiers de résistance, voir Resistance_Breakout_Strategy.
    
    Signaux générés:
    - SELL: Cassure confirmée de support + momentum baissier + volume
    - Pas de BUY (focus sur breakdowns baissiers)
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        
        # Paramètres de cassure support - AJUSTÉS
        self.breakdown_threshold = 0.005         # 0.5% sous support (moins strict)
        self.strong_breakdown_threshold = 0.012  # 1.2% pour cassure forte (moins strict)
        self.extreme_breakdown_threshold = 0.02  # 2% pour cassure extrême (moins strict)
        
        # Paramètres temporels
        self.max_time_near_support = 7          # Max barres près support avant cassure
        self.confirmation_bars = 1              # Barres pour confirmer cassure
        
        # Paramètres volume (confirmation cassure) - AJUSTÉS
        self.min_breakdown_volume = 1.3         # Volume 30% au-dessus normal (moins strict)
        self.strong_breakdown_volume = 2.0      # Volume 2x pour cassure forte (moins strict)
        self.volume_quality_threshold = 0.70    # Qualité volume minimum 70% (plus strict)
        
        # Paramètres momentum (continuation baissière) - OPTIMISÉS
        self.momentum_bearish_threshold = 35    # Momentum plus strict (35 au lieu de 40)
        self.roc_bearish_threshold = -0.015     # ROC plus strict (-1.5% au lieu de -1%)
        
        # Paramètres de support - OPTIMISÉS
        self.min_support_strength = 0.5         # Force minimum augmentée (50% au lieu de 40%)
        self.strong_support_bonus = 0.20        # Bonus augmenté (20% au lieu de 15%)
        self.min_break_probability = 0.40       # Probabilité minimum augmentée (40% au lieu de 30%)
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            # Support/Résistance (principal)
            'nearest_support': self.indicators.get('nearest_support'),
            'support_strength': self.indicators.get('support_strength'),
            'support_levels': self.indicators.get('support_levels'),
            'break_probability': self.indicators.get('break_probability'),
            'pivot_count': self.indicators.get('pivot_count'),
            
            # Bollinger Bands (support dynamique)
            'bb_lower': self.indicators.get('bb_lower'),
            'bb_position': self.indicators.get('bb_position'),
            'bb_width': self.indicators.get('bb_width'),
            'bb_breakout_direction': self.indicators.get('bb_breakout_direction'),
            
            # Momentum et ROC (continuation baissière)
            'momentum_score': self.indicators.get('momentum_score'),
            'momentum_10': self.indicators.get('momentum_10'),
            'roc_10': self.indicators.get('roc_10'),
            'roc_20': self.indicators.get('roc_20'),
            
            # Trend (confirme direction baissière)
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            
            # ADX (force de la cassure)
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            
            # Volume analysis (confirmation cassure)
            'volume_ratio': self.indicators.get('volume_ratio'),
            'relative_volume': self.indicators.get('relative_volume'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'volume_spike_multiplier': self.indicators.get('volume_spike_multiplier'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            
            # Volatilité (contexte cassure)
            'atr_14': self.indicators.get('atr_14'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            'atr_percentile': self.indicators.get('atr_percentile'),
            
            # Oscillateurs (survente potentielle après cassure)
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            'williams_r': self.indicators.get('williams_r'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            
            # Market regime et confluence
            'market_regime': self.indicators.get('market_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score'),
            'pattern_detected': self.indicators.get('pattern_detected'),
            'pattern_confidence': self.indicators.get('pattern_confidence')
        }
        
    def _detect_support_breakdown(self, values: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Détecte une cassure de support."""
        breakdown_score = 0.0
        breakdown_indicators = []
        
        # Vérification support principal
        nearest_support = values.get('nearest_support')
        if nearest_support is None:
            return {'is_breakdown': False, 'score': 0.0, 'indicators': []}
            
        try:
            support_level = float(nearest_support)
        except (ValueError, TypeError):
            return {'is_breakdown': False, 'score': 0.0, 'indicators': []}
            
        # Vérifier si le prix a cassé sous le support
        if current_price >= support_level:
            return {
                'is_breakdown': False,
                'score': 0.0,
                'indicators': ['Prix encore au-dessus support'],
                'support_level': support_level
            }
            
        # Distance de cassure (plus c'est loin, plus c'est significatif)
        breakdown_distance = (support_level - current_price) / support_level
        
        if breakdown_distance >= self.extreme_breakdown_threshold:
            breakdown_score += 0.4
            breakdown_indicators.append(f"Cassure extrême ({breakdown_distance*100:.1f}%)")
        elif breakdown_distance >= self.strong_breakdown_threshold:
            breakdown_score += 0.3
            breakdown_indicators.append(f"Cassure forte ({breakdown_distance*100:.1f}%)")
        elif breakdown_distance >= self.breakdown_threshold:
            breakdown_score += 0.2
            breakdown_indicators.append(f"Cassure confirmée ({breakdown_distance*100:.1f}%)")
        else:
            # Cassure trop faible
            return {
                'is_breakdown': False,
                'score': 0.0,
                'indicators': [f'Cassure insuffisante ({breakdown_distance*100:.2f}%)'],
                'support_level': support_level
            }
            
        # Force du support cassé (plus fort = cassure plus significative)
        support_strength = values.get('support_strength')
        if support_strength is not None:
            try:
                if isinstance(support_strength, str):
                    strength_map = {'WEAK': 0.2, 'MODERATE': 0.5, 'STRONG': 0.8, 'MAJOR': 1.0}
                    strength_val = strength_map.get(support_strength.upper(), 0.5)
                else:
                    strength_val = float(support_strength)
                    
                if strength_val >= 0.8:
                    breakdown_score += self.strong_support_bonus * 2
                    breakdown_indicators.append(f"Support très fort cassé ({strength_val:.2f})")
                elif strength_val >= self.min_support_strength:
                    breakdown_score += self.strong_support_bonus
                    breakdown_indicators.append(f"Support fort cassé ({strength_val:.2f})")
            except (ValueError, TypeError):
                pass
                
        # Probabilité de cassure (format décimal 0-1 depuis DB)
        break_probability = values.get('break_probability')
        if break_probability is not None:
            try:
                break_prob = float(break_probability)
                if break_prob >= 0.70:  # Haute probabilité (70%)
                    breakdown_score += 0.15
                    breakdown_indicators.append(f"Haute probabilité cassure ({break_prob*100:.0f}%)")
                elif break_prob >= self.min_break_probability:  # >= 0.30 (30%)
                    breakdown_score += 0.1
                    breakdown_indicators.append(f"Probabilité cassure modérée ({break_prob*100:.0f}%)")
            except (ValueError, TypeError):
                pass
                
        # Bollinger Band breakdown (support dynamique)
        bb_lower = values.get('bb_lower')
        bb_breakout_direction = values.get('bb_breakout_direction')
        
        if bb_lower is not None:
            try:
                bb_lower_val = float(bb_lower)
                if current_price < bb_lower_val:
                    breakdown_score += 0.1
                    breakdown_indicators.append("Cassure Bollinger basse")
                    
                if bb_breakout_direction == 'DOWN':
                    breakdown_score += 0.1
                    breakdown_indicators.append("BB breakout baissier")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_breakdown': breakdown_score >= 0.4,
            'score': breakdown_score,
            'indicators': breakdown_indicators,
            'support_level': support_level,
            'breakdown_distance_pct': breakdown_distance * 100
        }
        
    def _detect_bearish_momentum(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte le momentum baissier pour continuation."""
        momentum_score = 0
        momentum_indicators = []
        
        # Momentum score général (format 0-100, 50=neutre)
        momentum_val = values.get('momentum_score')
        if momentum_val is not None:
            try:
                momentum_float = float(momentum_val)
                if momentum_float <= self.momentum_bearish_threshold:  # <=40
                    momentum_score += 25
                    momentum_indicators.append(f"Momentum baissier ({momentum_float:.1f})")
            except (ValueError, TypeError):
                pass
                
        # ROC négatif (continuation baissière)
        roc_10 = values.get('roc_10')
        if roc_10 is not None:
            try:
                roc_val = float(roc_10)
                if roc_val <= self.roc_bearish_threshold:  # <=-0.01
                    momentum_score += 20
                    momentum_indicators.append(f"ROC baissier ({roc_val*100:.1f}%)")
            except (ValueError, TypeError):
                pass
                
        # Directional bias baissier
        directional_bias = values.get('directional_bias')
        if directional_bias and directional_bias.upper() == 'BEARISH':
            momentum_score += 15
            momentum_indicators.append("Bias directionnel baissier")
            
        # ADX DI confirmation
        plus_di = values.get('plus_di')
        minus_di = values.get('minus_di')
        
        if plus_di is not None and minus_di is not None:
            try:
                plus_di_val = float(plus_di)
                minus_di_val = float(minus_di)
                
                if minus_di_val > plus_di_val:  # DI- > DI+ = pression baissière
                    momentum_score += 15
                    momentum_indicators.append(f"DI- > DI+ ({minus_di_val:.1f} > {plus_di_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # Trend alignment baissier (format décimal)
        trend_alignment = values.get('trend_alignment')
        if trend_alignment is not None:
            try:
                trend_align = float(trend_alignment)
                if trend_align < -0.5:  # Alignment baissier fort (format décimal)
                    momentum_score += 10
                    momentum_indicators.append(f"Trend alignment baissier ({trend_align:.2f})")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_bearish_momentum': momentum_score >= 30,
            'score': momentum_score,
            'indicators': momentum_indicators
        }
        
    def _detect_volume_confirmation(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte confirmation volume pour la cassure."""
        volume_score = 0.0
        volume_indicators = []
        
        # Volume ratio principal
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.strong_breakdown_volume:
                    volume_score += 0.3
                    volume_indicators.append(f"Volume fort ({vol_ratio:.1f}x)")
                elif vol_ratio >= self.min_breakdown_volume:
                    volume_score += 0.2
                    volume_indicators.append(f"Volume élevé ({vol_ratio:.1f}x)")
            except (ValueError, TypeError):
                pass
                
        # Volume quality score
        volume_quality = values.get('volume_quality_score')
        if volume_quality is not None:
            try:
                volume_quality_score = float(volume_quality)
                if volume_quality_score >= 80:
                    volume_score += 0.2
                    volume_indicators.append(f"Volume qualité élevée ({volume_quality_score:.0f})")
                elif volume_quality_score >= 60:  # Ajuster le seuil pour format 0-100
                    volume_score += 0.1
                    volume_indicators.append(f"Volume qualité correcte ({volume_quality_score:.0f})")
            except (ValueError, TypeError):
                pass
                
        # Trade intensity
        trade_intensity = values.get('trade_intensity')
        if trade_intensity is not None:
            try:
                intensity = float(trade_intensity)
                if intensity >= 1.5:
                    volume_score += 0.15
                    volume_indicators.append(f"Intensité élevée ({intensity:.1f}x)")
            except (ValueError, TypeError):
                pass
                
        # Volume spike multiplier
        volume_spike = values.get('volume_spike_multiplier')
        if volume_spike is not None:
            try:
                spike_mult = float(volume_spike)
                if spike_mult >= 2.0:
                    volume_score += 0.1
                    volume_indicators.append(f"Volume spike ({spike_mult:.1f}x)")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_volume_confirmed': volume_score >= 0.25,
            'score': volume_score,
            'indicators': volume_indicators
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les cassures de support.
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
            
        # Étape 1: Détecter cassure de support
        breakdown_analysis = self._detect_support_breakdown(values, current_price)
        
        if not breakdown_analysis['is_breakdown']:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Pas de cassure support détectée: {', '.join(breakdown_analysis['indicators'][:2]) if breakdown_analysis['indicators'] else 'Support intact'}",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "current_price": current_price,
                    "breakdown_score": breakdown_analysis['score']
                }
            }
            
        # Étape 2: Confirmer momentum baissier
        momentum_analysis = self._detect_bearish_momentum(values)
        
        # Étape 3: Confirmer avec volume
        volume_analysis = self._detect_volume_confirmation(values)
        
        # Signal SELL si cassure + momentum baissier (volume optionnel mais bonifie)
        if breakdown_analysis['is_breakdown'] and momentum_analysis['is_bearish_momentum']:
            
            base_confidence = 0.40  # RÉDUIT - breakouts = plus risqués
            confidence_boost = 0.0
            
            # Score de cassure
            confidence_boost += breakdown_analysis['score'] * 0.4
            
            # Score momentum baissier
            confidence_boost += momentum_analysis['score'] * 0.4
            
            reason = f"Cassure support {breakdown_analysis['support_level']:.2f} ({breakdown_analysis['breakdown_distance_pct']:.1f}%)"
            reason += f" + {', '.join(momentum_analysis['indicators'][:1])}"
            
            # Bonus volume (pas obligatoire mais renforce)
            if volume_analysis['is_volume_confirmed']:
                confidence_boost += volume_analysis['score'] * 0.2
                reason += f" + {volume_analysis['indicators'][0]}"
                
            # Market regime breakout
            market_regime = values.get('market_regime')
            if market_regime and 'BREAKOUT_BEAR' in str(market_regime):
                confidence_boost += 0.1
                reason += " + regime breakout baissier"
                
            # Pattern detected
            pattern_detected = values.get('pattern_detected')
            if pattern_detected and 'breakout' in str(pattern_detected).lower():
                confidence_boost += 0.1
                reason += " + pattern breakout"
                
            # Confluence score
            confluence_score = values.get('confluence_score')
            if confluence_score is not None:
                try:
                    conf_val = float(confluence_score)
                    if conf_val > 70:
                        confidence_boost += 0.1
                        reason += " + haute confluence"
                except (ValueError, TypeError):
                    pass
                    
            confidence = self.calculate_confidence(base_confidence, 1.0 + confidence_boost)
            strength = self.get_strength_from_confidence(confidence)
            
            return {
                "side": "SELL",
                "confidence": confidence,
                "strength": strength,
                "reason": reason,
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "current_price": current_price,
                    "support_level": breakdown_analysis['support_level'],
                    "breakdown_distance_pct": breakdown_analysis['breakdown_distance_pct'],
                    "breakdown_score": breakdown_analysis['score'],
                    "momentum_score": momentum_analysis['score'],
                    "volume_score": volume_analysis['score'],
                    "breakdown_indicators": breakdown_analysis['indicators'],
                    "momentum_indicators": momentum_analysis['indicators'],
                    "volume_indicators": volume_analysis['indicators'],
                    "market_regime": market_regime,
                    "confluence_score": confluence_score
                }
            }
            
        # Diagnostic si conditions incomplètes
        missing_conditions = []
        if not breakdown_analysis['is_breakdown']:
            missing_conditions.append(f"Cassure insuffisante (score: {breakdown_analysis['score']:.2f})")
        if not momentum_analysis['is_bearish_momentum']:
            missing_conditions.append(f"Momentum pas assez baissier (score: {momentum_analysis['score']:.2f})")
            
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": f"Conditions incomplètes: {'; '.join(missing_conditions)}",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "current_price": current_price,
                "breakdown_score": breakdown_analysis['score'],
                "momentum_score": momentum_analysis['score'],
                "missing_conditions": missing_conditions
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        required_indicators = [
            'nearest_support', 'momentum_score', 'volume_ratio'
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
            logger.warning(f"{self.name}: Données OHLCV manquantes")
            return False
            
        return True
