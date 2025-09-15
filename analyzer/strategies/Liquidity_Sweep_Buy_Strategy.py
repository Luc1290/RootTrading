"""
Liquidity_Sweep_Buy_Strategy - Stratégie basée sur les liquidity sweeps haussiers.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class Liquidity_Sweep_Buy_Strategy(BaseStrategy):
    """
    Stratégie de liquidity sweep haussier - détecte les faux breakdowns suivis de retournements.
    
    Un liquidity sweep haussier se produit quand :
    1. Prix casse temporairement un support (sweep de liquidité baissière)
    2. Puis retourne rapidement au-dessus du support (faux breakdown)
    3. Continuation haussière attendue
    
    Signaux générés:
    - BUY: Après sweep de support + retour au-dessus + confirmations haussières
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres liquidity sweep - ASSOUPLIS pour plus de signaux
        self.sweep_threshold = 0.006      # 0.6% sous support pour sweep (réaliste crypto)
        self.recovery_threshold = 0.0    # Juste besoin de repasser > support
        self.max_sweep_duration = 4      # Max barres sous support (chasse rapide)
        self.min_volume_spike = 1.5      # Volume 50% au-dessus moyenne
        self.support_strength_min = 0.5  # Force minimum du support (MODERATE+)
        
    def _convert_support_strength_to_score(self, strength_str: str) -> float:
        """Convertit support_strength string en score numérique."""
        strength_map = {
            'WEAK': 0.2,
            'MODERATE': 0.5, 
            'STRONG': 0.8,
            'MAJOR': 1.0
        }
        return strength_map.get(str(strength_str).upper(), 0.3)
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs."""
        return {
            # Support/Résistance (clé pour liquidity sweep)
            'nearest_support': self.indicators.get('nearest_support'),
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            'support_strength': self.indicators.get('support_strength'),
            'resistance_strength': self.indicators.get('resistance_strength'),
            'break_probability': self.indicators.get('break_probability'),
            'pivot_count': self.indicators.get('pivot_count'),
            # Volume (crucial pour détecter les sweeps)
            'volume_ratio': self.indicators.get('volume_ratio'),
            'relative_volume': self.indicators.get('relative_volume'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'volume_spike_multiplier': self.indicators.get('volume_spike_multiplier'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            # Momentum et retournement
            'momentum_score': self.indicators.get('momentum_score'),
            'rsi_14': self.indicators.get('rsi_14'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            'williams_r': self.indicators.get('williams_r'),
            # MACD pour confirmation momentum
            'macd_line': self.indicators.get('macd_line'),
            'macd_signal': self.indicators.get('macd_signal'),
            'macd_histogram': self.indicators.get('macd_histogram'),
            # Contexte de marché
            'market_regime': self.indicators.get('market_regime'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            'directional_bias': self.indicators.get('directional_bias'),
            # Pattern recognition
            'pattern_detected': self.indicators.get('pattern_detected'),
            'pattern_confidence': self.indicators.get('pattern_confidence'),
            # Confluence
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score')
        }
        
    def _get_price_volume_data(self) -> Dict[str, Optional[float]]:
        """Récupère les données de prix et volume pour analyse sweep (7 barres)."""
        try:
            if self.data and all(key in self.data for key in ['close', 'low', 'high', 'volume']) and \
               all(len(self.data[key]) >= 8 for key in ['close', 'low', 'high', 'volume']):  # 8 pour avoir 7 barres précédentes
                
                return {
                    'current_price': float(self.data['close'][-1]),
                    'current_low': float(self.data['low'][-1]),
                    'current_high': float(self.data['high'][-1]),
                    'current_volume': float(self.data['volume'][-1]),
                    'prev_low_1': float(self.data['low'][-2]),
                    'prev_low_2': float(self.data['low'][-3]),
                    'prev_low_3': float(self.data['low'][-4]),
                    'prev_low_4': float(self.data['low'][-5]),
                    'prev_low_5': float(self.data['low'][-6]),
                    'prev_low_6': float(self.data['low'][-7]),
                    'prev_low_7': float(self.data['low'][-8]),
                    'prev_volume_1': float(self.data['volume'][-2]),
                    'prev_volume_2': float(self.data['volume'][-3]),
                    'prev_close_1': float(self.data['close'][-2]),
                    'prev_close_2': float(self.data['close'][-3])
                }
        except (IndexError, ValueError, TypeError):
            pass
        return {k: None for k in ['current_price', 'current_low', 'current_high', 'current_volume',
                                 'prev_low_1', 'prev_low_2', 'prev_low_3', 'prev_low_4', 'prev_low_5', 
                                 'prev_low_6', 'prev_low_7', 'prev_volume_1', 'prev_volume_2', 'prev_close_1', 'prev_close_2']}
        
    def _detect_liquidity_sweep_setup(self, price_data: Dict[str, Optional[float]], 
                                     support_level: float) -> Dict[str, Any]:
        """
        Détecte si on a un setup de liquidity sweep haussier.
        
        Returns:
            Dict avec 'is_sweep', 'sweep_type', 'sweep_strength', etc.
        """
        current_price = price_data['current_price']
        current_low = price_data['current_low']
        prev_low_1 = price_data['prev_low_1']
        prev_low_2 = price_data['prev_low_2']
        
        if any(v is None for v in [current_price, current_low, prev_low_1, prev_low_2]):
            return {'is_sweep': False, 'reason': 'Données prix incomplètes'}
            
        # Assertions pour mypy - on sait que les valeurs ne sont pas None
        assert current_price is not None
        assert current_low is not None
        assert prev_low_1 is not None
        assert prev_low_2 is not None
            
        # Détection du sweep : prix a cassé sous support récemment (7 barres)
        if support_level <= 0:
            return {'is_sweep': False, 'reason': 'Support level invalide pour calcul sweep'}
        sweep_distance = (support_level - current_low) / support_level
        recent_sweep = False
        sweep_bars_ago = 0
        
        # Vérifier si on a cassé sous support dans les 7 dernières barres
        lows = [current_low, prev_low_1, prev_low_2, 
                price_data.get('prev_low_3'), price_data.get('prev_low_4'),
                price_data.get('prev_low_5'), price_data.get('prev_low_6'), 
                price_data.get('prev_low_7')]
        for i, low in enumerate(lows):
            if low is not None and low < support_level * (1 - self.sweep_threshold):
                recent_sweep = True
                sweep_bars_ago = i
                break
                
        # Implémenter contrôle durée max sweep
        if recent_sweep and sweep_bars_ago > self.max_sweep_duration:
            return {'is_sweep': False, 'reason': f'Sweep trop ancien ({sweep_bars_ago} barres > {self.max_sweep_duration})'}
                
        if not recent_sweep:
            return {'is_sweep': False, 'reason': 'Pas de sweep récent détecté'}
            
        # Vérification recovery : prix est revenu au-dessus du support (ASSOUPLI)
        recovery_successful = current_price is not None and current_price > support_level * (1 + self.recovery_threshold)
        
        if not recovery_successful:
            return {'is_sweep': False, 'reason': 'Pas encore de recovery au-dessus support'}
            
        # Calcul de la force du sweep avec protection contre séquence vide
        sweep_lows = [low for low in lows if low is not None and low < support_level]
        if not sweep_lows:
            max_sweep_distance = 0.0  # Pas de sweep réel
        else:
            max_sweep_distance = max(abs(low - support_level) / support_level for low in sweep_lows)
        
        return {
            'is_sweep': True,
            'sweep_bars_ago': sweep_bars_ago,
            'max_sweep_distance': max_sweep_distance,
            'recovery_distance': (current_price - support_level) / support_level if current_price is not None else 0.0,
            'reason': f'Liquidity sweep détecté il y a {sweep_bars_ago} barres'
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les liquidity sweeps haussiers.
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
        price_data = self._get_price_volume_data()
        
        # Vérification support level - AVEC FALLBACK
        try:
            nearest_support = float(values['nearest_support']) if values['nearest_support'] is not None else None
            support_strength_raw = values['support_strength']
            
            # Fallback support si nearest_support manque
            if nearest_support is None or nearest_support == 0:  # Traiter 0 comme invalide
                # Utiliser le plus bas récent comme support dynamique (8 barres minimum)
                lows = self.data.get('low', [])
                if lows and len(lows) >= 8:  # Cohérent avec validation
                    try:
                        # Fallback support amélioré : utiliser percentile 10% au lieu du minimum
                        lookback = min(15, len(lows))
                        recent_lows = [float(low) for low in lows[-lookback:]]
                        import numpy as np
                        # Percentile 10% pour éviter les mèches extrêmes
                        nearest_support = float(np.percentile(recent_lows, 10))
                        support_strength_raw = 'WEAK'  # Plus réaliste pour fallback
                    except (ValueError, TypeError):
                        return {
                            "side": None,
                            "confidence": 0.0,
                            "strength": "weak",
                            "reason": "Erreur calcul fallback support depuis lows",
                            "metadata": {"strategy": self.name}
                        }
                else:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": "Pas de support disponible (ni fixe ni fallback) - données insuffisantes",
                        "metadata": {"strategy": self.name}
                    }
            
            # support_strength est en format string : WEAK/MODERATE/STRONG/MAJOR
            support_strength_score = self._convert_support_strength_to_score(support_strength_raw) if support_strength_raw is not None else 0.3
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion support: {e}",
                "metadata": {"strategy": self.name}
            }
            
        current_price = price_data['current_price']
        
        if nearest_support is None or current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Niveau support ou prix non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # Vérification force du support
        if support_strength_score is not None and support_strength_score < self.support_strength_min:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Support trop faible ({support_strength_raw}) pour liquidity sweep",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "support_strength": support_strength_raw,
                    "support_strength_score": support_strength_score,
                    "nearest_support": nearest_support
                }
            }
            
        # Détection du liquidity sweep setup
        sweep_analysis = self._detect_liquidity_sweep_setup(price_data, nearest_support)
        
        if not sweep_analysis['is_sweep']:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Pas de setup liquidity sweep: {sweep_analysis['reason']}",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "nearest_support": nearest_support,
                    "current_price": current_price
                }
            }
            
        # Si on a un sweep setup, générer signal BUY
        signal_side = "BUY"
        reason = f"Liquidity sweep haussier sur support {nearest_support:.2f}"
        base_confidence = 0.65  # Base réduite pour équilibrage avec autres stratégies
        confidence_boost = 0.0
        
        # Bonus selon la qualité du sweep
        sweep_distance = sweep_analysis['max_sweep_distance']
        recovery_distance = sweep_analysis['recovery_distance']
        
        if sweep_distance >= 0.01:  # Sweep > 1%
            confidence_boost += 0.15
            reason += f" - sweep profond ({sweep_distance*100:.1f}%)"
        elif sweep_distance >= 0.005:  # Sweep > 0.5%
            confidence_boost += 0.10
            reason += f" - sweep modéré ({sweep_distance*100:.1f}%)"
        else:
            confidence_boost += 0.05
            reason += f" - sweep léger ({sweep_distance*100:.1f}%)"
            
        # Bonus pour recovery forte
        if recovery_distance >= 0.01:
            confidence_boost += 0.10
            reason += f" + recovery forte (+{recovery_distance*100:.1f}%)"
        elif recovery_distance >= 0.005:
            confidence_boost += 0.08
            reason += f" + recovery modérée (+{recovery_distance*100:.1f}%)"
            
        # Confirmation volume STRICTE (crucial pour liquidity sweep)
        volume_ratio = values.get('volume_ratio')
        volume_spike_multiplier = values.get('volume_spike_multiplier')
        
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio < 0.7:  # Volume trop faible = rejet immédiat (assoupli)
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet liquidity sweep: volume trop faible ({vol_ratio:.1f}x < 0.7x)",
                        "metadata": {"strategy": self.name, "volume_ratio": vol_ratio}
                    }
                elif 0.9 <= vol_ratio < 1.2:  # Volume correct - petit bonus
                    confidence_boost += 0.03
                    reason += f" + volume correct ({vol_ratio:.1f}x)"
                elif vol_ratio >= 2.0:  # Volume exceptionnel
                    confidence_boost += 0.20
                    reason += f" + volume EXCEPTIONNEL ({vol_ratio:.1f}x)"
                elif vol_ratio >= self.min_volume_spike:  # Volume élevé (1.5x)
                    confidence_boost += 0.15
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                else:  # Volume correct mais pas exceptionnel
                    confidence_boost += 0.05
                    reason += f" + volume correct ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # Volume spike multiplier
        if volume_spike_multiplier is not None:
            try:
                spike_mult = float(volume_spike_multiplier)
                if spike_mult >= 3.0:
                    confidence_boost += 0.15
                    reason += f" + spike volume ({spike_mult:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # Confirmation oscillateurs avec rejets contradictoires
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if rsi > 70:  # RSI trop haut = rejet BUY sweep (assoupli)
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet liquidity sweep: RSI trop haut ({rsi:.1f}) pour BUY",
                        "metadata": {"strategy": self.name, "rsi": rsi}
                    }
                elif 40 <= rsi <= 65:  # RSI neutre - pas de bonus ni malus
                    pass  # Zone neutre
                elif rsi <= 30:  # Survente extrême
                    confidence_boost += 0.15
                    reason += f" + RSI survente ({rsi:.1f})"
                elif rsi <= 40:  # Survente modérée
                    confidence_boost += 0.10
                    reason += f" + RSI favorable ({rsi:.1f})"
            except (ValueError, TypeError):
                pass
                
        # Stochastic pour confirmation
        stoch_k = values.get('stoch_k')
        stoch_d = values.get('stoch_d')
        if stoch_k is not None and stoch_d is not None:
            try:
                k = float(stoch_k)
                d = float(stoch_d)
                if k <= 20 and d <= 20:
                    confidence_boost += 0.12
                    reason += f" + Stoch survente ({k:.1f},{d:.1f})"
            except (ValueError, TypeError):
                pass
                
        # Williams %R
        williams_r = values.get('williams_r')
        if williams_r is not None:
            try:
                wr = float(williams_r)
                if wr <= -80:
                    confidence_boost += 0.10
                    reason += f" + Williams%R survente ({wr:.1f})"
            except (ValueError, TypeError):
                pass
                
        # MACD pour momentum de retournement
        macd_histogram = values.get('macd_histogram')
        if macd_histogram is not None:
            try:
                histogram = float(macd_histogram)
                if histogram > 0:  # MACD commence à remonter
                    confidence_boost += 0.08
                    reason += " + MACD retournement"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec directional bias
        directional_bias = values.get('directional_bias')
        if directional_bias == "BULLISH":
            confidence_boost += 0.10
            reason += " + bias haussier"
        elif directional_bias == "BEARISH":
            confidence_boost -= 0.10
            reason += " mais bias baissier"
            
        # Pattern recognition
        pattern_detected = values.get('pattern_detected')
        pattern_confidence = values.get('pattern_confidence')
        if pattern_detected and pattern_confidence is not None:
            try:
                pattern_conf = float(pattern_confidence)
                if pattern_conf > 70:
                    confidence_boost += 0.10
                    reason += f" + pattern {pattern_detected}"
                elif pattern_conf > 50:
                    confidence_boost += 0.05
                    reason += f" + pattern faible {pattern_detected}"
            except (ValueError, TypeError):
                pass
                
        # Market regime avec pénalité (pas de rejet direct)
        market_regime = values.get('market_regime')
        if market_regime == "TRENDING_BEAR":
            confidence_boost -= 0.15
            reason += " (marché baissier, setup risqué)"
        elif market_regime == "TRENDING_BULL":
            confidence_boost += 0.08
            reason += " (marché haussier)"
        elif market_regime == "RANGING":
            confidence_boost += 0.10
            reason += " (marché en range - favorable aux sweeps)"
        elif market_regime == "TRANSITION":
            confidence_boost += 0.05
            reason += " (marché en transition)"
            
        # Signal strength (DB: WEAK/MODERATE/STRONG/VERY_STRONG/VERY_WEAK - UPPERCASE)
        signal_strength_calc = values.get('signal_strength')
        if signal_strength_calc is not None:
            sig_str = str(signal_strength_calc).upper()
            if sig_str == 'VERY_STRONG':
                confidence_boost += 0.15
                reason += " + signal très fort"
            elif sig_str == 'STRONG':
                confidence_boost += 0.10
                reason += " + signal fort"
            elif sig_str == 'MODERATE':
                confidence_boost += 0.05
                reason += " + signal modéré"
            # WEAK et VERY_WEAK = pas de bonus (signal faible)
                
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if confluence > 60:
                    confidence_boost += 0.12
                    reason += f" + confluence élevée ({confluence:.0f})"
                elif confluence > 45:
                    confidence_boost += 0.08
                    reason += f" + confluence modérée ({confluence:.0f})"
            except (ValueError, TypeError):
                pass
                
        confidence = min(1.0, self.calculate_confidence(base_confidence, 1 + confidence_boost))
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
                "nearest_support": nearest_support,
                "support_strength": support_strength_raw,
                "support_strength_score": support_strength_score,
                "sweep_bars_ago": sweep_analysis['sweep_bars_ago'],
                "max_sweep_distance_pct": sweep_analysis['max_sweep_distance'] * 100,
                "recovery_distance_pct": sweep_analysis['recovery_distance'] * 100,
                "volume_ratio": volume_ratio,
                "volume_spike_multiplier": volume_spike_multiplier,
                "rsi_14": rsi_14,
                "stoch_k": stoch_k,
                "stoch_d": stoch_d,
                "williams_r": williams_r,
                "macd_histogram": macd_histogram,
                "directional_bias": directional_bias,
                "pattern_detected": pattern_detected,
                "pattern_confidence": pattern_confidence,
                "market_regime": market_regime,
                "confluence_score": confluence_score
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que toutes les données requises sont présentes."""
        if not super().validate_data():
            return False
            
        # Liquidity sweep nécessite un support FIXE ou fallback possible
        if self.indicators.get('nearest_support') is None or self.indicators.get('nearest_support') == 0:
            # Vérifier si on peut faire un fallback avec les données low (8 barres minimum)
            if not self.data or 'low' not in self.data or not self.data['low'] or len(self.data['low']) < 8:
                logger.warning(f"{self.name}: nearest_support manquant et données low insuffisantes pour fallback")
                return False
                
        # Vérifier qu'on a des données OHLCV suffisantes pour analyse sweep
        required_ohlcv = ['close', 'low', 'high', 'volume']
        for key in required_ohlcv:
            if not self.data or key not in self.data or not self.data[key] or len(self.data[key]) < 8:
                logger.warning(f"{self.name}: Données {key} insuffisantes pour liquidity sweep (besoin 8 barres)")
                return False
                
        return True
