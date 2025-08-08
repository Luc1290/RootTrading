"""
TRIX_Crossover_Strategy - Stratégie basée sur TRIX simulé avec TEMA.
TRIX est un oscillateur basé sur le taux de changement pourcentuel d'une triple EMA lissée.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class TRIX_Crossover_Strategy(BaseStrategy):
    """
    Stratégie utilisant TRIX simulé avec TEMA pour détecter les changements de momentum.
    
    TRIX (Triple Exponential Average) :
    - TRIX = (TEMA_today - TEMA_yesterday) / TEMA_yesterday * 10000
    - Indicateur de momentum basé sur la pente de TEMA
    - Filtre le bruit mieux que MACD grâce au triple lissage
    
    Comme TRIX n'est pas directement disponible, nous le simulons avec :
    - TEMA 12 périodes comme base
    - ROC sur TEMA pour approximer TRIX
    - DEMA 12 comme signal line (plus rapide que TEMA)
    
    Signaux générés:
    - BUY: TRIX simulé crosses above zero + confirmations haussières
    - SELL: TRIX simulé crosses below zero + confirmations baissières
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        
        # Paramètres TRIX simulé DURCIS - Zone neutre ajoutée
        self.trix_bullish_threshold = 0.02      # Seuil haussier > 0.02% (plus strict)
        self.trix_bearish_threshold = -0.02     # Seuil baissier < -0.02% (plus strict) 
        self.neutral_zone = 0.02                # Zone neutre ±0.02% pour éviter le bruit
        self.strong_trix_threshold = 0.08       # TRIX fort > 0.08% (relevé)
        self.extreme_trix_threshold = 0.15      # TRIX extrême > 0.15% (relevé)
        
        # Paramètres crossover signal line
        self.signal_line_crossover = True       # Utiliser DEMA comme signal line
        self.min_tema_dema_separation = 0.005   # Séparation minimum TEMA/DEMA (5x plus strict)
        
        # Paramètres momentum confirmation
        self.momentum_alignment_required = True  # Momentum doit être aligné
        self.min_roc_confirmation = 0.005       # ROC minimum pour confirmation (format décimal)
        
        # Paramètres filtrage  
        self.min_trend_strength = 'WEAK'        # Trend strength minimum (STRING format schéma DB)
        self.volume_confirmation_threshold = 1.1 # Volume confirmation
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            # Moyennes mobiles (base TRIX)
            'tema_12': self.indicators.get('tema_12'),
            'dema_12': self.indicators.get('dema_12'),
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            
            # ROC et momentum (approximation TRIX)
            'roc_10': self.indicators.get('roc_10'),
            'roc_20': self.indicators.get('roc_20'),
            'momentum_10': self.indicators.get('momentum_10'),
            'momentum_score': self.indicators.get('momentum_score'),
            
            # Trend et direction
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            'trend_angle': self.indicators.get('trend_angle'),
            
            # MACD (comparaison avec TRIX)
            'macd_line': self.indicators.get('macd_line'),
            'macd_signal': self.indicators.get('macd_signal'),
            'macd_histogram': self.indicators.get('macd_histogram'),
            'macd_zero_cross': self.indicators.get('macd_zero_cross'),
            
            # Volume et confirmation
            'volume_ratio': self.indicators.get('volume_ratio'),
            'relative_volume': self.indicators.get('relative_volume'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            
            # Oscillateurs (confluence)
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            
            # Market context
            'market_regime': self.indicators.get('market_regime'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            'confluence_score': self.indicators.get('confluence_score'),
            'signal_strength': self.indicators.get('signal_strength')
        }
        
    def _calculate_trix_proxy(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule une approximation de TRIX basée sur TEMA et ROC."""
        tema_12 = values.get('tema_12')
        roc_10 = values.get('roc_10')
        momentum_10 = values.get('momentum_10')
        
        if tema_12 is None:
            return {'trix_value': None, 'trix_direction': None, 'trix_strength': 0}
            
        try:
            tema_val = float(tema_12)
            
            # Approximation TRIX avec ROC sur TEMA
            if roc_10 is not None:
                roc_val = float(roc_10)
                # ROC déjà en format décimal, utiliser directement
                trix_proxy = roc_val  # ROC en format décimal
            elif momentum_10 is not None:
                # Alternative avec momentum
                momentum_val = float(momentum_10)
                trix_proxy = momentum_val / 1000  # Normaliser momentum
            else:
                return {'trix_value': None, 'trix_direction': None, 'trix_strength': 0}
                
            # Direction TRIX avec ZONE NEUTRE élargie
            if abs(trix_proxy) < self.neutral_zone:
                # Zone neutre - pas de signal TRIX
                trix_direction = 'neutral'
                trix_strength = 0.1
            elif trix_proxy >= self.extreme_trix_threshold:
                trix_direction = 'extreme_bullish'  # Nouveau niveau
                trix_strength = 0.9  # Très fort
            elif trix_proxy >= self.strong_trix_threshold:
                trix_direction = 'strong_bullish'
                trix_strength = 0.7  # Réduit de 0.8
            elif trix_proxy >= self.trix_bullish_threshold:
                trix_direction = 'bullish'
                trix_strength = 0.4  # Réduit de 0.5
            elif trix_proxy <= -self.extreme_trix_threshold:
                trix_direction = 'extreme_bearish'  # Nouveau niveau
                trix_strength = 0.9  # Très fort
            elif trix_proxy <= -self.strong_trix_threshold:
                trix_direction = 'strong_bearish'
                trix_strength = 0.7  # Réduit de 0.8
            elif trix_proxy <= self.trix_bearish_threshold:
                trix_direction = 'bearish'
                trix_strength = 0.4  # Réduit de 0.5
            else:
                trix_direction = 'neutral'
                trix_strength = 0.1
                
            return {
                'trix_value': trix_proxy,
                'trix_direction': trix_direction,
                'trix_strength': trix_strength,
                'tema_value': tema_val
            }
            
        except (ValueError, TypeError):
            return {'trix_value': None, 'trix_direction': None, 'trix_strength': 0}
            
    def _detect_signal_line_crossover(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte crossover entre TEMA (TRIX base) et DEMA (signal line)."""
        tema_12 = values.get('tema_12')
        dema_12 = values.get('dema_12')
        
        if tema_12 is None or dema_12 is None:
            return {'is_crossover': False, 'direction': None, 'strength': 0}
            
        try:
            tema_val = float(tema_12)
            dema_val = float(dema_12)
            
            # Différence relative
            diff = (tema_val - dema_val) / tema_val
            
            # Détection crossover
            if diff > self.min_tema_dema_separation:
                # TEMA > DEMA = signal haussier
                crossover_direction = 'bullish'
                crossover_strength = min(abs(diff) * 100, 1.0)  # Normaliser
            elif diff < -self.min_tema_dema_separation:
                # TEMA < DEMA = signal baissier
                crossover_direction = 'bearish'
                crossover_strength = min(abs(diff) * 100, 1.0)
            else:
                # Pas de crossover significatif
                crossover_direction = 'neutral'
                crossover_strength = 0.1
                
            return {
                'is_crossover': crossover_strength > 0.2,
                'direction': crossover_direction,
                'strength': crossover_strength,
                'tema_dema_diff': diff
            }
            
        except (ValueError, TypeError):
            return {'is_crossover': False, 'direction': None, 'strength': 0}
            
    def _detect_momentum_alignment(self, values: Dict[str, Any], trix_direction: str) -> Dict[str, Any]:
        """Détecte l'alignement du momentum avec TRIX."""
        momentum_score = values.get('momentum_score')
        directional_bias = values.get('directional_bias')
        trend_strength = values.get('trend_strength')
        
        alignment_score = 0.0
        alignment_indicators = []
        
        # Momentum score alignment
        if momentum_score is not None:
            try:
                momentum_val = float(momentum_score)
                # momentum_score format 0-100, 50=neutre - SEUILS PLUS STRICTS
                if trix_direction in ['bullish', 'strong_bullish', 'extreme_bullish'] and momentum_val > 55:  # Plus strict
                    alignment_score += 0.20  # Réduit de 0.3
                    alignment_indicators.append(f"Momentum haussier ({momentum_val:.1f})")
                elif trix_direction in ['bearish', 'strong_bearish', 'extreme_bearish'] and momentum_val < 45:  # Plus strict
                    alignment_score += 0.20  # Réduit de 0.3
                    alignment_indicators.append(f"Momentum baissier ({momentum_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # Directional bias alignment - RÉDUIT
        if directional_bias:
            if trix_direction in ['bullish', 'strong_bullish', 'extreme_bullish'] and directional_bias == 'BULLISH':
                alignment_score += 0.18  # Réduit de 0.25
                alignment_indicators.append("Bias directionnel haussier")
            elif trix_direction in ['bearish', 'strong_bearish', 'extreme_bearish'] and directional_bias == 'BEARISH':
                alignment_score += 0.18  # Réduit de 0.25
                alignment_indicators.append("Bias directionnel baissier")
                
        # Trend strength confirmation (format STRING selon schéma DB)
        if trend_strength is not None:
            # trend_strength: WEAK/MODERATE/STRONG/VERY_STRONG - RÉDUIT
            if trend_strength == 'VERY_STRONG':
                alignment_score += 0.18  # Réduit de 0.25
                alignment_indicators.append(f"Trend très forte ({trend_strength})")
            elif trend_strength == 'STRONG':
                alignment_score += 0.14  # Réduit de 0.2
                alignment_indicators.append(f"Trend forte ({trend_strength})")
            elif trend_strength == 'MODERATE':
                alignment_score += 0.10  # Réduit de 0.15
                alignment_indicators.append(f"Trend modérée ({trend_strength})")
            elif trend_strength == 'WEAK':
                alignment_score += 0.06  # Réduit de 0.1
                alignment_indicators.append(f"Trend faible ({trend_strength})")
                
        # ROC confirmation
        roc_10 = values.get('roc_10')
        if roc_10 is not None:
            try:
                roc_val = float(roc_10)
                if trix_direction in ['bullish', 'strong_bullish', 'extreme_bullish'] and roc_val > self.min_roc_confirmation:
                    alignment_score += 0.12  # Réduit de 0.15
                    alignment_indicators.append(f"ROC haussier ({roc_val*100:.1f}%)")
                elif trix_direction in ['bearish', 'strong_bearish', 'extreme_bearish'] and roc_val < -self.min_roc_confirmation:
                    alignment_score += 0.12  # Réduit de 0.15
                    alignment_indicators.append(f"ROC baissier ({roc_val*100:.1f}%)")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_aligned': alignment_score >= 0.35,  # Seuil légèrement réduit (0.35 au lieu de 0.4)
            'score': alignment_score,
            'indicators': alignment_indicators
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur TRIX crossover simulé.
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
        
        # Calculer TRIX proxy
        trix_data = self._calculate_trix_proxy(values)
        if trix_data['trix_value'] is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "TEMA/ROC indisponible pour calcul TRIX",
                "metadata": {"strategy": self.name}
            }
            
        trix_direction = trix_data['trix_direction']
        if trix_direction == 'neutral':
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"TRIX neutre ({trix_data['trix_value']:.4f}) - pas de crossover",
                "metadata": {
                    "strategy": self.name,
                    "trix_value": trix_data['trix_value']
                }
            }
            
        # Détection signal line crossover (optionnel)
        crossover_data = self._detect_signal_line_crossover(values)
        
        # Vérifier alignment du momentum
        alignment_data = self._detect_momentum_alignment(values, trix_direction)
        
        if self.momentum_alignment_required and not alignment_data['is_aligned']:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"TRIX {trix_direction} mais momentum pas aligné",
                "metadata": {
                    "strategy": self.name,
                    "trix_direction": trix_direction,
                    "alignment_score": alignment_data['score']
                }
            }
            
        # Générer signal selon direction TRIX - INCLUANT NOUVEAUX NIVEAUX
        signal_side = None
        if trix_direction in ['bullish', 'strong_bullish', 'extreme_bullish']:
            signal_side = "BUY"
        elif trix_direction in ['bearish', 'strong_bearish', 'extreme_bearish']:
            signal_side = "SELL"
            
        if signal_side is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Direction TRIX indéterminée: {trix_direction}",
                "metadata": {"strategy": self.name}
            }
            
        # Calculer confidence - BASE RÉDUITE pour indicateur simulé
        base_confidence = 0.35  # Augmenté pour équilibrage avec autres stratégies
        confidence_boost = 0.0
        
        # Score TRIX strength - RÉDUIT
        confidence_boost += trix_data['trix_strength'] * 0.25  # Réduit de 0.4
        
        # Score alignment momentum - RÉDUIT
        confidence_boost += alignment_data['score'] * 0.20  # Réduit de 0.3
        
        reason = f"TRIX {trix_direction} ({trix_data['trix_value']:.4f})"
        
        if alignment_data['indicators']:
            reason += f" + {alignment_data['indicators'][0]}"
            
        # Bonus signal line crossover
        if self.signal_line_crossover and crossover_data['is_crossover']:
            if ((signal_side == "BUY" and crossover_data['direction'] == 'bullish') or
                (signal_side == "SELL" and crossover_data['direction'] == 'bearish')):
                confidence_boost += crossover_data['strength'] * 0.15  # Réduit de 0.2
                reason += " + crossover signal line"
                
        # Volume confirmation
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.volume_confirmation_threshold:
                    confidence_boost += 0.08  # Réduit de 0.1
                    reason += f" + volume ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # MACD confluence
        macd_line = values.get('macd_line')
        if macd_line is not None:
            try:
                macd_val = float(macd_line)
                if ((signal_side == "BUY" and macd_val > 0) or
                    (signal_side == "SELL" and macd_val < 0)):
                    confidence_boost += 0.07  # Réduit de 0.1
                    reason += " + MACD aligné"
            except (ValueError, TypeError):
                pass
                
        # Confluence score
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                conf_val = float(confluence_score)
                if conf_val > 75:  # Seuil plus strict
                    confidence_boost += 0.08  # Réduit de 0.1
                    reason += " + haute confluence"
            except (ValueError, TypeError):
                pass
                
        # NOUVEAU: Filtre final - rejeter si confidence insuffisante
        raw_confidence = base_confidence * (1.0 + confidence_boost)
        if raw_confidence < 0.35:  # Seuil minimum 35% pour TRIX simulé
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal TRIX rejeté - confiance insuffisante ({raw_confidence:.2f} < 0.35)",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "rejected_signal": signal_side,
                    "raw_confidence": raw_confidence,
                    "trix_value": trix_data['trix_value'],
                    "trix_direction": trix_direction
                }
            }
        
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
                "trix_value": trix_data['trix_value'],
                "trix_direction": trix_direction,
                "trix_strength": trix_data['trix_strength'],
                "tema_value": trix_data['tema_value'],
                "alignment_score": alignment_data['score'],
                "alignment_indicators": alignment_data['indicators'],
                "crossover_direction": crossover_data.get('direction'),
                "crossover_strength": crossover_data.get('strength'),
                "volume_ratio": volume_ratio,
                "confluence_score": confluence_score
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        required_indicators = [
            'tema_12', 'roc_10', 'momentum_score'
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
