"""
Volume_Spike_Validator - Validator basé sur les pics de volume et patterns d'activité.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class Volume_Spike_Validator(BaseValidator):
    """
    Validator pour les pics de volume - filtre selon l'intensité et qualité des spikes volume.
    
    Vérifie: Volume spikes, durée, qualité, timing, divergences
    Catégorie: technical
    
    Rejette les signaux en:
    - Absence de spike volume significatif
    - Spikes de mauvaise qualité ou erratiques
    - Timing inapproprié du spike
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "Volume_Spike_Validator"
        self.category = "technical"
        
        # Paramètres spike de base
        self.min_spike_multiplier = 1.5      # Spike minimum 50% au-dessus moyenne
        self.strong_spike_multiplier = 2.5   # Spike considéré fort
        self.exceptional_spike_multiplier = 4.0  # Spike exceptionnel
        self.max_spike_multiplier = 10.0     # Spike maximum acceptable
        
        # Paramètres durée et persistance
        self.min_spike_duration = 1          # Durée minimum spike (barres)
        self.optimal_spike_duration = 3      # Durée optimale spike
        self.max_spike_duration = 8          # Durée maximum spike
        self.spike_decay_threshold = 0.7     # Seuil décroissance spike
        
        # Paramètres qualité spike
        self.min_spike_quality = 40.0        # Qualité minimum spike (format 0-100)
        self.spike_consistency_threshold = 0.6  # Consistance spike minimum
        self.max_spike_volatility = 3.0      # Volatilité spike maximum
        self.min_spike_legitimacy = 50.0     # Légitimité spike minimum (format 0-100)
        
        # Paramètres timing
        self.recent_spike_window = 5         # Fenêtre spike récent (barres)
        self.spike_cooldown_period = 3       # Période cooldown entre spikes
        self.max_time_since_spike = 10       # Temps maximum depuis spike
        
        # Paramètres contexte marché
        self.min_relative_spike = 1.3        # Spike relatif au contexte marché
        self.spike_vs_trend_coherence = 0.6  # Cohérence spike/tendance
        self.unusual_activity_threshold = 2.0 # Seuil activité inhabituelle
        
        # Paramètres divergences
        self.max_price_volume_divergence = 0.4  # Max divergence prix/volume
        self.spike_confirmation_threshold = 0.7 # Confirmation spike par prix
        
        # Bonus/malus
        self.exceptional_spike_bonus = 0.30  # Bonus spike exceptionnel
        self.quality_spike_bonus = 0.25      # Bonus spike de qualité
        self.timing_spike_bonus = 0.20       # Bonus timing parfait
        self.confirmation_bonus = 0.15       # Bonus confirmation prix
        self.weak_spike_penalty = -0.25      # Pénalité spike faible
        self.poor_timing_penalty = -0.20     # Pénalité mauvais timing
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur les pics de volume et patterns d'activité.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon volume spike, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs spike depuis le contexte (noms DB réels)
            try:
                # Indicateurs de base
                volume_spike_multiplier = float(self.context.get('volume_spike_multiplier', 1.0)) if self.context.get('volume_spike_multiplier') is not None else None
                volume_buildup_periods = int(self.context.get('volume_buildup_periods', 0)) if self.context.get('volume_buildup_periods') is not None else None
                volume_quality_score = float(self.context.get('volume_quality_score', 0.5)) if self.context.get('volume_quality_score') is not None else None
                relative_volume = float(self.context.get('relative_volume', 1.0)) if self.context.get('relative_volume') is not None else None
                trade_intensity = float(self.context.get('trade_intensity', 0.5)) if self.context.get('trade_intensity') is not None else None
                
                # Contexte et confirmation
                momentum_score = float(self.context.get('momentum_score', 50)) if self.context.get('momentum_score') is not None else None
                signal_strength = self.context.get('signal_strength')  # STRING format
                pattern_confidence = float(self.context.get('pattern_confidence', 0.5)) if self.context.get('pattern_confidence') is not None else None
                regime_duration = int(self.context.get('regime_duration', 5)) if self.context.get('regime_duration') is not None else None
                
                # Indicateurs marché
                directional_bias = self.context.get('directional_bias', 'neutral')
                unusual_activity_detected = self.context.get('unusual_activity_detected', False)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion indicateurs pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', 'Unknown')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side:
                logger.warning(f"{self.name}: Signal side manquant pour {self.symbol}")
                return False
                
            # 1. Validation multiplier spike minimum
            if volume_spike_multiplier is not None and volume_spike_multiplier < self.min_spike_multiplier:
                logger.debug(f"{self.name}: Spike volume insuffisant ({self._safe_format(volume_spike_multiplier, '.2f')}x) pour {self.symbol}")
                if signal_confidence < 0.8:
                    return False
                    
            # Validation spike maximum (éviter manipulation)
            if volume_spike_multiplier is not None and volume_spike_multiplier > self.max_spike_multiplier:
                logger.debug(f"{self.name}: Spike volume excessif ({self._safe_format(volume_spike_multiplier, '.2f')}x) - possible manipulation pour {self.symbol}")
                if signal_confidence < 0.9:
                    return False
                    
            # 2. Validation durée spike
            if volume_buildup_periods is not None:
                if volume_buildup_periods < self.min_spike_duration:
                    logger.debug(f"{self.name}: Durée spike insuffisante ({volume_buildup_periods} barres) pour {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                elif volume_buildup_periods > self.max_spike_duration:
                    logger.debug(f"{self.name}: Durée spike excessive ({volume_buildup_periods} barres) pour {self.symbol}")
                    if signal_confidence < 0.6:
                        return False
                        
            # 3. Validation qualité spike
            if volume_quality_score is not None and volume_quality_score < self.min_spike_quality:
                logger.debug(f"{self.name}: Qualité spike insuffisante ({self._safe_format(volume_quality_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 4. Validation légitimité spike
            if pattern_confidence is not None and pattern_confidence < (self.min_spike_legitimacy / 100):
                logger.debug(f"{self.name}: Légitimité spike douteuse ({self._safe_format(pattern_confidence, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 5. Validation timing spike (regime_duration)
            if regime_duration is not None and regime_duration > self.max_time_since_spike:
                logger.debug(f"{self.name}: Spike trop ancien ({regime_duration} barres) pour {self.symbol}")
                if signal_confidence < 0.5:
                    return False
                    
            # 6. Validation spike relatif au contexte
            if relative_volume is not None and relative_volume < self.min_relative_spike:
                logger.debug(f"{self.name}: Spike relatif insuffisant ({self._safe_format(relative_volume, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 7. Validation activité marché
            if trade_intensity is not None and trade_intensity < 0.3:
                logger.debug(f"{self.name}: Activité marché faible ({self._safe_format(trade_intensity, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.5:
                    return False
                    
            # 8. Validation spécifique selon stratégie
            if volume_spike_multiplier is not None:
                strategy_spike_match = self._validate_strategy_spike_match(signal_strategy, volume_spike_multiplier)
            else:
                strategy_spike_match = True
            if not strategy_spike_match:
                logger.debug(f"{self.name}: Stratégie {signal_strategy} inadaptée aux spikes pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                        f"Spike: {self._safe_format(volume_spike_multiplier, '.2f')}x, "
                        f"Qualité: {self._safe_format(volume_quality_score, '.2f') if volume_quality_score is not None else 'N/A'}, "
                        f"Durée: {volume_buildup_periods or 'N/A'} barres, "
                        f"Timing: {regime_duration or 'N/A'} barres ago")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def _validate_strategy_spike_match(self, strategy: str, spike_multiplier: float) -> bool:
        """Valide l'adéquation stratégie/spike volume."""
        try:
            if not strategy:
                return True
                
            strategy_lower = strategy.lower()
            
            # Stratégies qui bénéficient des gros spikes
            high_volume_strategies = ['breakout', 'momentum', 'spike', 'volume', 'liquidity']
            
            # Stratégies sensibles aux spikes moyens
            medium_volume_strategies = ['trend', 'cross', 'macd']
            
            # Stratégies moins dépendantes du volume
            low_volume_strategies = ['bollinger', 'rsi', 'reversal', 'touch']
            
            if any(kw in strategy_lower for kw in high_volume_strategies):
                # Stratégies high-volume nécessitent spikes forts
                if spike_multiplier and spike_multiplier < self.strong_spike_multiplier:
                    return False
                    
            elif any(kw in strategy_lower for kw in medium_volume_strategies):
                # Stratégies medium-volume nécessitent spikes modérés
                if spike_multiplier and spike_multiplier < self.min_spike_multiplier + 0.5:
                    return False
                    
            # Stratégies low-volume acceptées avec spikes faibles
            return True
            
        except Exception:
            return True
            
    def _is_spike_directionally_coherent(self, signal_side: str, directional_bias: str, 
                                        price_momentum: float, spike_price_confirmation: float) -> bool:
        """Détermine si le spike de volume est cohérent avec la direction du signal."""
        try:
            # Facteurs de cohérence
            coherence_factors = 0
            total_factors = 0
            
            # Facteur 1: Cohérence avec bias directionnel
            if directional_bias and directional_bias != 'neutral':
                total_factors += 1
                if ((signal_side == "BUY" and directional_bias == "bullish") or 
                    (signal_side == "SELL" and directional_bias == "bearish")):
                    coherence_factors += 1
                    
            # Facteur 2: Cohérence avec momentum prix
            total_factors += 1
            if signal_side == "BUY" and price_momentum > 55:  # Momentum haussier
                coherence_factors += 1
            elif signal_side == "SELL" and price_momentum < 45:  # Momentum baissier
                coherence_factors += 1
            elif 45 <= price_momentum <= 55:  # Momentum neutre = cohérent pour les deux
                coherence_factors += 0.5
                
            # Facteur 3: Confirmation prix du spike
            if spike_price_confirmation is not None:
                total_factors += 1
                if spike_price_confirmation >= 0.6:  # Bonne confirmation
                    coherence_factors += 1
                elif spike_price_confirmation >= 0.4:  # Confirmation modérée
                    coherence_factors += 0.5
                    
            # Calculer ratio de cohérence
            if total_factors == 0:
                return True  # Pas de données pour juger = neutral
                
            coherence_ratio = coherence_factors / total_factors
            
            # Seuil de cohérence: 60% des facteurs doivent être favorables
            return coherence_ratio >= 0.6
            
        except Exception:
            return True  # En cas d'erreur, ne pas pénaliser
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur les pics de volume.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur volume spikes
            volume_spike_multiplier = float(self.context.get('volume_spike_multiplier', 1.0)) if self.context.get('volume_spike_multiplier') is not None else 1.0
            # spike_duration_bars → volume_buildup_periods
            spike_duration_bars = int(self.context.get('volume_buildup_periods', 1)) if self.context.get('volume_buildup_periods') is not None else 1
            # spike_quality_score → volume_quality_score
            spike_quality_score = float(self.context.get('volume_quality_score', 0.5)) if self.context.get('volume_quality_score') is not None else 0.5
            # spike_consistency → volume_quality_score
            spike_consistency = float(self.context.get('volume_quality_score', 0.5)) if self.context.get('volume_quality_score') is not None else 0.5
            # spike_legitimacy_score → pattern_confidence
            spike_legitimacy_score = float(self.context.get('pattern_confidence', 0.5)) if self.context.get('pattern_confidence') is not None else 0.5
            # time_since_spike → regime_duration
            time_since_spike = int(self.context.get('regime_duration', 5)) if self.context.get('regime_duration') is not None else 5
            # relative_spike_strength → relative_volume
            relative_spike_strength = float(self.context.get('relative_volume', 1.0)) if self.context.get('relative_volume') is not None else 1.0
            # spike_price_confirmation → pattern_confidence
            spike_price_confirmation = float(self.context.get('pattern_confidence', 0.5)) if self.context.get('pattern_confidence') is not None else 0.5
            unusual_activity_detected = self.context.get('unusual_activity_detected', False)
            
            signal_strategy = signal.get('strategy', '')
            signal_side = signal.get('side')
            
            base_score = 0.5  # Score de base si validé
            
            # CORRECTION: Bonus spike selon cohérence directionnelle + contexte
            # Obtenir le contexte directionnel pour interpréter le spike
            directional_bias = self.context.get('directional_bias', 'neutral')
            price_momentum = float(self.context.get('momentum_score', 50)) if self.context.get('momentum_score') is not None else 50
            
            # Déterminer si le spike est cohérent avec la direction
            spike_coherent = self._is_spike_directionally_coherent(
                signal_side, directional_bias, price_momentum, spike_price_confirmation
            )
            
            # Bonus multiplier spike (réduit si incohérent)
            if volume_spike_multiplier >= self.exceptional_spike_multiplier:
                if spike_coherent:
                    base_score += self.exceptional_spike_bonus  # Plein bonus si cohérent
                else:
                    base_score += self.exceptional_spike_bonus * 0.5  # Bonus réduit si incohérent
            elif volume_spike_multiplier >= self.strong_spike_multiplier:
                if spike_coherent:
                    base_score += 0.20
                else:
                    base_score += 0.10  # Bonus réduit
            elif volume_spike_multiplier >= self.min_spike_multiplier + 0.5:
                if spike_coherent:
                    base_score += 0.10
                else:
                    base_score += 0.05  # Bonus réduit
                
            # Bonus durée spike optimale
            if spike_duration_bars == self.optimal_spike_duration:
                base_score += 0.15  # Durée parfaite
            elif self.min_spike_duration <= spike_duration_bars <= self.max_spike_duration:
                base_score += 0.08  # Durée acceptable
                
            # Bonus qualité spike
            if spike_quality_score >= 80.0:
                base_score += self.quality_spike_bonus
            elif spike_quality_score >= 60.0:
                base_score += 0.15
                
            # Bonus consistance spike
            if spike_consistency >= 0.8:
                base_score += 0.12  # Spike très consistant
            elif spike_consistency >= self.spike_consistency_threshold:
                base_score += 0.08
                
            # Bonus légitimité spike
            if spike_legitimacy_score >= 80.0:
                base_score += 0.15  # Spike très légitime
            elif spike_legitimacy_score >= self.min_spike_legitimacy:
                base_score += 0.08
                
            # Bonus timing optimal
            if time_since_spike <= 2:
                base_score += self.timing_spike_bonus  # Timing parfait
            elif time_since_spike <= self.recent_spike_window:
                base_score += 0.12  # Bon timing
                
            # Bonus spike relatif fort
            if relative_spike_strength >= 2.0:
                base_score += 0.12  # Spike très fort relativement
            elif relative_spike_strength >= self.min_relative_spike:
                base_score += 0.08
                
            # Bonus confirmation prix
            if spike_price_confirmation >= 0.8:
                base_score += self.confirmation_bonus
            elif spike_price_confirmation >= self.spike_confirmation_threshold:
                base_score += 0.10
                
            # Bonus activité inhabituelle
            if unusual_activity_detected:
                base_score += 0.10  # Activité inhabituelle détectée
                
            # Bonus stratégie adaptée
            strategy_lower = signal_strategy.lower()
            if any(kw in strategy_lower for kw in ['breakout', 'momentum', 'spike', 'volume']):
                base_score += 0.08  # Stratégie adaptée aux spikes
                
            return max(0.0, min(1.0, base_score))
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur calcul score pour {self.symbol}: {e}")
            return 0.0
            
    def get_validation_reason(self, signal: Dict[str, Any], is_valid: bool) -> str:
        """
        Retourne une raison détaillée pour la validation/invalidation.
        
        Args:
            signal: Le signal évalué
            is_valid: Résultat de la validation
            
        Returns:
            Raison de la décision
        """
        try:
            signal_side = signal.get('side', 'N/A')
            signal_strategy = signal.get('strategy', 'N/A')
            
            volume_spike_multiplier = float(self.context.get('volume_spike_multiplier', 1.0)) if self.context.get('volume_spike_multiplier') is not None else None
            # spike_duration_bars → volume_buildup_periods
            spike_duration_bars = int(self.context.get('volume_buildup_periods', 0)) if self.context.get('volume_buildup_periods') is not None else None
            # spike_quality_score → volume_quality_score
            spike_quality_score = float(self.context.get('volume_quality_score', 0.5)) if self.context.get('volume_quality_score') is not None else None
            # time_since_spike → regime_duration
            time_since_spike = int(self.context.get('regime_duration', 0)) if self.context.get('regime_duration') is not None else None
            # spike_legitimacy_score → pattern_confidence
            spike_legitimacy_score = float(self.context.get('pattern_confidence', 0.5)) if self.context.get('pattern_confidence') is not None else None
            
            if is_valid:
                reason = f"Spike volume favorable"
                if volume_spike_multiplier:
                    spike_desc = "exceptionnel" if volume_spike_multiplier >= self.exceptional_spike_multiplier else "fort" if volume_spike_multiplier >= self.strong_spike_multiplier else "modéré"
                    reason += f" ({spike_desc}: {self._safe_format(volume_spike_multiplier, '.2f')}x)"
                if spike_quality_score:
                    reason += f", qualité: {self._safe_format(spike_quality_score, '.2f')}"
                if spike_duration_bars:
                    reason += f", durée: {spike_duration_bars} barres"
                if time_since_spike is not None:
                    timing_desc = "immédiat" if time_since_spike <= 1 else "récent" if time_since_spike <= 3 else "proche"
                    reason += f", timing: {timing_desc}"
                if spike_legitimacy_score:
                    reason += f", légitimité: {self._safe_format(spike_legitimacy_score, '.2f')}"
                    
                return f"{self.name}: Validé - {reason} pour {signal_strategy} {signal_side}"
            else:
                if volume_spike_multiplier and volume_spike_multiplier < self.min_spike_multiplier:
                    return f"{self.name}: Rejeté - Spike volume insuffisant ({self._safe_format(volume_spike_multiplier, '.2f')}x)"
                elif volume_spike_multiplier and volume_spike_multiplier > self.max_spike_multiplier:
                    return f"{self.name}: Rejeté - Spike volume excessif ({self._safe_format(volume_spike_multiplier, '.2f')}x) - manipulation possible"
                elif spike_quality_score and spike_quality_score < self.min_spike_quality:
                    return f"{self.name}: Rejeté - Qualité spike insuffisante ({self._safe_format(spike_quality_score, '.2f')})"
                elif spike_duration_bars and spike_duration_bars < self.min_spike_duration:
                    return f"{self.name}: Rejeté - Durée spike insuffisante ({spike_duration_bars} barres)"
                elif spike_duration_bars and spike_duration_bars > self.max_spike_duration:
                    return f"{self.name}: Rejeté - Durée spike excessive ({spike_duration_bars} barres)"
                elif time_since_spike and time_since_spike > self.max_time_since_spike:
                    return f"{self.name}: Rejeté - Spike trop ancien ({time_since_spike} barres)"
                elif spike_legitimacy_score and spike_legitimacy_score < self.min_spike_legitimacy:
                    return f"{self.name}: Rejeté - Légitimité spike douteuse ({self._safe_format(spike_legitimacy_score, '.2f')})"
                    
                return f"{self.name}: Rejeté - Critères volume spike non respectés"
                
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données de spike requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Au minimum, on a besoin d'un indicateur de spike (avec mappings)
        spike_indicators = [
            'volume_spike_multiplier', 'volume_spike_multiplier', 'volume_quality_score',
            'volume_buildup_periods', 'regime_duration', 'relative_volume'
        ]
        
        available_indicators = sum(1 for ind in spike_indicators 
                                 if ind in self.context and self.context[ind] is not None)
        
        if available_indicators < 2:
            logger.warning(f"{self.name}: Pas assez d'indicateurs de spike pour {self.symbol}")
            return False
            
        return True
