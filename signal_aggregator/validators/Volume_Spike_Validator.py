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
                
            # Extraction des indicateurs spike depuis le contexte
            try:
                # Spike de base
                volume_spike_multiplier = float(self.context.get('volume_spike_multiplier', 1.0)) if self.context.get('volume_spike_multiplier') is not None else None
                current_volume_spike = float(self.context.get('current_volume_spike', 1.0)) if self.context.get('current_volume_spike') is not None else None
                volume_spike_strength = float(self.context.get('volume_spike_strength', 0)) if self.context.get('volume_spike_strength') is not None else None
                
                # Durée et persistance
                spike_duration_bars = int(self.context.get('spike_duration_bars', 0)) if self.context.get('spike_duration_bars') is not None else None
                spike_decay_rate = float(self.context.get('spike_decay_rate', 0)) if self.context.get('spike_decay_rate') is not None else None
                spike_persistence_score = float(self.context.get('spike_persistence_score', 0)) if self.context.get('spike_persistence_score') is not None else None
                
                # Qualité spike
                spike_quality_score = float(self.context.get('spike_quality_score', 0.5)) if self.context.get('spike_quality_score') is not None else None
                spike_consistency = float(self.context.get('spike_consistency', 0.5)) if self.context.get('spike_consistency') is not None else None
                spike_legitimacy_score = float(self.context.get('spike_legitimacy_score', 0.5)) if self.context.get('spike_legitimacy_score') is not None else None
                
                # Timing
                time_since_spike = int(self.context.get('time_since_spike', 0)) if self.context.get('time_since_spike') is not None else None
                recent_spike_count = int(self.context.get('recent_spike_count', 0)) if self.context.get('recent_spike_count') is not None else None
                last_spike_bars_ago = int(self.context.get('last_spike_bars_ago', 999)) if self.context.get('last_spike_bars_ago') is not None else None
                
                # Contexte marché
                relative_spike_strength = float(self.context.get('relative_spike_strength', 1.0)) if self.context.get('relative_spike_strength') is not None else None
                market_activity_level = float(self.context.get('market_activity_level', 0.5)) if self.context.get('market_activity_level') is not None else None
                unusual_activity_detected = self.context.get('unusual_activity_detected', False)
                
                # Divergences et confirmations
                price_volume_divergence = float(self.context.get('price_volume_divergence', 0)) if self.context.get('price_volume_divergence') is not None else None
                spike_price_confirmation = float(self.context.get('spike_price_confirmation', 0.5)) if self.context.get('spike_price_confirmation') is not None else None
                volume_trend_coherence = float(self.context.get('volume_trend_coherence', 0.5)) if self.context.get('volume_trend_coherence') is not None else None
                
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
                    
            # 2. Validation force spike actuel
            if current_volume_spike is not None and current_volume_spike < self.min_spike_multiplier:
                logger.debug(f"{self.name}: Spike actuel insuffisant ({self._safe_format(current_volume_spike, '.2f')}x) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 3. Validation durée spike
            if spike_duration_bars is not None:
                if spike_duration_bars < self.min_spike_duration:
                    logger.debug(f"{self.name}: Durée spike insuffisante ({spike_duration_bars} barres) pour {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                elif spike_duration_bars > self.max_spike_duration:
                    logger.debug(f"{self.name}: Durée spike excessive ({spike_duration_bars} barres) pour {self.symbol}")
                    if signal_confidence < 0.6:
                        return False
                        
            # 4. Validation décroissance spike
            if spike_decay_rate is not None and spike_decay_rate > (1 - self.spike_decay_threshold):
                logger.debug(f"{self.name}: Décroissance spike trop rapide ({self._safe_format(spike_decay_rate, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 5. Validation qualité spike
            if spike_quality_score is not None and spike_quality_score < self.min_spike_quality:
                logger.debug(f"{self.name}: Qualité spike insuffisante ({self._safe_format(spike_quality_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            if spike_consistency is not None and spike_consistency < self.spike_consistency_threshold:
                logger.debug(f"{self.name}: Consistance spike insuffisante ({self._safe_format(spike_consistency, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 6. Validation légitimité spike
            if spike_legitimacy_score is not None and spike_legitimacy_score < self.min_spike_legitimacy:
                logger.debug(f"{self.name}: Légitimité spike douteuse ({self._safe_format(spike_legitimacy_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 7. Validation timing spike
            if time_since_spike is not None and time_since_spike > self.max_time_since_spike:
                logger.debug(f"{self.name}: Spike trop ancien ({time_since_spike} barres) pour {self.symbol}")
                if signal_confidence < 0.5:
                    return False
                    
            # Validation cooldown entre spikes
            if last_spike_bars_ago is not None and last_spike_bars_ago < self.spike_cooldown_period:
                logger.debug(f"{self.name}: Spike trop récent ({last_spike_bars_ago} barres) - cooldown pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 8. Validation spike relatif au contexte
            if relative_spike_strength is not None and relative_spike_strength < self.min_relative_spike:
                logger.debug(f"{self.name}: Spike relatif insuffisant ({self._safe_format(relative_spike_strength, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 9. Validation activité inhabituelle
            if market_activity_level is not None and market_activity_level < 0.3:
                logger.debug(f"{self.name}: Activité marché faible ({self._safe_format(market_activity_level, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.5:
                    return False
                    
            # 10. Validation divergence prix/volume
            if price_volume_divergence is not None and abs(price_volume_divergence) > self.max_price_volume_divergence:
                logger.debug(f"{self.name}: Divergence prix/volume excessive ({self._safe_format(price_volume_divergence, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 11. Validation confirmation prix
            if spike_price_confirmation is not None and spike_price_confirmation < self.spike_confirmation_threshold:
                logger.debug(f"{self.name}: Confirmation prix insuffisante ({self._safe_format(spike_price_confirmation, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 12. Validation cohérence spike/tendance
            if volume_trend_coherence is not None and volume_trend_coherence < self.spike_vs_trend_coherence:
                logger.debug(f"{self.name}: Cohérence spike/tendance insuffisante ({self._safe_format(volume_trend_coherence, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 13. Validation spécifique selon stratégie
            strategy_spike_match = self._validate_strategy_spike_match(signal_strategy, volume_spike_multiplier)
            if not strategy_spike_match:
                logger.debug(f"{self.name}: Stratégie {signal_strategy} inadaptée aux spikes pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                        f"Spike: {self._safe_format(volume_spike_multiplier, '.2f')}x, "
                        f"Qualité: {self._safe_format(spike_quality_score, '.2f') if spike_quality_score is not None else 'N/A'}, "
                        f"Durée: {spike_duration_bars or 'N/A'} barres, "
                        f"Timing: {time_since_spike or 'N/A'} barres ago")
            
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
            spike_duration_bars = int(self.context.get('spike_duration_bars', 1)) if self.context.get('spike_duration_bars') is not None else 1
            spike_quality_score = float(self.context.get('spike_quality_score', 0.5)) if self.context.get('spike_quality_score') is not None else 0.5
            spike_consistency = float(self.context.get('spike_consistency', 0.5)) if self.context.get('spike_consistency') is not None else 0.5
            spike_legitimacy_score = float(self.context.get('spike_legitimacy_score', 0.5)) if self.context.get('spike_legitimacy_score') is not None else 0.5
            time_since_spike = int(self.context.get('time_since_spike', 5)) if self.context.get('time_since_spike') is not None else 5
            relative_spike_strength = float(self.context.get('relative_spike_strength', 1.0)) if self.context.get('relative_spike_strength') is not None else 1.0
            spike_price_confirmation = float(self.context.get('spike_price_confirmation', 0.5)) if self.context.get('spike_price_confirmation') is not None else 0.5
            unusual_activity_detected = self.context.get('unusual_activity_detected', False)
            
            signal_strategy = signal.get('strategy', '')
            
            base_score = 0.5  # Score de base si validé
            
            # Bonus multiplier spike
            if volume_spike_multiplier >= self.exceptional_spike_multiplier:
                base_score += self.exceptional_spike_bonus
            elif volume_spike_multiplier >= self.strong_spike_multiplier:
                base_score += 0.20
            elif volume_spike_multiplier >= self.min_spike_multiplier + 0.5:
                base_score += 0.10
                
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
            spike_duration_bars = int(self.context.get('spike_duration_bars', 0)) if self.context.get('spike_duration_bars') is not None else None
            spike_quality_score = float(self.context.get('spike_quality_score', 0.5)) if self.context.get('spike_quality_score') is not None else None
            time_since_spike = int(self.context.get('time_since_spike', 0)) if self.context.get('time_since_spike') is not None else None
            spike_legitimacy_score = float(self.context.get('spike_legitimacy_score', 0.5)) if self.context.get('spike_legitimacy_score') is not None else None
            
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
            
        # Au minimum, on a besoin d'un indicateur de spike
        spike_indicators = [
            'volume_spike_multiplier', 'current_volume_spike', 'spike_quality_score',
            'spike_duration_bars', 'time_since_spike', 'relative_spike_strength'
        ]
        
        available_indicators = sum(1 for ind in spike_indicators 
                                 if ind in self.context and self.context[ind] is not None)
        
        if available_indicators < 2:
            logger.warning(f"{self.name}: Pas assez d'indicateurs de spike pour {self.symbol}")
            return False
            
        return True
