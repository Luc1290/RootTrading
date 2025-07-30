"""
Regime_Strength_Validator - Validator basé sur la force des régimes de marché.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class Regime_Strength_Validator(BaseValidator):
    """
    Validator pour la force des régimes de marché - filtre selon la solidité et persistance des régimes.
    
    Vérifie: Force régime, persistance, transition, cohérence multi-indicateurs
    Catégorie: regime
    
    Rejette les signaux en:
    - Régimes faibles ou instables
    - Transitions de régimes en cours
    - Incohérence entre différents indicateurs de régime
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "Regime_Strength_Validator"
        self.category = "regime"
        
        # Paramètres force de régime
        self.min_regime_strength = 0.4           # Force minimum régime
        self.strong_regime_threshold = 0.7       # Régime considéré fort
        self.very_strong_regime_threshold = 0.85 # Régime très fort
        self.min_regime_confidence = 0.5         # Confidence minimum régime
        self.high_regime_confidence = 0.8        # Confidence élevée régime
        
        # Paramètres persistance et durée
        self.min_regime_duration = 5             # Durée minimum régime (barres)
        self.optimal_regime_duration = 20        # Durée optimale régime (barres)
        self.max_regime_duration = 150           # Durée maximum avant staleness
        self.stability_lookback = 10             # Lookback stabilité régime
        
        # Paramètres transitions
        self.transition_detection_threshold = 0.3  # Seuil détection transition
        self.max_transition_volatility = 0.6      # Volatilité max en transition
        self.regime_change_cooldown = 3           # Cooldown après changement
        
        # Régimes favorables/défavorables
        self.favorable_regimes = ["trending", "bullish", "bearish", "expansion", "momentum", "stable"]
        self.unfavorable_regimes = ["chaotic", "whipsaw", "noise", "compression", "uncertain"]
        self.neutral_regimes = ["ranging", "consolidation", "sideways", "neutral"]
        
        # Paramètres cohérence multi-indicateurs
        self.min_regime_consensus = 0.6          # Consensus minimum entre indicateurs
        self.regime_divergence_threshold = 0.4   # Seuil divergence régimes
        
        # Bonus/malus
        self.strong_regime_bonus = 0.25          # Bonus régime fort
        self.persistence_bonus = 0.20            # Bonus persistance
        self.consensus_bonus = 0.18              # Bonus consensus régimes
        self.transition_penalty = -0.30          # Pénalité transition
        self.weak_regime_penalty = -0.25         # Pénalité régime faible
        self.unfavorable_regime_penalty = -0.20  # Pénalité régime défavorable
    
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur la force et stabilité des régimes de marché.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon regime strength, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs de régime depuis le contexte
            try:
                # Régime principal et force
                current_regime = self.context.get('current_regime')
                regime_strength_raw = self.context.get('regime_strength')
                regime_strength = self._convert_regime_strength_to_score(regime_strength_raw) if regime_strength_raw else None
                regime_confidence = float(self.context.get('regime_confidence', 0)) if self.context.get('regime_confidence') is not None else None
                regime_persistence = float(self.context.get('regime_persistence', 0)) if self.context.get('regime_persistence') is not None else None
                
                # Durée et stabilité
                regime_duration_bars = int(self.context.get('regime_duration_bars', 0)) if self.context.get('regime_duration_bars') is not None else None
                regime_stability = float(self.context.get('regime_stability', 0)) if self.context.get('regime_stability') is not None else None
                last_regime_change_bars = int(self.context.get('last_regime_change_bars', 999)) if self.context.get('last_regime_change_bars') is not None else None
                
                # Transitions et changements
                regime_transition_probability = float(self.context.get('regime_transition_probability', 0)) if self.context.get('regime_transition_probability') is not None else None
                regime_in_transition = self.context.get('regime_in_transition', False)
                transition_direction = self.context.get('transition_direction')  # 'to_bullish', 'to_bearish', etc.
                
                # Régimes multiples (volatilité, trend, momentum)
                volatility_regime = self.context.get('volatility_regime')
                trend_regime = self.context.get('trend_regime')
                momentum_regime = self.context.get('momentum_regime')
                
                # Consensus et cohérence
                regime_consensus_score = float(self.context.get('regime_consensus_score', 0)) if self.context.get('regime_consensus_score') is not None else None
                regime_divergence_score = float(self.context.get('regime_divergence_score', 0)) if self.context.get('regime_divergence_score') is not None else None
                
                # Indicateurs de force du régime
                regime_momentum = float(self.context.get('regime_momentum', 0)) if self.context.get('regime_momentum') is not None else None
                regime_volatility_score = float(self.context.get('regime_volatility_score', 0.5)) if self.context.get('regime_volatility_score') is not None else None
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion indicateurs pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', 'Unknown')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side:
                logger.warning(f"{self.name}: Signal side manquant pour {self.symbol}")
                return False
                
            # 1. Validation force du régime principal
            if regime_strength is not None and regime_strength < self.min_regime_strength:
                logger.debug(f"{self.name}: Force régime insuffisante ({self._safe_format(regime_strength, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.8:
                    return False
                    
            # 2. Validation confidence du régime
            if regime_confidence is not None and regime_confidence < self.min_regime_confidence:
                logger.debug(f"{self.name}: Confidence régime insuffisante ({self._safe_format(regime_confidence, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 3. Validation type de régime
            if current_regime:
                if current_regime in self.unfavorable_regimes:
                    logger.debug(f"{self.name}: Régime défavorable ({current_regime}) pour {self.symbol}")
                    if signal_confidence < 0.9:  # Très strict pour régimes défavorables
                        return False
                elif current_regime in self.neutral_regimes:
                    logger.debug(f"{self.name}: Régime neutre ({current_regime}) pour {self.symbol}")
                    if signal_confidence < 0.6:
                        return False
                        
            # 4. Validation durée du régime
            if regime_duration_bars is not None:
                if regime_duration_bars < self.min_regime_duration:
                    logger.debug(f"{self.name}: Régime trop récent ({regime_duration_bars or 'N/A'} barres) pour {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                elif regime_duration_bars > self.max_regime_duration:
                    logger.debug(f"{self.name}: Régime trop ancien ({regime_duration_bars or 'N/A'} barres) pour {self.symbol}")
                    if signal_confidence < 0.5:
                        return False
                        
            # 5. Validation stabilité du régime
            if regime_stability is not None and regime_stability < 0.4:
                logger.debug(f"{self.name}: Régime instable ({self._safe_format(regime_stability, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 6. Validation transitions de régime
            if regime_in_transition:
                logger.debug(f"{self.name}: Régime en transition pour {self.symbol}")
                if signal_confidence < 0.8:  # Très prudent en transition
                    return False
                    
            if regime_transition_probability is not None and regime_transition_probability > self.transition_detection_threshold:
                logger.debug(f"{self.name}: Probabilité transition élevée ({self._safe_format(regime_transition_probability, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 7. Validation cooldown après changement de régime
            if last_regime_change_bars is not None and last_regime_change_bars < self.regime_change_cooldown:
                logger.debug(f"{self.name}: Changement régime récent ({last_regime_change_bars} barres) pour {self.symbol}")
                if signal_confidence < 0.8:
                    return False
                    
            # 8. Validation consensus entre régimes multiples
            regime_coherence = self._validate_regime_coherence(
                current_regime, volatility_regime, trend_regime, momentum_regime
            )
            if not regime_coherence:
                logger.debug(f"{self.name}: Incohérence entre régimes pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 9. Validation consensus score
            if regime_consensus_score is not None and regime_consensus_score < self.min_regime_consensus:
                logger.debug(f"{self.name}: Consensus régimes insuffisant ({self._safe_format(regime_consensus_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 10. Validation divergence entre régimes
            if regime_divergence_score is not None and regime_divergence_score > self.regime_divergence_threshold:
                logger.debug(f"{self.name}: Divergence régimes excessive ({self._safe_format(regime_divergence_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 11. Validation direction de transition vs signal
            if transition_direction:
                transition_coherence = self._validate_transition_coherence(signal_side, transition_direction)
                if not transition_coherence:
                    logger.debug(f"{self.name}: Transition {transition_direction} incohérente avec signal {signal_side} pour {self.symbol}")
                    if signal_confidence < 0.8:
                        return False
                        
            # 12. Validation persistance du régime
            if regime_persistence is not None and regime_persistence < 0.3:
                logger.debug(f"{self.name}: Persistance régime faible ({self._safe_format(regime_persistence, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 13. Validation cohérence stratégie/régime
            strategy_regime_match = self._validate_strategy_regime_match(signal_strategy, current_regime)
            if not strategy_regime_match:
                logger.debug(f"{self.name}: Stratégie {signal_strategy} inadaptée au régime {current_regime} pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 14. Validation momentum du régime
            if regime_momentum is not None:
                momentum_coherence = self._validate_momentum_coherence(signal_side, regime_momentum)
                if not momentum_coherence:
                    logger.debug(f"{self.name}: Momentum régime incohérent avec signal pour {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                        
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                        f"Régime: {current_regime or 'N/A'}, "
                        f"Force: {self._safe_format(regime_strength, '.2f') if regime_strength is not None else 'N/A'}, "
                        f"Durée: {regime_duration_bars if regime_duration_bars is not None else 'N/A'}b, "
                        f"Stabilité: {self._safe_format(regime_stability, '.2f') if regime_stability is not None else 'N/A'}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def _validate_regime_coherence(self, main_regime: str, vol_regime: str, 
                                  trend_regime: str, momentum_regime: str) -> bool:
        """Valide la cohérence entre différents types de régimes."""
        try:
            incoherences = 0
            
            # Vérifier cohérence main vs volatility
            if main_regime and vol_regime:
                if main_regime in ["trending", "momentum"] and vol_regime == "low":
                    incoherences += 1  # Trending avec volatilité faible = incohérent
                elif main_regime == "ranging" and vol_regime == "extreme":
                    incoherences += 1  # Ranging avec volatilité extrême = incohérent
                    
            # Vérifier cohérence main vs trend
            if main_regime and trend_regime:
                if main_regime == "ranging" and trend_regime in ["strong_bullish", "strong_bearish"]:
                    incoherences += 1  # Ranging avec trend fort = incohérent
                elif main_regime in ["bullish", "bearish"] and trend_regime == "sideways":
                    incoherences += 1  # Directionnel avec trend sideways = incohérent
                    
            # Vérifier cohérence trend vs momentum
            if trend_regime and momentum_regime:
                if trend_regime in ["bullish", "strong_bullish"] and momentum_regime == "bearish":
                    incoherences += 1  # Trend haussier + momentum baissier = incohérent
                elif trend_regime in ["bearish", "strong_bearish"] and momentum_regime == "bullish":
                    incoherences += 1  # Trend baissier + momentum haussier = incohérent
                    
            # Accepter jusqu'à 1 incohérence mineure
            return incoherences <= 1
            
        except Exception:
            return True  # En cas d'erreur, ne pas bloquer
            
    def _validate_transition_coherence(self, signal_side: str, transition_direction: str) -> bool:
        """Valide la cohérence entre direction signal et transition de régime."""
        try:
            if not transition_direction:
                return True
                
            # Extraire direction de transition
            if "bullish" in transition_direction.lower() or "up" in transition_direction.lower():
                transition_bias = "bullish"
            elif "bearish" in transition_direction.lower() or "down" in transition_direction.lower():
                transition_bias = "bearish"
            else:
                return True  # Transition neutre, accepter
                
            # Vérifier cohérence
            if signal_side == "BUY" and transition_bias == "bearish":
                return False  # BUY pendant transition bearish
            elif signal_side == "SELL" and transition_bias == "bullish":
                return False  # SELL pendant transition bullish
                
            return True
            
        except Exception:
            return True
            
    def _validate_strategy_regime_match(self, strategy: str, regime: str) -> bool:
        """Valide l'adéquation stratégie/régime."""
        try:
            if not strategy or not regime:
                return True
                
            strategy_lower = strategy.lower()
            regime_lower = regime.lower()
            
            # Stratégies trend following
            if any(kw in strategy_lower for kw in ['trend', 'macd', 'ema', 'cross', 'slope']):
                # Fonctionnent mieux en régimes trending
                if regime_lower in ["ranging", "sideways", "consolidation"]:
                    return False
                    
            # Stratégies mean reversion
            elif any(kw in strategy_lower for kw in ['bollinger', 'rsi', 'reversal', 'touch']):
                # Fonctionnent mieux en régimes ranging
                if regime_lower in ["trending", "momentum", "expansion"]:
                    return False
                    
            # Stratégies breakout
            elif any(kw in strategy_lower for kw in ['breakout', 'donchian', 'atr']):
                # Fonctionnent mieux en régimes de compression vers expansion
                if regime_lower in ["chaotic", "whipsaw", "noise"]:
                    return False
                    
            return True
            
        except Exception:
            return True
            
    def _validate_momentum_coherence(self, signal_side: str, regime_momentum: float) -> bool:
        """Valide la cohérence entre direction signal et momentum de régime."""
        try:
            if regime_momentum is None:
                return True
                
            # Seuils momentum
            strong_positive = 0.3
            strong_negative = -0.3
            
            # Vérifier cohérence
            if signal_side == "BUY" and regime_momentum < strong_negative:
                return False  # BUY avec momentum très négatif
            elif signal_side == "SELL" and regime_momentum > strong_positive:
                return False  # SELL avec momentum très positif
                
            return True
            
        except Exception:
            return True
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur la force des régimes.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur régimes
            regime_strength_raw = self.context.get('regime_strength')
            regime_strength = self._convert_regime_strength_to_score(regime_strength_raw) if regime_strength_raw else 0.5
            regime_confidence = float(self.context.get('regime_confidence', 0.5)) if self.context.get('regime_confidence') is not None else 0.5
            regime_persistence = float(self.context.get('regime_persistence', 0.5)) if self.context.get('regime_persistence') is not None else 0.5
            regime_stability = float(self.context.get('regime_stability', 0.5)) if self.context.get('regime_stability') is not None else 0.5
            regime_duration_bars = int(self.context.get('regime_duration_bars', 10)) if self.context.get('regime_duration_bars') is not None else 10
            regime_consensus_score = float(self.context.get('regime_consensus_score', 0.6)) if self.context.get('regime_consensus_score') is not None else 0.6
            regime_transition_probability = float(self.context.get('regime_transition_probability', 0.2)) if self.context.get('regime_transition_probability') is not None else 0.2
            last_regime_change_bars = int(self.context.get('last_regime_change_bars', 10)) if self.context.get('last_regime_change_bars') is not None else 10
            
            current_regime = self.context.get('current_regime', 'neutral')
            signal_strategy = signal.get('strategy', '')
            
            base_score = 0.5  # Score de base si validé
            
            # Bonus force du régime
            if regime_strength >= self.very_strong_regime_threshold:
                base_score += self.strong_regime_bonus + 0.05  # Bonus supplémentaire très fort
            elif regime_strength >= self.strong_regime_threshold:
                base_score += self.strong_regime_bonus
            elif regime_strength >= self.min_regime_strength + 0.2:
                base_score += 0.15
                
            # Bonus confidence du régime
            if regime_confidence >= self.high_regime_confidence:
                base_score += 0.15  # Confidence très élevée
            elif regime_confidence >= self.min_regime_confidence + 0.2:
                base_score += 0.10
                
            # Bonus persistance
            if regime_persistence >= 0.8:
                base_score += self.persistence_bonus
            elif regime_persistence >= 0.6:
                base_score += 0.12
                
            # Bonus stabilité
            if regime_stability >= 0.8:
                base_score += 0.15  # Régime très stable
            elif regime_stability >= 0.6:
                base_score += 0.10  # Régime stable
                
            # Bonus durée optimale
            if self.optimal_regime_duration * 0.5 <= regime_duration_bars <= self.optimal_regime_duration * 2:
                base_score += 0.12  # Durée dans la zone optimale
            elif self.min_regime_duration <= regime_duration_bars <= self.max_regime_duration:
                base_score += 0.08  # Durée acceptable
                
            # Bonus type de régime
            if current_regime in self.favorable_regimes:
                base_score += 0.15  # Régime favorable
            elif current_regime in self.neutral_regimes:
                base_score += 0.05  # Régime neutre
            else:
                # Régime défavorable déjà pénalisé dans validation
                pass
                
            # Bonus consensus élevé
            if regime_consensus_score >= 0.8:
                base_score += self.consensus_bonus
            elif regime_consensus_score >= 0.7:
                base_score += 0.12
                
            # Malus transition récente ou probable
            if regime_transition_probability > self.transition_detection_threshold:
                base_score += self.transition_penalty * (regime_transition_probability - self.transition_detection_threshold)
                
            # Bonus stabilité post-changement
            if last_regime_change_bars >= self.regime_change_cooldown * 2:
                base_score += 0.08  # Régime bien établi
                
            # Bonus adéquation stratégie/régime
            if self._validate_strategy_regime_match(signal_strategy, current_regime):
                base_score += 0.10  # Stratégie adaptée au régime
                
            # Validation régimes multiples
            vol_regime = self.context.get('volatility_regime')
            trend_regime = self.context.get('trend_regime')
            momentum_regime = self.context.get('momentum_regime')
            
            if self._validate_regime_coherence(current_regime, vol_regime, trend_regime, momentum_regime):
                base_score += 0.08  # Cohérence entre régimes
                
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
            
            current_regime = self.context.get('current_regime', 'N/A')
            regime_strength_raw = self.context.get('regime_strength')
            regime_strength = self._convert_regime_strength_to_score(regime_strength_raw) if regime_strength_raw else None
            regime_confidence = float(self.context.get('regime_confidence', 0)) if self.context.get('regime_confidence') is not None else None
            regime_duration_bars = int(self.context.get('regime_duration_bars', 0)) if self.context.get('regime_duration_bars') is not None else None
            regime_transition_probability = float(self.context.get('regime_transition_probability', 0)) if self.context.get('regime_transition_probability') is not None else None
            regime_in_transition = self.context.get('regime_in_transition', False)
            
            if is_valid:
                reason = f"Régime favorable"
                if current_regime != 'N/A':
                    reason += f" (type: {current_regime})"
                if regime_strength is not None:
                    reason += f", force: {self._safe_format(regime_strength, '.2f')}"
                if regime_confidence is not None:
                    reason += f", confidence: {self._safe_format(regime_confidence, '.2f')}"
                if regime_duration_bars is not None:
                    reason += f", durée: {regime_duration_bars}b"
                    
                strategy_match = self._validate_strategy_regime_match(signal_strategy, current_regime)
                if strategy_match:
                    reason += " - stratégie adaptée"
                    
                return f"{self.name}: Validé - {reason} pour {signal_strategy} {signal_side}"
            else:
                if current_regime in self.unfavorable_regimes:
                    return f"{self.name}: Rejeté - Régime défavorable ({current_regime})"
                elif regime_strength is not None and regime_strength < self.min_regime_strength:
                    return f"{self.name}: Rejeté - Force régime insuffisante ({self._safe_format(regime_strength, '.2f')})"
                elif regime_confidence is not None and regime_confidence < self.min_regime_confidence:
                    return f"{self.name}: Rejeté - Confidence régime insuffisante ({self._safe_format(regime_confidence, '.2f')})"
                elif regime_in_transition:
                    return f"{self.name}: Rejeté - Régime en transition"
                elif regime_transition_probability is not None and regime_transition_probability > self.transition_detection_threshold:
                    return f"{self.name}: Rejeté - Probabilité transition élevée ({self._safe_format(regime_transition_probability, '.2f')})"
                elif regime_duration_bars is not None and regime_duration_bars < self.min_regime_duration:
                    return f"{self.name}: Rejeté - Régime trop récent ({regime_duration_bars or 'N/A'} barres)"
                    
                return f"{self.name}: Rejeté - Critères force régime non respectés"
                
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données de régime requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Au minimum, on a besoin d'un indicateur de régime
        regime_indicators = [
            'current_regime', 'regime_strength', 'regime_confidence', 
            'regime_duration_bars', 'regime_stability'
        ]
        
        available_indicators = sum(1 for ind in regime_indicators 
                                 if ind in self.context and self.context[ind] is not None)
        
        if available_indicators < 2:
            logger.warning(f"{self.name}: Pas assez d'indicateurs de régime pour {self.symbol}")
            return False
            
        return True
