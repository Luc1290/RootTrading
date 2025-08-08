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
        
        # Paramètres force de régime - AJUSTÉS POUR RANGING
        self.min_regime_strength = 0.35          # Force minimum régime RÉDUITE pour crypto ranging (35% au lieu de 55%)
        self.strong_regime_threshold = 0.60      # Régime considéré fort RÉDUIT (60% au lieu de 75%)
        self.very_strong_regime_threshold = 0.75 # Régime très fort RÉDUIT (75% au lieu de 88%)
        self.min_regime_confidence = 45.0        # Confidence minimum régime RÉDUITE (45% au lieu de 65%)
        self.high_regime_confidence = 65.0       # Confidence élevée régime RÉDUITE (65% au lieu de 85%)
        
        # Paramètres persistance et durée - AJUSTÉS POUR RANGING
        self.min_regime_duration = 3             # Durée minimum régime RÉDUITE pour crypto rapide (3 au lieu de 8 barres)
        self.optimal_regime_duration = 15        # Durée optimale régime RÉDUITE (15 au lieu de 25 barres)
        self.max_regime_duration = 100           # Durée maximum RÉDUITE (100 au lieu de 120 barres)
        self.stability_lookback = 8              # Lookback stabilité régime RÉDUIT (8 au lieu de 12)
        
        # Paramètres transitions - OPTIMISÉS
        self.transition_detection_threshold = 0.25  # Seuil détection transition PLUS STRICT (25% au lieu de 30%)
        self.max_transition_volatility = 0.55      # Volatilité max en transition PLUS STRICTE (55% au lieu de 60%)
        self.regime_change_cooldown = 5           # Cooldown après changement AUGMENTÉ (5 au lieu de 3 barres)
        
        # Régimes favorables/défavorables - AJUSTÉS POUR CRYPTO RANGING
        self.favorable_regimes = ["TRENDING_BULL", "TRENDING_BEAR", "BREAKOUT_BULL", "BREAKOUT_BEAR", "RANGING"]  # RANGING ajouté aux favorables
        self.unfavorable_regimes = ["VOLATILE", "UNKNOWN", "EXTREME"]
        self.neutral_regimes = ["TRANSITION", "COMPRESSION"]
        
        # Paramètres cohérence multi-indicateurs - AJUSTÉS POUR RANGING
        self.min_regime_consensus = 50.0         # Consensus minimum RÉDUIT pour crypto ranging (50% au lieu de 70%)
        self.regime_divergence_threshold = 0.50   # Seuil divergence régimes PLUS PERMISSIF (50% au lieu de 35%)
        
        # Bonus/malus
        self.strong_regime_bonus = 0.25          # Bonus régime fort
        self.persistence_bonus = 0.20          # Bonus persistance
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
                # current_regime → market_regime
                current_regime = self.context.get('market_regime')
                regime_strength_raw = self.context.get('regime_strength')
                regime_strength = self._convert_regime_strength_to_score(regime_strength_raw) if regime_strength_raw else None
                # regime_confidence existe déjà !
                regime_confidence = float(self.context.get('regime_confidence', 50.0)) if self.context.get('regime_confidence') is not None else None
                # regime_persistence → regime_strength (même concept)
                regime_persistence = float(self.context.get('regime_strength', 50.0)) if self.context.get('regime_strength') is not None else None
                
                # Durée et stabilité
                # regime_duration_bars → regime_duration (existe déjà!)
                regime_duration_bars = int(self.context.get('regime_duration', 0)) if self.context.get('regime_duration') is not None else None
                # regime_stability → regime_strength
                regime_stability = float(self.context.get('regime_strength', 50.0)) if self.context.get('regime_strength') is not None else None
                # last_regime_change_bars → regime_duration
                last_regime_change_bars = int(self.context.get('regime_duration', 999)) if self.context.get('regime_duration') is not None else None
                
                # Transitions et changements
                # regime_transition_probability → pattern_confidence
                regime_transition_probability = float(self.context.get('pattern_confidence', 0)) if self.context.get('pattern_confidence') is not None else None
                regime_in_transition = self.context.get('regime_in_transition', False)
                # transition_direction → directional_bias
                transition_direction = self.context.get('directional_bias')  # 'to_bullish', 'to_bearish', etc.
                
                # Régimes multiples (volatilité, trend, momentum)
                volatility_regime = self.context.get('volatility_regime')
                # trend_regime → market_regime (utiliser le régime principal)
                trend_regime = self.context.get('market_regime')
                # momentum_regime → momentum_score
                momentum_regime = self.context.get('momentum_score')
                
                # Consensus et cohérence
                # regime_consensus_score → confluence_score
                regime_consensus_score = float(self.context.get('confluence_score', 60.0)) if self.context.get('confluence_score') is not None else None
                # regime_divergence_score → signal_strength
                regime_divergence_score = float(self.context.get('signal_strength', 0)) if self.context.get('signal_strength') is not None else None
                
                # Indicateurs de force du régime
                # regime_momentum → momentum_score
                regime_momentum = float(self.context.get('momentum_score', 0)) if self.context.get('momentum_score') is not None else None
                # regime_volatility_score → volatility_regime (convertir en score)
                volatility_raw = self.context.get('volatility_regime', 'normal')
                if isinstance(volatility_raw, str):
                    regime_volatility_score = self._convert_volatility_regime_to_score(volatility_raw)
                else:
                    regime_volatility_score = float(volatility_raw) if volatility_raw is not None else 0.5
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion indicateurs pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', 'Unknown')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side:
                logger.warning(f"{self.name}: Signal side manquant pour {self.symbol}")
                return False
                
            # 1. Validation force du régime principal - AJUSTÉ POUR RANGING
            if regime_strength is not None and regime_strength < self.min_regime_strength:
                logger.debug(f"{self.name}: Force régime insuffisante ({self._safe_format(regime_strength, '.2f')}) pour {self.symbol}")
                # Plus permissif en régime ranging/crypto
                if signal_confidence < 0.50:  # RÉDUIT pour crypto (50% au lieu de 75%)
                    return False
                    
            # 2. Validation confidence du régime - AJUSTÉ POUR RANGING
            if regime_confidence is not None and regime_confidence < self.min_regime_confidence:
                logger.debug(f"{self.name}: Confidence régime insuffisante ({self._safe_format(regime_confidence, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.45:  # RÉDUIT pour crypto ranging (45% au lieu de 75%)
                    return False
                    
            # 3. Validation type de régime - AJUSTÉ POUR RANGING
            if current_regime:
                if current_regime in self.unfavorable_regimes:
                    logger.debug(f"{self.name}: Régime défavorable ({current_regime}) pour {self.symbol}")
                    if signal_confidence < 0.60:  # RÉDUIT pour crypto (60% au lieu de 82%)
                        return False
                elif current_regime in self.neutral_regimes:
                    logger.debug(f"{self.name}: Régime neutre ({current_regime}) pour {self.symbol}")
                    if signal_confidence < 0.45:  # RÉDUIT pour crypto (45% au lieu de 70%)
                        return False
                        
            # 4. Validation durée du régime - AJUSTÉ POUR RANGING
            if regime_duration_bars is not None:
                if regime_duration_bars < self.min_regime_duration:
                    logger.debug(f"{self.name}: Régime trop récent ({regime_duration_bars or 'N/A'} barres) pour {self.symbol}")
                    if signal_confidence < 0.45:  # RÉDUIT pour crypto rapide (45% au lieu de 68%)
                        return False
                elif regime_duration_bars > self.max_regime_duration:
                    logger.debug(f"{self.name}: Régime trop ancien ({regime_duration_bars or 'N/A'} barres) pour {self.symbol}")
                    if signal_confidence < 0.50:  # LÉGÈREMENT RÉDUIT (50% au lieu de 60%)
                        return False
                        
            # 5. Validation stabilité du régime - AJUSTÉ POUR RANGING
            if regime_stability is not None and regime_stability < 35.0:  # RÉDUIT pour crypto volatil (35% au lieu de 50%)
                logger.debug(f"{self.name}: Régime très instable ({self._safe_format(regime_stability, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.50:  # RÉDUIT pour crypto (50% au lieu de 75%)
                    return False
                    
            # 6. Validation transitions de régime - AJUSTÉ POUR RANGING
            if regime_in_transition:
                logger.debug(f"{self.name}: Régime en transition pour {self.symbol}")
                # Transitions fréquentes en crypto, plus permissif
                if signal_confidence < 0.50:  # RÉDUIT pour crypto (50% au lieu de 75%)
                    return False
                    
            if regime_transition_probability is not None and regime_transition_probability > self.transition_detection_threshold:
                logger.debug(f"{self.name}: Probabilité transition élevée ({self._safe_format(regime_transition_probability, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.45:  # RÉDUIT pour crypto (45% au lieu de 75%)
                    return False
                    
            # 7. Validation cooldown après changement de régime - AJUSTÉ POUR RANGING
            if last_regime_change_bars is not None and last_regime_change_bars < self.regime_change_cooldown:
                logger.debug(f"{self.name}: Changement régime récent ({last_regime_change_bars} barres) pour {self.symbol}")
                # Crypto change rapidement, plus permissif
                if signal_confidence < 0.40:  # RÉDUIT pour crypto rapide (40% au lieu de 73%)
                    return False
                    
            # 8. Validation consensus entre régimes multiples - PLUS STRICT
            if (current_regime and volatility_regime and trend_regime and momentum_regime):
                regime_coherence = self._validate_regime_coherence(
                    current_regime, volatility_regime, trend_regime, momentum_regime
                )
            else:
                regime_coherence = True
            if not regime_coherence:
                logger.debug(f"{self.name}: Incohérence entre régimes pour {self.symbol}")
                if signal_confidence < 0.50:  # RÉDUIT pour crypto (50% au lieu de 78%)
                    return False
                    
            # 9. Validation consensus score - AJUSTÉ POUR RANGING
            if regime_consensus_score is not None and regime_consensus_score < self.min_regime_consensus:
                logger.debug(f"{self.name}: Consensus régimes insuffisant ({self._safe_format(regime_consensus_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.45:  # RÉDUIT pour crypto (45% au lieu de 68%)
                    return False
                    
            # 10. Validation divergence entre régimes - AJUSTÉ POUR RANGING
            if regime_divergence_score is not None and regime_divergence_score > self.regime_divergence_threshold:
                logger.debug(f"{self.name}: Divergence régimes acceptable en crypto ({self._safe_format(regime_divergence_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.50:  # RÉDUIT pour crypto (50% au lieu de 75%)
                    return False
                    
            # 11. Validation direction de transition vs signal - PLUS STRICT
            if transition_direction:
                transition_coherence = self._validate_transition_coherence(signal_side, transition_direction)
                if not transition_coherence:
                    logger.debug(f"{self.name}: Transition {transition_direction} incohérente avec signal {signal_side} pour {self.symbol}")
                    if signal_confidence < 0.83:  # AUGMENTÉ de 80% à 83%
                        return False
                        
            # 12. Validation persistance du régime - PLUS STRICT
            if regime_persistence is not None and regime_persistence < 40.0:  # AUGMENTÉ de 30% à 40%
                logger.debug(f"{self.name}: Persistance régime faible ({self._safe_format(regime_persistence, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.68:  # AUGMENTÉ de 60% à 68%
                    return False
                    
            # 13. Validation cohérence stratégie/régime - PLUS STRICT
            if current_regime:
                strategy_regime_match = self._validate_strategy_regime_match(signal_strategy, current_regime)
            else:
                strategy_regime_match = True
            if not strategy_regime_match:
                logger.debug(f"{self.name}: Stratégie {signal_strategy} inadaptée au régime {current_regime} pour {self.symbol}")
                if signal_confidence < 0.68:  # AUGMENTÉ de 60% à 68%
                    return False
                    
            # 14. Validation momentum du régime - PLUS STRICT
            if regime_momentum is not None:
                momentum_coherence = self._validate_momentum_coherence(signal_side, regime_momentum)
                if not momentum_coherence:
                    logger.debug(f"{self.name}: Momentum régime incohérent avec signal pour {self.symbol}")
                    if signal_confidence < 0.75:  # AUGMENTÉ de 70% à 75%
                        return False
                        
            # NOUVEAU: Validation finale - rejet des signaux moyennement confiants avec régime fragile
            overall_regime_quality = self._calculate_overall_regime_quality(
                regime_strength, regime_confidence, regime_persistence, regime_stability
            )
            if overall_regime_quality < 0.6 and signal_confidence < 0.75:
                logger.debug(f"{self.name}: Régime fragile ({overall_regime_quality:.2f}) + signal confidence insuffisante pour {self.symbol}")
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
                
            # Extraire direction de transition (format: BULLISH/BEARISH/NEUTRAL)
            direction_upper = str(transition_direction).upper()
            if direction_upper == "BULLISH":
                transition_bias = "bullish"
            elif direction_upper == "BEARISH":
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
                
            # Seuils momentum (format 0-100, 50=neutre)
            strong_positive = 70  # Momentum haussier fort
            strong_negative = 30  # Momentum baissier fort
            
            # Vérifier cohérence
            if signal_side == "BUY" and regime_momentum < strong_negative:
                return False  # BUY avec momentum très baissier
            elif signal_side == "SELL" and regime_momentum > strong_positive:
                return False  # SELL avec momentum très haussier
                
            return True
            
        except Exception:
            return True
            
    def _get_directional_regime_multiplier(self, signal_side: str, current_regime: str) -> float:
        """Calcule un multiplicateur de bonus selon la cohérence signal/régime."""
        try:
            if not signal_side or not current_regime:
                return 1.0
                
            regime_lower = current_regime.lower()
            
            # Régimes directionnels favorables
            if signal_side == "BUY":
                if regime_lower in ["bullish", "trending", "momentum", "expansion"]:
                    return 1.3  # Bonus 30% pour BUY dans régimes haussiers
                elif regime_lower in ["bearish"]:
                    return 0.6  # Malus 40% pour BUY dans régime bearish
                elif regime_lower in ["ranging", "consolidation", "neutral"]:
                    return 0.9  # Léger malus pour BUY en range
                    
            elif signal_side == "SELL":
                if regime_lower in ["bearish", "trending", "momentum", "expansion"]:
                    return 1.3  # Bonus 30% pour SELL dans régimes baissiers
                elif regime_lower in ["bullish"]:
                    return 0.6  # Malus 40% pour SELL dans régime bullish
                elif regime_lower in ["ranging", "consolidation", "neutral"]:
                    return 0.9  # Léger malus pour SELL en range
                    
            return 1.0  # Neutre par défaut
            
        except Exception:
            return 1.0
            
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
            regime_confidence = float(self.context.get('regime_confidence', 50.0)) if self.context.get('regime_confidence') is not None else 50.0
            # regime_persistence → regime_strength (même concept)
            regime_persistence = float(self.context.get('regime_strength', 50.0)) if self.context.get('regime_strength') is not None else 50.0
            # regime_stability → regime_strength
            regime_stability = float(self.context.get('regime_strength', 50.0)) if self.context.get('regime_strength') is not None else 50.0
            # regime_duration_bars → regime_duration (existe déjà!)
            regime_duration_bars = int(self.context.get('regime_duration', 10)) if self.context.get('regime_duration') is not None else 10
            # regime_consensus_score → confluence_score
            regime_consensus_score = float(self.context.get('confluence_score', 60.0)) if self.context.get('confluence_score') is not None else 60.0
            # regime_transition_probability → pattern_confidence
            regime_transition_probability = float(self.context.get('pattern_confidence', 0.2)) if self.context.get('pattern_confidence') is not None else 0.2
            # last_regime_change_bars → regime_duration
            last_regime_change_bars = int(self.context.get('regime_duration', 10)) if self.context.get('regime_duration') is not None else 10
            
            current_regime = self.context.get('market_regime', 'RANGING')
            signal_strategy = signal.get('strategy', '')
            signal_side = signal.get('side')
            
            base_score = 0.48  # Score de base réduit (48% au lieu de 50%)
            
            # CORRECTION: Bonus force du régime avec logique directionnelle
            force_bonus_multiplier = self._get_directional_regime_multiplier(signal_side, current_regime)
            
            if regime_strength >= self.very_strong_regime_threshold:
                base_score += (self.strong_regime_bonus + 0.05) * force_bonus_multiplier
            elif regime_strength >= self.strong_regime_threshold:
                base_score += self.strong_regime_bonus * force_bonus_multiplier
            elif regime_strength >= self.min_regime_strength + 0.2:
                base_score += 0.15 * force_bonus_multiplier
                
            # Bonus confidence du régime (réduit si incohérence directionnelle)
            confidence_bonus_multiplier = 1.0 if force_bonus_multiplier >= 1.0 else 0.7
            if regime_confidence >= self.high_regime_confidence:
                base_score += 0.15 * confidence_bonus_multiplier
            elif regime_confidence >= self.min_regime_confidence + 0.2:
                base_score += 0.10 * confidence_bonus_multiplier
                
            # Bonus persistance (avec cohérence directionnelle)
            if regime_persistence >= 80.0:
                base_score += self.persistence_bonus * force_bonus_multiplier
            elif regime_persistence >= 60.0:
                base_score += 0.12 * force_bonus_multiplier
                
            # Bonus stabilité
            if regime_stability >= 80.0:
                base_score += 0.15  # Régime très stable
            elif regime_stability >= 60.0:
                base_score += 0.10  # Régime stable
                
            # Bonus durée optimale
            if self.optimal_regime_duration * 0.5 <= regime_duration_bars <= self.optimal_regime_duration * 2:
                base_score += 0.12  # Durée dans la zone optimale
            elif self.min_regime_duration <= regime_duration_bars <= self.max_regime_duration:
                base_score += 0.08  # Durée acceptable
                
            # CORRECTION: Bonus type de régime avec logique directionnelle (format: market_regime)
            if current_regime in self.favorable_regimes:
                # Bonus additionnel pour cohérence directionnelle
                if signal_side == "BUY" and current_regime in ["TRENDING_BULL", "BREAKOUT_BULL"]:
                    base_score += 0.20  # Excellent pour BUY
                elif signal_side == "SELL" and current_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]:
                    base_score += 0.20  # Excellent pour SELL  
                else:
                    base_score += 0.15  # Régime favorable général
            elif current_regime in self.neutral_regimes:
                base_score += 0.05  # Régime neutre
            elif current_regime == "TRENDING_BULL":
                # Régime trending bull : favoriser BUY, pénaliser SELL
                if signal_side == "BUY":
                    base_score += 0.15  # Bonus BUY en régime trending bull
                elif signal_side == "SELL":
                    base_score -= 0.05  # Malus SELL en régime trending bull
            elif current_regime == "TRENDING_BEAR":
                # Régime trending bear : favoriser SELL, pénaliser BUY
                if signal_side == "SELL":
                    base_score += 0.15  # Bonus SELL en régime trending bear
                elif signal_side == "BUY":
                    base_score -= 0.05  # Malus BUY en régime trending bear
                
            # Bonus consensus élevé
            if regime_consensus_score >= 80.0:
                base_score += self.consensus_bonus
            elif regime_consensus_score >= 70.0:
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
                
            # CORRECTION: Bonus momentum du régime avec logique directionnelle
            regime_momentum = self.context.get('momentum_score')
            if regime_momentum is not None:
                momentum_coherence = self._validate_momentum_coherence(signal_side, regime_momentum)
                if momentum_coherence:
                    # Bonus pour cohérence momentum + direction (format 0-100)
                    if signal_side == "BUY" and regime_momentum > 60:
                        base_score += 0.10  # BUY avec momentum haussier
                    elif signal_side == "SELL" and regime_momentum < 40:
                        base_score += 0.10  # SELL avec momentum baissier
                else:
                    # Malus pour incohérence momentum
                    base_score -= 0.08
                
            # CORRECTION: Validation transitions avec bonus directionnel (format: BULLISH/BEARISH/NEUTRAL)
            transition_direction = self.context.get('directional_bias')
            if transition_direction and signal_side:
                transition_coherence = self._validate_transition_coherence(signal_side, transition_direction)
                if transition_coherence:
                    # Bonus pour transition cohérente
                    direction_upper = str(transition_direction).upper()
                    if signal_side == "BUY" and direction_upper == "BULLISH":
                        base_score += 0.08  # BUY avec transition bullish
                    elif signal_side == "SELL" and direction_upper == "BEARISH":
                        base_score += 0.08  # SELL avec transition bearish
                
            # Validation régimes multiples
            vol_regime = self.context.get('volatility_regime')
            # trend_regime → market_regime (utiliser le régime principal)
            trend_regime = self.context.get('market_regime')
            # momentum_regime → momentum_score
            momentum_regime = self.context.get('momentum_score')
            
            if (current_regime and vol_regime and trend_regime and momentum_regime and 
                self._validate_regime_coherence(current_regime, vol_regime, trend_regime, momentum_regime)):
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
            
            current_regime = self.context.get('market_regime', 'N/A')
            regime_strength_raw = self.context.get('regime_strength')
            regime_strength = self._convert_regime_strength_to_score(regime_strength_raw) if regime_strength_raw else None
            regime_confidence = float(self.context.get('regime_confidence', 50.0)) if self.context.get('regime_confidence') is not None else None
            # regime_duration_bars → regime_duration (existe déjà!)
            regime_duration_bars = int(self.context.get('regime_duration', 0)) if self.context.get('regime_duration') is not None else None
            # regime_transition_probability → pattern_confidence
            regime_transition_probability = float(self.context.get('pattern_confidence', 0)) if self.context.get('pattern_confidence') is not None else None
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
            
    def _calculate_overall_regime_quality(self, regime_strength: float, regime_confidence: float,
                                          regime_persistence: float, regime_stability: float) -> float:
        """Calcule la qualité globale du régime."""
        try:
            quality_score = 0.0
            components = 0
            
            if regime_strength is not None:
                quality_score += regime_strength
                components += 1
                
            if regime_confidence is not None:
                quality_score += (regime_confidence / 100)  # Normaliser 0-100 vers 0-1
                components += 1
                
            if regime_persistence is not None:
                quality_score += (regime_persistence / 100)  # Normaliser 0-100 vers 0-1
                components += 1
                
            if regime_stability is not None:
                quality_score += (regime_stability / 100)  # Normaliser 0-100 vers 0-1
                components += 1
                
            return quality_score / max(1, components)  # Moyenne des composants disponibles
        except:
            return 0.5  # Valeur par défaut
            
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
            'market_regime', 'regime_strength', 'regime_confidence', 
            'regime_duration', 'regime_strength'
        ]
        
        available_indicators = sum(1 for ind in regime_indicators 
                                 if ind in self.context and self.context[ind] is not None)
        
        if available_indicators < 2:
            logger.warning(f"{self.name}: Pas assez d'indicateurs de régime pour {self.symbol}")
            return False
            
        return True
            
    def _convert_regime_strength_to_score(self, regime_strength_raw) -> float:
        """Convertit une valeur de force de régime en score 0-1."""
        try:
            if regime_strength_raw is None:
                return 0.5
                
            if isinstance(regime_strength_raw, str):
                strength_lower = regime_strength_raw.lower()
                if strength_lower in ['very_strong', 'strong', 'high']:
                    return 0.8
                elif strength_lower in ['moderate', 'medium', 'normal']:
                    return 0.6
                elif strength_lower in ['weak', 'low']:
                    return 0.3
                elif strength_lower in ['very_weak', 'very_low']:
                    return 0.1
                else:
                    return 0.5
            else:
                # Essayer de convertir directement
                score = float(regime_strength_raw)
                if score > 10:  # Format 0-100
                    return score / 100.0
                else:  # Format 0-1
                    return score
                    
        except (ValueError, TypeError):
            return 0.5
            
    def _convert_volatility_regime_to_score(self, volatility_regime: str) -> float:
        """Convertit un régime de volatilité en score 0-1."""
        try:
            if not volatility_regime:
                return 0.5
                
            vol_lower = volatility_regime.lower()
            
            if vol_lower in ['extreme', 'very_high']:
                return 0.9
            elif vol_lower in ['high', 'elevated']:
                return 0.7
            elif vol_lower in ['normal', 'medium', 'moderate']:
                return 0.5
            elif vol_lower in ['low', 'calm']:
                return 0.3
            elif vol_lower in ['very_low', 'dormant']:
                return 0.1
            else:
                return 0.5
                
        except Exception:
            return 0.5
