"""
Trend_Alignment_Validator - Validator basé sur l'alignement des tendances multi-timeframes.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class Trend_Alignment_Validator(BaseValidator):
    """
    Validator pour l'alignement des tendances - filtre selon la cohérence des trends sur différents timeframes.
    
    Vérifie: Alignement EMA, direction MACD, force trend, consensus multi-TF
    Catégorie: trend
    
    Rejette les signaux en:
    - Tendances contradictoires entre timeframes
    - Trend faible ou en transition
    - Manque de consensus directionnel
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "Trend_Alignment_Validator"
        self.category = "trend"
        
        # Paramètres alignement tendance DURCIS
        self.min_trend_strength = 0.55        # Force minimum tendance relevé (55%)
        self.strong_trend_threshold = 0.75    # Tendance considérée forte relevé (75%)
        self.min_trend_consensus = 0.70       # Consensus minimum entre TF relevé (70%)
        self.min_aligned_timeframes = 3       # Minimum timeframes alignés relevé (3)
        
        # Paramètres EMA alignment DURCIS
        self.min_ema_separation = 0.008       # 0.8% séparation minimum EMA relevé
        self.optimal_ema_separation = 0.025   # 2.5% séparation optimale EMA relevé
        self.ema_slope_threshold = 0.002      # Seuil pente EMA significative relevé
        
        # Paramètres MACD alignment DURCIS
        self.min_macd_histogram_strength = 0.15 # Force minimum histogramme MACD relevé
        self.macd_signal_coherence_threshold = 0.85 # Cohérence MACD/signal relevé
        
        # Paramètres multi-timeframe DURCIS
        self.timeframe_weights = {            # Poids par timeframe
            '1m': 0.1, '5m': 0.15, '15m': 0.2, '1h': 0.25, '4h': 0.2, '1d': 0.1
        }
        self.min_weighted_consensus = 0.75    # Consensus pondéré minimum relevé (75%)
        
        # Paramètres transition et momentum DURCIS
        self.max_trend_transition_probability = 0.25 # Max probabilité transition réduite
        self.min_momentum_alignment = 0.6     # Alignement momentum minimum relevé (60%)
        self.trend_maturity_threshold = 15    # Barres maturité trend minimum relevé
        
        # Bonus/malus RÉDUITS
        self.perfect_alignment_bonus = 0.18   # Bonus alignement parfait réduit (18%)
        self.strong_trend_bonus = 0.15        # Bonus tendance forte réduit (15%)
        self.multi_tf_consensus_bonus = 0.12  # Bonus consensus multi-TF réduit (12%)
        self.weak_trend_penalty = -0.15       # Pénalité tendance faible réduite (-15%)
        self.misalignment_penalty = -0.18     # Pénalité désalignement réduite (-18%)
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur l'alignement des tendances multi-timeframes.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon trend alignment, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs de tendance depuis le contexte
            try:
                # Tendance principale
                # primary_trend_direction → directional_bias
                primary_trend_direction = self.context.get('directional_bias')  # 'bullish', 'bearish', 'neutral'
                # primary_trend_strength → trend_strength
                primary_trend_strength_raw = self.context.get('trend_strength')
                primary_trend_strength = self._convert_trend_strength_to_score(str(primary_trend_strength_raw)) if primary_trend_strength_raw is not None else None
                # primary_trend_age → regime_duration
                primary_trend_age = int(self.context.get('regime_duration', 0)) if self.context.get('regime_duration') is not None else None
                
                # EMA alignment
                # ema_alignment_score → trend_alignment
                ema_alignment_score = float(self.context.get('trend_alignment', 50.0)) if self.context.get('trend_alignment') is not None else None
                # ema_separation_ratio → trend_strength (éviter conversion si catégoriel)
                trend_strength_raw = self.context.get('trend_strength', 0)
                ema_separation_ratio = self._safe_float_conversion(trend_strength_raw, 0.01)
                # ema_slope_strength → trend_angle
                ema_slope_strength = float(self.context.get('trend_angle', 0)) if self.context.get('trend_angle') is not None else None
                
                # MACD alignment
                # macd_trend_coherence → macd_trend (catégoriel: BEARISH/BULLISH/NEUTRAL)
                macd_trend_raw = self.context.get('macd_trend', 'NEUTRAL')
                macd_trend_coherence = self._convert_macd_trend_to_score(str(macd_trend_raw)) if macd_trend_raw is not None else None
                # macd_histogram_strength → macd_histogram
                macd_histogram_strength = float(self.context.get('macd_histogram', 0)) if self.context.get('macd_histogram') is not None else None
                # macd_signal_alignment → macd_signal_cross (boolean)
                macd_signal_cross_bool = self.context.get('macd_signal_cross')
                macd_signal_alignment = 1.0 if macd_signal_cross_bool else 0.0 if macd_signal_cross_bool is not None else None
                
                # Multi-timeframe consensus
                # timeframe_consensus_score → confluence_score
                timeframe_consensus_score = float(self.context.get('confluence_score', 60.0)) if self.context.get('confluence_score') is not None else None
                # aligned_timeframes_count → pivot_count (utiliser comme proxy)
                aligned_timeframes_count = int(self.context.get('pivot_count', 2)) if self.context.get('pivot_count') is not None else None
                
                # Transitions et momentum
                # trend_transition_probability → pattern_confidence (convertir % vers probabilité)
                pattern_conf_raw = self.context.get('pattern_confidence')
                trend_transition_probability = (100.0 - float(pattern_conf_raw)) / 100.0 if pattern_conf_raw is not None else None
                # momentum_trend_alignment → momentum_score
                momentum_trend_alignment = float(self.context.get('momentum_score', 0)) if self.context.get('momentum_score') is not None else None
                
                # Tendances par timeframe (utiliser différents indicateurs comme proxy)
                # trend_1h → directional_bias (court terme)
                trend_1h = self.context.get('directional_bias')
                # trend_4h → trend_strength (moyen terme)
                trend_4h = self.context.get('trend_strength')
                # trend_1d → market_regime (long terme)
                trend_1d = self.context.get('market_regime')
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion indicateurs pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', 'Unknown')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side:
                logger.warning(f"{self.name}: Signal side manquant pour {self.symbol}")
                return False
                
            # 1. Validation force de la tendance principale
            if primary_trend_strength is not None and primary_trend_strength < self.min_trend_strength:
                logger.debug(f"{self.name}: Tendance principale faible ({self._safe_format(primary_trend_strength, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.75:  # Plus strict
                    return False
                    
            # 2. Validation cohérence direction tendance/signal
            if primary_trend_direction:
                direction_coherence = self._validate_trend_signal_coherence(signal_side, primary_trend_direction)
                if not direction_coherence:
                    logger.debug(f"{self.name}: Incohérence tendance {primary_trend_direction} / signal {signal_side} pour {self.symbol}")
                    if signal_confidence < 0.70:  # Plus strict
                        return False
                        
            # 3. Validation alignement EMA
            if ema_alignment_score is not None and ema_alignment_score < 50.0:  # Seuil relevé de 40 à 50
                logger.debug(f"{self.name}: Alignement EMA insuffisant ({self._safe_format(ema_alignment_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.65:  # Plus strict
                    return False
                    
            # Validation séparation EMA
            if ema_separation_ratio is not None and ema_separation_ratio < self.min_ema_separation:
                logger.debug(f"{self.name}: Séparation EMA insuffisante ({self._safe_format(ema_separation_ratio*100, '.2f')}%) pour {self.symbol}")
                if signal_confidence < 0.60:  # Plus strict
                    return False
                    
            # 4. Validation cohérence MACD
            if macd_trend_coherence is not None and macd_trend_coherence < self.macd_signal_coherence_threshold:
                logger.debug(f"{self.name}: Cohérence MACD/tendance insuffisante ({self._safe_format(macd_trend_coherence, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.65:  # Plus strict
                    return False
                    
            if macd_histogram_strength is not None and macd_histogram_strength < self.min_macd_histogram_strength:
                logger.debug(f"{self.name}: Force histogramme MACD insuffisante ({self._safe_format(macd_histogram_strength, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.70:  # Plus strict
                    return False
                    
            # 5. Validation consensus multi-timeframe
            if timeframe_consensus_score is not None and timeframe_consensus_score < self.min_trend_consensus:
                logger.debug(f"{self.name}: Consensus timeframes insuffisant ({self._safe_format(timeframe_consensus_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.70:  # Plus strict
                    return False
                    
            if aligned_timeframes_count is not None and aligned_timeframes_count < self.min_aligned_timeframes:
                logger.debug(f"{self.name}: Pas assez de timeframes alignés ({aligned_timeframes_count}) pour {self.symbol}")
                if signal_confidence < 0.75:  # Plus strict
                    return False
                    
            # 6. Validation probabilité transition
            if trend_transition_probability is not None and trend_transition_probability > self.max_trend_transition_probability:
                logger.debug(f"{self.name}: Probabilité transition élevée ({self._safe_format(trend_transition_probability, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.85:  # Plus strict
                    return False
                    
            # 7. Validation alignement momentum (format 0-100, convertir vers 0-1)
            if momentum_trend_alignment is not None:
                momentum_norm = momentum_trend_alignment / 100.0 if momentum_trend_alignment > 1 else momentum_trend_alignment
                if momentum_norm < self.min_momentum_alignment:
                    logger.debug(f"{self.name}: Alignement momentum insuffisant ({self._safe_format(momentum_norm, '.2f')}) pour {self.symbol}")
                    if signal_confidence < 0.70:  # Plus strict
                        return False
                    
            # 8. Validation maturité de la tendance
            if primary_trend_age is not None and primary_trend_age < self.trend_maturity_threshold:
                logger.debug(f"{self.name}: Tendance trop jeune ({primary_trend_age} barres) pour {self.symbol}")
                if signal_confidence < 0.75:  # Plus strict
                    return False
                    
            # 9. Validation cohérence timeframes spécifiques
            tf_coherence = self._validate_timeframe_coherence(signal_side, str(trend_1h) if trend_1h is not None else '', str(trend_4h) if trend_4h is not None else '', str(trend_1d) if trend_1d is not None else '')
            if not tf_coherence:
                logger.debug(f"{self.name}: Incohérence entre timeframes pour {self.symbol}")
                if signal_confidence < 0.70:  # Plus strict
                    return False
                    
            # 10. Validation adéquation stratégie/trend
            strategy_trend_match = self._validate_strategy_trend_match(signal_strategy, str(primary_trend_direction) if primary_trend_direction is not None else '', primary_trend_strength if primary_trend_strength is not None else 0.0)
            if not strategy_trend_match:
                logger.debug(f"{self.name}: Stratégie {signal_strategy} inadaptée à la tendance pour {self.symbol}")
                if signal_confidence < 0.70:  # Plus strict
                    return False
                    
            # 11. Validation pente EMA significative
            if ema_slope_strength is not None and ema_slope_strength < self.ema_slope_threshold:
                logger.debug(f"{self.name}: Pente EMA insignifiante ({self._safe_format(ema_slope_strength, '.4f')}) pour {self.symbol}")
                if signal_confidence < 0.65:  # Plus strict
                    return False
                    
            logger.debug(f"{self.name}: Signal validé pour {self.symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def _validate_trend_signal_coherence(self, signal_side: str, trend_direction: str) -> bool:
        """Valide la cohérence entre direction signal et tendance."""
        try:
            if not trend_direction:
                return True
                
            trend_upper = str(trend_direction).upper()
            
            # BUY signals cohérents avec tendances haussières (format: BULLISH/BEARISH/NEUTRAL)
            if signal_side == "BUY":
                if trend_upper == "BEARISH":
                    return False
                    
            # SELL signals cohérents avec tendances baissières (format: BULLISH/BEARISH/NEUTRAL)
            elif signal_side == "SELL":
                if trend_upper == "BULLISH":
                    return False
                    
            return True
            
        except Exception:
            return True
            
    def _validate_timeframe_coherence(self, signal_side: str, trend_1h: str, trend_4h: str, trend_1d: str) -> bool:
        """Valide la cohérence entre timeframes spécifiques."""
        try:
            trends = [trend_1h, trend_4h, trend_1d]
            valid_trends = [t for t in trends if t is not None]
            
            if len(valid_trends) < 2:
                return True  # Pas assez de données pour juger
                
            # Compter alignements favorables au signal
            favorable_count = 0
            total_count = len(valid_trends)
            
            for trend in valid_trends:
                trend_upper = str(trend).upper() if trend else ""
                
                if signal_side == "BUY":
                    # Gérer différents formats selon l'indicateur source
                    if trend_upper in ["BULLISH", "TRENDING_BULL", "BREAKOUT_BULL"]:
                        favorable_count += 1
                    elif trend_upper in ["BEARISH", "TRENDING_BEAR", "BREAKOUT_BEAR"]:
                        favorable_count -= 1  # Comptage négatif pour opposition
                    elif trend_upper in ["STRONG", "VERY_STRONG", "EXTREME"]:  # trend_strength
                        favorable_count += 1
                        
                elif signal_side == "SELL":
                    if trend_upper in ["BEARISH", "TRENDING_BEAR", "BREAKOUT_BEAR"]:
                        favorable_count += 1
                    elif trend_upper in ["BULLISH", "TRENDING_BULL", "BREAKOUT_BULL"]:
                        favorable_count -= 1
                    elif trend_upper in ["STRONG", "VERY_STRONG", "EXTREME"]:  # trend_strength
                        favorable_count += 1
                        
            # Au moins 60% des timeframes doivent être favorables ou neutres
            coherence_ratio = favorable_count / total_count
            return coherence_ratio >= 0.0  # Pas de forte opposition
            
        except Exception:
            return True
            
    def _validate_strategy_trend_match(self, strategy: str, trend_direction: str, trend_strength: float) -> bool:
        """Valide l'adéquation stratégie/tendance."""
        try:
            if not strategy:
                return True
                
            strategy_lower = strategy.lower()
            
            # Stratégies trend following
            if any(kw in strategy_lower for kw in ['trend', 'macd', 'ema', 'cross', 'slope', 'momentum']):
                # Nécessitent une tendance claire (format: BULLISH/BEARISH/NEUTRAL)
                if trend_direction and str(trend_direction).upper() == "NEUTRAL":
                    return False
                if trend_strength is not None and trend_strength < 0.3:
                    return False
                    
            # Stratégies mean reversion
            elif any(kw in strategy_lower for kw in ['bollinger', 'rsi', 'reversal', 'touch']):
                # Peuvent fonctionner même avec tendances faibles
                return True
                
            # Stratégies breakout
            elif any(kw in strategy_lower for kw in ['breakout', 'donchian', 'atr']):
                # Bénéficient de tendances en développement
                if trend_strength is not None and trend_strength > 0.8:
                    return False  # Tendance déjà très forte = moins de potentiel breakout
                    
            return True
            
        except Exception:
            return True
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur l'alignement des tendances.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur alignement tendances
            # primary_trend_strength → trend_strength
            primary_trend_strength_raw = self.context.get('trend_strength')
            primary_trend_strength = self._convert_trend_strength_to_score(str(primary_trend_strength_raw)) if primary_trend_strength_raw is not None else 0.5
            # ema_alignment_score → trend_alignment
            ema_alignment_score = float(self.context.get('trend_alignment', 50.0)) if self.context.get('trend_alignment') is not None else 50.0
            # ema_separation_ratio → trend_strength (éviter conversion si catégoriel) 
            trend_strength_raw = self.context.get('trend_strength', 0.01)
            ema_separation_ratio = self._safe_float_conversion(trend_strength_raw, 0.01)
            # macd_trend_coherence → macd_trend (catégoriel: BEARISH/BULLISH/NEUTRAL)
            macd_trend_raw = self.context.get('macd_trend', 'NEUTRAL')
            macd_trend_coherence = self._convert_macd_trend_to_score(str(macd_trend_raw))
            # timeframe_consensus_score → confluence_score
            timeframe_consensus_score = float(self.context.get('confluence_score', 60.0)) if self.context.get('confluence_score') is not None else 60.0
            # aligned_timeframes_count → pivot_count (utiliser comme proxy)
            aligned_timeframes_count = int(self.context.get('pivot_count', 2)) if self.context.get('pivot_count') is not None else 2
            # momentum_trend_alignment → momentum_score (format 0-100)
            momentum_score_raw = self.context.get('momentum_score')
            momentum_trend_alignment = float(momentum_score_raw) if momentum_score_raw is not None else 50.0
            
            # primary_trend_direction → directional_bias
            primary_trend_direction = self.context.get('directional_bias', 'neutral')
            signal_strategy = signal.get('strategy', '')
            signal_side = signal.get('side')
            
            base_score = 0.3  # Score de base réduit (0.3 au lieu de 0.5)
            
            # CORRECTION: Adapter le score selon la cohérence directionnelle
            direction_coherence = self._validate_trend_signal_coherence(signal_side, str(primary_trend_direction) if primary_trend_direction is not None else 'neutral')
            
            if not direction_coherence:
                # Signal contraire à la tendance = rejet total
                return 0.0  # Rejet complet si incohérent
            
            # Bonus force tendance principale (seulement si cohérent avec signal) - RÉDUIT
            if primary_trend_strength >= self.strong_trend_threshold:
                base_score += self.strong_trend_bonus  # Déjà réduit à 0.15
            elif primary_trend_strength >= self.min_trend_strength + 0.15:  # Seuil relevé
                base_score += 0.09  # Réduit de 0.15 à 0.09
                
            # Bonus alignement EMA (adapté selon direction) - RÉDUIT
            if ema_alignment_score >= 90.0:
                # Alignement parfait + direction cohérente = bonus maximal
                base_score += 0.12  # Réduit de 0.20 à 0.12
            elif ema_alignment_score >= 75.0:  # Seuil relevé de 70 à 75
                # Bon alignement + direction cohérente = bonus modéré
                base_score += 0.07  # Réduit de 0.12 à 0.07
                
            # Bonus séparation EMA optimale - RÉDUIT
            if self.min_ema_separation <= ema_separation_ratio <= self.optimal_ema_separation:
                base_score += 0.06  # Réduit de 0.10 à 0.06
                
            # Bonus cohérence MACD excellente - RÉDUIT
            if macd_trend_coherence >= 0.9:
                base_score += 0.09  # Réduit de 0.15 à 0.09
            elif macd_trend_coherence >= 0.85:  # Seuil relevé
                base_score += 0.06  # Réduit de 0.10 à 0.06
                
            # Bonus consensus multi-timeframe (format 0-100) - RÉDUIT
            consensus_norm = timeframe_consensus_score / 100.0 if timeframe_consensus_score > 1 else timeframe_consensus_score
            if consensus_norm >= 0.9:
                base_score += self.multi_tf_consensus_bonus  # Déjà réduit à 0.12
            elif consensus_norm >= 0.75:  # Seuil relevé
                base_score += 0.07  # Réduit de 0.12 à 0.07
                
            # Bonus nombre timeframes alignés - RÉDUIT
            if aligned_timeframes_count >= 5:  # Seuil relevé
                base_score += 0.09  # Réduit de 0.15 à 0.09
            elif aligned_timeframes_count >= 4:  # Seuil relevé
                base_score += 0.06  # Réduit de 0.10 à 0.06
                
            # Bonus alignement momentum - RÉDUIT
            if momentum_trend_alignment >= 85:  # Seuil relevé
                base_score += 0.07  # Réduit de 0.12 à 0.07
            elif momentum_trend_alignment >= 70:  # Seuil relevé
                base_score += 0.05  # Réduit de 0.08 à 0.05
                
            # Bonus cohérence parfaite signal/tendance - RÉDUIT
            if signal_side and self._validate_trend_signal_coherence(signal_side, str(primary_trend_direction) if primary_trend_direction is not None else 'NEUTRAL'):
                if primary_trend_direction and str(primary_trend_direction).upper() != "NEUTRAL":
                    base_score += 0.06  # Réduit de 0.10 à 0.06
                    
            # Bonus stratégie adaptée - RÉDUIT
            if self._validate_strategy_trend_match(signal_strategy, primary_trend_direction, primary_trend_strength):
                base_score += 0.05  # Réduit de 0.08 à 0.05
                
            # Bonus alignement global exceptionnel
            alignment_factors = [
                ema_alignment_score, macd_trend_coherence, 
                timeframe_consensus_score, momentum_trend_alignment
            ]
            avg_alignment = sum(f for f in alignment_factors if f is not None) / len([f for f in alignment_factors if f is not None])
            
            if avg_alignment >= 0.90:  # Seuil plus strict
                base_score += self.perfect_alignment_bonus  # Déjà réduit à 0.18
                
            # NOUVEAU: Filtre final - rejeter si score trop faible
            if base_score < 0.35:  # Score minimum 35% pour trend alignment
                return 0.0
                
            # NOUVEAU: Cap final pour éviter les scores excessifs
            final_score = max(0.0, min(0.85, base_score))  # Limité à 85% maximum
            return final_score
            
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
            
            # primary_trend_direction → directional_bias
            primary_trend_direction = self.context.get('directional_bias', 'N/A')
            # primary_trend_strength → trend_strength
            primary_trend_strength_raw = self.context.get('trend_strength', 0)
            primary_trend_strength = self._convert_trend_strength_to_score(str(primary_trend_strength_raw)) if primary_trend_strength_raw is not None else None
            # ema_alignment_score → trend_alignment
            ema_alignment_score = float(self.context.get('trend_alignment', 50.0)) if self.context.get('trend_alignment') is not None else None
            # timeframe_consensus_score → confluence_score
            timeframe_consensus_score = float(self.context.get('confluence_score', 60.0)) if self.context.get('confluence_score') is not None else None
            # aligned_timeframes_count → pivot_count (utiliser comme proxy)
            aligned_timeframes_count = int(self.context.get('pivot_count', 0)) if self.context.get('pivot_count') is not None else None
            
            if is_valid:
                reason = f"Alignement de tendance favorable"
                if primary_trend_direction != 'N/A':
                    reason += f" (direction: {primary_trend_direction})"
                if primary_trend_strength is not None:
                    reason += f", force: {self._safe_format(primary_trend_strength, '.2f')}"
                if ema_alignment_score is not None:
                    reason += f", EMA align: {self._safe_format(ema_alignment_score, '.2f')}"
                if timeframe_consensus_score is not None:
                    reason += f", consensus TF: {self._safe_format(timeframe_consensus_score, '.2f')}"
                if aligned_timeframes_count is not None:
                    reason += f", TF alignés: {aligned_timeframes_count}"
                    
                return f"{self.name}: Validé - {reason} pour {signal_strategy} {signal_side}"
            else:
                if primary_trend_strength is not None and primary_trend_strength < self.min_trend_strength:
                    return f"{self.name}: Rejeté - Tendance trop faible ({self._safe_format(primary_trend_strength, '.2f')})"
                elif primary_trend_direction != 'N/A' and not self._validate_trend_signal_coherence(signal_side, primary_trend_direction):
                    return f"{self.name}: Rejeté - Incohérence tendance {primary_trend_direction} / signal {signal_side}"
                elif ema_alignment_score is not None and ema_alignment_score < 40.0:
                    return f"{self.name}: Rejeté - Alignement EMA insuffisant ({self._safe_format(ema_alignment_score, '.2f')})"
                elif timeframe_consensus_score is not None and timeframe_consensus_score < self.min_trend_consensus:
                    return f"{self.name}: Rejeté - Consensus timeframes insuffisant ({self._safe_format(timeframe_consensus_score, '.2f')})"
                elif aligned_timeframes_count is not None and aligned_timeframes_count < self.min_aligned_timeframes:
                    return f"{self.name}: Rejeté - Pas assez de timeframes alignés ({aligned_timeframes_count})"
                    
                return f"{self.name}: Rejeté - Critères alignement tendance non respectés"
                
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données de tendance requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Au minimum, on a besoin d'un indicateur de tendance
        trend_indicators = [
            'directional_bias', 'trend_strength', 'trend_alignment',
            'confluence_score', 'macd_trend'
        ]
        
        available_indicators = sum(1 for ind in trend_indicators 
                                 if ind in self.context and self.context[ind] is not None)
        
        if available_indicators < 2:
            logger.warning(f"{self.name}: Pas assez d'indicateurs de tendance pour {self.symbol}")
            return False
            
        return True
            
    def _convert_trend_strength_to_score(self, trend_strength_str: str) -> float:
        """Convertit une chaîne de force de tendance en score numérique."""
        try:
            if not trend_strength_str:
                return 0.5
                
            trend_lower = trend_strength_str.lower()
            
            if trend_lower in ['very_strong', 'strong_bullish', 'strong_bearish']:
                return 0.9
            elif trend_lower in ['strong', 'bullish', 'bearish']:
                return 0.7
            elif trend_lower in ['moderate', 'weak_bullish', 'weak_bearish']:
                return 0.5
            elif trend_lower in ['weak', 'very_weak']:
                return 0.3
            elif trend_lower in ['neutral', 'sideways', 'absent']:
                return 0.1
            else:
                # Essayer de convertir directement en float
                try:
                    return float(trend_strength_str)
                except (ValueError, TypeError):
                    # Si conversion impossible, traiter comme neutre
                    return 0.5
                    
        except Exception:
            return 0.5
    
    def _safe_float_conversion(self, value, default: float) -> float:
        """Conversion sécurisée vers float, évite les erreurs avec valeurs catégorielles."""
        try:
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                # Essayer conversion directe
                try:
                    return float(value)
                except (ValueError, TypeError):
                    # Si c'est une valeur catégorielle, convertir via _convert_trend_strength_to_score
                    if value.lower() in ['bearish', 'bullish', 'neutral', 'strong', 'weak', 'moderate']:
                        return self._convert_trend_strength_to_score(value)
                    return default
            return default
        except Exception:
            return default
    
    def _convert_macd_trend_to_score(self, macd_trend: str) -> float:
        """Convertit une tendance MACD catégorielle en score numérique."""
        try:
            if not macd_trend:
                return 0.5
                
            trend_lower = macd_trend.lower()
            
            if trend_lower in ['bullish', 'up', 'positive']:
                return 0.8  # MACD haussier
            elif trend_lower in ['bearish', 'down', 'negative']:
                return 0.2  # MACD baissier
            elif trend_lower in ['neutral', 'sideways', 'flat']:
                return 0.5  # MACD neutre
            else:
                # Essayer de convertir directement en float
                try:
                    return float(macd_trend)
                except (ValueError, TypeError):
                    return 0.5  # Valeur par défaut
                    
        except Exception:
            return 0.5
