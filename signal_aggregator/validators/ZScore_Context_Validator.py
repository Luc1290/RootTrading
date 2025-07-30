"""
ZScore_Context_Validator - Validator basé sur l'analyse statistique Z-Score et contexte de marché.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class ZScore_Context_Validator(BaseValidator):
    """
    Validator pour l'analyse Z-Score contextuelle - filtre selon les anomalies statistiques et contexte.
    
    Vérifie: Z-Score prix/volume, normalité distributions, outliers, contexte statistique
    Catégorie: technical
    
    Rejette les signaux en:
    - Z-Scores extrêmes ou anormaux
    - Contexte statistique défavorable
    - Anomalies de distribution suspectes
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "ZScore_Context_Validator"
        self.category = "technical"
        
        # Paramètres Z-Score de base
        self.min_zscore_threshold = -3.0        # Z-Score minimum acceptable
        self.max_zscore_threshold = 3.0         # Z-Score maximum acceptable
        self.optimal_zscore_range_min = -2.0    # Zone optimale minimum
        self.optimal_zscore_range_max = 2.0     # Zone optimale maximum
        self.extreme_zscore_threshold = 2.5     # Seuil Z-Score extrême
        
        # Paramètres contexte prix
        self.price_zscore_significance = 1.96   # Seuil significativité prix (95%)
        self.price_zscore_extreme = 2.58        # Seuil extrême prix (99%)
        self.price_mean_reversion_threshold = 1.5  # Seuil mean reversion
        
        # Paramètres contexte volume
        self.volume_zscore_significance = 1.64  # Seuil significativité volume (90%)
        self.volume_zscore_extreme = 2.33       # Seuil extrême volume (98%)
        self.volume_anomaly_threshold = 2.0     # Seuil anomalie volume
        
        # Paramètres distribution et normalité
        self.min_normality_score = 0.3          # Score normalité minimum
        self.skewness_threshold = 1.0           # Seuil asymétrie distribution
        self.kurtosis_threshold = 3.0           # Seuil aplatissement distribution
        self.outlier_ratio_max = 0.1            # Ratio maximum outliers
        
        # Paramètres contexte temporel
        self.lookback_periods = [20, 50, 100]   # Périodes lookback Z-Score
        self.zscore_stability_threshold = 0.8   # Stabilité Z-Score
        self.context_persistence_min = 3        # Persistance contexte minimum
        
        # Paramètres confluence statistique
        self.min_statistical_confluence = 0.6   # Confluence statistique minimum
        self.zscore_coherence_threshold = 0.7   # Cohérence entre Z-Scores
        self.context_reliability_min = 0.5      # Fiabilité contexte minimum
        
        # Bonus/malus
        self.optimal_zscore_bonus = 0.25        # Bonus Z-Score optimal
        self.statistical_confluence_bonus = 0.20 # Bonus confluence statistique
        self.context_persistence_bonus = 0.15   # Bonus persistance contexte
        self.normality_bonus = 0.12             # Bonus normalité distribution
        self.extreme_zscore_penalty = -0.30     # Pénalité Z-Score extrême
        self.poor_context_penalty = -0.25       # Pénalité contexte défavorable
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur l'analyse Z-Score contextuelle et statistiques.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon Z-Score context, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs Z-Score depuis le contexte
            try:
                # Z-Scores principaux
                price_zscore = float(self.context.get('price_zscore', 0)) if self.context.get('price_zscore') is not None else None
                volume_zscore = float(self.context.get('volume_zscore', 0)) if self.context.get('volume_zscore') is not None else None
                returns_zscore = float(self.context.get('returns_zscore', 0)) if self.context.get('returns_zscore') is not None else None
                
                # Z-Scores multi-périodes
                zscore_20 = float(self.context.get('zscore_20', 0)) if self.context.get('zscore_20') is not None else None
                zscore_50 = float(self.context.get('zscore_50', 0)) if self.context.get('zscore_50') is not None else None
                zscore_100 = float(self.context.get('zscore_100', 0)) if self.context.get('zscore_100') is not None else None
                
                # Contexte statistique
                statistical_significance = float(self.context.get('statistical_significance', 0)) if self.context.get('statistical_significance') is not None else None
                distribution_normality = float(self.context.get('distribution_normality', 0.5)) if self.context.get('distribution_normality') is not None else None
                data_reliability_score = float(self.context.get('data_reliability_score', 0.5)) if self.context.get('data_reliability_score') is not None else None
                
                # Paramètres distribution
                distribution_skewness = float(self.context.get('distribution_skewness', 0)) if self.context.get('distribution_skewness') is not None else None
                distribution_kurtosis = float(self.context.get('distribution_kurtosis', 3)) if self.context.get('distribution_kurtosis') is not None else None
                outlier_ratio = float(self.context.get('outlier_ratio', 0)) if self.context.get('outlier_ratio') is not None else None
                
                # Contexte temporel
                zscore_stability = float(self.context.get('zscore_stability', 0.5)) if self.context.get('zscore_stability') is not None else None
                context_persistence = int(self.context.get('context_persistence', 0)) if self.context.get('context_persistence') is not None else None
                trend_zscore_coherence = float(self.context.get('trend_zscore_coherence', 0.5)) if self.context.get('trend_zscore_coherence') is not None else None
                
                # Confluence et anomalies
                statistical_confluence = float(self.context.get('statistical_confluence', 0.5)) if self.context.get('statistical_confluence') is not None else None
                anomaly_detected = self.context.get('anomaly_detected', False)
                mean_reversion_signal = self.context.get('mean_reversion_signal', False)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion indicateurs pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', 'Unknown')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side:
                logger.warning(f"{self.name}: Signal side manquant pour {self.symbol}")
                return False
                
            # 1. Validation Z-Score prix dans limites acceptables
            if price_zscore is not None:
                if price_zscore < self.min_zscore_threshold or price_zscore > self.max_zscore_threshold:
                    logger.debug(f"{self.name}: Z-Score prix hors limites ({self._safe_format(price_zscore, '.2f')}) pour {self.symbol}")
                    if signal_confidence < 0.9:
                        return False
                        
                # Validation Z-Score extrême
                if abs(price_zscore) > self.extreme_zscore_threshold:
                    logger.debug(f"{self.name}: Z-Score prix extrême ({self._safe_format(price_zscore, '.2f')}) pour {self.symbol}")
                    if signal_confidence < 0.8:
                        return False
                        
            # 2. Validation Z-Score volume
            if volume_zscore is not None:
                if abs(volume_zscore) > self.volume_zscore_extreme:
                    logger.debug(f"{self.name}: Z-Score volume extrême ({self._safe_format(volume_zscore, '.2f')}) pour {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                        
                # Validation anomalie volume
                if abs(volume_zscore) > self.volume_anomaly_threshold:
                    logger.debug(f"{self.name}: Anomalie volume détectée ({self._safe_format(volume_zscore, '.2f')}) pour {self.symbol}")
                    if signal_confidence < 0.6:
                        return False
                        
            # 3. Validation cohérence Z-Scores multi-périodes
            zscores = [z for z in [zscore_20, zscore_50, zscore_100] if z is not None]
            if len(zscores) >= 2:
                zscore_coherence = self._validate_zscore_coherence(zscores)
                if not zscore_coherence:
                    logger.debug(f"{self.name}: Incohérence Z-Scores multi-périodes pour {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                        
            # 4. Validation significativité statistique
            if statistical_significance is not None and statistical_significance < 0.05:
                logger.debug(f"{self.name}: Significativité statistique insuffisante ({self._safe_format(statistical_significance, '.3f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 5. Validation normalité distribution
            if distribution_normality is not None and distribution_normality < self.min_normality_score:
                logger.debug(f"{self.name}: Normalité distribution insuffisante ({self._safe_format(distribution_normality, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 6. Validation asymétrie et aplatissement
            if distribution_skewness is not None and abs(distribution_skewness) > self.skewness_threshold:
                logger.debug(f"{self.name}: Asymétrie distribution excessive ({self._safe_format(distribution_skewness, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            if distribution_kurtosis is not None and distribution_kurtosis > self.kurtosis_threshold * 2:
                logger.debug(f"{self.name}: Aplatissement distribution excessif ({self._safe_format(distribution_kurtosis, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 7. Validation ratio outliers
            if outlier_ratio is not None and outlier_ratio > self.outlier_ratio_max:
                logger.debug(f"{self.name}: Ratio outliers excessif ({self._safe_format(outlier_ratio*100, '.1f')}%) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 8. Validation stabilité Z-Score
            if zscore_stability is not None and zscore_stability < self.zscore_stability_threshold:
                logger.debug(f"{self.name}: Stabilité Z-Score insuffisante ({self._safe_format(zscore_stability, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 9. Validation persistance contexte
            if context_persistence is not None and context_persistence < self.context_persistence_min:
                logger.debug(f"{self.name}: Persistance contexte insuffisante ({context_persistence} périodes) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 10. Validation confluence statistique
            if statistical_confluence is not None and statistical_confluence < self.min_statistical_confluence:
                logger.debug(f"{self.name}: Confluence statistique insuffisante ({self._safe_format(statistical_confluence, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 11. Validation cohérence trend/Z-Score
            if trend_zscore_coherence is not None and trend_zscore_coherence < self.zscore_coherence_threshold:
                logger.debug(f"{self.name}: Cohérence trend/Z-Score insuffisante ({self._safe_format(trend_zscore_coherence, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 12. Validation fiabilité données
            if data_reliability_score is not None and data_reliability_score < self.context_reliability_min:
                logger.debug(f"{self.name}: Fiabilité données insuffisante ({self._safe_format(data_reliability_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 13. Validation cohérence signal/mean reversion
            if mean_reversion_signal and price_zscore is not None:
                reversion_coherence = self._validate_mean_reversion_coherence(signal_side, price_zscore)
                if not reversion_coherence:
                    logger.debug(f"{self.name}: Incohérence signal/mean reversion pour {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                        
            # 14. Validation spécifique selon stratégie
            strategy_zscore_match = self._validate_strategy_zscore_match(signal_strategy, price_zscore, volume_zscore)
            if not strategy_zscore_match:
                logger.debug(f"{self.name}: Stratégie {signal_strategy} inadaptée au contexte Z-Score pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                        f"Z-Score prix: {self._safe_format(price_zscore, '.2f') if price_zscore is not None else 'N/A'}, "
                        f"Z-Score volume: {self._safe_format(volume_zscore, '.2f') if volume_zscore is not None else 'N/A'}, "
                        f"Normalité: {self._safe_format(distribution_normality, '.2f') if distribution_normality is not None else 'N/A'}, "
                        f"Confluence: {self._safe_format(statistical_confluence, '.2f') if statistical_confluence is not None else 'N/A'}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def _validate_zscore_coherence(self, zscores: list) -> bool:
        """Valide la cohérence entre Z-Scores de différentes périodes."""
        try:
            if len(zscores) < 2:
                return True
                
            # Vérifier que les Z-Scores pointent dans la même direction générale
            positive_count = sum(1 for z in zscores if z > 0.5)
            negative_count = sum(1 for z in zscores if z < -0.5)
            neutral_count = len(zscores) - positive_count - negative_count
            
            # Au moins 60% des Z-Scores doivent être cohérents
            total = len(zscores)
            max_coherent = max(positive_count, negative_count, neutral_count)
            coherence_ratio = max_coherent / total
            
            return coherence_ratio >= 0.6
            
        except Exception:
            return True
            
    def _validate_mean_reversion_coherence(self, signal_side: str, price_zscore: float) -> bool:
        """Valide la cohérence signal/mean reversion selon Z-Score prix."""
        try:
            if not price_zscore:
                return True
                
            # Pour mean reversion, BUY quand Z-Score négatif (sous-évalué)
            # SELL quand Z-Score positif (sur-évalué)
            if signal_side == "BUY":
                return price_zscore < -self.price_mean_reversion_threshold
            elif signal_side == "SELL":
                return price_zscore > self.price_mean_reversion_threshold
                
            return True
            
        except Exception:
            return True
            
    def _validate_strategy_zscore_match(self, strategy: str, price_zscore: float, volume_zscore: float) -> bool:
        """Valide l'adéquation stratégie/contexte Z-Score."""
        try:
            if not strategy:
                return True
                
            strategy_lower = strategy.lower()
            
            # Stratégies mean reversion
            if any(kw in strategy_lower for kw in ['bollinger', 'rsi', 'reversal', 'touch', 'reversion']):
                # Fonctionnent mieux avec Z-Scores modérés à élevés
                if price_zscore is not None and abs(price_zscore) < 1.0:
                    return False  # Z-Score trop faible pour mean reversion
                    
            # Stratégies momentum/breakout
            elif any(kw in strategy_lower for kw in ['breakout', 'momentum', 'trend', 'cross']):
                # Éviter Z-Scores extrêmes (possible épuisement momentum)
                if price_zscore is not None and abs(price_zscore) > 2.5:
                    return False
                    
            # Stratégies volume
            elif any(kw in strategy_lower for kw in ['volume', 'spike']):
                # Nécessitent Z-Score volume significatif
                if volume_zscore is not None and abs(volume_zscore) < 1.0:
                    return False
                    
            return True
            
        except Exception:
            return True
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur l'analyse Z-Score contextuelle.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur contexte Z-Score
            price_zscore = float(self.context.get('price_zscore', 0)) if self.context.get('price_zscore') is not None else 0
            volume_zscore = float(self.context.get('volume_zscore', 0)) if self.context.get('volume_zscore') is not None else 0
            distribution_normality = float(self.context.get('distribution_normality', 0.5)) if self.context.get('distribution_normality') is not None else 0.5
            zscore_stability = float(self.context.get('zscore_stability', 0.5)) if self.context.get('zscore_stability') is not None else 0.5
            statistical_confluence = float(self.context.get('statistical_confluence', 0.5)) if self.context.get('statistical_confluence') is not None else 0.5
            context_persistence = int(self.context.get('context_persistence', 3)) if self.context.get('context_persistence') is not None else 3
            data_reliability_score = float(self.context.get('data_reliability_score', 0.5)) if self.context.get('data_reliability_score') is not None else 0.5
            
            signal_strategy = signal.get('strategy', '')
            
            base_score = 0.5  # Score de base si validé
            
            # Bonus Z-Score prix dans zone optimale
            if self.optimal_zscore_range_min <= price_zscore <= self.optimal_zscore_range_max:
                if abs(price_zscore) >= 1.0:  # Z-Score significatif mais pas extrême
                    base_score += self.optimal_zscore_bonus
                elif abs(price_zscore) >= 0.5:
                    base_score += 0.15
                    
            # Bonus Z-Score volume approprié
            if 1.0 <= abs(volume_zscore) <= 2.5:
                base_score += 0.15  # Volume Z-Score dans zone optimale
            elif 0.5 <= abs(volume_zscore) <= 3.0:
                base_score += 0.08
                
            # Bonus normalité distribution
            if distribution_normality >= 0.8:
                base_score += self.normality_bonus
            elif distribution_normality >= 0.6:
                base_score += 0.08
                
            # Bonus stabilité Z-Score
            if zscore_stability >= 0.9:
                base_score += 0.12  # Z-Score très stable
            elif zscore_stability >= self.zscore_stability_threshold:
                base_score += 0.08
                
            # Bonus confluence statistique
            if statistical_confluence >= 0.8:
                base_score += self.statistical_confluence_bonus
            elif statistical_confluence >= 0.7:
                base_score += 0.12
                
            # Bonus persistance contexte
            if context_persistence >= 5:
                base_score += self.context_persistence_bonus
            elif context_persistence >= self.context_persistence_min:
                base_score += 0.08
                
            # Bonus fiabilité données
            if data_reliability_score >= 0.8:
                base_score += 0.10  # Données très fiables
            elif data_reliability_score >= 0.6:
                base_score += 0.06
                
            # Bonus stratégie adaptée
            if self._validate_strategy_zscore_match(signal_strategy, price_zscore, volume_zscore):
                base_score += 0.08  # Stratégie bien adaptée au contexte Z-Score
                
            # Bonus mean reversion cohérent
            signal_side = signal.get('side')
            mean_reversion_signal = self.context.get('mean_reversion_signal', False)
            if mean_reversion_signal and self._validate_mean_reversion_coherence(signal_side, price_zscore):
                base_score += 0.10  # Signal cohérent avec mean reversion
                
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
            
            price_zscore = float(self.context.get('price_zscore', 0)) if self.context.get('price_zscore') is not None else None
            volume_zscore = float(self.context.get('volume_zscore', 0)) if self.context.get('volume_zscore') is not None else None
            distribution_normality = float(self.context.get('distribution_normality', 0.5)) if self.context.get('distribution_normality') is not None else None
            statistical_confluence = float(self.context.get('statistical_confluence', 0.5)) if self.context.get('statistical_confluence') is not None else None
            context_persistence = int(self.context.get('context_persistence', 0)) if self.context.get('context_persistence') is not None else None
            
            if is_valid:
                reason = f"Contexte Z-Score favorable"
                if price_zscore is not None:
                    zscore_desc = "extrême" if abs(price_zscore) > 2.5 else "fort" if abs(price_zscore) > 1.5 else "modéré"
                    reason += f" (prix {zscore_desc}: {self._safe_format(price_zscore, '.2f')})"
                if volume_zscore is not None:
                    reason += f", volume: {self._safe_format(volume_zscore, '.2f')}"
                if distribution_normality is not None:
                    reason += f", normalité: {self._safe_format(distribution_normality, '.2f')}"
                if statistical_confluence is not None:
                    reason += f", confluence: {self._safe_format(statistical_confluence, '.2f')}"
                if context_persistence is not None:
                    reason += f", persistance: {context_persistence}p"
                    
                return f"{self.name}: Validé - {reason} pour {signal_strategy} {signal_side}"
            else:
                if price_zscore is not None:
                    if price_zscore < self.min_zscore_threshold or price_zscore > self.max_zscore_threshold:
                        return f"{self.name}: Rejeté - Z-Score prix hors limites ({self._safe_format(price_zscore, '.2f')})"
                    elif abs(price_zscore) > self.extreme_zscore_threshold:
                        return f"{self.name}: Rejeté - Z-Score prix extrême ({self._safe_format(price_zscore, '.2f')})"
                        
                if volume_zscore is not None and abs(volume_zscore) > self.volume_zscore_extreme:
                    return f"{self.name}: Rejeté - Z-Score volume extrême ({self._safe_format(volume_zscore, '.2f')})"
                    
                if distribution_normality is not None and distribution_normality < self.min_normality_score:
                    return f"{self.name}: Rejeté - Normalité distribution insuffisante ({self._safe_format(distribution_normality, '.2f')})"
                    
                if statistical_confluence is not None and statistical_confluence < self.min_statistical_confluence:
                    return f"{self.name}: Rejeté - Confluence statistique insuffisante ({self._safe_format(statistical_confluence, '.2f')})"
                    
                if context_persistence is not None and context_persistence < self.context_persistence_min:
                    return f"{self.name}: Rejeté - Persistance contexte insuffisante ({context_persistence} périodes)"
                    
                return f"{self.name}: Rejeté - Critères Z-Score contexte non respectés"
                
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données Z-Score requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Au minimum, on a besoin d'un indicateur Z-Score
        zscore_indicators = [
            'price_zscore', 'volume_zscore', 'returns_zscore',
            'distribution_normality', 'statistical_confluence', 'zscore_stability'
        ]
        
        available_indicators = sum(1 for ind in zscore_indicators 
                                 if ind in self.context and self.context[ind] is not None)
        
        if available_indicators < 2:
            logger.warning(f"{self.name}: Pas assez d'indicateurs Z-Score pour {self.symbol}")
            return False
            
        return True
