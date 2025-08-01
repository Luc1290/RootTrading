"""
MultiTF_Consensus_Validator - Validator basé sur le consensus multi-timeframes.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class MultiTF_Consensus_Validator(BaseValidator):
    """
    Validator pour le consensus multi-timeframes - filtre selon l'alignement entre différents TF.
    
    Vérifie: Consensus TF, alignement MA, cohérence directionnelle
    Catégorie: technical
    
    Rejette les signaux en:
    - Consensus TF insuffisant
    - Divergences majeures entre timeframes
    - Alignement MA contradictoire
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "MultiTF_Consensus_Validator"
        self.category = "technical"
        
        # Paramètres consensus multi-TF
        self.min_consensus_score = 0.6      # Score consensus minimum
        self.min_tf_alignment = 0.65        # Alignement TF minimum
        self.min_ma_alignment = 0.7         # Alignement MA minimum
        self.min_signal_strength = 0.55     # Force signal minimum
        
        # Paramètres directionnels
        self.directional_consensus_weight = 0.3   # Poids consensus directionnel
        self.trend_consistency_min = 0.6          # Cohérence tendance minimum
        
        # Seuils critiques
        self.critical_divergence_threshold = 0.3  # Seuil divergence crítica
        self.strong_consensus_threshold = 0.8     # Consensus très fort
        self.perfect_alignment_threshold = 0.9    # Alignement parfait
        
        # Bonus/malus
        self.perfect_consensus_bonus = 0.30       # Bonus consensus parfait
        self.strong_alignment_bonus = 0.20        # Bonus alignement fort
        self.divergence_penalty = -0.25           # Pénalité divergence
        self.ma_confluence_bonus = 0.15           # Bonus confluence MA
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur le consensus multi-timeframes.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon consensus MultiTF, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs multi-TF depuis le contexte
            try:
                # Scores principaux multi-TF
                # consensus_score → confluence_score (score de confluence) - convertir 0-100 en 0-1
                confluence_raw = self.context.get('confluence_score')
                consensus_score = float(confluence_raw) / 100.0 if confluence_raw is not None else None
                # tf_alignment → trend_alignment (alignement de tendance)
                tf_alignment = float(self.context.get('trend_alignment', 0)) if self.context.get('trend_alignment') is not None else None
                trend_alignment = float(self.context.get('trend_alignment', 0)) if self.context.get('trend_alignment') is not None else None
                # Ligne 72: Vérification None et type avant conversion
                signal_strength_raw = self.context.get('signal_strength')
                signal_strength = self._convert_signal_strength_to_score(str(signal_strength_raw)) if signal_strength_raw is not None else None
                
                # Alignement moyennes mobiles (disponibles dans MultiTF_ConfluentEntry_Strategy)
                ma_alignment_score = self._calculate_ma_alignment()
                
                # Direction et tendance
                directional_bias = self.context.get('directional_bias')
                # Ligne 79: Vérification None et type avant conversion
                trend_strength_raw = self.context.get('trend_strength')
                trend_strength = self._convert_trend_strength_to_score(str(trend_strength_raw)) if trend_strength_raw is not None else None
                
                # Confluence générale
                confluence_score = float(self.context.get('confluence_score', 50.0)) if self.context.get('confluence_score') is not None else None
                pattern_confidence = float(self.context.get('pattern_confidence', 0)) if self.context.get('pattern_confidence') is not None else None
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion indicateurs pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', 'Unknown')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side:
                logger.warning(f"{self.name}: Signal side manquant pour {self.symbol}")
                return False
                
            # 1. Validation score consensus principal
            if consensus_score is not None and consensus_score < self.min_consensus_score:
                logger.debug(f"{self.name}: Consensus score insuffisant ({self._safe_format(consensus_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.8:  # Accepter seulement si très confiant
                    return False
                    
            # 2. Validation alignement timeframes
            if tf_alignment is not None and tf_alignment < self.min_tf_alignment:
                logger.debug(f"{self.name}: Alignement TF insuffisant ({self._safe_format(tf_alignment, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 3. Validation alignement tendance
            if trend_alignment is not None and trend_alignment < self.min_ma_alignment:
                logger.debug(f"{self.name}: Alignement tendance insuffisant ({self._safe_format(trend_alignment, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 4. Validation force signal multi-TF
            if signal_strength is not None and signal_strength < self.min_signal_strength:
                logger.debug(f"{self.name}: Force signal insuffisante ({self._safe_format(signal_strength, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 5. Validation alignement moyennes mobiles calculé
            if ma_alignment_score < self.min_ma_alignment:
                logger.debug(f"{self.name}: Alignement MA calculé insuffisant ({self._safe_format(ma_alignment_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 6. Validation consensus directionnel
            # Lignes 129: Vérification None et type avant appel
            directional_consensus = self._validate_directional_consensus(
                signal_side, str(directional_bias) if directional_bias is not None else "neutral", trend_strength or 0.0
            )
            if not directional_consensus:
                logger.debug(f"{self.name}: Consensus directionnel insuffisant pour {self.symbol}")
                if signal_confidence < 0.8:
                    return False
                    
            # 7. Détection divergences critiques
            # Ligne 138: Vérification None avant appel
            critical_divergence = self._detect_critical_divergences(
                consensus_score or 0.0, tf_alignment or 0.0, trend_alignment or 0.0, signal_strength or 0.0
            )
            if critical_divergence:
                logger.debug(f"{self.name}: Divergence critique détectée pour {self.symbol}")
                if signal_confidence < 0.9:  # Très strict pour divergences
                    return False
                    
            # 8. Validation spécifique pour stratégies multi-TF
            if self._is_multitf_strategy(signal_strategy):
                # Critères plus stricts pour stratégies multi-TF
                min_scores = [
                    (consensus_score, self.min_consensus_score + 0.1, "consensus"),
                    (tf_alignment, self.min_tf_alignment + 0.05, "tf_alignment"), 
                    (trend_alignment, self.min_ma_alignment + 0.05, "trend_alignment")
                ]
                
                for score, min_val, name in min_scores:
                    if score is not None and score < min_val:
                        logger.debug(f"{self.name}: Stratégie MultiTF mais {name} insuffisant ({self._safe_format(score, '.2f')}) pour {self.symbol}")
                        return False
                        
            # 9. Validation confluence si disponible
            if confluence_score is not None and confluence_score < 50.0:
                logger.debug(f"{self.name}: Confluence générale faible ({self._safe_format(confluence_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 10. Validation pattern confidence
            if pattern_confidence is not None and pattern_confidence < 40:
                logger.debug(f"{self.name}: Pattern confidence faible ({self._safe_format(pattern_confidence, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                        f"Consensus: {self._safe_format(consensus_score, '.2f') if consensus_score is not None else 'N/A'}, "
                        f"TF Alignment: {self._safe_format(tf_alignment, '.2f') if tf_alignment is not None else 'N/A'}, "
                        f"Trend Alignment: {self._safe_format(trend_alignment, '.2f') if trend_alignment is not None else 'N/A'}, "
                        f"MA Alignment: {self._safe_format(ma_alignment_score, '.2f')}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def _calculate_ma_alignment(self) -> float:
        """Calcule le score d'alignement des moyennes mobiles."""
        try:
            # Récupération des MA disponibles
            mas = {}
            ma_keys = ['ema_7', 'ema_12', 'ema_26', 'ema_50', 'ema_99', 'sma_20', 'sma_50', 'hull_20']
            
            for key in ma_keys:
                if key in self.context and self.context[key] is not None:
                    try:
                        mas[key] = float(self.context[key])
                    except (ValueError, TypeError):
                        continue
                        
            if len(mas) < 3:
                return 0.3  # Score neutre si pas assez de MA
                
            # Calcul alignement (ordre croissant ou décroissant)
            ma_values = list(mas.values())
            ma_values.sort()
            
            # Vérifier ordre des MA (rapides vs lentes)
            ma_order_score = 0.0
            total_comparisons = 0
            
            # Comparaisons entre MA rapides et lentes
            fast_mas = [v for k, v in mas.items() if k in ['ema_7', 'ema_12']]
            slow_mas = [v for k, v in mas.items() if k in ['ema_50', 'ema_99', 'sma_50']]
            
            if fast_mas and slow_mas:
                for fast in fast_mas:
                    for slow in slow_mas:
                        total_comparisons += 1
                        # Alignement haussier: MA rapides > MA lentes
                        # Alignement baissier: MA rapides < MA lentes
                        if abs(fast - slow) > abs(fast + slow) * 0.001:  # Différence significative
                            ma_order_score += 1
                            
            if total_comparisons > 0:
                return min(1.0, ma_order_score / total_comparisons)
            else:
                return 0.5  # Score neutre
                
        except Exception:
            return 0.3  # Score de sécurité
            
    def _validate_directional_consensus(self, signal_side: str, directional_bias: str, 
                                       trend_strength: float) -> bool:
        """Valide le consensus directionnel."""
        consensus_factors = 0
        total_factors = 0
        
        # Facteur 1: directional_bias
        if directional_bias:
            total_factors += 1
            if (signal_side == "BUY" and directional_bias == "bullish") or \
               (signal_side == "SELL" and directional_bias == "bearish"):
                consensus_factors += 1
                
        # Facteur 2: trend_strength
        if trend_strength is not None:
            total_factors += 1
            if trend_strength > self.trend_consistency_min:
                consensus_factors += 1
                
        # Facteur 3: alignement général (basé sur MA)
        ma_alignment = self._calculate_ma_alignment()
        total_factors += 1
        if ma_alignment > 0.6:
            consensus_factors += 1
            
        if total_factors == 0:
            return True  # Pas d'éléments pour juger
            
        consensus_ratio = consensus_factors / total_factors
        return consensus_ratio >= 0.6  # 60% de consensus minimum
        
    def _detect_critical_divergences(self, consensus_score: float, tf_alignment: float,
                                    trend_alignment: float, signal_strength: float) -> bool:
        """Détecte les divergences critiques entre indicateurs."""
        scores = [s for s in [consensus_score, tf_alignment, trend_alignment, signal_strength] 
                 if s is not None]
        
        if len(scores) < 2:
            return False
            
        # Calcul écart-type pour détecter divergences
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Divergence critique si écart-type > seuil
        return std_dev > self.critical_divergence_threshold
        
    def _is_multitf_strategy(self, strategy_name: str) -> bool:
        """Détermine si la stratégie est spécifiquement multi-timeframes."""
        multitf_keywords = ['multitf', 'multi_tf', 'confluence', 'consensus', 'timeframe']
        strategy_lower = strategy_name.lower()
        return any(keyword in strategy_lower for keyword in multitf_keywords)
        
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur le consensus multi-timeframes.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur consensus multi-TF
            # consensus_score → confluence_score (score de confluence)
            consensus_score = float(self.context.get('confluence_score', 0.5)) if self.context.get('confluence_score') is not None else 0.5
            # tf_alignment → trend_alignment (alignement de tendance)
            tf_alignment = float(self.context.get('trend_alignment', 0.5)) if self.context.get('trend_alignment') is not None else 0.5
            trend_alignment = float(self.context.get('trend_alignment', 0.5)) if self.context.get('trend_alignment') is not None else 0.5
            # Ligne 301: Vérification None et type avant conversion
            signal_strength_raw = self.context.get('signal_strength')
            signal_strength = self._convert_signal_strength_to_score(str(signal_strength_raw)) if signal_strength_raw is not None else 0.5
            confluence_score = float(self.context.get('confluence_score', 50.0)) if self.context.get('confluence_score') is not None else 50.0
            
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', '')
            directional_bias = self.context.get('directional_bias')
            # Ligne 307: Vérification None avant conversion
            trend_strength_raw = self.context.get('trend_strength')
            trend_strength = self._convert_trend_strength_to_score(trend_strength_raw) if trend_strength_raw is not None else 0.5
            
            base_score = 0.5  # Score de base si validé
            
            # Bonus consensus score
            if consensus_score >= self.perfect_alignment_threshold:
                base_score += self.perfect_consensus_bonus
            elif consensus_score >= self.strong_consensus_threshold:
                base_score += 0.20
            elif consensus_score >= self.min_consensus_score:
                base_score += 0.10
                
            # Bonus alignement TF
            if tf_alignment >= self.perfect_alignment_threshold:
                base_score += self.strong_alignment_bonus
            elif tf_alignment >= self.strong_consensus_threshold:
                base_score += 0.15
            elif tf_alignment >= self.min_tf_alignment:
                base_score += 0.08
                
            # Bonus alignement tendance
            if trend_alignment >= self.perfect_alignment_threshold:
                base_score += 0.18
            elif trend_alignment >= self.strong_consensus_threshold:
                base_score += 0.12
            elif trend_alignment >= self.min_ma_alignment:
                base_score += 0.08
                
            # Bonus alignement MA calculé
            ma_alignment_score = self._calculate_ma_alignment()
            if ma_alignment_score >= 0.9:
                base_score += self.ma_confluence_bonus
            elif ma_alignment_score >= 0.8:
                base_score += 0.10
            elif ma_alignment_score >= 0.7:
                base_score += 0.05
                
            # Bonus force signal
            if signal_strength >= 0.8:
                base_score += 0.12
            elif signal_strength >= 0.65:
                base_score += 0.08
                
            # Bonus consensus directionnel
            # Ligne 352: Vérification None et type avant appel
            directional_consensus = self._validate_directional_consensus(
                str(signal_side) if signal_side is not None else "UNKNOWN", str(directional_bias) if directional_bias is not None else "neutral", trend_strength
            )
            if directional_consensus:
                base_score += 0.10
                
            # Bonus confluence générale
            if confluence_score >= 80.0:
                base_score += 0.10
            elif confluence_score >= 60.0:
                base_score += 0.05
                
            # Bonus stratégie spécialisée multi-TF
            if self._is_multitf_strategy(signal_strategy):
                base_score += 0.12  # Bonus spécialisation
                
            # Malus divergences critiques
            critical_divergence = self._detect_critical_divergences(
                consensus_score, tf_alignment, trend_alignment, signal_strength
            )
            if critical_divergence:
                base_score += self.divergence_penalty
                
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
            # consensus_score → confluence_score (score de confluence)
            consensus_score = float(self.context.get('confluence_score', 0)) if self.context.get('confluence_score') is not None else None
            # tf_alignment → trend_alignment (alignement de tendance)
            tf_alignment = float(self.context.get('trend_alignment', 0)) if self.context.get('trend_alignment') is not None else None
            trend_alignment = float(self.context.get('trend_alignment', 0)) if self.context.get('trend_alignment') is not None else None
            # Ligne 395: Vérification None et type avant conversion
            signal_strength_raw = self.context.get('signal_strength')
            signal_strength = self._convert_signal_strength_to_score(str(signal_strength_raw)) if signal_strength_raw is not None else None
            signal_side = signal.get('side', 'N/A')
            signal_strategy = signal.get('strategy', 'N/A')
            
            if is_valid:
                reason = f"Consensus MultiTF favorable"
                
                scores = []
                if consensus_score is not None:
                    scores.append(f"consensus: {self._safe_format(consensus_score, '.2f')}")
                if tf_alignment is not None:
                    scores.append(f"TF alignment: {self._safe_format(tf_alignment, '.2f')}")
                if trend_alignment is not None:
                    scores.append(f"trend: {self._safe_format(trend_alignment, '.2f')}")
                if signal_strength is not None:
                    scores.append(f"force: {self._safe_format(signal_strength, '.2f')}")
                    
                if scores:
                    reason += f" ({', '.join(scores)})"
                    
                ma_alignment = self._calculate_ma_alignment()
                reason += f", MA alignment: {self._safe_format(ma_alignment, '.2f')}"
                
                if self._is_multitf_strategy(signal_strategy):
                    reason += " - stratégie spécialisée"
                    
                return f"{self.name}: Validé - {reason} pour {signal_strategy} {signal_side}"
            else:
                if consensus_score is not None and consensus_score < self.min_consensus_score:
                    return f"{self.name}: Rejeté - Consensus insuffisant ({self._safe_format(consensus_score, '.2f')})"
                elif tf_alignment is not None and tf_alignment < self.min_tf_alignment:
                    return f"{self.name}: Rejeté - Alignement TF insuffisant ({self._safe_format(tf_alignment, '.2f')})"
                elif trend_alignment is not None and trend_alignment < self.min_ma_alignment:
                    return f"{self.name}: Rejeté - Alignement tendance insuffisant ({self._safe_format(trend_alignment, '.2f')})"
                elif signal_strength is not None and signal_strength < self.min_signal_strength:
                    return f"{self.name}: Rejeté - Force signal insuffisante ({self._safe_format(signal_strength, '.2f')})"
                    
                ma_alignment = self._calculate_ma_alignment()
                if ma_alignment < self.min_ma_alignment:
                    return f"{self.name}: Rejeté - Alignement MA insuffisant ({self._safe_format(ma_alignment, '.2f')})"
                    
                # Ligne 437: Vérification None avant appel
                critical_divergence = self._detect_critical_divergences(
                    consensus_score or 0.0, tf_alignment or 0.0, trend_alignment or 0.0, signal_strength or 0.0
                )
                if critical_divergence:
                    return f"{self.name}: Rejeté - Divergences critiques détectées"
                    
                return f"{self.name}: Rejeté - Consensus multi-timeframes insuffisant"
                
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données multi-TF requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Indicateurs multi-TF requis (au moins 2 sur 4)
        multitf_indicators = [
            'confluence_score', 'trend_alignment', 'trend_alignment', 'signal_strength'
        ]
        
        available_indicators = sum(1 for ind in multitf_indicators 
                                 if ind in self.context and self.context[ind] is not None)
        
        if available_indicators < 2:
            logger.warning(f"{self.name}: Pas assez d'indicateurs multi-TF pour {self.symbol}")
            return False
            
        # Vérifier qu'on a au moins quelques MA pour calcul alignement
        ma_indicators = ['ema_12', 'ema_26', 'ema_50', 'sma_20', 'sma_50']
        ma_available = sum(1 for ma in ma_indicators 
                          if ma in self.context and self.context[ma] is not None)
        
        if ma_available < 2:
            logger.warning(f"{self.name}: Pas assez de moyennes mobiles pour {self.symbol}")
            return False
            
        return True
