"""
Market_Structure_Validator - Validator basé sur la structure de marché globale.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging
import sys
import os

# Ajouter le chemin pour importer la classification
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
from strategy_classification import (
    get_strategy_family,
    get_regime_adjustment,
    get_min_confidence_for_regime,
    is_strategy_optimal_for_regime,
    is_strategy_acceptable_for_regime
)

logger = logging.getLogger(__name__)


class Market_Structure_Validator(BaseValidator):
    """
    Validator pour la structure de marché - filtre selon l'état général et régimes de marché.
    
    Vérifie: Régime marché, alignement tendance, confluence multi-indicateurs
    Catégorie: regime
    
    Rejette les signaux en:
    - Régimes de marché défavorables
    - Conflits entre différents timeframes
    - Structure générale contradictoire
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "Market_Structure_Validator"
        self.category = "regime"
        
        # Paramètres régimes de marché - CRYPTO ULTRA-PERMISSIF
        self.favorable_regimes = ["TRENDING_BULL", "TRENDING_BEAR", "RANGING", "BREAKOUT_BULL", "BREAKOUT_BEAR", "TRANSITION", "VOLATILE"]
        self.unfavorable_regimes = ["UNKNOWN"]  # Seulement UNKNOWN vraiment problématique
        self.regime_strength_min = 0.30     # Force minimum TRÈS RÉDUITE pour crypto (30% au lieu de 45%)
        self.regime_confidence_min = 40     # Confidence minimum TRÈS RÉDUITE pour crypto (40% au lieu de 55%)
        
        # Paramètres alignement - CRYPTO ULTRA-PERMISSIF 
        self.min_trend_alignment = 30      # Alignement minimum TRÈS RÉDUIT pour crypto ranging (30% au lieu de 50%)
        self.min_signal_strength = 0.35     # Force signal minimum TRÈS RÉDUITE pour crypto (35% au lieu de 45%)
        self.min_confluence_score = 25.0    # Score confluence minimum TRÈS RÉDUIT pour crypto (25 au lieu de 40)
        
        # Paramètres volatilité - CRYPTO PERMISSIF
        self.max_volatility_regime_risk = ["chaotic"]  # Seulement chaotic vraiment risqué
        self.acceptable_volatility = ["low", "normal", "high", "expanding", "extreme"]  # extreme acceptable crypto
        
        # Seuils directionnels - AJUSTÉS POUR RANGING
        self.directional_bias_weight = 0.3  # Poids bias directionnel RÉDUIT pour ranging (30% au lieu de 40%)
        self.trend_angle_min = 4.0          # Angle tendance minimum RÉDUIT pour ranging (4° au lieu de 8°)
        
        # Bonus/malus - OPTIMISÉS
        self.perfect_alignment_bonus = 0.30  # Bonus alignement parfait AUGMENTÉ (30% au lieu de 25%)
        self.regime_mismatch_penalty = -0.35 # Pénalité régime inadapté AUGMENTÉE (-35% au lieu de -30%)
        self.confluence_bonus = 0.25         # Bonus confluence élevée AUGMENTÉ (25% au lieu de 20%)
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur la structure globale du marché.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon structure marché, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs de structure depuis le contexte
            try:
                # Régimes principaux
                market_regime = self.context.get('market_regime')
                regime_strength_raw = self.context.get('regime_strength')
                regime_strength = self._convert_regime_strength_to_score(str(regime_strength_raw)) if regime_strength_raw is not None else None
                regime_confidence = float(self.context.get('regime_confidence', 50.0)) if self.context.get('regime_confidence') is not None else None
                volatility_regime = self.context.get('volatility_regime')
                
                # Alignement et confluence
                trend_alignment = float(self.context.get('trend_alignment', 0)) if self.context.get('trend_alignment') is not None else None
                signal_strength_raw = self.context.get('signal_strength')
                signal_strength = self._convert_signal_strength_to_score(str(signal_strength_raw)) if signal_strength_raw is not None else None
                confluence_score = float(self.context.get('confluence_score', 50.0)) if self.context.get('confluence_score') is not None else None
                
                # Direction et tendance
                directional_bias = self.context.get('directional_bias')
                trend_strength_raw = self.context.get('trend_strength')
                trend_strength = self._convert_trend_strength_to_score(str(trend_strength_raw)) if trend_strength_raw is not None else None
                trend_angle = float(self.context.get('trend_angle', 0)) if self.context.get('trend_angle') is not None else None
                
                # Pattern recognition
                pattern_detected = self.context.get('pattern_detected')
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
                
            # 1. Validation régime de marché principal - AJUSTÉ POUR RANGING
            if market_regime:
                if market_regime in self.unfavorable_regimes:
                    logger.debug(f"{self.name}: Régime défavorable ({market_regime or 'N/A'}) pour {self.symbol}")
                    # Accepter seulement avec confidence élevée
                    if signal_confidence < 0.65:  # Réduit pour crypto défavorables
                        return False
                elif market_regime == "RANGING":
                    # Régime ranging - PERMISSIF pour crypto flat
                    if signal_confidence < 0.45:  # Très permissif pour ranging
                        logger.debug(f"{self.name}: Régime ranging + confidence très faible pour {self.symbol}")
                        return False
                elif market_regime not in self.favorable_regimes:
                    # Régime neutre/inconnu
                    if signal_confidence < 0.55:  # Réduit pour crypto (était 60%)
                        logger.debug(f"{self.name}: Régime neutre ({market_regime or 'N/A'}) + confidence faible pour {self.symbol}")
                        return False
                        
            # 2. Validation force et confidence du régime - AJUSTÉ POUR CRYPTO
            if regime_strength is not None and regime_strength < self.regime_strength_min:
                logger.debug(f"{self.name}: Force régime insuffisante ({self._safe_format(regime_strength, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.55:  # Réduit pour crypto (était 65%)
                    return False
                    
            if regime_confidence is not None and regime_confidence < self.regime_confidence_min:
                logger.debug(f"{self.name}: Confidence régime insuffisante ({self._safe_format(regime_confidence, '.0f')}%) pour {self.symbol}")
                if signal_confidence < 0.55:  # Réduit pour crypto (était 65%)
                    return False
                    
            # 3. Validation régime de volatilité - CRYPTO ULTRA-PERMISSIF
            if volatility_regime in self.max_volatility_regime_risk:
                logger.debug(f"{self.name}: Volatilité chaotique ({volatility_regime or 'N/A'}) pour {self.symbol}")
                if signal_confidence < 0.60:  # Encore plus permissif pour crypto (60% au lieu de 70%)
                    return False
            # Toutes les autres volatilités sont acceptables en crypto
            # Volatilité "extreme" est normale en crypto pendant les mouvements
                    
            # 4. Validation alignement tendance (format décimal) - AJUSTÉ POUR RANGING
            if trend_alignment is not None and abs(trend_alignment) < (self.min_trend_alignment / 100):
                logger.debug(f"{self.name}: Alignement tendance insuffisant ({self._safe_format(abs(trend_alignment), '.3f')}) pour {self.symbol}")
                # Plus permissif en marché ranging où l'alignement est naturellement faible
                if signal_confidence < 0.45:  # Très permissif pour faible alignement (était 55%)
                    return False
                    
            # 5. Validation force signal globale - AJUSTÉ POUR RANGING
            if signal_strength is not None and signal_strength < self.min_signal_strength:
                logger.debug(f"{self.name}: Force signal insuffisante ({self._safe_format(signal_strength, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.50:  # Plus permissif pour signal faible (était 65%)
                    return False
                    
            # 6. Validation score confluence - AJUSTÉ POUR RANGING  
            if confluence_score is not None and confluence_score < self.min_confluence_score:
                logger.debug(f"{self.name}: Score confluence insuffisant ({self._safe_format(confluence_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.45:  # Plus permissif pour faible confluence (était 55%)
                    return False
                    
            # 7. Validation cohérence bias directionnel AVEC RÉGIME
            # Appliquer les ajustements de confidence basés sur le régime
            regime_adjustment = get_regime_adjustment(signal_strategy, market_regime, signal_side)
            adjusted_confidence = signal_confidence * regime_adjustment
            
            # Obtenir le seuil minimum pour ce régime/direction
            min_required_confidence = get_min_confidence_for_regime(market_regime, signal_side)
            
            if directional_bias:
                if signal_side == "BUY" and directional_bias.upper() == "BEARISH":
                    logger.debug(f"{self.name}: BUY signal mais bias bearish pour {self.symbol}")
                    # Plus strict si contre-bias ET contre-régime
                    if adjusted_confidence < max(0.75, min_required_confidence * 1.2):
                        logger.debug(f"{self.name}: Confidence ajustée insuffisante ({adjusted_confidence:.2f})")
                        return False
                elif signal_side == "SELL" and directional_bias.upper() == "BULLISH":
                    logger.debug(f"{self.name}: SELL signal mais bias bullish pour {self.symbol}")
                    # Plus strict si contre-bias ET contre-régime
                    if adjusted_confidence < max(0.75, min_required_confidence * 1.2):
                        logger.debug(f"{self.name}: Confidence ajustée insuffisante ({adjusted_confidence:.2f})")
                        return False
                elif directional_bias.upper() == "NEUTRAL":
                    # NEUTRAL = utiliser seulement l'ajustement de régime
                    if adjusted_confidence < min_required_confidence:
                        logger.debug(f"{self.name}: Confidence insuffisante pour régime {market_regime}: {adjusted_confidence:.2f} < {min_required_confidence:.2f}")
                        return False
            else:
                # Pas de bias directionnel, utiliser le seuil de régime
                if adjusted_confidence < min_required_confidence:
                    logger.debug(f"{self.name}: Confidence insuffisante pour régime {market_regime}: {adjusted_confidence:.2f} < {min_required_confidence:.2f}")
                    return False
                        
            # 8. Validation force tendance générale - PLUS STRICT
            if trend_strength is not None and trend_strength < 0.4:  # AUGMENTÉ de 0.3 à 0.4
                logger.debug(f"{self.name}: Tendance générale faible ({self._safe_format(trend_strength, '.2f')}) pour {self.symbol}")
                # En tendance faible, favoriser stratégies de mean reversion
                if not self._is_meanreversion_strategy(signal_strategy):
                    if signal_confidence < 0.60:  # Réduit pour crypto (était 70%)
                        return False
            # NOUVEAU: Rejet si tendance très faible - MAIS adapter selon directional_bias
            if trend_strength is not None and trend_strength < 0.2:
                logger.debug(f"{self.name}: Tendance très faible ({self._safe_format(trend_strength, '.2f')}) pour {self.symbol}")
                # Si directional_bias cohérent avec signal, être plus permissif
                bias_coherent = False
                if directional_bias:
                    bias_coherent = ((signal_side == "SELL" and directional_bias.upper() == "BEARISH") or
                                   (signal_side == "BUY" and directional_bias.upper() == "BULLISH") or
                                   directional_bias.upper() == "NEUTRAL")  # NEUTRAL = toujours permissif
                
                threshold = 0.60 if bias_coherent else 0.70
                if signal_confidence < threshold:
                    return False
                        
            # 9. Validation angle tendance - PLUS STRICT
            if trend_angle is not None and abs(trend_angle) < self.trend_angle_min:
                logger.debug(f"{self.name}: Angle tendance faible ({self._safe_format(trend_angle, '.1f')}°) pour {self.symbol}")
                if signal_confidence < 0.55:  # Réduit pour crypto (était 60%)
                    return False
                    
            # 10. Validation pattern si détecté - PLUS STRICT
            if pattern_detected and pattern_confidence is not None:
                if pattern_confidence < 60:  # AUGMENTÉ de 50% à 60%
                    logger.debug(f"{self.name}: Pattern {pattern_detected} confidence faible ({self._safe_format(pattern_confidence, '.2f')}) pour {self.symbol}")
                    if signal_confidence < 0.65:  # Réduit pour crypto (était 75%)
                        return False
                        
            # 11. Validation croisée régimes - PLUS STRICT
            regime_conflict = self._detect_regime_conflicts(str(market_regime) if market_regime is not None else '', str(volatility_regime) if volatility_regime is not None else '', str(directional_bias) if directional_bias is not None else '')
            if regime_conflict:
                logger.debug(f"{self.name}: Conflit entre régimes détecté pour {self.symbol}")
                if signal_confidence < 0.70:  # Réduit pour crypto (était 85%)
                    return False
                    
            # 12. Validation spécifique selon type stratégie - ADAPTATIF AU RÉGIME
            strategy_regime_match = self._validate_strategy_regime_match(
                signal_strategy, str(market_regime) if market_regime is not None else '', str(volatility_regime) if volatility_regime is not None else ''
            )
            if not strategy_regime_match:
                logger.warning(f"{self.name}: Stratégie {signal_strategy} INADAPTÉE au régime {market_regime} pour {self.symbol}")
                # Rejet plus strict pour inadéquation stratégie/régime
                if adjusted_confidence < 0.85:  # Très strict si mauvaise combinaison
                    return False
                    
            # Validation finale - structure globale AJUSTÉE POUR RANGING
            overall_structure_quality = self._calculate_structure_quality(regime_strength, regime_confidence, trend_alignment, confluence_score)
            # Plus permissif : structure médiocre acceptable en ranging avec confidence modérée
            if overall_structure_quality < 0.4 and signal_confidence < 0.55:
                logger.debug(f"{self.name}: Structure très médiocre ({overall_structure_quality:.2f}) + signal confidence insuffisante pour {self.symbol}")
                return False
                    
            logger.info(f"{self.name}: Signal validé pour {self.symbol} - "
                       f"Stratégie: {signal_strategy} ({get_strategy_family(signal_strategy)}), "
                       f"Régime: {market_regime or 'N/A'}, "
                       f"Ajustement: x{regime_adjustment:.2f}, "
                       f"Conf ajustée: {adjusted_confidence:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def _detect_regime_conflicts(self, market_regime: str, volatility_regime: str, 
                                directional_bias: str) -> bool:
        """Détecte les conflits entre différents régimes."""
        conflicts = 0
        
        # Conflit market vs volatility
        if market_regime in ["TRENDING_BULL", "TRENDING_BEAR"] and volatility_regime == "low":
            conflicts += 1  # Trending avec volatilité faible = conflit
        elif market_regime == "RANGING" and volatility_regime == "extreme":
            conflicts += 1  # Ranging avec volatilité extrême = conflit
            
        # Conflit directional bias vs market regime
        if market_regime in ["TRENDING_BULL", "TRENDING_BEAR"] and directional_bias == "NEUTRAL":
            conflicts += 1  # Trending mais bias neutre = conflit
            
        return conflicts >= 2  # Conflit si 2+ incohérences
        
    def _validate_strategy_regime_match(self, strategy: str, market_regime: str, 
                                       volatility_regime: str) -> bool:
        """Valide l'adéquation stratégie/régime avec classification intelligente."""
        
        # Utiliser la nouvelle classification
        is_acceptable = is_strategy_acceptable_for_regime(strategy, market_regime)
        
        # Si la stratégie n'est pas dans les régimes acceptables, on vérifie la volatilité
        if not is_acceptable:
            # Certaines stratégies peuvent être bloquées par une volatilité extrême
            family = get_strategy_family(strategy)
            
            # Les stratégies de breakout ont besoin de volatilité
            if family == 'breakout' and volatility_regime in ['low', 'compression']:
                logger.debug(f"Stratégie breakout {strategy} inadaptée en volatilité {volatility_regime}")
                return False
                
            # Les stratégies de mean reversion n'aiment pas la volatilité extrême
            if family == 'mean_reversion' and volatility_regime in ['extreme', 'chaotic']:
                logger.debug(f"Stratégie mean-reversion {strategy} inadaptée en volatilité {volatility_regime}")
                return False
                
        return is_acceptable
        
    def _is_meanreversion_strategy(self, strategy_name: str) -> bool:
        """Détermine si la stratégie est de type mean reversion."""
        family = get_strategy_family(strategy_name)
        return family == 'mean_reversion'
        
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur la structure globale du marché.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur structure marché
            market_regime = self.context.get('market_regime')
            regime_strength_raw = self.context.get('regime_strength')
            regime_strength = self._convert_regime_strength_to_score(str(regime_strength_raw)) if regime_strength_raw is not None else 0.5
            regime_confidence = float(self.context.get('regime_confidence', 50.0)) if self.context.get('regime_confidence') is not None else 50.0
            volatility_regime = self.context.get('volatility_regime')
            trend_alignment = float(self.context.get('trend_alignment', 0.5)) if self.context.get('trend_alignment') is not None else 0.5
            signal_strength_raw = self.context.get('signal_strength')
            signal_strength = self._convert_signal_strength_to_score(str(signal_strength_raw)) if signal_strength_raw is not None else 0.5
            confluence_score = float(self.context.get('confluence_score', 50.0)) if self.context.get('confluence_score') is not None else 50.0
            directional_bias = self.context.get('directional_bias')
            
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', '')
            
            base_score = 0.45  # Score de base réduit (45% au lieu de 50%)
            
            # Bonus régime favorable
            if market_regime in self.favorable_regimes:
                base_score += 0.15
                if market_regime in ["TRENDING_BULL", "TRENDING_BEAR"]:
                    base_score += 0.05  # Bonus supplémentaire trending
            elif market_regime in self.unfavorable_regimes:
                base_score -= 0.10  # Pénalité régime défavorable
                
            # Bonus force et confidence régime (normaliser regime_confidence 0-100 vers 0-1)
            regime_quality = (regime_strength + (regime_confidence / 100)) / 2
            if regime_quality >= 0.8:
                base_score += 0.12
            elif regime_quality >= 0.6:
                base_score += 0.08
                
            # Bonus volatilité favorable
            if volatility_regime in self.acceptable_volatility:
                base_score += 0.08
                if volatility_regime == "normal":
                    base_score += 0.04  # Bonus supplémentaire normal
                    
            # Bonus alignement tendance (format décimal)
            alignment_abs = abs(trend_alignment)
            if alignment_abs >= 0.8:
                base_score += self.perfect_alignment_bonus
            elif alignment_abs >= 0.7:
                base_score += 0.15
            elif alignment_abs >= 0.6:
                base_score += 0.10
                
            # Bonus force signal
            if signal_strength >= 0.8:
                base_score += 0.12
            elif signal_strength >= 0.6:
                base_score += 0.08
                
            # Bonus confluence
            if confluence_score >= 80.0:
                base_score += self.confluence_bonus
            elif confluence_score >= 60.0:
                base_score += 0.10
                
            # CORRECTION: Bonus/Malus bias directionnel selon cohérence
            if directional_bias:
                if (signal_side == "BUY" and directional_bias.upper() == "BULLISH") or \
                   (signal_side == "SELL" and directional_bias.upper() == "BEARISH"):
                    base_score += 0.15  # Bonus cohérence directionnelle
                elif (signal_side == "BUY" and directional_bias.upper() == "BEARISH") or \
                     (signal_side == "SELL" and directional_bias.upper() == "BULLISH"):
                    base_score -= 0.25  # Malus fort pour incohérence directionnelle
                # Si directional_bias == "NEUTRAL", pas de bonus ni malus
                    
            # Bonus adéquation stratégie/régime
            strategy_match = self._validate_strategy_regime_match(
                signal_strategy, str(market_regime) if market_regime is not None else '', str(volatility_regime) if volatility_regime is not None else ''
            )
            if strategy_match:
                base_score += 0.10
                
            # Malus conflits régimes
            regime_conflict = self._detect_regime_conflicts(
                str(market_regime) if market_regime is not None else '', str(volatility_regime) if volatility_regime is not None else '', str(directional_bias) if directional_bias is not None else ''
            )
            if regime_conflict:
                base_score += self.regime_mismatch_penalty
                
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
            market_regime = self.context.get('market_regime', 'N/A')
            volatility_regime = self.context.get('volatility_regime', 'N/A')
            directional_bias = self.context.get('directional_bias', 'N/A')
            trend_alignment = float(self.context.get('trend_alignment', 0)) if self.context.get('trend_alignment') is not None else None
            confluence_score = float(self.context.get('confluence_score', 50.0)) if self.context.get('confluence_score') is not None else None
            signal_side = signal.get('side', 'N/A')
            signal_strategy = signal.get('strategy', 'N/A')
            
            if is_valid:
                reason = f"Structure favorable"
                
                if market_regime != 'N/A':
                    reason += f" (régime: {market_regime})"
                if volatility_regime != 'N/A':
                    reason += f", volatilité: {volatility_regime}"
                if directional_bias != 'N/A':
                    reason += f", bias: {directional_bias}"
                if trend_alignment:
                    reason += f", alignment: {self._safe_format(trend_alignment, '.2f')}"
                if confluence_score:
                    reason += f", confluence: {self._safe_format(confluence_score, '.2f')}"
                    
                strategy_match = self._validate_strategy_regime_match(
                    signal_strategy, market_regime, volatility_regime
                )
                if strategy_match:
                    reason += " - stratégie adaptée"
                    
                return f"{self.name}: Validé - {reason} pour {signal_strategy} {signal_side}"
            else:
                if market_regime in self.unfavorable_regimes:
                    return f"{self.name}: Rejeté - Régime défavorable ({market_regime or 'N/A'})"
                elif volatility_regime in self.max_volatility_regime_risk:
                    return f"{self.name}: Rejeté - Volatilité risquée ({volatility_regime or 'N/A'})"
                elif directional_bias and signal_side:
                    if (signal_side == "BUY" and directional_bias.upper() == "BEARISH") or \
                       (signal_side == "SELL" and directional_bias.upper() == "BULLISH"):
                        return f"{self.name}: Rejeté - Signal {signal_side} contradictoire avec bias {directional_bias or 'N/A'}"
                elif trend_alignment and abs(trend_alignment) < (self.min_trend_alignment / 100):
                    return f"{self.name}: Rejeté - Alignement tendance insuffisant ({self._safe_format(abs(trend_alignment), '.3f')})"
                elif confluence_score and confluence_score < self.min_confluence_score:
                    return f"{self.name}: Rejeté - Confluence insuffisante ({self._safe_format(confluence_score, '.2f')})"
                    
                strategy_match = self._validate_strategy_regime_match(
                    signal_strategy, market_regime, volatility_regime
                )
                if not strategy_match:
                    return f"{self.name}: Rejeté - Stratégie {signal_strategy} inadaptée au régime {market_regime or 'N/A'}"
                    
                return f"{self.name}: Rejeté - Structure de marché défavorable"
                
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def _calculate_structure_quality(self, regime_strength: float, regime_confidence: float, 
                                   trend_alignment: float, confluence_score: float) -> float:
        """Calcule la qualité globale de la structure de marché."""
        try:
            quality_score = 0.0
            components = 0
            
            if regime_strength is not None:
                quality_score += regime_strength
                components += 1
                
            if regime_confidence is not None:
                quality_score += (regime_confidence / 100)  # Normaliser 0-100 vers 0-1
                components += 1
                
            if trend_alignment is not None:
                quality_score += abs(trend_alignment)  # Alignement absolu
                components += 1
                
            if confluence_score is not None:
                quality_score += (confluence_score / 100)  # Normaliser 0-100 vers 0-1
                components += 1
                
            return quality_score / max(1, components)  # Moyenne des composants disponibles
        except:
            return 0.5  # Valeur par défaut
            
    def validate_data(self) -> bool:
        """
        Valide que les données de structure requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Au minimum, on a besoin d'un indicateur de régime ou structure
        structure_indicators = [
            'market_regime', 'trend_alignment', 'signal_strength', 
            'confluence_score', 'directional_bias'
        ]
        
        available_indicators = sum(1 for ind in structure_indicators 
                                 if ind in self.context and self.context[ind] is not None)
        
        if available_indicators < 2:
            logger.warning(f"{self.name}: Pas assez d'indicateurs de structure pour {self.symbol}")
            return False
            
        return True
