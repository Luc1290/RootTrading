"""
Market_Structure_Validator - Validator basé sur la structure de marché globale.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

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
        
        # Paramètres régimes de marché - AJUSTÉS POUR CRYPTO RANGING
        self.favorable_regimes = ["trending", "expansion", "normal", "ranging"]  # RANGING ajouté aux favorables
        self.unfavorable_regimes = ["compression", "chaotic", "extreme_volatility"]  # RANGING retiré
        self.regime_strength_min = 0.45     # Force minimum régime RÉDUITE pour crypto (45% au lieu de 55%)
        self.regime_confidence_min = 55     # Confidence minimum régime RÉDUITE pour crypto (55% au lieu de 65%)
        
        # Paramètres alignement - AJUSTÉS POUR RANGING
        self.min_trend_alignment = 50      # Alignement minimum tendance RÉDUIT pour ranging (50% au lieu de 70%)
        self.min_signal_strength = 0.45     # Force signal minimum RÉDUITE pour ranging (45% au lieu de 60%)
        self.min_confluence_score = 40.0    # Score confluence minimum RÉDUIT pour ranging (40 au lieu de 55)
        
        # Paramètres volatilité - OPTIMISÉS
        self.max_volatility_regime_risk = ["extreme", "chaotic"]
        self.acceptable_volatility = ["low", "normal", "high", "expanding"]
        
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
                elif market_regime == "ranging":
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
                    
            # 3. Validation régime de volatilité - AJUSTÉ POUR CRYPTO FLAT
            if volatility_regime in self.max_volatility_regime_risk:
                logger.debug(f"{self.name}: Régime volatilité risqué ({volatility_regime or 'N/A'}) pour {self.symbol}")
                if signal_confidence < 0.70:  # Réduit pour crypto (était 75%)
                    return False
            # Volatilité faible acceptable en marché ranging
            elif volatility_regime == "low":
                if signal_confidence < 0.40:  # Très permissif pour volatilité faible (était 50%)
                    logger.debug(f"{self.name}: Volatilité faible + signal très faible pour {self.symbol}")
                    return False
                    
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
                    
            # 7. Validation cohérence bias directionnel - RENFORCÉ
            if directional_bias:
                if signal_side == "BUY" and directional_bias.upper() == "BEARISH":
                    logger.debug(f"{self.name}: BUY signal mais bias bearish pour {self.symbol}")
                    if signal_confidence < 0.70:  # Réduit pour crypto (était 85%)
                        return False
                elif signal_side == "SELL" and directional_bias.upper() == "BULLISH":
                    logger.debug(f"{self.name}: SELL signal mais bias bullish pour {self.symbol}")
                    if signal_confidence < 0.70:  # Réduit pour crypto (était 85%)
                        return False
                        
            # 8. Validation force tendance générale - PLUS STRICT
            if trend_strength is not None and trend_strength < 0.4:  # AUGMENTÉ de 0.3 à 0.4
                logger.debug(f"{self.name}: Tendance générale faible ({self._safe_format(trend_strength, '.2f')}) pour {self.symbol}")
                # En tendance faible, favoriser stratégies de mean reversion
                if not self._is_meanreversion_strategy(signal_strategy):
                    if signal_confidence < 0.60:  # Réduit pour crypto (était 70%)
                        return False
            # NOUVEAU: Rejet si tendance très faible pour toutes stratégies
            if trend_strength is not None and trend_strength < 0.2:
                logger.debug(f"{self.name}: Tendance très faible ({self._safe_format(trend_strength, '.2f')}) - signal trop risqué pour {self.symbol}")
                if signal_confidence < 0.70:  # Réduit pour crypto (était 80%)
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
                    
            # 12. Validation spécifique selon type stratégie - PLUS STRICT
            strategy_regime_match = self._validate_strategy_regime_match(
                signal_strategy, str(market_regime) if market_regime is not None else '', str(volatility_regime) if volatility_regime is not None else ''
            )
            if not strategy_regime_match:
                logger.debug(f"{self.name}: Stratégie {signal_strategy} inadaptée au régime pour {self.symbol}")
                if signal_confidence < 0.65:  # Réduit pour crypto (était 75%)
                    return False
                    
            # Validation finale - structure globale AJUSTÉE POUR RANGING
            overall_structure_quality = self._calculate_structure_quality(regime_strength, regime_confidence, trend_alignment, confluence_score)
            # Plus permissif : structure médiocre acceptable en ranging avec confidence modérée
            if overall_structure_quality < 0.4 and signal_confidence < 0.55:
                logger.debug(f"{self.name}: Structure très médiocre ({overall_structure_quality:.2f}) + signal confidence insuffisante pour {self.symbol}")
                return False
                    
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                        f"Régime: {market_regime or 'N/A'}, "
                        f"Volatilité: {volatility_regime or 'N/A'}, "
                        f"Bias: {directional_bias or 'N/A'}, "
                        f"Alignment: {self._safe_format(trend_alignment, '.2f') if trend_alignment is not None else 'N/A'}")
            
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
        if market_regime in ["TRENDING_BULL", "TRENDING_BEAR"] and directional_bias == "neutral":
            conflicts += 1  # Trending mais bias neutre = conflit
            
        return conflicts >= 2  # Conflit si 2+ incohérences
        
    def _validate_strategy_regime_match(self, strategy: str, market_regime: str, 
                                       volatility_regime: str) -> bool:
        """Valide l'adéquation stratégie/régime."""
        strategy_lower = strategy.lower()
        
        # MEAN REVERSION - PLUS PERMISSIF pour crypto ranging
        if any(kw in strategy_lower for kw in ['reversal', 'rebound', 'oversold', 'overbought', 'touch', 'rejection']):
            return market_regime in ["ranging", "normal", "compression", "trending"]  # Ajout trending
        elif any(kw in strategy_lower for kw in ['bollinger', 'zscore', 'stoch', 'williams', 'cci']):
            return market_regime in ["ranging", "normal", "compression", "trending"]  # Ajout trending
            
        # TREND FOLLOWING - PLUS PERMISSIF pour crypto ranging
        elif (any(kw in strategy_lower for kw in ['macd', 'slope', 'adx', 'hull', 'tema', 'trix', 'ema_cross']) 
              and 'reversal' not in strategy_lower):
            return market_regime in ["trending", "expansion", "normal", "ranging"]  # Ajout ranging
        elif (any(kw in strategy_lower for kw in ['cross', 'crossover']) 
              and not any(rev in strategy_lower for rev in ['rsi', 'stoch', 'williams', 'reversal'])):
            return market_regime in ["trending", "expansion", "normal", "ranging"]  # Ajout ranging
            
        # BREAKOUT - Volatilité élevée requise
        elif any(kw in strategy_lower for kw in ['breakout', 'donchian', 'atr', 'range_break']):
            return volatility_regime not in ["low", "compression"]
            
        # MOMENTUM/THRESHOLD - Adaptables mais préfèrent volatilité
        elif any(kw in strategy_lower for kw in ['roc_threshold', 'spike', 'pump_dump']):
            return volatility_regime in ["normal", "high", "expanding"]
            
        # LIQUIDITY SWEEP - Haute volatilité
        elif any(kw in strategy_lower for kw in ['sweep', 'liquidity']):
            return volatility_regime in ["normal", "high", "expanding"]
            
        # CONFLUENCE/MULTI-TF - Adaptables à tous régimes
        elif any(kw in strategy_lower for kw in ['confluence', 'multi', 'vwap']):
            return True  # Adaptable à tous régimes
            
        return True  # Par défaut, accepter
        
    def _is_meanreversion_strategy(self, strategy_name: str) -> bool:
        """Détermine si la stratégie est de type mean reversion."""
        meanrev_keywords = ['bollinger', 'touch', 'reversal', 'rsi', 'oversold', 'overbought', 'rebound']
        strategy_lower = strategy_name.lower()
        return any(keyword in strategy_lower for keyword in meanrev_keywords)
        
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
