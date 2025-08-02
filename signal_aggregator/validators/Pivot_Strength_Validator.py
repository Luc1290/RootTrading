"""
Pivot_Strength_Validator - Validator basé sur la force des points pivots.
"""

from typing import Dict, Any, Optional
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class Pivot_Strength_Validator(BaseValidator):
    """
    Validator pour la force des points pivots - filtre selon la qualité des niveaux S/R.
    
    Vérifie: Force pivots, proximité niveaux clés, confluence S/R
    Catégorie: structure
    
    Rejette les signaux en:
    - Pivots faibles (peu de touches/tests)
    - Distance inappropriée des niveaux clés
    - Manque de confluence entre différents pivots
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "Pivot_Strength_Validator"
        self.category = "structure"
        
        # Paramètres force pivots
        self.min_pivot_strength = 0.3       # Force minimum pivot
        self.strong_pivot_threshold = 0.7   # Pivot considéré fort
        self.min_touch_count = 2            # Nombre minimum touches/tests
        self.optimal_touch_count = 4        # Nombre optimal touches
        
        # Paramètres distance
        self.min_distance_ratio = 0.002     # 0.2% distance minimum du pivot
        self.max_distance_ratio = 0.05      # 5% distance maximum du pivot
        self.optimal_distance_ratio = 0.01  # 1% distance optimale
        
        # Paramètres confluence
        self.min_confluence_pivots = 2      # Minimum pivots en confluence
        self.confluence_proximity = 0.005   # 0.5% proximité pour confluence
        self.min_confluence_score = 40.0    # Score confluence minimum (format 0-100)
        
        # Paramètres temporels
        self.max_pivot_age_bars = 50        # Age maximum pivot (barres)
        self.min_pivot_age_bars = 3         # Age minimum pivot (barres)
        self.recent_interaction_bars = 10   # Interaction récente avec pivot
        
        # Bonus/malus
        self.strong_pivot_bonus = 0.25      # Bonus pivot très fort
        self.confluence_bonus = 0.20        # Bonus confluence multiple
        self.recent_interaction_bonus = 0.15 # Bonus interaction récente
        self.weak_pivot_penalty = -0.25     # Pénalité pivot faible
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur la force des points pivots.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon pivot strength, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs pivot depuis le contexte
            try:
                # Pivots support principaux
                # pivot_support → nearest_support
                pivot_support = float(self.context.get('nearest_support', 0)) if self.context.get('nearest_support') is not None else None
                # pivot_support_strength → support_strength
                pivot_support_strength = float(self.context.get('support_strength', 0)) if self.context.get('support_strength') is not None else None
                # support_touch_count → volume_buildup_periods
                support_touch_count = int(self.context.get('volume_buildup_periods', 0)) if self.context.get('volume_buildup_periods') is not None else None
                
                # Pivots résistance principaux
                # pivot_resistance → nearest_resistance
                pivot_resistance = float(self.context.get('nearest_resistance', 0)) if self.context.get('nearest_resistance') is not None else None
                # pivot_resistance_strength → resistance_strength
                pivot_resistance_strength = float(self.context.get('resistance_strength', 0)) if self.context.get('resistance_strength') is not None else None
                # resistance_touch_count → volume_buildup_periods
                resistance_touch_count = int(self.context.get('volume_buildup_periods', 0)) if self.context.get('volume_buildup_periods') is not None else None
                
                # Confluence et niveaux multiples
                confluence_score = float(self.context.get('confluence_score', 50.0)) if self.context.get('confluence_score') is not None else None
                # pivot_confluence_count → pivot_count (existe déjà!)
                pivot_confluence_count = int(self.context.get('pivot_count', 0)) if self.context.get('pivot_count') is not None else None
                
                # Age et interaction temporelle
                # pivot_age_bars → regime_duration
                pivot_age_bars = int(self.context.get('regime_duration', 0)) if self.context.get('regime_duration') is not None else None
                # last_interaction_bars → regime_duration
                last_interaction_bars = int(self.context.get('regime_duration', 999)) if self.context.get('regime_duration') is not None else None
                
                # Niveaux généraux S/R
                nearest_support = float(self.context.get('nearest_support', 0)) if self.context.get('nearest_support') is not None else None
                nearest_resistance = float(self.context.get('nearest_resistance', 0)) if self.context.get('nearest_resistance') is not None else None
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion indicateurs pour {self.symbol}: {e}")
                return False
                
                        # Récupération prix actuel depuis data
            current_price = None
            if self.data:
                # Essayer d'abord la valeur scalaire (format préféré)
                if 'close' in self.data and self.data['close'] is not None:
                    try:
                        if isinstance(self.data['close'], (int, float)):
                            current_price = float(self.data['close'])
                        elif isinstance(self.data['close'], list) and len(self.data['close']) > 0:
                            current_price = self._get_current_price()
                    except (IndexError, ValueError, TypeError):
                        pass
                
                # Fallback: essayer current_price dans le contexte
                if current_price is None:
                    # current_price n'est pas dans analyzer_data, utiliser self.data['close']
                    if current_price is not None:
                        try:
                            current_price = float(current_price)
                        except (ValueError, TypeError):
                            current_price = None
                    
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', 'Unknown')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side or current_price is None:
                logger.warning(f"{self.name}: Signal side ou prix manquant pour {self.symbol}")
                return False
                
            # 1. Validation force pivots selon direction signal
            if signal_side == "BUY":
                # Pour BUY, vérifier pivot support
                if pivot_support_strength is not None and pivot_support_strength < self.min_pivot_strength:
                    logger.debug(f"{self.name}: BUY signal mais pivot support faible ({self._safe_format(pivot_support_strength, '.2f')}) pour {self.symbol}")
                    if signal_confidence < 0.8:
                        return False
                        
                # Vérification touches support
                if support_touch_count is not None and support_touch_count < self.min_touch_count:
                    logger.debug(f"{self.name}: BUY signal mais support touches insuffisantes ({support_touch_count}) pour {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                        
            elif signal_side == "SELL":
                # Pour SELL, vérifier pivot résistance
                if pivot_resistance_strength is not None and pivot_resistance_strength < self.min_pivot_strength:
                    logger.debug(f"{self.name}: SELL signal mais pivot résistance faible ({self._safe_format(pivot_resistance_strength, '.2f')}) pour {self.symbol}")
                    if signal_confidence < 0.8:
                        return False
                        
                # Vérification touches résistance
                if resistance_touch_count is not None and resistance_touch_count < self.min_touch_count:
                    logger.debug(f"{self.name}: SELL signal mais résistance touches insuffisantes ({resistance_touch_count}) pour {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                        
            # 2. Validation distance par rapport aux pivots
            relevant_pivot = pivot_support if signal_side == "BUY" else pivot_resistance
            if relevant_pivot is not None:
                distance_ratio = abs(current_price - relevant_pivot) / current_price
                
                if distance_ratio < self.min_distance_ratio:
                    logger.debug(f"{self.name}: Trop proche du pivot ({self._safe_format(distance_ratio*100, '.2f')}%) pour {self.symbol}")
                    if signal_confidence < 0.6:
                        return False
                elif distance_ratio > self.max_distance_ratio:
                    logger.debug(f"{self.name}: Trop loin du pivot ({self._safe_format(distance_ratio*100, '.2f')}%) pour {self.symbol}")
                    if signal_confidence < 0.5:
                        return False
                        
            # 3. Validation confluence pivots
            if confluence_score is not None and confluence_score < self.min_confluence_score:
                logger.debug(f"{self.name}: Score confluence pivots insuffisant ({self._safe_format(confluence_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            if pivot_confluence_count is not None and pivot_confluence_count < self.min_confluence_pivots:
                logger.debug(f"{self.name}: Pas assez de pivots en confluence ({pivot_confluence_count}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 4. Validation age des pivots
            if pivot_age_bars is not None:
                if pivot_age_bars < self.min_pivot_age_bars:
                    logger.debug(f"{self.name}: Pivot trop récent ({pivot_age_bars} barres) pour {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                elif pivot_age_bars > self.max_pivot_age_bars:
                    logger.debug(f"{self.name}: Pivot trop ancien ({pivot_age_bars} barres) pour {self.symbol}")
                    if signal_confidence < 0.6:
                        return False
                        
            # 5. Validation interaction récente avec pivots
            if last_interaction_bars is not None and last_interaction_bars > self.recent_interaction_bars * 2:
                logger.debug(f"{self.name}: Pas d'interaction récente avec pivots ({last_interaction_bars} barres) pour {self.symbol}")
                if signal_confidence < 0.5:
                    return False
                    
            # 6. Validation cohérence avec S/R généraux
            if (current_price is not None and nearest_support is not None and 
                nearest_resistance is not None and pivot_support is not None and 
                pivot_resistance is not None):
                sr_coherence = self._validate_sr_coherence(
                    signal_side, current_price, nearest_support, nearest_resistance, 
                    pivot_support, pivot_resistance
                )
            else:
                sr_coherence = True
            if not sr_coherence:
                logger.debug(f"{self.name}: Incohérence entre pivots et S/R généraux pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 7. Validation spécifique selon type stratégie
            strategy_pivot_match = self._validate_strategy_pivot_match(signal_strategy, signal_side)
            if not strategy_pivot_match:
                logger.debug(f"{self.name}: Stratégie {signal_strategy} inadaptée aux conditions pivot pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 8. Validation qualité globale structure pivot
            if (pivot_support_strength is not None and pivot_resistance_strength is not None and 
                support_touch_count is not None and resistance_touch_count is not None and 
                confluence_score is not None):
                pivot_quality = self._calculate_pivot_quality(
                    pivot_support_strength, pivot_resistance_strength, 
                    support_touch_count, resistance_touch_count, confluence_score
                )
            else:
                pivot_quality = 0.5  # valeur neutre
            if pivot_quality < 0.4:
                logger.debug(f"{self.name}: Qualité globale pivots insuffisante ({self._safe_format(pivot_quality, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                        f"Pivot {'Support' if signal_side == 'BUY' else 'Résistance'}: "
                        f"{self._safe_format(relevant_pivot, '.4f') if relevant_pivot is not None else 'N/A'}, "
                        f"Force: {self._safe_format((pivot_support_strength if signal_side == 'BUY' else pivot_resistance_strength), '.2f') if (pivot_support_strength if signal_side == 'BUY' else pivot_resistance_strength) else 'N/A'}, "
                        f"Confluence: {self._safe_format(confluence_score, '.2f') if confluence_score is not None else 'N/A'}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def _validate_sr_coherence(self, signal_side: str, current_price: float,
                              nearest_support: float, nearest_resistance: float,
                              pivot_support: float, pivot_resistance: float) -> bool:
        """Valide la cohérence entre pivots et niveaux S/R généraux."""
        try:
            if signal_side == "BUY":
                # Pour BUY, vérifier cohérence support
                if nearest_support and pivot_support:
                    # Support pivot doit être proche du support général
                    sr_difference = abs(nearest_support - pivot_support) / current_price
                    if sr_difference > 0.02:  # 2% de différence max
                        return False
                        
            elif signal_side == "SELL":
                # Pour SELL, vérifier cohérence résistance
                if nearest_resistance and pivot_resistance:
                    # Résistance pivot doit être proche de la résistance générale
                    sr_difference = abs(nearest_resistance - pivot_resistance) / current_price
                    if sr_difference > 0.02:  # 2% de différence max
                        return False
                        
            return True
        except:
            return True  # En cas d'erreur, ne pas bloquer
            
    def _validate_strategy_pivot_match(self, strategy: str, signal_side: str) -> bool:
        """Valide l'adéquation stratégie/utilisation pivots."""
        strategy_lower = strategy.lower()
        
        # Stratégies qui bénéficient particulièrement des pivots forts
        pivot_friendly = ['breakout', 'support', 'resistance', 'pivot', 'level', 'bounce', 
                         'reversal', 'rebound', 'oversold', 'overbought', 'touch', 'rejection']
        
        # Stratégies moins dépendantes des pivots  
        pivot_neutral = ['macd_crossover', 'ema_cross', 'slope', 'tema', 'hull', 'trix',
                        'adx', 'trend_following', 'moving_average']
        
        # Stratégies oscillateurs (pivot-friendly via niveaux psychologiques)
        oscillator_strategies = ['rsi', 'stoch', 'williams', 'cci', 'bollinger', 'zscore']
        
        # Classification des stratégies selon leur rapport aux pivots
        if any(kw in strategy_lower for kw in pivot_friendly):
            # Pivot-friendly : critères stricts déjà appliqués dans validate()
            return True
        elif any(kw in strategy_lower for kw in oscillator_strategies):
            # Oscillateurs : pivot-friendly via niveaux psychologiques (30/70, etc.)
            # Appliquer critères similaires aux pivot-friendly mais moins stricts
            return True
        elif any(kw in strategy_lower for kw in pivot_neutral):
            # Neutres : accepter plus facilement
            return True
            
        return True  # Par défaut accepter
        
    def _calculate_pivot_quality(self, support_strength: float, resistance_strength: float,
                                support_touches: int, resistance_touches: int,
                                confluence_score: float) -> float:
        """Calcule un score de qualité globale des pivots."""
        try:
            quality_factors = []
            
            # Facteur force support
            if support_strength is not None:
                quality_factors.append(min(1.0, support_strength / self.strong_pivot_threshold))
                
            # Facteur force résistance
            if resistance_strength is not None:
                quality_factors.append(min(1.0, resistance_strength / self.strong_pivot_threshold))
                
            # Facteur touches support
            if support_touches is not None:
                quality_factors.append(min(1.0, support_touches / self.optimal_touch_count))
                
            # Facteur touches résistance
            if resistance_touches is not None:
                quality_factors.append(min(1.0, resistance_touches / self.optimal_touch_count))
                
            # Facteur confluence
            if confluence_score is not None:
                quality_factors.append(confluence_score)
                
            if not quality_factors:
                return 0.5  # Score neutre si pas de données
                
            return sum(quality_factors) / len(quality_factors)
            
        except:
            return 0.5
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur la force des pivots.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur pivots
            # pivot_support_strength → support_strength
            pivot_support_strength = float(self.context.get('support_strength', 0.5)) if self.context.get('support_strength') is not None else 0.5
            # pivot_resistance_strength → resistance_strength
            pivot_resistance_strength = float(self.context.get('resistance_strength', 0.5)) if self.context.get('resistance_strength') is not None else 0.5
            # support_touch_count → volume_buildup_periods
            support_touch_count = int(self.context.get('volume_buildup_periods', 2)) if self.context.get('volume_buildup_periods') is not None else 2
            # resistance_touch_count → volume_buildup_periods
            resistance_touch_count = int(self.context.get('volume_buildup_periods', 2)) if self.context.get('volume_buildup_periods') is not None else 2
            confluence_score = float(self.context.get('confluence_score', 50.0)) if self.context.get('confluence_score') is not None else 50.0
            # pivot_confluence_count → pivot_count (existe déjà!)
            pivot_confluence_count = int(self.context.get('pivot_count', 1)) if self.context.get('pivot_count') is not None else 1
            # last_interaction_bars → regime_duration
            last_interaction_bars = int(self.context.get('regime_duration', 20)) if self.context.get('regime_duration') is not None else 20
            
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', '')
            
            base_score = 0.5  # Score de base si validé
            
            # CORRECTION: Bonus/Malus selon cohérence directionnelle pivot
            if signal_side == "BUY":
                # BUY: Fort support = bonus, Forte résistance proche = malus
                if pivot_support_strength >= self.strong_pivot_threshold:
                    base_score += self.strong_pivot_bonus  # Support fort = bon pour BUY
                elif pivot_support_strength >= self.min_pivot_strength + 0.2:
                    base_score += 0.15
                
                # Malus si résistance très forte proche (risque de rejet)
                if pivot_resistance_strength >= self.strong_pivot_threshold:
                    base_score -= 0.15  # Résistance forte = risque pour BUY
                    
                # Bonus touches/tests support
                if support_touch_count >= self.optimal_touch_count:
                    base_score += 0.15  # Support très testé = fiable pour BUY
                elif support_touch_count >= self.min_touch_count + 1:
                    base_score += 0.10
                    
            elif signal_side == "SELL":
                # SELL: Forte résistance = bonus, Fort support proche = malus
                if pivot_resistance_strength >= self.strong_pivot_threshold:
                    base_score += self.strong_pivot_bonus  # Résistance forte = bon pour SELL
                elif pivot_resistance_strength >= self.min_pivot_strength + 0.2:
                    base_score += 0.15
                
                # Malus si support très fort proche (risque de rebond)
                if pivot_support_strength >= self.strong_pivot_threshold:
                    base_score -= 0.15  # Support fort = risque pour SELL
                    
                # Bonus touches/tests résistance
                if resistance_touch_count >= self.optimal_touch_count:
                    base_score += 0.15  # Résistance très testée = fiable pour SELL
                elif resistance_touch_count >= self.min_touch_count + 1:
                    base_score += 0.10
                
            # Bonus confluence
            if confluence_score >= 70.0:  # Format 0-100
                base_score += self.confluence_bonus
            elif confluence_score >= 50.0:  # Format 0-100
                base_score += 0.10
                
            if pivot_confluence_count >= 3:
                base_score += 0.12  # Multiple pivots en confluence
                
            # Bonus interaction récente
            if last_interaction_bars <= self.recent_interaction_bars:
                base_score += self.recent_interaction_bonus
            elif last_interaction_bars <= self.recent_interaction_bars * 2:
                base_score += 0.08
                
            # Bonus distance optimale (calculée si pivot disponible)
            try:
                current_price = self._get_current_price() if self.data and 'close' in self.data and self.data['close'] else None
                # pivot_support → nearest_support
                pivot_support = float(self.context.get('nearest_support', 0)) if self.context.get('nearest_support') is not None else None
                # pivot_resistance → nearest_resistance
                pivot_resistance = float(self.context.get('nearest_resistance', 0)) if self.context.get('nearest_resistance') is not None else None
                
                if current_price:
                    relevant_pivot = pivot_support if signal_side == "BUY" else pivot_resistance
                    if relevant_pivot:
                        distance_ratio = abs(current_price - relevant_pivot) / current_price
                        if distance_ratio <= self.optimal_distance_ratio:
                            base_score += 0.12  # Distance parfaite
                        elif distance_ratio <= self.optimal_distance_ratio * 2:
                            base_score += 0.08  # Distance bonne
            except:
                pass
                
            # Bonus stratégie adaptée aux pivots
            strategy_lower = signal_strategy.lower()
            if any(kw in strategy_lower for kw in ['breakout', 'support', 'resistance', 'pivot', 'bounce']):
                base_score += 0.10  # Stratégie pivot-friendly
                
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
            
            # pivot_support_strength → support_strength
            pivot_support_strength = float(self.context.get('support_strength', 0)) if self.context.get('support_strength') is not None else None
            # pivot_resistance_strength → resistance_strength
            pivot_resistance_strength = float(self.context.get('resistance_strength', 0)) if self.context.get('resistance_strength') is not None else None
            # support_touch_count → volume_buildup_periods
            support_touch_count = int(self.context.get('volume_buildup_periods', 0)) if self.context.get('volume_buildup_periods') is not None else None
            # resistance_touch_count → volume_buildup_periods
            resistance_touch_count = int(self.context.get('volume_buildup_periods', 0)) if self.context.get('volume_buildup_periods') is not None else None
            confluence_score = float(self.context.get('confluence_score', 50.0)) if self.context.get('confluence_score') is not None else None
            
            if is_valid:
                relevant_strength = pivot_support_strength if signal_side == "BUY" else pivot_resistance_strength
                relevant_touches = support_touch_count if signal_side == "BUY" else resistance_touch_count
                level_type = "support" if signal_side == "BUY" else "résistance"
                
                reason = f"Pivot {level_type} favorable"
                if relevant_strength:
                    reason += f" (force: {self._safe_format(relevant_strength, '.2f')})"
                if relevant_touches:
                    reason += f", touches: {relevant_touches}"
                if confluence_score:
                    reason += f", confluence: {self._safe_format(confluence_score, '.2f')}"
                    
                return f"{self.name}: Validé - {reason} pour {signal_strategy} {signal_side}"
            else:
                if signal_side == "BUY" and pivot_support_strength and pivot_support_strength < self.min_pivot_strength:
                    return f"{self.name}: Rejeté - Pivot support faible ({self._safe_format(pivot_support_strength, '.2f')})"
                elif signal_side == "SELL" and pivot_resistance_strength and pivot_resistance_strength < self.min_pivot_strength:
                    return f"{self.name}: Rejeté - Pivot résistance faible ({self._safe_format(pivot_resistance_strength, '.2f')})"
                elif support_touch_count and signal_side == "BUY" and support_touch_count < self.min_touch_count:
                    return f"{self.name}: Rejeté - Support touches insuffisantes ({support_touch_count})"
                elif resistance_touch_count and signal_side == "SELL" and resistance_touch_count < self.min_touch_count:
                    return f"{self.name}: Rejeté - Résistance touches insuffisantes ({resistance_touch_count})"
                elif confluence_score and confluence_score < self.min_confluence_score:
                    return f"{self.name}: Rejeté - Confluence pivots insuffisante ({self._safe_format(confluence_score, '.2f')})"
                    
                return f"{self.name}: Rejeté - Critères pivot strength non respectés"
                
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données pivot requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Au minimum, on a besoin d'un indicateur de pivot (support ou résistance)
        pivot_indicators = [
            'nearest_support', 'nearest_resistance', 'support_strength', 
            'resistance_strength', 'confluence_score'
        ]
        
        available_indicators = sum(1 for ind in pivot_indicators 
                                 if ind in self.context and self.context[ind] is not None)
        
        if available_indicators < 2:
            logger.warning(f"{self.name}: Pas assez d'indicateurs pivot pour {self.symbol}")
            return False
            
        return True
        
    def _get_current_price(self) -> Optional[float]:
        """Helper method to get current price from data or context."""
        if self.data:
            # Essayer d'abord la valeur scalaire (format préféré)
            if 'close' in self.data and self.data['close'] is not None:
                try:
                    if isinstance(self.data['close'], (int, float)):
                        return float(self.data['close'])
                    elif isinstance(self.data['close'], list) and len(self.data['close']) > 0:
                        return float(self.data['close'][-1])
                except (IndexError, ValueError, TypeError):
                    pass
            
            # Fallback: essayer current_price dans le contexte
            # current_price n'est pas dans analyzer_data, utiliser self.data['close']
            if current_price is not None:
                try:
                    return float(current_price)
                except (ValueError, TypeError):
                    pass
        return None
