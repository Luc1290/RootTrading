"""
Psychological_Level_Validator - Validator basé sur les niveaux psychologiques.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class Psychological_Level_Validator(BaseValidator):
    """
    Validator pour les niveaux psychologiques - filtre selon la proximité des niveaux ronds.
    
    Vérifie: Proximité niveaux ronds, force psychologique, réaction historique
    Catégorie: structure
    
    Rejette les signaux en:
    - Prix loin des niveaux psychologiques majeurs
    - Niveaux psychologiques faibles ou peu testés
    - Mauvais timing par rapport aux niveaux ronds
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "Psychological_Level_Validator"
        self.category = "structure"
        
        # Paramètres niveaux psychologiques
        self.major_round_levels = [1000, 500, 100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01]  # Niveaux ronds majeurs
        self.minor_round_levels = [25, 75, 2.5, 7.5, 0.25, 0.75, 0.025, 0.075]  # Niveaux ronds mineurs
        self.proximity_threshold_major = 0.005   # 0.5% proximité niveaux majeurs
        self.proximity_threshold_minor = 0.002   # 0.2% proximité niveaux mineurs
        
        # Paramètres force psychologique
        self.min_psychological_strength = 0.4    # Force minimum niveau psychologique
        self.strong_psychological_threshold = 0.7 # Niveau psychologique fort
        self.min_reaction_count = 2              # Minimum réactions historiques
        self.optimal_reaction_count = 5          # Nombre optimal réactions
        
        # Paramètres distance et timing
        self.max_distance_ratio = 0.02           # 2% distance maximum du niveau
        self.optimal_distance_ratio = 0.005      # 0.5% distance optimale
        self.reaction_lookback_bars = 100        # Barres lookback pour réactions
        
        # Paramètres spécifiques crypto/forex
        self.crypto_major_levels = [100000, 50000, 20000, 10000, 5000, 1000, 500, 100, 50, 10, 1, 0.1, 0.01, 0.001]
        self.forex_major_levels = [2.0, 1.5, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        
        # Bonus/malus
        self.major_level_bonus = 0.25            # Bonus niveau majeur
        self.minor_level_bonus = 0.15            # Bonus niveau mineur
        self.strong_reaction_bonus = 0.20        # Bonus réactions fortes
        self.proximity_bonus = 0.18              # Bonus proximité optimale
        self.weak_level_penalty = -0.20          # Pénalité niveau faible
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur la proximité des niveaux psychologiques.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon psychological levels, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs psychologiques depuis le contexte
            try:
                # Niveaux psychologiques principaux
                nearest_psychological_level = float(self.context.get('nearest_psychological_level', 0)) if self.context.get('nearest_psychological_level') is not None else None
                psychological_level_strength = float(self.context.get('psychological_level_strength', 0)) if self.context.get('psychological_level_strength') is not None else None
                psychological_level_type = self.context.get('psychological_level_type')  # 'major', 'minor', 'custom'
                
                # Réactions historiques
                level_reaction_count = int(self.context.get('level_reaction_count', 0)) if self.context.get('level_reaction_count') is not None else None
                level_bounce_strength = float(self.context.get('level_bounce_strength', 0)) if self.context.get('level_bounce_strength') is not None else None
                last_reaction_bars = int(self.context.get('last_reaction_bars', 999)) if self.context.get('last_reaction_bars') is not None else None
                
                # Distance et timing
                distance_to_level = float(self.context.get('distance_to_level', 0)) if self.context.get('distance_to_level') is not None else None
                level_approach_angle = float(self.context.get('level_approach_angle', 0)) if self.context.get('level_approach_angle') is not None else None
                
                # Confluence avec autres niveaux
                multiple_levels_confluence = self.context.get('multiple_levels_confluence', False)
                psychological_confluence_score = float(self.context.get('psychological_confluence_score', 0)) if self.context.get('psychological_confluence_score') is not None else None
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion indicateurs pour {self.symbol}: {e}")
                return False
                
            # Récupération prix actuel depuis data
            current_price = None
            if self.data and 'close' in self.data and self.data['close']:
                try:
                    current_price = float(self.data['close'][-1])
                except (IndexError, ValueError, TypeError):
                    pass
                    
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', 'Unknown')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side or current_price is None:
                logger.warning(f"{self.name}: Signal side ou prix manquant pour {self.symbol}")
                return False
                
            # 1. Détection automatique niveaux psychologiques si pas fournis
            if nearest_psychological_level is None:
                nearest_psychological_level, psychological_level_type = self._find_nearest_psychological_level(current_price)
                
            if nearest_psychological_level is None:
                logger.debug(f"{self.name}: Aucun niveau psychologique détecté pour {self.symbol}")
                return True  # Ne pas bloquer si pas de niveau détecté
                
            # 2. Validation distance du niveau psychologique
            if distance_to_level is None and nearest_psychological_level:
                distance_to_level = abs(current_price - nearest_psychological_level) / current_price
                
            if distance_to_level is not None:
                if distance_to_level > self.max_distance_ratio:
                    logger.debug(f"{self.name}: Trop loin du niveau psychologique ({distance_to_level*100:.2f}%) pour {self.symbol}")
                    if signal_confidence < 0.5:
                        return False
                        
            # 3. Validation force du niveau psychologique
            if psychological_level_strength is not None and psychological_level_strength < self.min_psychological_strength:
                logger.debug(f"{self.name}: Niveau psychologique faible ({psychological_level_strength:.2f}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 4. Validation réactions historiques
            if level_reaction_count is not None and level_reaction_count < self.min_reaction_count:
                logger.debug(f"{self.name}: Pas assez de réactions historiques ({level_reaction_count}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 5. Validation force des rebonds
            if level_bounce_strength is not None and level_bounce_strength < 0.3:
                logger.debug(f"{self.name}: Rebonds historiques faibles ({level_bounce_strength:.2f}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 6. Validation timing depuis dernière réaction
            if last_reaction_bars is not None and last_reaction_bars > self.reaction_lookback_bars:
                logger.debug(f"{self.name}: Dernière réaction trop ancienne ({last_reaction_bars} barres) pour {self.symbol}")
                if signal_confidence < 0.5:
                    return False
                    
            # 7. Validation angle d'approche du niveau
            if level_approach_angle is not None:
                # Angle trop steep peut indiquer un dépassement probable
                if abs(level_approach_angle) > 45:  # Plus de 45 degrés
                    logger.debug(f"{self.name}: Angle d'approche trop steep ({level_approach_angle:.1f}°) pour {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                        
            # 8. Validation type de niveau psychologique
            level_importance = self._assess_level_importance(psychological_level_type, nearest_psychological_level)
            if level_importance < 0.3:
                logger.debug(f"{self.name}: Niveau psychologique peu important ({level_importance:.2f}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 9. Validation confluence psychologique
            if psychological_confluence_score is not None and psychological_confluence_score < 0.4:
                logger.debug(f"{self.name}: Confluence psychologique insuffisante ({psychological_confluence_score:.2f}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 10. Validation cohérence signal/niveau selon direction
            level_coherence = self._validate_signal_level_coherence(
                signal_side, current_price, nearest_psychological_level, psychological_level_type
            )
            if not level_coherence:
                logger.debug(f"{self.name}: Cohérence signal/niveau psychologique insuffisante pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 11. Validation spécifique selon stratégie
            strategy_psych_match = self._validate_strategy_psychological_match(signal_strategy, distance_to_level)
            if not strategy_psych_match:
                logger.debug(f"{self.name}: Stratégie {signal_strategy} inadaptée aux niveaux psychologiques pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                        f"Niveau: {nearest_psychological_level:.4f if nearest_psychological_level is not None else 'N/A'}, "
                        f"Type: {psychological_level_type or 'N/A'}, "
                        f"Distance: {distance_to_level*100:.2f if distance_to_level is not None else 'N/A'}%, "
                        f"Force: {psychological_level_strength:.2f if psychological_level_strength is not None else 'N/A'}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def _find_nearest_psychological_level(self, price: float) -> tuple:
        """Trouve le niveau psychologique le plus proche."""
        try:
            # Déterminer les niveaux appropriés selon le prix
            if price > 1000:
                relevant_levels = [l for l in self.crypto_major_levels if l >= 1]
            elif price > 10:
                relevant_levels = self.major_round_levels + self.minor_round_levels
            elif price > 1:
                relevant_levels = [l for l in self.major_round_levels + self.minor_round_levels if l <= 10]
            else:
                relevant_levels = [l for l in self.major_round_levels + self.minor_round_levels if l <= 1]
                
            if not relevant_levels:
                return None, None
                
            # Trouver le niveau le plus proche
            distances = [(abs(price - level), level) for level in relevant_levels]
            distances.sort()
            
            if not distances:
                return None, None
                
            closest_distance, closest_level = distances[0]
            distance_ratio = closest_distance / price
            
            # Vérifier si assez proche pour être pertinent
            if distance_ratio > self.max_distance_ratio:
                return None, None
                
            # Déterminer le type de niveau
            level_type = "major" if closest_level in self.major_round_levels else "minor"
            if closest_level in self.crypto_major_levels:
                level_type = "major"
                
            return closest_level, level_type
            
        except Exception:
            return None, None
            
    def _assess_level_importance(self, level_type: str, level_value: float) -> float:
        """Évalue l'importance d'un niveau psychologique."""
        try:
            if not level_type or not level_value:
                return 0.5
                
            base_importance = 0.5
            
            # Bonus selon type
            if level_type == "major":
                base_importance += 0.3
            elif level_type == "minor":
                base_importance += 0.2
                
            # Bonus selon valeur ronde
            if level_value in [1000, 500, 100, 50, 10, 5, 1, 0.5, 0.1]:
                base_importance += 0.2  # Niveaux très ronds
            elif str(level_value).endswith('000') or str(level_value).endswith('00'):
                base_importance += 0.15  # Niveaux multiples de 100/1000
                
            return min(1.0, base_importance)
            
        except Exception:
            return 0.5
            
    def _validate_signal_level_coherence(self, signal_side: str, current_price: float, 
                                        level: float, level_type: str) -> bool:
        """Valide la cohérence entre signal et niveau psychologique."""
        try:
            if not level or not signal_side:
                return True
                
            # Pour BUY signals, favoriser près support psychologique
            # Pour SELL signals, favoriser près résistance psychologique
            if signal_side == "BUY":
                # BUY acceptable si prix au-dessus du niveau ou très proche
                if current_price < level:
                    distance_below = (level - current_price) / level
                    if distance_below > 0.01:  # Plus de 1% sous niveau
                        return False
                        
            elif signal_side == "SELL":
                # SELL acceptable si prix en-dessous du niveau ou très proche
                if current_price > level:
                    distance_above = (current_price - level) / level
                    if distance_above > 0.01:  # Plus de 1% au-dessus niveau
                        return False
                        
            return True
            
        except Exception:
            return True
            
    def _validate_strategy_psychological_match(self, strategy: str, distance_to_level: float) -> bool:
        """Valide l'adéquation stratégie/niveaux psychologiques."""
        strategy_lower = strategy.lower()
        
        # Stratégies qui bénéficient des niveaux psychologiques
        psych_friendly = ['breakout', 'reversal', 'bounce', 'support', 'resistance', 'psychological']
        
        # Stratégies moins sensibles aux niveaux psychologiques
        psych_neutral = ['macd', 'rsi', 'moving_average', 'cross', 'trend']
        
        # Si stratégie sensible aux niveaux, critères plus stricts
        if any(kw in strategy_lower for kw in psych_friendly):
            if distance_to_level and distance_to_level > self.proximity_threshold_major:
                return False  # Stratégie psych-friendly mais trop loin
                
        # Stratégies neutres acceptées plus facilement
        if any(kw in strategy_lower for kw in psych_neutral):
            return True
            
        return True  # Par défaut accepter
        
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur les niveaux psychologiques.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur niveaux psychologiques
            psychological_level_strength = float(self.context.get('psychological_level_strength', 0.5)) if self.context.get('psychological_level_strength') is not None else 0.5
            psychological_level_type = self.context.get('psychological_level_type', 'minor')
            level_reaction_count = int(self.context.get('level_reaction_count', 2)) if self.context.get('level_reaction_count') is not None else 2
            level_bounce_strength = float(self.context.get('level_bounce_strength', 0.5)) if self.context.get('level_bounce_strength') is not None else 0.5
            distance_to_level = float(self.context.get('distance_to_level', 0.01)) if self.context.get('distance_to_level') is not None else 0.01
            psychological_confluence_score = float(self.context.get('psychological_confluence_score', 0.5)) if self.context.get('psychological_confluence_score') is not None else 0.5
            last_reaction_bars = int(self.context.get('last_reaction_bars', 50)) if self.context.get('last_reaction_bars') is not None else 50
            
            signal_strategy = signal.get('strategy', '')
            
            base_score = 0.5  # Score de base si validé
            
            # Bonus type de niveau
            if psychological_level_type == "major":
                base_score += self.major_level_bonus
            elif psychological_level_type == "minor":
                base_score += self.minor_level_bonus
                
            # Bonus force du niveau
            if psychological_level_strength >= self.strong_psychological_threshold:
                base_score += 0.15  # Niveau très fort
            elif psychological_level_strength >= self.min_psychological_strength + 0.2:
                base_score += 0.10  # Niveau fort
                
            # Bonus réactions historiques
            if level_reaction_count >= self.optimal_reaction_count:
                base_score += self.strong_reaction_bonus
            elif level_reaction_count >= self.min_reaction_count + 1:
                base_score += 0.10
                
            # Bonus force rebonds
            if level_bounce_strength >= 0.7:
                base_score += 0.12  # Rebonds très forts
            elif level_bounce_strength >= 0.5:
                base_score += 0.08  # Rebonds forts
                
            # Bonus proximité optimale
            if distance_to_level <= self.optimal_distance_ratio:
                base_score += self.proximity_bonus
            elif distance_to_level <= self.proximity_threshold_major:
                base_score += 0.10
                
            # Bonus confluence psychologique
            if psychological_confluence_score >= 0.7:
                base_score += 0.12  # Confluence forte
            elif psychological_confluence_score >= 0.5:
                base_score += 0.08  # Confluence modérée
                
            # Bonus réaction récente
            if last_reaction_bars <= 20:
                base_score += 0.10  # Réaction très récente
            elif last_reaction_bars <= 50:
                base_score += 0.06  # Réaction récente
                
            # Bonus stratégie adaptée
            strategy_lower = signal_strategy.lower()
            if any(kw in strategy_lower for kw in ['breakout', 'reversal', 'bounce', 'psychological']):
                base_score += 0.08  # Stratégie adaptée aux niveaux psychologiques
                
            # Détection automatique si pas de données contextuelles
            try:
                current_price = float(self.data['close'][-1]) if self.data and 'close' in self.data and self.data['close'] else None
                if current_price:
                    auto_level, auto_type = self._find_nearest_psychological_level(current_price)
                    if auto_level and auto_type == "major":
                        base_score += 0.08  # Bonus détection automatique niveau majeur
            except:
                pass
                
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
            
            nearest_psychological_level = float(self.context.get('nearest_psychological_level', 0)) if self.context.get('nearest_psychological_level') is not None else None
            psychological_level_type = self.context.get('psychological_level_type', 'N/A')
            psychological_level_strength = float(self.context.get('psychological_level_strength', 0)) if self.context.get('psychological_level_strength') is not None else None
            distance_to_level = float(self.context.get('distance_to_level', 0)) if self.context.get('distance_to_level') is not None else None
            level_reaction_count = int(self.context.get('level_reaction_count', 0)) if self.context.get('level_reaction_count') is not None else None
            
            if is_valid:
                reason = f"Niveau psychologique favorable"
                if nearest_psychological_level:
                    reason += f" (niveau: {nearest_psychological_level:.4f})"
                if psychological_level_type != 'N/A':
                    reason += f", type: {psychological_level_type}"
                if distance_to_level:
                    reason += f", distance: {distance_to_level*100:.2f}%"
                if psychological_level_strength:
                    reason += f", force: {psychological_level_strength:.2f}"
                if level_reaction_count:
                    reason += f", réactions: {level_reaction_count}"
                    
                return f"{self.name}: Validé - {reason} pour {signal_strategy} {signal_side}"
            else:
                if distance_to_level and distance_to_level > self.max_distance_ratio:
                    return f"{self.name}: Rejeté - Trop loin du niveau psychologique ({distance_to_level*100:.2f}%)"
                elif psychological_level_strength and psychological_level_strength < self.min_psychological_strength:
                    return f"{self.name}: Rejeté - Niveau psychologique faible ({psychological_level_strength:.2f})"
                elif level_reaction_count and level_reaction_count < self.min_reaction_count:
                    return f"{self.name}: Rejeté - Pas assez de réactions historiques ({level_reaction_count})"
                elif not nearest_psychological_level:
                    return f"{self.name}: Rejeté - Aucun niveau psychologique pertinent détecté"
                    
                return f"{self.name}: Rejeté - Critères niveaux psychologiques non respectés"
                
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données psychologiques requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Pour niveaux psychologiques, on peut fonctionner avec détection automatique
        # Vérifier qu'on a au moins les données de prix
        if not self.data or 'close' not in self.data or not self.data['close']:
            logger.warning(f"{self.name}: Données de prix manquantes pour {self.symbol}")
            return False
            
        # Indicateurs optionnels (si pas présents, détection automatique)
        optional_indicators = [
            'nearest_psychological_level', 'psychological_level_strength', 
            'level_reaction_count', 'distance_to_level'
        ]
        
        available_indicators = sum(1 for ind in optional_indicators 
                                 if ind in self.context and self.context[ind] is not None)
        
        # Accepter même sans indicateurs contextuels (détection automatique)
        logger.debug(f"{self.name}: {available_indicators}/{len(optional_indicators)} indicateurs psychologiques disponibles pour {self.symbol}")
        
        return True
