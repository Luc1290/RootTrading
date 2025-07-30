"""
S_R_Level_Proximity_Validator - Validator basé sur la proximité des niveaux Support/Résistance.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class S_R_Level_Proximity_Validator(BaseValidator):
    """
    Validator basé sur la proximité des niveaux S/R - filtre les signaux selon leur position relative aux supports/résistances.
    
    Vérifie: Proximité support/résistance, force des niveaux, probabilité de cassure
    Catégorie: structure
    
    Rejette les signaux en:
    - BUY trop proche d'une résistance forte sans momentum de cassure
    - SELL trop proche d'un support fort sans momentum de cassure
    - Signaux dans une zone de consolidation étroite
    - Niveaux faibles ou peu fiables
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "S_R_Level_Proximity_Validator"
        self.category = "structure"
        
        # Paramètres de proximité (en % du prix)
        self.proximity_threshold = 0.005      # 0.5% de proximité critique
        self.close_proximity_threshold = 0.01 # 1.0% de proximité proche
        self.safe_distance_threshold = 0.02   # 2.0% de distance sûre
        
        # Paramètres de force des niveaux
        self.min_level_strength = 0.3        # Force minimum d'un niveau
        self.strong_level_threshold = 0.7    # Seuil niveau fort
        self.very_strong_level_threshold = 0.9 # Seuil niveau très fort
        
        # Paramètres de cassure
        self.min_break_probability = 0.6     # Probabilité min de cassure
        self.high_break_probability = 0.8    # Probabilité haute de cassure
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur la proximité des niveaux S/R.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon S/R, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs S/R depuis le contexte
            try:
                nearest_support = float(self.context.get('nearest_support', 0)) if self.context.get('nearest_support') is not None else None
                nearest_resistance = float(self.context.get('nearest_resistance', 0)) if self.context.get('nearest_resistance') is not None else None
                support_strength = self._convert_strength_to_score(self.context.get('support_strength')) if self.context.get('support_strength') is not None else None
                resistance_strength = self._convert_strength_to_score(self.context.get('resistance_strength')) if self.context.get('resistance_strength') is not None else None
                break_probability = float(self.context.get('break_probability', 0)) if self.context.get('break_probability') is not None else None
                
                # Prix actuel depuis les données OHLC
                current_price = None
                if 'close' in self.data:
                    current_price = float(self.data['close'])
                elif 'price' in signal:
                    current_price = float(signal['price'])
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion S/R pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_confidence = signal.get('confidence', 0.0)
            signal_strength = signal.get('strength', 'moderate')
            
            if not signal_side or current_price is None:
                logger.warning(f"{self.name}: Signal side ou prix manquant pour {self.symbol}")
                return False
                
            # 1. Validation des signaux BUY
            if signal_side == "BUY":
                # Vérifier proximité avec résistance
                if nearest_resistance is not None:
                    resistance_distance_pct = abs(nearest_resistance - current_price) / current_price
                    
                    # BUY trop proche d'une résistance forte = dangereux
                    if resistance_distance_pct <= self.proximity_threshold:
                        if resistance_strength is not None and resistance_strength >= self.strong_level_threshold:
                            # Sauf si forte probabilité de cassure
                            if break_probability is None or break_probability < self.min_break_probability:
                                logger.debug(f"{self.name}: BUY rejeté - trop proche résistance forte ({self._safe_format(resistance_distance_pct, '.3f')}%) pour {self.symbol}")
                                return False
                            else:
                                logger.debug(f"{self.name}: BUY accepté - proche résistance mais forte probabilité cassure ({self._safe_format(break_probability, '.2f')}) pour {self.symbol}")
                                
                    # Zone de proximité modérée - besoin de conditions favorables
                    elif resistance_distance_pct <= self.close_proximity_threshold:
                        if resistance_strength is not None and resistance_strength >= self.very_strong_level_threshold:
                            # Résistance très forte - besoin de signal très confiant
                            if signal_confidence < 0.8 or signal_strength != 'very_strong':
                                logger.debug(f"{self.name}: BUY rejeté - résistance très forte proche sans signal fort pour {self.symbol}")
                                return False
                                
                # Vérifier proximité avec support (favorable pour BUY)
                if nearest_support is not None:
                    support_distance_pct = abs(current_price - nearest_support) / current_price
                    
                    # BUY près d'un support fort = favorable
                    if support_distance_pct <= self.close_proximity_threshold:
                        if support_strength is not None and support_strength >= self.strong_level_threshold:
                            logger.debug(f"{self.name}: BUY favorisé - proche support fort ({self._safe_format(support_distance_pct, '.3f')}%) pour {self.symbol}")
                            return True
                            
            # 2. Validation des signaux SELL
            elif signal_side == "SELL":
                # Vérifier proximité avec support
                if nearest_support is not None:
                    support_distance_pct = abs(current_price - nearest_support) / current_price
                    
                    # SELL trop proche d'un support fort = dangereux
                    if support_distance_pct <= self.proximity_threshold:
                        if support_strength is not None and support_strength >= self.strong_level_threshold:
                            # Sauf si forte probabilité de cassure
                            if break_probability is None or break_probability < self.min_break_probability:
                                logger.debug(f"{self.name}: SELL rejeté - trop proche support fort ({self._safe_format(support_distance_pct, '.3f')}%) pour {self.symbol}")
                                return False
                            else:
                                logger.debug(f"{self.name}: SELL accepté - proche support mais forte probabilité cassure ({self._safe_format(break_probability, '.2f')}) pour {self.symbol}")
                                
                    # Zone de proximité modérée - besoin de conditions favorables
                    elif support_distance_pct <= self.close_proximity_threshold:
                        if support_strength is not None and support_strength >= self.very_strong_level_threshold:
                            # Support très fort - besoin de signal très confiant
                            if signal_confidence < 0.8 or signal_strength != 'very_strong':
                                logger.debug(f"{self.name}: SELL rejeté - support très fort proche sans signal fort pour {self.symbol}")
                                return False
                                
                # Vérifier proximité avec résistance (favorable pour SELL)
                if nearest_resistance is not None:
                    resistance_distance_pct = abs(nearest_resistance - current_price) / current_price
                    
                    # SELL près d'une résistance forte = favorable
                    if resistance_distance_pct <= self.close_proximity_threshold:
                        if resistance_strength is not None and resistance_strength >= self.strong_level_threshold:
                            logger.debug(f"{self.name}: SELL favorisé - proche résistance forte ({self._safe_format(resistance_distance_pct, '.3f')}%) pour {self.symbol}")
                            return True
                            
            # 3. Vérification zone de consolidation étroite
            if nearest_support is not None and nearest_resistance is not None:
                sr_range_pct = abs(nearest_resistance - nearest_support) / current_price
                
                # Range très étroit - signaux moins fiables
                if sr_range_pct <= 0.01:  # 1% de range
                    if signal_confidence < 0.7:
                        logger.debug(f"{self.name}: Signal rejeté - range S/R trop étroit ({self._safe_format(sr_range_pct, '.3f')}%) + confidence faible pour {self.symbol}")
                        return False
                        
            # 4. Vérification force des niveaux
            if signal_side == "BUY" and support_strength is not None:
                if support_strength < self.min_level_strength:
                    logger.debug(f"{self.name}: BUY rejeté - support trop faible ({self._safe_format(support_strength, '.2f')}) pour {self.symbol}")
                    return False
                    
            elif signal_side == "SELL" and resistance_strength is not None:
                if resistance_strength < self.min_level_strength:
                    logger.debug(f"{self.name}: SELL rejeté - résistance trop faible ({self._safe_format(resistance_strength, '.2f')}) pour {self.symbol}")
                    return False
                    
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                        f"Support: {self._safe_format(nearest_support, '.2f') if nearest_support is not None else 'N/A'} ({self._safe_format(support_strength, '.2f') if support_strength is not None else 'N/A'}), "
                        f"Résistance: {self._safe_format(nearest_resistance, '.2f') if nearest_resistance is not None else 'N/A'} ({self._safe_format(resistance_strength, '.2f') if resistance_strength is not None else 'N/A'}), "
                        f"Prix: {self._safe_format(current_price, '.2f')}, Side: {signal_side}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur la proximité S/R.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur S/R
            nearest_support = float(self.context.get('nearest_support', 0)) if self.context.get('nearest_support') is not None else None
            nearest_resistance = float(self.context.get('nearest_resistance', 0)) if self.context.get('nearest_resistance') is not None else None
            support_strength = self._convert_strength_to_score(self.context.get('support_strength')) if self.context.get('support_strength') is not None else 0
            resistance_strength = self._convert_strength_to_score(self.context.get('resistance_strength')) if self.context.get('resistance_strength') is not None else 0
            break_probability = float(self.context.get('break_probability', 0)) if self.context.get('break_probability') is not None else None
            
            current_price = float(self.data.get('close', signal.get('price', 0)))
            signal_side = signal.get('side')
            
            base_score = 0.5  # Score de base si validé
            
            # Scoring selon position par rapport aux niveaux
            if signal_side == "BUY" and nearest_support is not None:
                support_distance_pct = abs(current_price - nearest_support) / current_price
                
                # Bonus proximité support fort
                if support_distance_pct <= self.close_proximity_threshold:
                    if support_strength >= self.very_strong_level_threshold:
                        base_score += 0.3  # Excellent
                    elif support_strength >= self.strong_level_threshold:
                        base_score += 0.2  # Très bon
                    else:
                        base_score += 0.1  # Bon
                        
                # Bonus éloignement de résistance
                if nearest_resistance is not None:
                    resistance_distance_pct = abs(nearest_resistance - current_price) / current_price
                    if resistance_distance_pct >= self.safe_distance_threshold:
                        base_score += 0.1
                        
            elif signal_side == "SELL" and nearest_resistance is not None:
                resistance_distance_pct = abs(nearest_resistance - current_price) / current_price
                
                # Bonus proximité résistance forte
                if resistance_distance_pct <= self.close_proximity_threshold:
                    if resistance_strength >= self.very_strong_level_threshold:
                        base_score += 0.3  # Excellent
                    elif resistance_strength >= self.strong_level_threshold:
                        base_score += 0.2  # Très bon
                    else:
                        base_score += 0.1  # Bon
                        
                # Bonus éloignement de support
                if nearest_support is not None:
                    support_distance_pct = abs(current_price - nearest_support) / current_price
                    if support_distance_pct >= self.safe_distance_threshold:
                        base_score += 0.1
                        
            # Bonus probabilité de cassure élevée
            if break_probability is not None:
                if break_probability >= self.high_break_probability:
                    base_score += 0.2  # Forte probabilité cassure
                elif break_probability >= self.min_break_probability:
                    base_score += 0.1  # Probabilité modérée
                    
            # Bonus range large (évite consolidation)
            if nearest_support is not None and nearest_resistance is not None:
                sr_range_pct = abs(nearest_resistance - nearest_support) / current_price
                if sr_range_pct >= 0.03:  # 3% de range ou plus
                    base_score += 0.05
                    
            return min(1.0, max(0.0, base_score))
            
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
            current_price = float(self.data.get('close', signal.get('price', 0)))
            nearest_support = float(self.context.get('nearest_support', 0)) if self.context.get('nearest_support') is not None else None
            nearest_resistance = float(self.context.get('nearest_resistance', 0)) if self.context.get('nearest_resistance') is not None else None
            support_strength = float(self.context.get('support_strength', 0)) if self.context.get('support_strength') is not None else None
            resistance_strength = float(self.context.get('resistance_strength', 0)) if self.context.get('resistance_strength') is not None else None
            
            if is_valid:
                if signal_side == "BUY" and nearest_support is not None:
                    support_distance_pct = abs(current_price - nearest_support) / current_price
                    if support_distance_pct <= self.close_proximity_threshold and support_strength and support_strength >= self.strong_level_threshold:
                        return f"{self.name}: Validé - BUY proche support fort (distance: {support_distance_pct:.2%}, force: {self._safe_format(support_strength, '.2f')})"
                        
                elif signal_side == "SELL" and nearest_resistance is not None:
                    resistance_distance_pct = abs(nearest_resistance - current_price) / current_price
                    if resistance_distance_pct <= self.close_proximity_threshold and resistance_strength and resistance_strength >= self.strong_level_threshold:
                        return f"{self.name}: Validé - SELL proche résistance forte (distance: {resistance_distance_pct:.2%}, force: {self._safe_format(resistance_strength, '.2f')})"
                        
                return f"{self.name}: Validé - Position S/R favorable pour signal {signal_side}"
            else:
                if signal_side == "BUY" and nearest_resistance is not None:
                    resistance_distance_pct = abs(nearest_resistance - current_price) / current_price
                    if resistance_distance_pct <= self.proximity_threshold:
                        return f"{self.name}: Rejeté - BUY trop proche résistance (distance: {resistance_distance_pct:.2%})"
                        
                elif signal_side == "SELL" and nearest_support is not None:
                    support_distance_pct = abs(current_price - nearest_support) / current_price
                    if support_distance_pct <= self.proximity_threshold:
                        return f"{self.name}: Rejeté - SELL trop proche support (distance: {support_distance_pct:.2%})"
                        
                return f"{self.name}: Rejeté - Position S/R défavorable"
                
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données S/R requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Vérifier qu'au moins un niveau S/R est présent
        has_support = 'nearest_support' in self.context and self.context['nearest_support'] is not None
        has_resistance = 'nearest_resistance' in self.context and self.context['nearest_resistance'] is not None
        
        if not has_support and not has_resistance:
            logger.warning(f"{self.name}: Aucun niveau S/R disponible pour {self.symbol}")
            return False
            
        # Vérifier prix disponible
        has_price = 'close' in self.data or ('price' in self.context and self.context['price'] is not None)
        if not has_price:
            logger.warning(f"{self.name}: Prix manquant pour {self.symbol}")
            return False
            
        return True
