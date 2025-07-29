"""
ADX_TrendStrength_Validator - Validator basé sur la force de tendance ADX.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class ADX_TrendStrength_Validator(BaseValidator):
    """
    Validator pour la force de tendance ADX - filtre les signaux selon la force de la tendance.
    
    Vérifie: Force tendance ADX, cohérence DI, alignement directionnel
    Catégorie: trend
    
    Rejette les signaux en:
    - Tendance faible (ADX < 20)
    - DI contradictoires avec le signal
    - Tendance déclinante (ADX en baisse)
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "ADX_TrendStrength_Validator"
        self.category = "trend"
        
        # Paramètres ADX
        self.min_adx_strength = 20.0      # ADX minimum pour tendance
        self.strong_adx_threshold = 30.0  # ADX fort
        self.extreme_adx_threshold = 50.0 # ADX extrême
        self.di_separation_min = 5.0      # Séparation min +DI/-DI
        self.trend_consistency_bonus = 0.2 # Bonus cohérence
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur la force de tendance ADX.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon ADX, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs ADX depuis le contexte
            try:
                adx_14 = float(self.context.get('adx_14', 0)) if self.context.get('adx_14') is not None else None
                plus_di = float(self.context.get('plus_di', 0)) if self.context.get('plus_di') is not None else None
                minus_di = float(self.context.get('minus_di', 0)) if self.context.get('minus_di') is not None else None
                trend_strength = self._convert_trend_strength_to_score(self.context.get('trend_strength')) if self.context.get('trend_strength') is not None else None
                directional_bias = self.context.get('directional_bias')
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion ADX pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side or adx_14 is None:
                logger.warning(f"{self.name}: Signal side ou ADX manquant pour {self.symbol}")
                return False
                
            # 1. Vérification force tendance ADX
            if adx_14 < self.min_adx_strength:
                logger.debug(f"{self.name}: ADX trop faible ({adx_14:.1f}) pour {self.symbol} - tendance insuffisante")
                return False
                
            # 2. Vérification cohérence directionnelle avec DI
            if plus_di is not None and minus_di is not None:
                di_separation = abs(plus_di - minus_di)
                
                if signal_side == "BUY":
                    # Pour BUY: +DI doit être > -DI
                    if plus_di <= minus_di:
                        logger.debug(f"{self.name}: BUY signal mais +DI ({plus_di:.1f}) <= -DI ({minus_di:.1f}) pour {self.symbol}")
                        return False
                        
                elif signal_side == "SELL":
                    # Pour SELL: -DI doit être > +DI
                    if minus_di <= plus_di:
                        logger.debug(f"{self.name}: SELL signal mais -DI ({minus_di:.1f}) <= +DI ({plus_di:.1f}) pour {self.symbol}")
                        return False
                        
                # Vérification séparation suffisante des DI
                if di_separation < self.di_separation_min:
                    logger.debug(f"{self.name}: Séparation DI insuffisante ({di_separation:.1f}) pour {self.symbol}")
                    return False
                    
            # 3. Vérification cohérence avec directional_bias
            if directional_bias:
                if signal_side == "BUY" and directional_bias == "bearish":
                    logger.debug(f"{self.name}: BUY signal mais bias bearish pour {self.symbol}")
                    return False
                elif signal_side == "SELL" and directional_bias == "bullish":
                    logger.debug(f"{self.name}: SELL signal mais bias bullish pour {self.symbol}")
                    return False
                    
            # 4. Bonus pour tendance très forte
            if adx_14 >= self.strong_adx_threshold:
                # Tendance forte - signal validé
                logger.debug(f"{self.name}: Tendance forte ADX ({adx_14:.1f}) - signal validé pour {self.symbol}")
                
                if adx_14 >= self.extreme_adx_threshold:
                    logger.debug(f"{self.name}: Tendance extrême ADX ({adx_14:.1f}) pour {self.symbol}")
                    
            # 5. Vérification trend_strength pour cohérence
            if trend_strength is not None:
                if trend_strength < 0.3:  # Tendance très faible selon autre mesure
                    logger.debug(f"{self.name}: Trend strength faible ({trend_strength:.2f}) malgré ADX pour {self.symbol}")
                    # Ne pas rejeter mais noter l'incohérence
                    
            # 6. Validation finale selon confidence du signal
            if signal_confidence < 0.3 and adx_14 < self.strong_adx_threshold:
                # Signal faible + tendance modérée = rejet
                logger.debug(f"{self.name}: Signal confidence faible ({signal_confidence:.2f}) + ADX modéré ({adx_14:.1f}) pour {self.symbol}")
                return False
                
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - ADX: {adx_14:.1f}, "
                        f"+DI: {plus_di:.1f if plus_di else 'N/A'}, "
                        f"-DI: {minus_di:.1f if minus_di else 'N/A'}, "
                        f"Side: {signal_side}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur la force ADX.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur ADX
            adx_14 = float(self.context.get('adx_14', 0)) if self.context.get('adx_14') is not None else 0
            plus_di = float(self.context.get('plus_di', 0)) if self.context.get('plus_di') is not None else None
            minus_di = float(self.context.get('minus_di', 0)) if self.context.get('minus_di') is not None else None
            
            base_score = 0.5  # Score de base si validé
            
            # Bonus selon force ADX
            if adx_14 >= self.extreme_adx_threshold:
                base_score += 0.4  # Tendance extrême
            elif adx_14 >= self.strong_adx_threshold:
                base_score += 0.3  # Tendance forte
            else:
                base_score += 0.1  # Tendance modérée
                
            # Bonus séparation DI
            if plus_di is not None and minus_di is not None:
                di_separation = abs(plus_di - minus_di)
                if di_separation >= 15.0:
                    base_score += 0.1  # Excellente séparation
                elif di_separation >= 10.0:
                    base_score += 0.05  # Bonne séparation
                    
            # Bonus cohérence directional_bias
            directional_bias = self.context.get('directional_bias')
            signal_side = signal.get('side')
            if directional_bias and signal_side:
                if (signal_side == "BUY" and directional_bias == "bullish") or \
                   (signal_side == "SELL" and directional_bias == "bearish"):
                    base_score += self.trend_consistency_bonus
                    
            return min(1.0, base_score)
            
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
            adx_14 = float(self.context.get('adx_14', 0)) if self.context.get('adx_14') is not None else 0
            plus_di = float(self.context.get('plus_di', 0)) if self.context.get('plus_di') is not None else None
            minus_di = float(self.context.get('minus_di', 0)) if self.context.get('minus_di') is not None else None
            signal_side = signal.get('side', 'N/A')
            
            if is_valid:
                strength_desc = "extrême" if adx_14 >= self.extreme_adx_threshold else \
                               "forte" if adx_14 >= self.strong_adx_threshold else "modérée"
                               
                reason = f"Tendance {strength_desc} (ADX: {adx_14:.1f})"
                
                if plus_di is not None and minus_di is not None:
                    di_sep = abs(plus_di - minus_di)
                    reason += f", DI séparation: {di_sep:.1f}"
                    
                return f"{self.name}: Validé - {reason} pour signal {signal_side}"
            else:
                if adx_14 < self.min_adx_strength:
                    return f"{self.name}: Rejeté - Tendance trop faible (ADX: {adx_14:.1f})"
                elif plus_di is not None and minus_di is not None:
                    if signal_side == "BUY" and plus_di <= minus_di:
                        return f"{self.name}: Rejeté - BUY mais +DI ({plus_di:.1f}) <= -DI ({minus_di:.1f})"
                    elif signal_side == "SELL" and minus_di <= plus_di:
                        return f"{self.name}: Rejeté - SELL mais -DI ({minus_di:.1f}) <= +DI ({plus_di:.1f})"
                        
                return f"{self.name}: Rejeté - Critères ADX non respectés"
                
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données ADX requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Vérification présence ADX
        if 'adx_14' not in self.context or self.context['adx_14'] is None:
            logger.warning(f"{self.name}: ADX_14 manquant pour {self.symbol}")
            return False
            
        return True
