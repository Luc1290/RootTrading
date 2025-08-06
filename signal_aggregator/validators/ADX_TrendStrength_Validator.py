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
        
        # Paramètres ADX - OPTIMISÉS
        self.min_adx_strength = 25.0      # ADX minimum augmenté (25 au lieu de 20)
        self.strong_adx_threshold = 35.0  # ADX fort augmenté (35 au lieu de 30)
        self.extreme_adx_threshold = 55.0 # ADX extrême augmenté (55 au lieu de 50)
        self.di_separation_min = 8.0      # Séparation min DI augmentée (8 au lieu de 5)
        self.strong_di_separation = 15.0  # Séparation forte DI
        self.trend_consistency_bonus = 0.25 # Bonus cohérence augmenté
        self.weak_signal_adx_threshold = 40.0  # ADX requis pour signaux faibles
        
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
                trend_strength = self._convert_trend_strength_to_score(str(self.context.get('trend_strength'))) if self.context.get('trend_strength') is not None else None
                directional_bias = self.context.get('directional_bias')
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion ADX pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side or adx_14 is None:
                logger.warning(f"{self.name}: Signal side ou ADX manquant pour {self.symbol}")
                return False
                
            # 1. Vérification force tendance ADX - PLUS STRICT
            if adx_14 < self.min_adx_strength:
                logger.debug(f"{self.name}: ADX trop faible ({self._safe_format(adx_14, '.1f')} < {self.min_adx_strength}) pour {self.symbol} - tendance insuffisante")
                return False
                
            # 2. Vérification cohérence directionnelle avec DI
            if plus_di is not None and minus_di is not None:
                di_separation = abs(plus_di - minus_di)
                
                if signal_side == "BUY":
                    # Pour BUY: +DI doit être > -DI
                    if plus_di <= minus_di:
                        logger.debug(f"{self.name}: BUY signal mais +DI ({self._safe_format(plus_di, '.1f')}) <= -DI ({self._safe_format(minus_di, '.1f')}) pour {self.symbol}")
                        return False
                        
                elif signal_side == "SELL":
                    # Pour SELL: -DI doit être > +DI
                    if minus_di <= plus_di:
                        logger.debug(f"{self.name}: SELL signal mais -DI ({self._safe_format(minus_di, '.1f')}) <= +DI ({self._safe_format(plus_di, '.1f')}) pour {self.symbol}")
                        return False
                        
                # Vérification séparation suffisante des DI - PLUS STRICT
                if di_separation < self.di_separation_min:
                    logger.debug(f"{self.name}: Séparation DI insuffisante ({self._safe_format(di_separation, '.1f')} < {self.di_separation_min}) pour {self.symbol}")
                    return False
                    
                # NOUVEAU: Vérification séparation forte pour signaux faibles
                if signal_confidence < 0.5 and di_separation < self.strong_di_separation:
                    logger.debug(f"{self.name}: Signal faible ({self._safe_format(signal_confidence, '.2f')}) nécessite DI séparation forte ({self._safe_format(di_separation, '.1f')} < {self.strong_di_separation}) pour {self.symbol}")
                    return False
                    
            # 3. Vérification cohérence avec directional_bias
            if directional_bias:
                if signal_side == "BUY" and directional_bias.upper() == "BEARISH":
                    logger.debug(f"{self.name}: BUY signal mais bias bearish pour {self.symbol}")
                    return False
                elif signal_side == "SELL" and directional_bias.upper() == "BULLISH":
                    logger.debug(f"{self.name}: SELL signal mais bias bullish pour {self.symbol}")
                    return False
                    
            # 4. Bonus pour tendance très forte
            if adx_14 >= self.strong_adx_threshold:
                # Tendance forte - signal validé
                logger.debug(f"{self.name}: Tendance forte ADX ({self._safe_format(adx_14, '.1f')}) - signal validé pour {self.symbol}")
                
                if adx_14 >= self.extreme_adx_threshold:
                    logger.debug(f"{self.name}: Tendance extrême ADX ({self._safe_format(adx_14, '.1f')}) pour {self.symbol}")
                    
            # 5. Vérification trend_strength pour cohérence - PLUS STRICT
            if trend_strength is not None:
                if trend_strength < 0.2:  # Seuil abaissé mais avec rejet
                    logger.debug(f"{self.name}: Trend strength très faible ({self._safe_format(trend_strength, '.2f')}) incohérent avec ADX pour {self.symbol}")
                    return False  # NOUVEAU: Rejeter les incohérences fortes
                elif trend_strength < 0.35 and signal_confidence < 0.5:
                    logger.debug(f"{self.name}: Trend strength faible ({self._safe_format(trend_strength, '.2f')}) + signal faible pour {self.symbol}")
                    return False  # NOUVEAU: Rejeter si double faiblesse
                    
            # 6. Validation finale selon confidence du signal - CRITÈRES DURCIS
            if signal_confidence < 0.4:  # Seuil augmenté (40% au lieu de 30%)
                if adx_14 < self.weak_signal_adx_threshold:  # ADX fort requis pour signaux faibles
                    logger.debug(f"{self.name}: Signal confidence faible ({self._safe_format(signal_confidence, '.2f')}) nécessite ADX fort (ADX: {self._safe_format(adx_14, '.1f')} < {self.weak_signal_adx_threshold}) pour {self.symbol}")
                    return False
                    
            # NOUVEAU: Rejet signaux très faibles même avec ADX correct
            if signal_confidence < 0.25:
                logger.debug(f"{self.name}: Signal confidence trop faible ({self._safe_format(signal_confidence, '.2f')} < 0.25) pour {self.symbol}")
                return False
                
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - ADX: {self._safe_format(adx_14, '.1f')}, "
                        f"+DI: {self._safe_format(plus_di, '.1f') if plus_di is not None else 'N/A'}, "
                        f"-DI: {self._safe_format(minus_di, '.1f') if minus_di is not None else 'N/A'}, "
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
            
            base_score = 0.4  # Score de base réduit (40% au lieu de 50%)
            
            # Bonus selon force ADX - VALORISATION AUGMENTÉE
            if adx_14 >= self.extreme_adx_threshold:
                base_score += 0.5  # AUGMENTÉ - Tendance extrême
            elif adx_14 >= self.strong_adx_threshold:
                base_score += 0.35  # AUGMENTÉ - Tendance forte
            else:
                base_score += 0.05  # RÉDUIT - Tendance modérée moins valorisée
                
            # Bonus séparation DI - VALORISATION AUGMENTÉE
            if plus_di is not None and minus_di is not None:
                di_separation = abs(plus_di - minus_di)
                if di_separation >= 20.0:  # Seuil augmenté
                    base_score += 0.15  # AUGMENTÉ - Excellente séparation
                elif di_separation >= self.strong_di_separation:  # 15.0
                    base_score += 0.10  # AUGMENTÉ - Très bonne séparation
                elif di_separation >= 10.0:
                    base_score += 0.05  # Bonne séparation
                # NOUVEAU: Pénalité si séparation très faible
                elif di_separation < self.di_separation_min:
                    base_score -= 0.1  # Pénalité séparation insuffisante
                    
            # Bonus cohérence directional_bias - PLUS VALORISÉ
            directional_bias = self.context.get('directional_bias')
            signal_side = signal.get('side')
            if directional_bias and signal_side:
                if (signal_side == "BUY" and directional_bias.upper() == "BULLISH") or \
                   (signal_side == "SELL" and directional_bias.upper() == "BEARISH"):
                    base_score += self.trend_consistency_bonus  # 0.25
                # NOUVEAU: Pénalité si bias contradictoire (ne devrait pas arriver car déjà rejeté)
                elif (signal_side == "BUY" and directional_bias.upper() == "BEARISH") or \
                     (signal_side == "SELL" and directional_bias.upper() == "BULLISH"):
                    base_score -= 0.2  # Pénalité contradiction
                    
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
                               
                reason = f"Tendance {strength_desc} (ADX: {self._safe_format(adx_14, '.1f')})"
                
                if plus_di is not None and minus_di is not None:
                    di_sep = abs(plus_di - minus_di)
                    reason += f", DI séparation: {self._safe_format(di_sep, '.1f')}"
                    
                return f"{self.name}: Validé - {reason} pour signal {signal_side}"
            else:
                if adx_14 < self.min_adx_strength:
                    return f"{self.name}: Rejeté - Tendance trop faible (ADX: {self._safe_format(adx_14, '.1f')} < {self.min_adx_strength})"
                elif plus_di is not None and minus_di is not None:
                    if signal_side == "BUY" and plus_di <= minus_di:
                        return f"{self.name}: Rejeté - BUY mais +DI ({self._safe_format(plus_di, '.1f')}) <= -DI ({self._safe_format(minus_di, '.1f')})"
                    elif signal_side == "SELL" and minus_di <= plus_di:
                        return f"{self.name}: Rejeté - SELL mais -DI ({self._safe_format(minus_di, '.1f')}) <= +DI ({self._safe_format(plus_di, '.1f')})"
                        
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
