"""
Trend_Smoothness_Validator - Validator basé sur la fluidité et régularité des tendances.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class Trend_Smoothness_Validator(BaseValidator):
    """
    Validator basé sur la fluidité des tendances - filtre les signaux selon la régularité et qualité du trend.
    
    Vérifie: Fluidité trend, alignement moyennes mobiles, angle trend, volatilité relative
    Catégorie: trend
    
    Rejette les signaux en:
    - Tendances chaotiques ou très volatiles
    - Moyennes mobiles désalignées ou croisées
    - Angle de tendance trop faible ou instable
    - Trend de courte durée sans confirmation
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "Trend_Smoothness_Validator"
        self.category = "trend"
        
        # Paramètres de fluidité
        self.min_trend_strength = 0.4         # Force minimum du trend
        self.smooth_trend_threshold = 0.7     # Seuil trend fluide
        self.very_smooth_threshold = 0.85     # Seuil trend très fluide
        
        # Paramètres d'angle de tendance
        self.min_trend_angle = 15.0           # Angle minimum (degrés)
        self.strong_trend_angle = 30.0        # Angle fort
        self.steep_trend_angle = 45.0         # Angle raide
        
        # Paramètres d'alignement
        self.min_trend_alignment = 0.60      # Alignement minimum des MAs (format 0-1)
        self.strong_alignment_threshold = 0.8 # Alignement fort
        
        # Paramètres de volatilité
        self.max_volatility_ratio = 2.0       # Ratio volatilité max acceptable
        self.low_volatility_threshold = 0.8   # Seuil volatilité faible
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur la fluidité des tendances.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon la fluidité, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs de tendance depuis le contexte
            try:
                # Gestion robuste de trend_strength (peut être "absent", None, ou numérique)
                raw_trend_strength = self.context.get('trend_strength')
                if raw_trend_strength in [None, 'absent', 'unknown', '']:
                    trend_strength = None
                else:
                    trend_strength = self._convert_trend_strength_to_score(str(raw_trend_strength))
                
                # Gestion des valeurs NULL dans trend_angle
                trend_angle = float(self.context.get('trend_angle', 0)) if self.context.get('trend_angle') is not None else None
                trend_alignment = float(self.context.get('trend_alignment', 0)) if self.context.get('trend_alignment') is not None else None
                directional_bias = self.context.get('directional_bias')
                
                # Moyennes mobiles pour vérifier alignement
                ema_7 = float(self.context.get('ema_7', 0)) if self.context.get('ema_7') is not None else None
                ema_12 = float(self.context.get('ema_12', 0)) if self.context.get('ema_12') is not None else None
                ema_26 = float(self.context.get('ema_26', 0)) if self.context.get('ema_26') is not None else None
                ema_50 = float(self.context.get('ema_50', 0)) if self.context.get('ema_50') is not None else None
                
                # Volatilité et régime
                volatility_regime = self.context.get('volatility_regime')
                atr_percentile = float(self.context.get('atr_percentile', 0)) if self.context.get('atr_percentile') is not None else None
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion indicateurs pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side:
                logger.warning(f"{self.name}: Signal side manquant pour {self.symbol}")
                return False
                
            # 1. Vérification force de tendance (seulement si disponible)
            if trend_strength is not None:
                if trend_strength < self.min_trend_strength:
                    logger.debug(f"{self.name}: Tendance trop faible ({self._safe_format(trend_strength, '.2f')}) pour {self.symbol}")
                    return False
            else:
                # Si trend_strength non disponible, utiliser d'autres indicateurs
                logger.debug(f"{self.name}: trend_strength non disponible, vérification avec autres indicateurs pour {self.symbol}")
                    
            # 2. Vérification angle de tendance (seulement si disponible)
            if trend_angle is not None:
                abs_angle = abs(trend_angle)
                if abs_angle < self.min_trend_angle:
                    logger.debug(f"{self.name}: Angle de tendance trop faible ({self._safe_format(abs_angle, '.1f')}°) pour {self.symbol}")
                    return False
                    
                # Vérification cohérence angle/signal
                if signal_side == "BUY" and trend_angle < -self.min_trend_angle:
                    logger.debug(f"{self.name}: BUY signal mais angle bearish ({self._safe_format(trend_angle, '.1f')}°) pour {self.symbol}")
                    return False
                elif signal_side == "SELL" and trend_angle > self.min_trend_angle:
                    logger.debug(f"{self.name}: SELL signal mais angle bullish ({self._safe_format(trend_angle, '.1f')}°) pour {self.symbol}")
                    return False
                    
            # 3. Vérification alignement des moyennes mobiles
            if ema_7 is not None and ema_12 is not None and ema_26 is not None and ema_50 is not None:
                # Pour BUY: EMAs doivent être ordonnées (7 > 12 > 26 > 50)
                if signal_side == "BUY":
                    if not (ema_7 > ema_12 > ema_26):
                        # Permettre un léger désalignement si trend_alignment est bon
                        if trend_alignment is None or trend_alignment < self.min_trend_alignment:
                            logger.debug(f"{self.name}: BUY signal mais EMAs mal alignées pour {self.symbol}")
                            return False
                            
                # Pour SELL: EMAs doivent être ordonnées (7 < 12 < 26 < 50)
                elif signal_side == "SELL":
                    if not (ema_7 < ema_12 < ema_26):
                        # Permettre un léger désalignement si trend_alignment est bon
                        if trend_alignment is None or trend_alignment < self.min_trend_alignment:
                            logger.debug(f"{self.name}: SELL signal mais EMAs mal alignées pour {self.symbol}")
                            return False
                            
            # 4. Vérification alignement général des tendances
            if trend_alignment is not None:
                # Si les données principales manquent, être plus permissif sur l'alignement
                alignment_threshold = self.min_trend_alignment
                if trend_strength is None and trend_angle is None:
                    alignment_threshold = 0.3  # Seuil réduit si données principales manquantes
                    logger.debug(f"{self.name}: Seuil alignement réduit (données principales manquantes) pour {self.symbol}")
                
                if trend_alignment < alignment_threshold:
                    logger.debug(f"{self.name}: Alignement des tendances insuffisant ({self._safe_format(trend_alignment, '.2f')} < {alignment_threshold}) pour {self.symbol}")
                    return False
                    
            # 5. Vérification cohérence directional_bias
            if directional_bias:
                if signal_side == "BUY" and directional_bias.upper() == "BEARISH":
                    logger.debug(f"{self.name}: BUY signal mais bias bearish pour {self.symbol}")
                    return False
                elif signal_side == "SELL" and directional_bias.upper() == "BULLISH":
                    logger.debug(f"{self.name}: SELL signal mais bias bullish pour {self.symbol}")
                    return False
                    
            # 6. Vérification régime de volatilité
            if volatility_regime == "high" and signal_confidence < 0.8:
                logger.debug(f"{self.name}: Volatilité élevée nécessite confidence élevée pour {self.symbol}")
                return False
                
            # 7. Vérification percentile ATR (éviter volatilité extrême)
            if atr_percentile is not None:
                if atr_percentile > 90.0:  # ATR dans les 10% les plus élevés
                    if signal_confidence < 0.75:
                        logger.debug(f"{self.name}: ATR extrême ({self._safe_format(atr_percentile, '.1f')}%) nécessite confidence élevée pour {self.symbol}")
                        return False
                        
            # 8. Bonus pour tendances très fluides
            smooth_bonus = False
            if trend_strength is not None and trend_strength >= self.very_smooth_threshold:
                logger.debug(f"{self.name}: Tendance très fluide détectée ({self._safe_format(trend_strength, '.2f')}) pour {self.symbol}")
                smooth_bonus = True
                
            if trend_alignment is not None and trend_alignment >= self.strong_alignment_threshold:
                logger.debug(f"{self.name}: Alignement fort détecté ({self._safe_format(trend_alignment, '.2f')}) pour {self.symbol}")
                smooth_bonus = True
                
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                        f"Trend strength: {self._safe_format(trend_strength, '.2f') if trend_strength is not None else 'N/A'}, "
                        f"Angle: {self._safe_format(trend_angle, '.1f') if trend_angle is not None else 'N/A'}°, "
                        f"Alignment: {self._safe_format(trend_alignment, '.2f') if trend_alignment is not None else 'N/A'}, "
                        f"Side: {signal_side}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur la fluidité des tendances.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur la fluidité
            trend_strength_value = self.context.get('trend_strength')
            trend_strength = self._convert_trend_strength_to_score(str(trend_strength_value)) if trend_strength_value is not None else 0.5
            trend_angle = float(self.context.get('trend_angle', 0)) if self.context.get('trend_angle') is not None else None
            trend_alignment = float(self.context.get('trend_alignment', 0)) if self.context.get('trend_alignment') is not None else 0.5
            atr_percentile = float(self.context.get('atr_percentile', 0)) if self.context.get('atr_percentile') is not None else 50
            
            signal_side = signal.get('side')
            base_score = 0.5  # Score de base si validé
            
            # Bonus selon force de tendance
            if trend_strength >= self.very_smooth_threshold:
                base_score += 0.3  # Tendance très fluide
            elif trend_strength >= self.smooth_trend_threshold:
                base_score += 0.2  # Tendance fluide
            elif trend_strength >= self.min_trend_strength:
                base_score += 0.1  # Tendance acceptable
                
            # Bonus selon alignement
            if trend_alignment >= self.strong_alignment_threshold:
                base_score += 0.2  # Alignement fort
            elif trend_alignment >= self.min_trend_alignment:
                base_score += 0.1  # Alignement acceptable
                
            # Bonus selon angle de tendance
            if trend_angle is not None:
                abs_angle = abs(trend_angle)
                if abs_angle >= self.steep_trend_angle:
                    base_score += 0.15  # Tendance raide
                elif abs_angle >= self.strong_trend_angle:
                    base_score += 0.1   # Tendance forte
                elif abs_angle >= self.min_trend_angle:
                    base_score += 0.05  # Tendance acceptable
                    
            # Malus volatilité élevée
            if atr_percentile > 80.0:
                base_score -= 0.1  # Volatilité élevée
            elif atr_percentile < 20.0:
                base_score += 0.05  # Volatilité faible (bonus)
                
            # Vérification EMAs pour bonus supplémentaire
            ema_7 = float(self.context.get('ema_7', 0)) if self.context.get('ema_7') is not None else None
            ema_12 = float(self.context.get('ema_12', 0)) if self.context.get('ema_12') is not None else None
            ema_26 = float(self.context.get('ema_26', 0)) if self.context.get('ema_26') is not None else None
            
            if ema_7 is not None and ema_12 is not None and ema_26 is not None:
                # Bonus alignement parfait des EMAs
                if signal_side == "BUY" and ema_7 > ema_12 > ema_26:
                    base_score += 0.1
                elif signal_side == "SELL" and ema_7 < ema_12 < ema_26:
                    base_score += 0.1
                    
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
            trend_strength_value = self.context.get('trend_strength')
            trend_strength = self._convert_trend_strength_to_score(str(trend_strength_value)) if trend_strength_value is not None else None
            trend_angle = float(self.context.get('trend_angle', 0)) if self.context.get('trend_angle') is not None else None
            trend_alignment = float(self.context.get('trend_alignment', 0)) if self.context.get('trend_alignment') is not None else None
            
            if is_valid:
                # Déterminer qualité de la tendance
                if trend_strength is not None and trend_strength >= self.very_smooth_threshold:
                    quality = "très fluide"
                elif trend_strength is not None and trend_strength >= self.smooth_trend_threshold:
                    quality = "fluide"
                else:
                    quality = "acceptable"
                    
                reason = f"Tendance {quality}"
                
                if trend_angle is not None:
                    reason += f" (angle: {self._safe_format(trend_angle, '.1f')}°)"
                    
                if trend_alignment is not None:
                    reason += f" (alignement: {self._safe_format(trend_alignment, '.2f')})"
                    
                return f"{self.name}: Validé - {reason} pour signal {signal_side}"
            else:
                if trend_strength is not None and trend_strength < self.min_trend_strength:
                    return f"{self.name}: Rejeté - Tendance trop faible ({self._safe_format(trend_strength, '.2f')})"
                elif trend_angle is not None and abs(trend_angle) < self.min_trend_angle:
                    return f"{self.name}: Rejeté - Angle tendance insuffisant ({self._safe_format(trend_angle, '.1f')}°)"
                elif trend_alignment is not None and trend_alignment < self.min_trend_alignment:
                    return f"{self.name}: Rejeté - Alignement insuffisant ({self._safe_format(trend_alignment, '.2f')})"
                else:
                    return f"{self.name}: Rejeté - Tendance non fluide ou incohérente"
                    
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
            
        # Vérifier qu'au moins un indicateur de tendance est présent
        has_trend_strength = 'trend_strength' in self.context and self.context['trend_strength'] is not None
        has_trend_angle = 'trend_angle' in self.context and self.context['trend_angle'] is not None
        has_trend_alignment = 'trend_alignment' in self.context and self.context['trend_alignment'] is not None
        has_emas = ('ema_7' in self.context and 'ema_12' in self.context and 
                   'ema_26' in self.context and self.context['ema_7'] is not None)
        
        if not (has_trend_strength or has_trend_angle or has_trend_alignment or has_emas):
            logger.warning(f"{self.name}: Aucun indicateur de tendance disponible pour {self.symbol}")
            return False
            
        return True
