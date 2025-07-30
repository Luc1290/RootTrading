"""
Volume_Quality_Score_Validator - Validator basé sur la qualité du volume pour filtrer les signaux selon l'activité et la liquidité.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class Volume_Quality_Score_Validator(BaseValidator):
    """
    Validator basé sur la qualité du volume - filtre les signaux selon la liquidité et l'authenticité du volume.
    
    Vérifie: Score qualité volume, patterns volume, contexte volume, ratio quote/base
    Catégorie: volume
    
    Rejette les signaux en:
    - Volume de mauvaise qualité (manipulation/bots)
    - Volume insuffisant ou anormalement faible
    - Patterns de volume suspects
    - Divergence prix/volume défavorable
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "Volume_Quality_Score_Validator"
        self.category = "volume"
        
        # Paramètres de qualité volume
        self.min_quality_score = 0.4          # Score qualité minimum
        self.good_quality_score = 0.6         # Score bonne qualité
        self.excellent_quality_score = 0.8    # Score excellente qualité
        
        # Paramètres de volume relatif
        self.min_relative_volume = 0.5        # Volume min vs moyenne
        self.high_relative_volume = 2.0       # Volume élevé vs moyenne
        self.extreme_relative_volume = 5.0    # Volume extrême vs moyenne
        
        # Paramètres patterns volume
        self.min_volume_consistency = 0.3     # Consistance minimum
        self.spike_threshold = 3.0            # Seuil spike volume
        self.buildup_periods_min = 3          # Périodes buildup minimum
        
        # Paramètres divergence
        self.max_price_volume_divergence = 0.3 # Divergence max acceptable
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur la qualité du volume.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon le volume, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs de volume depuis le contexte
            try:
                volume_quality_score = float(self.context.get('volume_quality_score', 0)) if self.context.get('volume_quality_score') is not None else None
                relative_volume = float(self.context.get('relative_volume', 1.0)) if self.context.get('relative_volume') is not None else 1.0
                volume_pattern = self.context.get('volume_pattern')
                volume_context = self.context.get('volume_context')
                
                # Indicateurs additionnels volume
                volume_ratio = float(self.context.get('volume_ratio', 1.0)) if self.context.get('volume_ratio') is not None else 1.0
                quote_volume_ratio = float(self.context.get('quote_volume_ratio', 1.0)) if self.context.get('quote_volume_ratio') is not None else None
                avg_trade_size = float(self.context.get('avg_trade_size', 0)) if self.context.get('avg_trade_size') is not None else None
                trade_intensity = float(self.context.get('trade_intensity', 0)) if self.context.get('trade_intensity') is not None else None
                volume_spike_multiplier = float(self.context.get('volume_spike_multiplier', 1.0)) if self.context.get('volume_spike_multiplier') is not None else 1.0
                volume_buildup_periods = int(self.context.get('volume_buildup_periods', 0)) if self.context.get('volume_buildup_periods') is not None else 0
                
                # OBV et AD
                obv = float(self.context.get('obv', 0)) if self.context.get('obv') is not None else None
                obv_oscillator = float(self.context.get('obv_oscillator', 0)) if self.context.get('obv_oscillator') is not None else None
                ad_line = float(self.context.get('ad_line', 0)) if self.context.get('ad_line') is not None else None
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion volume pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_confidence = signal.get('confidence', 0.0)
            signal_strength = signal.get('strength', 'moderate')
            
            if not signal_side:
                logger.warning(f"{self.name}: Signal side manquant pour {self.symbol}")
                return False
                
            # 1. Vérification score qualité volume
            if volume_quality_score is not None:
                if volume_quality_score < self.min_quality_score:
                    logger.debug(f"{self.name}: Qualité volume insuffisante ({self._safe_format(volume_quality_score, '.2f')}) pour {self.symbol}")
                    return False
                    
                # Volume excellent - signal favorisé
                if volume_quality_score >= self.excellent_quality_score:
                    logger.debug(f"{self.name}: Excellente qualité volume ({self._safe_format(volume_quality_score, '.2f')}) pour {self.symbol}")
                    
            # 2. Vérification volume relatif
            if relative_volume < self.min_relative_volume:
                logger.debug(f"{self.name}: Volume relatif trop faible ({self._safe_format(relative_volume, '.2f')}) pour {self.symbol}")
                return False
                
            # Volume extrême - vigilance manipulation
            if relative_volume >= self.extreme_relative_volume:
                if signal_confidence < 0.8:
                    logger.debug(f"{self.name}: Volume extrême ({self._safe_format(relative_volume, '.2f')}x) nécessite confidence élevée pour {self.symbol}")
                    return False
                    
            # 3. Vérification patterns volume
            if volume_pattern:
                # Patterns suspects
                suspicious_patterns = ['wash_trading', 'bot_activity', 'fake_volume', 'pump_activity']
                if volume_pattern in suspicious_patterns:
                    logger.debug(f"{self.name}: Pattern volume suspect détecté ({volume_pattern}) pour {self.symbol}")
                    return False
                    
                # Patterns favorables selon signal
                if signal_side == "BUY":
                    favorable_patterns = ['accumulation', 'buildup', 'breakout_volume', 'institutional_buying']
                    if volume_pattern in favorable_patterns:
                        logger.debug(f"{self.name}: Pattern volume favorable pour BUY ({volume_pattern}) pour {self.symbol}")
                        return True
                        
                elif signal_side == "SELL":
                    favorable_patterns = ['distribution', 'selling_pressure', 'institutional_selling', 'breakdown_volume']
                    if volume_pattern in favorable_patterns:
                        logger.debug(f"{self.name}: Pattern volume favorable pour SELL ({volume_pattern}) pour {self.symbol}")
                        return True
                        
            # 4. Vérification contexte volume
            if volume_context:
                # Contextes défavorables
                if volume_context in ['no_interest', 'dead_market', 'low_liquidity']:
                    logger.debug(f"{self.name}: Contexte volume défavorable ({volume_context}) pour {self.symbol}")
                    return False
                    
                # Contextes nécessitant prudence
                if volume_context in ['unusual_activity', 'spike', 'abnormal']:
                    if signal_confidence < 0.7:
                        logger.debug(f"{self.name}: Contexte volume inhabituel nécessite confidence élevée pour {self.symbol}")
                        return False
                        
            # 5. Vérification spike volume
            if volume_spike_multiplier >= self.spike_threshold:
                # Spike important - vérifier authenticité
                if volume_quality_score is not None and volume_quality_score < self.good_quality_score:
                    logger.debug(f"{self.name}: Spike volume ({self._safe_format(volume_spike_multiplier, '.1f')}x) avec qualité faible pour {self.symbol}")
                    return False
                    
                # Spike authentique mais besoin de confirmation
                if signal_strength in ['weak', 'very_weak']:
                    logger.debug(f"{self.name}: Spike volume nécessite signal fort pour {self.symbol}")
                    return False
                    
            # 6. Vérification buildup volume
            if volume_buildup_periods > 0:
                # Buildup en cours - favorable si suffisant
                if volume_buildup_periods >= self.buildup_periods_min:
                    logger.debug(f"{self.name}: Buildup volume favorable ({volume_buildup_periods} périodes) pour {self.symbol}")
                else:
                    # Buildup trop court
                    if signal_confidence < 0.6:
                        logger.debug(f"{self.name}: Buildup volume insuffisant nécessite confidence pour {self.symbol}")
                        return False
                        
            # 7. Vérification OBV/AD divergences
            if obv_oscillator is not None:
                # Divergence OBV défavorable
                if signal_side == "BUY" and obv_oscillator < -0.2:
                    logger.debug(f"{self.name}: Divergence OBV négative pour signal BUY pour {self.symbol}")
                    return False
                elif signal_side == "SELL" and obv_oscillator > 0.2:
                    logger.debug(f"{self.name}: Divergence OBV positive pour signal SELL pour {self.symbol}")
                    return False
                    
            # 8. Vérification trade intensity
            if trade_intensity is not None:
                # Intensité très faible = peu d'intérêt
                if trade_intensity < 0.1:
                    logger.debug(f"{self.name}: Intensité trading trop faible ({self._safe_format(trade_intensity, '.2f')}) pour {self.symbol}")
                    return False
                    
            # 9. Vérification cohérence quote/base volume
            if quote_volume_ratio is not None:
                # Ratio anormal peut indiquer manipulation
                if quote_volume_ratio < 0.5 or quote_volume_ratio > 2.0:
                    if signal_confidence < 0.7:
                        logger.debug(f"{self.name}: Ratio quote/base volume anormal ({self._safe_format(quote_volume_ratio, '.2f')}) pour {self.symbol}")
                        return False
                        
            logger.debug(f"{self.name}: Signal validé pour {self.symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur la qualité du volume.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur le volume
            volume_quality_score = float(self.context.get('volume_quality_score', 0.5)) if self.context.get('volume_quality_score') is not None else 0.5
            relative_volume = float(self.context.get('relative_volume', 1.0)) if self.context.get('relative_volume') is not None else 1.0
            volume_pattern = self.context.get('volume_pattern')
            volume_spike_multiplier = float(self.context.get('volume_spike_multiplier', 1.0)) if self.context.get('volume_spike_multiplier') is not None else 1.0
            volume_buildup_periods = int(self.context.get('volume_buildup_periods', 0)) if self.context.get('volume_buildup_periods') is not None else 0
            trade_intensity = float(self.context.get('trade_intensity', 0)) if self.context.get('trade_intensity') is not None else None
            
            signal_side = signal.get('side')
            base_score = 0.5  # Score de base si validé
            
            # Bonus qualité volume
            if volume_quality_score >= self.excellent_quality_score:
                base_score += 0.3  # Excellente qualité
            elif volume_quality_score >= self.good_quality_score:
                base_score += 0.2  # Bonne qualité
            elif volume_quality_score >= self.min_quality_score:
                base_score += 0.1  # Qualité acceptable
                
            # Bonus volume relatif
            if relative_volume >= self.high_relative_volume:
                base_score += 0.15  # Volume élevé
            elif relative_volume >= 1.5:
                base_score += 0.1   # Volume supérieur moyenne
            elif relative_volume >= 1.0:
                base_score += 0.05  # Volume normal
                
            # Bonus patterns favorables
            if volume_pattern:
                buy_patterns = ['accumulation', 'buildup', 'breakout_volume', 'institutional_buying']
                sell_patterns = ['distribution', 'selling_pressure', 'institutional_selling', 'breakdown_volume']
                
                if (signal_side == "BUY" and volume_pattern in buy_patterns) or \
                   (signal_side == "SELL" and volume_pattern in sell_patterns):
                    base_score += 0.2  # Pattern très favorable
                    
            # Bonus spike authentique
            if volume_spike_multiplier >= self.spike_threshold and volume_quality_score >= self.good_quality_score:
                base_score += 0.1  # Spike de bonne qualité
                
            # Bonus buildup
            if volume_buildup_periods >= self.buildup_periods_min:
                base_score += 0.1  # Accumulation/distribution confirmée
                
            # Bonus trade intensity
            if trade_intensity is not None:
                if trade_intensity >= 1.0:
                    base_score += 0.1  # Forte activité
                elif trade_intensity >= 0.5:
                    base_score += 0.05  # Activité normale
                    
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
            volume_quality_score = float(self.context.get('volume_quality_score', 0)) if self.context.get('volume_quality_score') is not None else 0
            relative_volume = float(self.context.get('relative_volume', 1.0)) if self.context.get('relative_volume') is not None else 1.0
            volume_pattern = self.context.get('volume_pattern', 'N/A')
            volume_context = self.context.get('volume_context', 'N/A')
            
            if is_valid:
                # Déterminer qualité volume
                if volume_quality_score >= self.excellent_quality_score:
                    quality = "excellente"
                elif volume_quality_score >= self.good_quality_score:
                    quality = "bonne"
                else:
                    quality = "acceptable"
                    
                # Déterminer conditions
                if relative_volume >= self.high_relative_volume:
                    condition = f"volume élevé ({self._safe_format(relative_volume, '.1f')}x)"
                elif volume_pattern and volume_pattern != 'N/A':
                    condition = f"pattern {volume_pattern}"
                else:
                    condition = f"volume {quality}"
                    
                return f"{self.name}: Validé - {condition} (score: {self._safe_format(volume_quality_score, '.2f')}) pour signal {signal_side}"
            else:
                if volume_quality_score < self.min_quality_score:
                    return f"{self.name}: Rejeté - Qualité volume insuffisante ({self._safe_format(volume_quality_score, '.2f')})"
                elif relative_volume < self.min_relative_volume:
                    return f"{self.name}: Rejeté - Volume trop faible ({self._safe_format(relative_volume, '.2f')}x moyenne)"
                elif volume_pattern in ['wash_trading', 'bot_activity', 'fake_volume']:
                    return f"{self.name}: Rejeté - Pattern volume suspect ({volume_pattern})"
                elif volume_context in ['no_interest', 'dead_market']:
                    return f"{self.name}: Rejeté - Contexte volume défavorable ({volume_context})"
                else:
                    return f"{self.name}: Rejeté - Conditions volume inadéquates"
                    
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données de volume requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Vérifier qu'au moins un indicateur de volume est présent
        has_quality_score = 'volume_quality_score' in self.context and self.context['volume_quality_score'] is not None
        has_relative_volume = 'relative_volume' in self.context and self.context['relative_volume'] is not None
        has_volume_pattern = 'volume_pattern' in self.context and self.context['volume_pattern'] is not None
        has_volume_ratio = 'volume_ratio' in self.context and self.context['volume_ratio'] is not None
        has_obv = 'obv' in self.context and self.context['obv'] is not None
        
        if not (has_quality_score or has_relative_volume or has_volume_pattern or has_volume_ratio or has_obv):
            logger.warning(f"{self.name}: Aucun indicateur de volume disponible pour {self.symbol}")
            return False
            
        return True
