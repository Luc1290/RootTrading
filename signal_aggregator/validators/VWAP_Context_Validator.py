"""
VWAP_Context_Validator - Validator basé sur le contexte VWAP pour valider les signaux selon le prix moyen pondéré par volume.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class VWAP_Context_Validator(BaseValidator):
    """
    Validator basé sur le contexte VWAP - valide les signaux selon leur position relative au VWAP et aux bandes.
    
    Vérifie: Position vs VWAP, bandes VWAP, divergences prix/VWAP, anchored VWAP
    Catégorie: volume
    
    Rejette les signaux en:
    - BUY bien au-dessus VWAP sans momentum
    - SELL bien en-dessous VWAP sans momentum
    - Divergences VWAP défavorables
    - Position dans bandes VWAP extrêmes
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "VWAP_Context_Validator"
        self.category = "volume"
        
        # Paramètres position VWAP (en % du prix)
        self.close_to_vwap_threshold = 0.005   # 0.5% proche du VWAP
        self.moderate_distance_threshold = 0.02 # 2% distance modérée
        self.far_from_vwap_threshold = 0.05    # 5% distance importante
        self.extreme_distance_threshold = 0.1  # 10% distance extrême
        
        # Paramètres bandes VWAP
        self.upper_band_threshold = 0.8        # Position haute dans les bandes
        self.lower_band_threshold = 0.2        # Position basse dans les bandes
        self.extreme_upper_threshold = 0.95    # Position extrême haute
        self.extreme_lower_threshold = 0.05    # Position extrême basse
        
        # Paramètres momentum vs VWAP
        self.min_momentum_for_distance = 0.3   # Momentum min si loin du VWAP
        self.strong_momentum_threshold = 0.7   # Momentum fort
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur le contexte VWAP.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon VWAP, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs VWAP depuis le contexte
            try:
                vwap_10 = float(self.context.get('vwap_10', 0)) if self.context.get('vwap_10') is not None else None
                vwap_quote_10 = float(self.context.get('vwap_quote_10', 0)) if self.context.get('vwap_quote_10') is not None else None
                anchored_vwap = float(self.context.get('anchored_vwap', 0)) if self.context.get('anchored_vwap') is not None else None
                vwap_upper_band = float(self.context.get('vwap_upper_band', 0)) if self.context.get('vwap_upper_band') is not None else None
                vwap_lower_band = float(self.context.get('vwap_lower_band', 0)) if self.context.get('vwap_lower_band') is not None else None
                
                # Prix actuel
                current_price = None
                if 'close' in self.data:
                    current_price = float(self.data['close'])
                elif 'price' in signal:
                    current_price = float(signal['price'])
                    
                # Indicateurs complémentaires
                momentum_score = float(self.context.get('momentum_score', 0)) if self.context.get('momentum_score') is not None else None
                trend_strength = self._convert_trend_strength_to_score(self.context.get('trend_strength')) if self.context.get('trend_strength') is not None else None
                volume_quality_score = float(self.context.get('volume_quality_score', 0)) if self.context.get('volume_quality_score') is not None else None
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion VWAP pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_confidence = signal.get('confidence', 0.0)
            signal_strength = signal.get('strength', 'moderate')
            
            if not signal_side or current_price is None:
                logger.warning(f"{self.name}: Signal side ou prix manquant pour {self.symbol}")
                return False
                
            # Utiliser VWAP principal (vwap_10 prioritaire, sinon anchored_vwap)
            main_vwap = vwap_10 if vwap_10 is not None else anchored_vwap
            if main_vwap is None:
                logger.warning(f"{self.name}: Aucun VWAP disponible pour {self.symbol}")
                return False
                
            # Calcul distance relative au VWAP
            vwap_distance_pct = abs(current_price - main_vwap) / main_vwap
            price_above_vwap = current_price > main_vwap
            
            # 1. Validation position vs VWAP selon signal
            if signal_side == "BUY":
                # BUY bien au-dessus VWAP = risqué sauf momentum fort
                if price_above_vwap and vwap_distance_pct >= self.far_from_vwap_threshold:
                    if momentum_score is None or momentum_score < self.min_momentum_for_distance:
                        logger.debug(f"{self.name}: BUY trop au-dessus VWAP ({vwap_distance_pct:.2%}) sans momentum pour {self.symbol}")
                        return False
                        
                    # Distance extrême nécessite momentum très fort
                    if vwap_distance_pct >= self.extreme_distance_threshold:
                        if momentum_score < self.strong_momentum_threshold:
                            logger.debug(f"{self.name}: BUY extrêmement au-dessus VWAP ({vwap_distance_pct:.2%}) pour {self.symbol}")
                            return False
                            
                # BUY proche ou en-dessous VWAP = favorable
                elif not price_above_vwap or vwap_distance_pct <= self.moderate_distance_threshold:
                    logger.debug(f"{self.name}: BUY favorable vs VWAP ({vwap_distance_pct:.2%}) pour {self.symbol}")
                    
            elif signal_side == "SELL":
                # SELL bien en-dessous VWAP = risqué sauf momentum fort
                if not price_above_vwap and vwap_distance_pct >= self.far_from_vwap_threshold:
                    if momentum_score is None or momentum_score > -self.min_momentum_for_distance:
                        logger.debug(f"{self.name}: SELL trop en-dessous VWAP ({vwap_distance_pct:.2%}) sans momentum pour {self.symbol}")
                        return False
                        
                    # Distance extrême nécessite momentum très fort
                    if vwap_distance_pct >= self.extreme_distance_threshold:
                        if momentum_score > -self.strong_momentum_threshold:
                            logger.debug(f"{self.name}: SELL extrêmement en-dessous VWAP ({vwap_distance_pct:.2%}) pour {self.symbol}")
                            return False
                            
                # SELL proche ou au-dessus VWAP = favorable
                elif price_above_vwap or vwap_distance_pct <= self.moderate_distance_threshold:
                    logger.debug(f"{self.name}: SELL favorable vs VWAP ({vwap_distance_pct:.2%}) pour {self.symbol}")
                    
            # 2. Validation bandes VWAP si disponibles
            if vwap_upper_band is not None and vwap_lower_band is not None:
                # Calcul position dans les bandes (0 = lower, 1 = upper)
                band_position = (current_price - vwap_lower_band) / (vwap_upper_band - vwap_lower_band)
                
                # Position extrême dans les bandes
                if signal_side == "BUY" and band_position >= self.extreme_upper_threshold:
                    if signal_confidence < 0.8:
                        logger.debug(f"{self.name}: BUY en position extrême haute bandes VWAP ({band_position:.2f}) pour {self.symbol}")
                        return False
                        
                elif signal_side == "SELL" and band_position <= self.extreme_lower_threshold:
                    if signal_confidence < 0.8:
                        logger.debug(f"{self.name}: SELL en position extrême basse bandes VWAP ({band_position:.2f}) pour {self.symbol}")
                        return False
                        
                # Position haute/basse modérée
                elif signal_side == "BUY" and band_position >= self.upper_band_threshold:
                    if signal_strength in ['weak', 'very_weak']:
                        logger.debug(f"{self.name}: BUY position haute nécessite signal fort pour {self.symbol}")
                        return False
                        
                elif signal_side == "SELL" and band_position <= self.lower_band_threshold:
                    if signal_strength in ['weak', 'very_weak']:
                        logger.debug(f"{self.name}: SELL position basse nécessite signal fort pour {self.symbol}")
                        return False
                        
            # 3. Validation anchored VWAP vs VWAP standard si les deux disponibles
            if vwap_10 is not None and anchored_vwap is not None:
                vwap_divergence = abs(vwap_10 - anchored_vwap) / vwap_10
                
                # Divergence importante entre VWAPs
                if vwap_divergence > 0.02:  # 2% de divergence
                    # Vérifier cohérence avec le signal
                    if signal_side == "BUY":
                        if current_price < min(vwap_10, anchored_vwap):
                            logger.debug(f"{self.name}: BUY favorable - prix sous les deux VWAPs pour {self.symbol}")
                        elif current_price > max(vwap_10, anchored_vwap) and signal_confidence < 0.7:
                            logger.debug(f"{self.name}: BUY au-dessus des deux VWAPs nécessite confidence pour {self.symbol}")
                            return False
                            
                    elif signal_side == "SELL":
                        if current_price > max(vwap_10, anchored_vwap):
                            logger.debug(f"{self.name}: SELL favorable - prix au-dessus des deux VWAPs pour {self.symbol}")
                        elif current_price < min(vwap_10, anchored_vwap) and signal_confidence < 0.7:
                            logger.debug(f"{self.name}: SELL sous les deux VWAPs nécessite confidence pour {self.symbol}")
                            return False
                            
            # 4. Validation quote VWAP si disponible
            if vwap_quote_10 is not None and vwap_10 is not None:
                quote_vwap_ratio = vwap_quote_10 / vwap_10
                
                # Ratio anormal peut indiquer problème de liquidité
                if quote_vwap_ratio < 0.8 or quote_vwap_ratio > 1.2:
                    if signal_confidence < 0.6:
                        logger.debug(f"{self.name}: Ratio VWAP quote/base anormal ({quote_vwap_ratio:.2f}) pour {self.symbol}")
                        return False
                        
            # 5. Intégration qualité volume si disponible
            if volume_quality_score is not None:
                # Volume de mauvaise qualité + position défavorable VWAP = rejet
                if volume_quality_score < 0.5:
                    if ((signal_side == "BUY" and price_above_vwap and vwap_distance_pct > self.moderate_distance_threshold) or
                        (signal_side == "SELL" and not price_above_vwap and vwap_distance_pct > self.moderate_distance_threshold)):
                        logger.debug(f"{self.name}: Qualité volume faible + position VWAP défavorable pour {self.symbol}")
                        return False
                        
            # 6. Validation trend strength vs position VWAP
            if trend_strength is not None:
                # Tendance faible + position défavorable = rejet
                if trend_strength < 0.4:
                    if vwap_distance_pct >= self.far_from_vwap_threshold:
                        logger.debug(f"{self.name}: Tendance faible + loin du VWAP pour {self.symbol}")
                        return False
                        
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                        f"Prix: {current_price:.2f}, "
                        f"VWAP: {main_vwap:.2f}, "
                        f"Distance: {vwap_distance_pct:.2%}, "
                        f"Side: {signal_side}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur le contexte VWAP.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur VWAP
            vwap_10 = float(self.context.get('vwap_10', 0)) if self.context.get('vwap_10') is not None else None
            anchored_vwap = float(self.context.get('anchored_vwap', 0)) if self.context.get('anchored_vwap') is not None else None
            vwap_upper_band = float(self.context.get('vwap_upper_band', 0)) if self.context.get('vwap_upper_band') is not None else None
            vwap_lower_band = float(self.context.get('vwap_lower_band', 0)) if self.context.get('vwap_lower_band') is not None else None
            momentum_score = float(self.context.get('momentum_score', 0)) if self.context.get('momentum_score') is not None else 0
            
            current_price = float(self.data.get('close', signal.get('price', 0)))
            signal_side = signal.get('side')
            
            main_vwap = vwap_10 if vwap_10 is not None else anchored_vwap
            if main_vwap is None:
                return 0.5  # Score neutre si pas de VWAP
                
            base_score = 0.5  # Score de base si validé
            
            # Calcul distance et position
            vwap_distance_pct = abs(current_price - main_vwap) / main_vwap
            price_above_vwap = current_price > main_vwap
            
            # Scoring selon position favorable
            if signal_side == "BUY":
                if not price_above_vwap:
                    # BUY en-dessous VWAP = excellent
                    base_score += 0.3
                elif vwap_distance_pct <= self.close_to_vwap_threshold:
                    # BUY très proche VWAP = très bon
                    base_score += 0.25
                elif vwap_distance_pct <= self.moderate_distance_threshold:
                    # BUY proche VWAP = bon
                    base_score += 0.15
                else:
                    # BUY loin VWAP mais avec momentum
                    if momentum_score >= self.strong_momentum_threshold:
                        base_score += 0.1
                        
            elif signal_side == "SELL":
                if price_above_vwap:
                    # SELL au-dessus VWAP = excellent
                    base_score += 0.3
                elif vwap_distance_pct <= self.close_to_vwap_threshold:
                    # SELL très proche VWAP = très bon
                    base_score += 0.25
                elif vwap_distance_pct <= self.moderate_distance_threshold:
                    # SELL proche VWAP = bon
                    base_score += 0.15
                else:
                    # SELL loin VWAP mais avec momentum
                    if momentum_score <= -self.strong_momentum_threshold:
                        base_score += 0.1
                        
            # Bonus position dans bandes VWAP
            if vwap_upper_band is not None and vwap_lower_band is not None:
                band_position = (current_price - vwap_lower_band) / (vwap_upper_band - vwap_lower_band)
                
                if signal_side == "BUY":
                    if band_position <= 0.3:  # Position basse favorable pour BUY
                        base_score += 0.1
                    elif band_position <= 0.5:  # Position moyenne acceptable
                        base_score += 0.05
                        
                elif signal_side == "SELL":
                    if band_position >= 0.7:  # Position haute favorable pour SELL
                        base_score += 0.1
                    elif band_position >= 0.5:  # Position moyenne acceptable
                        base_score += 0.05
                        
            # Bonus cohérence VWAPs multiples
            if vwap_10 is not None and anchored_vwap is not None:
                vwap_coherence = 1 - min(abs(vwap_10 - anchored_vwap) / vwap_10, 0.1)
                if vwap_coherence > 0.98:  # VWAPs très cohérents
                    base_score += 0.05
                    
            # Bonus momentum cohérent
            if signal_side == "BUY" and momentum_score > 0.5:
                base_score += 0.05
            elif signal_side == "SELL" and momentum_score < -0.5:
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
            vwap_10 = float(self.context.get('vwap_10', 0)) if self.context.get('vwap_10') is not None else None
            anchored_vwap = float(self.context.get('anchored_vwap', 0)) if self.context.get('anchored_vwap') is not None else None
            
            main_vwap = vwap_10 if vwap_10 is not None else anchored_vwap
            if main_vwap is None:
                return f"{self.name}: VWAP non disponible"
                
            vwap_distance_pct = abs(current_price - main_vwap) / main_vwap
            price_above_vwap = current_price > main_vwap
            position_desc = "au-dessus" if price_above_vwap else "en-dessous"
            
            if is_valid:
                if signal_side == "BUY" and not price_above_vwap:
                    condition = f"prix {position_desc} VWAP favorable pour BUY"
                elif signal_side == "SELL" and price_above_vwap:
                    condition = f"prix {position_desc} VWAP favorable pour SELL"
                elif vwap_distance_pct <= self.close_to_vwap_threshold:
                    condition = f"très proche VWAP ({vwap_distance_pct:.1%})"
                else:
                    condition = f"position VWAP acceptable ({vwap_distance_pct:.1%})"
                    
                return f"{self.name}: Validé - {condition} pour signal {signal_side}"
            else:
                if vwap_distance_pct >= self.extreme_distance_threshold:
                    return f"{self.name}: Rejeté - Distance VWAP extrême ({vwap_distance_pct:.1%})"
                elif signal_side == "BUY" and price_above_vwap and vwap_distance_pct >= self.far_from_vwap_threshold:
                    return f"{self.name}: Rejeté - BUY trop au-dessus VWAP ({vwap_distance_pct:.1%})"
                elif signal_side == "SELL" and not price_above_vwap and vwap_distance_pct >= self.far_from_vwap_threshold:
                    return f"{self.name}: Rejeté - SELL trop en-dessous VWAP ({vwap_distance_pct:.1%})"
                else:
                    return f"{self.name}: Rejeté - Position VWAP défavorable"
                    
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données VWAP requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Vérifier qu'au moins un VWAP est présent
        has_vwap_10 = 'vwap_10' in self.context and self.context['vwap_10'] is not None
        has_anchored_vwap = 'anchored_vwap' in self.context and self.context['anchored_vwap'] is not None
        has_vwap_quote = 'vwap_quote_10' in self.context and self.context['vwap_quote_10'] is not None
        
        if not (has_vwap_10 or has_anchored_vwap or has_vwap_quote):
            logger.warning(f"{self.name}: Aucun VWAP disponible pour {self.symbol}")
            return False
            
        # Vérifier prix disponible
        has_price = 'close' in self.data or ('price' in self.context and self.context['price'] is not None)
        if not has_price:
            logger.warning(f"{self.name}: Prix manquant pour {self.symbol}")
            return False
            
        return True
