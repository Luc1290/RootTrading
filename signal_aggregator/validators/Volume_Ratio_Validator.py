"""
Volume_Ratio_Validator - Validator basé sur les ratios de volume pour valider la cohérence et l'intérêt du marché.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class Volume_Ratio_Validator(BaseValidator):
    """
    Validator basé sur les ratios de volume - valide les signaux selon les comparaisons de volume historiques.
    
    Vérifie: Ratios volume/moyenne, cohérence quote/base, taille trades, intensité relative
    Catégorie: volume
    
    Rejette les signaux en:
    - Volume très faible vs moyenne historique
    - Ratios anormaux indiquant manipulation
    - Incohérence entre volume base et quote
    - Activité de trading insuffisante
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "Volume_Ratio_Validator"
        self.category = "volume"
        
        # Paramètres ratios volume
        self.min_volume_ratio = 0.3           # Ratio volume minimum vs moyenne (30%)
        self.low_volume_ratio = 0.5           # Ratio volume faible (50%)
        self.normal_volume_ratio = 1.0        # Ratio volume normal
        self.high_volume_ratio = 2.0          # Ratio volume élevé
        self.extreme_volume_ratio = 5.0       # Ratio volume extrême
        
        # Paramètres quote/base volume
        self.min_quote_volume_ratio = 0.3     # Ratio quote/base minimum (30%)
        self.max_quote_volume_ratio = 3.0     # Ratio quote/base maximum
        self.ideal_quote_ratio_min = 0.8      # Ratio idéal minimum
        self.ideal_quote_ratio_max = 1.2      # Ratio idéal maximum
        
        # Paramètres taille moyenne des trades
        self.min_avg_trade_multiplier = 0.5   # Multiplicateur taille trade min
        self.large_trade_multiplier = 2.0     # Multiplicateur gros trades
        self.whale_trade_multiplier = 5.0     # Multiplicateur trades baleine
        
        # Paramètres intensité
        self.min_trade_intensity = 0.2        # Intensité trading minimum
        self.high_intensity_threshold = 1.5   # Seuil haute intensité
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur les ratios de volume.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon les ratios, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs de ratio volume depuis le contexte
            try:
                volume_ratio = float(self.context.get('volume_ratio', 1.0)) if self.context.get('volume_ratio') is not None else 1.0
                quote_volume_ratio = float(self.context.get('quote_volume_ratio', 1.0)) if self.context.get('quote_volume_ratio') is not None else None
                relative_volume = float(self.context.get('relative_volume', 1.0)) if self.context.get('relative_volume') is not None else 1.0
                
                # Indicateurs taille trades
                avg_trade_size = float(self.context.get('avg_trade_size', 0)) if self.context.get('avg_trade_size') is not None else None
                trade_intensity = float(self.context.get('trade_intensity', 0)) if self.context.get('trade_intensity') is not None else None
                
                # Volumes moyens historiques pour comparaison
                avg_volume_20 = float(self.context.get('avg_volume_20', 0)) if self.context.get('avg_volume_20') is not None else None
                
                # Volume actuel
                current_volume = None
                if 'volume' in self.data:
                    current_volume = float(self.data['volume'])
                    
                # Calculs dérivés
                volume_vs_avg_ratio = None
                if current_volume is not None and avg_volume_20 is not None and avg_volume_20 > 0:
                    volume_vs_avg_ratio = current_volume / avg_volume_20
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion ratios volume pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_confidence = signal.get('confidence', 0.0)
            signal_strength = signal.get('strength', 'moderate')
            
            if not signal_side:
                logger.warning(f"{self.name}: Signal side manquant pour {self.symbol}")
                return False
                
            # 1. Vérification ratio volume principal
            if volume_ratio < self.min_volume_ratio:
                logger.debug(f"{self.name}: Volume ratio trop faible ({self._safe_format(volume_ratio, '.2f')}) pour {self.symbol}")
                return False
                
            # Volume extrêmement élevé - suspicion manipulation
            if volume_ratio >= self.extreme_volume_ratio:
                if signal_confidence < 0.85:
                    logger.debug(f"{self.name}: Volume ratio extrême ({self._safe_format(volume_ratio, '.2f')}) nécessite confidence très élevée pour {self.symbol}")
                    return False
                    
            # 2. Vérification volume relatif calculé
            if relative_volume < self.min_volume_ratio:
                logger.debug(f"{self.name}: Volume relatif trop faible ({self._safe_format(relative_volume, '.2f')}) pour {self.symbol}")
                return False
                
            # 3. Vérification volume vs moyenne historique si disponible
            if volume_vs_avg_ratio is not None:
                if volume_vs_avg_ratio < self.min_volume_ratio:
                    logger.debug(f"{self.name}: Volume vs moyenne historique trop faible ({self._safe_format(volume_vs_avg_ratio, '.2f')}) pour {self.symbol}")
                    return False
                    
                # Volume très élevé vs historique
                if volume_vs_avg_ratio >= self.extreme_volume_ratio:
                    if signal_confidence < 0.8:
                        logger.debug(f"{self.name}: Volume vs historique extrême ({self._safe_format(volume_vs_avg_ratio, '.2f')}) pour {self.symbol}")
                        return False
                        
            # 4. Vérification ratio quote/base volume avec logique directionnelle
            if quote_volume_ratio is not None:
                # Ratio anormal indique possibles manipulations
                if quote_volume_ratio < self.min_quote_volume_ratio or quote_volume_ratio > self.max_quote_volume_ratio:
                    logger.debug(f"{self.name}: Ratio quote/base anormal ({self._safe_format(quote_volume_ratio, '.2f')}) pour {self.symbol}")
                    return False
                    
                # Logique directionnelle pour quote/base ratio
                if signal_side == "BUY":
                    # Pour BUY: ratio élevé (>1.0) indique forte demande en quote currency
                    if quote_volume_ratio > 1.2:
                        logger.debug(f"{self.name}: Quote/base ratio élevé favorable pour BUY ({self._safe_format(quote_volume_ratio, '.2f')}) pour {self.symbol}")
                    elif quote_volume_ratio < 0.8:
                        logger.debug(f"{self.name}: Quote/base ratio faible défavorable pour BUY ({self._safe_format(quote_volume_ratio, '.2f')}) pour {self.symbol}")
                        if signal_confidence < 0.7:
                            return False
                            
                elif signal_side == "SELL":
                    # Pour SELL: ratio faible (<1.0) peut indiquer sortie de positions
                    if quote_volume_ratio < 0.8:
                        logger.debug(f"{self.name}: Quote/base ratio faible cohérent avec SELL ({self._safe_format(quote_volume_ratio, '.2f')}) pour {self.symbol}")
                    elif quote_volume_ratio > 1.5:
                        logger.debug(f"{self.name}: Quote/base ratio élevé incohérent avec SELL ({self._safe_format(quote_volume_ratio, '.2f')}) pour {self.symbol}")
                        if signal_confidence < 0.75:
                            return False
                    
                # Ratio idéal - signal favorisé
                if self.ideal_quote_ratio_min <= quote_volume_ratio <= self.ideal_quote_ratio_max:
                    logger.debug(f"{self.name}: Ratio quote/base idéal ({self._safe_format(quote_volume_ratio, '.2f')}) pour {self.symbol}")
                    
            # 5. Vérification taille moyenne des trades avec logique directionnelle
            if avg_trade_size is not None:
                # Note: avg_trade_size devrait être comparé à une baseline
                # Pour simplifier, on assume qu'il est déjà normalisé vs moyenne
                
                if signal_side == "BUY":
                    # Pour BUY: gros trades = accumulation institutionnelle (favorable)
                    if avg_trade_size >= self.large_trade_multiplier:
                        logger.debug(f"{self.name}: Gros trades favorables pour BUY ({self._safe_format(avg_trade_size, '.2f')}) - accumulation institutionnelle possible pour {self.symbol}")
                    elif avg_trade_size < self.min_avg_trade_multiplier:
                        # Petits trades pour BUY = retail FOMO (défavorable)
                        logger.debug(f"{self.name}: Petits trades défavorables pour BUY ({self._safe_format(avg_trade_size, '.2f')}) - possible retail FOMO pour {self.symbol}")
                        if signal_confidence < 0.75:
                            return False
                            
                elif signal_side == "SELL":
                    # Pour SELL: petits trades = distribution discrète (favorable)
                    if avg_trade_size < self.min_avg_trade_multiplier:
                        logger.debug(f"{self.name}: Petits trades cohérents avec SELL ({self._safe_format(avg_trade_size, '.2f')}) - distribution discrète pour {self.symbol}")
                    elif avg_trade_size >= self.whale_trade_multiplier:
                        # Très gros trades pour SELL = whale dump (dangereux mais peut être opportunité)
                        logger.debug(f"{self.name}: Trades whale détectés pour SELL ({self._safe_format(avg_trade_size, '.2f')}) pour {self.symbol}")
                        if signal_strength in ['weak', 'very_weak']:
                            logger.debug(f"{self.name}: Whale trades + signal faible = risque élevé pour {self.symbol}")
                            return False
                        
            # 6. Vérification intensité de trading avec logique directionnelle
            if trade_intensity is not None:
                if trade_intensity < self.min_trade_intensity:
                    logger.debug(f"{self.name}: Intensité trading trop faible ({self._safe_format(trade_intensity, '.2f')}) pour {self.symbol}")
                    return False
                    
                # Logique directionnelle pour intensité
                if signal_side == "BUY":
                    # Pour BUY: haute intensité = FOMO/momentum (nécessite prudence)
                    if trade_intensity >= self.high_intensity_threshold:
                        logger.debug(f"{self.name}: Haute intensité pour BUY ({self._safe_format(trade_intensity, '.2f')}) - possible FOMO pour {self.symbol}")
                        if signal_confidence < 0.8:  # Plus strict pour BUY avec haute intensité
                            logger.debug(f"{self.name}: Haute intensité BUY nécessite confidence élevée pour {self.symbol}")
                            return False
                    elif trade_intensity >= 1.0:
                        # Intensité modérée favorable pour BUY
                        logger.debug(f"{self.name}: Intensité modérée favorable pour BUY ({self._safe_format(trade_intensity, '.2f')}) pour {self.symbol}")
                        
                elif signal_side == "SELL":
                    # Pour SELL: haute intensité = panique/capitulation (opportunité si signal fort)
                    if trade_intensity >= self.high_intensity_threshold:
                        logger.debug(f"{self.name}: Haute intensité pour SELL ({self._safe_format(trade_intensity, '.2f')}) - possible panique pour {self.symbol}")
                        if signal_strength in ['strong', 'very_strong']:
                            logger.debug(f"{self.name}: Haute intensité + signal fort = opportunité SELL pour {self.symbol}")
                        elif signal_confidence < 0.75:
                            return False
                        
            # 7. Logique spécifique selon type de signal
            if signal_side == "BUY":
                # Pour BUY, volume élevé est généralement favorable
                if volume_ratio >= self.high_volume_ratio and signal_confidence >= 0.6:
                    logger.debug(f"{self.name}: Volume élevé favorable pour BUY ({self._safe_format(volume_ratio, '.2f')}) pour {self.symbol}")
                    return True
                    
            elif signal_side == "SELL":
                # Pour SELL, volume élevé peut indiquer panique ou distribution
                if volume_ratio >= self.high_volume_ratio:
                    # Volume élevé pour SELL est OK si signal fort
                    if signal_strength in ['strong', 'very_strong']:
                        logger.debug(f"{self.name}: Volume élevé + signal fort favorable pour SELL pour {self.symbol}")
                    elif signal_confidence < 0.7:
                        logger.debug(f"{self.name}: Volume élevé pour SELL nécessite confidence pour {self.symbol}")
                        return False
                        
            # 8. Validation finale cohérence globale avec logique directionnelle
            ratios_consistent = True
            
            # Vérifier cohérence entre différents ratios
            if quote_volume_ratio is not None and volume_ratio > 0:
                ratio_difference = abs(quote_volume_ratio - volume_ratio) / volume_ratio
                if ratio_difference > 0.5:  # 50% d'écart
                    if signal_confidence < 0.6:
                        logger.debug(f"{self.name}: Incohérence ratios volume ({self._safe_format(ratio_difference, '.2f')}) pour {self.symbol}")
                        ratios_consistent = False
                        
            # Validation croisée relative_volume avec signal direction
            if relative_volume is not None:
                if signal_side == "BUY":
                    # Pour BUY: volume relatif très faible est suspect
                    if relative_volume < 0.5 and signal_confidence < 0.7:
                        logger.debug(f"{self.name}: Volume relatif très faible pour BUY ({self._safe_format(relative_volume, '.2f')}) pour {self.symbol}")
                        return False
                elif signal_side == "SELL":
                    # Pour SELL: volume relatif peut être plus faible (distribution discrète)
                    if relative_volume < 0.3 and signal_confidence < 0.6:
                        logger.debug(f"{self.name}: Volume relatif extrêmement faible même pour SELL ({self._safe_format(relative_volume, '.2f')}) pour {self.symbol}")
                        return False
                        
            if not ratios_consistent and signal_strength in ['weak', 'very_weak']:
                return False
                
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                        f"Volume ratio: {self._safe_format(volume_ratio, '.2f')}, "
                        f"Quote ratio: {self._safe_format(quote_volume_ratio, '.2f') if quote_volume_ratio is not None else 'N/A'}, "
                        f"Relative: {self._safe_format(relative_volume, '.2f')}, "
                        f"Side: {signal_side}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur les ratios de volume.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur les ratios
            volume_ratio = float(self.context.get('volume_ratio', 1.0)) if self.context.get('volume_ratio') is not None else 1.0
            quote_volume_ratio = float(self.context.get('quote_volume_ratio', 1.0)) if self.context.get('quote_volume_ratio') is not None else 1.0
            relative_volume = float(self.context.get('relative_volume', 1.0)) if self.context.get('relative_volume') is not None else 1.0
            trade_intensity = float(self.context.get('trade_intensity', 0)) if self.context.get('trade_intensity') is not None else None
            avg_trade_size = float(self.context.get('avg_trade_size', 1.0)) if self.context.get('avg_trade_size') is not None else 1.0
            
            signal_side = signal.get('side')
            base_score = 0.5  # Score de base si validé
            
            # Bonus volume ratio
            if volume_ratio >= self.high_volume_ratio:
                if signal_side == "BUY":
                    base_score += 0.2  # Volume élevé favorable pour BUY
                else:
                    base_score += 0.1  # Volume élevé neutre pour SELL
            elif volume_ratio >= self.normal_volume_ratio:
                base_score += 0.15  # Volume normal bon
            elif volume_ratio >= self.low_volume_ratio:
                base_score += 0.05  # Volume faible acceptable
                
            # Bonus ratio quote/base avec logique directionnelle
            if quote_volume_ratio is not None:
                if signal_side == "BUY":
                    # Pour BUY: favoriser ratio > 1.0
                    if quote_volume_ratio > 1.2:
                        base_score += 0.20  # Forte demande quote
                    elif quote_volume_ratio >= 1.0:
                        base_score += 0.10  # Demande normale
                    elif quote_volume_ratio < 0.8:
                        base_score -= 0.10  # Faible demande
                        
                elif signal_side == "SELL":
                    # Pour SELL: ratio < 1.0 peut être favorable
                    if quote_volume_ratio < 0.8:
                        base_score += 0.15  # Distribution active
                    elif quote_volume_ratio <= 1.0:
                        base_score += 0.08  # Distribution normale
                    elif quote_volume_ratio > 1.5:
                        base_score -= 0.10  # Incohérent avec SELL
                    
            # Bonus volume relatif avec logique directionnelle
            if relative_volume is not None:
                if signal_side == "BUY":
                    if relative_volume >= 2.0:
                        base_score += 0.15  # Volume relatif élevé favorable pour BUY
                    elif relative_volume >= 1.0:
                        base_score += 0.08  # Volume relatif normal OK pour BUY
                    elif relative_volume < 0.5:
                        base_score -= 0.15  # Volume relatif très faible défavorable pour BUY
                        
                elif signal_side == "SELL":
                    if relative_volume >= 2.0:
                        base_score += 0.10  # Volume élevé peut indiquer capitulation
                    elif relative_volume >= 0.7:
                        base_score += 0.05  # Volume modéré OK pour SELL
                    # Pour SELL, volume faible peut être OK (distribution discrète)
                
            # Bonus trade intensity avec logique directionnelle
            if trade_intensity is not None:
                if signal_side == "BUY":
                    if trade_intensity >= self.high_intensity_threshold:
                        base_score += 0.05  # Haute intensité = FOMO potentiel, bonus modéré
                    elif trade_intensity >= 1.0:
                        base_score += 0.10  # Intensité modérée idéale pour BUY
                    elif trade_intensity >= 0.5:
                        base_score += 0.05  # Intensité normale acceptable
                        
                elif signal_side == "SELL":
                    if trade_intensity >= self.high_intensity_threshold:
                        base_score += 0.15  # Haute intensité = panique, bon pour SELL fort
                    elif trade_intensity >= 0.5:
                        base_score += 0.08  # Intensité normale OK pour SELL
                    
            # Bonus taille trades avec logique directionnelle
            if avg_trade_size is not None:
                if signal_side == "BUY":
                    if avg_trade_size >= self.large_trade_multiplier:
                        base_score += 0.15  # Gros trades = accumulation institutionnelle
                    elif avg_trade_size >= 1.0:
                        base_score += 0.08  # Taille normale favorable
                    elif avg_trade_size < self.min_avg_trade_multiplier:
                        base_score -= 0.10  # Petits trades = retail FOMO
                        
                elif signal_side == "SELL":
                    if avg_trade_size < self.min_avg_trade_multiplier:
                        base_score += 0.10  # Petits trades = distribution discrète
                    elif avg_trade_size <= 1.0:
                        base_score += 0.05  # Taille normale OK
                    elif avg_trade_size >= self.whale_trade_multiplier:
                        base_score += 0.08  # Whale dump peut créer opportunité
                
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
            volume_ratio = float(self.context.get('volume_ratio', 1.0)) if self.context.get('volume_ratio') is not None else 1.0
            quote_volume_ratio = float(self.context.get('quote_volume_ratio', 1.0)) if self.context.get('quote_volume_ratio') is not None else None
            relative_volume = float(self.context.get('relative_volume', 1.0)) if self.context.get('relative_volume') is not None else 1.0
            
            if is_valid:
                # Déterminer condition dominante
                if volume_ratio >= self.high_volume_ratio:
                    condition = f"volume élevé ({self._safe_format(volume_ratio, '.1f')}x)"
                elif quote_volume_ratio and self.ideal_quote_ratio_min <= quote_volume_ratio <= self.ideal_quote_ratio_max:
                    condition = f"ratio quote/base idéal ({self._safe_format(quote_volume_ratio, '.2f')})"
                elif relative_volume >= 1.5:
                    condition = f"volume relatif élevé ({self._safe_format(relative_volume, '.1f')}x)"
                else:
                    condition = f"ratios acceptables (vol: {self._safe_format(volume_ratio, '.1f')}x)"
                    
                return f"{self.name}: Validé - {condition} pour signal {signal_side}"
            else:
                if volume_ratio < self.min_volume_ratio:
                    return f"{self.name}: Rejeté - Volume ratio trop faible ({self._safe_format(volume_ratio, '.2f')})"
                elif quote_volume_ratio and (quote_volume_ratio < self.min_quote_volume_ratio or quote_volume_ratio > self.max_quote_volume_ratio):
                    return f"{self.name}: Rejeté - Ratio quote/base anormal ({self._safe_format(quote_volume_ratio, '.2f')})"
                elif relative_volume < self.min_volume_ratio:
                    return f"{self.name}: Rejeté - Volume relatif insuffisant ({self._safe_format(relative_volume, '.2f')})"
                elif volume_ratio >= self.extreme_volume_ratio:
                    return f"{self.name}: Rejeté - Volume ratio extrême suspect ({self._safe_format(volume_ratio, '.2f')})"
                else:
                    return f"{self.name}: Rejeté - Ratios volume inadéquats"
                    
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données de ratio volume requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Vérifier qu'au moins un indicateur de ratio volume est présent
        has_volume_ratio = 'volume_ratio' in self.context and self.context['volume_ratio'] is not None
        has_relative_volume = 'relative_volume' in self.context and self.context['relative_volume'] is not None
        has_quote_ratio = 'quote_volume_ratio' in self.context and self.context['quote_volume_ratio'] is not None
        has_avg_volume = 'avg_volume_20' in self.context and self.context['avg_volume_20'] is not None
        has_current_volume = 'volume' in self.data and self.data['volume'] is not None
        
        if not (has_volume_ratio or has_relative_volume or has_quote_ratio or (has_avg_volume and has_current_volume)):
            logger.warning(f"{self.name}: Aucun indicateur de ratio volume disponible pour {self.symbol}")
            return False
            
        return True
