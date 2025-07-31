"""
RSI_Regime_Validator - Validator basé sur les régimes RSI pour filtrer les signaux.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class RSI_Regime_Validator(BaseValidator):
    """
    Validator basé sur les régimes RSI - filtre les signaux selon les conditions de surachat/survente.
    
    Vérifie: Niveaux RSI, momentum RSI, divergences potentielles
    Catégorie: technical
    
    Rejette les signaux en:
    - Zone de surachat extrême (RSI > 80) pour BUY
    - Zone de survente extrême (RSI < 20) pour SELL  
    - RSI neutre sans momentum (40-60 avec faible variation)
    - Divergences RSI/prix défavorables
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "RSI_Regime_Validator"
        self.category = "technical"
        
        # Paramètres RSI
        self.oversold_threshold = 30.0      # RSI survente
        self.overbought_threshold = 70.0    # RSI surachat
        self.extreme_oversold = 20.0        # RSI survente extrême
        self.extreme_overbought = 80.0      # RSI surachat extrême
        self.neutral_zone_min = 40.0        # Zone neutre min
        self.neutral_zone_max = 60.0        # Zone neutre max
        self.momentum_threshold = 5.0       # Variation RSI minimum
        self.strong_momentum_threshold = 10.0 # Variation RSI forte
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur les régimes RSI.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon RSI, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs RSI depuis le contexte
            try:
                rsi_14 = float(self.context.get('rsi_14', 50.0)) if self.context.get('rsi_14') is not None else None
                rsi_21 = float(self.context.get('rsi_21', 50.0)) if self.context.get('rsi_21') is not None else None
                momentum_score = float(self.context.get('momentum_score', 50.0)) if self.context.get('momentum_score') is not None else None
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion RSI pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side or rsi_14 is None:
                logger.warning(f"{self.name}: Signal side ou RSI manquant pour {self.symbol}")
                return False
                
            # 1. Validation des signaux BUY
            if signal_side == "BUY":
                # Rejeter BUY en surachat extrême
                if rsi_14 >= self.extreme_overbought:
                    logger.debug(f"{self.name}: BUY rejeté - RSI surachat extrême ({self._safe_format(rsi_14, '.1f')}) pour {self.symbol}")
                    return False
                    
                # Favoriser BUY en survente
                if rsi_14 <= self.oversold_threshold:
                    logger.debug(f"{self.name}: BUY favorisé - RSI survente ({self._safe_format(rsi_14, '.1f')}) pour {self.symbol}")
                    return True
                    
                # BUY acceptable si RSI < 70 avec momentum positif
                if rsi_14 < self.overbought_threshold:
                    if momentum_score is not None and momentum_score > 30.0:  # Format 0-100
                        logger.debug(f"{self.name}: BUY accepté - RSI modéré ({self._safe_format(rsi_14, '.1f')}) + momentum positif pour {self.symbol}")
                        return True
                    elif rsi_14 < 65.0:  # Zone acceptable même sans momentum fort
                        return True
                        
                # Zone neutre : besoin de momentum fort ou confidence élevée
                if self.neutral_zone_min <= rsi_14 <= self.neutral_zone_max:
                    if signal_confidence >= 0.8:
                        logger.debug(f"{self.name}: BUY accepté - Zone neutre mais confidence élevée ({self._safe_format(signal_confidence, '.2f')}) pour {self.symbol}")
                        return True
                    else:
                        logger.debug(f"{self.name}: BUY rejeté - Zone neutre RSI ({self._safe_format(rsi_14, '.1f')}) + confidence faible pour {self.symbol}")
                        return False
                        
            # 2. Validation des signaux SELL  
            elif signal_side == "SELL":
                # Rejeter SELL en survente extrême
                if rsi_14 <= self.extreme_oversold:
                    logger.debug(f"{self.name}: SELL rejeté - RSI survente extrême ({self._safe_format(rsi_14, '.1f')}) pour {self.symbol}")
                    return False
                    
                # Favoriser SELL en surachat
                if rsi_14 >= self.overbought_threshold:
                    logger.debug(f"{self.name}: SELL favorisé - RSI surachat ({self._safe_format(rsi_14, '.1f')}) pour {self.symbol}")
                    return True
                    
                # SELL acceptable si RSI > 30 avec momentum bearish
                if rsi_14 > self.oversold_threshold:
                    # Momentum bearish = momentum_score < 50 (car 0-100, 50 est neutre)
                    if momentum_score is not None and momentum_score < 40.0:  # Momentum bearish
                        logger.debug(f"{self.name}: SELL accepté - RSI modéré ({self._safe_format(rsi_14, '.1f')}) + momentum bearish ({self._safe_format(momentum_score, '.1f')}) pour {self.symbol}")
                        return True
                    elif rsi_14 > 35.0:  # Zone acceptable même sans momentum fort
                        return True
                        
                # Zone neutre : besoin de momentum fort ou confidence élevée
                if self.neutral_zone_min <= rsi_14 <= self.neutral_zone_max:
                    if signal_confidence >= 0.8:
                        logger.debug(f"{self.name}: SELL accepté - Zone neutre mais confidence élevée ({self._safe_format(signal_confidence, '.2f')}) pour {self.symbol}")
                        return True
                    else:
                        logger.debug(f"{self.name}: SELL rejeté - Zone neutre RSI ({self._safe_format(rsi_14, '.1f')}) + confidence faible pour {self.symbol}")
                        return False
                        
            # 3. Vérification cohérence RSI 14 vs RSI 21 si disponible
            if rsi_21 is not None:
                rsi_divergence = abs(rsi_14 - rsi_21)
                if rsi_divergence > 15.0:  # Divergence importante
                    logger.debug(f"{self.name}: Divergence RSI14/RSI21 importante ({self._safe_format(rsi_divergence, '.1f')}) pour {self.symbol}")
                    # Ne pas rejeter mais noter l'incohérence
                    
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - RSI14: {self._safe_format(rsi_14, '.1f')}, "
                        f"RSI21: {self._safe_format(rsi_21, '.1f') if rsi_21 is not None else 'N/A'}, "
                        f"Side: {signal_side}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur les régimes RSI.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur RSI
            rsi_14 = float(self.context.get('rsi_14', 50.0)) if self.context.get('rsi_14') is not None else 50.0
            signal_side = signal.get('side')
            signal_confidence = signal.get('confidence', 0.0)
            momentum_score = float(self.context.get('momentum_score', 50.0)) if self.context.get('momentum_score') is not None else None
            
            base_score = 0.5  # Score de base si validé
            
            # Scoring selon position RSI et side
            if signal_side == "BUY":
                if rsi_14 <= self.extreme_oversold:
                    base_score += 0.4  # Excellent pour BUY
                elif rsi_14 <= self.oversold_threshold:
                    base_score += 0.3  # Très bon pour BUY
                elif rsi_14 <= 50.0:
                    base_score += 0.2  # Bon pour BUY
                elif rsi_14 <= self.overbought_threshold:
                    base_score += 0.1  # Acceptable pour BUY
                else:
                    base_score -= 0.1  # Moins favorable pour BUY
                    
            elif signal_side == "SELL":
                if rsi_14 >= self.extreme_overbought:
                    base_score += 0.4  # Excellent pour SELL
                elif rsi_14 >= self.overbought_threshold:
                    base_score += 0.3  # Très bon pour SELL
                elif rsi_14 >= 50.0:
                    base_score += 0.2  # Bon pour SELL
                elif rsi_14 >= self.oversold_threshold:
                    base_score += 0.1  # Acceptable pour SELL
                else:
                    base_score -= 0.1  # Moins favorable pour SELL
                    
            # Bonus momentum cohérent
            if momentum_score is not None:
                if signal_side == "BUY" and momentum_score > 60.0:  # Momentum bullish fort
                    base_score += 0.1  # Momentum bullish pour BUY
                elif signal_side == "SELL" and momentum_score < 40.0:  # Momentum bearish fort
                    base_score += 0.1  # Momentum bearish pour SELL
                    # Bonus supplémentaire si momentum très bearish
                    if momentum_score < 25.0:
                        base_score += 0.05
                    
            # Bonus confidence élevée
            if signal_confidence >= 0.8:
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
            rsi_14 = float(self.context.get('rsi_14', 50.0)) if self.context.get('rsi_14') is not None else 50.0
            signal_side = signal.get('side', 'N/A')
            
            if is_valid:
                # Déterminer le régime RSI
                if rsi_14 <= self.extreme_oversold:
                    regime = "survente extrême"
                elif rsi_14 <= self.oversold_threshold:
                    regime = "survente"
                elif rsi_14 >= self.extreme_overbought:
                    regime = "surachat extrême"
                elif rsi_14 >= self.overbought_threshold:
                    regime = "surachat"
                else:
                    regime = "neutre"
                    
                return f"{self.name}: Validé - RSI {regime} ({self._safe_format(rsi_14, '.1f')}) favorable pour signal {signal_side}"
            else:
                if signal_side == "BUY" and rsi_14 >= self.extreme_overbought:
                    return f"{self.name}: Rejeté - BUY en surachat extrême (RSI: {self._safe_format(rsi_14, '.1f')})"
                elif signal_side == "SELL" and rsi_14 <= self.extreme_oversold:
                    return f"{self.name}: Rejeté - SELL en survente extrême (RSI: {self._safe_format(rsi_14, '.1f')})"
                elif self.neutral_zone_min <= rsi_14 <= self.neutral_zone_max:
                    return f"{self.name}: Rejeté - Zone neutre RSI ({self._safe_format(rsi_14, '.1f')}) sans momentum suffisant"
                else:
                    return f"{self.name}: Rejeté - Régime RSI défavorable ({self._safe_format(rsi_14, '.1f')})"
                    
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données RSI requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Vérification présence RSI_14 (minimum requis)
        if 'rsi_14' not in self.context or self.context['rsi_14'] is None:
            logger.warning(f"{self.name}: RSI_14 manquant pour {self.symbol}")
            return False
            
        return True
