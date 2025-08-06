"""
ATR_Volatility_Validator - Validator basé sur la volatilité ATR.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class ATR_Volatility_Validator(BaseValidator):
    """
    Validator pour la volatilité ATR - filtre les signaux selon les conditions de volatilité.
    
    Vérifie: Niveau ATR, régime de volatilité, contexte de marché
    Catégorie: volatility
    
    Rejette les signaux en:
    - Volatilité trop faible (pas de mouvement attendu)
    - Volatilité extrême (risque trop élevé)
    - Régime de volatilité inadapté au signal
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "ATR_Volatility_Validator"
        self.category = "volatility"
        
        # Paramètres ATR DURCIS (en percentiles)
        self.min_atr_percentile = 30.0     # ATR minimum relevé (30% au lieu de 20%)
        self.max_atr_percentile = 85.0     # ATR maximum réduit (85% au lieu de 90%)
        self.optimal_atr_min = 40.0        # Zone optimale relevée (40% au lieu de 30%)
        self.optimal_atr_max = 75.0        # Zone optimale réduite (75% au lieu de 80%)
        
        # Seuils absolus ATR DURCIS
        self.min_atr_absolute = 0.002      # 0.2% minimum (doublé)
        self.extreme_atr_threshold = 0.04  # 4% volatilité extrême (réduit)
        
        # Bonus/malus
        self.volatility_bonus = 0.2        # Bonus zone optimale
        self.extreme_penalty = -0.3        # Pénalité volatilité extrême
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur les conditions de volatilité ATR.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon ATR, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs ATR depuis le contexte
            try:
                # Log temporaire pour debug
                logger.debug(f"{self.name}: Contexte reçu pour {self.symbol}: {list(self.context.keys())[:10]}...")
                
                atr_14 = float(self.context.get('atr_14', 0)) if self.context.get('atr_14') is not None else None
                atr_percentile = float(self.context.get('atr_percentile', 50)) if self.context.get('atr_percentile') is not None else None
                volatility_regime = self.context.get('volatility_regime')
                natr = float(self.context.get('natr', 0)) if self.context.get('natr') is not None else None
                
                logger.debug(f"{self.name}: ATR values - atr_14: {atr_14}, atr_percentile: {atr_percentile}, regime: {volatility_regime}")
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion ATR pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', 'Unknown')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side or atr_14 is None:
                logger.warning(f"{self.name}: Signal side ou ATR manquant pour {self.symbol}")
                return False
                
            # 1. Vérification ATR minimum absolu
            if atr_14 < self.min_atr_absolute:
                logger.debug(f"{self.name}: ATR trop faible ({self._safe_format(atr_14, '.4f')}) pour {self.symbol} - marché dormant")
                return False
                
            # 2. Vérification volatilité extrême
            if atr_14 > self.extreme_atr_threshold:
                logger.debug(f"{self.name}: ATR extrême ({self._safe_format(atr_14, '.4f')}) pour {self.symbol} - risque trop élevé")
                # Critères ALIGNÉS avec signal_aggregator (seuil critical 75%)
                if signal_confidence < 0.75:  # Aligné avec nouveaux seuils
                    logger.debug(f"{self.name}: Signal confidence insuffisante ({self._safe_format(signal_confidence, '.2f')}) pour volatilité extrême")
                    return False
                    
            # 3. Vérification percentile ATR
            if atr_percentile is not None:
                if atr_percentile < self.min_atr_percentile:
                    logger.debug(f"{self.name}: ATR percentile trop bas ({self._safe_format(atr_percentile, '.1f')}) pour {self.symbol}")
                    return False
                    
                if atr_percentile > self.max_atr_percentile:
                    logger.debug(f"{self.name}: ATR percentile trop élevé ({self._safe_format(atr_percentile, '.1f')}) pour {self.symbol}")
                    # Critères ALIGNÉS - volatilité très élevée
                    if signal_confidence < 0.80:  # Maintenu à 80% (très strict)
                        return False
                        
            # 4. Validation selon le régime de volatilité
            if volatility_regime:
                if volatility_regime == "low":
                    # Volatilité faible - ALIGNÉ avec seuils
                    if signal_confidence < 0.65:  # Aligné avec seuil important (65%)
                        logger.debug(f"{self.name}: Régime volatilité faible mais confidence insuffisante ({self._safe_format(signal_confidence, '.2f')}) pour {self.symbol}")
                        return False
                        
                elif volatility_regime == "extreme":
                    # Volatilité extrême - TRÈS sélectif
                    if signal_confidence < 0.80:  # Maintenu strict
                        logger.debug(f"{self.name}: Régime volatilité extrême mais confidence insuffisante ({self._safe_format(signal_confidence, '.2f')}) pour {self.symbol}")
                        return False
                        
                elif volatility_regime == "high":
                    # Volatilité élevée - ALIGNÉ avec seuils
                    if signal_confidence < 0.55:  # Aligné avec seuil standard (55%)
                        logger.debug(f"{self.name}: Régime volatilité élevée mais confidence très faible ({self._safe_format(signal_confidence, '.2f')}) pour {self.symbol}")
                        return False
                        
            # 5. Validation spécifique selon la stratégie
            if self._is_breakout_strategy(signal_strategy):
                # Stratégies de breakout nécessitent de la volatilité
                if atr_percentile is not None and atr_percentile < 40.0:
                    logger.debug(f"{self.name}: Stratégie breakout mais ATR percentile faible ({self._safe_format(atr_percentile, '.1f')}) pour {self.symbol}")
                    return False
                    
            elif self._is_meanreversion_strategy(signal_strategy):
                # Stratégies de mean reversion préfèrent volatilité modérée
                if atr_percentile is not None and atr_percentile > 85.0:
                    logger.debug(f"{self.name}: Stratégie mean reversion mais ATR percentile très élevé ({self._safe_format(atr_percentile, '.1f')}) pour {self.symbol}")
                    return False
                    
            # 6. Vérification NATR (Normalized ATR) si disponible
            if natr is not None:
                if natr < 0.5:  # NATR très faible
                    logger.debug(f"{self.name}: NATR très faible ({self._safe_format(natr, '.2f')}) pour {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                elif natr > 8.0:  # NATR très élevé
                    logger.debug(f"{self.name}: NATR très élevé ({self._safe_format(natr, '.2f')}) pour {self.symbol}")
                    if signal_confidence < 0.8:
                        return False
                        
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - ATR: {self._safe_format(atr_14, '.4f')}, "
                        f"Percentile: {self._safe_format(atr_percentile, '.1f') if atr_percentile is not None else 'N/A'}, "
                        f"Régime: {volatility_regime or 'N/A'}, "
                        f"Strategy: {signal_strategy}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def _is_breakout_strategy(self, strategy_name: str) -> bool:
        """Détermine si la stratégie est de type breakout."""
        breakout_keywords = ['breakout', 'break', 'donchian', 'channel', 'resistance', 'support']
        strategy_lower = strategy_name.lower()
        return any(keyword in strategy_lower for keyword in breakout_keywords)
        
    def _is_meanreversion_strategy(self, strategy_name: str) -> bool:
        """Détermine si la stratégie est de type mean reversion."""
        meanrev_keywords = ['bollinger', 'touch', 'reversal', 'rsi', 'oversold', 'overbought', 'rebound']
        strategy_lower = strategy_name.lower()
        return any(keyword in strategy_lower for keyword in meanrev_keywords)
        
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur la volatilité ATR.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur ATR
            atr_14 = float(self.context.get('atr_14', 0)) if self.context.get('atr_14') is not None else 0
            atr_percentile = float(self.context.get('atr_percentile', 50)) if self.context.get('atr_percentile') is not None else 50
            volatility_regime = self.context.get('volatility_regime')
            
            base_score = 0.3  # Score de base réduit (0.3 au lieu de 0.5)
            
            # CORRECTION: Bonus selon percentile ATR avec logique directionnelle
            signal_side = signal.get('side')
            
            if self.optimal_atr_min <= atr_percentile <= self.optimal_atr_max:
                # Zone optimale de volatilité - ajustement selon direction
                optimal_center = (self.optimal_atr_min + self.optimal_atr_max) / 2
                distance_from_center = abs(atr_percentile - optimal_center)
                max_distance = (self.optimal_atr_max - self.optimal_atr_min) / 2
                
                # Score de base pour zone optimale
                optimal_bonus = self.volatility_bonus * (1 - distance_from_center / max_distance)
                
                # Ajustement directionnel dans la zone optimale - RÉDUIT
                if signal_side == "BUY":
                    # BUY préfère le bas de la zone optimale (volatilité en expansion)
                    if atr_percentile < optimal_center:
                        base_score += optimal_bonus * 0.9  # Réduit de 1.2 à 0.9
                    else:
                        base_score += optimal_bonus * 0.6  # Réduit de 0.8 à 0.6
                elif signal_side == "SELL":
                    # SELL préfère le haut de la zone optimale (volatilité élevée)
                    if atr_percentile > optimal_center:
                        base_score += optimal_bonus * 0.9  # Réduit de 1.2 à 0.9
                    else:
                        base_score += optimal_bonus * 0.6  # Réduit de 0.8 à 0.6
                else:
                    base_score += optimal_bonus * 0.7  # Réduit pour pas de direction
                
            elif atr_percentile > self.optimal_atr_max:
                # Volatilité élevée - ajustement directionnel RÉDUIT
                if signal_side == "SELL":
                    base_score += 0.06  # Réduit de 0.10 à 0.06
                elif signal_side == "BUY":
                    base_score += 0.01  # Réduit de 0.02 à 0.01
                else:
                    base_score += 0.03  # Réduit de 0.05 à 0.03
                    
            elif atr_percentile < self.optimal_atr_min:
                # Volatilité faible - ajustement directionnel RÉDUIT
                if signal_side == "BUY":
                    base_score -= 0.02  # Réduit la pénalité de 0.03 à 0.02
                elif signal_side == "SELL":
                    base_score -= 0.05  # Réduit la pénalité de 0.08 à 0.05
                else:
                    base_score -= 0.03  # Réduit la pénalité de 0.05 à 0.03
                
            # Ajustement selon régime de volatilité (selon schéma: low/normal/high/extreme) - RÉDUIT
            if volatility_regime == "normal":
                base_score += 0.06  # Réduit de 0.1 à 0.06
            elif volatility_regime == "low":
                # Volatilité faible - signaux moins favorables RÉDUIT
                if signal_side == "BUY":
                    base_score += 0.02  # Réduit de 0.03 à 0.02
                elif signal_side == "SELL":
                    base_score -= 0.03  # Réduit la pénalité de 0.05 à 0.03
                else:
                    base_score += 0.01  # Réduit de 0.02 à 0.01
            elif volatility_regime == "high":
                # Volatilité élevée - signaux directionnels favorisés RÉDUIT
                if signal_side in ["BUY", "SELL"]:
                    base_score += 0.08  # Réduit de 0.12 à 0.08
                else:
                    base_score += 0.05  # Réduit de 0.08 à 0.05
            elif volatility_regime == "extreme":
                # Volatilité extrême = risque mais opportunités RÉDUIT
                if signal_side == "BUY":
                    # BUY en volatilité extrême = risqué mais potentiel si confidence élevée
                    if signal.get('confidence', 0) >= 0.8:
                        base_score += 0.03  # Réduit de 0.05 à 0.03
                    else:
                        base_score += self.extreme_penalty * 0.7  # Réduire la pénalité
                elif signal_side == "SELL":
                    # SELL en volatilité extrême = souvent panic selling
                    if signal.get('confidence', 0) >= 0.8:
                        base_score += 0.06  # Réduit de 0.10 à 0.06
                    else:
                        base_score += self.extreme_penalty * 0.7  # Réduire la pénalité
                
            # CORRECTION: Bonus cohérence stratégie/volatilité avec logique directionnelle
            signal_strategy = signal.get('strategy', '')
            
            # Logique directionnelle pour ATR selon le type de signal - RÉDUIT
            if signal_side == "BUY":
                # BUY : Favoriser volatilité croissante ou normale RÉDUIT
                if atr_percentile >= 50 and atr_percentile <= 80:
                    base_score += 0.07  # Réduit de 0.12 à 0.07
                elif atr_percentile > 80:
                    # Très haute volatilité pour BUY
                    if self._is_breakout_strategy(signal_strategy):
                        base_score += 0.09  # Réduit de 0.15 à 0.09
                    else:
                        base_score += 0.03  # Réduit de 0.05 à 0.03
                elif atr_percentile < 30:
                    # Faible volatilité pour BUY
                    if self._is_meanreversion_strategy(signal_strategy):
                        base_score += 0.05  # Réduit de 0.08 à 0.05
                    else:
                        base_score -= 0.03  # Réduit la pénalité de 0.05 à 0.03
                        
            elif signal_side == "SELL":
                # SELL : Favoriser volatilité élevée (panic) ou expansion RÉDUIT
                if atr_percentile >= 70:
                    base_score += 0.09  # Réduit de 0.15 à 0.09
                elif atr_percentile >= 50:
                    base_score += 0.06  # Réduit de 0.10 à 0.06
                elif atr_percentile < 30:
                    # Faible volatilité pour SELL
                    if self._is_meanreversion_strategy(signal_strategy):
                        base_score += 0.05  # Réduit de 0.08 à 0.05
                    else:
                        base_score -= 0.05  # Réduit la pénalité de 0.08 à 0.05
                        
            # Bonus spécifiques stratégie RÉDUIT
            if self._is_breakout_strategy(signal_strategy):
                if atr_percentile >= 50:
                    base_score += 0.03  # Réduit de 0.05 à 0.03
            elif self._is_meanreversion_strategy(signal_strategy):
                if 30 <= atr_percentile <= 70:
                    base_score += 0.03  # Réduit de 0.05 à 0.03
                
            # NOUVEAU: Cap final pour éviter les scores excessifs
            final_score = max(0.0, min(0.85, base_score))  # Limité à 85% maximum
            return final_score
            
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
            atr_14 = float(self.context.get('atr_14', 0)) if self.context.get('atr_14') is not None else 0
            atr_percentile = float(self.context.get('atr_percentile', 50)) if self.context.get('atr_percentile') is not None else None
            volatility_regime = self.context.get('volatility_regime', 'N/A')
            signal_strategy = signal.get('strategy', 'N/A')
            
            if is_valid:
                regime_desc = f"régime {volatility_regime}" if volatility_regime != 'N/A' else "régime normal"
                percentile_desc = f"percentile {self._safe_format(atr_percentile, '.1f')}" if atr_percentile is not None else "N/A"
                
                reason = f"Volatilité acceptable (ATR: {self._safe_format(atr_14, '.4f')}, {percentile_desc}, {regime_desc})"
                
                if self.optimal_atr_min <= (atr_percentile or 50) <= self.optimal_atr_max:
                    reason += " - zone optimale"
                    
                return f"{self.name}: Validé - {reason} pour {signal_strategy}"
            else:
                if atr_14 < self.min_atr_absolute:
                    return f"{self.name}: Rejeté - ATR trop faible ({self._safe_format(atr_14, '.4f')}) - marché dormant"
                elif atr_14 > self.extreme_atr_threshold:
                    return f"{self.name}: Rejeté - ATR extrême ({self._safe_format(atr_14, '.4f')}) - risque élevé"
                elif atr_percentile and atr_percentile < self.min_atr_percentile:
                    return f"{self.name}: Rejeté - ATR percentile trop bas ({self._safe_format(atr_percentile, '.1f')})"
                elif atr_percentile and atr_percentile > self.max_atr_percentile:
                    return f"{self.name}: Rejeté - ATR percentile trop élevé ({self._safe_format(atr_percentile, '.1f')})"
                elif volatility_regime == "extreme":
                    return f"{self.name}: Rejeté - Volatilité extrême + confidence insuffisante"
                    
                return f"{self.name}: Rejeté - Critères volatilité non respectés"
                
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données ATR requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Vérification présence ATR
        if 'atr_14' not in self.context or self.context['atr_14'] is None:
            logger.warning(f"{self.name}: ATR_14 manquant pour {self.symbol}")
            return False
            
        return True
