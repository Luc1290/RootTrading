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
        
        # Paramètres ATR (en percentiles)
        self.min_atr_percentile = 20.0     # ATR minimum (éviter marchés dormants)
        self.max_atr_percentile = 90.0     # ATR maximum (éviter volatilité extrême)
        self.optimal_atr_min = 30.0        # Zone optimale minimum
        self.optimal_atr_max = 80.0        # Zone optimale maximum
        
        # Seuils absolus ATR
        self.min_atr_absolute = 0.001      # 0.1% minimum de volatilité
        self.extreme_atr_threshold = 0.05  # 5% volatilité extrême
        
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
                atr_14 = float(self.context.get('atr_14', 0)) if self.context.get('atr_14') is not None else None
                atr_percentile = float(self.context.get('atr_percentile', 50)) if self.context.get('atr_percentile') is not None else None
                volatility_regime = self.context.get('volatility_regime')
                natr = float(self.context.get('natr', 0)) if self.context.get('natr') is not None else None
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
                logger.debug(f"{self.name}: ATR trop faible ({atr_14:.4f}) pour {self.symbol} - marché dormant")
                return False
                
            # 2. Vérification volatilité extrême
            if atr_14 > self.extreme_atr_threshold:
                logger.debug(f"{self.name}: ATR extrême ({atr_14:.4f}) pour {self.symbol} - risque trop élevé")
                # Ne pas rejeter automatiquement, mais appliquer des critères plus stricts
                if signal_confidence < 0.7:
                    logger.debug(f"{self.name}: Signal confidence insuffisante ({signal_confidence:.2f}) pour volatilité extrême")
                    return False
                    
            # 3. Vérification percentile ATR
            if atr_percentile is not None:
                if atr_percentile < self.min_atr_percentile:
                    logger.debug(f"{self.name}: ATR percentile trop bas ({atr_percentile:.1f}) pour {self.symbol}")
                    return False
                    
                if atr_percentile > self.max_atr_percentile:
                    logger.debug(f"{self.name}: ATR percentile trop élevé ({atr_percentile:.1f}) pour {self.symbol}")
                    # Critères plus stricts pour volatilité très élevée
                    if signal_confidence < 0.8:
                        return False
                        
            # 4. Validation selon le régime de volatilité
            if volatility_regime:
                if volatility_regime == "low":
                    # Volatilité faible - accepter seulement signaux très confiants
                    if signal_confidence < 0.6:
                        logger.debug(f"{self.name}: Régime volatilité faible mais confidence insuffisante ({signal_confidence:.2f}) pour {self.symbol}")
                        return False
                        
                elif volatility_regime == "extreme":
                    # Volatilité extrême - très sélectif
                    if signal_confidence < 0.8:
                        logger.debug(f"{self.name}: Régime volatilité extrême mais confidence insuffisante ({signal_confidence:.2f}) pour {self.symbol}")
                        return False
                        
                elif volatility_regime == "high":
                    # Volatilité élevée - moyennement sélectif
                    if signal_confidence < 0.5:
                        logger.debug(f"{self.name}: Régime volatilité élevée mais confidence très faible ({signal_confidence:.2f}) pour {self.symbol}")
                        return False
                        
            # 5. Validation spécifique selon la stratégie
            if self._is_breakout_strategy(signal_strategy):
                # Stratégies de breakout nécessitent de la volatilité
                if atr_percentile is not None and atr_percentile < 40.0:
                    logger.debug(f"{self.name}: Stratégie breakout mais ATR percentile faible ({atr_percentile:.1f}) pour {self.symbol}")
                    return False
                    
            elif self._is_meanreversion_strategy(signal_strategy):
                # Stratégies de mean reversion préfèrent volatilité modérée
                if atr_percentile is not None and atr_percentile > 85.0:
                    logger.debug(f"{self.name}: Stratégie mean reversion mais ATR percentile très élevé ({atr_percentile:.1f}) pour {self.symbol}")
                    return False
                    
            # 6. Vérification NATR (Normalized ATR) si disponible
            if natr is not None:
                if natr < 0.5:  # NATR très faible
                    logger.debug(f"{self.name}: NATR très faible ({natr:.2f}) pour {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                elif natr > 8.0:  # NATR très élevé
                    logger.debug(f"{self.name}: NATR très élevé ({natr:.2f}) pour {self.symbol}")
                    if signal_confidence < 0.8:
                        return False
                        
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - ATR: {atr_14:.4f}, "
                        f"Percentile: {atr_percentile:.1f if atr_percentile is not None else 'N/A'}, "
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
            
            base_score = 0.5  # Score de base si validé
            
            # Bonus selon percentile ATR (zone optimale)
            if self.optimal_atr_min <= atr_percentile <= self.optimal_atr_max:
                # Zone optimale de volatilité
                optimal_center = (self.optimal_atr_min + self.optimal_atr_max) / 2
                distance_from_center = abs(atr_percentile - optimal_center)
                max_distance = (self.optimal_atr_max - self.optimal_atr_min) / 2
                
                # Score plus élevé proche du centre de la zone optimale
                optimal_bonus = self.volatility_bonus * (1 - distance_from_center / max_distance)
                base_score += optimal_bonus
                
            elif atr_percentile > self.optimal_atr_max:
                # Volatilité élevée - bonus réduit
                base_score += 0.05
            elif atr_percentile < self.optimal_atr_min:
                # Volatilité faible - légère pénalité
                base_score -= 0.05
                
            # Ajustement selon régime de volatilité
            if volatility_regime == "normal":
                base_score += 0.1  # Régime normal favorable
            elif volatility_regime == "expanding":
                base_score += 0.15  # Volatilité en expansion favorable
            elif volatility_regime == "contracting":
                base_score += 0.05  # Volatilité en contraction moins favorable
            elif volatility_regime == "extreme":
                base_score += self.extreme_penalty  # Pénalité volatilité extrême
                
            # Bonus cohérence stratégie/volatilité
            signal_strategy = signal.get('strategy', '')
            if self._is_breakout_strategy(signal_strategy) and atr_percentile >= 50:
                base_score += 0.1  # Breakout avec volatilité suffisante
            elif self._is_meanreversion_strategy(signal_strategy) and 30 <= atr_percentile <= 70:
                base_score += 0.1  # Mean reversion avec volatilité modérée
                
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
            atr_14 = float(self.context.get('atr_14', 0)) if self.context.get('atr_14') is not None else 0
            atr_percentile = float(self.context.get('atr_percentile', 50)) if self.context.get('atr_percentile') is not None else None
            volatility_regime = self.context.get('volatility_regime', 'N/A')
            signal_strategy = signal.get('strategy', 'N/A')
            
            if is_valid:
                regime_desc = f"régime {volatility_regime}" if volatility_regime != 'N/A' else "régime normal"
                percentile_desc = f"percentile {atr_percentile:.1f}" if atr_percentile is not None else "N/A"
                
                reason = f"Volatilité acceptable (ATR: {atr_14:.4f}, {percentile_desc}, {regime_desc})"
                
                if self.optimal_atr_min <= (atr_percentile or 50) <= self.optimal_atr_max:
                    reason += " - zone optimale"
                    
                return f"{self.name}: Validé - {reason} pour {signal_strategy}"
            else:
                if atr_14 < self.min_atr_absolute:
                    return f"{self.name}: Rejeté - ATR trop faible ({atr_14:.4f}) - marché dormant"
                elif atr_14 > self.extreme_atr_threshold:
                    return f"{self.name}: Rejeté - ATR extrême ({atr_14:.4f}) - risque élevé"
                elif atr_percentile and atr_percentile < self.min_atr_percentile:
                    return f"{self.name}: Rejeté - ATR percentile trop bas ({atr_percentile:.1f})"
                elif atr_percentile and atr_percentile > self.max_atr_percentile:
                    return f"{self.name}: Rejeté - ATR percentile trop élevé ({atr_percentile:.1f})"
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
