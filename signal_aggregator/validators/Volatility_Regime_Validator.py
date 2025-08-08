"""
Volatility_Regime_Validator - Validator basé sur les régimes de volatilité pour adapter la validation aux conditions de marché.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class Volatility_Regime_Validator(BaseValidator):
    """
    Validator basé sur les régimes de volatilité - adapte la validation selon les conditions de volatilité du marché.
    
    Vérifie: Régime volatilité, ATR percentile, Bollinger Bands expansion, cohérence signal/volatilité
    Catégorie: volatility
    
    Rejette les signaux en:
    - Volatilité extrême sans confidence adaptée
    - Signaux faibles en période de haute volatilité
    - Breakouts en période de compression excessive
    - Incohérence entre signal et expansion/contraction
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "Volatility_Regime_Validator"
        self.category = "volatility"
        
        # Paramètres ATR
        self.low_atr_percentile = 20.0        # ATR faible (< 20e percentile)
        self.high_atr_percentile = 80.0       # ATR élevé (> 80e percentile)
        self.extreme_atr_percentile = 95.0    # ATR extrême (> 95e percentile)
        
        # Paramètres de confidence selon volatilité
        self.high_vol_min_confidence = 0.75   # Confidence min en haute volatilité
        self.extreme_vol_min_confidence = 0.85 # Confidence min en volatilité extrême
        self.low_vol_min_confidence = 0.5     # Confidence min en basse volatilité
        
        # Paramètres Bollinger Bands
        self.bb_squeeze_threshold = 0.1       # Seuil compression BB
        self.bb_expansion_threshold = 0.3     # Seuil expansion BB
        self.bb_extreme_expansion = 0.5       # Expansion extrême BB
        
        # Paramètres de régime
        self.regime_stability_threshold = 0.7 # Stabilité régime minimum
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur les régimes de volatilité.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon la volatilité, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs de volatilité depuis le contexte
            try:
                volatility_regime = self.context.get('volatility_regime')  # 'low', 'normal', 'high', 'extreme'
                atr_14 = float(self.context.get('atr_14', 0)) if self.context.get('atr_14') is not None else None
                atr_percentile = float(self.context.get('atr_percentile', 50)) if self.context.get('atr_percentile') is not None else 50
                natr = float(self.context.get('natr', 0)) if self.context.get('natr') is not None else None
                
                # Bollinger Bands
                bb_width = float(self.context.get('bb_width', 0)) if self.context.get('bb_width') is not None else None
                bb_squeeze = self.context.get('bb_squeeze', False)
                bb_expansion = self.context.get('bb_expansion', False)
                bb_position = float(self.context.get('bb_position', 0.5)) if self.context.get('bb_position') is not None else None
                
                # Contexte additionnel
                market_regime = self.context.get('market_regime')
                regime_strength_raw = self.context.get('regime_strength')
                regime_strength = self._convert_regime_strength_to_score(str(regime_strength_raw)) if regime_strength_raw is not None else None
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion volatilité pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_confidence = signal.get('confidence', 0.0)
            signal_strength = signal.get('strength', 'moderate')
            
            if not signal_side:
                logger.warning(f"{self.name}: Signal side manquant pour {self.symbol}")
                return False
                
            # 1. Validation selon régime de volatilité général
            if volatility_regime:
                if volatility_regime == "extreme":
                    # Volatilité extrême - très strict
                    if signal_confidence < self.extreme_vol_min_confidence:
                        logger.debug(f"{self.name}: Volatilité extrême nécessite confidence ≥ {self.extreme_vol_min_confidence} (actuel: {self._safe_format(signal_confidence, '.2f')}) pour {self.symbol}")
                        return False
                elif volatility_regime == "high":
                    # Haute volatilité - strict
                    if signal_confidence < self.high_vol_min_confidence:
                        logger.debug(f"{self.name}: Haute volatilité nécessite confidence ≥ {self.high_vol_min_confidence} (actuel: {self._safe_format(signal_confidence, '.2f')}) pour {self.symbol}")
                        return False
                elif volatility_regime in ["low", "normal"]:
                    # Basse/Normale volatilité - plus permissif
                    if signal_confidence < self.low_vol_min_confidence:
                        logger.debug(f"{self.name}: Volatilité {volatility_regime} nécessite confidence ≥ {self.low_vol_min_confidence} (actuel: {self._safe_format(signal_confidence, '.2f')}) pour {self.symbol}")
                        return False
                        
                    # Signaux faibles rejetés en haute volatilité
                    if signal_strength in ['weak', 'very_weak']:
                        logger.debug(f"{self.name}: Signal faible rejeté en haute volatilité pour {self.symbol}")
                        return False
                        
                elif volatility_regime == "low":
                    # Basse volatilité - attention aux faux breakouts
                    if signal_confidence < self.low_vol_min_confidence:
                        logger.debug(f"{self.name}: Basse volatilité nécessite confidence ≥ {self.low_vol_min_confidence} pour {self.symbol}")
                        return False
                        
            # 2. Validation selon percentile ATR
            if atr_percentile >= self.extreme_atr_percentile:
                # Volatilité extrême
                if signal_confidence < self.extreme_vol_min_confidence:
                    logger.debug(f"{self.name}: ATR extrême ({self._safe_format(atr_percentile, '.1f')}%) nécessite confidence ≥ {self.extreme_vol_min_confidence} pour {self.symbol}")
                    return False
                    
            elif atr_percentile >= self.high_atr_percentile:
                # Volatilité élevée
                if signal_confidence < self.high_vol_min_confidence:
                    logger.debug(f"{self.name}: ATR élevé ({self._safe_format(atr_percentile, '.1f')}%) nécessite confidence ≥ {self.high_vol_min_confidence} pour {self.symbol}")
                    return False
                    
            elif atr_percentile <= self.low_atr_percentile:
                # Volatilité très faible - méfiance breakouts
                if bb_squeeze and signal_strength not in ['strong', 'very_strong']:
                    logger.debug(f"{self.name}: Compression + ATR faible nécessite signal fort pour {self.symbol}")
                    return False
                    
            # 3. Validation Bollinger Bands
            if bb_width is not None:
                # Compression extrême - signaux difficiles
                if bb_width <= self.bb_squeeze_threshold:
                    if signal_confidence < 0.7:
                        logger.debug(f"{self.name}: Compression BB extrême nécessite confidence élevée pour {self.symbol}")
                        return False
                        
                # Expansion extrême - attention aux reversals
                elif bb_width >= self.bb_extreme_expansion:
                    if signal_side == "BUY" and bb_position is not None and bb_position > 0.80:
                        # BUY près de la bande supérieure en expansion = dangereux
                        if signal_confidence < 0.70:
                            logger.debug(f"{self.name}: BUY près BB supérieure en expansion nécessite forte confidence pour {self.symbol}")
                            return False
                            
                    elif signal_side == "SELL" and bb_position is not None and bb_position < 0.20:
                        # SELL près de la bande inférieure en expansion = dangereux
                        if signal_confidence < 0.70:
                            logger.debug(f"{self.name}: SELL près BB inférieure en expansion nécessite forte confidence pour {self.symbol}")
                            return False
                            
            # 4. Cohérence signal avec expansion/contraction
            if bb_squeeze and bb_expansion:
                # Transition squeeze -> expansion - peut être favorable
                logger.debug(f"{self.name}: Transition squeeze->expansion détectée pour {self.symbol}")
            elif bb_squeeze:
                # Pure compression - signaux moins fiables sauf breakout confirmé
                if signal_strength == 'weak':
                    logger.debug(f"{self.name}: Signal faible rejeté pendant compression BB pour {self.symbol}")
                    return False
                    
            # 5. Validation régime de marché cohérent
            if market_regime and regime_strength is not None:
                if regime_strength < self.regime_stability_threshold:
                    # Régime instable - exigences accrues
                    if signal_confidence < 0.7:
                        logger.debug(f"{self.name}: Régime instable nécessite confidence élevée pour {self.symbol}")
                        return False
                        
                # Cohérence signal/régime
                if market_regime in ["TRENDING_BULL", "TRENDING_BEAR"]:
                    # En trending, favoriser signaux dans le sens
                    pass  # Validation OK
                elif market_regime == "RANGING":
                    # En ranging, attention aux breakouts
                    if signal_strength not in ['strong', 'very_strong'] and atr_percentile < 30:
                        logger.debug(f"{self.name}: Range + faible volatilité nécessite signal fort pour {self.symbol}")
                        return False
                        
            # 6. Validation NATR si disponible
            if natr is not None:
                # NATR très élevé - prudence
                if natr > 5.0:  # 5% de NATR = très volatil
                    if signal_confidence < 0.70:
                        logger.debug(f"{self.name}: NATR élevé ({self._safe_format(natr, '.2f')}%) nécessite forte confidence pour {self.symbol}")
                        return False
                        
            # Debug pour identifier la variable problématique
            try:
                logger.debug(f"{self.name}: DEBUG - atr_percentile type: {type(atr_percentile)}, value: {atr_percentile}")
                logger.debug(f"{self.name}: DEBUG - bb_width type: {type(bb_width)}, value: {bb_width}")
                
                logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                            f"Régime: {volatility_regime or 'N/A'}, "
                            f"ATR percentile: {self._safe_format(atr_percentile, '.1f') if atr_percentile is not None else 'N/A'}%, "
                            f"BB width: {self._safe_format(bb_width, '.3f') if bb_width is not None else 'N/A'}, "
                            f"Side: {signal_side}")
            except Exception as e:
                logger.error(f"{self.name}: Erreur debug log: {e}")
                logger.debug(f"{self.name}: Signal validé pour {self.symbol} (log simplifié)")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur les régimes de volatilité.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur la volatilité
            volatility_regime = self.context.get('volatility_regime')
            atr_percentile = float(self.context.get('atr_percentile', 50)) if self.context.get('atr_percentile') is not None else 50
            bb_width = float(self.context.get('bb_width', 0)) if self.context.get('bb_width') is not None else None
            bb_squeeze = self.context.get('bb_squeeze', False)
            bb_expansion = self.context.get('bb_expansion', False)
            regime_strength_raw = self.context.get('regime_strength')
            regime_strength = self._convert_regime_strength_to_score(str(regime_strength_raw)) if regime_strength_raw is not None else 0.5
            
            signal_confidence = signal.get('confidence', 0.0)
            signal_strength = signal.get('strength', 'moderate')
            signal_side = signal.get('side')
            bb_position = float(self.context.get('bb_position', 0.5)) if self.context.get('bb_position') is not None else 0.5
            
            base_score = 0.5  # Score de base si validé
            
            # CORRECTION: Scoring selon régime de volatilité avec logique directionnelle
            if volatility_regime == "normal":
                base_score += 0.2  # Régime optimal pour tous
            elif volatility_regime == "low":
                if bb_expansion:
                    # Expansion après compression : favoriser breakout directionnel
                    if (signal_side == "BUY" and bb_position <= 0.6) or \
                       (signal_side == "SELL" and bb_position >= 0.4):
                        base_score += 0.20  # Breakout directionnel favorable
                    else:
                        base_score += 0.10  # Breakout mais position moins idéale
                else:
                    # Basse volatilité sans expansion
                    if signal_strength in ['strong', 'very_strong']:
                        base_score += 0.08  # Signal fort en basse vol
                    else:
                        base_score += 0.02  # Basse volatilité peu favorable
            elif volatility_regime == "high":
                # Haute volatilité : favoriser les stratégies adaptées
                if signal_confidence >= 0.8:
                    # Confiance élevée : adapter selon direction et position
                    if signal_side == "BUY" and bb_position <= 0.3:
                        base_score += 0.15  # BUY en zone basse haute vol = opportunité
                    elif signal_side == "SELL" and bb_position >= 0.7:
                        base_score += 0.15  # SELL en zone haute haute vol = opportunité
                    else:
                        base_score += 0.08  # Haute vol maîtrisée position neutre
                else:
                    base_score -= 0.05  # Haute volatilité risquée sans confidence
            elif volatility_regime == "extreme":
                # Volatilité extrême : très sélectif selon direction
                if signal_confidence >= 0.85:
                    if (signal_side == "BUY" and bb_position <= 0.2) or \
                       (signal_side == "SELL" and bb_position >= 0.8):
                        base_score += 0.10  # Positions extrêmes en vol extrême
                    else:
                        base_score -= 0.05  # Vol extrême position défavorable
                else:
                    base_score -= 0.10  # Vol extrême sans très haute confidence
                    
            # CORRECTION: Scoring selon percentile ATR avec logique directionnelle
            if 30 <= atr_percentile <= 70:
                base_score += 0.15  # Zone ATR optimale pour tous
            elif atr_percentile < 20:
                # Volatilité très faible : favoriser signaux forts directionnels
                if signal_strength in ['strong', 'very_strong']:
                    if (signal_side == "BUY" and bb_position <= 0.4) or \
                       (signal_side == "SELL" and bb_position >= 0.6):
                        base_score += 0.10  # Signal fort directionnel en faible vol
                    else:
                        base_score += 0.05  # Signal fort mais position moins idéale
                else:
                    base_score += 0.02  # Volatilité très faible signal moyen
            elif atr_percentile > 90:
                # Volatilité extrême : adapter selon position et direction
                if (signal_side == "BUY" and bb_position <= 0.2) or \
                   (signal_side == "SELL" and bb_position >= 0.8):
                    base_score -= 0.05  # Positions extrêmes moins risquées
                else:
                    base_score -= 0.15  # Vol extrême position défavorable
                
            # CORRECTION: Bonus/malus Bollinger Bands avec logique directionnelle
            if bb_width is not None:
                if bb_squeeze and bb_expansion:
                    # Breakout de squeeze : favoriser selon position + direction
                    if signal_side == "BUY" and bb_position <= 0.6:
                        base_score += 0.25  # BUY breakout depuis zone basse/médiane
                    elif signal_side == "SELL" and bb_position >= 0.4:
                        base_score += 0.25  # SELL breakout depuis zone haute/médiane
                    else:
                        base_score += 0.15  # Breakout général mais position moins idéale
                        
                elif bb_expansion and not bb_squeeze:
                    # Expansion normale : vérifier position selon direction
                    if signal_side == "BUY" and bb_position <= 0.5:
                        base_score += 0.15  # BUY en expansion position favorable
                    elif signal_side == "SELL" and bb_position >= 0.5:
                        base_score += 0.15  # SELL en expansion position favorable
                    elif signal_side == "BUY" and bb_position >= 0.8:
                        base_score -= 0.05  # BUY en expansion près BB supérieure = risqué
                    elif signal_side == "SELL" and bb_position <= 0.2:
                        base_score -= 0.05  # SELL en expansion près BB inférieure = risqué
                    else:
                        base_score += 0.08  # Expansion normale position neutre
                        
                elif bb_squeeze:
                    # Compression : favoriser signaux forts avec position directionnelle
                    if signal_strength in ['strong', 'very_strong']:
                        if signal_side == "BUY" and bb_position <= 0.4:
                            base_score += 0.10  # BUY fort en compression zone basse
                        elif signal_side == "SELL" and bb_position >= 0.6:
                            base_score += 0.10  # SELL fort en compression zone haute
                        else:
                            base_score += 0.05  # Signal fort pendant compression position neutre
                    else:
                        base_score -= 0.05  # Signal faible pendant compression
                        
            # Bonus stabilité régime
            if regime_strength >= 0.8:
                base_score += 0.1  # Régime stable
            elif regime_strength < 0.5:
                base_score -= 0.05  # Régime instable
                
            # CORRECTION: Bonus confidence adaptée à la volatilité et direction
            if volatility_regime in ["high", "extreme"] and signal_confidence >= 0.85:
                # Haute/extrême volatilité avec très haute confidence
                if (signal_side == "BUY" and bb_position <= 0.5) or \
                   (signal_side == "SELL" and bb_position >= 0.5):
                    base_score += 0.12  # Excellente cohérence direction/vol/confidence
                else:
                    base_score += 0.08  # Bonne confidence en haute vol
            elif volatility_regime == "low" and signal_confidence >= 0.6:
                # Basse volatilité : favoriser breakouts directionnels confiants
                if bb_expansion or (bb_width is not None and bb_width > self.bb_squeeze_threshold * 1.5):
                    if (signal_side == "BUY" and bb_position <= 0.6) or \
                       (signal_side == "SELL" and bb_position >= 0.4):
                        base_score += 0.10  # Confident + expansion directionnelle
                    else:
                        base_score += 0.05  # Confident + expansion position neutre
                else:
                    base_score += 0.03  # Confident en basse vol sans expansion
                
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
            signal_confidence = signal.get('confidence', 0.0)
            volatility_regime = self.context.get('volatility_regime', 'unknown')
            atr_percentile = float(self.context.get('atr_percentile', 50)) if self.context.get('atr_percentile') is not None else 50
            bb_squeeze = self.context.get('bb_squeeze', False)
            bb_expansion = self.context.get('bb_expansion', False)
            
            if is_valid:
                regime_desc = f"régime {volatility_regime}" if volatility_regime != 'unknown' else f"ATR {self._safe_format(atr_percentile, '.0f')}e percentile"
                
                if bb_squeeze and bb_expansion:
                    condition = "breakout de compression"
                elif volatility_regime == "normal":
                    condition = "volatilité optimale"
                elif atr_percentile >= self.extreme_atr_percentile:
                    condition = "haute volatilité maîtrisée"
                else:
                    condition = "conditions acceptables"
                    
                return f"{self.name}: Validé - {condition} ({regime_desc}, confidence: {self._safe_format(signal_confidence, '.2f')}) pour signal {signal_side}"
            else:
                if atr_percentile >= self.extreme_atr_percentile and signal_confidence < self.extreme_vol_min_confidence:
                    return f"{self.name}: Rejeté - Volatilité extrême ({self._safe_format(atr_percentile, '.0f')}%) nécessite confidence ≥ {self.extreme_vol_min_confidence}"
                elif volatility_regime == "high" and signal_confidence < self.high_vol_min_confidence:
                    return f"{self.name}: Rejeté - Haute volatilité nécessite confidence ≥ {self.high_vol_min_confidence}"
                elif bb_squeeze and signal_confidence < 0.7:
                    return f"{self.name}: Rejeté - Compression BB nécessite confidence élevée"
                else:
                    return f"{self.name}: Rejeté - Conditions volatilité défavorables"
                    
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données de volatilité requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Vérifier qu'au moins un indicateur de volatilité est présent
        has_regime = 'volatility_regime' in self.context and self.context['volatility_regime'] is not None
        has_atr = 'atr_14' in self.context and self.context['atr_14'] is not None
        has_atr_percentile = 'atr_percentile' in self.context and self.context['atr_percentile'] is not None
        has_bb_width = 'bb_width' in self.context and self.context['bb_width'] is not None
        has_natr = 'natr' in self.context and self.context['natr'] is not None
        
        if not (has_regime or has_atr or has_atr_percentile or has_bb_width or has_natr):
            logger.warning(f"{self.name}: Aucun indicateur de volatilité disponible pour {self.symbol}")
            return False
            
        return True
    
    def _convert_regime_strength_to_score(self, regime_strength_str: str) -> float:
        """Convertit une force de régime en score numérique."""
        try:
            if not regime_strength_str:
                return 0.5
                
            strength_lower = regime_strength_str.lower()
            
            if strength_lower in ['very_strong', 'strong']:
                return 0.9
            elif strength_lower in ['moderate', 'medium']:
                return 0.6
            elif strength_lower in ['weak', 'very_weak']:
                return 0.3
            elif strength_lower in ['neutral', 'absent']:
                return 0.1
            else:
                # Essayer de convertir directement en float
                try:
                    return float(regime_strength_str)
                except (ValueError, TypeError):
                    return 0.5
                    
        except Exception:
            return 0.5
