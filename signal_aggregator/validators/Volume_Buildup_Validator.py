"""
Volume_Buildup_Validator - Validator basé sur l'accumulation de volume et patterns de liquidité.
"""

from typing import Dict, Any, Optional
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class Volume_Buildup_Validator(BaseValidator):
    """
    Validator pour l'accumulation de volume - filtre selon les patterns de volume et liquidité.
    
    Vérifie: Volume trend, accumulation patterns, buy/sell pressure, liquidité
    Catégorie: technical
    
    Rejette les signaux en:
    - Volume insuffisant ou décroissant
    - Absence de buildup patterns
    - Déséquilibre buy/sell pressure défavorable
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "Volume_Buildup_Validator"
        self.category = "technical"
        
        # Paramètres volume de base - OPTIMISÉS
        self.min_volume_ratio = 1.4          # Volume 40% au-dessus moyenne AUGMENTÉ (1.4x au lieu de 1.2x)
        self.strong_volume_ratio = 2.2       # Volume considéré fort AUGMENTÉ (2.2x au lieu de 2.0x)
        self.exceptional_volume_ratio = 3.5  # Volume exceptionnel AUGMENTÉ (3.5x au lieu de 3.0x)
        self.min_volume_trend = 0.15         # Trend volume croissant minimum AUGMENTÉ (15% au lieu de 10%)
        
        # Paramètres accumulation/distribution - OPTIMISÉS
        self.min_accumulation_score = 50.0   # Score accumulation minimum AUGMENTÉ (50 au lieu de 40)
        self.strong_accumulation_threshold = 0.75  # Accumulation forte AUGMENTÉE (75% au lieu de 70%)
        self.min_buy_pressure = 0.50         # Pression acheteuse minimum AUGMENTÉE (50% au lieu de 45%)
        self.optimal_buy_pressure = 0.70     # Pression acheteuse optimale AUGMENTÉE (70% au lieu de 65%)
        
        # Paramètres buildup patterns - OPTIMISÉS
        self.min_buildup_bars = 5            # Barres minimum pour buildup AUGMENTÉ (5 au lieu de 3)
        self.optimal_buildup_bars = 8        # Barres optimales buildup AUGMENTÉ (8 au lieu de 5)
        self.buildup_slope_threshold = 0.08  # Pente minimum buildup AUGMENTÉE (8% au lieu de 5%)
        self.buildup_consistency = 0.70      # Consistance buildup minimum AUGMENTÉE (70% au lieu de 60%)
        
        # Paramètres liquidité et profondeur - OPTIMISÉS
        self.min_liquidity_score = 40.0      # Score liquidité minimum AUGMENTÉ (40 au lieu de 30)
        self.min_bid_ask_ratio = 0.85        # Ratio bid/ask minimum AUGMENTÉ (85% au lieu de 80%)
        self.max_spread_ratio = 0.0015       # Spread maximum RÉDUIT (0.15% au lieu de 0.2%)
        self.min_market_depth = 0.6          # Profondeur marché minimum AUGMENTÉE (60% au lieu de 50%)
        
        # Paramètres OBV et flux - OPTIMISÉS
        self.min_obv_trend = 0.05            # OBV trend neutre minimum AUGMENTÉ (5% au lieu de 0%)
        self.obv_divergence_threshold = 0.25  # Seuil divergence OBV/prix PLUS STRICT (25% au lieu de 30%)
        self.min_money_flow = 0.35           # Money flow minimum AUGMENTÉ (35% au lieu de 30%)
        self.strong_money_flow = 0.75        # Money flow fort AUGMENTÉ (75% au lieu de 70%)
        
        # Paramètres qualité volume - OPTIMISÉS
        self.min_volume_quality = 0.5        # Qualité volume minimum AUGMENTÉE (50% au lieu de 40%)
        self.min_trade_size_ratio = 0.85     # Ratio taille trades minimum AUGMENTÉ (85% au lieu de 80%)
        self.max_volume_volatility = 1.8     # Volatilité volume maximum RÉDUITE (1.8x au lieu de 2.0x)
        
        # Bonus/malus
        self.exceptional_volume_bonus = 0.30  # Bonus volume exceptionnel
        self.strong_accumulation_bonus = 0.25 # Bonus forte accumulation
        self.buildup_pattern_bonus = 0.20    # Bonus pattern buildup
        self.liquidity_bonus = 0.15          # Bonus bonne liquidité
        self.weak_volume_penalty = -0.25     # Pénalité volume faible
        self.distribution_penalty = -0.30    # Pénalité distribution
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur l'accumulation de volume et patterns de liquidité.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon volume buildup, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs volume depuis le contexte
            try:
                # Volume de base
                volume_ratio = float(self.context.get('volume_ratio', 1.0)) if self.context.get('volume_ratio') is not None else None
                # volume_trend → volume_pattern (pattern de volume - toujours string)
                volume_trend_raw = self.context.get('volume_pattern')
                # volume_pattern est toujours une string catégorielle: 'DECLINING', 'NORMAL', 'INCREASING' 
                volume_trend = str(volume_trend_raw) if volume_trend_raw is not None else 'NORMAL'
                # volume_sma_20 → avg_volume_20 (existe déjà!)
                volume_sma_20 = float(self.context.get('avg_volume_20', 0)) if self.context.get('avg_volume_20') is not None else None
                
                # Accumulation/Distribution
                # accumulation_distribution_score → ad_line
                accumulation_distribution_score = float(self.context.get('ad_line', 50.0)) if self.context.get('ad_line') is not None else None
                # buy_sell_pressure → volume_pattern (utiliser mapping)
                buy_sell_pressure_raw = self.context.get('volume_pattern', 'NORMAL')
                buy_sell_pressure = self._convert_volume_pattern_to_pressure(str(buy_sell_pressure_raw))
                # volume_weighted_price → vwap_10
                volume_weighted_price = float(self.context.get('vwap_10', 0)) if self.context.get('vwap_10') is not None else None
                
                # Buildup patterns
                # volume_buildup_bars → volume_buildup_periods (existe déjà!)
                volume_buildup_bars = int(self.context.get('volume_buildup_periods', 0)) if self.context.get('volume_buildup_periods') is not None else None
                # volume_buildup_slope → trend_angle
                volume_buildup_slope = float(self.context.get('trend_angle', 0)) if self.context.get('trend_angle') is not None else None
                # volume_buildup_consistency → volume_quality_score (format 0-100, normaliser vers 0-1)
                volume_quality_raw = self.context.get('volume_quality_score')
                volume_buildup_consistency = float(volume_quality_raw) / 100.0 if volume_quality_raw is not None and volume_quality_raw > 1 else float(volume_quality_raw) if volume_quality_raw is not None else None
                
                # Liquidité
                # liquidity_score → trade_intensity
                liquidity_score = float(self.context.get('trade_intensity', 50.0)) if self.context.get('trade_intensity') is not None else None
                # bid_ask_ratio → quote_volume_ratio
                bid_ask_ratio = float(self.context.get('quote_volume_ratio', 1.0)) if self.context.get('quote_volume_ratio') is not None else None
                # spread_ratio → volume_quality_score
                spread_ratio = float(self.context.get('volume_quality_score', 0.001)) if self.context.get('volume_quality_score') is not None else None
                # market_depth_score → volume_quality_score
                market_depth_score = float(self.context.get('volume_quality_score', 0.5)) if self.context.get('volume_quality_score') is not None else None
                
                # OBV et flux
                # obv_trend → obv_oscillator
                obv_trend = float(self.context.get('obv_oscillator', 0)) if self.context.get('obv_oscillator') is not None else None
                # obv_price_divergence → signal_strength
                obv_price_divergence = float(self.context.get('signal_strength', 0)) if self.context.get('signal_strength') is not None else None
                # money_flow_index → mfi_14 (existe déjà!)
                money_flow_index = float(self.context.get('mfi_14', 0.5)) if self.context.get('mfi_14') is not None else None
                # chaikin_money_flow → ad_line
                chaikin_money_flow = float(self.context.get('ad_line', 0)) if self.context.get('ad_line') is not None else None
                
                # Qualité volume
                volume_quality_score = float(self.context.get('volume_quality_score', 50.0)) if self.context.get('volume_quality_score') is not None else None
                # average_trade_size_ratio → avg_trade_size
                average_trade_size_ratio = float(self.context.get('avg_trade_size', 1.0)) if self.context.get('avg_trade_size') is not None else None
                # volume_volatility → volatility_regime (catégoriel)
                volume_volatility_raw = self.context.get('volatility_regime', 'normal')
                volume_volatility = self._convert_volatility_to_score(str(volume_volatility_raw))
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion indicateurs pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', 'Unknown')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side:
                logger.warning(f"{self.name}: Signal side manquant pour {self.symbol}")
                return False
                
            # 1. Validation ratio volume minimum - PLUS STRICT
            if volume_ratio is not None and volume_ratio < self.min_volume_ratio:
                logger.debug(f"{self.name}: Volume insuffisant ({self._safe_format(volume_ratio, '.2f')}x) pour {self.symbol}")
                if signal_confidence < 0.85:  # AUGMENTÉ de 80% à 85%
                    return False
                    
            # 2. Validation trend volume (catégoriel) - PLUS STRICT
            if volume_trend is not None and volume_trend.upper() in ['DECLINING', 'FALLING', 'WEAK']:
                logger.debug(f"{self.name}: Trend volume décroissant ({volume_trend}) pour {self.symbol}")
                if signal_confidence < 0.78:  # AUGMENTÉ de 70% à 78%
                    return False
                    
            # 3. Validation accumulation/distribution selon signal - PLUS STRICT
            if accumulation_distribution_score is not None:
                # ad_line: valeurs positives = accumulation, négatives = distribution
                if signal_side == "BUY" and accumulation_distribution_score < 0:
                    logger.debug(f"{self.name}: A/D Line négative ({self._safe_format(accumulation_distribution_score, '.2f')}) défavorable pour BUY {self.symbol}")
                    if signal_confidence < 0.75:  # AUGMENTÉ de 70% à 75%
                        return False
                elif signal_side == "SELL" and accumulation_distribution_score > 0:
                    logger.debug(f"{self.name}: A/D Line positive ({self._safe_format(accumulation_distribution_score, '.2f')}) défavorable pour SELL {self.symbol}")
                    if signal_confidence < 0.75:  # AUGMENTÉ de 70% à 75%
                        return False
                        
            # 4. Validation buy/sell pressure - PLUS STRICT
            if buy_sell_pressure is not None:
                if signal_side == "BUY" and buy_sell_pressure < self.min_buy_pressure:
                    logger.debug(f"{self.name}: Pression acheteuse insuffisante ({self._safe_format(buy_sell_pressure, '.2f')}) pour {self.symbol}")
                    if signal_confidence < 0.68:  # AUGMENTÉ de 60% à 68%
                        return False
                elif signal_side == "SELL" and buy_sell_pressure > (1 - self.min_buy_pressure):
                    logger.debug(f"{self.name}: Pression vendeuse insuffisante ({self._safe_format(1-buy_sell_pressure, '.2f')}) pour {self.symbol}")
                    if signal_confidence < 0.68:  # AUGMENTÉ de 60% à 68%
                        return False
                        
            # 5. Validation buildup pattern
            if volume_buildup_bars is not None and volume_buildup_bars < self.min_buildup_bars:
                logger.debug(f"{self.name}: Pattern buildup insuffisant ({volume_buildup_bars} barres) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            if volume_buildup_slope is not None and volume_buildup_slope < self.buildup_slope_threshold:
                logger.debug(f"{self.name}: Pente buildup insuffisante ({self._safe_format(volume_buildup_slope, '.3f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            if volume_buildup_consistency is not None and volume_buildup_consistency < self.buildup_consistency:
                logger.debug(f"{self.name}: Consistance buildup insuffisante ({self._safe_format(volume_buildup_consistency, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 6. Validation liquidité
            if liquidity_score is not None and liquidity_score < self.min_liquidity_score:
                logger.debug(f"{self.name}: Score liquidité insuffisant ({self._safe_format(liquidity_score, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.5:
                    return False
                    
            if spread_ratio is not None and spread_ratio > self.max_spread_ratio:
                logger.debug(f"{self.name}: Spread excessif ({self._safe_format(spread_ratio*100, '.3f')}%) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            if bid_ask_ratio is not None and bid_ask_ratio < self.min_bid_ask_ratio:
                logger.debug(f"{self.name}: Ratio bid/ask défavorable ({self._safe_format(bid_ask_ratio, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.5:
                    return False
                    
            # 7. Validation OBV trend selon signal
            if obv_trend is not None:
                if signal_side == "BUY" and obv_trend < self.min_obv_trend:
                    logger.debug(f"{self.name}: OBV trend baissier ({self._safe_format(obv_trend, '.2f')}) pour BUY {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                elif signal_side == "SELL" and obv_trend > -self.min_obv_trend:
                    logger.debug(f"{self.name}: OBV trend haussier ({self._safe_format(obv_trend, '.2f')}) pour SELL {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                        
            # 8. Validation divergence OBV/prix
            if obv_price_divergence is not None and abs(obv_price_divergence) > self.obv_divergence_threshold:
                logger.debug(f"{self.name}: Divergence OBV/prix excessive ({self._safe_format(obv_price_divergence, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 9. Validation money flow (mfi_14: format 0-100, 50=neutre)
            if money_flow_index is not None:
                # Conversion si nécessaire: MFI peut être 0-100 ou 0-1
                mfi_norm = money_flow_index / 100.0 if money_flow_index > 1 else money_flow_index
                if signal_side == "BUY" and mfi_norm < self.min_money_flow:
                    logger.debug(f"{self.name}: Money flow insuffisant ({self._safe_format(mfi_norm, '.2f')}) pour BUY {self.symbol}")
                    if signal_confidence < 0.6:
                        return False
                elif signal_side == "SELL" and mfi_norm > (1 - self.min_money_flow):
                    logger.debug(f"{self.name}: Money flow excessif ({self._safe_format(mfi_norm, '.2f')}) pour SELL {self.symbol}")
                    if signal_confidence < 0.6:
                        return False
                        
            # 10. Validation qualité volume (format 0-100)
            if volume_quality_score is not None:
                quality_norm = volume_quality_score / 100.0 if volume_quality_score > 1 else volume_quality_score
                if quality_norm < self.min_volume_quality:
                    logger.debug(f"{self.name}: Qualité volume insuffisante ({self._safe_format(quality_norm, '.2f')}) pour {self.symbol}")
                    if signal_confidence < 0.5:
                        return False
                    
            # 11. Validation volatilité volume
            if volume_volatility is not None and volume_volatility > self.max_volume_volatility:
                logger.debug(f"{self.name}: Volatilité volume excessive ({self._safe_format(volume_volatility, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 12. Validation cohérence VWAP
            current_price = None
            if self.data and 'close' in self.data and self.data['close']:
                try:
                    current_price = self._get_current_price()
                except (IndexError, ValueError, TypeError):
                    pass
                    
            if current_price and volume_weighted_price:
                vwap_coherence = self._validate_vwap_coherence(signal_side, current_price, volume_weighted_price)
                if not vwap_coherence:
                    logger.debug(f"{self.name}: Incohérence VWAP pour {signal_side} {self.symbol}")
                    if signal_confidence < 0.68:  # AUGMENTÉ de 60% à 68%
                        return False
                        
            # NOUVEAU: Validation finale - rejet des signaux moyennement confiants avec volume faible
            overall_volume_quality = self._calculate_overall_volume_quality(
                volume_ratio, accumulation_distribution_score, buy_sell_pressure, 
                volume_buildup_bars, liquidity_score
            )
            if overall_volume_quality < 0.55 and signal_confidence < 0.72:
                logger.debug(f"{self.name}: Volume qualité médiocre ({overall_volume_quality:.2f}) + signal confidence insuffisante pour {self.symbol}")
                return False
                        
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                        f"Volume: {self._safe_format(volume_ratio, '.2f')}x, "
                        f"Accumulation: {self._safe_format(accumulation_distribution_score, '.2f') if accumulation_distribution_score is not None else 'N/A'}, "
                        f"Buy pressure: {self._safe_format(buy_sell_pressure, '.2f') if buy_sell_pressure is not None else 'N/A'}, "
                        f"Buildup: {volume_buildup_bars or 'N/A'} barres")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def _validate_vwap_coherence(self, signal_side: str, current_price: float, vwap: float) -> bool:
        """Valide la cohérence prix/VWAP selon direction signal."""
        try:
            if not vwap or vwap <= 0:
                return True
                
            # Pour BUY, préférer prix proche ou sous VWAP
            if signal_side == "BUY":
                if current_price > vwap * 1.02:  # Prix >2% au-dessus VWAP
                    return False
                    
            # Pour SELL, préférer prix proche ou au-dessus VWAP
            elif signal_side == "SELL":
                if current_price < vwap * 0.98:  # Prix >2% sous VWAP
                    return False
                    
            return True
            
        except Exception:
            return True
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur l'accumulation de volume.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur volume buildup
            volume_ratio = float(self.context.get('volume_ratio', 1.0)) if self.context.get('volume_ratio') is not None else 1.0
            # volume_trend → volume_pattern (pattern de volume - toujours string)
            volume_trend_raw = self.context.get('volume_pattern')
            # volume_pattern est toujours une string catégorielle: 'DECLINING', 'NORMAL', 'INCREASING' 
            volume_trend = str(volume_trend_raw) if volume_trend_raw is not None else 'NORMAL'
            # accumulation_distribution_score → ad_line
            accumulation_distribution_score = float(self.context.get('ad_line', 50.0)) if self.context.get('ad_line') is not None else 50.0
            # buy_sell_pressure → volume_pattern (utiliser mapping)
            buy_sell_pressure_raw = self.context.get('volume_pattern', 'NORMAL')
            buy_sell_pressure = self._convert_volume_pattern_to_pressure(str(buy_sell_pressure_raw))
            volume_buildup_bars = int(self.context.get('volume_buildup_periods', 0)) if self.context.get('volume_buildup_periods') is not None else 0
            # volume_buildup_consistency → volume_quality_score (format 0-100, normaliser vers 0-1)
            volume_quality_raw = self.context.get('volume_quality_score')
            volume_buildup_consistency = float(volume_quality_raw) / 100.0 if volume_quality_raw is not None and volume_quality_raw > 1 else float(volume_quality_raw) if volume_quality_raw is not None else 0.5
            # liquidity_score → trade_intensity
            liquidity_score = float(self.context.get('trade_intensity', 50.0)) if self.context.get('trade_intensity') is not None else 50.0
            # money_flow_index → mfi_14 (existe déjà!)
            money_flow_index = float(self.context.get('mfi_14', 0.5)) if self.context.get('mfi_14') is not None else 0.5
            volume_quality_score = float(self.context.get('volume_quality_score', 50.0)) if self.context.get('volume_quality_score') is not None else 50.0
            
            signal_side = signal.get('side')
            
            base_score = 0.46  # Score de base réduit (46% au lieu de 50%)
            
            # Bonus volume ratio
            if volume_ratio >= self.exceptional_volume_ratio:
                base_score += self.exceptional_volume_bonus
            elif volume_ratio >= self.strong_volume_ratio:
                base_score += 0.20
            elif volume_ratio >= self.min_volume_ratio + 0.3:
                base_score += 0.10
                
            # Bonus trend volume croissant (catégoriel)
            if volume_trend and volume_trend.upper() in ['INCREASING', 'RISING', 'STRONG', 'BULLISH']:
                base_score += 0.15  # Trend volume très positif
            elif volume_trend and volume_trend.upper() in ['NORMAL', 'STABLE', 'MODERATE']:
                base_score += 0.10  # Trend volume positif
                
            # Bonus accumulation/distribution selon signal (ad_line: positif=accumulation, négatif=distribution)
            if signal_side == "BUY":
                if accumulation_distribution_score > 100:  # Forte accumulation
                    base_score += self.strong_accumulation_bonus
                elif accumulation_distribution_score > 0:  # Accumulation modérée
                    base_score += 0.15
            elif signal_side == "SELL":
                if accumulation_distribution_score < -100:  # Forte distribution
                    base_score += self.strong_accumulation_bonus
                elif accumulation_distribution_score < 0:  # Distribution modérée
                    base_score += 0.15
                    
            # Bonus buy/sell pressure
            if signal_side == "BUY" and buy_sell_pressure >= self.optimal_buy_pressure:
                base_score += 0.15  # Pression acheteuse optimale
            elif signal_side == "SELL" and buy_sell_pressure <= (1 - self.optimal_buy_pressure):
                base_score += 0.15  # Pression vendeuse optimale
                
            # Bonus buildup pattern
            if volume_buildup_bars >= self.optimal_buildup_bars:
                base_score += self.buildup_pattern_bonus
            elif volume_buildup_bars >= self.min_buildup_bars:
                base_score += 0.10
                
            # Bonus consistance buildup
            if volume_buildup_consistency >= 0.8:
                base_score += 0.12  # Buildup très consistant
            elif volume_buildup_consistency >= self.buildup_consistency:
                base_score += 0.08
                
            # Bonus liquidité
            if liquidity_score >= 70.0:
                base_score += self.liquidity_bonus
            elif liquidity_score >= 50.0:
                base_score += 0.08
                
            # Bonus money flow (mfi_14: format 0-100, normaliser)
            mfi_norm = money_flow_index / 100.0 if money_flow_index > 1 else money_flow_index
            if signal_side == "BUY" and mfi_norm >= self.strong_money_flow:
                base_score += 0.12  # Money flow très favorable
            elif signal_side == "SELL" and mfi_norm <= (1 - self.strong_money_flow):
                base_score += 0.12
                
            # Bonus qualité volume (format 0-100, normaliser)
            quality_norm = volume_quality_score / 100.0 if volume_quality_score > 1 else volume_quality_score
            if quality_norm >= 0.8:
                base_score += 0.10  # Volume de très haute qualité
            elif quality_norm >= 0.6:
                base_score += 0.06
                
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
            signal_side = signal.get('side', 'N/A')
            signal_strategy = signal.get('strategy', 'N/A')
            
            volume_ratio = float(self.context.get('volume_ratio', 1.0)) if self.context.get('volume_ratio') is not None else None
            # accumulation_distribution_score → ad_line
            accumulation_distribution_score = float(self.context.get('ad_line', 50.0)) if self.context.get('ad_line') is not None else None
            # buy_sell_pressure → volume_pattern (utiliser mapping)
            buy_sell_pressure_raw = self.context.get('volume_pattern', 'NORMAL')
            buy_sell_pressure = self._convert_volume_pattern_to_pressure(str(buy_sell_pressure_raw))
            volume_buildup_bars = int(self.context.get('volume_buildup_periods', 0)) if self.context.get('volume_buildup_periods') is not None else None
            # liquidity_score → trade_intensity
            liquidity_score = float(self.context.get('trade_intensity', 50.0)) if self.context.get('trade_intensity') is not None else None
            
            if is_valid:
                reason = f"Volume buildup favorable"
                if volume_ratio:
                    reason += f" (ratio: {self._safe_format(volume_ratio, '.2f')}x)"
                if accumulation_distribution_score:
                    acc_desc = "accumulation" if accumulation_distribution_score > 50.0 else "distribution"
                    reason += f", {acc_desc}: {self._safe_format(abs(accumulation_distribution_score), '.2f')}"
                if buy_sell_pressure:
                    pressure_desc = "achat" if buy_sell_pressure > 0.5 else "vente"
                    reason += f", pression {pressure_desc}: {self._safe_format(buy_sell_pressure, '.2f')}"
                if volume_buildup_bars:
                    reason += f", buildup: {volume_buildup_bars} barres"
                if liquidity_score:
                    reason += f", liquidité: {self._safe_format(liquidity_score, '.2f')}"
                    
                return f"{self.name}: Validé - {reason} pour {signal_strategy} {signal_side}"
            else:
                if volume_ratio and volume_ratio < self.min_volume_ratio:
                    return f"{self.name}: Rejeté - Volume insuffisant ({self._safe_format(volume_ratio, '.2f')}x)"
                elif signal_side == "BUY" and accumulation_distribution_score and accumulation_distribution_score < self.min_accumulation_score:
                    return f"{self.name}: Rejeté - Score accumulation insuffisant ({self._safe_format(accumulation_distribution_score, '.2f')})"
                elif signal_side == "SELL" and accumulation_distribution_score and accumulation_distribution_score > -self.min_accumulation_score:
                    return f"{self.name}: Rejeté - Score distribution insuffisant ({self._safe_format(accumulation_distribution_score, '.2f')})"
                elif buy_sell_pressure:
                    if signal_side == "BUY" and buy_sell_pressure < self.min_buy_pressure:
                        return f"{self.name}: Rejeté - Pression acheteuse insuffisante ({self._safe_format(buy_sell_pressure, '.2f')})"
                    elif signal_side == "SELL" and buy_sell_pressure > (1 - self.min_buy_pressure):
                        return f"{self.name}: Rejeté - Pression vendeuse insuffisante ({self._safe_format(1-buy_sell_pressure, '.2f')})"
                elif volume_buildup_bars and volume_buildup_bars < self.min_buildup_bars:
                    return f"{self.name}: Rejeté - Pattern buildup insuffisant ({volume_buildup_bars} barres)"
                elif liquidity_score and liquidity_score < self.min_liquidity_score:
                    return f"{self.name}: Rejeté - Liquidité insuffisante ({self._safe_format(liquidity_score, '.2f')})"
                    
                return f"{self.name}: Rejeté - Critères volume buildup non respectés"
                
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
            
        # Au minimum, on a besoin d'un indicateur de volume (avec mappings)
        volume_indicators = [
            'volume_ratio', 'volume_pattern', 'ad_line',
            'volume_pattern', 'trade_intensity', 'mfi_14'
        ]
        
        available_indicators = sum(1 for ind in volume_indicators 
                                 if ind in self.context and self.context[ind] is not None)
        
        if available_indicators < 2:
            logger.warning(f"{self.name}: Pas assez d'indicateurs de volume pour {self.symbol}")
            return False
            
        return True
    
    def _calculate_overall_volume_quality(self, volume_ratio: float, accumulation_score: float,
                                         buy_sell_pressure: float, buildup_bars: int, 
                                         liquidity_score: float) -> float:
        """Calcule la qualité globale du volume."""
        try:
            quality_score = 0.0
            components = 0
            
            if volume_ratio is not None:
                # Normaliser volume_ratio (1.0 = neutre)
                quality_score += min(1.0, volume_ratio / 2.0)  # Cap à 1.0 pour ratio >= 2.0
                components += 1
                
            if accumulation_score is not None:
                # Utiliser valeur absolue de l'A/D Line, normaliser vers 0-1
                quality_score += min(1.0, abs(accumulation_score) / 200.0)
                components += 1
                
            if buy_sell_pressure is not None:
                # Utiliser distance depuis 0.5 (neutre) comme métrique de force
                quality_score += abs(buy_sell_pressure - 0.5) * 2  # 0.5 devient 0, 1.0 ou 0.0 devient 1.0
                components += 1
                
            if buildup_bars is not None:
                # Normaliser buildup bars (8 barres optimales = 1.0)
                quality_score += min(1.0, buildup_bars / 8.0)
                components += 1
                
            if liquidity_score is not None:
                # Normaliser score liquidité (format 0-100)
                score_norm = liquidity_score / 100.0 if liquidity_score > 1 else liquidity_score
                quality_score += score_norm
                components += 1
                
            return quality_score / max(1, components)  # Moyenne des composants disponibles
        except:
            return 0.5  # Valeur par défaut
        
    def _get_current_price(self) -> Optional[float]:
        """Helper method to get current price from data or context."""
        if self.data:
            # Essayer d'abord la valeur scalaire (format préféré)
            if 'close' in self.data and self.data['close'] is not None:
                try:
                    if isinstance(self.data['close'], (int, float)):
                        return float(self.data['close'])
                    elif isinstance(self.data['close'], list) and len(self.data['close']) > 0:
                        return float(self.data['close'][-1])
                except (IndexError, ValueError, TypeError):
                    pass
            
            # Fallback: essayer current_price dans le contexte
            # current_price n'est pas dans analyzer_data, utiliser self.data['close']
            if current_price is not None:
                try:
                    return float(current_price)
                except (ValueError, TypeError):
                    pass
        return None
    
    def _convert_volume_pattern_to_pressure(self, volume_pattern: str) -> float:
        """Convertit un pattern de volume en score de pression achat/vente."""
        try:
            if not volume_pattern:
                return 0.5
                
            pattern_lower = volume_pattern.lower()
            
            if pattern_lower in ['increasing', 'rising', 'strong', 'bullish']:
                return 0.8  # Forte pression acheteuse
            elif pattern_lower in ['declining', 'falling', 'weak', 'bearish']:
                return 0.2  # Forte pression vendeuse
            elif pattern_lower in ['normal', 'stable', 'moderate', 'neutral']:
                return 0.5  # Pression équilibrée
            else:
                # Essayer de convertir directement en float
                try:
                    return float(volume_pattern)
                except (ValueError, TypeError):
                    return 0.5  # Valeur par défaut
                    
        except Exception:
            return 0.5
    
    def _convert_volatility_to_score(self, volatility_regime: str) -> float:
        """Convertit un régime de volatilité en score numérique."""
        try:
            if not volatility_regime:
                return 1.0
                
            vol_lower = volatility_regime.lower()
            
            if vol_lower in ['high', 'very_high', 'extreme']:
                return 2.5  # Haute volatilité
            elif vol_lower in ['normal', 'moderate', 'average']:
                return 1.0  # Volatilité normale
            elif vol_lower in ['low', 'very_low', 'minimal']:
                return 0.5  # Faible volatilité
            else:
                # Essayer de convertir directement en float
                try:
                    return float(volatility_regime)
                except (ValueError, TypeError):
                    return 1.0  # Valeur par défaut
                    
        except Exception:
            return 1.0
