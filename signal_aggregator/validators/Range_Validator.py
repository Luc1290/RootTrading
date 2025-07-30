"""
Range_Validator - Validator basé sur les conditions de range de marché.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class Range_Validator(BaseValidator):
    """
    Validator pour les conditions de range - filtre selon l'état de consolidation du marché.
    
    Vérifie: Bornes de range, volatilité, breakout potentiel, position dans range
    Catégorie: structure
    
    Rejette les signaux en:
    - Range trop étroit ou trop large
    - Position inappropriée dans le range
    - Breakout imminent probable
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "Range_Validator"
        self.category = "structure"
        
        # Paramètres range de base
        self.min_range_width = 0.005         # 0.5% largeur minimum
        self.max_range_width = 0.08          # 8% largeur maximum
        self.optimal_range_width = 0.02      # 2% largeur optimale
        self.range_age_min = 5               # Age minimum range (barres)
        self.range_age_max = 200             # Age maximum range (barres)
        
        # Paramètres position dans range
        self.range_center_zone = 0.3         # 30% zone centrale
        self.range_edge_zone = 0.2           # 20% zone de bord
        self.safe_range_zone_min = 0.25      # 25% zone sûre minimum
        self.safe_range_zone_max = 0.75      # 75% zone sûre maximum
        
        # Paramètres breakout
        self.breakout_probability_max = 0.7  # Probabilité breakout max
        self.breakout_strength_threshold = 0.6  # Force breakout seuil
        self.false_breakout_ratio = 0.3      # Ratio faux breakouts
        
        # Paramètres volatilité dans range
        self.min_range_volatility = 0.008    # 0.8% volatilité minimum
        self.max_range_volatility = 0.05     # 5% volatilité maximum
        self.range_compression_threshold = 0.6  # Seuil compression
        
        # Paramètres tests des bornes
        self.min_boundary_tests = 2          # Tests minimum des bornes
        self.optimal_boundary_tests = 4      # Tests optimaux des bornes
        self.boundary_respect_ratio = 0.7    # Ratio respect des bornes
        
        # Bonus/malus
        self.optimal_range_bonus = 0.20      # Bonus range optimal
        self.center_position_bonus = 0.15    # Bonus position centrale
        self.strong_boundaries_bonus = 0.18  # Bonus bornes solides
        self.breakout_risk_penalty = -0.25   # Pénalité risque breakout
        self.edge_position_penalty = -0.15   # Pénalité position bord
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur les conditions de range du marché.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon range conditions, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs de range depuis le contexte
            try:
                # Bornes du range
                range_high = float(self.context.get('range_high', 0)) if self.context.get('range_high') is not None else None
                range_low = float(self.context.get('range_low', 0)) if self.context.get('range_low') is not None else None
                range_width = float(self.context.get('range_width', 0)) if self.context.get('range_width') is not None else None
                range_center = float(self.context.get('range_center', 0)) if self.context.get('range_center') is not None else None
                
                # Informations temporelles et force
                range_age_bars = int(self.context.get('range_age_bars', 0)) if self.context.get('range_age_bars') is not None else None
                range_strength = float(self.context.get('range_strength', 0)) if self.context.get('range_strength') is not None else None
                boundary_test_count = int(self.context.get('boundary_test_count', 0)) if self.context.get('boundary_test_count') is not None else None
                
                # Position et mouvement dans range
                range_position = float(self.context.get('range_position', 0.5)) if self.context.get('range_position') is not None else None  # 0-1
                range_movement_direction = self.context.get('range_movement_direction')  # 'up', 'down', 'neutral'
                time_in_range = int(self.context.get('time_in_range', 0)) if self.context.get('time_in_range') is not None else None
                
                # Probabilité breakout et volatilité
                breakout_probability = float(self.context.get('breakout_probability', 0.5)) if self.context.get('breakout_probability') is not None else None
                range_volatility = float(self.context.get('range_volatility', 0.02)) if self.context.get('range_volatility') is not None else None
                range_compression = float(self.context.get('range_compression', 0.5)) if self.context.get('range_compression') is not None else None
                
                # Tests et respect des bornes
                upper_boundary_tests = int(self.context.get('upper_boundary_tests', 0)) if self.context.get('upper_boundary_tests') is not None else None
                lower_boundary_tests = int(self.context.get('lower_boundary_tests', 0)) if self.context.get('lower_boundary_tests') is not None else None
                boundary_respect_rate = float(self.context.get('boundary_respect_rate', 0.7)) if self.context.get('boundary_respect_rate') is not None else None
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion indicateurs pour {self.symbol}: {e}")
                return False
                
                        # Récupération prix actuel depuis data
            current_price = None
            if self.data:
                # Essayer d'abord la valeur scalaire (format préféré)
                if 'close' in self.data and self.data['close'] is not None:
                    try:
                        if isinstance(self.data['close'], (int, float)):
                            current_price = float(self.data['close'])
                        elif isinstance(self.data['close'], list) and len(self.data['close']) > 0:
                            current_price = self._get_current_price()
                    except (IndexError, ValueError, TypeError):
                        pass
                
                # Fallback: essayer current_price dans le contexte
                if current_price is None:
                    current_price = self.context.get('current_price')
                    if current_price is not None:
                        try:
                            current_price = float(current_price)
                        except (ValueError, TypeError):
                            current_price = None
                
                # Fallback: essayer current_price dans le contexte
                if current_price is None:
                    current_price = self.context.get('current_price')
                    if current_price is not None:
                        try:
                            current_price = float(current_price)
                        except (ValueError, TypeError):
                            current_price = None
                    
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', 'Unknown')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side or current_price is None:
                logger.warning(f"{self.name}: Signal side ou prix manquant pour {self.symbol}")
                return False
                
            # 1. Calcul automatique des paramètres de range si manquants
            if range_high is None or range_low is None:
                auto_range = self._detect_current_range(current_price)
                if auto_range:
                    range_high, range_low, range_width, range_center = auto_range
                else:
                    logger.debug(f"{self.name}: Impossible de détecter un range pour {self.symbol}")
                    return True  # Ne pas bloquer si pas de range détecté
            
            # Calcul position dans range si manquante
            if range_position is None and range_high and range_low:
                if range_high > range_low:
                    range_position = (current_price - range_low) / (range_high - range_low)
                else:
                    range_position = 0.5
                    
            # 2. Validation largeur du range
            if range_width is not None:
                range_width_ratio = range_width / current_price if current_price > 0 else 0
                
                if range_width_ratio < self.min_range_width:
                    logger.debug(f"{self.name}: Range trop étroit ({self._safe_format(range_width_ratio*100, '.2f')}%) pour {self.symbol}")
                    if signal_confidence < 0.8:
                        return False
                elif range_width_ratio > self.max_range_width:
                    logger.debug(f"{self.name}: Range trop large ({self._safe_format(range_width_ratio*100, '.2f')}%) pour {self.symbol}")
                    if signal_confidence < 0.6:
                        return False
                        
            # 3. Validation age du range
            if range_age_bars is not None:
                if range_age_bars < self.range_age_min:
                    logger.debug(f"{self.name}: Range trop récent ({range_age_bars} barres) pour {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                elif range_age_bars > self.range_age_max:
                    logger.debug(f"{self.name}: Range trop ancien ({range_age_bars} barres) pour {self.symbol}")
                    if signal_confidence < 0.5:
                        return False
                        
            # 4. Validation position dans le range
            if range_position is not None:
                # Position trop près des bords
                if range_position < self.range_edge_zone or range_position > (1 - self.range_edge_zone):
                    logger.debug(f"{self.name}: Position près du bord du range ({self._safe_format(range_position, '.2f')}) pour {self.symbol}")
                    if not self._is_breakout_strategy(signal_strategy):
                        if signal_confidence < 0.7:
                            return False
                            
                # Position dans zone dangereuse selon direction signal
                if signal_side == "BUY" and range_position > 0.8:
                    logger.debug(f"{self.name}: BUY signal près du haut du range ({self._safe_format(range_position, '.2f')}) pour {self.symbol}")
                    if signal_confidence < 0.6:
                        return False
                elif signal_side == "SELL" and range_position < 0.2:
                    logger.debug(f"{self.name}: SELL signal près du bas du range ({self._safe_format(range_position, '.2f')}) pour {self.symbol}")
                    if signal_confidence < 0.6:
                        return False
                        
            # 5. Validation probabilité de breakout
            if breakout_probability is not None and breakout_probability > self.breakout_probability_max:
                logger.debug(f"{self.name}: Probabilité breakout élevée ({self._safe_format(breakout_probability, '.2f')}) pour {self.symbol}")
                if not self._is_breakout_strategy(signal_strategy):
                    if signal_confidence < 0.8:
                        return False
                        
            # 6. Validation volatilité dans range
            if range_volatility is not None:
                if range_volatility < self.min_range_volatility:
                    logger.debug(f"{self.name}: Volatilité range trop faible ({self._safe_format(range_volatility*100, '.2f')}%) pour {self.symbol}")
                    if signal_confidence < 0.7:
                        return False
                elif range_volatility > self.max_range_volatility:
                    logger.debug(f"{self.name}: Volatilité range excessive ({self._safe_format(range_volatility*100, '.2f')}%) pour {self.symbol}")
                    if signal_confidence < 0.6:
                        return False
                        
            # 7. Validation compression du range
            if range_compression is not None and range_compression > self.range_compression_threshold:
                logger.debug(f"{self.name}: Range en compression ({self._safe_format(range_compression, '.2f')}) pour {self.symbol}")
                # Range compressé peut indiquer un breakout imminent
                if not self._is_breakout_strategy(signal_strategy):
                    if signal_confidence < 0.8:
                        return False
                        
            # 8. Validation tests des bornes
            total_boundary_tests = 0
            if upper_boundary_tests is not None:
                total_boundary_tests += upper_boundary_tests
            if lower_boundary_tests is not None:
                total_boundary_tests += lower_boundary_tests
            if boundary_test_count is not None:
                total_boundary_tests = max(total_boundary_tests, boundary_test_count)
                
            if total_boundary_tests > 0 and total_boundary_tests < self.min_boundary_tests:
                logger.debug(f"{self.name}: Pas assez de tests des bornes ({total_boundary_tests}) pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 9. Validation respect des bornes
            if boundary_respect_rate is not None and boundary_respect_rate < self.boundary_respect_ratio:
                logger.debug(f"{self.name}: Faible respect des bornes ({self._safe_format(boundary_respect_rate, '.2f')}) pour {self.symbol}")
                if signal_confidence < 0.7:
                    return False
                    
            # 10. Validation cohérence stratégie/range selon position
            strategy_range_match = self._validate_strategy_range_match(
                signal_strategy, signal_side, range_position, breakout_probability
            )
            if not strategy_range_match:
                logger.debug(f"{self.name}: Stratégie {signal_strategy} inadaptée aux conditions range pour {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 11. Validation mouvement dans range
            if range_movement_direction:
                movement_coherence = self._validate_movement_coherence(
                    signal_side, range_movement_direction, range_position
                )
                if not movement_coherence:
                    logger.debug(f"{self.name}: Mouvement range incohérent avec signal pour {self.symbol}")
                    if signal_confidence < 0.6:
                        return False
                        
            # 12. Validation temps passé dans range
            if time_in_range is not None and time_in_range > self.range_age_max * 1.5:
                logger.debug(f"{self.name}: Temps excessif dans range ({time_in_range} barres) pour {self.symbol}")
                if signal_confidence < 0.5:
                    return False
                    
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                        f"Range: {self._safe_format(range_low, '.4f') if range_low is not None else 'N/A'}-{self._safe_format(range_high, '.4f') if range_high is not None else 'N/A'}, "
                        f"Position: {self._safe_format(range_position, '.2f') if range_position is not None else 'N/A'}, "
                        f"Largeur: {self._safe_format(range_width_ratio*100, '.2f') if range_width_ratio is not None else 'N/A'}%, "
                        f"Age: {range_age_bars or 'N/A'} barres")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def _detect_current_range(self, current_price: float) -> tuple:
        """Détecte automatiquement le range actuel basé sur les prix récents."""
        try:
            if not self.data or 'high' not in self.data or 'low' not in self.data:
                return None
                
            # Utiliser les 20 dernières barres pour détecter le range
            lookback = min(20, len(self.data['high']))
            if lookback < 5:
                return None
                
            recent_highs = self.data['high'][-lookback:]
            recent_lows = self.data['low'][-lookback:]
            
            range_high = max(recent_highs)
            range_low = min(recent_lows)
            range_width = range_high - range_low
            range_center = (range_high + range_low) / 2
            
            # Vérifier que c'est un range valide
            range_width_ratio = range_width / current_price if current_price > 0 else 0
            if range_width_ratio < self.min_range_width or range_width_ratio > self.max_range_width:
                return None
                
            return (range_high, range_low, range_width, range_center)
            
        except Exception:
            return None
            
    def _is_breakout_strategy(self, strategy_name: str) -> bool:
        """Détermine si la stratégie est de type breakout."""
        breakout_keywords = ['breakout', 'break', 'channel', 'donchian', 'resistance', 'support']
        strategy_lower = strategy_name.lower()
        return any(keyword in strategy_lower for keyword in breakout_keywords)
        
    def _validate_strategy_range_match(self, strategy: str, signal_side: str, 
                                     range_position: float, breakout_prob: float) -> bool:
        """Valide l'adéquation stratégie/conditions de range."""
        strategy_lower = strategy.lower()
        
        # Stratégies de mean reversion (favorisent ranges)
        if any(kw in strategy_lower for kw in ['bollinger', 'rsi', 'reversal', 'touch', 'rebound']):
            # Mean reversion fonctionne mieux dans ranges stables
            if breakout_prob and breakout_prob > 0.8:
                return False  # Breakout imminent
            return True
            
        # Stratégies de breakout
        elif any(kw in strategy_lower for kw in ['breakout', 'donchian', 'channel']):
            # Breakout strategies préfèrent ranges en compression
            if range_position is not None:
                # Accepter près des bords pour breakout
                if (signal_side == "BUY" and range_position > 0.7) or \
                   (signal_side == "SELL" and range_position < 0.3):
                    return True
            return True
            
        # Stratégies trend following
        elif any(kw in strategy_lower for kw in ['macd', 'ema', 'trend', 'cross']):
            # Trend following moins efficace dans ranges
            if range_position is not None and 0.3 <= range_position <= 0.7:
                return False  # Centre du range = pas de trend
            return True
            
        return True  # Par défaut accepter
        
    def _validate_movement_coherence(self, signal_side: str, movement_direction: str, 
                                   range_position: float) -> bool:
        """Valide la cohérence entre mouvement range et signal."""
        try:
            if not movement_direction or range_position is None:
                return True
                
            # Pour BUY signals
            if signal_side == "BUY":
                # Favoriser si mouvement vers haut ou neutre
                if movement_direction == "down" and range_position < 0.3:
                    return False  # Mouvement bas + signal BUY en bas de range
                    
            # Pour SELL signals
            elif signal_side == "SELL":
                # Favoriser si mouvement vers bas ou neutre
                if movement_direction == "up" and range_position > 0.7:
                    return False  # Mouvement haut + signal SELL en haut de range
                    
            return True
            
        except Exception:
            return True
            
    def _get_current_price(self) -> float:
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
            current_price = self.context.get('current_price')
            if current_price is not None:
                try:
                    return float(current_price)
                except (ValueError, TypeError):
                    pass
        return None
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur les conditions de range.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur range
            range_width = float(self.context.get('range_width', 0.02)) if self.context.get('range_width') is not None else 0.02
            range_position = float(self.context.get('range_position', 0.5)) if self.context.get('range_position') is not None else 0.5
            range_strength = float(self.context.get('range_strength', 0.5)) if self.context.get('range_strength') is not None else 0.5
            range_age_bars = int(self.context.get('range_age_bars', 20)) if self.context.get('range_age_bars') is not None else 20
            boundary_test_count = int(self.context.get('boundary_test_count', 2)) if self.context.get('boundary_test_count') is not None else 2
            boundary_respect_rate = float(self.context.get('boundary_respect_rate', 0.7)) if self.context.get('boundary_respect_rate') is not None else 0.7
            breakout_probability = float(self.context.get('breakout_probability', 0.5)) if self.context.get('breakout_probability') is not None else 0.5
            range_volatility = float(self.context.get('range_volatility', 0.02)) if self.context.get('range_volatility') is not None else 0.02
            
            signal_strategy = signal.get('strategy', '')
            
            base_score = 0.5  # Score de base si validé
            
            # Bonus largeur range optimale
            current_price = self._get_current_price() or 1
            range_width_ratio = range_width / current_price if current_price > 0 else 0.02
            
            if abs(range_width_ratio - self.optimal_range_width) <= 0.005:
                base_score += self.optimal_range_bonus
            elif self.min_range_width <= range_width_ratio <= self.max_range_width:
                base_score += 0.10
                
            # Bonus position dans range
            if 0.4 <= range_position <= 0.6:  # Zone centrale
                base_score += self.center_position_bonus
            elif 0.25 <= range_position <= 0.75:  # Zone sûre
                base_score += 0.08
            else:
                # Position près des bords - bonus seulement pour breakout strategies
                if self._is_breakout_strategy(signal_strategy):
                    base_score += 0.10
                else:
                    base_score += self.edge_position_penalty
                    
            # Bonus force et tests des bornes
            if boundary_respect_rate >= 0.8:
                base_score += self.strong_boundaries_bonus
            elif boundary_respect_rate >= 0.7:
                base_score += 0.12
                
            if boundary_test_count >= self.optimal_boundary_tests:
                base_score += 0.12  # Range bien testé
            elif boundary_test_count >= self.min_boundary_tests:
                base_score += 0.08
                
            # Bonus age range approprié
            if self.range_age_min * 2 <= range_age_bars <= self.range_age_max // 2:
                base_score += 0.10  # Age optimal
                
            # Bonus/malus probabilité breakout selon stratégie
            if self._is_breakout_strategy(signal_strategy):
                if breakout_probability >= 0.6:
                    base_score += 0.15  # Bon pour breakout
            else:
                if breakout_probability <= 0.4:
                    base_score += 0.12  # Stable pour mean reversion
                elif breakout_probability >= 0.7:
                    base_score += self.breakout_risk_penalty  # Risque pour non-breakout
                    
            # Bonus volatilité appropriée
            if self.min_range_volatility * 1.5 <= range_volatility <= self.max_range_volatility * 0.7:
                base_score += 0.08  # Volatilité optimale
                
            # Bonus force du range
            if range_strength >= 0.7:
                base_score += 0.12  # Range très solide
            elif range_strength >= 0.5:
                base_score += 0.08  # Range solide
                
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
            
            range_position = float(self.context.get('range_position', 0.5)) if self.context.get('range_position') is not None else None
            range_width = float(self.context.get('range_width', 0)) if self.context.get('range_width') is not None else None
            range_age_bars = int(self.context.get('range_age_bars', 0)) if self.context.get('range_age_bars') is not None else None
            breakout_probability = float(self.context.get('breakout_probability', 0.5)) if self.context.get('breakout_probability') is not None else None
            boundary_respect_rate = float(self.context.get('boundary_respect_rate', 0.7)) if self.context.get('boundary_respect_rate') is not None else None
            
            if is_valid:
                reason = f"Conditions range favorables"
                if range_position is not None:
                    position_desc = "centrale" if 0.4 <= range_position <= 0.6 else "bord" if range_position < 0.2 or range_position > 0.8 else "normale"
                    reason += f" (position {position_desc}: {self._safe_format(range_position, '.2f')})"
                if range_width is not None:
                    current_price = self._get_current_price() or 1
                    width_pct = (range_width / current_price * 100) if current_price > 0 else 0
                    reason += f", largeur: {self._safe_format(width_pct, '.1f')}%"
                if range_age_bars is not None:
                    reason += f", âge: {range_age_bars}b"
                if boundary_respect_rate is not None:
                    reason += f", respect bornes: {self._safe_format(boundary_respect_rate, '.2f')}"
                    
                return f"{self.name}: Validé - {reason} pour {signal_strategy} {signal_side}"
            else:
                current_price = self._get_current_price() or 1
                
                if range_width is not None:
                    width_ratio = range_width / current_price if current_price > 0 else 0
                    if width_ratio < self.min_range_width:
                        return f"{self.name}: Rejeté - Range trop étroit ({self._safe_format(width_ratio*100, '.2f')}%)"
                    elif width_ratio > self.max_range_width:
                        return f"{self.name}: Rejeté - Range trop large ({self._safe_format(width_ratio*100, '.2f')}%)"
                        
                if range_position is not None:
                    if signal_side == "BUY" and range_position > 0.8:
                        return f"{self.name}: Rejeté - BUY près du haut du range ({self._safe_format(range_position, '.2f')})"
                    elif signal_side == "SELL" and range_position < 0.2:
                        return f"{self.name}: Rejeté - SELL près du bas du range ({self._safe_format(range_position, '.2f')})"
                        
                if breakout_probability and breakout_probability > self.breakout_probability_max:
                    return f"{self.name}: Rejeté - Probabilité breakout élevée ({self._safe_format(breakout_probability, '.2f')})"
                    
                if range_age_bars is not None:
                    if range_age_bars < self.range_age_min:
                        return f"{self.name}: Rejeté - Range trop récent ({range_age_bars} barres)"
                    elif range_age_bars > self.range_age_max:
                        return f"{self.name}: Rejeté - Range trop ancien ({range_age_bars} barres)"
                        
                return f"{self.name}: Rejeté - Critères range non respectés"
                
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données de range requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Pour range, on peut fonctionner avec détection automatique
        # Vérifier qu'on a au moins les données OHLC
        required_data = ['high', 'low', 'close']
        for data_type in required_data:
            if not self.data or data_type not in self.data or not self.data[data_type]:
                logger.warning(f"{self.name}: Données {data_type} manquantes pour {self.symbol}")
                return False
                
        # Indicateurs optionnels (si pas présents, détection automatique)
        optional_indicators = [
            'range_high', 'range_low', 'range_position', 'range_width', 'breakout_probability'
        ]
        
        available_indicators = sum(1 for ind in optional_indicators 
                                 if ind in self.context and self.context[ind] is not None)
        
        logger.debug(f"{self.name}: {available_indicators}/{len(optional_indicators)} indicateurs range disponibles pour {self.symbol}")
        
        return True
