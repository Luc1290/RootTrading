"""
Liquidity_Sweep_Validator - Validator basé sur la détection des liquidity sweeps.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class Liquidity_Sweep_Validator(BaseValidator):
    """
    Validator pour les liquidity sweeps - valide les conditions de marché pour les faux breakdowns/breakouts.
    
    Vérifie: Support/résistance, volume spikes, patterns de retournement
    Catégorie: structure
    
    Rejette les signaux en:
    - Support/résistance faibles (pas de liquidity pool)
    - Volume insuffisant pour le sweep
    - Timing inapproprié après sweep
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "Liquidity_Sweep_Validator"
        self.category = "structure"
        
        # Paramètres support/résistance
        self.min_support_strength = 0.3     # Force minimum support/résistance
        self.min_break_probability = 0.2    # Probabilité cassure minimum
        self.max_break_probability = 0.8    # Maximum (au-delà = vraie cassure)
        
        # Paramètres volume
        self.min_volume_spike = 1.5         # Volume 50% au-dessus normale
        self.optimal_volume_spike = 2.5     # Volume optimal pour sweep
        self.min_volume_quality = 0.4       # Qualité volume minimum
        
        # Paramètres timing
        self.max_time_since_sweep = 5       # Max barres depuis sweep
        self.recovery_confirmation_min = 2  # Min barres pour confirmer recovery
        
        # Paramètres prix
        self.min_sweep_distance = 0.002     # 0.2% minimum sweep sous support
        self.max_sweep_distance = 0.02      # 2% maximum (au-delà = vraie cassure)
        self.min_recovery_distance = 0.001  # 0.1% minimum recovery au-dessus
        
        # Bonus/malus
        self.perfect_sweep_bonus = 0.3      # Bonus sweep parfait
        self.volume_confirmation_bonus = 0.2 # Bonus volume élevé
        self.timing_penalty = -0.2          # Pénalité timing tardif
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur les conditions de liquidity sweep.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon liquidity sweep, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs depuis le contexte
            try:
                nearest_support = float(self.context.get('nearest_support', 0)) if self.context.get('nearest_support') is not None else None
                nearest_resistance = float(self.context.get('nearest_resistance', 0)) if self.context.get('nearest_resistance') is not None else None
                support_strength = self._convert_strength_to_score(self.context.get('support_strength')) if self.context.get('support_strength') is not None else None
                resistance_strength = self._convert_strength_to_score(self.context.get('resistance_strength')) if self.context.get('resistance_strength') is not None else None
                break_probability = float(self.context.get('break_probability', 0.5)) if self.context.get('break_probability') is not None else None
                
                # Volume indicators
                volume_ratio = float(self.context.get('volume_ratio', 1.0)) if self.context.get('volume_ratio') is not None else None
                volume_spike_multiplier = float(self.context.get('volume_spike_multiplier', 1.0)) if self.context.get('volume_spike_multiplier') is not None else None
                volume_quality_score = float(self.context.get('volume_quality_score', 0.5)) if self.context.get('volume_quality_score') is not None else None
                trade_intensity = float(self.context.get('trade_intensity', 0.5)) if self.context.get('trade_intensity') is not None else None
                
                # Market context
                market_regime = self.context.get('market_regime')
                volatility_regime = self.context.get('volatility_regime')
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion indicateurs pour {self.symbol}: {e}")
                return False
                
            # Récupération prix actuel depuis data
            current_price = None
            if self.data and 'close' in self.data and self.data['close']:
                try:
                    current_price = float(self.data['close'][-1])
                except (IndexError, ValueError, TypeError):
                    pass
                    
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', 'Unknown')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side or current_price is None:
                logger.warning(f"{self.name}: Signal side ou prix manquant pour {self.symbol}")
                return False
                
            # 1. Vérification que c'est une stratégie de liquidity sweep
            if not self._is_liquidity_sweep_strategy(signal_strategy):
                # Pour autres stratégies, validation moins stricte mais toujours pertinente
                logger.debug(f"{self.name}: Stratégie non-sweep ({signal_strategy}) - validation basique pour {self.symbol}")
                return self._validate_basic_structure(signal, current_price, nearest_support, nearest_resistance)
                
            # 2. Validation spécifique pour liquidity sweep strategies
            if signal_side == "BUY":
                return self._validate_bullish_sweep(
                    signal, current_price, nearest_support, support_strength, 
                    break_probability, volume_ratio, volume_spike_multiplier, volume_quality_score
                )
            elif signal_side == "SELL":
                return self._validate_bearish_sweep(
                    signal, current_price, nearest_resistance, resistance_strength,
                    break_probability, volume_ratio, volume_spike_multiplier, volume_quality_score
                )
                
            return False
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def _is_liquidity_sweep_strategy(self, strategy_name: str) -> bool:
        """Détermine si la stratégie est de type liquidity sweep."""
        sweep_keywords = ['sweep', 'liquidity', 'false_break', 'fakeout', 'trap', 'stop_hunt']
        strategy_lower = strategy_name.lower()
        return any(keyword in strategy_lower for keyword in sweep_keywords)
        
    def _validate_basic_structure(self, signal: Dict[str, Any], current_price: float, 
                                 nearest_support: float, nearest_resistance: float) -> bool:
        """Validation basique de structure pour stratégies non-sweep."""
        signal_side = signal.get('side')
        
        # Vérification basique support/résistance
        if signal_side == "BUY" and nearest_support is not None:
            if current_price < nearest_support * 0.95:  # Trop loin sous support
                logger.debug(f"{self.name}: BUY trop loin sous support pour {self.symbol}")
                return False
        elif signal_side == "SELL" and nearest_resistance is not None:
            if current_price > nearest_resistance * 1.05:  # Trop loin au-dessus résistance
                logger.debug(f"{self.name}: SELL trop loin au-dessus résistance pour {self.symbol}")
                return False
                
        return True
        
    def _validate_bullish_sweep(self, signal: Dict[str, Any], current_price: float,
                               nearest_support: float, support_strength: float,
                               break_probability: float, volume_ratio: float,
                               volume_spike_multiplier: float, volume_quality_score: float) -> bool:
        """Validation spécifique pour liquidity sweep haussier."""
        if nearest_support is None:
            logger.debug(f"{self.name}: Support manquant pour sweep haussier {self.symbol}")
            return False
            
        # 1. Vérification force du support
        if support_strength is not None and support_strength < self.min_support_strength:
            logger.debug(f"{self.name}: Support trop faible ({support_strength:.2f}) pour sweep {self.symbol}")
            return False
            
        # 2. Vérification probabilité de cassure (zone optimale pour sweep)
        if break_probability is not None:
            if break_probability < self.min_break_probability:
                logger.debug(f"{self.name}: Probabilité cassure trop faible ({break_probability:.2f}) pour {self.symbol}")
                return False
            elif break_probability > self.max_break_probability:
                logger.debug(f"{self.name}: Probabilité cassure trop élevée ({break_probability:.2f}) - vraie cassure probable {self.symbol}")
                return False
                
        # 3. Vérification position prix par rapport au support
        if current_price < nearest_support:
            # Prix sous support - vérifier que c'est un sweep récent et en recovery
            sweep_distance = (nearest_support - current_price) / nearest_support
            
            if sweep_distance < self.min_sweep_distance:
                logger.debug(f"{self.name}: Sweep distance trop faible ({sweep_distance*100:.2f}%) pour {self.symbol}")
                return False
            elif sweep_distance > self.max_sweep_distance:
                logger.debug(f"{self.name}: Sweep distance excessive ({sweep_distance*100:.2f}%) - vraie cassure {self.symbol}")
                return False
                
        elif current_price > nearest_support:
            # Prix au-dessus support - vérifier recovery après sweep
            recovery_distance = (current_price - nearest_support) / nearest_support
            
            if recovery_distance < self.min_recovery_distance:
                logger.debug(f"{self.name}: Recovery insuffisante ({recovery_distance*100:.2f}%) pour {self.symbol}")
                return False
                
        # 4. Vérification volume (crucial pour liquidity sweep)
        if volume_ratio is not None and volume_ratio < self.min_volume_spike:
            logger.debug(f"{self.name}: Volume spike insuffisant ({volume_ratio:.1f}x) pour sweep {self.symbol}")
            return False
            
        if volume_quality_score is not None and volume_quality_score < self.min_volume_quality:
            logger.debug(f"{self.name}: Qualité volume insuffisante ({volume_quality_score:.2f}) pour sweep {self.symbol}")
            return False
            
        # 5. Confirmation signal confidence pour sweep
        signal_confidence = signal.get('confidence', 0.0)
        if signal_confidence < 0.6:  # Sweep nécessite confidence élevée
            logger.debug(f"{self.name}: Confidence trop faible ({signal_confidence:.2f}) pour sweep {self.symbol}")
            return False
            
        logger.debug(f"{self.name}: Sweep haussier validé pour {self.symbol} - "
                    f"Support: {nearest_support:.4f}, Prix: {current_price:.4f}, "
                    f"Volume: {volume_ratio:.1f}x, Confidence: {signal_confidence:.2f}")
        
        return True
        
    def _validate_bearish_sweep(self, signal: Dict[str, Any], current_price: float,
                               nearest_resistance: float, resistance_strength: float,
                               break_probability: float, volume_ratio: float,
                               volume_spike_multiplier: float, volume_quality_score: float) -> bool:
        """Validation spécifique pour liquidity sweep baissier."""
        if nearest_resistance is None:
            logger.debug(f"{self.name}: Résistance manquante pour sweep baissier {self.symbol}")
            return False
            
        # 1. Vérification force de la résistance
        if resistance_strength is not None and resistance_strength < self.min_support_strength:
            logger.debug(f"{self.name}: Résistance trop faible ({resistance_strength:.2f}) pour sweep {self.symbol}")
            return False
            
        # 2. Vérification probabilité de cassure
        if break_probability is not None:
            if break_probability < self.min_break_probability:
                logger.debug(f"{self.name}: Probabilité cassure trop faible ({break_probability:.2f}) pour {self.symbol}")
                return False
            elif break_probability > self.max_break_probability:
                logger.debug(f"{self.name}: Probabilité cassure trop élevée ({break_probability:.2f}) - vraie cassure probable {self.symbol}")
                return False
                
        # 3. Vérification position prix par rapport à la résistance
        if current_price > nearest_resistance:
            # Prix au-dessus résistance - vérifier que c'est un sweep récent et en recovery
            sweep_distance = (current_price - nearest_resistance) / nearest_resistance
            
            if sweep_distance < self.min_sweep_distance:
                logger.debug(f"{self.name}: Sweep distance trop faible ({sweep_distance*100:.2f}%) pour {self.symbol}")
                return False
            elif sweep_distance > self.max_sweep_distance:
                logger.debug(f"{self.name}: Sweep distance excessive ({sweep_distance*100:.2f}%) - vraie cassure {self.symbol}")
                return False
                
        elif current_price < nearest_resistance:
            # Prix sous résistance - vérifier recovery après sweep
            recovery_distance = (nearest_resistance - current_price) / nearest_resistance
            
            if recovery_distance < self.min_recovery_distance:
                logger.debug(f"{self.name}: Recovery insuffisante ({recovery_distance*100:.2f}%) pour {self.symbol}")
                return False
                
        # 4. Volume et confidence similaires au sweep haussier
        if volume_ratio is not None and volume_ratio < self.min_volume_spike:
            logger.debug(f"{self.name}: Volume spike insuffisant ({volume_ratio:.1f}x) pour sweep {self.symbol}")
            return False
            
        if volume_quality_score is not None and volume_quality_score < self.min_volume_quality:
            logger.debug(f"{self.name}: Qualité volume insuffisante ({volume_quality_score:.2f}) pour sweep {self.symbol}")
            return False
            
        signal_confidence = signal.get('confidence', 0.0)
        if signal_confidence < 0.6:
            logger.debug(f"{self.name}: Confidence trop faible ({signal_confidence:.2f}) pour sweep {self.symbol}")
            return False
            
        logger.debug(f"{self.name}: Sweep baissier validé pour {self.symbol} - "
                    f"Résistance: {nearest_resistance:.4f}, Prix: {current_price:.4f}, "
                    f"Volume: {volume_ratio:.1f}x, Confidence: {signal_confidence:.2f}")
        
        return True
        
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur la qualité du liquidity sweep.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur liquidity sweep
            support_strength = self._convert_strength_to_score(self.context.get('support_strength')) if self.context.get('support_strength') is not None else 0.5
            resistance_strength = self._convert_strength_to_score(self.context.get('resistance_strength')) if self.context.get('resistance_strength') is not None else 0.5
            break_probability = float(self.context.get('break_probability', 0.5)) if self.context.get('break_probability') is not None else 0.5
            volume_ratio = float(self.context.get('volume_ratio', 1.0)) if self.context.get('volume_ratio') is not None else 1.0
            volume_spike_multiplier = float(self.context.get('volume_spike_multiplier', 1.0)) if self.context.get('volume_spike_multiplier') is not None else 1.0
            volume_quality_score = float(self.context.get('volume_quality_score', 0.5)) if self.context.get('volume_quality_score') is not None else 0.5
            
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', '')
            
            base_score = 0.5  # Score de base si validé
            
            # Bonus force support/résistance
            relevant_strength = support_strength if signal_side == "BUY" else resistance_strength
            if relevant_strength >= 0.7:
                base_score += 0.15  # Support/résistance très fort
            elif relevant_strength >= 0.5:
                base_score += 0.10  # Support/résistance fort
                
            # Bonus probabilité cassure dans zone optimale
            if 0.3 <= break_probability <= 0.7:
                # Zone optimale pour sweep (ni trop sûr, ni trop risqué)
                optimal_center = 0.5
                distance_from_optimal = abs(break_probability - optimal_center)
                max_distance = 0.2
                
                probability_bonus = 0.15 * (1 - distance_from_optimal / max_distance)
                base_score += probability_bonus
                
            # Bonus volume
            if volume_ratio >= self.optimal_volume_spike:
                base_score += self.volume_confirmation_bonus  # Volume exceptionnel
            elif volume_ratio >= self.min_volume_spike:
                base_score += 0.10  # Volume suffisant
                
            # Bonus volume spike multiplier
            if volume_spike_multiplier >= 3.0:
                base_score += 0.10  # Spike très marqué
                
            # Bonus qualité volume
            if volume_quality_score >= 0.8:
                base_score += 0.08  # Volume de très bonne qualité
                
            # Bonus stratégie spécialisée
            if self._is_liquidity_sweep_strategy(signal_strategy):
                base_score += self.perfect_sweep_bonus  # Stratégie spécialisée
                
            # Bonus confidence élevée (important pour sweep)
            signal_confidence = signal.get('confidence', 0.0)
            if signal_confidence >= 0.8:
                base_score += 0.1  # Très bonne confidence
                
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
            
            support_strength = float(self.context.get('support_strength', 0)) if self.context.get('support_strength') is not None else None
            resistance_strength = float(self.context.get('resistance_strength', 0)) if self.context.get('resistance_strength') is not None else None
            break_probability = float(self.context.get('break_probability', 0.5)) if self.context.get('break_probability') is not None else None
            volume_ratio = float(self.context.get('volume_ratio', 1.0)) if self.context.get('volume_ratio') is not None else None
            
            if is_valid:
                relevant_strength = support_strength if signal_side == "BUY" else resistance_strength
                level_type = "support" if signal_side == "BUY" else "résistance"
                
                reason = f"Structure favorable pour sweep"
                if relevant_strength:
                    reason += f" ({level_type}: {relevant_strength:.2f})"
                if break_probability:
                    reason += f", probabilité cassure: {break_probability:.2f}"
                if volume_ratio:
                    reason += f", volume: {volume_ratio:.1f}x"
                    
                if self._is_liquidity_sweep_strategy(signal_strategy):
                    reason += " - stratégie spécialisée"
                    
                return f"{self.name}: Validé - {reason} pour {signal_strategy} {signal_side}"
            else:
                if signal_side == "BUY" and support_strength and support_strength < self.min_support_strength:
                    return f"{self.name}: Rejeté - Support trop faible ({support_strength:.2f})"
                elif signal_side == "SELL" and resistance_strength and resistance_strength < self.min_support_strength:
                    return f"{self.name}: Rejeté - Résistance trop faible ({resistance_strength:.2f})"
                elif break_probability and break_probability > self.max_break_probability:
                    return f"{self.name}: Rejeté - Probabilité cassure trop élevée ({break_probability:.2f}) - vraie cassure"
                elif volume_ratio and volume_ratio < self.min_volume_spike:
                    return f"{self.name}: Rejeté - Volume spike insuffisant ({volume_ratio:.1f}x)"
                elif signal.get('confidence', 0) < 0.6:
                    return f"{self.name}: Rejeté - Confidence insuffisante pour sweep"
                    
                return f"{self.name}: Rejeté - Critères liquidity sweep non respectés"
                
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données de structure requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Pour liquidity sweep, on a besoin au minimum de support ou résistance
        has_support = 'nearest_support' in self.context and self.context['nearest_support'] is not None
        has_resistance = 'nearest_resistance' in self.context and self.context['nearest_resistance'] is not None
        
        if not has_support and not has_resistance:
            logger.warning(f"{self.name}: Ni support ni résistance disponible pour {self.symbol}")
            return False
            
        return True
