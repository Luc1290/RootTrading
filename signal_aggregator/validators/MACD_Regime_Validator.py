"""
MACD_Regime_Validator - Validator basé sur l'état et régime MACD.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class MACD_Regime_Validator(BaseValidator):
    """
    Validator pour le régime MACD - filtre selon l'état momentum et les conditions MACD.
    
    Vérifie: État MACD, croisements, histogram, trend MACD
    Catégorie: technical
    
    Rejette les signaux en:
    - MACD contradictoire avec signal
    - Histogram défavorable
    - Croisements récents opposés
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "MACD_Regime_Validator"
        self.category = "technical"
        
        # Paramètres MACD - OPTIMISÉS POUR RÉDUIRE SURTRADING
        self.min_macd_separation = 0.0005    # Séparation minimum MACD/Signal (5x plus permissif)
        self.histogram_threshold = 0.0003    # Seuil histogram significatif (3x plus permissif)
        self.zero_line_bonus_threshold = 0.003  # Bonus zone favorable (3x plus permissif)
        
        # Paramètres croisements - ALLÉGÉS
        self.recent_cross_penalty_bars = 2   # Pénalité croisement récent réduite
        self.cross_confirmation_strength = 0.001  # Force minimum croisement réduite
        
        # Seuils momentum - PLUS PERMISSIFS
        self.strong_momentum_threshold = 0.005    # MACD momentum fort (2x plus permissif)
        self.weak_momentum_threshold = 0.0002   # MACD momentum faible plus permissif
        self.permissive_mode = True              # NOUVEAU: Mode permissif activé
        
        # Bonus/malus
        self.zero_line_bonus = 0.15          # Bonus position favorable MACD
        self.histogram_bonus = 0.10          # Bonus histogram favorable
        self.cross_confirmation_bonus = 0.20  # Bonus croisement confirmé
        self.contradictory_penalty = -0.30   # Pénalité contradiction MACD
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur l'état du régime MACD.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon MACD, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs MACD depuis le contexte
            try:
                macd_line = float(self.context.get('macd_line', 0)) if self.context.get('macd_line') is not None else None
                macd_signal = float(self.context.get('macd_signal', 0)) if self.context.get('macd_signal') is not None else None
                macd_histogram = float(self.context.get('macd_histogram', 0)) if self.context.get('macd_histogram') is not None else None
                macd_trend = self.context.get('macd_trend')
                macd_zero_cross = self.context.get('macd_zero_cross')
                macd_signal_cross = self.context.get('macd_signal_cross')
                
                # PPO pour confirmation (MACD normalisé)
                ppo = float(self.context.get('ppo', 0)) if self.context.get('ppo') is not None else None
                
                # Context momentum
                momentum_score = float(self.context.get('momentum_score', 50.0)) if self.context.get('momentum_score') is not None else None
                
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion MACD pour {self.symbol}: {e}")
                return False
                
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', 'Unknown')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side or macd_line is None or macd_signal is None:
                logger.warning(f"{self.name}: Signal side ou MACD manquant pour {self.symbol}")
                return False
                
            # 1. Vérification séparation MACD/Signal - MODE PERMISSIF
            macd_separation = abs(macd_line - macd_signal)
            if self.permissive_mode:
                # Mode permissif : accepter même si MACD très proche de Signal
                if macd_separation < self.min_macd_separation and signal_confidence < 0.5:
                    logger.debug(f"{self.name}: MACD très proche Signal mais accepté en mode permissif ({self._safe_format(macd_separation, '.5f')}) pour {self.symbol}")
                    # Ne pas rejeter - laisser passer
            else:
                # Mode strict (ancienne logique)
                if macd_separation < self.min_macd_separation:
                    logger.debug(f"{self.name}: MACD trop proche Signal ({self._safe_format(macd_separation, '.5f')}) pour {self.symbol} - signal peu fiable")
                    if signal_confidence < 0.70:
                        return False
                    
            # 2. Validation direction MACD vs Signal
            macd_above_signal = macd_line > macd_signal
            
            if self.permissive_mode:
                # Mode permissif : accepter divergences MACD/Signal mineures
                if signal_side == "BUY" and not macd_above_signal:
                    # MACD sous Signal mais BUY demandé - peut être anticipation
                    if macd_separation > self.cross_confirmation_strength * 2:  # Seulement si très éloigné
                        logger.debug(f"{self.name}: BUY signal mais MACD très sous Signal pour {self.symbol}")
                        return False
                elif signal_side == "SELL" and macd_above_signal:
                    # MACD au-dessus Signal mais SELL demandé - peut être anticipation
                    if macd_separation > self.cross_confirmation_strength * 2:  # Seulement si très éloigné
                        logger.debug(f"{self.name}: SELL signal mais MACD très au-dessus Signal pour {self.symbol}")
                        return False
            else:
                # Mode strict (ancienne logique)
                if signal_side == "BUY" and not macd_above_signal:
                    logger.debug(f"{self.name}: BUY signal mais MACD sous Signal pour {self.symbol}")
                    if macd_separation > self.cross_confirmation_strength:
                        return False
                elif signal_side == "SELL" and macd_above_signal:
                    logger.debug(f"{self.name}: SELL signal mais MACD au-dessus Signal pour {self.symbol}")
                    if macd_separation > self.cross_confirmation_strength:
                        return False
                    
            # 3. Validation histogram coherence - MODE PERMISSIF
            if macd_histogram is not None and not self.permissive_mode:
                # Seulement en mode strict
                if signal_side == "BUY":
                    if macd_histogram < -self.histogram_threshold and signal_confidence < 0.5:
                        logger.debug(f"{self.name}: BUY signal mais histogram très négatif ({self._safe_format(macd_histogram, '.5f')}) pour {self.symbol}")
                        return False
                elif signal_side == "SELL":
                    if macd_histogram > self.histogram_threshold and signal_confidence < 0.5:
                        logger.debug(f"{self.name}: SELL signal mais histogram très positif ({self._safe_format(macd_histogram, '.5f')}) pour {self.symbol}")
                        return False
            # En mode permissif, ignorer histogram (trop volatile)
                            
            # 4. Validation macd_trend coherence
            if macd_trend:
                if signal_side == "BUY" and macd_trend == "BEARISH":
                    logger.debug(f"{self.name}: BUY signal mais MACD trend bearish pour {self.symbol}")
                    return False
                elif signal_side == "SELL" and macd_trend == "BULLISH":
                    logger.debug(f"{self.name}: SELL signal mais MACD trend bullish pour {self.symbol}")
                    return False
                    
            # 5. Validation croisements récents (macd_signal_cross est BOOLEAN)
            if macd_signal_cross is not None:
                # macd_signal_cross = True indique un croisement récent
                # La direction doit être déduite de macd_line vs macd_signal
                if macd_signal_cross:
                    # Croisement récent détecté - vérifier cohérence avec signal
                    macd_above_signal = macd_line > macd_signal
                    if signal_side == "BUY" and not macd_above_signal:
                        logger.debug(f"{self.name}: BUY signal mais croisement MACD vers le bas récent pour {self.symbol}")
                        if signal_confidence < 0.70:
                            return False
                    elif signal_side == "SELL" and macd_above_signal:
                        logger.debug(f"{self.name}: SELL signal mais croisement MACD vers le haut récent pour {self.symbol}")
                        if signal_confidence < 0.70:
                            return False
                        
            # 6. Validation position zéro line
            zero_line_favorable = self._check_zero_line_position(signal_side, macd_line)
            if not zero_line_favorable and abs(macd_line) > self.zero_line_bonus_threshold:
                # Position défavorable par rapport à zéro
                logger.debug(f"{self.name}: Position MACD défavorable par rapport zéro pour {signal_side} {self.symbol}")
                if signal_confidence < 0.6:
                    return False
                    
            # 7. Validation avec PPO - SEUILS PLUS PERMISSIFS
            if ppo is not None:
                if signal_side == "BUY" and ppo < -0.02:  # Seuil 4x plus permissif (-2% au lieu de -0.5%)
                    logger.debug(f"{self.name}: BUY signal mais PPO extrêmement négatif ({self._safe_format(ppo, '.4f')}) pour {self.symbol}")
                    if signal_confidence < 0.4:  # Seuil confidence plus permissif
                        return False
                elif signal_side == "SELL" and ppo > 0.02:  # Seuil 4x plus permissif (+2% au lieu de +0.5%)
                    logger.debug(f"{self.name}: SELL signal mais PPO extrêmement positif ({self._safe_format(ppo, '.4f')}) pour {self.symbol}")
                    if signal_confidence < 0.4:  # Seuil confidence plus permissif
                        return False
                        
            # 8. Validation momentum score coherence - SEUILS PLUS PERMISSIFS
            if momentum_score is not None:
                if signal_side == "BUY" and momentum_score < 10.0:  # Seulement momentum extrêmement bearish (10 au lieu de 20)
                    logger.debug(f"{self.name}: BUY signal mais momentum extrêmement bearish ({self._safe_format(momentum_score, '.1f')}) pour {self.symbol}")
                    return False
                elif signal_side == "SELL" and momentum_score > 90.0:  # Seulement momentum extrêmement bullish (90 au lieu de 80)
                    logger.debug(f"{self.name}: SELL signal mais momentum extrêmement bullish ({self._safe_format(momentum_score, '.1f')}) pour {self.symbol}")
                    return False
                    
            # 9. Validation spécifique pour stratégies MACD
            if self._is_macd_strategy(signal_strategy):
                # Critères plus stricts pour stratégies basées sur MACD
                if macd_separation < self.cross_confirmation_strength:
                    logger.debug(f"{self.name}: Stratégie MACD mais séparation insuffisante pour {self.symbol}")
                    return False
                    
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - "
                        f"MACD: {self._safe_format(macd_line, '.5f')}, Signal: {self._safe_format(macd_signal, '.5f')}, "
                        f"Histogram: {self._safe_format(macd_histogram, '.5f') if macd_histogram is not None else 'N/A'}, "
                        f"Trend: {macd_trend or 'N/A'}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def _check_zero_line_position(self, signal_side: str, macd_line: float) -> bool:
        """Vérifie si la position MACD par rapport à zéro est favorable."""
        if signal_side == "BUY":
            # Pour BUY, préférable que MACD soit au-dessus de zéro ou proche
            return macd_line > -self.zero_line_bonus_threshold
        elif signal_side == "SELL":
            # Pour SELL, préférable que MACD soit en-dessous de zéro ou proche
            return macd_line < self.zero_line_bonus_threshold
        return True
        
    def _is_macd_strategy(self, strategy_name: str) -> bool:
        """Détermine si la stratégie est spécifiquement basée sur MACD."""
        macd_keywords = ['macd', 'moving_average_convergence', 'ppo']
        strategy_lower = strategy_name.lower()
        return any(keyword in strategy_lower for keyword in macd_keywords)
        
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur l'état du régime MACD.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur MACD
            macd_line = float(self.context.get('macd_line', 0)) if self.context.get('macd_line') is not None else 0
            macd_signal = float(self.context.get('macd_signal', 0)) if self.context.get('macd_signal') is not None else 0
            macd_histogram = float(self.context.get('macd_histogram', 0)) if self.context.get('macd_histogram') is not None else 0
            macd_trend = self.context.get('macd_trend')
            ppo = float(self.context.get('ppo', 0)) if self.context.get('ppo') is not None else None
            
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', '')
            
            base_score = 0.5  # Score de base si validé
            
            # Bonus séparation MACD/Signal
            macd_separation = abs(macd_line - macd_signal)
            if macd_separation >= self.strong_momentum_threshold:
                base_score += 0.15  # Momentum très fort
            elif macd_separation >= self.cross_confirmation_strength:
                base_score += 0.10  # Momentum confirmé
                
            # Bonus histogram favorable
            if signal_side == "BUY" and macd_histogram > self.histogram_threshold:
                base_score += self.histogram_bonus
            elif signal_side == "SELL" and macd_histogram < -self.histogram_threshold:
                base_score += self.histogram_bonus
                
            # Bonus position zéro line favorable
            zero_line_favorable = self._check_zero_line_position(str(signal_side) if signal_side is not None else "", macd_line if macd_line is not None else 0.0)
            if zero_line_favorable:
                if signal_side == "BUY" and macd_line > self.zero_line_bonus_threshold:
                    base_score += self.zero_line_bonus  # MACD bien au-dessus zéro
                elif signal_side == "SELL" and macd_line < -self.zero_line_bonus_threshold:
                    base_score += self.zero_line_bonus  # MACD bien en-dessous zéro
                elif abs(macd_line) <= self.zero_line_bonus_threshold:
                    base_score += 0.05  # Proche de zéro (neutre)
                    
            # Bonus macd_trend coherent
            if macd_trend:
                if (signal_side == "BUY" and macd_trend == "BULLISH") or \
                   (signal_side == "SELL" and macd_trend == "BEARISH"):
                    base_score += 0.15  # Trend MACD confirme signal
                    
            # Bonus PPO coherent
            if ppo is not None:
                if (signal_side == "BUY" and ppo > 0.001) or \
                   (signal_side == "SELL" and ppo < -0.001):
                    base_score += 0.08  # PPO confirme direction
                    
            # Bonus stratégie spécialisée MACD
            if self._is_macd_strategy(signal_strategy):
                base_score += 0.10  # Bonus spécialisation
                
            # Bonus momentum fort (momentum_score est 0-100, 50=neutre)
            momentum_score = self.context.get('momentum_score')
            if momentum_score is not None:
                try:
                    momentum = float(momentum_score)
                    if (signal_side == "BUY" and momentum > 60) or \
                       (signal_side == "SELL" and momentum < 40):
                        base_score += 0.08  # Momentum favorable
                except (ValueError, TypeError):
                    pass
                    
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
            macd_line = float(self.context.get('macd_line', 0)) if self.context.get('macd_line') is not None else 0
            macd_signal = float(self.context.get('macd_signal', 0)) if self.context.get('macd_signal') is not None else 0
            macd_histogram = float(self.context.get('macd_histogram', 0)) if self.context.get('macd_histogram') is not None else None
            macd_trend = self.context.get('macd_trend', 'N/A')
            signal_side = signal.get('side', 'N/A')
            signal_strategy = signal.get('strategy', 'N/A')
            
            if is_valid:
                macd_above = "au-dessus" if macd_line > macd_signal else "en-dessous"
                separation = abs(macd_line - macd_signal)
                
                reason = f"MACD {macd_above} Signal (écart: {self._safe_format(separation, '.5f')})"
                
                if macd_histogram is not None:
                    hist_desc = "positif" if macd_histogram > 0 else "négatif"
                    reason += f", histogram {hist_desc} ({self._safe_format(macd_histogram, '.5f')})"
                    
                if macd_trend != 'N/A':
                    reason += f", trend {macd_trend}"
                    
                zero_desc = "favorable" if self._check_zero_line_position(signal_side, macd_line) else "défavorable"
                reason += f", position zéro {zero_desc}"
                
                if self._is_macd_strategy(signal_strategy):
                    reason += " - stratégie spécialisée"
                    
                return f"{self.name}: Validé - {reason} pour {signal_strategy} {signal_side}"
            else:
                if macd_line is not None and macd_signal is not None:
                    if signal_side == "BUY" and macd_line <= macd_signal:
                        return f"{self.name}: Rejeté - BUY mais MACD ({self._safe_format(macd_line, '.5f')}) <= Signal ({self._safe_format(macd_signal, '.5f')})"
                    elif signal_side == "SELL" and macd_line >= macd_signal:
                        return f"{self.name}: Rejeté - SELL mais MACD ({self._safe_format(macd_line, '.5f')}) >= Signal ({self._safe_format(macd_signal, '.5f')})"
                        
                if macd_trend:
                    if (signal_side == "BUY" and macd_trend == "BEARISH") or \
                       (signal_side == "SELL" and macd_trend == "BULLISH"):
                        return f"{self.name}: Rejeté - Signal {signal_side} contradictoire avec trend MACD {macd_trend}"
                        
                if macd_histogram is not None:
                    if (signal_side == "BUY" and macd_histogram < -self.histogram_threshold) or \
                       (signal_side == "SELL" and macd_histogram > self.histogram_threshold):
                        return f"{self.name}: Rejeté - Histogram MACD défavorable ({self._safe_format(macd_histogram, '.5f')})"
                        
                separation = abs(macd_line - macd_signal) if macd_line is not None and macd_signal is not None else 0
                if separation < self.min_macd_separation:
                    return f"{self.name}: Rejeté - MACD trop proche Signal ({self._safe_format(separation, '.5f')}) - signal peu fiable"
                    
                return f"{self.name}: Rejeté - Conditions MACD non respectées"
                
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données MACD requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Vérification présence MACD essentiels
        required_indicators = ['macd_line', 'macd_signal']
        for indicator in required_indicators:
            if indicator not in self.context or self.context[indicator] is None:
                logger.warning(f"{self.name}: {indicator} manquant pour {self.symbol}")
                return False
                
        return True
