"""
Bollinger_Width_Validator - Validator basé sur la largeur des Bollinger Bands.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class Bollinger_Width_Validator(BaseValidator):
    """
    Validator pour la largeur des Bollinger Bands - filtre selon l'état de volatilité et expansion.
    
    Vérifie: Largeur BB, squeeze/expansion, breakout potentiel
    Catégorie: volatility
    
    Rejette les signaux en:
    - Squeeze trop serré (pas de mouvement attendu)
    - Expansion excessive (volatilité dangereuse)
    - Position prix inappropriée dans les bandes
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        self.name = "Bollinger_Width_Validator"
        self.category = "volatility"
        
        # Paramètres Bollinger Bands - DURCIS pour crypto
        self.min_bb_width = 0.025       # 2.5% largeur minimum (augmenté x2.5)
        self.max_bb_width = 0.12        # 12% largeur maximum (réduit)
        self.squeeze_threshold = 0.04   # 4% seuil squeeze (doublé)
        self.expansion_threshold = 0.10 # 10% seuil expansion (augmenté)
        
        # Paramètres position prix
        self.extreme_position_threshold = 0.95  # Position extrême dans BB
        self.safe_position_min = 0.2           # Zone sûre minimum
        self.safe_position_max = 0.8           # Zone sûre maximum
        
        # Bonus/malus
        self.expansion_bonus = 0.2      # Bonus expansion favorable
        self.squeeze_penalty = -0.2     # Pénalité squeeze
        self.breakout_bonus = 0.25      # Bonus breakout confirmé
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal basé sur la largeur et état des Bollinger Bands.
        
        Args:
            signal: Signal à valider contenant strategy, symbol, side, etc.
            
        Returns:
            True si le signal est valide selon Bollinger Bands, False sinon
        """
        try:
            if not self.validate_data():
                logger.warning(f"{self.name}: Données insuffisantes pour {self.symbol}")
                return False
                
            # Extraction des indicateurs Bollinger depuis le contexte
            try:
                bb_upper = float(self.context.get('bb_upper', 0)) if self.context.get('bb_upper') is not None else None
                bb_middle = float(self.context.get('bb_middle', 0)) if self.context.get('bb_middle') is not None else None
                bb_lower = float(self.context.get('bb_lower', 0)) if self.context.get('bb_lower') is not None else None
                bb_width = float(self.context.get('bb_width', 0)) if self.context.get('bb_width') is not None else None
                bb_position = float(self.context.get('bb_position', 0.5)) if self.context.get('bb_position') is not None else None
                bb_squeeze = self.context.get('bb_squeeze', False)
                bb_expansion = self.context.get('bb_expansion', False)
                bb_breakout_direction = self.context.get('bb_breakout_direction')
            except (ValueError, TypeError) as e:
                logger.warning(f"{self.name}: Erreur conversion Bollinger pour {self.symbol}: {e}")
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
                            current_price = float(self.data['close'][-1])
                    except (IndexError, ValueError, TypeError):
                        pass
                
                # Fallback: current_price n'est pas dans le contexte analyzer_data
                # Le prix actuel vient de self.data['close']
                # if current_price is None:
                #     current_price = self.context.get('current_price')
                #     if current_price is not None:
                #         try:
                #             current_price = float(current_price)
                #         except (ValueError, TypeError):
                #             current_price = None
                    
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', 'Unknown')
            signal_confidence = signal.get('confidence', 0.0)
            
            if not signal_side or bb_width is None:
                logger.warning(f"{self.name}: Signal side ou BB width manquant pour {self.symbol}")
                return False
                
            # bb_width est déjà en format relatif (% du prix moyen)
            bb_width_pct = bb_width
            
            if bb_width_pct is None:
                logger.warning(f"{self.name}: BB width manquant pour {self.symbol}")
                return False
            
            # 1. Vérification largeur BB minimum - PLUS STRICT
            if bb_width_pct < self.min_bb_width:
                logger.debug(f"{self.name}: BB width trop faible ({self._safe_format(bb_width_pct, '.3f')}) pour {self.symbol} - marché compressé")
                # En squeeze sévère, REJETER tous les signaux sauf exceptionnels
                if signal_confidence < 0.9:  # Augmenté de 0.8 à 0.9
                    return False
                    
            # 2. Vérification largeur BB maximum - PLUS STRICT
            if bb_width_pct > self.max_bb_width:
                logger.debug(f"{self.name}: BB width excessive ({self._safe_format(bb_width_pct, '.3f')}) pour {self.symbol} - volatilité dangereuse")
                # Volatilité extrême, REJETER TOUS les signaux
                return False  # Aucune exception en volatilité extrême
                    
            # 3. Gestion état squeeze - BEAUCOUP PLUS STRICT
            if bb_squeeze or bb_width_pct < self.squeeze_threshold:
                logger.debug(f"{self.name}: BB en squeeze pour {self.symbol}")
                # En squeeze, REJETER la plupart des signaux
                if not self._is_breakout_strategy(signal_strategy):
                    if signal_confidence < 0.85:  # Seuil très élevé
                        logger.debug(f"{self.name}: Squeeze - stratégie non-breakout rejetée pour {self.symbol}")
                        return False
                else:
                    # Même pour breakout, exiger confidence élevée
                    if signal_confidence < 0.75:  # NOUVEAU
                        logger.debug(f"{self.name}: Squeeze - même breakout doit être très confiant pour {self.symbol}")
                        return False
                        
            # 4. Gestion état expansion - PLUS STRICT contre mean reversion
            if bb_expansion or bb_width_pct > self.expansion_threshold:
                logger.debug(f"{self.name}: BB en expansion pour {self.symbol}")
                # En expansion, FORTEMENT pénaliser mean reversion
                if self._is_meanreversion_strategy(signal_strategy):
                    logger.debug(f"{self.name}: Expansion - mean reversion fortement pénalisée pour {self.symbol}")
                    if signal_confidence < 0.8:  # Augmenté de 0.6 à 0.8
                        return False
                        
            # 5. Validation position prix dans les bandes
            if bb_position is not None:
                if bb_position > self.extreme_position_threshold:
                    # Prix très proche BB supérieure
                    if signal_side == "BUY":
                        logger.debug(f"{self.name}: BUY signal mais prix près BB supérieure ({self._safe_format(bb_position, '.2f')}) pour {self.symbol}")
                        # BUY près du haut = très risqué, même pour breakout
                        if signal_confidence < 0.9:  # Très strict
                            return False
                elif bb_position < (1 - self.extreme_position_threshold):
                    # Prix très proche BB inférieure
                    if signal_side == "SELL":
                        logger.debug(f"{self.name}: SELL signal mais prix près BB inférieure ({self._safe_format(bb_position, '.2f')}) pour {self.symbol}")
                        # SELL près du bas = très risqué, même pour breakout
                        if signal_confidence < 0.9:  # Très strict
                            return False
                            
            # 6. Validation breakout direction (format: UP/DOWN/NONE)
            if bb_breakout_direction:
                bb_direction = str(bb_breakout_direction).upper()
                if signal_side == "BUY" and bb_direction == "DOWN":
                    logger.debug(f"{self.name}: BUY signal mais breakout direction DOWN pour {self.symbol}")
                    return False
                elif signal_side == "SELL" and bb_direction == "UP":
                    logger.debug(f"{self.name}: SELL signal mais breakout direction UP pour {self.symbol}")
                    return False
                    
            # 7. Validation cohérence prix/bandes si disponible
            if current_price and bb_upper and bb_lower and bb_middle:
                if signal_side == "BUY":
                    # Pour BUY, prix ne doit pas être trop près de BB supérieure
                    distance_to_upper = (bb_upper - current_price) / (bb_upper - bb_middle)
                    if distance_to_upper < 0.1:  # Très proche BB supérieure
                        if not self._is_breakout_strategy(signal_strategy):
                            logger.debug(f"{self.name}: BUY très proche BB supérieure sans stratégie breakout pour {self.symbol}")
                            return False
                            
                elif signal_side == "SELL":
                    # Pour SELL, prix ne doit pas être trop près de BB inférieure
                    distance_to_lower = (current_price - bb_lower) / (bb_middle - bb_lower)
                    if distance_to_lower < 0.1:  # Très proche BB inférieure
                        if not self._is_breakout_strategy(signal_strategy):
                            logger.debug(f"{self.name}: SELL très proche BB inférieure sans stratégie breakout pour {self.symbol}")
                            return False
                            
            # 8. Validation spécifique selon stratégie
            if self._is_bollinger_strategy(signal_strategy):
                # Stratégies spécifiquement basées sur Bollinger
                if bb_width_pct < self.min_bb_width * 2:  # Critères plus stricts
                    logger.debug(f"{self.name}: Stratégie Bollinger mais width insuffisante pour {self.symbol}")
                    return False
                    
            logger.debug(f"{self.name}: Signal validé pour {self.symbol} - BB Width: {self._safe_format(bb_width, '.3f')}, "
                        f"Position: {self._safe_format(bb_position, '.2f') if bb_position is not None else 'N/A'}, "
                        f"Squeeze: {bb_squeeze}, Expansion: {bb_expansion}, "
                        f"Breakout: {bb_breakout_direction or 'N/A'}")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name}: Erreur validation signal pour {self.symbol}: {e}")
            return False
            
    def _is_breakout_strategy(self, strategy_name: str) -> bool:
        """Détermine si la stratégie est de type breakout."""
        breakout_keywords = ['breakout', 'break', 'donchian', 'channel', 'resistance', 'support', 'sweep']
        strategy_lower = strategy_name.lower()
        return any(keyword in strategy_lower for keyword in breakout_keywords)
        
    def _is_meanreversion_strategy(self, strategy_name: str) -> bool:
        """Détermine si la stratégie est de type mean reversion."""
        meanrev_keywords = ['touch', 'reversal', 'rsi', 'oversold', 'overbought', 'rebound']
        strategy_lower = strategy_name.lower()
        return any(keyword in strategy_lower for keyword in meanrev_keywords)
        
    def _is_bollinger_strategy(self, strategy_name: str) -> bool:
        """Détermine si la stratégie est spécifiquement basée sur Bollinger."""
        bollinger_keywords = ['bollinger', 'bb_', 'band']
        strategy_lower = strategy_name.lower()
        return any(keyword in strategy_lower for keyword in bollinger_keywords)
        
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur l'état des Bollinger Bands.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        try:
            if not self.validate_signal(signal):
                return 0.0
                
            # Calcul du score basé sur Bollinger Bands
            bb_width = float(self.context.get('bb_width', 0)) if self.context.get('bb_width') is not None else 0
            bb_position = float(self.context.get('bb_position', 0.5)) if self.context.get('bb_position') is not None else 0.5
            bb_squeeze = self.context.get('bb_squeeze', False)
            bb_expansion = self.context.get('bb_expansion', False)
            bb_breakout_direction = self.context.get('bb_breakout_direction')
            
            base_score = 0.3  # Score de base réduit de 0.5 à 0.3
            
            # Ajustement selon largeur BB (bb_width déjà en format relatif)
            if self.squeeze_threshold <= bb_width <= self.expansion_threshold:
                # Zone optimale de largeur
                optimal_center = (self.squeeze_threshold + self.expansion_threshold) / 2
                distance_from_optimal = abs(bb_width - optimal_center)
                max_distance = (self.expansion_threshold - self.squeeze_threshold) / 2
                
                # Score plus élevé proche du centre optimal - BONUS RÉDUIT
                width_bonus = 0.08 * (1 - distance_from_optimal / max_distance)  # Réduit de 0.15
                base_score += width_bonus
                
            # Bonus/malus selon état BB - PLUS CONSERVATEUR
            if bb_expansion and not bb_squeeze:
                base_score += 0.1  # Réduit de self.expansion_bonus (0.2)
            elif bb_squeeze and not bb_expansion:
                # Squeeze = généralement défavorable
                signal_strategy = signal.get('strategy', '')
                if self._is_breakout_strategy(signal_strategy):
                    base_score += 0.05  # Bonus réduit pour breakout
                else:
                    base_score -= 0.15  # Pénalité augmentée
                    
            # Bonus breakout direction cohérente - RÉDUIT
            if bb_breakout_direction:
                signal_side = signal.get('side')
                bb_direction = str(bb_breakout_direction).upper()
                if (signal_side == "BUY" and bb_direction == "UP") or \
                   (signal_side == "SELL" and bb_direction == "DOWN"):
                    base_score += 0.15  # Réduit de self.breakout_bonus (0.25)
                    
            # CORRECTION: Ajustement selon position prix dans BB + direction signal
            signal_side = signal.get('side')
            signal_strategy = signal.get('strategy', '')
            
            if bb_position is not None:
                # Logique directionnelle pour positions BB
                if signal_side == "BUY":
                    # BUY: Meilleur près de la bande inférieure (prix bas)
                    if bb_position <= 0.25:  # Zone très basse - seuil plus strict
                        base_score += 0.12  # Réduit de 0.15
                    elif bb_position <= 0.4:  # Zone basse élargie
                        base_score += 0.06  # Réduit de 0.08
                    elif bb_position >= 0.75:  # Proche bande supérieure - seuil plus strict
                        if self._is_breakout_strategy(signal_strategy):
                            base_score += 0.05  # Réduit de 0.10
                        else:
                            base_score -= 0.20  # Pénalité augmentée
                            
                elif signal_side == "SELL":
                    # SELL: Meilleur près de la bande supérieure (prix haut)
                    if bb_position >= 0.75:  # Zone très haute - seuil plus strict
                        base_score += 0.12  # Réduit de 0.15
                    elif bb_position >= 0.6:  # Zone haute élargie
                        base_score += 0.06  # Réduit de 0.08
                    elif bb_position <= 0.25:  # Proche bande inférieure - seuil plus strict
                        if self._is_breakout_strategy(signal_strategy):
                            base_score += 0.05  # Réduit de 0.10
                        else:
                            base_score -= 0.20  # Pénalité augmentée
                            
                # Position centrale (0.45-0.55) → neutre pour les deux directions - ZONE RÉDUITE
                if 0.45 <= bb_position <= 0.55:
                    base_score += 0.02  # Bonus réduit pour position neutre
                        
            # Bonus stratégie spécialisée Bollinger - RÉDUIT
            signal_strategy = signal.get('strategy', '')
            if self._is_bollinger_strategy(signal_strategy):
                base_score += 0.05  # Réduit de 0.1 à 0.05
                
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
            bb_width = float(self.context.get('bb_width', 0)) if self.context.get('bb_width') is not None else 0
            bb_position = float(self.context.get('bb_position', 0.5)) if self.context.get('bb_position') is not None else None
            bb_squeeze = self.context.get('bb_squeeze', False)
            bb_expansion = self.context.get('bb_expansion', False)
            bb_breakout_direction = self.context.get('bb_breakout_direction')
            signal_strategy = signal.get('strategy', 'N/A')
            signal_side = signal.get('side', 'N/A')
            
            if is_valid:
                width_desc = "squeeze" if bb_squeeze is not None else "expansion" if bb_expansion is not None else "normale"
                position_desc = f"position {self._safe_format(bb_position, '.2f')}" if bb_position is not None else "N/A"
                breakout_desc = f"breakout {bb_breakout_direction}" if bb_breakout_direction is not None else "pas de breakout"
                
                reason = f"BB {width_desc} (width: {self._safe_format(bb_width, '.3f')}, {position_desc}, {breakout_desc})"
                
                return f"{self.name}: Validé - {reason} pour {signal_strategy} {signal_side}"
            else:
                if bb_width < self.min_bb_width:
                    return f"{self.name}: Rejeté - BB width trop faible ({self._safe_format(bb_width, '.3f')}) - marché compressé"
                elif bb_width > self.max_bb_width:
                    return f"{self.name}: Rejeté - BB width excessive ({self._safe_format(bb_width, '.3f')}) - volatilité dangereuse"
                elif bb_squeeze and signal.get('confidence', 0) < 0.7:
                    return f"{self.name}: Rejeté - Squeeze BB + confidence insuffisante"
                elif bb_breakout_direction:
                    bb_direction = str(bb_breakout_direction).upper()
                    if (signal_side == "BUY" and bb_direction == "DOWN") or \
                       (signal_side == "SELL" and bb_direction == "UP"):
                        return f"{self.name}: Rejeté - Signal {signal_side} contradictoire avec breakout {bb_direction}"
                elif bb_position and bb_position > self.extreme_position_threshold:
                    return f"{self.name}: Rejeté - Position extrême BB supérieure ({self._safe_format(bb_position, '.2f')})"
                elif bb_position and bb_position < (1 - self.extreme_position_threshold):
                    return f"{self.name}: Rejeté - Position extrême BB inférieure ({self._safe_format(bb_position, '.2f')})"
                    
                return f"{self.name}: Rejeté - Critères Bollinger Bands non respectés"
                
        except Exception as e:
            return f"{self.name}: Erreur évaluation - {e}"
            
    def validate_data(self) -> bool:
        """
        Valide que les données Bollinger Bands requises sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not super().validate_data():
            return False
            
        # Vérification présence BB width (indicateur principal)
        if 'bb_width' not in self.context or self.context['bb_width'] is None:
            logger.warning(f"{self.name}: BB_width manquant pour {self.symbol}")
            return False
            
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
            
            # Fallback: pas de current_price dans analyzer_data
            # current_price n'est pas disponible dans analyzer_data
        return 0.0  # Retourner 0.0 au lieu de None pour correspondre au type de retour float
