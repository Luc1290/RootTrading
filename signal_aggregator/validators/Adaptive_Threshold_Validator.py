"""
Adaptive_Threshold_Validator - Validateur avec seuils adaptatifs par timeframe.
Adapte les seuils RSI, MACD, etc. selon le timeframe pour réduire le bruit.
"""

from typing import Dict, Any, Optional
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class Adaptive_Threshold_Validator(BaseValidator):
    """
    Validateur qui adapte les seuils des indicateurs selon le timeframe.
    
    Principe :
    - 1m : Seuils extrêmes (20/80 RSI) pour éviter le bruit
    - 5m : Seuils standards (30/70 RSI)
    - 15m+ : Seuils conservateurs (35/65 RSI) pour plus de fiabilité
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        
        # Configuration des seuils par timeframe
        self.timeframe_thresholds = {
            '1m': {
                'rsi_oversold': 20,
                'rsi_overbought': 80,
                'stoch_k_oversold': 15,
                'stoch_k_overbought': 85,
                'min_macd_distance': 0.002,
                'min_volume_ratio': 1.5,
                'noise_filter': True
            },
            '3m': {
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'stoch_k_oversold': 20,
                'stoch_k_overbought': 80,
                'min_macd_distance': 0.0015,
                'min_volume_ratio': 1.3,
                'noise_filter': True
            },
            '5m': {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'stoch_k_oversold': 25,
                'stoch_k_overbought': 75,
                'min_macd_distance': 0.001,
                'min_volume_ratio': 1.2,
                'noise_filter': False
            },
            '15m': {
                'rsi_oversold': 35,
                'rsi_overbought': 65,
                'stoch_k_oversold': 30,
                'stoch_k_overbought': 70,
                'min_macd_distance': 0.0008,
                'min_volume_ratio': 1.1,
                'noise_filter': False
            },
            '30m': {
                'rsi_oversold': 35,
                'rsi_overbought': 65,
                'stoch_k_oversold': 30,
                'stoch_k_overbought': 70,
                'min_macd_distance': 0.0005,
                'min_volume_ratio': 1.0,
                'noise_filter': False
            },
            '1h': {
                'rsi_oversold': 40,
                'rsi_overbought': 60,
                'stoch_k_oversold': 35,
                'stoch_k_overbought': 65,
                'min_macd_distance': 0.0005,
                'min_volume_ratio': 1.0,
                'noise_filter': False
            },
            '1d': {
                'rsi_oversold': 40,
                'rsi_overbought': 60,
                'stoch_k_oversold': 35,
                'stoch_k_overbought': 65,
                'min_macd_distance': 0.0003,
                'min_volume_ratio': 1.0,
                'noise_filter': False
            }
        }
        
        # Timeframe par défaut si non détecté
        self.default_timeframe = '5m'
        
    def _get_timeframe_from_signal(self, signal: Dict[str, Any]) -> str:
        """Extrait le timeframe du signal."""
        # Essayer plusieurs sources
        timeframe = signal.get('timeframe')
        if not timeframe:
            timeframe = signal.get('metadata', {}).get('timeframe')
        if not timeframe:
            # Fallback sur le contexte si disponible
            timeframe = self.context.get('timeframe')
        
        return timeframe or self.default_timeframe
        
    def _get_thresholds(self, timeframe: str) -> Dict[str, Any]:
        """Récupère les seuils pour un timeframe donné."""
        return self.timeframe_thresholds.get(timeframe, self.timeframe_thresholds[self.default_timeframe])
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal avec des seuils adaptatifs selon le timeframe.
        
        Args:
            signal: Signal à valider
            
        Returns:
            True si le signal respecte les seuils adaptatifs
        """
        try:
            signal_side = signal.get('side')
            if not signal_side:
                return False
                
            timeframe = self._get_timeframe_from_signal(signal)
            thresholds = self._get_thresholds(timeframe)
            
            # Récupération des indicateurs
            rsi_14 = self.context.get('rsi_14')
            stoch_k = self.context.get('stoch_k')
            macd_line = self.context.get('macd_line')
            macd_signal = self.context.get('macd_signal')
            volume_ratio = self.context.get('volume_ratio')
            atr_14 = self.context.get('atr_14')
            
            # 1. Validation RSI adaptative - LOGIQUE CORRIGÉE (plus de rejet absolu)
            rsi_penalty = 0.0
            if rsi_14 is not None:
                try:
                    rsi_val = float(rsi_14)
                    
                    # NOUVEAU: Logique progressive au lieu de rejet brutal
                    if signal_side == 'BUY':
                        # BUY acceptable si RSI >= seuil neutre minimum
                        if rsi_val < thresholds['rsi_neutral_min']:
                            # RSI trop bas même pour BUY = suspect
                            if rsi_val < 15:  # Extrême dangereux
                                logger.debug(f"Signal BUY rejeté: RSI extrêmement bas {rsi_val:.1f} < 15 ({timeframe})")
                                return False
                            # Sinon, juste une pénalité légère
                            rsi_penalty = 0.05
                        # Pas de rejet si RSI dans la zone acceptable
                        
                    elif signal_side == 'SELL':
                        # SELL acceptable si RSI <= seuil neutre maximum  
                        if rsi_val > thresholds['rsi_neutral_max']:
                            # RSI trop haut même pour SELL = suspect
                            if rsi_val > 85:  # Extrême dangereux
                                logger.debug(f"Signal SELL rejeté: RSI extrêmement haut {rsi_val:.1f} > 85 ({timeframe})")
                                return False
                            # Sinon, juste une pénalité légère
                            rsi_penalty = 0.05
                        # Pas de rejet si RSI dans la zone acceptable
                            
                except (ValueError, TypeError):
                    pass
                    
            # 2. Validation Stochastic K adaptative - LOGIQUE CORRIGÉE
            stoch_penalty = 0.0
            if stoch_k is not None:
                try:
                    stoch_val = float(stoch_k)
                    
                    # NOUVEAU: Plus de rejet brutal, juste pénalités pour valeurs extrêmes
                    if signal_side == 'BUY':
                        if stoch_val > 90:  # Stoch trop haut pour BUY
                            logger.debug(f"Signal BUY pénalisé: Stoch K très haut {stoch_val:.1f} ({timeframe})")
                            stoch_penalty = 0.1  # Pénalité au lieu de rejet
                        elif stoch_val > 80:
                            stoch_penalty = 0.05  # Pénalité légère
                    elif signal_side == 'SELL':
                        if stoch_val < 10:  # Stoch trop bas pour SELL
                            logger.debug(f"Signal SELL pénalisé: Stoch K très bas {stoch_val:.1f} ({timeframe})")
                            stoch_penalty = 0.1  # Pénalité au lieu de rejet
                        elif stoch_val < 20:
                            stoch_penalty = 0.05  # Pénalité légère
                        
                except (ValueError, TypeError):
                    pass
                    
            # 3. Validation MACD distance adaptative - LOGIQUE ASSOUPLIE
            macd_penalty = 0.0
            if macd_line is not None and macd_signal is not None:
                try:
                    macd_distance = abs(float(macd_line) - float(macd_signal))
                    min_distance = thresholds['min_macd_distance']
                    
                    # NOUVEAU: Pénalité progressive au lieu de rejet brutal
                    if macd_distance < min_distance:
                        if macd_distance < min_distance * 0.3:  # Très collées
                            logger.debug(f"Signal rejeté: MACD trop collées {macd_distance:.4f} < {min_distance * 0.3:.4f} ({timeframe})")
                            return False  # Rejet seulement si extrêmement collées
                        else:
                            # Pénalité proportionnelle à la distance
                            distance_ratio = macd_distance / min_distance
                            macd_penalty = 0.1 * (1 - distance_ratio)
                            logger.debug(f"Signal pénalisé: distance MACD faible {macd_distance:.4f} ({timeframe})")
                        
                except (ValueError, TypeError):
                    pass
                    
            # 4. Validation volume adaptative - LOGIQUE ASSOUPLIE
            volume_penalty = 0.0
            if volume_ratio is not None:
                try:
                    vol_ratio = float(volume_ratio)
                    min_vol_ratio = thresholds['min_volume_ratio']
                    
                    # NOUVEAU: Pénalité progressive au lieu de rejet brutal
                    if vol_ratio < min_vol_ratio:
                        if vol_ratio < min_vol_ratio * 0.5:  # Volume très faible
                            logger.debug(f"Signal rejeté: volume extrêmement faible {vol_ratio:.2f} < {min_vol_ratio * 0.5:.2f} ({timeframe})")
                            return False  # Rejet seulement si volume dérisoire
                        else:
                            # Pénalité proportionnelle au volume
                            volume_ratio_penalty = (min_vol_ratio - vol_ratio) / min_vol_ratio
                            volume_penalty = 0.15 * volume_ratio_penalty
                            logger.debug(f"Signal pénalisé: volume faible {vol_ratio:.2f} < {min_vol_ratio:.2f} ({timeframe})")
                        
                except (ValueError, TypeError):
                    pass
                    
            # 5. Filtre anti-bruit pour timeframes courts
            if thresholds['noise_filter'] and atr_14 is not None:
                try:
                    atr_val = float(atr_14)
                    current_price = self.context.get('current_price')
                    
                    if current_price:
                        price_val = float(current_price)
                        atr_percentage = (atr_val / price_val) * 100
                        
                        # Si ATR trop faible, c'est du bruit
                        min_atr_percentage = 0.1 if timeframe == '1m' else 0.05
                        if atr_percentage < min_atr_percentage:
                            logger.debug(f"Signal rejeté: ATR trop faible {atr_percentage:.3f}% < {min_atr_percentage}% ({timeframe})")
                            return False
                            
                except (ValueError, TypeError):
                    pass
            
            # NOUVEAU: Validation avec pénalités accumulées au lieu de rejets brutaux
            total_penalty = rsi_penalty + stoch_penalty + macd_penalty + volume_penalty
            confidence_adjustment = thresholds.get('confidence_penalty', 0.0)
            
            # Stocker les pénalités pour usage dans get_validation_score
            if not hasattr(self, '_current_penalties'):
                self._current_penalties = {}
            self._current_penalties[signal.get('strategy', 'unknown')] = {
                'total_penalty': total_penalty,
                'confidence_adjustment': confidence_adjustment,
                'timeframe': timeframe
            }
            
            return True  # Plus de rejets brutaux, tout passe avec pénalités
            
        except Exception as e:
            logger.error(f"Erreur validation seuils adaptatifs: {e}")
            return True  # En cas d'erreur, on laisse passer
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score basé sur la conformité aux seuils adaptatifs.
        
        Returns:
            Score entre 0 et 1
        """
        try:
            signal_side = signal.get('side')
            if not signal_side:
                return 0.0
                
            timeframe = self._get_timeframe_from_signal(signal)
            thresholds = self._get_thresholds(timeframe)
            
            score = 0.5
            factors_checked = 0
            
            # Score RSI
            rsi_14 = self.context.get('rsi_14')
            if rsi_14 is not None:
                try:
                    rsi_val = float(rsi_14)
                    factors_checked += 1
                    
                    if signal_side == 'BUY':
                        # Plus le RSI est bas (survente), meilleur le score
                        if rsi_val <= thresholds['rsi_oversold']:
                            extremity = (thresholds['rsi_oversold'] - rsi_val) / thresholds['rsi_oversold']
                            score += 0.3 * extremity
                        else:
                            score -= 0.2
                    elif signal_side == 'SELL':
                        # Plus le RSI est haut (surachat), meilleur le score
                        if rsi_val >= thresholds['rsi_overbought']:
                            extremity = (rsi_val - thresholds['rsi_overbought']) / (100 - thresholds['rsi_overbought'])
                            score += 0.3 * extremity
                        else:
                            score -= 0.2
                            
                except (ValueError, TypeError):
                    pass
                    
            # Score volume
            volume_ratio = self.context.get('volume_ratio')
            if volume_ratio is not None:
                try:
                    vol_ratio = float(volume_ratio)
                    min_vol = thresholds['min_volume_ratio']
                    factors_checked += 1
                    
                    if vol_ratio >= min_vol:
                        # Bonus selon l'excès de volume
                        volume_bonus = min(0.2, (vol_ratio - min_vol) * 0.1)
                        score += volume_bonus
                    else:
                        score -= 0.15
                        
                except (ValueError, TypeError):
                    pass
                    
            # Ajustement selon le timeframe (favoriser les TF longs)
            tf_bonus = {
                '1m': -0.1,   # Pénalité pour le bruit
                '3m': -0.05,
                '5m': 0.0,    # Neutre
                '15m': 0.05,  # Léger bonus
                '30m': 0.1,   # Bonus
                '1h': 0.1,
                '1d': 0.15
            }.get(timeframe, 0.0)
            
            score += tf_bonus
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Erreur calcul score seuils adaptatifs: {e}")
            return 0.5
            
    def get_validation_reason(self, signal: Dict[str, Any], is_valid: bool) -> str:
        """
        Fournit une explication de la validation adaptative.
        
        Returns:
            Raison de la validation/invalidation
        """
        try:
            signal_side = signal.get('side', 'N/A')
            timeframe = self._get_timeframe_from_signal(signal)
            thresholds = self._get_thresholds(timeframe)
            
            rsi_14 = self.context.get('rsi_14')
            volume_ratio = self.context.get('volume_ratio')
            
            if is_valid:
                reason_parts = [f"Signal {signal_side} conforme seuils {timeframe}"]
                
                if rsi_14 is not None:
                    try:
                        rsi_val = float(rsi_14)
                        if signal_side == 'BUY' and rsi_val <= thresholds['rsi_oversold']:
                            reason_parts.append(f"RSI {rsi_val:.1f} <= {thresholds['rsi_oversold']}")
                        elif signal_side == 'SELL' and rsi_val >= thresholds['rsi_overbought']:
                            reason_parts.append(f"RSI {rsi_val:.1f} >= {thresholds['rsi_overbought']}")
                    except (ValueError, TypeError):
                        pass
                        
                if volume_ratio is not None:
                    try:
                        vol_ratio = float(volume_ratio)
                        if vol_ratio >= thresholds['min_volume_ratio']:
                            reason_parts.append(f"Vol {vol_ratio:.1f}x >= {thresholds['min_volume_ratio']}")
                    except (ValueError, TypeError):
                        pass
                        
                return " | ".join(reason_parts)
            else:
                return f"Signal {signal_side} non conforme seuils {timeframe} " \
                       f"(RSI {rsi_14}, Vol {volume_ratio})"
                       
        except Exception as e:
            return f"Erreur validation adaptative: {str(e)}"