"""
Global_Trend_Validator - Validateur de tendance globale strict.
Vérifie que les signaux sont alignés avec la tendance principale du marché.
"""

from typing import Dict, Any, Optional
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class Global_Trend_Validator(BaseValidator):
    """
    Validateur qui vérifie l'alignement des signaux avec la tendance globale.
    
    Rejette les signaux contra-trend en tendance forte:
    - Rejette les BUY en forte tendance baissière
    - Rejette les SELL en forte tendance haussière
    - Plus permissif en marché en range
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        
        # Seuils de validation - OPTIMISÉS POUR RÉDUIRE SURTRADING
        self.min_trend_alignment_bull = 0.3   # Moins strict: 0.3 au lieu de 0.5  
        self.min_trend_alignment_bear = -0.3  # Moins strict: -0.3 au lieu de -0.5
        self.min_adx_trend = 30              # Plus strict: 30 au lieu de 25 (tendance vraiment forte)
        self.min_regime_confidence = 75      # Plus strict: 75% au lieu de 60% (confiance vraiment élevée)
        # Filtres EMA adaptés
        self.use_ema_filter = True          # Activer le filtre EMA50/EMA99
        self.ema_divergence_penalty = 0.3   # Pénalité réduite: 0.3 au lieu de 0.5
        self.ranging_mode_permissive = True  # NOUVEAU: Mode permissif en ranging
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide que le signal est aligné avec la tendance globale.
        
        Args:
            signal: Signal à valider
            
        Returns:
            True si le signal est aligné avec la tendance, False sinon
        """
        try:
            signal_side = signal.get('side')
            if not signal_side:
                return False
                
            # Récupération des indicateurs de tendance
            market_regime = self.context.get('market_regime')
            regime_strength = self.context.get('regime_strength')
            regime_confidence = self.context.get('regime_confidence', 0)
            trend_alignment = self.context.get('trend_alignment', 0)
            adx_value = self.context.get('adx_14', 0)
            trend_strength = self.context.get('trend_strength')
            
            # En marché en range - LOGIQUE OPTIMISÉE
            if market_regime == 'RANGING':
                if self.ranging_mode_permissive:
                    # NOUVEAU: Mode permissif en ranging - accepter BUY et SELL
                    # C'est l'essence du trading en range !
                    logger.debug(f"Signal {signal_side} accepté en mode RANGING permissif")
                    return True
                else:
                    # Ancienne logique strict (gardée pour compatibilité)
                    ema_50 = self.context.get('ema_50')
                    ema_99 = self.context.get('ema_99')
                    current_price = self.context.get('current_price')
                    
                    if self.use_ema_filter and ema_50 and ema_99 and current_price:
                        try:
                            ema50_val = float(ema_50)
                            ema99_val = float(ema_99)
                            price_val = float(current_price)
                            
                            # Seuil de séparation EMA pour éviter les faux signaux
                            ema_separation = abs(ema50_val - ema99_val) / max(ema50_val, ema99_val)
                            
                            # Si EMA trop proches = vrai range, accepter les deux
                            if ema_separation < 0.01:  # EMA à moins de 1% l'une de l'autre
                                return True
                            
                            # Sinon logique directionnelle soft
                            if ema50_val > ema99_val and signal_side == 'SELL':
                                logger.debug(f"Signal SELL rejeté en RANGING: tendance EMA haussière (sep: {ema_separation*100:.1f}%)")
                                return False
                            elif ema50_val < ema99_val and signal_side == 'BUY':
                                logger.debug(f"Signal BUY rejeté en RANGING: tendance EMA baissière (sep: {ema_separation*100:.1f}%)")
                                return False
                                
                        except (ValueError, TypeError):
                            pass
                
            # Vérifier la force de la tendance
            is_strong_trend = False
            if adx_value and adx_value >= self.min_adx_trend:
                is_strong_trend = True
            elif regime_confidence and regime_confidence >= self.min_regime_confidence:
                is_strong_trend = True
            elif trend_strength and str(trend_strength).lower() in ['strong', 'very_strong']:
                is_strong_trend = True
                
            # Si pas de tendance forte - LOGIQUE ALLÉGÉE
            if not is_strong_trend:
                # NOUVEAU: Sans tendance forte, être plus permissif
                # Les stratégies ont déjà leurs propres filtres
                ema_50 = self.context.get('ema_50')
                ema_99 = self.context.get('ema_99')
                current_price = self.context.get('current_price')
                
                if self.use_ema_filter and ema_50 and ema_99 and current_price:
                    try:
                        ema50_val = float(ema_50)
                        ema99_val = float(ema_99)
                        price_val = float(current_price)
                        
                        # Seuil de divergence extrême pour rejeter
                        price_ema50_distance = abs(price_val - ema50_val) / ema50_val
                        
                        # Rejeter seulement les signaux VRAIMENT contra-trend
                        if (price_val < ema50_val * 0.98 and ema50_val < ema99_val * 0.995 and signal_side == 'BUY') or \
                           (price_val > ema50_val * 1.02 and ema50_val > ema99_val * 1.005 and signal_side == 'SELL'):
                            # Prix très éloigné de l'EMA dans le mauvais sens
                            logger.debug(f"Signal {signal_side} rejeté: divergence EMA extrême (prix: {price_val:.2f}, EMA50: {ema50_val:.2f})")
                            return False
                            
                    except (ValueError, TypeError):
                        pass
                        
                return True
                
            # Filtre EMA 50/99 pour tendance de fond
            ema_50 = self.context.get('ema_50')
            ema_99 = self.context.get('ema_99')
            current_price = self.context.get('current_price')
            
            ema_trend_up = False
            ema_trend_down = False
            
            if self.use_ema_filter and ema_50 and ema_99 and current_price:
                try:
                    ema50_val = float(ema_50)
                    ema99_val = float(ema_99)
                    price_val = float(current_price)
                    
                    # Déterminer la tendance EMA
                    if ema50_val > ema99_val and price_val > ema50_val:
                        ema_trend_up = True
                    elif ema50_val < ema99_val and price_val < ema50_val:
                        ema_trend_down = True
                        
                    # Rejeter signaux fortement contra-EMA
                    if ema_trend_down and signal_side == 'BUY':
                        # Prix sous EMA50 < EMA99 = forte tendance baissière
                        logger.debug(f"Signal BUY rejeté: prix {price_val:.2f} sous EMA50 {ema50_val:.2f} < EMA99 {ema99_val:.2f}")
                        return False
                    elif ema_trend_up and signal_side == 'SELL':
                        # Prix au-dessus EMA50 > EMA99 = forte tendance haussière
                        logger.debug(f"Signal SELL rejeté: prix {price_val:.2f} au-dessus EMA50 {ema50_val:.2f} > EMA99 {ema99_val:.2f}")
                        return False
                        
                except (ValueError, TypeError):
                    pass
            
            # Vérification stricte en tendance forte (gardée en complément)
            bullish_regimes = ['TRENDING_BULL', 'BREAKOUT_BULL', 'BULLISH']
            bearish_regimes = ['TRENDING_BEAR', 'BREAKOUT_BEAR', 'BEARISH']
            
            if market_regime in bullish_regimes or trend_alignment > self.min_trend_alignment_bull:
                # Tendance haussière forte : rejeter les SELL
                if signal_side == 'SELL':
                    logger.debug(f"Signal SELL rejeté en tendance haussière forte "
                               f"(regime={market_regime}, alignment={trend_alignment:.1f})")
                    return False
                    
            elif market_regime in bearish_regimes or trend_alignment < self.min_trend_alignment_bear:
                # Tendance baissière forte : rejeter les BUY
                if signal_side == 'BUY':
                    logger.debug(f"Signal BUY rejeté en tendance baissière forte "
                               f"(regime={market_regime}, alignment={trend_alignment:.1f})")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation tendance globale: {e}")
            return False  # En cas d'erreur, on rejette par sécurité
            
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation basé sur l'alignement avec la tendance.
        
        Returns:
            Score entre 0 et 1
        """
        try:
            signal_side = signal.get('side')
            if not signal_side:
                return 0.0
                
            market_regime = self.context.get('market_regime')
            trend_alignment = self.context.get('trend_alignment', 0)
            regime_confidence = self.context.get('regime_confidence', 50)
            adx_value = self.context.get('adx_14', 20)
            
            # Filtre EMA pour le score
            ema_50 = self.context.get('ema_50')
            ema_99 = self.context.get('ema_99')
            current_price = self.context.get('current_price')
            
            base_score = 0.5
            ema_penalty = 0.0
            
            # Pénalité EMA si contra-tendance
            if self.use_ema_filter and ema_50 and ema_99 and current_price:
                try:
                    ema50_val = float(ema_50)
                    ema99_val = float(ema_99)
                    price_val = float(current_price)
                    
                    # EMA50 > EMA99 et prix > EMA50 = tendance haussière
                    if ema50_val > ema99_val and price_val > ema50_val and signal_side == 'SELL':
                        ema_penalty = self.ema_divergence_penalty  # -0.5
                    # EMA50 < EMA99 et prix < EMA50 = tendance baissière  
                    elif ema50_val < ema99_val and price_val < ema50_val and signal_side == 'BUY':
                        ema_penalty = self.ema_divergence_penalty  # -0.5
                        
                except (ValueError, TypeError):
                    pass
            
            # Score basé sur l'alignement
            bullish_regimes = ['TRENDING_BULL', 'BREAKOUT_BULL', 'BULLISH']
            bearish_regimes = ['TRENDING_BEAR', 'BREAKOUT_BEAR', 'BEARISH']
            
            if market_regime == 'RANGING':
                # En range, score plus élevé - c'est l'environnement idéal pour oscillateurs
                base_score = 0.75  # Augmenté de 0.6 à 0.75
            elif (market_regime in bullish_regimes and signal_side == 'BUY') or \
                 (market_regime in bearish_regimes and signal_side == 'SELL'):
                # Parfaitement aligné
                base_score = 1.0
                
                # Bonus selon la force de l'alignement (format décimal)
                alignment_abs = abs(trend_alignment)
                if alignment_abs > 0.5:
                    base_score = 1.0
                elif alignment_abs > 0.3:
                    base_score = 0.9
                else:
                    base_score = 0.8
                    
            elif (market_regime in bullish_regimes and signal_side == 'SELL') or \
                 (market_regime in bearish_regimes and signal_side == 'BUY'):
                # Contra-trend
                base_score = 0.0
                
                # Si la tendance n'est pas très forte, on peut donner un petit score
                if adx_value < 30 and regime_confidence < 70:
                    base_score = 0.3
                elif adx_value < 25 and regime_confidence < 60:
                    base_score = 0.4
                    
            # Ajustement selon ADX
            if adx_value:
                if adx_value > 40:  # Très forte tendance
                    if base_score >= 0.8:
                        base_score = min(1.0, base_score * 1.1)
                    else:
                        base_score = max(0.0, base_score * 0.8)
                elif adx_value < 20:  # Tendance faible
                    base_score = max(0.5, base_score * 0.9)
            
            # Appliquer la pénalité EMA
            final_score = base_score - ema_penalty
                    
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            logger.error(f"Erreur calcul score tendance globale: {e}")
            return 0.5
            
    def get_validation_reason(self, signal: Dict[str, Any], is_valid: bool) -> str:
        """
        Fournit une explication de la validation.
        
        Returns:
            Raison de la validation/invalidation
        """
        try:
            signal_side = signal.get('side', 'N/A')
            market_regime = self.context.get('market_regime', 'UNKNOWN')
            trend_alignment = self.context.get('trend_alignment', 0)
            adx_value = self.context.get('adx_14', 0)
            regime_confidence = self.context.get('regime_confidence', 0)
            
            bullish_regimes = ['TRENDING_BULL', 'BREAKOUT_BULL', 'BULLISH']
            bearish_regimes = ['TRENDING_BEAR', 'BREAKOUT_BEAR', 'BEARISH']
            
            if is_valid:
                if market_regime == 'RANGING':
                    return f"Signal {signal_side} accepté en marché en range"
                elif (market_regime in bullish_regimes and signal_side == 'BUY') or \
                     (market_regime in bearish_regimes and signal_side == 'SELL'):
                    return f"Signal {signal_side} aligné avec tendance {market_regime} " \
                           f"(alignment={trend_alignment:.1f}, ADX={adx_value:.1f})"
                else:
                    return f"Signal {signal_side} accepté, tendance faible " \
                           f"(ADX={adx_value:.1f}, confidence={regime_confidence:.0f}%)"
            else:
                return f"Signal {signal_side} contra-trend rejeté en {market_regime} " \
                       f"(alignment={trend_alignment:.1f}, ADX={adx_value:.1f})"
                       
        except Exception as e:
            return f"Erreur validation tendance: {str(e)}"