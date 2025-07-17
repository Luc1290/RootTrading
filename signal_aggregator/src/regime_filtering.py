#!/usr/bin/env python3
"""
Module pour le filtrage des signaux basé sur les régimes de marché.
Contient la logique de filtrage adaptative selon les conditions de marché.
"""

import logging
from typing import Dict, Any, Optional, List
from enhanced_regime_detector import MarketRegime

logger = logging.getLogger(__name__)


class RegimeFiltering:
    """Classe pour le filtrage des signaux basé sur les régimes de marché"""
    
    # Strategy groupings by market condition
    STRATEGY_GROUPS = {
        'trend': ['EMA_Cross', 'MACD', 'Breakout'],
        'mean_reversion': ['Bollinger', 'RSI', 'Divergence'],
        'adaptive': ['Ride_or_React']
    }
    
    def __init__(self, technical_analysis=None):
        self.technical_analysis = technical_analysis
    
    async def apply_enhanced_regime_filtering(self, signal: Dict[str, Any], regime, regime_metrics: Dict[str, float], 
                                             is_ultra_confluent: bool, signal_score: Optional[float], 
                                             strategy_count: int = 1) -> bool:
        """
        Applique un filtrage intelligent basé sur les régimes Enhanced.
        
        Args:
            signal: Signal à filtrer
            regime: Régime Enhanced détecté
            regime_metrics: Métriques du régime
            is_ultra_confluent: Si le signal est ultra-confluent
            signal_score: Score du signal (si disponible)
            strategy_count: Nombre de stratégies qui s'accordent sur ce signal
            
        Returns:
            True si le signal doit être accepté, False sinon
        """
        try:
            symbol = signal['symbol']
            signal_strength = signal.get('strength', 'moderate')
            signal_confidence = signal.get('confidence', 0.5)
            strategy = signal.get('strategy', 'Unknown')
            # Normaliser le nom de stratégie (retirer _Strategy)
            strategy = strategy.replace('_Strategy', '')
            side = signal.get('side', 'UNKNOWN')
            
            # NOUVEAU: Récupérer données techniques pour validation Enhanced
            technical_context = await self.technical_analysis.get_technical_context(symbol) if self.technical_analysis else {}
            
            # Seuils adaptatifs selon le régime Enhanced + contexte technique
            if regime.name == 'STRONG_TREND_UP':
                # Tendance haussière forte: ATTENTION - éviter d'acheter au sommet
                if side == 'SELL':
                    # En forte tendance haussière, c'est le moment de prendre des profits
                    min_confidence = 0.65  # Plus permissif pour SELL (prise de profits)
                    required_strength = ['moderate', 'strong', 'very_strong']
                    logger.debug(f"💪 {regime.name}: SELL favorisé pour prise de profits pour {symbol}")
                else:  # BUY
                    # En forte tendance haussière, éviter d'acheter (prix déjà haut)
                    min_confidence = 0.85  # Très strict pour BUY quand prix très haut
                    required_strength = ['very_strong']
                    logger.debug(f"💪 {regime.name}: BUY fortement pénalisé (prix déjà haut) pour {symbol}")
                
            elif regime.name == 'TREND_UP':
                # Tendance haussière modérée: équilibrer BUY/SELL mais attention au prix
                if side == 'SELL':
                    min_confidence = 0.70  # Modérément permissif pour SELL (prise de profits)
                    required_strength = ['moderate', 'strong', 'very_strong']
                    logger.debug(f"📈 {regime.name}: SELL modérément favorisé pour profits pour {symbol}")
                else:  # BUY
                    # En tendance haussière, être prudent avec les BUY
                    min_confidence = 0.75  # Plus strict pour BUY (prix en hausse)
                    required_strength = ['strong', 'very_strong']
                    logger.debug(f"📈 {regime.name}: BUY pénalisé (prix en hausse) pour {symbol}")
                
            elif regime.name == 'WEAK_TREND_UP':
                # Tendance haussière faible: plus équilibré
                if side == 'SELL':
                    min_confidence = 0.65  # Légèrement permissif pour SELL
                    required_strength = ['moderate', 'strong', 'very_strong']
                    logger.debug(f"📊 {regime.name}: SELL légèrement favorisé pour {symbol}")
                else:  # BUY
                    # En tendance faible, BUY acceptable mais prudent
                    min_confidence = 0.70  # Modérément strict pour BUY
                    required_strength = ['moderate', 'strong', 'very_strong']
                    logger.debug(f"📊 {regime.name}: BUY modérément accepté pour {symbol}")
                
            elif regime.name == 'RANGE_TIGHT':
                # Gestion spéciale pour ADX très faible (marché plat)
                adx = regime_metrics.get('adx', 0)
                if adx <= 5:  # ADX près de 0
                    # Exiger confirmation volume élevé
                    volume_ratio = signal.get('metadata', {}).get('volume_ratio', 1.0)
                    if volume_ratio < 2.0:
                        logger.info(f"🚫 Signal rejeté en RANGE_TIGHT: ADX={adx:.1f} et volume_ratio={volume_ratio:.1f} < 2.0")
                        return False
                    
                    # Marquer pour réduction de poids 0.5x
                    signal['metadata'] = signal.get('metadata', {})
                    signal['metadata']['adx_weight_modifier'] = 0.5
                    logger.info(f"⚖️ ADX faible ({adx:.1f}): poids réduit à 0.5x pour {symbol}")
                
                # Range serré: ASSOUPLI pour mean-reversion
                if strategy in self.STRATEGY_GROUPS.get('mean_reversion', []):
                    # ASSOUPLI pour stratégies de mean-reversion
                    min_confidence = 0.6  # ASSOUPLI de 0.75 à 0.6
                    required_strength = ['moderate', 'strong', 'very_strong']  # Ajouter moderate
                    logger.debug(f"🔒 {regime.name}: seuils ASSOUPLIS pour mean-reversion {symbol}")
                else:
                    # ASSOUPLI pour autres stratégies aussi
                    min_confidence = 0.7  # ASSOUPLI de 0.8 à 0.7
                    required_strength = ['strong', 'very_strong']
                    logger.debug(f"🔒 {regime.name}: seuils ASSOUPLIS (0.7) pour {symbol}")
                
            elif regime.name == 'RANGE_VOLATILE':
                # Range volatil: sélectif mais moins que tight
                min_confidence = 0.7
                required_strength = ['strong', 'very_strong']
                logger.debug(f"⚡ {regime.name}: seuils stricts pour {symbol}")
                
            elif regime.name in ['WEAK_TREND_DOWN', 'TREND_DOWN', 'STRONG_TREND_DOWN']:
                # Tendances baissières: INVERSER LA LOGIQUE - favoriser les BUY opportunistes, bloquer les SELL
                if side == 'BUY':
                    # En tendance baissière, c'est le moment d'acheter (prix bas)
                    if regime.name == 'STRONG_TREND_DOWN':
                        min_confidence = 0.65  # Plus permissif pour BUY quand prix très bas
                        required_strength = ['moderate', 'strong', 'very_strong']
                        logger.debug(f"📉 {regime.name}: BUY opportuniste favorisé (prix bas) pour {symbol}")
                    else:
                        min_confidence = 0.70  # Modérément permissif pour BUY
                        required_strength = ['moderate', 'strong', 'very_strong']
                else:  # SELL
                    # En tendance baissière, éviter de vendre (prix déjà bas)
                    if regime.name == 'STRONG_TREND_DOWN':
                        min_confidence = 0.85  # Très strict pour SELL quand prix très bas
                        required_strength = ['very_strong']
                        logger.debug(f"📉 {regime.name}: SELL fortement pénalisé (prix déjà bas) pour {symbol}")
                    else:
                        min_confidence = 0.80  # Strict pour SELL
                        required_strength = ['strong', 'very_strong']
                logger.debug(f"📉 {regime.name}: logique inversée BUY/SELL pour {symbol}")
                
            else:
                # Régime inconnu ou UNDEFINED: seuils par défaut
                min_confidence = 0.6
                required_strength = ['strong', 'very_strong']
                logger.debug(f"❓ {regime.name}: seuils par défaut pour {symbol}")
            
            # Exception pour signaux ultra-confluents de haute qualité
            if is_ultra_confluent and signal_score:
                if signal_score >= 85:
                    # Signaux excellents: réduire les seuils
                    min_confidence *= 0.8
                    if 'moderate' not in required_strength:
                        required_strength.append('moderate')
                    logger.info(f"⭐ Signal ultra-confluent excellent (score={signal_score:.1f}): seuils réduits pour {symbol}")
                elif signal_score >= 75:
                    # Signaux très bons: réduire modérément
                    min_confidence *= 0.9
                    logger.info(f"✨ Signal ultra-confluent très bon (score={signal_score:.1f}): seuils ajustés pour {symbol}")
            
            # Appliquer les filtres
            if signal_confidence < min_confidence:
                logger.info(f"🚫 Signal rejeté en {regime.name}: confiance {signal_confidence:.2f} < {min_confidence:.2f} "
                           f"pour {strategy} {side} {symbol}")
                return False
                
            # NOUVEAU: Accepter les signaux 'moderate' si 2+ stratégies s'accordent
            if signal_strength == 'moderate' and strategy_count >= 2:
                logger.info(f"✅ Signal 'moderate' accepté avec {strategy_count} stratégies en {regime.name}: "
                           f"{strategy} {side} {symbol}")
            elif signal_strength not in required_strength:
                logger.info(f"🚫 Signal rejeté en {regime.name}: force '{signal_strength}' insuffisante "
                           f"(requis: {required_strength}) pour {strategy} {side} {symbol}")
                return False
            
            # Signal accepté
            adx = regime_metrics.get('adx', 0)
            logger.info(f"✅ Signal accepté en {regime.name} (ADX={adx:.1f}): "
                       f"{strategy} {side} {symbol} force={signal_strength} confiance={signal_confidence:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur dans le filtrage Enhanced: {e}")
            return True  # En cas d'erreur, laisser passer le signal
    
    def get_regime_threshold(self, regime: Any) -> float:
        """Retourne le seuil de vote minimum selon le régime"""
        if MarketRegime is None:
            return 0.5  # Seuil par défaut
            
        thresholds = {
            MarketRegime.STRONG_TREND_UP: 0.6,
            MarketRegime.STRONG_TREND_DOWN: 0.6,
            MarketRegime.TREND_UP: 0.5,
            MarketRegime.TREND_DOWN: 0.5,
            MarketRegime.WEAK_TREND_UP: 0.4,
            MarketRegime.WEAK_TREND_DOWN: 0.4,
            MarketRegime.RANGE_TIGHT: 0.7,  # Plus strict en range serré
            MarketRegime.RANGE_VOLATILE: 0.6,
            MarketRegime.UNDEFINED: 0.8  # Très prudent si indéfini
        }
        return thresholds.get(regime, 0.5)
    
    def get_single_strategy_threshold(self, regime: Any) -> float:
        """Retourne le seuil de confiance pour les signaux d'une seule stratégie selon le régime"""
        if MarketRegime is None:
            return 0.8
            
        thresholds = {
            MarketRegime.STRONG_TREND_UP: 0.7,
            MarketRegime.STRONG_TREND_DOWN: 0.7,
            MarketRegime.TREND_UP: 0.75,
            MarketRegime.TREND_DOWN: 0.75,
            MarketRegime.WEAK_TREND_UP: 0.8,
            MarketRegime.WEAK_TREND_DOWN: 0.8,
            MarketRegime.RANGE_TIGHT: 0.85,  # Très strict en range serré
            MarketRegime.RANGE_VOLATILE: 0.8,
            MarketRegime.UNDEFINED: 0.9  # Très prudent si indéfini
        }
        return thresholds.get(regime, 0.8)
    
    def is_strategy_active(self, strategy: str, regime: str) -> bool:
        """Check if a strategy should be active in current regime"""
        
        # Adaptive strategies are always active
        if strategy in self.STRATEGY_GROUPS['adaptive']:
            return True
            
        # Handle enhanced regime codes
        if regime.startswith('STRONG_TREND') or regime.startswith('TREND'):
            return strategy in self.STRATEGY_GROUPS['trend']
            
        # Handle range regimes (RANGE_TIGHT, RANGE_VOLATILE, etc.)
        elif regime.startswith('RANGE'):
            return strategy in self.STRATEGY_GROUPS['mean_reversion']
            
        # Handle other enhanced regimes
        elif regime in ['BREAKOUT_UP', 'BREAKOUT_DOWN']:
            # Breakout regimes favor trend strategies
            return strategy in self.STRATEGY_GROUPS['trend']
        elif regime == 'VOLATILE':
            # In volatile markets, adaptive strategies work best
            return strategy in self.STRATEGY_GROUPS['adaptive']
            
        # In undefined regime, all strategies are active
        return True
    
    def apply_regime_confidence_boost(self, confidence: float, regime: Any, metrics: Dict[str, float]) -> float:
        """Applique un boost de confiance basé sur les métriques du régime"""
        # Boost basé sur la force de la tendance (ADX)
        from shared.src.config import ADX_STRONG_TREND_THRESHOLD, ADX_TREND_THRESHOLD
        adx = metrics.get('adx', 20)
        if adx > ADX_STRONG_TREND_THRESHOLD:  # Tendance très forte
            confidence *= 1.1
        elif adx > ADX_TREND_THRESHOLD:  # Tendance forte
            confidence *= 1.05
        
        # Boost basé sur le momentum (ROC)
        roc = abs(metrics.get('roc', 0))
        if roc > 5:  # Momentum fort
            confidence *= 1.05
        
        # Penalty pour les régimes indéfinis ou instables
        if MarketRegime is not None:
            if regime == MarketRegime.UNDEFINED:
                confidence *= 0.9
            elif regime in [MarketRegime.RANGE_VOLATILE]:
                # Ne pas pénaliser les stratégies de mean-reversion en range
                # Note: cette logique est maintenant dans _apply_enhanced_regime_filtering
                confidence *= 0.95
        
        return min(1.0, confidence)  # Cap à 1.0
    
    def determine_signal_strength(self, confidence: float, regime: Any) -> str:
        """Détermine la force du signal basée sur la confiance et le régime"""
        # Seuils standardisés alignés avec analyzer
        # moderate ≥ 0.55, strong ≥ 0.75, very_strong ≥ 0.9
        if confidence >= 0.9:
            return 'very_strong'
        elif confidence >= 0.75:
            return 'strong'
        elif confidence >= 0.55:
            return 'moderate'
        else:
            return 'weak'