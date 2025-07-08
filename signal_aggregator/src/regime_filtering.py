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
                # Tendance haussière forte: favoriser les BUY, pénaliser les SELL
                if side == 'SELL':
                    min_confidence = 0.80  # Pénaliser SELL en forte tendance haussière
                    required_strength = ['very_strong']
                    logger.debug(f"💪 {regime.name}: SELL pénalisé, seuils stricts pour {symbol}")
                else:  # BUY
                    # Validation MACD pour confirmer la force de tendance
                    if self.technical_analysis and self.technical_analysis.validate_macd_trend(technical_context, 'bullish'):
                        min_confidence = 0.35  # Encore plus permissif si MACD confirme
                        logger.debug(f"💪 {regime.name}: MACD confirme, seuils très assouplis pour {symbol}")
                    else:
                        min_confidence = 0.4
                        logger.debug(f"💪 {regime.name}: seuils assouplis pour {symbol}")
                    required_strength = ['weak', 'moderate', 'strong', 'very_strong']
                
            elif regime.name == 'TREND_UP':
                # Tendance haussière: favoriser les BUY, pénaliser modérément les SELL
                if side == 'SELL':
                    min_confidence = 0.75  # Pénaliser SELL en tendance haussière
                    required_strength = ['strong', 'very_strong']
                    logger.debug(f"📈 {regime.name}: SELL pénalisé, seuils élevés pour {symbol}")
                else:  # BUY
                    # Validation OBV pour confirmer le volume
                    if self.technical_analysis and self.technical_analysis.validate_obv_trend(technical_context, side):
                        min_confidence = 0.45  # Bonus si OBV confirme
                        logger.debug(f"📈 {regime.name}: OBV confirme, seuils bonus (0.45) pour {symbol}")
                    else:
                        min_confidence = 0.5  # ASSOUPLI à 0.50 (était 0.7)
                        logger.debug(f"📈 {regime.name}: seuils ASSOUPLIS (0.5) pour {symbol}")
                    required_strength = ['moderate', 'strong', 'very_strong']
                
            elif regime.name == 'WEAK_TREND_UP':
                # Tendance haussière faible: légère pénalisation des SELL
                if side == 'SELL':
                    min_confidence = 0.70  # Pénaliser légèrement SELL en tendance haussière faible
                    required_strength = ['strong', 'very_strong']
                    logger.debug(f"📊 {regime.name}: SELL légèrement pénalisé pour {symbol}")
                else:  # BUY
                    # Validation ROC pour détecter l'accélération
                    roc_boost = self.technical_analysis.check_roc_acceleration(technical_context, side) if self.technical_analysis else False
                    if roc_boost:
                        min_confidence = 0.50  # Bonus si ROC détecte accélération
                        logger.debug(f"📊 {regime.name}: ROC accélération détectée, seuils bonus (0.50) pour {symbol}")
                    else:
                        min_confidence = 0.55  # ASSOUPLI à 0.55 (était 0.65)
                        logger.debug(f"📊 {regime.name}: seuils ASSOUPLIS (0.55) pour {symbol}")
                    required_strength = ['moderate', 'strong', 'very_strong']
                
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
                # Tendances baissières: favoriser les SELL, bloquer les BUY faibles
                if side == 'BUY':
                    min_confidence = 0.80  # Assoupli de 0.85 à 0.80 pour les BUY en downtrend
                    required_strength = ['very_strong']
                else:  # SELL
                    min_confidence = 0.7  # Seuil ajusté pour les SELL (0.7 recommandé)
                    required_strength = ['moderate', 'strong', 'very_strong']
                logger.debug(f"📉 {regime.name}: adaptation BUY/SELL pour {symbol}")
                
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
        adx = metrics.get('adx', 20)
        if adx > 40:  # Tendance très forte
            confidence *= 1.1
        elif adx > 30:  # Tendance forte
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