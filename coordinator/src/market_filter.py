"""
Module de filtrage des signaux basé sur les conditions de marché.
Extrait de signal_handler.py pour améliorer la modularité.
"""
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from shared.src.schemas import StrategySignal
from shared.src.enums import OrderSide, SignalStrength

logger = logging.getLogger(__name__)


class MarketFilter:
    """
    Gère le filtrage des signaux basé sur les conditions de marché.
    Permet de filtrer les signaux en fonction du mode de marché (ride/react),
    des actions recommandées et de l'âge des données.
    """
    
    def __init__(self, max_filter_age: int = 900):
        """
        Initialise le filtre de marché.
        
        Args:
            max_filter_age: Âge maximum des données de filtrage en secondes (défaut: 15 minutes)
        """
        self.max_filter_age = max_filter_age
        self.market_filters: Dict[str, Dict[str, Any]] = {}
        self.last_refresh_attempt: Dict[str, float] = {}
        
    def should_filter_signal(self, signal: StrategySignal) -> bool:
        """
        Détermine si un signal doit être filtré en utilisant les données enrichies.
        
        OPTIMISATION: Utilise les données techniques du signal_aggregator.
        
        Args:
            signal: Signal à évaluer (avec métadonnées techniques)
            
        Returns:
            True si le signal doit être filtré (ignoré), False sinon
        """
        # NOUVEAU: Filtrage intelligent pour signaux agrégés basé sur leurs métadonnées
        if signal.strategy.startswith("Aggregated_"):
            return self._apply_technical_filtering(signal)
            
        # Vérifier si nous avons des informations de filtrage pour ce symbole
        if signal.symbol not in self.market_filters:
            logger.debug(f"Aucune information de filtrage pour {signal.symbol}")
            return False
            
        filter_info = self.market_filters[signal.symbol]
        
        # Vérifier si les informations sont obsolètes
        if self._is_filter_outdated(signal.symbol):
            logger.warning(f"Informations de filtrage obsolètes pour {signal.symbol}")
            # En mode de secours, filtrer uniquement les signaux faibles
            signal_strength = signal.strength if hasattr(signal.strength, 'value') else SignalStrength(signal.strength) if isinstance(signal.strength, str) else signal.strength
            if signal_strength == SignalStrength.WEAK:
                logger.info(f"Signal {signal.side} filtré en mode de secours (force insuffisante)")
                return True
            return False
            
        # Appliquer les règles de filtrage basées sur le mode de marché
        legacy_filter = self._apply_filter_rules(signal, filter_info)
        
        # NOUVEAU: Compléter avec filtrage technique si disponible
        technical_filter = self._apply_technical_filtering(signal)
        
        # Combiner les deux filtres (OR logic - filtré si l'un des deux dit oui)
        return legacy_filter or technical_filter
        
    def update_market_filter(self, symbol: str, filter_data: Dict[str, Any]):
        """
        Met à jour les informations de filtrage pour un symbole.
        
        Args:
            symbol: Symbole à mettre à jour
            filter_data: Nouvelles données de filtrage
        """
        filter_data['updated_at'] = time.time()
        self.market_filters[symbol] = filter_data
        logger.info(f"Filtre de marché mis à jour pour {symbol}: {filter_data}")
    
    def _apply_technical_filtering(self, signal: StrategySignal) -> bool:
        """
        Applique un filtrage basé sur les indicateurs techniques enrichis du signal_aggregator.
        
        OPTIMISATION: Réutilise les calculs déjà effectués.
        
        Args:
            signal: Signal avec métadonnées techniques
            
        Returns:
            True si le signal doit être filtré
        """
        try:
            # Vérifier la présence des métadonnées
            if not hasattr(signal, 'metadata') or not signal.metadata:
                # Pas de métadonnées = filtrage conservateur
                signal_strength = getattr(signal, 'strength', None)
                if signal_strength and str(signal_strength).lower() == 'weak':
                    logger.info(f"🚫 Signal {signal.side} filtré: force faible sans métadonnées")
                    return True
                return False
            
            metadata = signal.metadata
            
            # 1. Filtrage basé sur le régime Enhanced
            regime = metadata.get('regime')
            regime_metrics = metadata.get('regime_metrics', {})
            
            if regime == 'UNDEFINED' or regime == 'VOLATILE':
                # Marché instable: filtrer les signaux faibles (confidence < 0.65 pour crypto)
                signal_confidence = getattr(signal, 'confidence', 0.5)
                if signal_confidence < 0.65:  # Seuil crypto pour marché instable
                    logger.info(f"🚫 Signal {signal.side} filtré: confiance {signal_confidence:.2f} en marché {regime}")
                    return True
            
            # 2. Filtrage basé sur l'ADX (force de tendance)
            adx = regime_metrics.get('adx')
            if adx is not None:
                if adx < 20 and not metadata.get('ultra_confluence'):
                    logger.info(f"🚫 Signal {signal.side} filtré: ADX trop faible ({adx:.1f}) sans ultra-confluence")
                    return True
                elif adx > 70:
                    # Tendance très forte: risque de retournement
                    logger.info(f"⚠️ Signal {signal.side} filtré: ADX très élevé ({adx:.1f}) - risque retournement")
                    return True
            
            # 3. Filtrage basé sur le volume
            volume_analysis = metadata.get('volume_analysis', {})
            if volume_analysis:
                avg_volume_ratio = volume_analysis.get('avg_volume_ratio', 1.0)
                if avg_volume_ratio < 0.8:
                    logger.info(f"🚫 Signal {signal.side} filtré: volume faible (ratio={avg_volume_ratio:.2f})")
                    return True
            
            # 4. Filtrage par confiance multi-stratégies (CRYPTO OPTIMISÉ)
            strategy_count = metadata.get('strategy_count', 1)
            signal_confidence = getattr(signal, 'confidence', 0.5)
            
            # Seuils crypto plus stricts
            if strategy_count == 1 and signal_confidence < 0.75:  # Relevé de 0.8 à 0.75 pour crypto
                logger.info(f"🚫 Signal {signal.side} filtré: une seule stratégie avec confiance {signal_confidence:.2f}")
                return True
            elif strategy_count >= 2 and signal_confidence < 0.55:  # Multi-stratégies : seuil 0.55
                logger.info(f"🚫 Signal {signal.side} filtré: {strategy_count} stratégies avec confiance {signal_confidence:.2f}")
                return True
            
            # 5. Exception pour signaux institutionnels
            if metadata.get('institutional_grade') or metadata.get('excellent_grade'):
                logger.info(f"🎆 Signal {signal.side} exempté du filtrage: qualité institutionnelle/excellente")
                return False
            
            # Signal passé avec succès
            logger.debug(f"✅ Signal {signal.side} passé le filtrage technique")
            return False
            
        except Exception as e:
            logger.error(f"Erreur filtrage technique: {e}")
            return False  # En cas d'erreur, ne pas filtrer
        
    def _is_filter_outdated(self, symbol: str) -> bool:
        """
        Vérifie si les données de filtrage sont obsolètes.
        
        Args:
            symbol: Symbole à vérifier
            
        Returns:
            True si les données sont obsolètes
        """
        if symbol not in self.market_filters:
            return True
            
        filter_info = self.market_filters[symbol]
        age = time.time() - filter_info.get('updated_at', 0)
        return age > self.max_filter_age
        
    def _apply_filter_rules(self, signal: StrategySignal, filter_info: Dict[str, Any]) -> bool:
        """
        Applique les règles de filtrage au signal.
        
        Args:
            signal: Signal à évaluer
            filter_info: Informations de filtrage
            
        Returns:
            True si le signal doit être filtré
        """
        # Filtrage basé sur le mode de marché
        mode = filter_info.get('mode')
        
        # En mode "ride", filtrer les signaux contre-tendance faibles
        if mode == 'ride':
            signal_side = signal.side if hasattr(signal.side, 'value') else OrderSide(signal.side) if isinstance(signal.side, str) else signal.side
            signal_strength = signal.strength if hasattr(signal.strength, 'value') else SignalStrength(signal.strength) if isinstance(signal.strength, str) else signal.strength
            if signal_side == OrderSide.SELL and signal_strength != SignalStrength.VERY_STRONG:
                logger.info(f"🔍 Signal {signal.side} filtré: marché en mode RIDE pour {signal.symbol}")
                return True
                
        # Filtrage basé sur les actions recommandées
        action = filter_info.get('action')
        
        if action == 'no_trading':
            logger.info(f"🔍 Signal {signal.side} filtré: action 'no_trading' active pour {signal.symbol}")
            return True
            
        elif action == 'buy_only' and (signal.side if hasattr(signal.side, 'value') else OrderSide(signal.side) if isinstance(signal.side, str) else signal.side) == OrderSide.SELL:
            logger.info(f"🔍 Signal {signal.side} filtré: seuls les achats autorisés pour {signal.symbol}")
            return True
            
        elif action == 'sell_only' and (signal.side if hasattr(signal.side, 'value') else OrderSide(signal.side) if isinstance(signal.side, str) else signal.side) == OrderSide.BUY:
            logger.info(f"🔍 Signal {signal.side} filtré: seules les ventes autorisées pour {signal.symbol}")
            return True
            
        return False
        
    def get_filter_status(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtient le statut des filtres de marché.
        
        Args:
            symbol: Symbole spécifique ou None pour tous
            
        Returns:
            Statut des filtres
        """
        if symbol:
            if symbol in self.market_filters:
                filter_info = self.market_filters[symbol].copy()
                filter_info['is_outdated'] = self._is_filter_outdated(symbol)
                return {symbol: filter_info}
            return {symbol: {"status": "no_filter"}}
            
        # Retourner le statut de tous les filtres
        status = {}
        for sym, filter_info in self.market_filters.items():
            info = filter_info.copy()
            info['is_outdated'] = self._is_filter_outdated(sym)
            status[sym] = info
            
        return status
        
    def clear_outdated_filters(self):
        """
        Supprime les filtres obsolètes du cache.
        """
        symbols_to_remove = []
        current_time = time.time()
        
        for symbol, filter_info in self.market_filters.items():
            if current_time - filter_info.get('updated_at', 0) > self.max_filter_age * 2:
                symbols_to_remove.append(symbol)
                
        for symbol in symbols_to_remove:
            del self.market_filters[symbol]
            logger.info(f"Filtre obsolète supprimé pour {symbol}")
            
        if symbols_to_remove:
            logger.info(f"Suppression de {len(symbols_to_remove)} filtres obsolètes")