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
        Détermine si un signal doit être filtré.
        
        Args:
            signal: Signal à évaluer
            
        Returns:
            True si le signal doit être filtré (ignoré), False sinon
        """
        # Les signaux agrégés sont exemptés du filtrage
        if signal.strategy.startswith("Aggregated_"):
            logger.info(f"✅ Signal agrégé exempté du filtrage: {signal.strategy}")
            return False
            
        # Vérifier si nous avons des informations de filtrage pour ce symbole
        if signal.symbol not in self.market_filters:
            logger.debug(f"Aucune information de filtrage pour {signal.symbol}")
            return False
            
        filter_info = self.market_filters[signal.symbol]
        
        # Vérifier si les informations sont obsolètes
        if self._is_filter_outdated(signal.symbol):
            logger.warning(f"Informations de filtrage obsolètes pour {signal.symbol}")
            # En mode de secours, filtrer uniquement les signaux faibles
            if signal.strength == SignalStrength.WEAK:
                logger.info(f"Signal {signal.side} filtré en mode de secours (force insuffisante)")
                return True
            return False
            
        # Appliquer les règles de filtrage basées sur le mode de marché
        return self._apply_filter_rules(signal, filter_info)
        
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
            if signal.side == OrderSide.SELL and signal.strength != SignalStrength.VERY_STRONG:
                logger.info(f"🔍 Signal {signal.side} filtré: marché en mode RIDE pour {signal.symbol}")
                return True
                
        # Filtrage basé sur les actions recommandées
        action = filter_info.get('action')
        
        if action == 'no_trading':
            logger.info(f"🔍 Signal {signal.side} filtré: action 'no_trading' active pour {signal.symbol}")
            return True
            
        elif action == 'buy_only' and signal.side == OrderSide.SELL:
            logger.info(f"🔍 Signal {signal.side} filtré: seuls les achats autorisés pour {signal.symbol}")
            return True
            
        elif action == 'sell_only' and signal.side == OrderSide.BUY:
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