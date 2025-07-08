#!/usr/bin/env python3
"""
Module pour l'accumulation des données de marché historiques.
Sépare la logique d'accumulation de données du signal aggregator principal.
"""

import logging
from typing import Dict, List, Any
from collections import defaultdict, deque
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class MarketDataAccumulator:
    """Accumule les données de marché pour construire un historique"""
    
    def __init__(self, max_history: int = 200):
        self.max_history = max_history
        self.data_history = defaultdict(lambda: deque(maxlen=max_history))
        self.last_update = defaultdict(float)
    
    def add_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Ajoute des données de marché à l'historique"""
        try:
            timestamp = data.get('timestamp', time.time())
            
            # Éviter les doublons (même timestamp)
            if timestamp <= self.last_update[symbol]:
                return
                
            # Enrichir les données avec timestamp normalisé
            enriched_data = data.copy()
            enriched_data['timestamp'] = timestamp
            enriched_data['datetime'] = datetime.fromtimestamp(timestamp)
            
            # Ajouter à l'historique
            self.data_history[symbol].append(enriched_data)
            self.last_update[symbol] = timestamp
            
        except Exception as e:
            logger.error(f"Erreur ajout données historiques {symbol}: {e}")
    
    def get_history(self, symbol: str, limit: int = None) -> List[Dict[str, Any]]:
        """Récupère l'historique des données pour un symbole"""
        history = list(self.data_history[symbol])
        if limit and len(history) > limit:
            return history[-limit:]
        return history
    
    def get_history_count(self, symbol: str) -> int:
        """Retourne le nombre de points historiques disponibles"""
        return len(self.data_history[symbol])
    
    def clear_history(self, symbol: str = None) -> None:
        """Efface l'historique pour un symbole ou tous les symboles"""
        if symbol:
            if symbol in self.data_history:
                self.data_history[symbol].clear()
                self.last_update[symbol] = 0
        else:
            self.data_history.clear()
            self.last_update.clear()
    
    def get_latest_data(self, symbol: str) -> Dict[str, Any]:
        """Récupère les dernières données pour un symbole"""
        history = self.data_history.get(symbol)
        if history and len(history) > 0:
            return history[-1]
        return {}
    
    def get_symbols(self) -> List[str]:
        """Retourne la liste des symboles suivis"""
        return list(self.data_history.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques sur l'accumulateur"""
        total_points = sum(len(history) for history in self.data_history.values())
        return {
            'symbols_tracked': len(self.data_history),
            'total_data_points': total_points,
            'max_history_per_symbol': self.max_history,
            'symbols': {
                symbol: {
                    'data_points': len(history),
                    'last_update': self.last_update.get(symbol, 0)
                } for symbol, history in self.data_history.items()
            }
        }