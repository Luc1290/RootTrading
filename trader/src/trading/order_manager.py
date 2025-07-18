"""
Gestionnaire d'ordres simplifié pour le trader.
Rôle : Recevoir ordres du coordinator → Exécuter via OrderExecutor → Monitoring prix.
"""
import logging
import time
from typing import Dict, List, Any, Optional

from shared.src.config import SYMBOLS, TRADING_MODE
from shared.src.enums import OrderSide
from shared.src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY

from trader.src.exchange.binance_executor import BinanceExecutor
from trader.src.trading.order_executor import OrderExecutor
from trader.src.trading.price_monitor import PriceMonitor

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Gestionnaire d'ordres simplifié.
    Plus de cycles complexes : juste exécuter les ordres du coordinator.
    """
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialise le gestionnaire d'ordres.
        
        Args:
            symbols: Liste des symboles à surveiller
        """
        self.symbols = symbols or SYMBOLS
        self.start_time = time.time()
        
        # Initialiser les composants
        demo_mode = TRADING_MODE.lower() == 'demo'
        self.binance_executor = BinanceExecutor(
            api_key=BINANCE_API_KEY, 
            api_secret=BINANCE_SECRET_KEY, 
            demo_mode=demo_mode
        )
        
        # Exécuteur d'ordres simplifié
        self.order_executor = OrderExecutor(self.binance_executor)
        
        # Moniteur de prix pour les données de marché
        self.price_monitor = PriceMonitor(
            symbols=self.symbols,
            price_update_callback=self.handle_price_update
        )
        
        # Cache des derniers prix
        self.last_prices: Dict[str, float] = {}
        self.last_price_update = time.time()
        
        # Configuration pour les pauses de trading
        self.paused_symbols: set[str] = set()
        self.paused_strategies: set[str] = set()
        self.paused_all = False
        
        logger.info(f"✅ OrderManager initialisé (mode simplifié) pour {len(self.symbols)} symboles")
    
    def is_trading_paused(self, symbol: str, strategy: Optional[str] = None) -> bool:
        """
        Vérifie si le trading est en pause.
        
        Args:
            symbol: Symbole à vérifier
            strategy: Stratégie à vérifier (optionnel)
            
        Returns:
            True si en pause, False sinon
        """
        if self.paused_all:
            return True
        
        if symbol in self.paused_symbols:
            return True
        
        if strategy and strategy in self.paused_strategies:
            return True
        
        return False
    
    def pause_symbol(self, symbol: str) -> None:
        """Met en pause le trading pour un symbole."""
        self.paused_symbols.add(symbol)
        logger.info(f"⏸️ Trading en pause pour {symbol}")
    
    def resume_symbol(self, symbol: str) -> None:
        """Reprend le trading pour un symbole."""
        self.paused_symbols.discard(symbol)
        logger.info(f"▶️ Trading repris pour {symbol}")
    
    def pause_strategy(self, strategy: str) -> None:
        """Met en pause le trading pour une stratégie."""
        self.paused_strategies.add(strategy)
        logger.info(f"⏸️ Trading en pause pour la stratégie {strategy}")
    
    def resume_strategy(self, strategy: str) -> None:
        """Reprend le trading pour une stratégie."""
        self.paused_strategies.discard(strategy)
        logger.info(f"▶️ Trading repris pour la stratégie {strategy}")
    
    def pause_all(self) -> None:
        """Met en pause tout le trading."""
        self.paused_all = True
        logger.info("⏸️ Trading en pause (global)")
    
    def resume_all(self) -> None:
        """Reprend tout le trading."""
        self.paused_all = False
        self.paused_symbols.clear()
        self.paused_strategies.clear()
        logger.info("▶️ Trading repris (global)")
    
    def handle_price_update(self, symbol: str, price: float) -> None:
        """
        Traite une mise à jour de prix.
        
        Args:
            symbol: Symbole mis à jour
            price: Nouveau prix
        """
        self.last_prices[symbol] = price
        self.last_price_update = time.time()
        
        # Plus de trailing stops ou de cycles - juste du monitoring
        logger.debug(f"📊 Prix mis à jour: {symbol} = {price}")
    
    def create_order(self, order_data: Dict[str, Any]) -> Optional[str]:
        """
        Crée un ordre.
        
        Args:
            order_data: Données de l'ordre du coordinator
            
        Returns:
            ID de l'ordre créé ou None
        """
        try:
            symbol = order_data.get("symbol")
            strategy = order_data.get("strategy", "Manual")
            
            # Vérifier si le trading est en pause
            if symbol and self.is_trading_paused(symbol, strategy):
                logger.warning(f"❌ Trading en pause pour {symbol}/{strategy}")
                return None
            
            # Vérifier le symbole supporté
            if symbol not in self.symbols:
                logger.error(f"❌ Symbole non supporté: {symbol}")
                return None
            
            # Déléguer à l'exécuteur
            return self.order_executor.execute_order(order_data)
            
        except Exception as e:
            logger.error(f"❌ Erreur création ordre: {str(e)}")
            return None
    
    def get_order_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des ordres.
        
        Args:
            limit: Nombre maximum d'ordres
            
        Returns:
            Liste des ordres récents
        """
        return self.order_executor.get_order_history(limit)
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère le statut d'un ordre.
        
        Args:
            order_id: ID de l'ordre
            
        Returns:
            Statut de l'ordre ou None
        """
        return self.order_executor.get_order_status(order_id)
    
    def get_current_prices(self) -> Dict[str, float]:
        """
        Récupère les prix actuels.
        
        Returns:
            Dict {symbol: price}
        """
        return self.last_prices.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques du gestionnaire.
        
        Returns:
            Statistiques
        """
        executor_stats = self.order_executor.get_stats()
        
        return {
            "uptime": time.time() - self.start_time,
            "symbols": self.symbols,
            "last_price_update": self.last_price_update,
            "paused_symbols": list(self.paused_symbols),
            "paused_strategies": list(self.paused_strategies),
            "paused_all": self.paused_all,
            "executor_stats": executor_stats
        }
    
    def start(self) -> None:
        """Démarre le gestionnaire d'ordres."""
        logger.info("🚀 Démarrage du gestionnaire d'ordres (simplifié)...")
        
        # Démarrer le moniteur de prix
        self.price_monitor.start()
        
        logger.info("✅ Gestionnaire d'ordres démarré")
    
    def stop(self) -> None:
        """Arrête le gestionnaire d'ordres."""
        logger.info("Arrêt du gestionnaire d'ordres...")
        
        # Arrêter le moniteur de prix
        if self.price_monitor:
            self.price_monitor.stop()
        
        logger.info("✅ Gestionnaire d'ordres arrêté")