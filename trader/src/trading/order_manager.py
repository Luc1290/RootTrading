"""
Gestionnaire d'ordres pour le trader.
Gère les cycles de trading via l'API REST.
Version simplifiée utilisant les nouveaux modules.
"""
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Union

from shared.src.config import SYMBOLS, TRADE_QUANTITIES, TRADE_QUANTITY, TRADING_MODE
from shared.src.enums import OrderSide, CycleStatus
from shared.src.schemas import TradeOrder

from trader.src.trading.cycle_manager import CycleManager
from trader.src.exchange.binance_executor import BinanceExecutor
from trader.src.trading.price_monitor import PriceMonitor
from trader.src.trading.reconciliation import ExchangeReconciliation
from trader.src.utils.safety import safe_execute, notify_error
from shared.src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY

# Configuration du logging
logger = logging.getLogger(__name__)

class OrderManager:
    """
    Gestionnaire d'ordres.
    Gère les cycles de trading via l'API REST.
    """
    
    def __init__(self, symbols: List[str] = None):
        """
        Initialise le gestionnaire d'ordres.
        
        Args:
            symbols: Liste des symboles à surveiller
        """
        self.symbols = symbols or SYMBOLS
        self.start_time = time.time()
        
        # Initialiser les composants
        # Vérifier si on est en mode demo selon la config
        demo_mode = TRADING_MODE.lower() == 'demo'
        self.binance_executor = BinanceExecutor(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY, demo_mode=demo_mode)
        self.cycle_manager = CycleManager(binance_executor=self.binance_executor)
        
        # Le trader reçoit maintenant les ordres uniquement via l'API REST du coordinator
        
        # Initialiser le moniteur de prix
        self.price_monitor = PriceMonitor(
            symbols=self.symbols,
            price_update_callback=self.handle_price_update
        )
        
        # Configuration pour les pauses de trading
        self.paused_symbols = set()
        self.paused_strategies = set()
        self.pause_until = 0  # Timestamp pour la reprise automatique
        self.paused_all = False
        
        # Pour accéder facilement aux derniers prix
        self.last_prices = {}
        self.last_price_update = time.time()
        
        # Initialiser le service de réconciliation
        self.reconciliation_service = ExchangeReconciliation(
            cycle_repository=self.cycle_manager.repository,
            binance_executor=self.binance_executor,
            reconciliation_interval=300,  # Réconciliation toutes les 5 minutes
            cycle_manager=self.cycle_manager  # Passer le cycle_manager pour nettoyer le cache
        )
        
        logger.info(f"✅ OrderManager initialisé pour {len(self.symbols)} symboles: {', '.join(self.symbols)}")
    
    def is_trading_paused(self, symbol: str, strategy: str) -> bool:
        """
        Vérifie si le trading est en pause pour un symbole ou une stratégie donnée.
        
        Args:
            symbol: Symbole à vérifier
            strategy: Stratégie à vérifier
            
        Returns:
            True si le trading est en pause, False sinon
        """
        # Vérifier si la pause globale est active
        if self.paused_all:
            # Vérifier si la pause a une durée et si elle est expirée
            if self.pause_until > 0 and time.time() > self.pause_until:
                self.resume_all()
                return False
            return True
        
        # Vérifier si le symbole est en pause
        if symbol in self.paused_symbols:
            return True
        
        # Vérifier si la stratégie est en pause
        if strategy in self.paused_strategies:
            return True
        
        return False
    
    def pause_symbol(self, symbol: str, duration: int = 0) -> None:
        """
        Met en pause le trading pour un symbole spécifique.
        
        Args:
            symbol: Symbole à mettre en pause
            duration: Durée de la pause en secondes (0 = indéfini)
        """
        self.paused_symbols.add(symbol)
        
        if duration > 0:
            # Planifier une reprise automatique
            import threading
            threading.Timer(duration, self.resume_symbol, args=[symbol]).start()
        
        logger.info(f"Trading en pause pour le symbole {symbol}" + 
                   (f" pendant {duration}s" if duration > 0 else ""))
    
    def pause_strategy(self, strategy: str, duration: int = 0) -> None:
        """
        Met en pause le trading pour une stratégie spécifique.
        
        Args:
            strategy: Stratégie à mettre en pause
            duration: Durée de la pause en secondes (0 = indéfini)
        """
        self.paused_strategies.add(strategy)
        
        if duration > 0:
            # Planifier une reprise automatique
            import threading
            threading.Timer(duration, self.resume_strategy, args=[strategy]).start()
        
        logger.info(f"Trading en pause pour la stratégie {strategy}" + 
                   (f" pendant {duration}s" if duration > 0 else ""))
    
    def pause_all(self, duration: int = 0) -> None:
        """
        Met en pause le trading pour tous les symboles et stratégies.
        
        Args:
            duration: Durée de la pause en secondes (0 = indéfini)
        """
        self.paused_all = True
        
        if duration > 0:
            self.pause_until = time.time() + duration
            # Planifier une reprise automatique
            import threading
            threading.Timer(duration, self.resume_all).start()
        
        logger.info(f"Trading en pause pour tous les symboles et stratégies" + 
                  (f" pendant {duration}s" if duration > 0 else ""))
    
    def resume_symbol(self, symbol: str) -> None:
        """
        Reprend le trading pour un symbole spécifique.
        
        Args:
            symbol: Symbole à reprendre
        """
        if symbol in self.paused_symbols:
            self.paused_symbols.remove(symbol)
            logger.info(f"Trading repris pour le symbole {symbol}")
    
    def resume_strategy(self, strategy: str) -> None:
        """
        Reprend le trading pour une stratégie spécifique.
        
        Args:
            strategy: Stratégie à reprendre
        """
        if strategy in self.paused_strategies:
            self.paused_strategies.remove(strategy)
            logger.info(f"Trading repris pour la stratégie {strategy}")
    
    def resume_all(self) -> None:
        """
        Reprend le trading pour tous les symboles et stratégies.
        """
        self.paused_all = False
        self.paused_symbols.clear()
        self.paused_strategies.clear()
        self.pause_until = 0
        logger.info("Trading repris pour tous les symboles et stratégies")
    
    # handle_signal supprimée - Les ordres viennent maintenant via l'API REST
    
    def handle_price_update(self, symbol: str, price: float) -> None:
        """
        Traite une mise à jour de prix.
        
        Args:
            symbol: Symbole mis à jour
            price: Nouveau prix
        """
        # Mettre à jour le cache de prix
        self.last_prices[symbol] = price
        self.last_price_update = time.time()
        
        # Transmettre au gestionnaire de cycles
        self.cycle_manager.process_price_update(symbol, price)
    
    def calculate_min_profitable_change(self, symbol: str) -> float:
        """
        Calcule le pourcentage minimal de changement de prix nécessaire pour être rentable,
        en tenant compte des frais d'achat et de vente.
    
        Args:
            symbol: Le symbole de la paire de trading
        
        Returns:
            Le pourcentage minimal nécessaire (ex: 0.3 pour 0.3%)
        """
        maker_fee, taker_fee = self.binance_executor.get_trade_fee(symbol)
    
        # Calculer l'impact total des frais (achat + vente)
        # Pour être sûr, on considère les frais taker qui sont généralement plus élevés
        total_fee_impact = (1 + taker_fee) * (1 + taker_fee) - 1
    
        # Ajouter une petite marge de sécurité (20% supplémentaire)
        min_profitable_change = total_fee_impact * 1.2 * 100  # Convertir en pourcentage
    
        return min_profitable_change
    
    def create_manual_order(self, symbol: str, side: OrderSide, quantity: float, 
                           price: Optional[float] = None, strategy: str = "Manual",
                           stop_price: Optional[float] = None) -> str:
        """
        Crée un ordre manuel (hors signal).
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            side: Côté de l'ordre (BUY ou SELL)
            quantity: Quantité à trader
            price: Prix (optionnel, sinon au marché)
            strategy: Nom de la stratégie (optionnel, défaut: "Manual")
            
        Returns:
            ID du cycle créé ou message d'erreur
        """
        try:
            # Vérifier si le symbole est supporté
            if symbol not in self.symbols:
                return f"Symbole non supporté: {symbol}"
            
            # Utiliser le dernier prix connu si non spécifié
            if price is None:
                price = self.last_prices.get(symbol)
                if price is None:
                    return "Prix non spécifié et aucun prix récent disponible"
            
            # Vérifier et ajuster la quantité selon les contraintes de l'exchange
            constraints = self.binance_executor.symbol_constraints
            min_quantity = constraints.calculate_min_quantity(symbol, price)
            
            # Utiliser la plus grande des deux valeurs
            adjusted_quantity = max(quantity, min_quantity)
            
            # Log si la quantité a été ajustée
            if adjusted_quantity > quantity:
                logger.info(f"Quantité ajustée pour {symbol}: {quantity} → {adjusted_quantity} (minimum requis)")
            
            # Calculer le stop par défaut si non spécifié (plus de target avec TrailingStop pur)
            if stop_price is None:
                # Par défaut: -2% pour BUY, +2% pour SELL
                stop_price = price * 0.98 if side == OrderSide.BUY else price * 1.02
                logger.info(f"Stop loss calculé automatiquement: {stop_price:.8f}")
                
            # Créer un cycle avec TrailingStop pur (plus de target_price)
            cycle = self.cycle_manager.create_cycle(
                symbol=symbol,
                strategy=strategy,
                side=side,
                price=price,
                quantity=adjusted_quantity,
                stop_price=stop_price
            )
            
            if cycle:
                return cycle.id
            else:
                return "Échec de création du cycle"
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de l'ordre manuel: {str(e)}")
            return f"Erreur: {str(e)}"
    
    def get_active_cycles(self, symbol: Optional[str] = None, strategy: Optional[str] = None):
        """
        Récupère les cycles actifs, avec filtrage optionnel.
        
        Args:
            symbol: Filtrer par symbole (optionnel)
            strategy: Filtrer par stratégie (optionnel)
            
        Returns:
            Liste des cycles actifs filtrés
        """
        return self.cycle_manager.get_active_cycles(symbol, strategy)
    
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """
        Récupère la liste des ordres actifs.
        
        Returns:
            Liste des ordres actifs au format JSON
        """
        active_cycles = self.cycle_manager.get_active_cycles()
        
        # Convertir les cycles en dictionnaires pour JSON
        orders = []
        for cycle in active_cycles:
            orders.append({
                "id": cycle.id,
                "symbol": cycle.symbol,
                "strategy": cycle.strategy,
                "status": cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status),
                # Vérification robuste du statut pour déterminer côté (insensible à la casse)
                "side": "BUY" if (
                    hasattr(cycle.status, 'value') and cycle.status.value.lower() in ['active_BUY', 'waiting_BUY'] or
                    isinstance(cycle.status, str) and cycle.status.lower() in ['active_BUY', 'waiting_BUY']
                ) else "SELL",
                "entry_price": cycle.entry_price,
                "current_price": self.last_prices.get(cycle.symbol),
                "quantity": cycle.quantity,
                "stop_price": cycle.stop_price,
                "created_at": cycle.created_at.isoformat() if cycle.created_at else None
            })
        
        return orders
    
    def start(self) -> None:
        """
        Démarre le gestionnaire d'ordres.
        """
        logger.info("🚀 Démarrage du gestionnaire d'ordres...")
        
        # Démarrer le moniteur de prix
        self.price_monitor.start()
        
        # Les ordres viennent uniquement via l'API REST du coordinator
        
        # Démarrer le service de réconciliation
        self.reconciliation_service.start()
        
        # Force une réconciliation initiale pour nettoyer les cycles fantômes
        # Utiliser un thread séparé pour éviter de bloquer le démarrage
        def delayed_reconciliation():
            time.sleep(5)  # Attendre 5 secondes après le démarrage
            try:
                logger.info("🔄 Lancement de la réconciliation initiale...")
                self.reconciliation_service.force_reconciliation()
            except Exception as e:
                logger.error(f"❌ Erreur lors de la réconciliation initiale: {str(e)}")
        
        threading.Thread(target=delayed_reconciliation, daemon=True).start()
        
        logger.info("✅ Gestionnaire d'ordres démarré")
    
    def stop(self) -> None:
        """
        Arrête le gestionnaire d'ordres.
        """
        logger.info("Arrêt du gestionnaire d'ordres...")
        
        # Plus de processeur de signaux - architecture REST uniquement
        
        # Arrêter le moniteur de prix
        self.price_monitor.stop()
        
        # Arrêter le service de réconciliation
        self.reconciliation_service.stop()
        
        # Fermer le gestionnaire de cycles
        self.cycle_manager.close()
        
        logger.info("✅ Gestionnaire d'ordres arrêté")