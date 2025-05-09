"""
Gestionnaire d'ordres pour le trader.
Reçoit les signaux, les valide, et crée des cycles de trading.
"""
import logging
import threading
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import queue

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import SYMBOLS, TRADE_QUANTITIES, TRADE_QUANTITY
from shared.src.redis_client import RedisClient
from shared.src.enums import OrderSide, SignalStrength, CycleStatus
from shared.src.schemas import StrategySignal, TradeOrder

from trader.src.cycle_manager import CycleManager
from trader.src.binance_executor import BinanceExecutor

# Configuration du logging
logger = logging.getLogger(__name__)

class OrderManager:
    """
    Gestionnaire d'ordres.
    Reçoit les signaux de trading, les valide, et crée des cycles de trading.
    """
    
    def __init__(self, symbols: List[str] = None, redis_client: RedisClient = None, 
                 cycle_manager: CycleManager = None):
        """
        Initialise le gestionnaire d'ordres.
        
        Args:
            symbols: Liste des symboles à surveiller
            redis_client: Client Redis préexistant (optionnel)
            cycle_manager: Gestionnaire de cycles préexistant (optionnel)
        """
        self.symbols = symbols or SYMBOLS
        self.redis_signal_client = redis_client or RedisClient()
        self.redis_price_client = RedisClient() 
        self.cycle_manager = cycle_manager or CycleManager()
        self.binance_executor = self.cycle_manager.binance_executor
        
        # Configuration du canal Redis pour les signaux
        self.signal_channel = "roottrading:analyze:signal"
        
        # Configuration du canal Redis pour les mises à jour de prix
        self.price_channels = [f"roottrading:market:data:{symbol.lower()}" for symbol in self.symbols]
        
        # File d'attente thread-safe pour les signaux
        self.signal_queue = queue.Queue()
        
        # Dictionnaire pour stocker les derniers prix par symbole
        self.last_prices: Dict[str, float] = {}
        self.last_price_update = time.time()
        
        # Thread pour le traitement des signaux
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # Configuration pour les pauses de trading
        self.paused_symbols = set()
        self.paused_strategies = set()
        self.pause_until = 0  # Timestamp pour la reprise automatique
        self.paused_all = False
        
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
            resume_time = time.time() + duration
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
    
    def log_system_status(self) -> None:
        """
        Journalise l'état du système pour diagnostiquer les problèmes.
        """
        try:
            # Récupérer et loguer les cycles actifs
            active_cycles = self.cycle_manager.get_active_cycles()
            logger.info(f"État du système - Cycles actifs: {len(active_cycles)}")
        
            for cycle in active_cycles:
                logger.info(f"Cycle {cycle.id}: {cycle.symbol} {cycle.status.value}, "
                       f"entrée à {cycle.entry_price}, quantité: {cycle.quantity}")
        
            # Loguer les derniers prix connus
            logger.info(f"Derniers prix connus: {len(self.last_prices)} symboles")
            for symbol, price in self.last_prices.items():
                logger.debug(f"Prix de {symbol}: {price}")
        
            # Vérifier l'état des abonnements Redis
            if hasattr(self.redis_signal_client, 'pubsub') and self.redis_signal_client.pubsub:
                channels = self.redis_signal_client.pubsub.channels.keys() if hasattr(self.redis_signal_client.pubsub, 'channels') else []
                logger.info(f"Abonnements Redis actifs: {len(channels)} canaux")
                for channel in channels:
                    logger.debug(f"Abonné à: {channel}")
    
        except Exception as e:
            logger.error(f"Erreur lors de la journalisation de l'état du système: {str(e)}")
    
    def _process_signal(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Callback pour traiter les signaux reçus de Redis.
        Valide et ajoute les signaux à la file d'attente pour traitement.
    
        Args:
            channel: Canal Redis d'où provient le signal
            data: Données du signal
        """
        try:
            # Logging pour debug - afficher les données brutes reçues en mode debug uniquement
            logger.debug(f"Signal brut reçu: {json.dumps(data)}")

            # 🚨 Filtres de protection : ignorer les données de prix ou autres messages non pertinents
            if not isinstance(data, dict) or "strategy" not in data or "side" not in data or "timestamp" not in data or "price" not in data:
                logger.debug(f"⏭️ Message ignoré: n'est pas un signal valide")
                return
        
            # Vérifier les champs obligatoires
            required_fields = ["strategy", "symbol", "side", "timestamp", "price"]
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                logger.warning(f"❌ Champs obligatoires manquants dans le signal: {missing_fields}")
                return
            
            # Convertir la chaîne ISO en objet datetime
            if "timestamp" in data and isinstance(data["timestamp"], str):
                try:
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                except ValueError:
                    logger.error(f"❌ Format de timestamp invalide: {data['timestamp']}")
                    return

            # Valider le signal avec Pydantic
            signal = StrategySignal(**data)
            
            # Vérifier si le trading est en pause pour ce signal
            if self.is_trading_paused(signal.symbol, signal.strategy):
                logger.info(f"⏸️ Signal ignoré: trading en pause pour {signal.symbol}/{signal.strategy}")
                return
        
            # Ajouter à la file d'attente pour traitement
            self.signal_queue.put(signal)
        
            logger.info(f"📨 Signal reçu: {signal.side} {signal.symbol} @ {signal.price} ({signal.strategy})")
    
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement du signal: {str(e)}")
            logger.error(f"Données reçues problématiques: {data}")
    
    def _process_price_update(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Callback pour traiter les mises à jour de prix reçues de Redis.
        Met à jour les derniers prix et vérifie les stop-loss/take-profit.
        
        Args:
            channel: Canal Redis d'où provient la mise à jour
            data: Données de prix
        """
        try:
            # Ne traiter que les chandeliers fermés
            if not data.get('is_closed', False):
                return
            
            symbol = data.get('symbol')
            price = data.get('close')
            
            if not symbol or price is None:
                return
            
            # Mettre à jour le dernier prix
            self.last_prices[symbol] = price
            self.last_price_update = time.time()
            
            # Traiter la mise à jour de prix dans le gestionnaire de cycles
            self.cycle_manager.process_price_update(symbol, price)
            
            logger.debug(f"💰 Prix mis à jour: {symbol} @ {price}")
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de la mise à jour de prix: {str(e)}")
    
    def _signal_processing_loop(self) -> None:
        """
        Boucle de traitement des signaux de trading.
        Cette méthode s'exécute dans un thread séparé.
        """
        logger.info("Démarrage de la boucle de traitement des signaux")
        
        self.last_status_log = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Récupérer un signal de la file d'attente avec timeout
                try:
                    signal = self.signal_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Vérifier à nouveau si le trading est en pause (pourrait avoir changé)
                if self.is_trading_paused(signal.symbol, signal.strategy):
                    logger.info(f"⏸️ Signal ignoré dans la boucle: trading en pause pour {signal.symbol}/{signal.strategy}")
                    self.signal_queue.task_done()
                    continue
                
                # Traiter le signal
                # Traiter le signal en toute sécurité
                try:
                    self.safe_handle_signal(signal)
                except Exception as e:
                    logger.error(f"❌ Erreur inattendue lors du traitement du signal sécurisé: {str(e)}")

                
                # Marquer la tâche comme terminée
                self.signal_queue.task_done()

                # Loguer l'état du système toutes les 2 minutes (120 secondes)
                current_time = time.time()
                if current_time - self.last_status_log > 120:
                    self.log_system_status()
                    self.last_status_log = current_time
                
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle de traitement des signaux: {str(e)}")
                time.sleep(1)  # Pause pour éviter une boucle d'erreur infinie
        
        logger.info("Boucle de traitement des signaux arrêtée")

    # Calculer le pourcentage minimal de changement de prix nécessaire pour être rentable
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
    
    def safe_handle_signal(self, signal: StrategySignal) -> None:
        """
        Version sécurisée du traitement de signal : ignore tout échec de création de cycle.
        """
        try:
            if signal.symbol not in self.symbols:
                logger.warning(f"⚠️ Signal ignoré: symbole non supporté {signal.symbol}")
                return

            if signal.strength in [SignalStrength.WEAK]:
                logger.info(f"⚠️ Signal ignoré: force insuffisante ({signal.strength})")
                return

            if isinstance(signal.side, str):
                signal.side = OrderSide(signal.side)

            current_price = self.last_prices.get(signal.symbol, signal.price)
            quantity = TRADE_QUANTITIES.get(signal.symbol, TRADE_QUANTITY)

            metadata = signal.metadata or {}
            target_price = float(metadata.get('target_price', current_price * 1.03 if signal.side == OrderSide.BUY else current_price * 0.97))
            stop_price = float(metadata.get('stop_price', current_price * 0.98 if signal.side == OrderSide.BUY else current_price * 1.02))
            trailing_delta = float(metadata['trailing_delta']) if 'trailing_delta' in metadata else None

            min_change = self.calculate_min_profitable_change(signal.symbol)
            target_price_percent = abs((target_price - current_price) / current_price * 100)
            if target_price_percent < min_change:
                logger.info(f"⚠️ Signal ignoré: gain potentiel {target_price_percent:.2f}% inférieur au seuil minimal {min_change:.2f}%")
                return
            
            # Protection anti-doublon via Redis
            signal_fingerprint = f"{signal.symbol}:{signal.strategy}:{signal.side.value}:{int(signal.price)}"
            lock_key = f"roottrading:signal_lock:{signal_fingerprint}"
            lock_ttl = 30  # secondes

            if self.redis_signal_client.get(lock_key):
                logger.warning(f"⛔ Signal ignoré (doublon): {signal_fingerprint}")
                return

            # Poser un verrou pour éviter les doublons
            self.redis_signal_client.set(lock_key, "1", ex=lock_ttl)

            pocket = metadata.get('pocket', 'active')
            cycle = self.cycle_manager.create_cycle(
                symbol=signal.symbol,
                strategy=signal.strategy,
                side=signal.side,
                price=current_price,
                quantity=quantity,
                pocket=pocket,
                target_price=target_price,
                stop_price=stop_price,
                trailing_delta=trailing_delta
            )

            if cycle is None:
                logger.warning(f"❌ Aucun cycle créé pour le signal {signal.symbol} ({signal.strategy})")
                return

            logger.info(f"✅ Nouveau cycle actif créé: {cycle.id}")

        except Exception as e:
            logger.error(f"❌ Erreur dans safe_handle_signal: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
    def start(self) -> None:
        """
        Démarre le gestionnaire d'ordres.
        S'abonne aux canaux Redis et lance les threads de traitement.
        """
        logger.info("🚀 Démarrage du gestionnaire d'ordres...")
        
        # Réinitialiser l'événement d'arrêt
        self.stop_event.clear()
        
        # S'abonner au canal des signaux
        self.redis_signal_client.subscribe(self.signal_channel, self._process_signal)
        logger.info(f"✅ Abonné au canal Redis des signaux: {self.signal_channel}")
        
        # S'abonner aux canaux des mises à jour de prix
        self.redis_price_client.subscribe(self.price_channels, self._process_price_update)
        logger.info(f"✅ Abonné aux canaux Redis des prix: {len(self.price_channels)} canaux")
        
        # Démarrer le thread de traitement des signaux
        self.processing_thread = threading.Thread(
            target=self._signal_processing_loop,
            daemon=True,
            name="SignalProcessor"
        )
        self.processing_thread.start()
        
        logger.info("✅ Gestionnaire d'ordres démarré")
    
    def stop(self) -> None:
        """
        Arrête le gestionnaire d'ordres.
        """
        logger.info("Arrêt du gestionnaire d'ordres...")
        
        # Signaler l'arrêt aux threads
        self.stop_event.set()
        
        # Attendre que le thread de traitement se termine
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            logger.info("Thread de traitement des signaux arrêté")
        
        # Se désabonner des canaux Redis
        self.redis_signal_client.unsubscribe()
        self.redis_signal_client.close()

        self.redis_price_client.unsubscribe()
        self.redis_price_client.close()
        
        # Fermer le gestionnaire de cycles
        self.cycle_manager.close()
        
        logger.info("✅ Gestionnaire d'ordres arrêté")
    
    def create_manual_order(self, symbol: str, side: OrderSide, quantity: float, 
                           price: Optional[float] = None) -> str:
        """
        Crée un ordre manuel (hors signal).
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            side: Côté de l'ordre (BUY ou SELL)
            quantity: Quantité à trader
            price: Prix (optionnel, sinon au marché)
            
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
            
            # Créer un cycle avec la stratégie "Manual"
            cycle = self.cycle_manager.create_cycle(
                symbol=symbol,
                strategy="Manual",
                side=side,
                price=price,
                quantity=quantity,
                pocket="active"  # Utiliser la poche active par défaut
            )
            
            if cycle:
                return cycle.id
            else:
                return "Échec de création du cycle"
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de l'ordre manuel: {str(e)}")
            return f"Erreur: {str(e)}"
    
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
                "side": "BUY" if cycle.status in [CycleStatus.ACTIVE_BUY, CycleStatus.WAITING_BUY] else "SELL",
                "entry_price": cycle.entry_price,
                "current_price": self.last_prices.get(cycle.symbol),
                "quantity": cycle.quantity,
                "target_price": cycle.target_price,
                "stop_price": cycle.stop_price,
                "created_at": cycle.created_at.isoformat() if cycle.created_at else None
            })
        
        return orders