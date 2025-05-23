"""
Module de gestion des signaux de trading.
Reçoit les signaux, les valide, et coordonne la création des cycles de trading.
"""
import logging
import json
import requests
import threading
import time
from typing import Dict, Any, Optional, List
import queue

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.redis_client import RedisClient
from shared.src.config import TRADING_MODE
from shared.src.enums import OrderSide, SignalStrength, CycleStatus
from coordinator.src.cycle_sync_monitor import CycleSyncMonitor
from shared.src.schemas import StrategySignal, TradeOrder

from coordinator.src.pocket_checker import PocketChecker

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SignalHandler:
    """
    Gestionnaire des signaux de trading.
    Reçoit les signaux, les valide, et coordonne la création des cycles de trading.
    """
    
    def __init__(self, trader_api_url: str = "http://trader:5002", 
                 portfolio_api_url: str = "http://portfolio:8000"):
        """
        Initialise le gestionnaire de signaux.
        
        Args:
            trader_api_url: URL de l'API du service Trader
            portfolio_api_url: URL de l'API du service Portfolio
        """
        self.trader_api_url = trader_api_url
        self.portfolio_api_url = portfolio_api_url
        
        # Client Redis pour les communications
        self.redis_client = RedisClient()
        self.redis_client.subscribe("roottrading:order:failed", self.handle_order_failed)
        
        # S'abonner aux événements de cycles pour rester synchronisé
        self.redis_client.subscribe("roottrading:cycle:created", self.handle_cycle_created)
        self.redis_client.subscribe("roottrading:cycle:closed", self.handle_cycle_closed)
        self.redis_client.subscribe("roottrading:cycle:canceled", self.handle_cycle_canceled)
        self.redis_client.subscribe("roottrading:cycle:failed", self.handle_cycle_failed)
        
        # Canal Redis pour les signaux
        self.signal_channel = "roottrading:analyze:signal"
        
        # File d'attente thread-safe pour les signaux
        self.signal_queue = queue.Queue()
        
        # Thread pour le traitement des signaux
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # Gestionnaire de poches
        self.pocket_checker = PocketChecker(portfolio_api_url)
        
        # Moniteur de synchronisation des cycles (solution définitive)
        self.sync_monitor = CycleSyncMonitor(
            trader_api_url=trader_api_url,
            check_interval=30  # Vérification toutes les 30 secondes
        )
        
        # Cache des prix actuels
        self.price_cache = {}
        
        # Configuration du mode de trading
        self.demo_mode = TRADING_MODE.lower() == 'demo'
        
        # Stratégies spéciales pour le filtrage
        self.filter_strategies = ['Ride_or_React_Strategy']
        self.market_filters = {}  # {symbol: {filter_data}}

        # Circuit breakers pour éviter les appels répétés à des services en échec
        self.trader_circuit = CircuitBreaker()
        self.portfolio_circuit = CircuitBreaker()
        
        # S'abonner aux notifications de mise à jour du portfolio (nouvelle fonctionnalité)
        self.redis_client.subscribe(
            "roottrading:notification:balance_updated", 
            self._handle_portfolio_update
        )
        
        logger.info(f"✅ SignalHandler initialisé en mode {'DÉMO' if self.demo_mode else 'RÉEL'}")
    
    def _handle_portfolio_update(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Gère les notifications de mise à jour du portfolio.
        Déclenche une mise à jour du cache et une réallocation.
        
        Args:
            channel: Canal Redis
            data: Données de notification
        """
        try:
            logger.info(f"📢 Notification de mise à jour du portfolio reçue: {data}")
            
            # Invalider le cache des poches
            if hasattr(self, 'pocket_checker') and self.pocket_checker:
                self.pocket_checker.last_cache_update = 0
                
                # Déclencher une réallocation des fonds
                logger.info("Réallocation des fonds suite à notification...")
                self.pocket_checker.reallocate_funds()
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de la notification: {str(e)}")
    
    def _process_signal(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Callback pour traiter les signaux reçus de Redis.
        Ajoute les signaux à la file d'attente pour traitement.
    
        Args:
            channel: Canal Redis d'où provient le signal
            data: Données du signal
        """
        try:
            # Convertir les chaînes de caractères en énumérations avant de créer le signal
            if 'side' in data and isinstance(data['side'], str):
                try:
                    data['side'] = OrderSide(data['side'])
                except ValueError:
                    logger.error(f"Valeur d'énumération invalide pour side: {data['side']}")
                    return
                
            if 'strength' in data and isinstance(data['strength'], str):
                try:
                    data['strength'] = SignalStrength(data['strength'])
                except ValueError:
                    logger.error(f"Valeur d'énumération invalide pour strength: {data['strength']}")
                    # Utiliser une valeur par défaut au lieu d'abandonner
                    data['strength'] = SignalStrength.MODERATE
        
            # Valider le signal
            signal = StrategySignal(**data)
        
            # Traiter les signaux de filtrage séparément
            if signal.strategy in self.filter_strategies:
                self._update_market_filters(signal)
                return
        
            # Ajouter à la file d'attente pour traitement
            self.signal_queue.put(signal)
        
            # Mettre à jour le cache des prix
            self.price_cache[signal.symbol] = signal.price
        
            logger.info(f"📨 Signal reçu: {signal.side} {signal.symbol} @ {signal.price} ({signal.strategy})")
    
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement du signal: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _update_market_filters(self, signal: StrategySignal) -> None:
        """
        Met à jour les filtres de marché basés sur des stratégies spéciales comme Ride or React.
        Version améliorée avec meilleure gestion de l'obsolescence.
        
        Args:
            signal: Signal de la stratégie de filtrage
        """
        if signal.strategy in self.filter_strategies:
            # Vérifier que les métadonnées sont présentes
            if not signal.metadata:
                logger.warning(f"Signal de filtrage sans métadonnées reçu pour {signal.symbol}, ignoré")
                return
            
            # Stocker les informations de mode dans le dictionnaire de filtres
            mode = signal.metadata.get('mode', 'react')
            action = signal.metadata.get('action', 'normal_trading')
            
            # Vérifier si les données sont cohérentes
            if mode not in ['ride', 'react', 'neutral']:
                logger.warning(f"Mode de filtrage inconnu: {mode}, utilisation de 'react' par défaut")
                mode = 'react'
            
            if action not in ['normal_trading', 'no_trading', 'buy_only', 'sell_only']:
                logger.warning(f"Action de filtrage inconnue: {action}, utilisation de 'normal_trading' par défaut")
                action = 'normal_trading'
            
            # Mise à jour du filtre avec les nouvelles données
            self.market_filters[signal.symbol] = {
                'mode': mode,
                'action': action,
                'updated_at': time.time(),
                'is_obsolete': False,
                'source': signal.strategy
            }
            
            # Si des infos supplémentaires sont disponibles, les stocker aussi
            if 'trend_strength' in signal.metadata:
                self.market_filters[signal.symbol]['strength'] = float(signal.metadata['trend_strength'])
            
            logger.info(f"🔍 Filtre de marché mis à jour pour {signal.symbol}: "
                    f"mode={mode}, action={action}")
            
            # Publier la mise à jour sur Redis pour informer les autres composants
            try:
                from shared.src.redis_client import RedisClient
                redis_client = RedisClient()
                redis_client.publish("roottrading:market:filters", {
                    "symbol": signal.symbol,
                    "mode": mode,
                    "action": action,
                    "updated_at": time.time(),
                    "source": signal.strategy
                })
            except Exception as e:
                logger.warning(f"⚠️ Impossible de publier la mise à jour de filtre sur Redis: {str(e)}")
    
    def _should_filter_signal(self, signal: StrategySignal) -> bool:
        """
        Détermine si un signal doit être filtré en fonction des conditions de marché.
        Version améliorée avec gestion de l'obsolescence des filtres.
        
        Args:
            signal: Signal à évaluer
            
        Returns:
            True si le signal doit être filtré (ignoré), False sinon
        """
        # Vérifier si nous avons des informations de filtrage pour ce symbole
        if signal.symbol not in self.market_filters:
            # Aucune information de filtrage, essayer de récupérer des données récentes
            self._refresh_market_filter(signal.symbol)
            return False  # Ne pas filtrer si pas de données
        
        filter_info = self.market_filters[signal.symbol]
        
        # Vérifier si les informations de filtrage sont récentes
        # Réduire à 15 minutes (900 secondes) au lieu de 30 minutes
        max_age = 900  # 15 minutes
        if time.time() - filter_info.get('updated_at', 0) > max_age:
            logger.warning(f"Informations de filtrage obsolètes pour {signal.symbol}, tentative de rafraîchissement")
            
            # Essayer de rafraîchir les données de filtrage
            refreshed = self._refresh_market_filter(signal.symbol)
            
            if not refreshed:
                # Si le rafraîchissement échoue, utiliser un mode de fallback basé sur la force du signal
                logger.warning(f"Impossible de rafraîchir les informations de filtrage pour {signal.symbol}, utilisation du mode de secours")
                
                # En mode de secours, n'ignorer que les signaux très faibles
                if signal.strength == SignalStrength.WEAK:
                    logger.info(f"Signal {signal.side} ignoré en mode de secours (force insuffisante)")
                    return True
                
                # Laisser passer les autres signaux
                return False
            
            # Récupérer les informations rafraîchies
            filter_info = self.market_filters[signal.symbol]
        
        # En mode "ride", filtrer certains signaux contre-tendance
        if filter_info.get('mode') == 'ride':
            # Si dans une tendance haussière forte, filtrer les signaux SELL (sauf très forts)
            if signal.side == OrderSide.SELL and signal.strength != SignalStrength.VERY_STRONG:
                logger.info(f"🔍 Signal {signal.side} filtré: marché en mode RIDE pour {signal.symbol}")
                return True
        # En mode "react", aucun filtrage supplémentaire n'est nécessaire
        
        # Si une action spécifique est recommandée
        if 'action' in filter_info:
            action = filter_info.get('action')
            
            # Si l'action est "no_trading", filtrer tous les signaux
            if action == 'no_trading':
                logger.info(f"🔍 Signal {signal.side} filtré: action 'no_trading' active pour {signal.symbol}")
                return True
            
            # Si l'action est "buy_only", filtrer les signaux de vente
            elif action == 'buy_only' and signal.side == OrderSide.SELL:
                logger.info(f"🔍 Signal {signal.side} filtré: seuls les achats sont autorisés pour {signal.symbol}")
                return True
            
            # Si l'action est "sell_only", filtrer les signaux d'achat
            elif action == 'sell_only' and signal.side == OrderSide.BUY:
                logger.info(f"🔍 Signal {signal.side} filtré: seules les ventes sont autorisées pour {signal.symbol}")
                return True
        
        # Si aucune condition de filtrage n'a été rencontrée
        return False
    
    def _refresh_market_filter(self, symbol: str) -> bool:
        """
        Tente de rafraîchir les informations de filtrage pour un symbole.
        
        Args:
            symbol: Symbole pour lequel rafraîchir les données
            
        Returns:
            True si le rafraîchissement a réussi, False sinon
        """
        try:
            # Vérifier si le circuit breaker est ouvert
            if not self.trader_circuit.can_execute():
                logger.warning(f"Circuit breaker actif, impossible de rafraîchir les filtres")
                return False
            
            # Récupérer les dernières données de marché
            url = f"{self.trader_api_url}/market/filter/{symbol}"
            filter_data = self._make_request_with_retry(url, timeout=2.0)
            
            if not filter_data:
                logger.warning(f"Aucune donnée de filtrage disponible pour {symbol}")
                return False
            
            # Mettre à jour le filtre avec les nouvelles données
            self.market_filters[symbol] = {
                'mode': filter_data.get('mode', 'react'),  # Mode par défaut: react
                'action': filter_data.get('action', 'normal_trading'),
                'strength': filter_data.get('trend_strength', 0.0),
                'updated_at': time.time()  # Mettre à jour le timestamp
            }
            
            logger.info(f"✅ Informations de filtrage rafraîchies pour {symbol}: mode={self.market_filters[symbol]['mode']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du rafraîchissement des filtres pour {symbol}: {str(e)}")
            
            # En cas d'échec, marquer le filtre comme obsolète mais ne pas le supprimer complètement
            if symbol in self.market_filters:
                # Conserver les anciennes données mais les marquer comme explicitement obsolètes
                self.market_filters[symbol]['is_obsolete'] = True
            
            return False
    
    def _calculate_trade_amount(self, signal: StrategySignal) -> float:
        """
        Calcule le montant à trader basé sur le signal.
        
        Args:
            signal: Signal de trading
            
        Returns:
            Montant en USDC à réserver
        """
        # Valeurs par défaut
        default_amount = 100.0  # 100 USDC par défaut
        
        # Ajuster en fonction de la force du signal
        if signal.strength == SignalStrength.WEAK:
            amount = default_amount * 0.5
        elif signal.strength == SignalStrength.MODERATE:
            amount = default_amount * 0.8
        elif signal.strength == SignalStrength.STRONG:
            amount = default_amount * 1.0
        elif signal.strength == SignalStrength.VERY_STRONG:
            amount = default_amount * 1.2
        else:
            amount = default_amount
        
        # TODO: Logique d'ajustement plus complexe basée sur le portefeuille total
        # et les limites de risque par trade
        
        return amount
    
    def _make_request_with_retry(self, url, method="GET", json_data=None, params=None, max_retries=3, timeout=5.0):
        """
        Effectue une requête HTTP avec mécanisme de retry.
        
        Args:
            url: URL de la requête
            method: Méthode HTTP (GET, POST, DELETE)
            json_data: Données JSON pour POST
            params: Paramètres de requête
            max_retries: Nombre maximum de tentatives
            timeout: Timeout en secondes
            
        Returns:
            Réponse JSON ou None en cas d'échec
        """
        retry_count = 0
        last_exception = None
        
        while retry_count < max_retries:
            try:
                if method == "GET":
                    response = requests.get(url, params=params, timeout=timeout)
                elif method == "POST":
                    response = requests.post(url, json=json_data, params=params, timeout=timeout)
                elif method == "DELETE":
                    response = requests.delete(url, params=params, timeout=timeout)
                else:
                    raise ValueError(f"Méthode non supportée: {method}")
                    
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                last_exception = e
                retry_count += 1
                wait_time = 0.5 * (2 ** retry_count)  # Backoff exponentiel
                logger.warning(f"Tentative {retry_count}/{max_retries} échouée: {str(e)}. Nouvelle tentative dans {wait_time}s")
                time.sleep(wait_time)
        
        logger.error(f"Échec après {max_retries} tentatives: {str(last_exception)}")
        return None
    
    def _create_trade_cycle(self, signal: StrategySignal) -> Optional[str]:
        """
        Crée un cycle de trading à partir d'un signal.

        Args:
            signal: Signal de trading validé

        Returns:
            ID du cycle créé ou None en cas d'échec
        """
        # Vérifier le circuit breaker pour le portfolio
        if not self.portfolio_circuit.can_execute():
            logger.warning(f"Circuit ouvert pour le service Portfolio, signal ignoré")
            return None

        try:
            # Calculer le montant à trader
            trade_amount = self._calculate_trade_amount(signal)
    
            # Déterminer la poche à utiliser
            try:
                pocket_type = self.pocket_checker.determine_best_pocket(trade_amount)
                # Appel au portfolio réussi
                self.portfolio_circuit.record_success()
            except Exception as e:
                self.portfolio_circuit.record_failure()
                logger.error(f"❌ Erreur lors de l'interaction avec Portfolio: {str(e)}")
                return None
    
            if not pocket_type:
                logger.warning(f"❌ Aucune poche disponible pour un trade de {trade_amount:.2f} USDC")
                return None
    
            # Convertir le montant en quantité
            quantity = trade_amount / signal.price
    
            # Calculer le stop-loss et take-profit
            stop_price = signal.metadata.get('stop_price')
            target_price = signal.metadata.get('target_price')
    
            # Préparer la requête pour le Trader
            # Important: Convertir les enums en chaînes explicitement
            order_data = {
                "symbol": signal.symbol,
                "side": signal.side.value if hasattr(signal.side, 'value') else str(signal.side),
                "quantity": quantity,
                "price": signal.price,
                "strategy": signal.strategy,
                "timestamp": int(time.time() * 1000)  # Un timestamp actuel en millisecondes
            }
    
            # Réserver les fonds dans la poche
            temp_cycle_id = f"temp_{int(time.time())}"
            try:
                reserved = self.pocket_checker.reserve_funds(trade_amount, temp_cycle_id, pocket_type)
                # Autre appel au portfolio réussi
                self.portfolio_circuit.record_success()
            except Exception as e:
                self.portfolio_circuit.record_failure()
                logger.error(f"❌ Erreur lors de la réservation des fonds: {str(e)}")
                return None
    
            if not reserved:
                logger.error(f"❌ Échec de réservation des fonds pour le trade")
                return None
    
            # Vérifier le circuit breaker pour le Trader
            if not self.trader_circuit.can_execute():
                logger.warning(f"Circuit ouvert pour le service Trader, libération des fonds réservés")
                self.pocket_checker.release_funds(trade_amount, temp_cycle_id, pocket_type)
                return None
    
            # Créer le cycle via l'API du Trader avec retry
            try:
                logger.info(f"Envoi de la requête au Trader: {order_data}")
                result = self._make_request_with_retry(
                    f"{self.trader_api_url}/order",
                    method="POST",
                    json_data=order_data,
                    timeout=10.0  # Timeout plus long pour la création de l'ordre
                )
                
                if not result:
                    logger.error("❌ Échec de la création du cycle: aucune réponse du Trader")
                    # Libérer les fonds réservés
                    self.pocket_checker.release_funds(trade_amount, temp_cycle_id, pocket_type)
                    return None
                
                cycle_id = result.get('order_id')
        
                # Appel au trader réussi
                self.trader_circuit.record_success()
        
                if not cycle_id:
                    logger.error("❌ Réponse invalide du Trader: pas d'ID de cycle")
                    # Libérer les fonds réservés
                    self.pocket_checker.release_funds(trade_amount, temp_cycle_id, pocket_type)
                    return None
        
                # Mettre à jour la réservation avec l'ID réel du cycle
                try:
                    self.pocket_checker.release_funds(trade_amount, temp_cycle_id, pocket_type)
                    self.pocket_checker.reserve_funds(trade_amount, cycle_id, pocket_type)
                    self.portfolio_circuit.record_success()
                except Exception as e:
                    self.portfolio_circuit.record_failure()
                    logger.error(f"❌ Erreur lors de la mise à jour de la réservation: {str(e)}")
                    # Tenter d'annuler le cycle créé
                    try:
                        self._make_request_with_retry(
                            f"{self.trader_api_url}/order/{cycle_id}",
                            method="DELETE"
                        )
                    except:
                        pass
                    return None
        
                logger.info(f"✅ Cycle de trading créé: {cycle_id} ({signal.side} {signal.symbol})")
                return cycle_id
            
            except requests.RequestException as e:
                self.trader_circuit.record_failure()
                logger.error(f"❌ Erreur lors de la création du cycle: {str(e)}")
                # Libérer les fonds réservés
                self.pocket_checker.release_funds(trade_amount, temp_cycle_id, pocket_type)
                return None
    
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création du cycle de trading: {str(e)}")
            return None
    
    def _signal_processing_loop(self) -> None:
        """
        Boucle de traitement des signaux de trading.
        Cette méthode s'exécute dans un thread séparé.
        """
        logger.info("Démarrage de la boucle de traitement des signaux")
        
        while not self.stop_event.is_set():
            try:
                # Récupérer un signal de la file d'attente avec timeout
                try:
                    signal = self.signal_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Vérifier si le signal doit être filtré
                if self._should_filter_signal(signal):
                    self.signal_queue.task_done()
                    continue
                
                # Vérifier la force du signal
                if signal.strength in [SignalStrength.WEAK]:
                    logger.info(f"⚠️ Signal ignoré: trop faible ({signal.strength})")
                    self.signal_queue.task_done()
                    continue
                
                # Créer un cycle de trading
                cycle_id = self._create_trade_cycle(signal)
                
                if cycle_id:
                    logger.info(f"✅ Trade exécuté pour le signal {signal.strategy} sur {signal.symbol}")
                else:
                    logger.warning(f"⚠️ Échec d'exécution du trade pour le signal {signal.strategy}")
                
                # Marquer la tâche comme terminée
                self.signal_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle de traitement des signaux: {str(e)}")
                time.sleep(1)  # Pause pour éviter une boucle d'erreur infinie
        
        logger.info("Boucle de traitement des signaux arrêtée")
    
    def start(self) -> None:
        """
        Démarre le gestionnaire de signaux.
        """
        logger.info("🚀 Démarrage du gestionnaire de signaux...")
        
        # S'abonner au canal des signaux
        self.redis_client.subscribe(self.signal_channel, self._process_signal)
        
        # Démarrer le thread de traitement des signaux
        self.stop_event.clear()
        self.processing_thread = threading.Thread(
            target=self._signal_processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        # Démarrer le moniteur de synchronisation
        self.sync_monitor.start()
        
        logger.info("✅ Gestionnaire de signaux démarré")
    
    def stop(self) -> None:
        """
        Arrête le gestionnaire de signaux.
        """
        logger.info("Arrêt du gestionnaire de signaux...")
        
        # Signaler l'arrêt aux threads
        self.stop_event.set()
        
        # Attendre que le thread de traitement se termine
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Arrêter le moniteur de synchronisation
        self.sync_monitor.stop()
        
        # Se désabonner du canal Redis
        self.redis_client.unsubscribe()
        
        logger.info("✅ Gestionnaire de signaux arrêté")

    def handle_order_failed(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite les notifications d'échec d'ordre.
    
        Args:
            channel: Canal Redis d'où provient la notification
            data: Données de la notification
        """
        try:
            cycle_id = data.get("cycle_id")
            symbol = data.get("symbol")
            reason = data.get("reason", "Raison inconnue")
        
            if not cycle_id:
                logger.warning("❌ Message d'échec d'ordre reçu sans cycle_id")
                return
            
            logger.info(f"⚠️ Ordre échoué pour le cycle {cycle_id}: {reason}")
        
            # Déterminer si c'est un cycle temporaire ou confirmé
            if cycle_id.startswith("temp_"):
                # Cycle temporaire, libérer les fonds
                amount = data.get("amount", 0)
                if amount > 0:
                    self.pocket_checker.release_funds(amount, cycle_id, "active")
                    logger.info(f"✅ {amount} USDC libérés pour le cycle temporaire {cycle_id} après échec")
            else:
                # Cycle confirmé, annuler le cycle via l'API Trader
                try:
                    self._make_request_with_retry(
                        f"{self.trader_api_url}/order/{cycle_id}",
                        method="DELETE"
                    )
                    logger.info(f"✅ Cycle {cycle_id} annulé après échec d'ordre")
                except Exception as e:
                    logger.error(f"❌ Erreur lors de l'annulation du cycle {cycle_id}: {str(e)}")
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de l'échec d'ordre: {str(e)}")
    
    def handle_cycle_created(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite la création d'un cycle pour maintenir la synchronisation.
        """
        cycle_id = data.get('cycle_id')
        logger.debug(f"📌 Cycle créé: {cycle_id}")
        # La réservation est déjà faite, on note juste l'événement
        
    def handle_cycle_closed(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite la fermeture d'un cycle et force une réconciliation des poches.
        """
        cycle_id = data.get('cycle_id')
        symbol = data.get('symbol')
        profit_loss = data.get('profit_loss', 0)
        
        logger.info(f"💰 Cycle fermé: {cycle_id} ({symbol}) - P&L: {profit_loss:.2f}")
        
        # Forcer une réconciliation pour mettre à jour les poches
        self.pocket_checker.force_refresh()
        
    def handle_cycle_canceled(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite l'annulation d'un cycle et libère les fonds.
        """
        cycle_id = data.get('cycle_id')
        logger.info(f"🚫 Cycle annulé: {cycle_id}")
        
        # Forcer une réconciliation pour libérer les fonds
        self.pocket_checker.force_refresh()
        
    def handle_cycle_failed(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite l'échec d'un cycle.
        """
        cycle_id = data.get('cycle_id')
        logger.info(f"❌ Cycle échoué: {cycle_id}")
        
        # Forcer une réconciliation
        self.pocket_checker.force_refresh()

class CircuitBreaker:
    """Circuit breaker pour éviter les appels répétés à des services en échec."""
    
    def __init__(self, max_failures=3, reset_timeout=60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.open_since = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def record_success(self):
        """Enregistre un succès et réinitialise le circuit."""
        self.failure_count = 0
        self.state = "CLOSED"
        self.open_since = None
    
    def record_failure(self):
        """Enregistre un échec et ouvre le circuit si nécessaire."""
        self.failure_count += 1
        if self.failure_count >= self.max_failures:
            self.state = "OPEN"
            self.open_since = time.time()
    
    def can_execute(self):
        """Vérifie si une opération peut être exécutée."""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            # Vérifier si le temps de reset est écoulé
            if time.time() - self.open_since > self.reset_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        
        # HALF_OPEN: permettre un essai
        return True