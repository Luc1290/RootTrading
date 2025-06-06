"""
Module de vérification et de gestion des poches de capital.
Vérifie la disponibilité des fonds dans les poches avant d'autoriser les trades.
"""
import logging
import threading
import requests
import json
import time
from shared.src.redis_client import RedisClient
import json
from typing import Dict, Any, Optional
from urllib.parse import urljoin

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import POCKET_CONFIG

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PocketChecker:
    """
    Vérifie et gère les poches de capital pour les trades.
    Communique avec le service Portfolio pour réserver et libérer des fonds.
    """
    
    def __init__(self, portfolio_api_url: str = "http://portfolio:8000"):
        """
        Initialise le vérificateur de poches.
        
        Args:
            portfolio_api_url: URL de l'API du service Portfolio
        """
        self.portfolio_api_url = portfolio_api_url
        
        # Configuration des poches
        self.pocket_config = POCKET_CONFIG
        self.active_pocket = "active"
        self.buffer_pocket = "buffer"
        self.safety_pocket = "safety"
        
        # Cache pour éviter trop d'appels API
        self.pocket_cache = {}
        self.cache_expiry = 5  # Réduit à 5 secondes pour éviter les données obsolètes
        self.last_cache_update = 0
        # --- Cache Redis ------------------------------------------------
        self.redis = RedisClient()
        self._redis_key = "roottrading:pockets:snapshot"
        # Réactiver Redis PubSub pour une synchronisation fiable
        threading.Thread(
            target=self._subscribe_updates,
            daemon=True
        ).start()
        
        # Utiliser Kafka ET Redis pour une redondance
        self._start_kafka_listener()

        
        logger.info(f"✅ PocketChecker initialisé - API Portfolio: {portfolio_api_url}")
    
    def _start_kafka_listener(self):
        """Démarre l'écoute des événements Kafka pour les mises à jour de pockets."""
        try:
            from shared.src.kafka_client import KafkaClientPool
            
            def kafka_listener():
                kafka = KafkaClientPool.get_instance()
                
                def handle_pocket_update(topic, message):
                    try:
                        logger.info(f"🔍 Message Kafka reçu sur {topic}: {str(message)[:200]}...")
                        data = json.loads(message) if isinstance(message, str) else message
                        event_type = data.get('type', '')
                        logger.info(f"🔍 Type d'événement: {event_type}")
                        
                        if event_type in ['pocket.updated', 'pocket.reserved', 'pocket.released']:
                            # Invalider le cache local
                            self.last_cache_update = 0
                            logger.info(f"📨 Cache invalidé par événement Kafka: {event_type}")
                            
                            # Si possible, mettre à jour directement depuis l'événement
                            if 'pockets' in data:
                                self._update_cache_from_event(data['pockets'])
                                logger.info(f"✅ Cache mis à jour directement depuis Kafka")
                        else:
                            logger.warning(f"⚠️ Événement Kafka ignoré - type: {event_type}")
                                
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement de l'événement Kafka: {e}")
                
                # S'abonner au topic des pockets
                kafka.consume(["portfolio.pockets"], handle_pocket_update)
                logger.info("📡 Écoute Kafka démarrée pour portfolio.pockets")
                
            # Démarrer dans un thread séparé
            thread = threading.Thread(target=kafka_listener, daemon=True)
            thread.start()
            
        except Exception as e:
            logger.warning(f"⚠️ Impossible de démarrer l'écoute Kafka: {e}")
    
    def _update_cache_from_event(self, pockets_data):
        """Met à jour le cache local avec les données reçues par événement."""
        try:
            if isinstance(pockets_data, list):
                self.pocket_cache = {p["pocket_type"]: p for p in pockets_data}
                self.last_cache_update = time.time()
                logger.debug("✅ Cache mis à jour depuis événement Kafka")
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du cache: {e}")

    def _make_request_with_retry(self, url, method="GET", json_data=None, params=None, max_retries=3, timeout=5.0):
        """
        Effectue une requête HTTP avec mécanisme de retry.
        
        Args:
            url: URL de la requête
            method: Méthode HTTP (GET, POST, PUT)
            json_data: Données JSON pour la requête
            params: Paramètres de la requête
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
                elif method == "PUT":
                    response = requests.put(url, json=json_data, params=params, timeout=timeout)
                else:
                    raise ValueError(f"Méthode non supportée: {method}")
                    
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                last_exception = e
                retry_count += 1
                wait_time = 0.5 * (2 ** retry_count)  # Backoff exponentiel: 1s, 2s, 4s...
                logger.warning(f"Tentative {retry_count}/{max_retries} échouée: {str(e)}. Nouvelle tentative dans {wait_time}s")
                time.sleep(wait_time)
        
        logger.error(f"Échec après {max_retries} tentatives: {str(last_exception)}")
        return None
        
    def _subscribe_updates(self) -> None:
        """Écoute le canal Redis pour invalider le cache local."""
        def _handler(channel, data):
            self.last_cache_update = 0   # force refresh au prochain call
            logger.debug("♻️ Cache local invalidé par message Redis")

        self.redis.subscribe("roottrading:pockets.update", _handler)

    def _load_from_redis(self) -> bool:
        """
        Tente de charger les poches depuis Redis.
        Retourne True si succès, False sinon.
        """
        raw = self.redis.get(self._redis_key)
        if not raw:
            return False

        try:
            # Si raw est déjà une chaîne (à cause de decode_responses=True), 
            # on doit faire json.loads
            if isinstance(raw, str):
                pockets = json.loads(raw)
            else:
                # Si c'est déjà un objet Python (liste/dict), on l'utilise directement
                pockets = raw
                
            # Transformer en dict {pocket_type: data}
            self.pocket_cache = {p["pocket_type"]: p for p in pockets}
            self.last_cache_update = time.time()
            logger.debug("✅ Poches chargées depuis Redis")
            return True
        except Exception as e:
            logger.warning(f"Impossible de décoder le snapshot Redis : {e}")
            return False

    
    def _refresh_cache(self) -> Optional[bool]:
        """
        Rafraîchit le cache des poches avec meilleure gestion des erreurs et timeouts.
        
        Returns:
            True si mise à jour réussie, None en cas d'erreur critique, False si pas nécessaire
        """
        now = time.time()
        # 1) Essayer Redis d'abord
        if self._load_from_redis():
            return True            # cache rempli → terminé
        
        # Si le cache n'a pas expiré, ne rien faire
        if now - self.last_cache_update <= self.cache_expiry:
            return False
        
        try:
            # Ajouter un timeout explicite
            response = requests.get(
                urljoin(self.portfolio_api_url, "/pockets"), 
                timeout=5.0  # Timeout de 5 secondes
            )
            response.raise_for_status()
            
            pockets = response.json()
            self.pocket_cache = {pocket["pocket_type"]: pocket for pocket in pockets}
            self.last_cache_update = now
            
            logger.info(f"Cache des poches mis à jour: {len(pockets)} poches")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Erreur lors de la récupération des poches: {str(e)}")
            # Invalider le cache en cas d'erreur critique pour forcer une nouvelle tentative
            if isinstance(e, (requests.ConnectionError, requests.Timeout)):
                logger.warning("Erreur critique, invalidation du cache")
                self.last_cache_update = 0
                return None
            # Conserver l'ancien cache si erreur
            return False
    
    def force_refresh(self) -> None:
        """
        Force un rafraîchissement immédiat du cache des poches.
        Méthode utile après des événements importants (cycles fermés, annulés, etc.)
        """
        self.last_cache_update = 0
        self._refresh_cache()
        logger.debug("♻️ Cache des poches forcé à se rafraîchir")
    
    def get_available_funds(self, pocket_type: str = "active") -> float:
        """
        Récupère les fonds disponibles dans une poche de manière plus robuste.
        
        Args:
            pocket_type: Type de poche ('active', 'buffer', 'safety')
            
        Returns:
            Montant disponible
        """
        # Tenter de rafraîchir le cache si nécessaire
        refresh_result = self._refresh_cache()
        
        # Si refresh_result est None, c'est qu'il y a eu une erreur de connexion
        # Dans ce cas, on peut retenter une deuxième fois après une courte pause
        if refresh_result is None:
            logger.warning(f"Échec du rafraîchissement du cache, nouvelle tentative après pause...")
            time.sleep(1.0)
            self._refresh_cache()
        
        if pocket_type in self.pocket_cache:
            available = self.pocket_cache[pocket_type]["available_value"]
            logger.info(f"Fonds disponibles dans la poche {pocket_type}: {available:.2f} USDC (depuis cache)")
            return available
        
        # Si la poche n'est pas dans le cache, essayer de la récupérer directement avec retry
        try:
            pockets_data = self._make_request_with_retry(
                urljoin(self.portfolio_api_url, f"/pockets"),
                timeout=5.0
            )
            
            if not pockets_data:
                logger.error(f"Impossible de récupérer les données des poches")
                return 0.0
                
            pockets = pockets_data
            
            for pocket in pockets:
                if pocket["pocket_type"] == pocket_type:
                    available = pocket["available_value"]
                    logger.info(f"Fonds disponibles dans la poche {pocket_type}: {available:.2f} USDC (depuis API)")
                    return available
            
            logger.warning(f"Poche {pocket_type} non trouvée")
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des fonds disponibles: {str(e)}")
            return 0.0
    
    def check_funds_availability(self, amount: float, pocket_type: str = "active") -> bool:
        """
        Vérifie si les fonds sont disponibles dans une poche.
        
        Args:
            amount: Montant nécessaire
            pocket_type: Type de poche ('active', 'buffer', 'safety')
            
        Returns:
            True si les fonds sont disponibles, False sinon
        """
        available = self.get_available_funds(pocket_type)
        
        # Ajouter une marge de sécurité de 1% pour tenir compte des fluctuations de prix
        required = amount * 1.01
        
        logger.info(f"Vérification de disponibilité: {required:.2f} USDC requis, {available:.2f} USDC disponibles dans la poche {pocket_type}")
        
        return available >= required
    
    def reserve_funds(self, amount: float, cycle_id: str, pocket_type: str = "active", asset: str = "USDC") -> bool:
        """
        Réserve des fonds dans une poche pour un cycle de trading.
        
        Args:
            amount: Montant à réserver
            cycle_id: ID du cycle de trading
            pocket_type: Type de poche ('active', 'buffer', 'safety')
            
        Returns:
            True si la réservation a réussi, False sinon
        """
        # Vérifier d'abord la disponibilité
        if not self.check_funds_availability(amount, pocket_type):
            logger.warning(f"❌ Fonds insuffisants dans la poche {pocket_type} pour réserver {amount:.2f} USDC")
            return False
        
        # Réserver les fonds avec retry
        try:
            url = urljoin(self.portfolio_api_url, f"/pockets/{pocket_type}/reserve")
            params = {"amount": amount, "cycle_id": cycle_id, "asset": asset}
            
            result = self._make_request_with_retry(url, method="POST", params=params)
            
            if not result:
                logger.error(f"❌ Échec de la réservation des fonds: aucune réponse")
                return False
            
            # Forcer un rafraîchissement immédiat du cache
            self.force_refresh()
            
            logger.info(f"✅ {amount:.2f} USDC réservés dans la poche {pocket_type} pour le cycle {cycle_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la réservation des fonds: {str(e)}")
            return False
    
    def release_funds(self, amount: float, cycle_id: str, pocket_type: str = "active", asset: str = "USDC") -> bool:
        """
        Libère des fonds réservés dans une poche.
        
        Args:
            amount: Montant à libérer
            cycle_id: ID du cycle de trading
            pocket_type: Type de poche ('active', 'buffer', 'safety')
            
        Returns:
            True si la libération a réussi, False sinon
        """
        try:
            url = urljoin(self.portfolio_api_url, f"/pockets/{pocket_type}/release")
            params = {"amount": amount, "cycle_id": cycle_id, "asset": asset}
            
            result = self._make_request_with_retry(url, method="POST", params=params)
            
            if not result:
                logger.error(f"❌ Échec de la libération des fonds: aucune réponse")
                return False
            
            # Forcer un rafraîchissement immédiat du cache
            self.force_refresh()
            
            logger.info(f"✅ {amount:.2f} USDC libérés dans la poche {pocket_type} pour le cycle {cycle_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la libération des fonds: {str(e)}")
            return False
    
    def determine_best_pocket(self, amount: float, asset: str = "USDC") -> Optional[str]:
        """
        Détermine la meilleure poche à utiliser pour un trade.
        Essaie d'abord la poche active, puis la tampon si nécessaire.
        
        Args:
            amount: Montant nécessaire
            
        Returns:
            Type de poche à utiliser ou None si aucune poche n'a assez de fonds
        """
        # Essayer la poche active en premier
        if self.check_funds_availability(amount, self.active_pocket):
            return self.active_pocket
        
        # Essayer la poche tampon en second
        if self.check_funds_availability(amount, self.buffer_pocket):
            logger.info(f"Utilisation de la poche tampon pour {amount:.2f} USDC (poche active insuffisante)")
            return self.buffer_pocket
        
        # NOUVEAU: Si aucune poche n'a de fonds, forcer une vérification directe (cache possiblement obsolète)
        logger.warning(f"⚠️ Aucune poche n'a de fonds selon le cache - Vérification directe...")
        self.last_cache_update = 0  # Forcer un refresh immédiat
        
        # Réessayer après avoir invalidé le cache
        if self.check_funds_availability(amount, self.active_pocket):
            logger.info(f"✅ Fonds trouvés dans la poche active après refresh du cache")
            return self.active_pocket
        
        if self.check_funds_availability(amount, self.buffer_pocket):
            logger.info(f"✅ Fonds trouvés dans la poche tampon après refresh du cache") 
            return self.buffer_pocket
        
        # Ne pas utiliser la poche de sécurité automatiquement
        # Elle devrait être utilisée manuellement ou dans des cas spécifiques
        
        logger.warning(f"❌ Aucune poche n'a suffisamment de fonds pour {amount:.2f} USDC")
        return None        
    
    def reallocate_funds(self) -> bool:
        """
        Déclenche une réallocation des fonds entre les poches avec mécanisme de retry.
        Utile après des changements importants dans le portefeuille.
        
        Returns:
            True si la réallocation a réussi, False sinon
        """
        try:
            # Obtenir la valeur totale du portefeuille avec retry
            portfolio_data = self._make_request_with_retry(
                urljoin(self.portfolio_api_url, "/summary"),
                timeout=5.0
            )
            
            if not portfolio_data:
                logger.error("Impossible de récupérer les données du portefeuille")
                return False
                
            total_value = portfolio_data.get("total_value", 0)
            
            if total_value <= 0:
                logger.warning(f"Valeur totale du portefeuille nulle ou négative: {total_value}, utilisation d'une valeur par défaut")
                total_value = 100.0  # Valeur par défaut comme dans votre code original
            
            # Demander la mise à jour de l'allocation des poches avec retry
            allocation_result = self._make_request_with_retry(
                urljoin(self.portfolio_api_url, "/pockets/allocation"),
                method="PUT",
                params={"total_value": total_value},
                timeout=5.0
            )
            
            if not allocation_result:
                logger.error("Échec de la mise à jour de l'allocation")
                return False
            
            # Invalider le cache
            self.last_cache_update = 0
            
            # Recharger le cache immédiatement pour éviter les incohérences
            self._refresh_cache()
            
            logger.info(f"✅ Réallocation des poches effectuée (valeur totale: {total_value:.2f} USDC)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la réallocation des fonds: {str(e)}")
            return False
    
    def reconcile_pockets(self, force=False) -> bool:
        """
        Réconcilie les poches avec les cycles actifs en base de données.
        Correction manuelle de la désynchronisation entre cycles et poches.
        
        Args:
            force: Si True, force la réconciliation même si les différences sont minimes
        
        Returns:
            True si la réconciliation a réussi, False sinon
        """
        try:
            # Récupérer les poches avec retry
            pockets_data = self._make_request_with_retry(
                urljoin(self.portfolio_api_url, "/pockets")
            )
            
            if not pockets_data:
                logger.error("Impossible de récupérer les données des poches")
                return False
                
            pockets = pockets_data
        
            # Récupérer les cycles actifs depuis le trader avec retry
            trader_api_url = os.getenv("TRADER_API_URL", "http://trader:5002")
            active_cycles_data = self._make_request_with_retry(
                urljoin(trader_api_url, "/orders")
            )
            
            if active_cycles_data is None:
                logger.error("Impossible de récupérer les cycles actifs")
                return False
            
            # Vérifier si les données des cycles sont valides
            if not isinstance(active_cycles_data, list):
                logger.error(f"Format des cycles actifs invalide: {type(active_cycles_data)}")
                active_cycles = []
            else:    
                # Filtrer les cycles terminés comme le fait CycleSyncMonitor
                terminal_statuses = {'completed', 'canceled', 'failed', 'error'}
                active_cycles = [
                    cycle for cycle in active_cycles_data 
                    if cycle.get('status', '').lower() not in terminal_statuses
                ]
        
            # Calculer le nombre total de cycles actifs
            active_cycle_count = len(active_cycles)
        
            # Calculer le nombre total de cycles dans les poches
            pocket_cycle_count = sum(p.get("active_cycles", 0) for p in pockets)
            
            # Calculer le montant total utilisé dans les trades
            trade_used_value = 0
            try:
                trade_used_value = sum(
                    float(cycle.get("quantity", 0)) * float(cycle.get("entry_price", 0))
                    for cycle in active_cycles
                    if "quantity" in cycle and "entry_price" in cycle
                )
            except Exception as e:
                logger.warning(f"Impossible de calculer le montant total des trades: {str(e)}")
            
            # Calculer le montant total utilisé dans les poches
            pocket_used_value = sum(float(p.get("used_value", 0)) for p in pockets)
            
            # Vérifier si une réconciliation est nécessaire
            cycles_synced = abs(active_cycle_count - pocket_cycle_count) <= 1
            
            # Vérifier la synchronisation des montants (avec une tolérance de 5%)
            amount_diff_percent = 0
            if trade_used_value > 0 and pocket_used_value > 0:
                amount_diff_percent = abs(trade_used_value - pocket_used_value) / max(trade_used_value, pocket_used_value) * 100
            elif trade_used_value > 0 and pocket_used_value == 0:
                amount_diff_percent = 100  # 100% de différence si une valeur est nulle
            elif pocket_used_value > 0 and trade_used_value == 0:
                amount_diff_percent = 100
            
            amounts_synced = amount_diff_percent <= 5.0
            
            # Si tout est synchronisé et que force est False, ne rien faire
            if cycles_synced and amounts_synced and not force:
                logger.info("✅ Poches déjà synchronisées, pas besoin de réconciliation")
                return True
            
            # Construire un dictionnaire des cycles par poche pour la réconciliation
            # Format: {pocket_type: [cycle_ids]}
            cycles_by_pocket = {"active": [], "buffer": [], "safety": []}
            
            # Assigner chaque cycle à sa poche
            for cycle in active_cycles:
                pocket = cycle.get("pocket", "active")  # Par défaut, utiliser la poche active
                if pocket not in cycles_by_pocket:
                    cycles_by_pocket[pocket] = []
                cycles_by_pocket[pocket].append(cycle.get("id"))
            
            # Calculer les montants par poche
            amounts_by_pocket = {"active": 0.0, "buffer": 0.0, "safety": 0.0}
            
            for cycle in active_cycles:
                pocket = cycle.get("pocket", "active")
                try:
                    symbol = cycle.get("symbol", "")
                    quantity = float(cycle.get("quantity", 0))
                    entry_price = float(cycle.get("entry_price", 0))
                    
                    # Calculer le montant en quote asset
                    amount = quantity * entry_price
                    
                    # Convertir en USDC si nécessaire
                    amount_usdc = self._convert_to_usdc(amount, symbol)
                    if amount_usdc > 0:
                        logger.debug(f"Cycle {cycle.get('id')}: {quantity} * {entry_price} = {amount:.8f} {symbol[-3:]} = {amount_usdc:.2f} USDC")
                        amount = amount_usdc
                    else:
                        logger.warning(f"Cycle {cycle.get('id')}: Impossible de convertir {amount:.8f} {symbol} en USDC")
                    
                    if pocket in amounts_by_pocket:
                        amounts_by_pocket[pocket] += amount
                except (ValueError, TypeError) as e:
                    logger.error(f"Erreur de calcul pour le cycle {cycle.get('id')}: {str(e)}")
                    continue
            
            # Préparer les données pour la réconciliation
            reconcile_data = {
                "cycles": cycles_by_pocket,
                "amounts": amounts_by_pocket,
                "force": force
            }
            
            logger.info(f"Demande de réconciliation avec données: {json.dumps(reconcile_data)}")
            
            # Envoyer la requête de réconciliation avec retry
            reconcile_result = self._make_request_with_retry(
                urljoin(self.portfolio_api_url, "/pockets/reconcile"),
                method="POST",
                json_data=reconcile_data,
                timeout=30.0  # Timeout plus long pour cette opération importante
            )
            
            if not reconcile_result:
                logger.error("❌ Échec de la réconciliation des poches")
                return False
                
            # Loguer les résultats de la réconciliation
            logger.info(f"✅ Réconciliation des poches réussie: {active_cycle_count} cycles, "
                    f"{trade_used_value:.2f} USDC")
            
            # Invalider le cache immédiatement
            self.last_cache_update = 0
            
            # Recharger le cache après réconciliation
            self._refresh_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la réconciliation des poches: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
    def check_pocket_synchronization(self) -> bool:
        """
        Vérifie si les poches sont synchronisées avec les trades actifs.
        Version améliorée qui déclenche une réconciliation avancée si nécessaire.
        
        Returns:
            True si la synchronisation est correcte, False sinon
        """
        try:
            # Récupérer les poches avec retry
            pockets_data = self._make_request_with_retry(
                urljoin(self.portfolio_api_url, "/pockets")
            )
            
            if not pockets_data:
                logger.error("Impossible de récupérer les données des poches")
                return False
                
            pockets = pockets_data
        
            # Récupérer les cycles actifs depuis le trader avec retry
            trader_api_url = os.getenv("TRADER_API_URL", "http://trader:5002")
            active_cycles_data = self._make_request_with_retry(
                urljoin(trader_api_url, "/orders")
            )
            
            if active_cycles_data is None:
                logger.error("Impossible de récupérer les cycles actifs")
                return False
                
            active_cycles = active_cycles_data
        
            # Calculer le nombre total de cycles actifs
            active_cycle_count = len(active_cycles)
            
            # Compter aussi les cycles avec des données valides pour diagnostic
            valid_cycle_count = len([
                cycle for cycle in active_cycles
                if ("quantity" in cycle and "entry_price" in cycle and 
                    float(cycle.get("quantity", 0)) > 0 and float(cycle.get("entry_price", 0)) > 0)
            ])
        
            # Calculer le nombre total de cycles dans les poches
            pocket_cycle_count = sum(p.get("active_cycles", 0) for p in pockets)
            
            # Calculer le montant total utilisé dans les trades
            # Filtrer les cycles avec des données valides (quantity > 0 et entry_price > 0)
            valid_cycles = [
                cycle for cycle in active_cycles
                if ("quantity" in cycle and "entry_price" in cycle and 
                    float(cycle.get("quantity", 0)) > 0 and float(cycle.get("entry_price", 0)) > 0)
            ]
            
            trade_used_value = 0
            try:
                for cycle in valid_cycles:
                    symbol = cycle.get("symbol", "")
                    quantity = float(cycle.get("quantity", 0))
                    entry_price = float(cycle.get("entry_price", 0))
                    amount = quantity * entry_price
                    
                    # Convertir en USDC
                    amount_usdc = self._convert_to_usdc(amount, symbol)
                    trade_used_value += amount_usdc
                    
            except Exception as e:
                logger.warning(f"Impossible de calculer le montant total des trades: {str(e)}")
            
            # Calculer le montant total utilisé dans les poches
            pocket_used_value = sum(float(p.get("used_value", 0)) for p in pockets)
            
            # Vérifier si les nombres correspondent (avec une marge d'erreur de 1)
            cycles_synced = abs(active_cycle_count - pocket_cycle_count) <= 1
            
            # Vérifier la synchronisation des montants (avec une tolérance de 5%)
            amount_diff_percent = 0
            if trade_used_value == 0 and pocket_used_value == 0:
                # Cas particulier: si les deux sont à 0, il n'y a pas de différence
                amount_diff_percent = 0
            elif trade_used_value > 0 and pocket_used_value > 0:
                amount_diff_percent = abs(trade_used_value - pocket_used_value) / max(trade_used_value, pocket_used_value) * 100
            elif trade_used_value > 0 and pocket_used_value == 0:
                amount_diff_percent = 100  # 100% de différence si une valeur est nulle
            elif pocket_used_value > 0 and trade_used_value == 0:
                amount_diff_percent = 100
            
            amounts_synced = amount_diff_percent <= 5.0
        
            if not cycles_synced or not amounts_synced:
                # Ne pas logger comme warning si la différence est de 1 cycle seulement
                log_level = logging.INFO if abs(active_cycle_count - pocket_cycle_count) == 1 else logging.WARNING
                
                # Collecter les IDs des cycles pour diagnostic
                active_cycle_ids = [cycle.get('id', 'unknown') for cycle in active_cycles]
                
                logger.log(
                    log_level,
                    f"{'ℹ️' if log_level == logging.INFO else '⚠️'} Différence détectée: "
                    f"{active_cycle_count} cycles actifs vs {pocket_cycle_count} dans les poches. "
                    f"Montant utilisé: {trade_used_value:.2f} vs {pocket_used_value:.2f} (diff: {amount_diff_percent:.2f}%)"
                )
                
                # Log détaillé pour diagnostic (seulement si différence > 1 cycle)
                if abs(active_cycle_count - pocket_cycle_count) > 1:
                    logger.debug(f"🔍 Cycles Trader: {active_cycle_ids}")
                    logger.debug(f"🔍 Cycles valides (avec prix/quantité): {len(valid_cycles)} sur {len(active_cycles)}")
                
                # Si la différence est de plus d'un cycle OU si les montants diffèrent de plus de 5%
                if abs(active_cycle_count - pocket_cycle_count) > 1 or amount_diff_percent > 5.0:
                    logger.info("🔄 Tentative de réconciliation automatique...")
                    # Utiliser notre nouvelle méthode de réconciliation avancée
                    reconciliation_success = self.reconcile_pockets()
                    
                    if reconciliation_success:
                        logger.info("✅ Réconciliation avancée des poches réussie")
                    else:
                        logger.error("❌ Échec de la réconciliation avancée des poches")
                    
                    return reconciliation_success
                else:
                    # Différence mineure acceptable
                    logger.debug("Différence mineure acceptable, pas de réconciliation nécessaire")
                    return True
        
            logger.info("✅ Poches correctement synchronisées")
            return True
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification de synchronisation: {str(e)}")
            return False
    
    def _convert_to_usdc(self, amount: float, symbol: str) -> float:
        """
        Convertit un montant en USDC selon la paire de trading.
        
        Args:
            amount: Le montant dans la devise de base
            symbol: Le symbole de la paire (ex: ETHBTC, BTCUSDC)
            
        Returns:
            Le montant converti en USDC
        """
        if symbol.endswith("USDC"):
            return amount
            
        if symbol.endswith("BTC"):
            try:
                response = requests.get(f"http://trader:5002/price/BTCUSDC", timeout=2.0)
                if response.status_code == 200:
                    btc_price = float(response.json().get('price', 0))
                    if btc_price > 0:
                        return amount * btc_price
            except Exception as e:
                logger.warning(f"Impossible d'obtenir le prix BTC: {e}")
                
        elif symbol.endswith("ETH"):
            try:
                response = requests.get(f"http://trader:5002/price/ETHUSDC", timeout=2.0)
                if response.status_code == 200:
                    eth_price = float(response.json().get('price', 0))
                    if eth_price > 0:
                        return amount * eth_price
            except Exception as e:
                logger.warning(f"Impossible d'obtenir le prix ETH: {e}")
                
        # Si conversion impossible, retourner 0 pour éviter les erreurs
        logger.warning(f"Impossible de convertir {amount} {symbol} en USDC")
        return 0

