"""
Module de vérification et de gestion des poches de capital.
Vérifie la disponibilité des fonds dans les poches avant d'autoriser les trades.
"""
import logging
import requests
import json
import time
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
        self.cache_expiry = 30  # Réduit de 60 à 30 secondes pour plus de réactivité
        self.last_cache_update = 0
        
        logger.info(f"✅ PocketChecker initialisé - API Portfolio: {portfolio_api_url}")

    def _make_request_with_retry(self, url, method="GET", params=None, max_retries=3, timeout=5.0):
        """
        Effectue une requête HTTP avec mécanisme de retry.
        
        Args:
            url: URL de la requête
            method: Méthode HTTP (GET, POST, PUT)
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
                    response = requests.post(url, params=params, timeout=timeout)
                elif method == "PUT":
                    response = requests.put(url, params=params, timeout=timeout)
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
    
    def _refresh_cache(self) -> Optional[bool]:
        """
        Rafraîchit le cache des poches avec meilleure gestion des erreurs et timeouts.
        
        Returns:
            True si mise à jour réussie, None en cas d'erreur critique, False si pas nécessaire
        """
        now = time.time()
        
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
    
    def reserve_funds(self, amount: float, cycle_id: str, pocket_type: str = "active") -> bool:
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
            params = {"amount": amount, "cycle_id": cycle_id}
            
            result = self._make_request_with_retry(url, method="POST", params=params)
            
            if not result:
                logger.error(f"❌ Échec de la réservation des fonds: aucune réponse")
                return False
            
            # Invalider le cache
            self.last_cache_update = 0
            
            logger.info(f"✅ {amount:.2f} USDC réservés dans la poche {pocket_type} pour le cycle {cycle_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la réservation des fonds: {str(e)}")
            return False
    
    def release_funds(self, amount: float, cycle_id: str, pocket_type: str = "active") -> bool:
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
            params = {"amount": amount, "cycle_id": cycle_id}
            
            result = self._make_request_with_retry(url, method="POST", params=params)
            
            if not result:
                logger.error(f"❌ Échec de la libération des fonds: aucune réponse")
                return False
            
            # Invalider le cache
            self.last_cache_update = 0
            
            logger.info(f"✅ {amount:.2f} USDC libérés dans la poche {pocket_type} pour le cycle {cycle_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la libération des fonds: {str(e)}")
            return False
    
    def determine_best_pocket(self, amount: float) -> Optional[str]:
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
        
    def check_pocket_synchronization(self) -> bool:
        """
        Vérifie si les poches sont synchronisées avec les trades actifs.
        Version améliorée avec vérification des montants.
    
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
                urljoin(trader_api_url, "/orders?status=active")
            )
            
            if not active_cycles_data:
                logger.error("Impossible de récupérer les cycles actifs")
                return False
                
            active_cycles = active_cycles_data
        
            # Calculer le nombre total de cycles actifs
            active_cycle_count = len(active_cycles)
        
            # Calculer le nombre total de cycles dans les poches
            pocket_cycle_count = sum(p.get("active_cycles", 0) for p in pockets)
            
            # Calculer le montant total utilisé dans les trades (si les données sont disponibles)
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
            
            # Vérifier si les nombres correspondent (avec une marge d'erreur de 1)
            cycles_synced = abs(active_cycle_count - pocket_cycle_count) <= 1
            
            # Vérifier la synchronisation des montants (avec une tolérance de 5%)
            amount_diff_percent = 0
            if trade_used_value > 0:
                amount_diff_percent = abs(trade_used_value - pocket_used_value) / trade_used_value * 100
            
            amounts_synced = amount_diff_percent <= 5.0
        
            if not cycles_synced or (trade_used_value > 0 and not amounts_synced):
                logger.warning(
                    f"⚠️ Désynchronisation détectée: "
                    f"{active_cycle_count} cycles actifs vs {pocket_cycle_count} dans les poches. "
                    f"Montant utilisé: {trade_used_value:.2f} vs {pocket_used_value:.2f} (diff: {amount_diff_percent:.2f}%)"
                )
            
                # Demander une réconciliation avec retry
                reconcile_result = self._make_request_with_retry(
                    urljoin(self.portfolio_api_url, "/pockets/reconcile"),
                    method="POST"
                )
                
                if reconcile_result:
                    logger.info("✅ Réconciliation des poches demandée")
                    
                    # Invalider le cache immédiatement
                    self.last_cache_update = 0
                    # Recharger le cache après réconciliation
                    self._refresh_cache()
                    
                    return False
                else:
                    logger.error("❌ Échec de la demande de réconciliation")
                    return False
        
            logger.info("✅ Poches correctement synchronisées")
            return True
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification de synchronisation: {str(e)}")
            return False