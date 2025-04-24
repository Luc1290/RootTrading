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
        self.cache_expiry = 60  # Secondes
        self.last_cache_update = 0
        
        logger.info(f"✅ PocketChecker initialisé - API Portfolio: {portfolio_api_url}")
    
    def _refresh_cache(self) -> None:
        """
        Rafraîchit le cache des poches si nécessaire.
        """
        now = time.time()
        
        # Si le cache a expiré, le mettre à jour
        if now - self.last_cache_update > self.cache_expiry:
            try:
                response = requests.get(urljoin(self.portfolio_api_url, "/pockets"))
                response.raise_for_status()
                
                pockets = response.json()
                self.pocket_cache = {pocket["pocket_type"]: pocket for pocket in pockets}
                self.last_cache_update = now
                
                logger.info(f"Cache des poches mis à jour: {len(pockets)} poches")
                
            except requests.RequestException as e:
                logger.error(f"Erreur lors de la récupération des poches: {str(e)}")
                # Conserver l'ancien cache si erreur
    
    def get_available_funds(self, pocket_type: str = "active") -> float:
        """
        Récupère les fonds disponibles dans une poche.
        
        Args:
            pocket_type: Type de poche ('active', 'buffer', 'safety')
            
        Returns:
            Montant disponible
        """
        self._refresh_cache()
        
        if pocket_type in self.pocket_cache:
            return self.pocket_cache[pocket_type]["available_value"]
        
        # Si la poche n'est pas dans le cache, essayer de la récupérer directement
        try:
            response = requests.get(urljoin(self.portfolio_api_url, f"/pockets"))
            response.raise_for_status()
            
            pockets = response.json()
            for pocket in pockets:
                if pocket["pocket_type"] == pocket_type:
                    return pocket["available_value"]
            
            logger.warning(f"Poche {pocket_type} non trouvée")
            return 0.0
            
        except requests.RequestException as e:
            logger.error(f"Erreur lors de la récupération des fonds disponibles: {str(e)}")
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
        
        # Réserver les fonds
        try:
            url = urljoin(self.portfolio_api_url, f"/pockets/{pocket_type}/reserve")
            params = {"amount": amount, "cycle_id": cycle_id}
            
            response = requests.post(url, params=params)
            response.raise_for_status()
            
            # Invalider le cache
            self.last_cache_update = 0
            
            logger.info(f"✅ {amount:.2f} USDC réservés dans la poche {pocket_type} pour le cycle {cycle_id}")
            return True
            
        except requests.RequestException as e:
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
            
            response = requests.post(url, params=params)
            response.raise_for_status()
            
            # Invalider le cache
            self.last_cache_update = 0
            
            logger.info(f"✅ {amount:.2f} USDC libérés dans la poche {pocket_type} pour le cycle {cycle_id}")
            return True
            
        except requests.RequestException as e:
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
        Déclenche une réallocation des fonds entre les poches.
        Utile après des changements importants dans le portefeuille.
        
        Returns:
            True si la réallocation a réussi, False sinon
        """
        try:
            # Obtenir la valeur totale du portefeuille
            portfolio_response = requests.get(urljoin(self.portfolio_api_url, "/summary"))
            portfolio_response.raise_for_status()
            
            portfolio_data = portfolio_response.json()
            total_value = portfolio_data["total_value"]
            
            # Demander la mise à jour de l'allocation des poches
            allocation_url = urljoin(self.portfolio_api_url, "/pockets/allocation")
            allocation_response = requests.put(allocation_url, params={"total_value": total_value})
            allocation_response.raise_for_status()
            
            # Invalider le cache
            self.last_cache_update = 0
            
            logger.info(f"✅ Réallocation des poches effectuée (valeur totale: {total_value:.2f} USDC)")
            return True
            
        except requests.RequestException as e:
            logger.error(f"❌ Erreur lors de la réallocation des fonds: {str(e)}")
            return False