"""
Client centralisé pour les appels aux services externes.
Évite la duplication de code et centralise la gestion des erreurs et retry.
"""
import logging
import requests
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ServiceEndpoint:
    """Configuration d'un endpoint de service."""
    service_name: str
    base_url: str
    timeout: float = 5.0
    max_retries: int = 3


class CircuitBreaker:
    """Circuit breaker pour éviter les appels répétés à des services en échec."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call_succeeded(self):
        """Enregistre un appel réussi."""
        self.failure_count = 0
        self.state = "CLOSED"
        
    def call_failed(self):
        """Enregistre un appel échoué."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker OPEN après {self.failure_count} échecs")
            
    def can_execute(self) -> bool:
        """Vérifie si on peut exécuter un appel."""
        if self.state == "CLOSED":
            return True
            
        if self.state == "OPEN":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.reset_timeout):
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker passé en HALF_OPEN")
                return True
            return False
            
        # HALF_OPEN
        return True


class ServiceClient:
    """
    Client centralisé pour tous les appels aux services externes.
    Gère les retry, circuit breakers et cache.
    """
    
    def __init__(self, trader_url: str = "http://trader:5002", 
                 portfolio_url: str = "http://portfolio:8000",
                 analyzer_url: str = "http://analyzer:8001"):
        """
        Initialise le client de services.
        
        Args:
            trader_url: URL du service Trader
            portfolio_url: URL du service Portfolio
            analyzer_url: URL du service Analyzer
        """
        self.endpoints = {
            "trader": ServiceEndpoint("trader", trader_url, timeout=10.0),
            "portfolio": ServiceEndpoint("portfolio", portfolio_url, timeout=5.0),
            "analyzer": ServiceEndpoint("analyzer", analyzer_url, timeout=5.0)
        }
        
        # Circuit breakers par service
        self.circuit_breakers = {
            name: CircuitBreaker() for name in self.endpoints.keys()
        }
        
        # Cache simple avec TTL
        self._cache = {}
        self._cache_ttl = {}
        
    def _make_request(self, service: str, endpoint: str, method: str = "GET",
                     json_data: Dict = None, params: Dict = None) -> Optional[Dict[str, Any]]:
        """
        Effectue une requête HTTP avec retry et circuit breaker.
        
        Args:
            service: Nom du service (trader, portfolio, analyzer)
            endpoint: Endpoint à appeler (ex: /cycles)
            method: Méthode HTTP
            json_data: Données JSON à envoyer
            params: Paramètres de requête
            
        Returns:
            Réponse JSON ou None si échec
        """
        if service not in self.endpoints:
            logger.error(f"Service inconnu: {service}")
            return None
            
        circuit_breaker = self.circuit_breakers[service]
        if not circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker OPEN pour {service}, requête bloquée")
            return None
            
        service_config = self.endpoints[service]
        url = f"{service_config.base_url}{endpoint}"
        
        for attempt in range(service_config.max_retries):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    json=json_data,
                    params=params,
                    timeout=service_config.timeout
                )
                
                response.raise_for_status()
                circuit_breaker.call_succeeded()
                return response.json()
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout appelant {service}{endpoint} (tentative {attempt + 1})")
                time.sleep(0.5 * (attempt + 1))
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Erreur appelant {service}{endpoint}: {str(e)}")
                if attempt == service_config.max_retries - 1:
                    circuit_breaker.call_failed()
                time.sleep(0.5 * (attempt + 1))
                
        return None
    
    # === API Trader ===
    
    def get_active_cycles(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupère les cycles actifs depuis le trader.
        
        Args:
            symbol: Filtrer par symbole (optionnel)
            
        Returns:
            Liste des cycles actifs
        """
        cache_key = f"cycles:{symbol or 'all'}"
        
        # Vérifier le cache
        if cache_key in self._cache:
            if time.time() - self._cache_ttl[cache_key] < 2.0:  # Cache 2 secondes
                return self._cache[cache_key]
        
        params = {"symbol": symbol} if symbol else None
        response = self._make_request("trader", "/cycles", params=params)
        
        if response:
            if "cycles" in response:
                cycles = response["cycles"]
            elif "data" in response and "cycles" in response["data"]:
                cycles = response["data"]["cycles"]
            else:
                cycles = response if isinstance(response, list) else []
            # Mettre en cache
            self._cache[cache_key] = cycles
            self._cache_ttl[cache_key] = time.time()
            return cycles
            
        return []
    
    def create_order(self, order_data: Dict[str, Any]) -> Optional[str]:
        """
        Crée un nouvel ordre via le trader.
        
        Args:
            order_data: Données de l'ordre
            
        Returns:
            ID de l'ordre créé ou None
        """
        response = self._make_request("trader", "/order", method="POST", json_data=order_data)
        
        if response and response.get("order_id"):
            return response["order_id"]
            
        return None
    
    def reinforce_cycle(self, cycle_id: str, symbol: str, side: str, 
                       quantity: float, price: Optional[float] = None,
                       metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Renforce un cycle existant via DCA.
        
        Args:
            cycle_id: ID du cycle à renforcer
            symbol: Symbole de trading
            side: Côté de l'ordre (BUY/SELL)
            quantity: Quantité à ajouter
            price: Prix limite (optionnel)
            metadata: Métadonnées supplémentaires
            
        Returns:
            Résultat du renforcement
        """
        reinforce_data = {
            "cycle_id": cycle_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity
        }
        
        if price:
            reinforce_data["price"] = price
            
        if metadata:
            reinforce_data["metadata"] = metadata
            
        response = self._make_request("trader", "/reinforce", method="POST", json_data=reinforce_data)
        
        if response:
            return response
        else:
            return {"success": False, "error": "Service indisponible"}
    
    def close_cycle(self, cycle_id: str, close_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Ferme un cycle existant via le trader.
        
        Args:
            cycle_id: ID du cycle à fermer
            close_data: Données de fermeture (reason, price optionnel)
            
        Returns:
            Résultat de la fermeture
        """
        endpoint = f"/close/{cycle_id}"
        
        response = self._make_request("trader", endpoint, method="POST", json_data=close_data or {})
        
        if response:
            return response
        else:
            return {"success": False, "error": "Service indisponible"}
    
    def close_cycle_accounting(self, cycle_id: str, price: float, reason: str = "Fermeture comptable") -> Dict[str, Any]:
        """
        Ferme un cycle de manière comptable sans ordre réel.
        
        Args:
            cycle_id: ID du cycle à fermer
            price: Prix pour le calcul du P&L
            reason: Raison de la fermeture
            
        Returns:
            Résultat de la fermeture
        """
        endpoint = f"/close_accounting/{cycle_id}"
        close_data = {
            "price": price,
            "reason": reason
        }
        
        response = self._make_request("trader", endpoint, method="POST", json_data=close_data)
        
        if response:
            return response
        else:
            return {"success": False, "error": "Service indisponible"}
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Récupère les prix actuels depuis le trader.
        
        Args:
            symbols: Liste des symboles
            
        Returns:
            Dict {symbol: price}
        """
        response = self._make_request("trader", "/prices", params={"symbols": ",".join(symbols)})
        return response.get("prices", {}) if response else {}
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Récupère le prix actuel d'un symbole.
        
        Args:
            symbol: Symbole à vérifier
            
        Returns:
            Prix actuel ou None
        """
        prices = self.get_current_prices([symbol])
        return prices.get(symbol)
    
    # === API Portfolio ===
    
    def get_portfolio_balance(self, asset: str) -> Optional[float]:
        """
        Récupère le solde d'un actif depuis le portfolio.
        
        Args:
            asset: Nom de l'actif (ex: USDC)
            
        Returns:
            Solde disponible ou None
        """
        response = self._make_request("portfolio", f"/balance/{asset}")
        
        if response and "free" in response:
            return response.get("free", 0.0)
            
        return None
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Récupère le résumé du portfolio.
        
        Returns:
            Résumé du portfolio
        """
        cache_key = "portfolio:summary"
        
        # Cache de 5 secondes pour le résumé
        if cache_key in self._cache:
            if time.time() - self._cache_ttl[cache_key] < 5.0:
                return self._cache[cache_key]
        
        response = self._make_request("portfolio", "/summary")
        
        if response:
            self._cache[cache_key] = response
            self._cache_ttl[cache_key] = time.time()
            return response
            
        return {}
    
    def check_balance_for_trade(self, symbol: str, side: str, amount: float) -> Dict[str, Any]:
        """
        Vérifie si les balances sont suffisantes pour un trade.
        Utilise les endpoints existants du portfolio pour vérifier.

        Args:
            symbol: Symbole de trading
            side: BUY ou SELL
            amount: Montant en USDC (pour BUY) ou quantité (pour SELL)
            
        Returns:
            Dict avec can_trade, available_balance, required_amount
        """
        try:
            # Récupérer toutes les balances
            all_balances = self.get_all_balances()
            if not all_balances:
                return {"can_trade": False, "reason": "Impossible de récupérer les balances"}
            
            # Déterminer l'asset nécessaire
            if side.upper() == "BUY":
                # Pour BUY, on a besoin de l'asset de quote (USDC généralement)
                required_asset = "USDC" if symbol.endswith("USDC") else "USDT"
                required_amount = amount
            else:
                # Pour SELL, on a besoin de l'asset de base
                required_asset = symbol.replace("USDC", "").replace("USDT", "")
                required_amount = amount
            
            # Vérifier la balance disponible
            available_balance = all_balances.get(required_asset, {}).get("free", 0.0)
            
            can_trade = available_balance >= required_amount
            
            return {
                "can_trade": can_trade,
                "available_balance": available_balance,
                "required_amount": required_amount,
                "required_asset": required_asset,
                "reason": "Balance suffisante" if can_trade else f"Balance insuffisante: {available_balance} < {required_amount} {required_asset}"
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de balance: {str(e)}")
            return {"can_trade": False, "reason": f"Erreur: {str(e)}"}
        
    def get_all_balances(self) -> Dict[str, Dict[str, float]]:
        """
        Récupère toutes les balances depuis le portfolio.
        
        Returns:
            Dict {asset: {free: float}}
        """
        response = self._make_request("portfolio", "/balances")
        
        if response:
            # Response is a list of balance objects, convert to dict
            if isinstance(response, list):
                balances = {}
                for balance in response:
                    if isinstance(balance, dict) and "asset" in balance:
                        asset = balance["asset"]
                        balances[asset] = {
                            "free": balance.get("free", 0.0)
                        }
                return balances
            return response
            
        return {}
    
    # get_positions() supprimée - le coordinator n'a plus besoin de vérifier les positions
    
    # === Méthodes utilitaires ===
    
    def invalidate_cache(self, pattern: Optional[str] = None):
        """
        Invalide le cache.
        
        Args:
            pattern: Pattern pour invalider (ex: "cycles:*"). Si None, invalide tout.
        """
        if pattern is None:
            self._cache.clear()
            self._cache_ttl.clear()
        else:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._cache[key]
                del self._cache_ttl[key]
    
    def get_service_health(self) -> Dict[str, bool]:
        """
        Vérifie l'état de santé de tous les services.
        
        Returns:
            Dict {service_name: is_healthy}
        """
        health_status = {}
        
        for service_name in self.endpoints.keys():
            response = self._make_request(service_name, "/health")
            health_status[service_name] = response is not None
            
        return health_status