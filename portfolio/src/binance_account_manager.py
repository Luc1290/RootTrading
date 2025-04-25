"""
Module de gestion de compte Binance.
Fournit des fonctions pour interagir avec l'API Binance et récupérer les données de compte.
"""
import logging
import time
import hmac
import hashlib
import requests
import json
from typing import Dict, List, Any, Optional
import traceback

# Configuration du logging
logger = logging.getLogger(__name__)

class BinanceAccountManager:
    """
    Gestionnaire de compte Binance.
    Permet de récupérer les balances et autres informations du compte.
    """
    
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.binance.com"):
        """
        Initialise le gestionnaire de compte Binance.
        
        Args:
            api_key: Clé API Binance
            api_secret: Clé secrète API Binance
            base_url: URL de base de l'API Binance
        """
        if not api_key or not api_secret:
            raise ValueError("Les clés API Binance sont requises")
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.recvWindow = 10000  # Fenêtre de réception (ms) pour les requêtes signées
        
        logger.info(f"BinanceAccountManager initialisé pour {base_url}")
    
    def _generate_signature(self, query_string: str) -> str:
        """
        Génère une signature HMAC-SHA256 pour une requête API Binance.
        
        Args:
            query_string: Chaîne de requête à signer
            
        Returns:
            Signature hexadécimale
        """
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint: str, method: str = "GET", signed: bool = False, 
                     params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Effectue une requête vers l'API Binance.
        
        Args:
            endpoint: Point de terminaison de l'API
            method: Méthode HTTP (GET, POST, etc.)
            signed: Si True, la requête nécessite une signature
            params: Paramètres de la requête
            
        Returns:
            Réponse JSON de l'API
            
        Raises:
            Exception: Si la requête échoue
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        params = params or {}
        
        # Requête signée
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = self.recvWindow
            
            # Construire la chaîne de requête
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            
            # Ajouter la signature
            signature = self._generate_signature(query_string)
            params['signature'] = signature
        
        try:
            # Effectuer la requête
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=headers, params=params)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, params=params)
            else:
                raise ValueError(f"Méthode HTTP non supportée: {method}")
            
            # Vérifier le statut de la réponse
            response.raise_for_status()
            
            # Analyser et retourner la réponse JSON
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            error_message = f"Erreur HTTP {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_message = f"{error_message}: {error_data.get('msg', str(error_data))}"
            except:
                error_message = f"{error_message}: {e.response.text}"
            
            logger.error(error_message)
            raise Exception(error_message)
        
        except Exception as e:
            logger.error(f"Erreur lors de la requête à l'API Binance: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Récupère les informations du compte.
        
        Returns:
            Informations du compte
            
        Raises:
            Exception: Si la requête échoue
        """
        endpoint = "/api/v3/account"
        return self._make_request(endpoint, signed=True)
    
    def get_balances(self) -> List[Dict[str, Any]]:
        """
        Récupère les balances du compte.
        Ne retourne que les actifs avec un solde non nul.
        
        Returns:
            Liste des balances non nulles
            
        Raises:
            Exception: Si la requête échoue
        """
        try:
            account_info = self.get_account_info()
            balances = []
            
            for balance in account_info.get("balances", []):
                free = float(balance.get("free", 0))
                locked = float(balance.get("locked", 0))
                total = free + locked
                
                # Ne conserver que les balances non nulles
                if total > 0:
                    balances.append({
                        "asset": balance.get("asset"),
                        "free": free,
                        "locked": locked,
                        "total": total
                    })
            
            logger.info(f"Récupéré {len(balances)} balances non nulles depuis Binance")
            return balances
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des balances: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def get_ticker_prices(self) -> Dict[str, float]:
        """
        Récupère les prix actuels de tous les symboles.
        
        Returns:
            Dictionnaire {symbole: prix}
            
        Raises:
            Exception: Si la requête échoue
        """
        try:
            endpoint = "/api/v3/ticker/price"
            response = self._make_request(endpoint)
            
            # Convertir la réponse en dictionnaire {symbole: prix}
            prices = {}
            for item in response:
                symbol = item.get("symbol")
                price = float(item.get("price", 0))
                prices[symbol] = price
            
            logger.info(f"Récupéré les prix pour {len(prices)} symboles depuis Binance")
            return prices
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des prix: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def get_ticker_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations de marché pour un symbole spécifique.
        
        Args:
            symbol: Symbole (ex: BTCUSDC)
            
        Returns:
            Informations de marché ou None si erreur
            
        Raises:
            Exception: Si la requête échoue
        """
        try:
            endpoint = "/api/v3/ticker/24hr"
            params = {"symbol": symbol}
            return self._make_request(endpoint, params=params)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations pour {symbol}: {str(e)}")
            return None
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupère les ordres ouverts.
        
        Args:
            symbol: Symbole spécifique (optionnel)
            
        Returns:
            Liste des ordres ouverts
            
        Raises:
            Exception: Si la requête échoue
        """
        endpoint = "/api/v3/openOrders"
        params = {}
        
        if symbol:
            params["symbol"] = symbol
        
        try:
            return self._make_request(endpoint, signed=True, params=params)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des ordres ouverts: {str(e)}")
            return []
    
    def get_order_history(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des ordres pour un symbole.
        
        Args:
            symbol: Symbole
            limit: Nombre maximum d'ordres à récupérer
            
        Returns:
            Liste des ordres
            
        Raises:
            Exception: Si la requête échoue
        """
        endpoint = "/api/v3/allOrders"
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        try:
            return self._make_request(endpoint, signed=True, params=params)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique des ordres: {str(e)}")
            return []
    
    def calculate_asset_values(self) -> List[Dict[str, Any]]:
        """
        Calcule la valeur de chaque actif en USDC.
        
        Returns:
            Liste des balances avec valeur en USDC
            
        Raises:
            Exception: Si la requête échoue
        """
        try:
            # Récupérer les balances
            balances = self.get_balances()
            
            # Récupérer les prix actuels
            prices = self.get_ticker_prices()
            
            # Calculer la valeur en USDC pour chaque actif
            for balance in balances:
                asset = balance["asset"]
                total = balance["total"]
                
                # Cas particulier pour USDC
                if asset == "USDC":
                    balance["value_usdc"] = total
                    continue
                
                # Chercher le prix en USDC
                symbol = f"{asset}USDC"
                if symbol in prices:
                    balance["value_usdc"] = total * prices[symbol]
                else:
                    # Essayer de trouver un chemin indirect via USDT
                    symbol_usdt = f"{asset}USDT"
                    usdt_usdc = "USDTUSDC"
                    
                    if symbol_usdt in prices and usdt_usdc in prices:
                        value_usdt = total * prices[symbol_usdt]
                        balance["value_usdc"] = value_usdt * prices[usdt_usdc]
                    else:
                        # Pas de prix disponible
                        balance["value_usdc"] = None
            
            logger.info(f"Valeurs en USDC calculées pour {len(balances)} actifs")
            return balances
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des valeurs en USDC: {str(e)}")
            logger.error(traceback.format_exc())
            return []

# Test simple de la classe
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Charger les variables d'environnement
    load_dotenv()
    
    # Configurer le logging
    logging.basicConfig(level=logging.INFO)
    
    # Récupérer les clés API depuis les variables d'environnement
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")
    
    if not api_key or not api_secret:
        print("Les clés API Binance ne sont pas configurées dans les variables d'environnement")
        exit(1)
    
    try:
        # Créer le gestionnaire de compte
        manager = BinanceAccountManager(api_key, api_secret)
        
        # Récupérer les balances
        balances = manager.get_balances()
        print(f"Balances récupérées: {len(balances)}")
        
        # Afficher les balances
        for balance in balances:
            print(f"{balance['asset']}: {balance['total']}")
        
        # Récupérer les prix
        prices = manager.get_ticker_prices()
        print(f"Prix récupérés: {len(prices)}")
        
        # Afficher quelques prix
        for symbol, price in list(prices.items())[:5]:
            print(f"{symbol}: {price}")
        
        # Calculer les valeurs en USDC
        valued_balances = manager.calculate_asset_values()
        print(f"Balances avec valeur en USDC: {len(valued_balances)}")
        
        # Afficher les valeurs en USDC
        for balance in valued_balances:
            if "value_usdc" in balance and balance["value_usdc"] is not None:
                print(f"{balance['asset']}: {balance['total']} ({balance['value_usdc']:.2f} USDC)")
        
        print("Test réussi!")
        
    except Exception as e:
        print(f"Erreur: {str(e)}")
        print(traceback.format_exc())