"""
Module de gestion de compte Binance.
Fournit des fonctions pour interagir avec l'API Binance et récupérer les données de compte.
Version optimisée avec gestion d'erreur robuste, retry automatique et cache.
"""
import logging
import time
import hmac
import hashlib
import requests  # type: ignore
from typing import Dict, List, Any, Optional, Union
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Configuration du logging
logger = logging.getLogger(__name__)

class BinanceApiError(Exception):
    """Erreur spécifique à l'API Binance."""
    def __init__(self, message, status_code=None, response_text=None):
        self.message = message
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(self.message)

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
        self.request_timeout = 30  # Timeout en secondes pour les requêtes HTTP
        
        # Cache des prix
        self._prices_cache: Dict[str, float] = {}
        self._prices_cache_time = 0
        self._prices_cache_ttl = 60  # Durée de vie du cache en secondes
        
        logger.info(f"BinanceAccountManager initialisé pour {base_url}")
    
    def _generate_signature(self, query_string: str) -> str:
        """
        Génère une signature HMAC-SHA256 pour une requête API Binance.
    
        Args:
            query_string: Chaîne de requête à signer
        
        Returns:
            Signature hexadécimale
        """
        # S'assurer que le secret est valide
        if not self.api_secret:
            logger.error("API Secret est vide ou invalide")
            raise ValueError("API Secret invalide")
    
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
        return signature
    
    def _make_request(self, endpoint: str, method: str = "GET", signed: bool = False, 
                     params: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Effectue une requête vers l'API Binance avec retry automatique.
        
        Args:
            endpoint: Point de terminaison de l'API
            method: Méthode HTTP (GET, POST, etc.)
            signed: Si True, la requête nécessite une signature
            params: Paramètres de la requête
            max_retries: Nombre maximum de tentatives
            
        Returns:
            Réponse JSON de l'API
            
        Raises:
            BinanceApiError: Si la requête échoue après toutes les tentatives
        """
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        params = params or {}
        
        retry_count = 0
        last_exception = None
        
        while retry_count < max_retries:
            try:
                # Requête signée
                if signed:
                    # Générer un nouveau timestamp à chaque tentative
                    params['timestamp'] = int(time.time() * 1000)
                    params['recvWindow'] = self.recvWindow
                    
                    # Construire la chaîne de requête
                    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                    
                    # Ajouter la signature
                    signature = self._generate_signature(query_string)
                    params['signature'] = signature
                
                # Effectuer la requête avec timeout
                if method == "GET":
                    response = requests.get(url, headers=headers, params=params, timeout=self.request_timeout)
                elif method == "POST":
                    response = requests.post(url, headers=headers, params=params, timeout=self.request_timeout)
                elif method == "DELETE":
                    response = requests.delete(url, headers=headers, params=params, timeout=self.request_timeout)
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
                except Exception:
                    error_message = f"{error_message}: {e.response.text}"
                
                # Déterminer si on doit réessayer
                status_code = e.response.status_code
                retry = False
                
                # Réessayer pour les erreurs 429 (rate limit) et 5xx (serveur)
                if status_code == 429 or status_code >= 500:
                    retry = True
                
                if retry and retry_count < max_retries - 1:
                    retry_count += 1
                    wait_time = min(2 ** retry_count, 60)  # Backoff exponentiel, max 60s
                    logger.warning(f"⚠️ {error_message}. Nouvelle tentative {retry_count}/{max_retries} dans {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                # Erreur terminale
                last_exception = BinanceApiError(
                    error_message, 
                    status_code=status_code,
                    response_text=e.response.text
                )
                break
                
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                # Toujours réessayer pour les erreurs de connexion/timeout
                if retry_count < max_retries - 1:
                    retry_count += 1
                    wait_time = min(2 ** retry_count, 60)
                    logger.warning(f"⚠️ Erreur de connexion: {str(e)}. Nouvelle tentative {retry_count}/{max_retries} dans {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                last_exception = BinanceApiError(f"Erreur de connexion après {max_retries} tentatives: {str(e)}")
                break
            
            except Exception as e:
                # Ne pas réessayer les autres erreurs
                last_exception = BinanceApiError(f"Erreur inattendue: {str(e)}")
                logger.error(traceback.format_exc())
                break
        
        # Si on arrive ici, c'est que toutes les tentatives ont échoué
        if last_exception:
            logger.error(f"❌ Échec de la requête à l'API Binance après {retry_count + 1} tentatives: {str(last_exception)}")
            raise last_exception
        
        # Ce code ne devrait jamais être atteint
        raise BinanceApiError("Erreur inconnue lors de la requête à l'API Binance")
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Récupère les informations du compte.
        
        Returns:
            Informations du compte
            
        Raises:
            BinanceApiError: Si la requête échoue
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
            BinanceApiError: Si la requête échoue
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
            
        except BinanceApiError as e:
            logger.error(f"❌ Erreur lors de la récupération des balances: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Erreur inattendue lors de la récupération des balances: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def get_ticker_prices(self, use_cache: bool = True) -> Dict[str, float]:
        """
        Récupère les prix actuels de tous les symboles.
        Utilise un cache pour améliorer les performances.
        
        Args:
            use_cache: Si True, utilise le cache quand il est disponible
            
        Returns:
            Dictionnaire {symbole: prix}
            
        Raises:
            BinanceApiError: Si la requête échoue
        """
        # Vérifier le cache
        current_time = time.time()
        if use_cache and self._prices_cache and (current_time - self._prices_cache_time < self._prices_cache_ttl):
            logger.debug(f"Utilisation du cache des prix ({len(self._prices_cache)} symboles)")
            return self._prices_cache
        
        try:
            endpoint = "/api/v3/ticker/price"
            response = self._make_request(endpoint)
            assert isinstance(response, list), "Response should be a list for ticker/price endpoint"
            
            # Convertir la réponse en dictionnaire {symbole: prix}
            prices = {}
            for item in response:
                symbol = item.get("symbol")
                price = float(item.get("price", 0))
                prices[symbol] = price
            
            # Mettre à jour le cache
            self._prices_cache = prices
            self._prices_cache_time = int(current_time)
            
            logger.info(f"Récupéré les prix pour {len(prices)} symboles depuis Binance")
            return prices
            
        except BinanceApiError as e:
            logger.error(f"❌ Erreur lors de la récupération des prix: {str(e)}")
            
            # Si le cache existe et n'est pas trop vieux (moins de 5 minutes), l'utiliser malgré l'erreur
            if self._prices_cache and (current_time - self._prices_cache_time < 60):
                logger.warning(f"⚠️ Utilisation du cache des prix après échec de la requête ({len(self._prices_cache)} symboles)")
                return self._prices_cache
            
            raise
        except Exception as e:
            logger.error(f"❌ Erreur inattendue lors de la récupération des prix: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Si le cache existe et n'est pas trop vieux (moins de 5 minutes), l'utiliser malgré l'erreur
            if self._prices_cache and (current_time - self._prices_cache_time < 60):
                logger.warning(f"⚠️ Utilisation du cache des prix après erreur inattendue ({len(self._prices_cache)} symboles)")
                return self._prices_cache
            
            return {}
    
    @lru_cache(maxsize=100)
    def get_ticker_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations de marché pour un symbole spécifique.
        Utilise une mise en cache pour les requêtes fréquentes.
        
        Args:
            symbol: Symbole (ex: BTCUSDC)
            
        Returns:
            Informations de marché ou None si erreur
            
        Raises:
            BinanceApiError: Si la requête échoue
        """
        try:
            endpoint = "/api/v3/ticker/24hr"
            params = {"symbol": symbol}
            return self._make_request(endpoint, params=params)
            
        except BinanceApiError as e:
            logger.error(f"❌ Erreur lors de la récupération des informations pour {symbol}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Erreur inattendue lors de la récupération des informations pour {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupère les ordres ouverts.
        
        Args:
            symbol: Symbole spécifique (optionnel)
            
        Returns:
            Liste des ordres ouverts
            
        Raises:
            BinanceApiError: Si la requête échoue
        """
        endpoint = "/api/v3/openOrders"
        params = {}
        
        if symbol:
            params["symbol"] = symbol
        
        try:
            result = self._make_request(endpoint, signed=True, params=params)
            return result if isinstance(result, list) else [result] if result else []
            
        except BinanceApiError as e:
            logger.error(f"❌ Erreur lors de la récupération des ordres ouverts: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Erreur inattendue lors de la récupération des ordres ouverts: {str(e)}")
            logger.error(traceback.format_exc())
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
            BinanceApiError: Si la requête échoue
        """
        endpoint = "/api/v3/allOrders"
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        try:
            result = self._make_request(endpoint, signed=True, params=params)
            return result if isinstance(result, list) else [result] if result else []
            
        except BinanceApiError as e:
            logger.error(f"❌ Erreur lors de la récupération de l'historique des ordres: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Erreur inattendue lors de la récupération de l'historique des ordres: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def calculate_asset_values(self) -> List[Dict[str, Any]]:
        """
        Calcule la valeur de chaque actif en USDC.
        Utilise des chemins de conversion alternatifs si nécessaire.
        
        Returns:
            Liste des balances avec valeur en USDC
            
        Raises:
            BinanceApiError: Si la requête aux API échoue
        """
        try:
            # Récupérer les balances avec retry
            max_retries = 3
            retry = 0
            balances = None
            
            while retry < max_retries and not balances:
                try:
                    balances = self.get_balances()
                    if not balances and retry < max_retries - 1:
                        retry += 1
                        wait_time = 2 ** retry
                        logger.warning(f"⚠️ Aucune balance reçue, nouvelle tentative {retry}/{max_retries} dans {wait_time}s...")
                        time.sleep(wait_time)
                except Exception as e:
                    retry += 1
                    if retry >= max_retries:
                        raise
                    wait_time = 2 ** retry
                    logger.warning(f"⚠️ Erreur lors de la tentative {retry}/{max_retries}: {str(e)}. Nouvelle tentative dans {wait_time}s...")
                    time.sleep(wait_time)
            
            if not balances:
                logger.error("❌ Impossible de récupérer les balances après plusieurs tentatives")
                return []
            
            # Récupérer les prix actuels avec retry
            prices = None
            retry = 0
            
            while retry < max_retries and not prices:
                try:
                    prices = self.get_ticker_prices()
                    if not prices and retry < max_retries - 1:
                        retry += 1
                        wait_time = 2 ** retry
                        logger.warning(f"⚠️ Aucun prix reçu, nouvelle tentative {retry}/{max_retries} dans {wait_time}s...")
                        time.sleep(wait_time)
                except Exception as e:
                    retry += 1
                    if retry >= max_retries:
                        raise
                    wait_time = 2 ** retry
                    logger.warning(f"⚠️ Erreur lors de la tentative {retry}/{max_retries}: {str(e)}. Nouvelle tentative dans {wait_time}s...")
                    time.sleep(wait_time)
            
            if not prices:
                logger.error("❌ Impossible de récupérer les prix après plusieurs tentatives")
                return balances  # Retourner les balances sans valeur USDC
            
            # Préparer les taux de conversion pour les chemins alternatifs
            conversion_rates = {
                'USDT_USDC': prices.get('USDTUSDC', 1.0),
                'BUSD_USDC': prices.get('BUSDUSDC', 1.0),
                'BTC_USDC': prices.get('BTCUSDC', None),
                'ETH_USDC': prices.get('ETHUSDC', None)
            }
            
            # Les paires de trading possibles (par ordre de priorité)
            trading_routes = [
                # Direct en USDC
                lambda asset: (f"{asset}USDC", 1.0),
                # Via USDT
                lambda asset: (f"{asset}USDT", conversion_rates['USDT_USDC']),
                # Via BUSD
                lambda asset: (f"{asset}BUSD", conversion_rates['BUSD_USDC']),
                # Via BTC
                lambda asset: (f"{asset}BTC", conversion_rates['BTC_USDC']),
                # Via ETH
                lambda asset: (f"{asset}ETH", conversion_rates['ETH_USDC'])
            ]
            
            # Utiliser un ThreadPoolExecutor pour le calcul en parallèle
            def process_balance(balance):
                asset = balance["asset"]
                total = balance["total"]
                
                # Cas particulier pour USDC
                if asset == "USDC":
                    balance["value_usdc"] = total
                    return balance
                
                # Essayer toutes les routes de trading possibles
                for route_func in trading_routes:
                    symbol, conversion_rate = route_func(asset)
                    
                    # Vérifier si cette route est valide
                    if symbol in prices and conversion_rate is not None:
                        asset_price = prices[symbol]
                        balance["value_usdc"] = total * asset_price * conversion_rate
                        return balance
                
                # Aucune route trouvée
                balance["value_usdc"] = None
                return balance
            
            # Traiter les balances en parallèle pour les grands portefeuilles
            if len(balances) > 20:
                with ThreadPoolExecutor(max_workers=10) as executor:
                    balances = list(executor.map(process_balance, balances))
            else:
                balances = [process_balance(balance) for balance in balances]
            
            # Compter les actifs avec valeur en USDC
            valued_count = sum(1 for b in balances if b.get("value_usdc") is not None)
            logger.info(f"✅ Valeurs en USDC calculées pour {valued_count}/{len(balances)} actifs")
            
            # Trier par valeur décroissante
            balances.sort(key=lambda b: b.get("value_usdc", 0) or 0, reverse=True)
            
            return balances
            
        except BinanceApiError as e:
            logger.error(f"❌ Erreur API Binance lors du calcul des valeurs: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Erreur inattendue lors du calcul des valeurs en USDC: {str(e)}")
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
        for balance in balances[:5]:  # Afficher les 5 premières pour éviter une sortie trop grande
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
        total_value = 0
        for balance in valued_balances:
            if "value_usdc" in balance and balance["value_usdc"] is not None:
                total_value += balance["value_usdc"]
                if balance["value_usdc"] > 10:  # N'afficher que les valeurs significatives
                    print(f"{balance['asset']}: {balance['total']} ({balance['value_usdc']:.2f} USDC)")
        
        print(f"Valeur totale estimée: {total_value:.2f} USDC")
        print("Test réussi!")
        
    except BinanceApiError as e:
        print(f"Erreur API Binance: {str(e)}")
        if hasattr(e, 'status_code'):
            print(f"Code d'erreur: {e.status_code}")
        if hasattr(e, 'response_text'):
            print(f"Réponse: {e.response_text}")
    except Exception as e:
        print(f"Erreur: {str(e)}")
        print(traceback.format_exc())