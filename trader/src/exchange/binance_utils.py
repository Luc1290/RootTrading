# trader/src/exchange/binance_utils.py
"""
Fonctions utilitaires pour interagir avec l'API Binance.
Contient les fonctions de bas niveau pour les requêtes, signatures, etc.
"""
import logging
import time
import hmac
import hashlib
import uuid
import requests
from decimal import Decimal, getcontext, ROUND_DOWN
from urllib.parse import urlencode
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union

from shared.src.enums import OrderSide, OrderStatus, TradeRole
from shared.src.schemas import TradeOrder, TradeExecution

# Configuration de la précision décimale
getcontext().prec = 28  # 28 chiffres significatifs suffisent largement

# Configuration du logging
logger = logging.getLogger(__name__)

class BinanceAPIError(Exception):
    """Exception personnalisée pour les erreurs de l'API Binance."""
    pass

class BinanceUtils:
    """
    Classe utilitaire pour les opérations de bas niveau avec l'API Binance.
    """
    # URLs de base de l'API Binance
    BASE_URL = "https://api.binance.com"
    API_V3 = "/api/v3"
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialise les utilitaires Binance.
        
        Args:
            api_key: Clé API Binance
            api_secret: Clé secrète Binance
        """
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Session HTTP pour les requêtes
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})
    
    def get_time_offset(self) -> int:
        """
        Calcule le décalage temporel entre l'horloge locale et le serveur Binance.
        
        Returns:
            Décalage en millisecondes (positif si le serveur est en avance)
        """
        try:
            server_time_url = f"{self.BASE_URL}{self.API_V3}/time"
            response = self.session.get(server_time_url)
            response.raise_for_status()
            
            server_time = response.json()['serverTime']
            local_time = int(time.time() * 1000)
            offset = server_time - local_time
            
            logger.info(f"⏱️ Décalage temporel avec Binance: {offset} ms")
            return offset
        except Exception as e:
            logger.warning(f"⚠️ Impossible de calculer le décalage temporel: {str(e)}")
            return 0
    
    def check_connectivity(self, demo_mode: bool = False) -> bool:
        """
        Vérifie la connectivité avec l'API Binance et les permissions de l'API.
        
        Args:
            demo_mode: Si True, ne vérifie que la connectivité de base.
            
        Returns:
            True si la connectivité est OK, False sinon
        """
        try:
            # Vérifier la connectivité générale
            ping_url = f"{self.BASE_URL}{self.API_V3}/ping"
            response = self.session.get(ping_url)
            response.raise_for_status()
            
            # Vérifier les informations du compte (nécessite des permissions)
            if not demo_mode:
                account_url = f"{self.BASE_URL}{self.API_V3}/account"
                timestamp = int(time.time() * 1000)
                params = {"timestamp": timestamp}
                params["signature"] = self.generate_signature(params)
                
                response = self.session.get(account_url, params=params)
                response.raise_for_status()
                
                account_info = response.json()
                permissions = account_info.get("permissions", [])
                can_trade = account_info.get("canTrade", False)
                can_withdraw = account_info.get("canWithdraw", False)
                can_deposit = account_info.get("canDeposit", False)

                logger.info(f"🔍 Permissions API : {permissions}")
                logger.info(f"🔒 canTrade={can_trade} | canWithdraw={can_withdraw} | canDeposit={can_deposit}")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Erreur de connexion à Binance: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Réponse: {e.response.text}")
            return False
    
    def generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Génère une signature HMAC-SHA256 pour authentifier les requêtes vers l'API Binance.
        
        Args:
            params: Paramètres de la requête
            
        Returns:
            Signature HMAC-SHA256 en hexadécimal
        """
        # Convertir tous les paramètres en strings et les encoder correctement
        try:
            # Convertir d'abord tous les paramètres en strings
            processed_params = {}
            for key, value in params.items():
                if isinstance(value, float):
                    # Gérer les valeurs flottantes sans notation scientifique
                    processed_params[key] = f"{value:.8f}".rstrip('0').rstrip('.') if '.' in f"{value:.8f}" else f"{value}"
                elif isinstance(value, bool):
                    # Convertir les booléens en lowercase string (true/false)
                    processed_params[key] = str(value).lower()
                else:
                    processed_params[key] = str(value)
            
            # Encoder les paramètres en query string
            query_string = urlencode(processed_params)
            # logger.debug(f"Génération de signature pour: {query_string}")  # Commenté pour réduire le bruit dans les logs
            
            # Générer la signature HMAC-SHA256
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return signature
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération de la signature: {str(e)}")
            raise
    
    def notify_order_failure(self, error, order_params, client_order_id=None):
        """
        Notifie d'un échec d'ordre via Redis.
        
        Args:
            error: L'erreur survenue
            order_params: Paramètres de l'ordre
            client_order_id: ID du client (optionnel)
        """
        try:
            # Extraire le cycle_id de manière plus fiable
            cycle_id = None
        
            # Si le client_order_id est fourni et commence par "entry_", extraire le cycle_id
            if client_order_id and client_order_id.startswith("entry_"):
                cycle_id = client_order_id.replace("entry_", "")
            # Sinon, essayer d'obtenir le newClientOrderId des paramètres
            elif order_params and "newClientOrderId" in order_params:
                client_id = order_params.get("newClientOrderId")
                if client_id and client_id.startswith("entry_"):
                    cycle_id = client_id.replace("entry_", "")
        
            # Si aucun cycle_id n'a été trouvé, générer un ID temporaire
            if not cycle_id:
                cycle_id = f"temp_{int(time.time() * 1000)}"

            notification = {
                "type": "order_failed",
                "cycle_id": cycle_id,
                "symbol": order_params.get("symbol") if order_params else None,
                "side": order_params.get("side") if order_params else None,
                "quantity": order_params.get("quantity") if order_params else None,
                "price": order_params.get("price") if order_params else None,
                "reason": f"Erreur Binance: {str(error)}",
                "timestamp": int(time.time() * 1000)
            }

            try:
                if order_params:
                    price = float(order_params.get("price", 0))
                    quantity = float(order_params.get("quantity", 0))
                    if price > 0 and quantity > 0:
                        notification["amount"] = price * quantity
            except (ValueError, TypeError):
                pass

            try:
                from shared.src.redis_client import RedisClient
                redis_client = RedisClient()
                redis_client.publish("roottrading:order:failed", notification)
                logger.info(f"✅ Notification d'échec d'ordre envoyée pour {cycle_id}")
            except Exception as e:
                logger.warning(f"⚠️ Impossible d'envoyer la notification Redis: {str(e)}")

        except Exception as e:
            logger.error(f"❌ Erreur lors de la notification d'échec d'ordre: {str(e)}")

    def get_current_price(self, symbol: str) -> float:
        """
        Récupère le prix actuel d'un symbole sur Binance.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            
        Returns:
            Prix actuel
        """
        try:
            price_url = f"{self.BASE_URL}{self.API_V3}/ticker/price"
            params = {"symbol": symbol}
            response = self.session.get(price_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return float(data['price'])
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur lors de la récupération du prix pour {symbol}: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Réponse: {e.response.text}")
            
            # En cas d'erreur, utiliser un prix de secours (pourrait être amélioré)
            if symbol.startswith("BTC"):
                return 60000.0
            elif symbol.startswith("ETH"):
                return 3000.0
            else:
                return 100.0  # Valeur par défaut
                
    def prepare_order_params(self, order: TradeOrder, time_offset: int = 0) -> Dict[str, Any]:
        """
        Prépare les paramètres d'un ordre pour envoi à Binance.
        
        Args:
            order: Ordre à préparer
            time_offset: Décalage temporel avec le serveur Binance
            
        Returns:
            Paramètres préparés pour l'envoi
        """
        # Préparation de l'ID client
        client_order_id = order.client_order_id or f"root_{uuid.uuid4().hex[:16]}"
        
        # Convertir side en string si c'est un enum
        side_str = order.side.value if hasattr(order.side, 'value') else str(order.side)
        
        # Convertir BUY/SELL vers BUY/SELL pour Binance
        if side_str == "BUY":
            side = "BUY"
        elif side_str == "SELL":
            side = "SELL"
        else:
            # Compatibilité avec l'ancien système BUY/SELL (si présent)
            side = side_str
        
        # Déterminer si c'est un ordre LIMIT ou MARKET
        order_type = "LIMIT" if order.price else "MARKET"
        
        # Préparer les paramètres de base
        params = {
            "symbol": order.symbol,
            "side": side,
            "type": order_type,
            "quantity": f"{order.quantity:.8f}".rstrip('0').rstrip('.'),
            "newClientOrderId": client_order_id,
            "timestamp": int(time.time() * 1000) + time_offset
        }
        
        # Ajouter les paramètres pour les ordres LIMIT
        if order_type == "LIMIT":
            params["price"] = f"{order.price:.8f}".rstrip('0').rstrip('.')
            params["timeInForce"] = "GTC"  # Good Till Canceled
        
        # Générer la signature
        params["signature"] = self.generate_signature(params)
        
        logger.debug(f"Paramètres de l'ordre préparés: {params}")
        return params
                
    def send_order_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Envoie une requête d'ordre à Binance.
        
        Args:
            params: Paramètres de l'ordre
            
        Returns:
            Réponse de Binance
            
        Raises:
            BinanceAPIError: En cas d'erreur de l'API Binance
        """
        try:
            order_url = f"{self.BASE_URL}{self.API_V3}/order"
            
            # Afficher les paramètres pour le débogage
            logger.info(f"📦 Paramètres POST vers Binance: {params}")
            
            # Envoyer la requête
            response = self.session.post(order_url, data=params)
            
            # Vérifier si la requête a réussi
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("msg", "Unknown error")
                    error_code = error_data.get("code", 0)
                    logger.error(f"Erreur Binance {response.status_code}: {error_code} - {error_msg}")
                except:
                    error_msg = f"HTTP {response.status_code}"
                    logger.error(f"Erreur Binance {response.status_code}: {error_msg}")
                
                self.notify_order_failure(error_msg, params, params.get("newClientOrderId"))
                raise BinanceAPIError(f"Erreur Binance: {error_msg}")
            
            # Traiter la réponse
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur lors de l'exécution: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Réponse: {e.response.text}")
            
            # Notifier l'échec
            self.notify_order_failure(e, params, params.get("newClientOrderId"))
            raise BinanceAPIError(f"Erreur de requête Binance: {str(e)}")
    
    def create_execution_from_response(self, data: Dict[str, Any]) -> TradeExecution:
        """
        Crée un objet TradeExecution à partir de la réponse de Binance.
        
        Args:
            data: Réponse de Binance
            
        Returns:
            Objet TradeExecution
        """
        # Calculer le prix si l'ordre est MARKET (cumulativeQuoteQty / executedQty)
        price = float(data['price']) if float(data.get('price', 0)) > 0 else (
            float(data['cummulativeQuoteQty']) / float(data['executedQty']) 
            if float(data.get('executedQty', 0)) > 0 else 0
        )
        
        # Pour les ordres non encore exécutés, utiliser origQty au lieu de executedQty
        quantity = float(data['executedQty']) if float(data.get('executedQty', 0)) > 0 else float(data.get('origQty', 0))
        
        # Convertir BUY/SELL de Binance vers BUY/SELL pour notre enum
        binance_side = data['side']
        if binance_side == "BUY":
            side = OrderSide.BUY
        elif binance_side == "SELL":
            side = OrderSide.SELL
        else:
            # Compatibilité au cas où la valeur serait déjà BUY/SELL
            # ou fallback pour des valeurs inattendues
            try:
                side = OrderSide(binance_side)
            except ValueError:
                # Si la conversion échoue, traiter comme inconnu et logger l'erreur
                logger.error(f"❌ Valeur OrderSide non reconnue de Binance: {binance_side}")
                raise ValueError(f"OrderSide invalide reçu de Binance: {binance_side}")
        
        # Créer et retourner l'exécution
        return TradeExecution(
            order_id=str(data['orderId']),
            symbol=data['symbol'],
            side=side,
            status=OrderStatus(data['status']),
            price=price,
            quantity=quantity,
            quote_quantity=float(data['cummulativeQuoteQty']),
            fee=None,  # Les frais ne sont pas inclus dans la réponse initiale
            fee_asset=None,
            role=None,
            timestamp=datetime.fromtimestamp(int(data['transactTime']) / 1000),
            demo=False
        )
                
    def fetch_order_status(self, symbol: str, order_id: str, time_offset: int = 0) -> Optional[TradeExecution]:
        """
        Récupère le statut d'un ordre sur Binance.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            order_id: ID de l'ordre
            time_offset: Décalage temporel
            
        Returns:
            Exécution mise à jour ou None si erreur
        """
        # Ne plus vérifier l'ID numérique car les vrais ordres Binance peuvent avoir des IDs élevés
        # La vérification des ordres démo doit se faire ailleurs (dans BinanceExecutor)
        # logger.debug(f"Vérification order_id: {order_id} (type: {type(order_id)})")  # Commenté pour réduire le bruit dans les logs
            
        try:
            order_url = f"{self.BASE_URL}{self.API_V3}/order"
            timestamp = int(time.time() * 1000) + time_offset
            
            params = {
                "symbol": symbol,
                "orderId": order_id,
                "timestamp": timestamp
            }
            
            # Générer la signature
            params["signature"] = self.generate_signature(params)
            
            # Envoyer la requête
            response = self.session.get(order_url, params=params)
            response.raise_for_status()
            
            # Traiter la réponse
            order_response = response.json()
            
            # Calculer le prix correctement
            price = float(order_response['price']) if float(order_response.get('price', 0)) > 0 else (
                float(order_response['cummulativeQuoteQty']) / max(float(order_response['executedQty']), 1e-8)
                if float(order_response.get('executedQty', 0)) > 0 else 0
            )
            
            # Convertir BUY/SELL de Binance vers BUY/SELL pour notre enum
            binance_side = order_response['side']
            if binance_side == "BUY":
                side = OrderSide.BUY
            elif binance_side == "SELL":
                side = OrderSide.SELL
            else:
                # Compatibilité au cas où la valeur serait déjà BUY/SELL
                # ou fallback pour des valeurs inattendues
                try:
                    side = OrderSide(binance_side)
                except ValueError:
                    # Si la conversion échoue, traiter comme inconnu et logger l'erreur
                    logger.error(f"❌ Valeur OrderSide non reconnue de Binance: {binance_side}")
                    raise ValueError(f"OrderSide invalide reçu de Binance: {binance_side}")
            
            # Préparer l'objet d'exécution
            execution = TradeExecution(
                order_id=str(order_response['orderId']),
                symbol=order_response['symbol'],
                side=side,
                status=OrderStatus(order_response['status']),
                price=price,
                quantity=float(order_response['executedQty']),
                quote_quantity=float(order_response['cummulativeQuoteQty']),
                fee=None,
                fee_asset=None,
                role=None,
                timestamp=datetime.fromtimestamp(order_response.get('time', time.time() * 1000) / 1000),
                demo=False
            )
            
            return execution
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération du statut de l'ordre: {str(e)}")
            return None
                
    def cancel_order(self, symbol: str, order_id: str, time_offset: int = 0) -> bool:
        """
        Annule un ordre sur Binance.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            order_id: ID de l'ordre
            time_offset: Décalage temporel
            
        Returns:
            True si l'annulation a réussi, False sinon
        """
        # Ne plus vérifier l'ID numérique car les vrais ordres Binance peuvent avoir des IDs élevés
        # La vérification des ordres démo doit se faire ailleurs (dans BinanceExecutor)
            
        try:
            cancel_url = f"{self.BASE_URL}{self.API_V3}/order"
            timestamp = int(time.time() * 1000) + time_offset
            
            params = {
                "symbol": symbol,
                "orderId": order_id,
                "timestamp": timestamp
            }
            
            # Générer la signature
            params["signature"] = self.generate_signature(params)
            
            # Envoyer la requête d'annulation
            response = self.session.delete(cancel_url, params=params)
            
            # Gérer les erreurs d'annulation
            if response.status_code != 200:
                try:
                    error_msg = response.json().get("msg", "Unknown error")
                except:
                    error_msg = f"HTTP {response.status_code}"
                
                logger.error(f"Erreur Binance lors de l'annulation: {error_msg}")
                
                # Notifier de l'échec si ce n'est pas une erreur "ordre déjà rempli"
                if "FILLED" not in error_msg:
                    self.notify_order_failure(error_msg, params, order_id)
                
                return False
            
            logger.info(f"✅ Ordre annulé sur Binance: {order_id}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'annulation de l'ordre: {str(e)}")
            
            # Notifier l'échec d'annulation
            self.notify_order_failure(e, {"symbol": symbol, "orderId": order_id}, order_id)
            
            return False
                
    def fetch_account_balances(self, time_offset: int = 0) -> Dict[str, Dict[str, float]]:
        """
        Récupère les soldes du compte Binance.
        
        Args:
            time_offset: Décalage temporel avec le serveur Binance
            
        Returns:
            Dictionnaire des soldes par actif
        """
        try:
            account_url = f"{self.BASE_URL}{self.API_V3}/account"
            timestamp = int(time.time() * 1000) + time_offset
            
            params = {"timestamp": timestamp}
            params["signature"] = self.generate_signature(params)
            
            response = self.session.get(account_url, params=params)
            response.raise_for_status()
            
            account_info = response.json()
            balances = {}
            
            for balance in account_info['balances']:
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                # Ne conserver que les actifs avec un solde
                if total > 0:
                    balances[balance['asset']] = {
                        "free": free,
                        "locked": locked,
                        "total": total
                    }
            
            return balances
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des soldes: {str(e)}")
            
            # En cas d'erreur, retourner un dictionnaire vide
            return {}
                
    def fetch_trade_fee(self, symbol: str, time_offset: int = 0) -> Tuple[float, float]:
        """
        Récupère les frais de trading pour un symbole.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            time_offset: Décalage temporel
            
        Returns:
            Tuple (maker_fee, taker_fee) en pourcentage
        """
        try:
            fee_url = f"{self.BASE_URL}/sapi/v1/asset/tradeFee"
            timestamp = int(time.time() * 1000) + time_offset
            
            params = {
                "symbol": symbol,
                "timestamp": timestamp
            }
            
            params["signature"] = self.generate_signature(params)
            
            response = self.session.get(fee_url, params=params)
            response.raise_for_status()
            
            fee_info = response.json()
            
            if fee_info and len(fee_info) > 0:
                maker_fee = float(fee_info[0]['makerCommission'])
                taker_fee = float(fee_info[0]['takerCommission'])
                return (maker_fee, taker_fee)
            
            # Si pas d'info spécifique, utiliser les frais standard
            return (0.001, 0.001)
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des frais de trading: {str(e)}")
            
            # En cas d'erreur, retourner des frais standard
            return (0.001, 0.001)
                
    def fetch_open_orders(self, symbol: Optional[str] = None, time_offset: int = 0) -> List[Dict[str, Any]]:
        """
        Récupère tous les ordres ouverts sur Binance.
        
        Args:
            symbol: Symbole optionnel pour filtrer (si None, récupère tous les ordres)
            time_offset: Décalage temporel avec le serveur Binance
            
        Returns:
            Liste des ordres ouverts
        """
        try:
            orders_url = f"{self.BASE_URL}{self.API_V3}/openOrders"
            timestamp = int(time.time() * 1000) + time_offset
            
            params = {"timestamp": timestamp}
            
            # Ajouter le symbole si spécifié
            if symbol:
                params["symbol"] = symbol
            
            # Générer la signature
            params["signature"] = self.generate_signature(params)
            
            # Envoyer la requête
            response = self.session.get(orders_url, params=params)
            response.raise_for_status()
            
            orders = response.json()
            logger.info(f"📊 {len(orders)} ordres ouverts trouvés" + (f" pour {symbol}" if symbol else ""))
            
            return orders
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des ordres ouverts: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Réponse: {e.response.text}")
            
            # En cas d'erreur, retourner une liste vide
            return []
                
    def fetch_exchange_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Récupère les informations de trading pour tous les symboles.
        
        Returns:
            Dictionnaire des informations par symbole
        """
        try:
            url = f"{self.BASE_URL}{self.API_V3}/exchangeInfo"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()

            symbol_info = {}
            for symbol_data in data.get("symbols", []):
                symbol_name = symbol_data["symbol"]
                filters = {f["filterType"]: f for f in symbol_data.get("filters", [])}
                
                price_filter = filters.get("PRICE_FILTER", {})
                lot_size_filter = filters.get("LOT_SIZE", {})
                notional_filter = filters.get("NOTIONAL", {})
                min_notional_filter = filters.get("MIN_NOTIONAL", {})

                info = {}
                if price_filter:
                    info["tick_size"] = float(price_filter.get("tickSize", 0.01))
                    info["min_price"] = float(price_filter.get("minPrice", 0.01))
                    info["max_price"] = float(price_filter.get("maxPrice", 100000.0))
                
                if lot_size_filter:
                    info["step_size"] = float(lot_size_filter.get("stepSize", 0.0001))
                    info["min_qty"] = float(lot_size_filter.get("minQty", 0.001))
                    info["max_qty"] = float(lot_size_filter.get("maxQty", 100000.0))
                
                if notional_filter:
                    info["min_notional"] = float(notional_filter.get("minNotional", 10.0))
                elif min_notional_filter:  # Parfois c'est un filtre distinct
                    info["min_notional"] = float(min_notional_filter.get("minNotional", 10.0))
                
                symbol_info[symbol_name] = info

            logger.info(f"✅ Informations de trading chargées pour {len(symbol_info)} symboles")
            return symbol_info

        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des informations d'échange: {str(e)}")
            # Retourner des informations par défaut pour les symboles courants
            default_info = {
                "BTCUSDC": {
                    "tick_size": 0.01,
                    "step_size": 0.00001,
                    "min_qty": 0.001,
                    "min_notional": 10.0
                },
                "ETHUSDC": {
                    "tick_size": 0.01,
                    "step_size": 0.001,
                    "min_qty": 0.01,
                    "min_notional": 10.0
                }
            }
            return default_info