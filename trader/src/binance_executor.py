"""
Module d'exécution des ordres sur Binance.
Gère l'envoi des ordres à Binance et le suivi de leur exécution.
"""
import logging
import math
import time
import hmac
import hashlib
import json
import uuid
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Union
import requests
from datetime import datetime
import asyncio

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY, TRADING_MODE
from shared.src.enums import OrderSide, OrderStatus, TradeRole
from shared.src.schemas import TradeOrder, TradeExecution

# Configuration du logging
logger = logging.getLogger(__name__)

class BinanceAPIError(Exception):
    """Exception personnalisée pour les erreurs de l'API Binance."""
    pass

class BinanceExecutor:
    """
    Exécuteur d'ordres sur Binance.
    Gère l'envoi et le suivi des ordres sur Binance, ou les simule en mode démo.
    """
    
    # URLs de base de l'API Binance
    BASE_URL = "https://api.binance.com"
    API_V3 = "/api/v3"
    
    def __init__(self, api_key: str = BINANCE_API_KEY, api_secret: str = BINANCE_SECRET_KEY, 
                 demo_mode: bool = TRADING_MODE.lower() == 'demo'):
        """
        Initialise l'exécuteur Binance.
        
        Args:
            api_key: Clé API Binance
            api_secret: Clé secrète Binance
            demo_mode: Mode démo (pas d'ordres réels)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Forcer le mode démo temporairement pour les tests
        # Décommentez cette ligne si vous voulez forcer le mode démo
        # self.demo_mode = True
        
        # Utiliser le mode configuré
        self.demo_mode = demo_mode
        
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})
        
        # Variables pour le mode démo
        self.demo_order_id = 10000000  # ID de départ pour les ordres démo
        self.demo_trades = {}  # Historique des trades en mode démo
        
        # Limites minimales pour les symboles courants
        self.min_quantities = {
            "BTCUSDC": 0.001,
            "ETHUSDC": 0.01,
        }
        
        # Vérifier la connectivité et les permissions
        self._check_connectivity()
        
        # Récupérer les informations de trading pour tous les symboles
        self.symbol_info = self._fetch_exchange_info()
        
        logger.info(f"✅ BinanceExecutor initialisé en mode {'DÉMO' if demo_mode else 'RÉEL'}")
    
    def _check_connectivity(self) -> None:
        """
        Vérifie la connectivité avec l'API Binance et les permissions de l'API.
        Lève une exception en cas d'erreur.
        """
        try:
            # Vérifier la connectivité générale
            ping_url = f"{self.BASE_URL}{self.API_V3}/ping"
            response = self.session.get(ping_url)
            response.raise_for_status()
            
            # Vérifier les informations du compte (nécessite des permissions)
            if not self.demo_mode:
                account_url = f"{self.BASE_URL}{self.API_V3}/account"
                timestamp = int(time.time() * 1000)
                params = {"timestamp": timestamp}
                params["signature"] = self._generate_signature(params)
                
                response = self.session.get(account_url, params=params)
                response.raise_for_status()
                
                account_info = response.json()
                permissions = account_info.get("permissions", ["SPOT"])
                
                if "SPOT" not in permissions:
                    logger.warning("⚠️ La clé API n'a pas les permissions SPOT")
                
                logger.info(f"✅ Connecté à Binance avec succès (permissions: {', '.join(permissions)})")
            else:
                logger.info("✅ Mode DÉMO: vérifié la connectivité Binance (sans authentification)")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur de connexion à Binance: {str(e)}")
            if e.response:
                logger.error(f"Réponse: {e.response.text}")
            raise BinanceAPIError(f"Impossible de se connecter à Binance: {str(e)}")
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Génère une signature HMAC SHA256 pour l'authentification Binance.
        
        Args:
            params: Paramètres de la requête
            
        Returns:
            Signature générée
        """
        # Trier les paramètres et les formater en chaîne de requête
        query_string = "&".join([f"{key}={params[key]}" for key in sorted(params.keys())])
        
        # Ajout de debug logs pour comprendre le problème
        logger.debug(f"Génération de signature pour: {query_string}")
        
        # S'assurer que le secret est valide
        if not self.api_secret:
            logger.error("API Secret est vide ou invalide")
            raise ValueError("API Secret invalide")
        
        # Générer la signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        logger.debug(f"Signature générée: {signature}")
        
        return signature
    
    def _notify_order_failure(self, error, order_params, client_order_id=None):
        """
        Notifie les autres services d'un échec d'ordre via Redis.
        
        Args:
            error: Erreur Binance
            order_params: Paramètres de l'ordre
            client_order_id: ID du cycle ou de l'ordre client (optionnel)
        """
        try:
            # Préparer les données de notification
            notification = {
                "type": "order_failed",
                "cycle_id": client_order_id or order_params.get("newClientOrderId", "unknown"),
                "symbol": order_params.get("symbol"),
                "side": order_params.get("side"),
                "quantity": order_params.get("quantity"),
                "price": order_params.get("price"),
                "reason": f"Erreur Binance: {str(error)}",
                "timestamp": int(time.time() * 1000)
            }
            
            # Calculer le montant en USDC si possible
            try:
                price = float(order_params.get("price", 0))
                quantity = float(order_params.get("quantity", 0))
                amount = price * quantity
                notification["amount"] = amount
            except:
                # Si conversion impossible, ne pas inclure le montant
                pass
            
            # Importer redis_client depuis le module partagé s'il existe
            try:
                from shared.src.redis_client import RedisClient
                redis_client = RedisClient()
                redis_client.publish("roottrading:order:failed", notification)
                logger.info(f"✅ Notification d'échec d'ordre envoyée pour {client_order_id}")
            except Exception as e:
                logger.warning(f"⚠️ Impossible d'envoyer la notification Redis: {str(e)}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la notification d'échec d'ordre: {str(e)}")
    
    def _generate_order_id(self) -> str:
        """
        Génère un ID d'ordre unique pour le mode démo.
        
        Returns:
            ID d'ordre unique
        """
        self.demo_order_id += 1
        return str(self.demo_order_id)
    
    def _get_min_quantity(self, symbol: str) -> float:
        """
        Retourne la quantité minimale pour un symbole.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            
        Returns:
            Quantité minimale
        """
        # Vérifier si nous avons des informations sur le symbole
        if symbol in self.symbol_info and "min_qty" in self.symbol_info[symbol]:
            return self.symbol_info[symbol]["min_qty"]
        
        # Sinon utiliser les valeurs par défaut
        return self.min_quantities.get(symbol, 0.001)
    
    def _get_min_notional(self, symbol: str) -> float:
        """
        Retourne la valeur minimale d'un ordre pour un symbole.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            
        Returns:
            Valeur minimale de l'ordre
        """
        # Vérifier si nous avons des informations sur le symbole
        if symbol in self.symbol_info and "min_notional" in self.symbol_info[symbol]:
            return self.symbol_info[symbol]["min_notional"]
        
        # Sinon utiliser une valeur par défaut
        return 10.0  # La plupart des paires Binance ont un minimum de 10 USDC
    
    def _simulate_order(self, order: TradeOrder) -> TradeExecution:
        """
        Simule l'exécution d'un ordre en mode démo.
        
        Args:
            order: Ordre à simuler
            
        Returns:
            Exécution simulée
        """
        # S'assurer que order.side est bien un enum OrderSide
        if isinstance(order.side, str):
            order.side = OrderSide(order.side)
            
        # Générer un ID d'ordre
        order_id = self._generate_order_id()
        
        # Récupérer le prix actuel (ou utiliser le prix de l'ordre)
        price = order.price
        if price is None:
            # En mode démo, utiliser le dernier prix connu ou un prix simulé
            price = self._get_current_price(order.symbol)
        
        # Calculer la quantité quote (USDC, etc.)
        quote_quantity = price * order.quantity
        
        # Simuler des frais (0.1% pour Binance)
        fee = quote_quantity * 0.001
        
        # Créer l'exécution
        execution = TradeExecution(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            status=OrderStatus.FILLED,
            price=price,
            quantity=order.quantity,
            quote_quantity=quote_quantity,
            fee=fee,
            fee_asset=order.symbol.replace("USDC", ""),  # BTC pour BTCUSDC
            role=TradeRole.TAKER,
            timestamp=datetime.now(),
            demo=True
        )
        
        # Stocker le trade en mémoire pour référence future
        self.demo_trades[order_id] = execution
        
        logger.info(f"✅ [DÉMO] Ordre simulé: {order.side.value if hasattr(order.side, 'value') else order.side} {order.quantity} {order.symbol} @ {price}")
        return execution
    
    def _get_current_price(self, symbol: str) -> float:
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
            if e.response:
                logger.error(f"Réponse: {e.response.text}")
            
            # En cas d'erreur, utiliser un prix de secours (pourrait être amélioré)
            if symbol.startswith("BTC"):
                return 60000.0
            elif symbol.startswith("ETH"):
                return 3000.0
            else:
                return 100.0  # Valeur par défaut
    
    def execute_order(self, order: TradeOrder) -> TradeExecution:
        """
        Exécute un ordre sur Binance ou le simule en mode démo.
        
        Args:
            order: Ordre à exécuter
            
        Returns:
            Exécution de l'ordre
        """
        # S'assurer que order.side est bien un enum OrderSide
        if isinstance(order.side, str):
            order.side = OrderSide(order.side)
        
        # En mode démo, simuler l'ordre
        if self.demo_mode:
            return self._simulate_order(order)
        
        # Récupérer les limites minimales pour ce symbole
        min_qty = self._get_min_quantity(order.symbol)
        min_notional = self._get_min_notional(order.symbol)
        
        # Vérifier et ajuster la quantité minimale
        original_quantity = order.quantity
        if order.quantity < min_qty:
            logger.warning(f"⚠️ Quantité {order.quantity} inférieure au minimum {min_qty} pour {order.symbol}")
            # Passer automatiquement en mode simulation pour cet ordre
            logger.info(f"Passage en mode simulation pour quantité insuffisante")
            return self._simulate_order(order)
        
        # Vérifier la valeur minimale de l'ordre
        current_price = order.price or self._get_current_price(order.symbol)
        notional_value = order.quantity * current_price
        
        if notional_value < min_notional:
            logger.warning(f"⚠️ Valeur de l'ordre {notional_value:.2f} inférieure au minimum {min_notional}")
            # Passer automatiquement en mode simulation pour cet ordre
            logger.info(f"Passage en mode simulation pour valeur insuffisante")
            return self._simulate_order(order)
        
        # Appliquer l'arrondi de quantité et prix selon les règles Binance
        quantity = self._round_quantity(order.symbol, order.quantity)
        price = None
        if order.price:
            price = self._round_price(order.symbol, order.price)
        
        # En mode réel, envoyer l'ordre à Binance
        try:
            order_url = f"{self.BASE_URL}{self.API_V3}/order"
            timestamp = int(time.time() * 1000)
            
            # Générer un client_order_id unique si non fourni
            client_order_id = order.client_order_id or f"root_{uuid.uuid4().hex[:16]}"
            
            # Formater la quantité correctement
            step_size = self.symbol_info.get(order.symbol, {}).get("step_size", 0.0001)
            precision = abs(int(round(-math.log10(step_size))))
            quantity_str = f"{quantity:.{precision}f}"
            
            # Préparer les paramètres de l'ordre
            params = {
                "symbol": order.symbol,
                "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
                "type": "LIMIT" if price else "MARKET",
                "quantity": quantity_str,
                "newClientOrderId": client_order_id,
                "timestamp": timestamp
            }
            
            # Ajouter le prix pour les ordres limites
            if price:
                tick_size = self.symbol_info.get(order.symbol, {}).get("tick_size", 0.01)
                price_precision = abs(int(round(-math.log10(tick_size))))
                params["price"] = f"{price:.{price_precision}f}"
                params["timeInForce"] = "GTC"  # Good Till Cancelled
            
            # Log des paramètres pour débogage
            logger.info(f"Paramètres de l'ordre: {params}")
            
            # Générer la signature
            params["signature"] = self._generate_signature(params)
            
            # Envoyer l'ordre
            response = self.session.post(order_url, params=params)
            
            # Si erreur 400, capturer et utiliser la simulation
            if response.status_code == 400:
                error_msg = response.json().get("msg", "Unknown error")
                logger.error(f"Erreur Binance 400: {error_msg}")
                
                # Notifier l'échec d'ordre
                self._notify_order_failure(error_msg, params, client_order_id)
                
                logger.info(f"Passage en mode simulation suite à l'erreur")
                return self._simulate_order(order)
                
            response.raise_for_status()
            
            # Traiter la réponse
            order_response = response.json()
            
            # Préparer l'objet d'exécution
            execution = TradeExecution(
                order_id=str(order_response['orderId']),
                symbol=order_response['symbol'],
                side=OrderSide(order_response['side']),
                status=OrderStatus(order_response['status']),
                price=float(order_response['price']) if float(order_response['price']) > 0 else float(order_response['cummulativeQuoteQty']) / float(order_response['executedQty']),
                quantity=float(order_response['executedQty']),
                quote_quantity=float(order_response['cummulativeQuoteQty']),
                fee=None,  # Les frais ne sont pas inclus dans la réponse de l'ordre
                fee_asset=None,
                role=None,  # Le rôle n'est pas inclus dans la réponse
                timestamp=datetime.fromtimestamp(order_response['transactTime'] / 1000),
                demo=False
            )
            
            logger.info(f"✅ Ordre exécuté sur Binance: {order.side.value if hasattr(order.side, 'value') else order.side} {quantity} {order.symbol} @ {execution.price}")
            return execution
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur lors de l'exécution de l'ordre: {str(e)}")
            if e.response:
                logger.error(f"Réponse: {e.response.text}")
            
            # Notifier l'échec d'ordre
            self._notify_order_failure(e, params, client_order_id)
            
            # En cas d'erreur, passer en mode simulation pour cet ordre
            logger.info(f"Passage en mode simulation suite à l'erreur")
            return self._simulate_order(order)
    
    def get_order_status(self, symbol: str, order_id: str) -> Optional[TradeExecution]:
        """
        Récupère le statut d'un ordre sur Binance.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            order_id: ID de l'ordre
            
        Returns:
            Exécution mise à jour ou None si non trouvée
        """
        # En mode démo, récupérer depuis la mémoire
        if self.demo_mode:
            if order_id in self.demo_trades:
                return self.demo_trades[order_id]
            return None
        
        # En mode réel, interroger Binance
        try:
            order_url = f"{self.BASE_URL}{self.API_V3}/order"
            timestamp = int(time.time() * 1000)
            
            params = {
                "symbol": symbol,
                "orderId": order_id,
                "timestamp": timestamp
            }
            
            # Générer la signature
            params["signature"] = self._generate_signature(params)
            
            # Envoyer la requête
            response = self.session.get(order_url, params=params)
            response.raise_for_status()
            
            # Traiter la réponse
            order_response = response.json()
            
            # Préparer l'objet d'exécution
            execution = TradeExecution(
                order_id=str(order_response['orderId']),
                symbol=order_response['symbol'],
                side=OrderSide(order_response['side']),
                status=OrderStatus(order_response['status']),
                price=float(order_response['price']) if float(order_response['price']) > 0 else float(order_response['cummulativeQuoteQty']) / max(float(order_response['executedQty']), 1e-8),
                quantity=float(order_response['executedQty']),
                quote_quantity=float(order_response['cummulativeQuoteQty']),
                fee=None,
                fee_asset=None,
                role=None,
                timestamp=datetime.fromtimestamp(order_response['time'] / 1000) if 'time' in order_response else datetime.now(),
                demo=False
            )
            
            return execution
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur lors de la récupération du statut de l'ordre: {str(e)}")
            if e.response:
                logger.error(f"Réponse: {e.response.text}")
            return None
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Annule un ordre sur Binance.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            order_id: ID de l'ordre
            
        Returns:
            True si l'annulation a réussi, False sinon
        """
        # En mode démo, simuler l'annulation
        if self.demo_mode:
            if order_id in self.demo_trades:
                trade = self.demo_trades[order_id]
                if trade.status not in [OrderStatus.FILLED, OrderStatus.CANCELED]:
                    trade.status = OrderStatus.CANCELED
                    logger.info(f"✅ [DÉMO] Ordre annulé: {order_id}")
                    return True
            return False
        
        # En mode réel, annuler sur Binance
        try:
            cancel_url = f"{self.BASE_URL}{self.API_V3}/order"
            timestamp = int(time.time() * 1000)
            
            params = {
                "symbol": symbol,
                "orderId": order_id,
                "timestamp": timestamp
            }
            
            # Générer la signature
            params["signature"] = self._generate_signature(params)
            
            # Envoyer la requête d'annulation
            response = self.session.delete(cancel_url, params=params)
            
            # Gérer les erreurs d'annulation
            if response.status_code != 200:
                error_msg = response.json().get("msg", "Unknown error")
                logger.error(f"Erreur Binance lors de l'annulation: {error_msg}")
                
                # Notifier de l'échec si ce n'est pas une erreur "ordre déjà rempli"
                if "FILLED" not in error_msg:
                    self._notify_order_failure(error_msg, params, order_id)
                
                return False
            
            response.raise_for_status()
            
            logger.info(f"✅ Ordre annulé sur Binance: {order_id}")
            return True
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur lors de l'annulation de l'ordre: {str(e)}")
            if e.response:
                logger.error(f"Réponse: {e.response.text}")
            
            # Notifier l'échec d'annulation
            self._notify_order_failure(e, params, order_id)
            
            return False
    
    def get_account_balances(self) -> Dict[str, Dict[str, float]]:
        """
        Récupère les soldes du compte Binance.
        
        Returns:
            Dictionnaire des soldes par actif
        """
        if self.demo_mode:
            # En mode démo, retourner des soldes fictifs
            return {
                "BTC": {"free": 0.01, "locked": 0.0, "total": 0.01},
                "ETH": {"free": 0.5, "locked": 0.0, "total": 0.5},
                "USDC": {"free": 1000.0, "locked": 0.0, "total": 1000.0}
            }
        
        try:
            account_url = f"{self.BASE_URL}{self.API_V3}/account"
            timestamp = int(time.time() * 1000)
            
            params = {"timestamp": timestamp}
            params["signature"] = self._generate_signature(params)
            
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
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur lors de la récupération des soldes: {str(e)}")
            if e.response:
                logger.error(f"Réponse: {e.response.text}")
            
            # En cas d'erreur, retourner un dictionnaire vide
            return {}
    
    def get_trade_fee(self, symbol: str) -> Tuple[float, float]:
        """
        Récupère les frais de trading pour un symbole.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            
        Returns:
            Tuple (maker_fee, taker_fee) en pourcentage
        """
        if self.demo_mode:
            # En mode démo, retourner des frais standard
            return (0.001, 0.001)  # 0.1%
        
        try:
            fee_url = f"{self.BASE_URL}/sapi/v1/asset/tradeFee"
            timestamp = int(time.time() * 1000)
            
            params = {
                "symbol": symbol,
                "timestamp": timestamp
            }
            
            params["signature"] = self._generate_signature(params)
            
            response = self.session.get(fee_url, params=params)
            response.raise_for_status()
            
            fee_info = response.json()
            
            if fee_info and len(fee_info) > 0:
                maker_fee = float(fee_info[0]['makerCommission'])
                taker_fee = float(fee_info[0]['takerCommission'])
                return (maker_fee, taker_fee)
            
            # Si pas d'info spécifique, utiliser les frais standard
            return (0.001, 0.001)
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erreur lors de la récupération des frais de trading: {str(e)}")
            if e.response:
                logger.error(f"Réponse: {e.response.text}")
            
            # En cas d'erreur, retourner des frais standard
            return (0.001, 0.001)
        
    def _fetch_exchange_info(self) -> Dict[str, Dict[str, Any]]:
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

    def _round_price(self, symbol: str, price: float) -> float:
        """
        Arrondit le prix selon le tickSize du symbole.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            price: Prix à arrondir
            
        Returns:
            Prix arrondi
        """
        info = self.symbol_info.get(symbol, {})
        tick_size = info.get("tick_size", 0.01)
        
        if tick_size == 0:
            return price
            
        precision = int(round(-math.log10(tick_size)))
        return math.floor(price / tick_size) * tick_size

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """
        Arrondit la quantité selon le stepSize du symbole.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            quantity: Quantité à arrondir
            
        Returns:
            Quantité arrondie
        """
        info = self.symbol_info.get(symbol, {})
        step_size = info.get("step_size", 0.0001)
        
        if step_size == 0:
            return quantity
            
        precision = int(round(-math.log10(step_size)))
        return math.floor(quantity / step_size) * step_size