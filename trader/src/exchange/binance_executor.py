# trader/src/exchange/binance_executor.py
"""
Module d'exécution principal des ordres sur Binance.
Version simplifiée qui délègue les tâches spécifiques à d'autres modules.
"""
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Imports des modules internes
from trader.src.exchange.binance_utils import BinanceUtils
from trader.src.exchange.constraints import BinanceSymbolConstraints
from trader.src.exchange.order_validation import OrderValidator
from shared.src.enums import OrderSide, OrderStatus, TradeRole
from shared.src.schemas import TradeOrder, TradeExecution

logger = logging.getLogger(__name__)

class BinanceExecutor:
    """
    Exécuteur d'ordres sur Binance.
    Gère l'envoi et le suivi des ordres sur Binance, ou les simule en mode démo.
    """
    # URLs de base de l'API Binance
    BASE_URL = "https://api.binance.com"
    API_V3 = "/api/v3"
    
    def __init__(self, api_key: str, api_secret: str, demo_mode: bool = False):
        """
        Initialise l'exécuteur Binance.
        
        Args:
            api_key: Clé API Binance
            api_secret: Clé secrète Binance
            demo_mode: Mode démo (pas d'ordres réels)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.demo_mode = demo_mode
        
        # Initialiser les classes utilitaires
        self.utils = BinanceUtils(api_key, api_secret)
        
        # Variables pour le mode démo
        self.demo_order_id = 10000000  # ID de départ pour les ordres démo
        self.demo_trades: Dict[str, Any] = {}  # Historique des trades en mode démo
        
        # Calculer le décalage temporel avec le serveur Binance
        self.time_offset = self.utils.get_time_offset()
        
        # Vérifier la connectivité et les permissions
        self._check_connectivity()
        
        # Récupérer les informations de trading pour tous les symboles
        self.symbol_info = self.utils.fetch_exchange_info()
        
        # Initialiser les contraintes avec les informations en temps réel
        self.symbol_constraints = BinanceSymbolConstraints(self.symbol_info)
        self.validator = OrderValidator(self.symbol_constraints)
        
        logger.info(f"✅ BinanceExecutor initialisé en mode {'DÉMO' if demo_mode else 'RÉEL'}")
    
    def _check_connectivity(self) -> None:
        """
        Vérifie la connectivité avec l'API Binance et les permissions de l'API.
        Lève une exception en cas d'erreur.
        """
        if self.utils.check_connectivity(self.demo_mode):
            logger.info(f"✅ {'Mode DÉMO: vérifié la connectivité Binance (sans authentification)' if self.demo_mode else 'Connecté à Binance avec succès'}")
        else:
            logger.error("❌ Échec de la vérification de la connectivité Binance")
            raise Exception("Impossible de se connecter à Binance")
    
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
        order_id = str(self.demo_order_id)
        self.demo_order_id += 1
        
        # Récupérer le prix actuel (ou utiliser le prix de l'ordre)
        price = order.price if order.price is not None else self.utils.get_current_price(order.symbol)
        
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
    
    def execute_order(self, order: TradeOrder) -> TradeExecution:
        """
        Exécute un ordre sur Binance ou le simule en mode démo.

        Args:
            order: Ordre à exécuter
        
        Returns:
            Exécution de l'ordre
        """
        try:
            # Mode démo
            if self.demo_mode:
                return self._simulate_order(order)
            
            # Mode réel
            # Normaliser l'ordre
            if isinstance(order.side, str):
                order.side = OrderSide(order.side)
                
            # Valider et ajuster l'ordre
            validated_order = self.validator.validate_and_adjust_order(order)
            
            # Préparer les paramètres d'ordre
            params = self.utils.prepare_order_params(validated_order, self.time_offset)
            
            # Envoyer l'ordre à Binance
            response = self.utils.send_order_request(params)
            
            # Transformer la réponse en objet TradeExecution
            execution = self.utils.create_execution_from_response(response)
            
            side_str = order.side.value if hasattr(order.side, 'value') else str(order.side)
            logger.info(f"✅ Ordre exécuté sur Binance: {side_str} {execution.quantity} {execution.symbol} @ {execution.price}")
            return execution
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution: {str(e)}")
            
            # NE PAS simuler l'ordre en cas d'erreur
            # Cela crée des incohérences (cycles non-démo avec ordres démo)
            raise e
    
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
            return self.demo_trades.get(order_id)
        
        # Vérifier si c'est un ordre créé en mode démo (stocké dans demo_trades)
        if order_id in self.demo_trades:
            logger.debug(f"⏭️ Ordre démo détecté ({order_id}), ignorer la vérification Binance")
            return self.demo_trades.get(order_id)
        
        # En mode réel, interroger Binance
        try:
            return self.utils.fetch_order_status(symbol, order_id, self.time_offset)
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération du statut de l'ordre: {str(e)}")
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
        
        # Vérifier si c'est un ordre créé en mode démo (stocké dans demo_trades)
        if order_id in self.demo_trades:
            logger.debug(f"⏭️ Ordre démo détecté ({order_id}), ignorer l'annulation Binance")
            return True  # Considérer comme réussi pour ne pas bloquer le processus
        
        # En mode réel, annuler sur Binance
        try:
            return self.utils.cancel_order(symbol, order_id, self.time_offset)
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'annulation de l'ordre: {str(e)}")
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
            return self.utils.fetch_account_balances(self.time_offset)
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des soldes: {str(e)}")
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
            return self.utils.fetch_trade_fee(symbol, self.time_offset)
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des frais de trading: {str(e)}")
            return (0.001, 0.001)  # Valeurs par défaut en cas d'erreur
    
