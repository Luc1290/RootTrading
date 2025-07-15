"""
Exécuteur d'ordres simplifié pour le trader.
Logique simple : reçoit ordre du coordinator → exécute sur Binance → stocke historique.
"""
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from shared.src.enums import OrderSide, OrderStatus
from shared.src.schemas import TradeOrder, TradeExecution
from shared.src.db_pool import transaction
from trader.src.exchange.binance_executor import BinanceExecutor

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Exécuteur d'ordres simplifié.
    Pas de cycles, pas de statuts complexes : juste exécuter et stocker.
    """
    
    def __init__(self, binance_executor: BinanceExecutor):
        """
        Initialise l'exécuteur d'ordres.
        
        Args:
            binance_executor: Exécuteur Binance
        """
        self.binance_executor = binance_executor
        
        # Historique des ordres exécutés
        self.order_history: List[Dict[str, Any]] = []
        
        logger.info("✅ OrderExecutor initialisé (logique simplifiée)")
    
    def execute_order(self, order_data: Dict[str, Any]) -> Optional[str]:
        """
        Exécute un ordre simple.
        
        Args:
            order_data: Données de l'ordre
            Format: {
                "symbol": "BTCUSDC",
                "side": "BUY" ou "SELL", 
                "quantity": 0.001,
                "price": 50000 (optionnel, sinon MARKET),
                "strategy": "Manual",
                "timestamp": 1234567890
            }
            
        Returns:
            ID de l'ordre exécuté ou None si échec
        """
        try:
            # Générer un ID unique pour l'ordre
            order_id = f"order_{uuid.uuid4().hex[:16]}"
            
            # Valider les données
            required_fields = ["symbol", "side", "quantity"]
            for field in required_fields:
                if field not in order_data:
                    logger.error(f"❌ Champ manquant dans l'ordre: {field}")
                    return None
            
            symbol = order_data["symbol"]
            side = OrderSide(order_data["side"])
            quantity = float(order_data["quantity"])
            price = float(order_data.get("price", 0)) if order_data.get("price") else None
            strategy = order_data.get("strategy", "Manual")
            
            side_str = side.value if hasattr(side, 'value') else str(side)
            logger.info(f"📤 Exécution ordre: {side_str} {quantity} {symbol} @ {price or 'MARKET'}")
            
            # Créer l'ordre TradeOrder
            trade_order = TradeOrder(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                client_order_id=order_id,
                strategy=strategy,
                demo=self.binance_executor.demo_mode
            )
            
            # Exécuter sur Binance
            execution = self.binance_executor.execute_order(trade_order)
            
            if not execution or not execution.order_id:
                logger.error(f"❌ Échec exécution ordre {order_id}")
                return None
            
            # Vérifier que l'ordre a été exécuté
            if execution.status not in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                logger.error(f"❌ Ordre {order_id} non exécuté: {execution.status}")
                return None
            
            # Sauvegarder en base de données
            self._save_execution(execution)
            
            # Ajouter à l'historique
            order_record = {
                "id": order_id,
                "binance_order_id": execution.order_id,
                "symbol": symbol,
                "side": side_str,
                "quantity": quantity,
                "price": execution.price,
                "strategy": strategy,
                "status": execution.status.value if hasattr(execution.status, 'value') else str(execution.status),
                "timestamp": datetime.now(timezone.utc),
                "demo": self.binance_executor.demo_mode
            }
            self.order_history.append(order_record)
            
            status_str = execution.status.value if hasattr(execution.status, 'value') else str(execution.status)
            logger.info(f"✅ Ordre exécuté: {order_id} (Binance: {execution.order_id}) - {status_str}")
            return order_id
            
        except Exception as e:
            logger.error(f"❌ Erreur exécution ordre: {str(e)}")
            return None
    
    def _save_execution(self, execution: TradeExecution) -> None:
        """
        Sauvegarde une exécution en base de données.
        
        Args:
            execution: Exécution à sauvegarder
        """
        try:
            with transaction() as cursor:
                cursor.execute("""
                    INSERT INTO trade_executions 
                    (order_id, symbol, side, status, price, quantity, quote_quantity, 
                     fee, fee_asset, role, timestamp, demo)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (order_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        price = EXCLUDED.price,
                        quantity = EXCLUDED.quantity,
                        quote_quantity = EXCLUDED.quote_quantity,
                        fee = EXCLUDED.fee,
                        timestamp = EXCLUDED.timestamp
                """, (
                    execution.order_id,
                    execution.symbol,
                    execution.side.value if hasattr(execution.side, 'value') else str(execution.side),
                    execution.status.value if hasattr(execution.status, 'value') else str(execution.status),
                    execution.price,
                    execution.quantity,
                    execution.quote_quantity,
                    execution.fee,
                    execution.fee_asset,
                    execution.role.value if execution.role and hasattr(execution.role, 'value') else (str(execution.role) if execution.role else None),
                    execution.timestamp,
                    execution.demo
                ))
            
            logger.debug(f"✅ Exécution {execution.order_id} sauvegardée")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde exécution: {str(e)}")
    
    def get_order_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des ordres.
        
        Args:
            limit: Nombre maximum d'ordres à retourner
            
        Returns:
            Liste des ordres récents
        """
        return self.order_history[-limit:]
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère le statut d'un ordre.
        
        Args:
            order_id: ID de l'ordre
            
        Returns:
            Statut de l'ordre ou None
        """
        for order in self.order_history:
            if order["id"] == order_id:
                return order
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques de l'exécuteur.
        
        Returns:
            Statistiques
        """
        total_orders = len(self.order_history)
        successful_orders = sum(1 for order in self.order_history 
                              if order["status"] in ["FILLED", "PARTIALLY_FILLED"])
        
        return {
            "total_orders": total_orders,
            "successful_orders": successful_orders,
            "success_rate": (successful_orders / total_orders * 100) if total_orders > 0 else 0,
            "demo_mode": self.binance_executor.demo_mode,
            "last_order_time": self.order_history[-1]["timestamp"].isoformat() if self.order_history else None
        }