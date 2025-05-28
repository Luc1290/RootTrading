#!/usr/bin/env python3
"""
Script pour nettoyer les ordres orphelins sur Binance
(ordres de cycles terminés qui n'ont pas été annulés)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import logging
import time
from datetime import datetime
from typing import List, Tuple

from shared.src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY, get_db_url
from shared.src.db_pool import DBContextManager
from trader.src.exchange.binance_executor import BinanceExecutor

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrphanOrderCleaner:
    def __init__(self, dry_run: bool = True):
        """
        Initialise le nettoyeur d'ordres orphelins.
        
        Args:
            dry_run: Si True, simule seulement (n'annule pas vraiment)
        """
        self.dry_run = dry_run
        self.db_url = get_db_url()
        self.binance = BinanceExecutor(BINANCE_API_KEY, BINANCE_SECRET_KEY, demo_mode=False)
        
    def find_orphan_orders(self) -> List[Tuple[str, str, str, str]]:
        """
        Trouve tous les ordres orphelins dans la DB.
        
        Returns:
            Liste de tuples (order_id, symbol, cycle_id, cycle_status)
        """
        query = """
        WITH orphan_orders AS (
            -- Ordres d'entrée NEW pour cycles terminés
            SELECT 
                te.order_id,
                te.symbol,
                tc.id as cycle_id,
                tc.status as cycle_status,
                'entry' as order_type
            FROM trade_cycles tc
            JOIN trade_executions te ON tc.entry_order_id = te.order_id
            WHERE tc.status IN ('completed', 'failed', 'canceled')
            AND te.status = 'NEW'
            AND tc.created_at > NOW() - INTERVAL '7 days'
            
            UNION ALL
            
            -- Ordres de sortie NEW pour cycles terminés
            SELECT 
                te.order_id,
                te.symbol,
                tc.id as cycle_id,
                tc.status as cycle_status,
                'exit' as order_type
            FROM trade_cycles tc
            JOIN trade_executions te ON tc.exit_order_id = te.order_id
            WHERE tc.status IN ('completed', 'failed', 'canceled')
            AND te.status = 'NEW'
            AND tc.created_at > NOW() - INTERVAL '7 days'
        )
        SELECT DISTINCT order_id, symbol, cycle_id, cycle_status
        FROM orphan_orders
        ORDER BY cycle_id DESC
        """
        
        orphan_orders = []
        
        with DBContextManager(self.db_url) as db:
            with db.get_connection() as conn:
                result = conn.execute_query(query, fetch_all=True)
                
                if result:
                    for row in result:
                        orphan_orders.append((
                            row['order_id'],
                            row['symbol'],
                            row['cycle_id'],
                            row['cycle_status']
                        ))
        
        logger.info(f"✅ Trouvé {len(orphan_orders)} ordres orphelins")
        return orphan_orders
    
    def cleanup_orders(self) -> int:
        """
        Annule tous les ordres orphelins sur Binance.
        
        Returns:
            Nombre d'ordres annulés avec succès
        """
        orphan_orders = self.find_orphan_orders()
        
        if not orphan_orders:
            logger.info("✅ Aucun ordre orphelin à nettoyer")
            return 0
        
        # Grouper par symbole pour optimiser
        orders_by_symbol = {}
        for order_id, symbol, cycle_id, cycle_status in orphan_orders:
            if symbol not in orders_by_symbol:
                orders_by_symbol[symbol] = []
            orders_by_symbol[symbol].append((order_id, cycle_id, cycle_status))
        
        total_canceled = 0
        total_errors = 0
        
        for symbol, orders in orders_by_symbol.items():
            logger.info(f"\n🔧 Traitement de {len(orders)} ordres pour {symbol}")
            
            for order_id, cycle_id, cycle_status in orders:
                try:
                    if self.dry_run:
                        logger.info(f"[DRY RUN] Annulerait l'ordre {order_id} (cycle {cycle_id[:8]}... - {cycle_status})")
                        total_canceled += 1
                    else:
                        # Vérifier d'abord le statut de l'ordre sur Binance
                        order_status = self.binance.get_order_status(symbol, order_id)
                        
                        if order_status and order_status.status.value == 'NEW':
                            # L'ordre est bien ouvert, on peut l'annuler
                            result = self.binance.cancel_order(symbol, order_id)
                            
                            if result:
                                logger.info(f"✅ Ordre {order_id} annulé (cycle {cycle_id[:8]}... - {cycle_status})")
                                total_canceled += 1
                                
                                # Mettre à jour le statut dans la DB
                                self._update_order_status(order_id, 'CANCELED')
                            else:
                                logger.error(f"❌ Échec d'annulation de l'ordre {order_id}")
                                total_errors += 1
                        else:
                            logger.info(f"⏭️ Ordre {order_id} n'est plus NEW, statut: {order_status.status if order_status else 'UNKNOWN'}")
                    
                    # Pause pour éviter de surcharger l'API
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"❌ Erreur lors du traitement de l'ordre {order_id}: {str(e)}")
                    total_errors += 1
        
        logger.info(f"\n📊 Résumé du nettoyage:")
        logger.info(f"   - Ordres annulés: {total_canceled}")
        logger.info(f"   - Erreurs: {total_errors}")
        logger.info(f"   - Total traité: {len(orphan_orders)}")
        
        return total_canceled
    
    def _update_order_status(self, order_id: str, new_status: str) -> None:
        """Met à jour le statut d'un ordre dans la DB."""
        query = """
        UPDATE trade_executions
        SET status = %s, updated_at = NOW()
        WHERE order_id = %s
        """
        
        with DBContextManager(self.db_url) as db:
            with db.get_connection() as conn:
                conn.execute_query(query, (new_status, order_id), commit=True)


def main():
    """Point d'entrée principal du script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Nettoie les ordres orphelins sur Binance")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Exécuter vraiment (sinon dry-run par défaut)"
    )
    
    args = parser.parse_args()
    
    logger.info("🧹 Démarrage du nettoyage des ordres orphelins")
    logger.info(f"Mode: {'EXÉCUTION' if args.execute else 'DRY RUN (simulation)'}")
    
    cleaner = OrphanOrderCleaner(dry_run=not args.execute)
    cleaner.cleanup_orders()


if __name__ == "__main__":
    main()