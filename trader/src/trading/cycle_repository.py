# trader/src/trading/cycle_repository.py
"""
Repository pour les cycles de trading.
S'occupe du stockage et de la récupération des cycles en base de données.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from shared.src.enums import OrderSide, OrderStatus, CycleStatus
from shared.src.schemas import TradeOrder, TradeExecution, TradeCycle
from shared.src.db_pool import DBContextManager, transaction

# Configuration du logging
logger = logging.getLogger(__name__)

class CycleRepository:
    """
    Repository pour les cycles de trading.
    Gère les opérations de base de données pour les cycles.
    """
    
    def __init__(self, db_url: str):
        """
        Initialise le repository.
        
        Args:
            db_url: URL de connexion à la base de données
        """
        self.db_url = db_url
        self._init_db_schema()
        logger.info("✅ CycleRepository initialisé")
    
    def _init_db_schema(self) -> None:
        """
        Initialise le schéma de la base de données.
        Crée les tables nécessaires si elles n'existent pas.
        """
        try:
            # Use a connection with autocommit=True for schema operations
            with DBContextManager(auto_transaction=False) as cursor:
                # Table des ordres/exécutions
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_executions (
                    order_id VARCHAR(50) PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    price NUMERIC(16, 8) NOT NULL,
                    quantity NUMERIC(16, 8) NOT NULL,
                    quote_quantity NUMERIC(16, 8) NOT NULL,
                    fee NUMERIC(16, 8),
                    fee_asset VARCHAR(10),
                    role VARCHAR(10),
                    timestamp TIMESTAMP NOT NULL,
                    cycle_id VARCHAR(50),
                    demo BOOLEAN NOT NULL DEFAULT FALSE
                );
                """)
                
                # Table des cycles de trading
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_cycles (
                    id VARCHAR(50) PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    strategy VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    entry_order_id VARCHAR(50),
                    exit_order_id VARCHAR(50),
                    entry_price NUMERIC(16, 8),
                    exit_price NUMERIC(16, 8),
                    quantity NUMERIC(16, 8),
                    target_price NUMERIC(16, 8),
                    stop_price NUMERIC(16, 8),
                    trailing_delta NUMERIC(16, 8),
                    profit_loss NUMERIC(16, 8),
                    profit_loss_percent NUMERIC(16, 8),
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    pocket VARCHAR(20),
                    demo BOOLEAN NOT NULL DEFAULT FALSE
                );
                """)
                
                # Créer un index sur status pour des requêtes plus rapides
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_cycles_status ON trade_cycles (status);
                """)
                
                # Créer un index sur le timestamp pour des requêtes chronologiques
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_executions_timestamp ON trade_executions (timestamp);
                """)
            
            logger.info("✅ Schéma de base de données initialisé")
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation du schéma de base de données: {str(e)}")
            raise
    
    def save_execution(self, execution: TradeExecution, cycle_id: Optional[str] = None) -> bool:
        """
        Enregistre une exécution d'ordre dans la base de données.
    
        Args:
            execution: Exécution à enregistrer
            cycle_id: ID du cycle associé (optionnel)
        
        Returns:
            True si l'enregistrement a réussi, False sinon
        """
        try:
            # Vérifier si l'exécution existe déjà
            exists = False
            with DBContextManager() as cursor:
                cursor.execute(
                    "SELECT order_id FROM trade_executions WHERE order_id = %s",
                    (execution.order_id,)
                )
                exists = cursor.fetchone() is not None
        
            # Convertir les objets enum en chaînes
            side = execution.side.value if hasattr(execution.side, 'value') else str(execution.side)
            status = execution.status.value if hasattr(execution.status, 'value') else str(execution.status)
            role = execution.role.value if execution.role and hasattr(execution.role, 'value') else None
            
            # Définir la requête SQL
            if exists:
                query = """
                UPDATE trade_executions SET
                    symbol = %s,
                    side = %s,
                    status = %s,
                    price = %s,
                    quantity = %s,
                    quote_quantity = %s,
                    fee = %s,
                    fee_asset = %s,
                    role = %s,
                    timestamp = %s,
                    cycle_id = %s,
                    demo = %s
                WHERE order_id = %s
                """
                
                params = (
                    execution.symbol,
                    side,
                    status,
                    execution.price,
                    execution.quantity,
                    execution.quote_quantity,
                    execution.fee,
                    execution.fee_asset,
                    role,
                    execution.timestamp,
                    cycle_id,
                    execution.demo,
                    execution.order_id
                )
            else:
                query = """
                INSERT INTO trade_executions
                (order_id, symbol, side, status, price, quantity, quote_quantity,
                fee, fee_asset, role, timestamp, cycle_id, demo)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                params = (
                    execution.order_id,
                    execution.symbol,
                    side,
                    status,
                    execution.price,
                    execution.quantity,
                    execution.quote_quantity,
                    execution.fee,
                    execution.fee_asset,
                    role,
                    execution.timestamp,
                    cycle_id,
                    execution.demo
                )
                
            # Exécuter la requête avec transaction explicite
            with transaction() as cursor:
                cursor.execute(query, params)
        
            logger.debug(f"✅ Exécution {execution.order_id} enregistrée en base de données")
            return True
    
        except Exception as e:
            logger.error("❌ Erreur lors de l'enregistrement de l'exécution en base de données")
            logger.exception(e)  # Ajoute la trace complète de l'exception
            return False
    
    def save_cycle(self, cycle: TradeCycle) -> bool:
        """
        Enregistre un cycle de trading dans la base de données.
    
        Args:
            cycle: Cycle à enregistrer
        
        Returns:
            True si l'enregistrement a réussi, False sinon
        """
        try:
            # Vérifier si le cycle existe déjà
            exists = False
            with DBContextManager() as cursor:
                cursor.execute(
                    "SELECT id FROM trade_cycles WHERE id = %s",
                    (cycle.id,)
                )
                exists = cursor.fetchone() is not None
        
            # Convertir l'enum en chaîne
            status = cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status)
            
            # Vérifier l'existence de l'attribut 'confirmed'
            confirmed = getattr(cycle, 'confirmed', False)
            
            # Définir la requête SQL
            if exists:
                query = """
                UPDATE trade_cycles SET
                    symbol = %s,
                    strategy = %s,
                    status = %s,
                    entry_order_id = %s,
                    exit_order_id = %s,
                    entry_price = %s,
                    exit_price = %s,
                    quantity = %s,
                    target_price = %s,
                    stop_price = %s,
                    trailing_delta = %s,
                    profit_loss = %s,
                    profit_loss_percent = %s,
                    created_at = %s,
                    updated_at = %s,
                    completed_at = %s,
                    pocket = %s,
                    demo = %s
                WHERE id = %s
                """
                
                params = (
                    cycle.symbol,
                    cycle.strategy,
                    status,
                    cycle.entry_order_id,
                    cycle.exit_order_id,
                    cycle.entry_price,
                    cycle.exit_price,
                    cycle.quantity,
                    cycle.target_price,
                    cycle.stop_price,
                    cycle.trailing_delta,
                    cycle.profit_loss,
                    cycle.profit_loss_percent,
                    cycle.created_at,
                    cycle.updated_at,
                    cycle.completed_at,
                    cycle.pocket,
                    cycle.demo,
                    cycle.id
                )
            else:
                query = """
                INSERT INTO trade_cycles
                (id, symbol, strategy, status, entry_order_id, exit_order_id,
                entry_price, exit_price, quantity, target_price, stop_price,
                trailing_delta, profit_loss, profit_loss_percent, created_at,
                updated_at, completed_at, pocket, demo)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                params = (
                    cycle.id,
                    cycle.symbol,
                    cycle.strategy,
                    status,
                    cycle.entry_order_id,
                    cycle.exit_order_id,
                    cycle.entry_price,
                    cycle.exit_price,
                    cycle.quantity,
                    cycle.target_price,
                    cycle.stop_price,
                    cycle.trailing_delta,
                    cycle.profit_loss,
                    cycle.profit_loss_percent,
                    cycle.created_at,
                    cycle.updated_at,
                    cycle.completed_at,
                    cycle.pocket,
                    cycle.demo
                )
        
            # Exécuter la requête avec transaction explicite
            with transaction() as cursor:
                cursor.execute(query, params)
        
            logger.debug(f"✅ Cycle {cycle.id} enregistré en base de données")
            return True
    
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'enregistrement du cycle en base de données: {str(e)}")
            return False
    
    def get_cycle(self, cycle_id: str) -> Optional[TradeCycle]:
        """
        Récupère un cycle par son ID.
        
        Args:
            cycle_id: ID du cycle
            
        Returns:
            Cycle ou None si non trouvé
        """
        try:
            with DBContextManager() as cursor:
                cursor.execute(
                    "SELECT * FROM trade_cycles WHERE id = %s",
                    (cycle_id,)
                )
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                # Convertir le résultat en dictionnaire
                column_names = [desc[0] for desc in cursor.description]
                cycle_data = dict(zip(column_names, result))
                
                # Convertir les données en objet TradeCycle
                return self._create_cycle_from_data(cycle_data)
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération du cycle {cycle_id}: {str(e)}")
            return None
    
    def get_active_cycles(self, symbol: Optional[str] = None, strategy: Optional[str] = None) -> List[TradeCycle]:
        """
        Récupère les cycles actifs, avec filtrage optionnel.
        
        Args:
            symbol: Filtrer par symbole (optionnel)
            strategy: Filtrer par stratégie (optionnel)
            
        Returns:
            Liste des cycles actifs filtrés
        """
        try:
            # Préparer la clause WHERE
            where_clauses = ["status NOT IN ('completed', 'canceled', 'failed')"]
            params = []
            
            if symbol:
                where_clauses.append("symbol = %s")
                params.append(symbol)
            
            if strategy:
                where_clauses.append("strategy = %s")
                params.append(strategy)
            
            where_clause = " AND ".join(where_clauses)
            
            # Exécuter la requête - IMPORTANT: auto_transaction=False
            with DBContextManager(auto_transaction=False) as cursor:
                query = f"""
                SELECT * FROM trade_cycles
                WHERE {where_clause}
                ORDER BY created_at DESC
                """
                
                cursor.execute(query, params)
                
                # Récupérer les résultats
                cycle_records = cursor.fetchall()
                
                # Obtenir les noms des colonnes
                column_names = [desc[0] for desc in cursor.description]
                
                # Convertir les résultats en objets TradeCycle
                cycles = []
                for record in cycle_records:
                    # Convertir le tuple en dictionnaire
                    cycle_data = dict(zip(column_names, record))
                    
                    # Convertir les données en objet TradeCycle
                    cycle = self._create_cycle_from_data(cycle_data)
                    cycles.append(cycle)
                
                return cycles
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des cycles actifs: {str(e)}")
            return []
    
    def _create_cycle_from_data(self, cycle_data: Dict[str, Any]) -> TradeCycle:
        """
        Crée un objet TradeCycle à partir des données de la base de données.
        
        Args:
            cycle_data: Données du cycle
            
        Returns:
            Objet TradeCycle
        """
        return TradeCycle(
            id=cycle_data['id'],
            symbol=cycle_data['symbol'],
            strategy=cycle_data['strategy'],
            status=CycleStatus(cycle_data['status']),
            entry_order_id=cycle_data['entry_order_id'],
            exit_order_id=cycle_data['exit_order_id'],
            entry_price=float(cycle_data['entry_price']) if cycle_data['entry_price'] else None,
            exit_price=float(cycle_data['exit_price']) if cycle_data['exit_price'] else None,
            quantity=float(cycle_data['quantity']) if cycle_data['quantity'] else None,
            target_price=float(cycle_data['target_price']) if cycle_data['target_price'] else None,
            stop_price=float(cycle_data['stop_price']) if cycle_data['stop_price'] else None,
            trailing_delta=float(cycle_data['trailing_delta']) if cycle_data['trailing_delta'] else None,
            profit_loss=float(cycle_data['profit_loss']) if cycle_data['profit_loss'] else None,
            profit_loss_percent=float(cycle_data['profit_loss_percent']) if cycle_data['profit_loss_percent'] else None,
            created_at=cycle_data['created_at'],
            updated_at=cycle_data['updated_at'],
            completed_at=cycle_data['completed_at'],
            pocket=cycle_data['pocket'],
            demo=cycle_data['demo']
        )