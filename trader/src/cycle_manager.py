"""
Gestionnaire des cycles de trading.
Suit l'état des cycles de trading depuis l'ouverture jusqu'à la fermeture.
Un cycle représente une position complète (entrée + sortie).
"""
import logging
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import threading
from threading import RLock

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import get_db_url, TRADING_MODE
from shared.src.enums import OrderSide, OrderStatus, CycleStatus
from shared.src.schemas import TradeOrder, TradeExecution, TradeCycle
from shared.src.db_pool import DBContextManager, DBConnectionPool

from trader.src.binance_executor import BinanceExecutor

# Configuration du logging
logger = logging.getLogger(__name__)

class CycleManager:
    """
    Gestionnaire des cycles de trading.
    Crée, met à jour et suit l'état des cycles de trading.
    """
    
    def __init__(
        self,
        db_url: str: Optional[Any] = None,
        binance_executor: BinanceExecutor: Optional[Any] = None
    ) -> None:
        """
        Initialise le gestionnaire de cycles.
        
        Args:
            db_url: URL de connexion à la base de données
            binance_executor: Exécuteur Binance préexistant (optionnel)
        """
        self.db_url = db_url or get_db_url()
        self.binance_executor = binance_executor or BinanceExecutor()
        self.demo_mode = TRADING_MODE.lower() == 'demo'
        
        # Dictionnaire des cycles actifs {id_cycle: cycle}
        self.active_cycles: Dict[str, TradeCycle] = {}
        
        # Mutex pour l'accès concurrent aux cycles
        self.cycles_lock = RLock()
        
        # Initialiser le pool de connexions DB
        try:
            self.db_pool = DBConnectionPool.get_instance()
            self._init_db_schema()
            self._load_active_cycles_from_db()
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur lors de l'initialisation de la base de données: {str(e)
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise}")
        
        logger.info(f"✅ CycleManager initialisé en mode {'DÉMO' if self.demo_mode else 'RÉEL'}")
    
    def _init_db_schema(self) -> None:
        """
        Initialise le schéma de la base de données.
        Crée les tables nécessaires si elles n'existent pas.
        """
        try:
            with DBContextManager() as cursor:
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
        
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur lors de l'initialisation du schéma de base de données: {str(e)
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise}")
            raise
    
    def _save_execution_to_db(self, execution: TradeExecution, cycle_id: Optional[str] = None) -> bool:
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
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
                
                # Convertir les objets enum en chaînes
                side = execution.side.value if hasattr(execution.side, 'value') else str(execution.side)
                status = execution.status.value if hasattr(execution.status, 'value') else str(execution.status)
                role = execution.role.value if execution.role and hasattr(execution.role, 'value') else None
                
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
                
                # Convertir les objets enum en chaînes
                side = execution.side.value if hasattr(execution.side, 'value') else str(execution.side)
                status = execution.status.value if hasattr(execution.status, 'value') else str(execution.status)
                role = execution.role.value if execution.role and hasattr(execution.role, 'value') else None
                
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
        
            # Exécuter la requête
            with DBContextManager() as cursor:
                cursor.execute(query, params)
        
            logger.debug(f"✅ Exécution {execution.order_id} enregistrée en base de données")
            return True
    
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur lors de l'enregistrement de l'exécution en base de données: {str(e)
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except Exception as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise}")
            return False
    
    def _load_active_cycles_from_db(self) -> None:
        """
        Charge les cycles actifs depuis la base de données.
        """
        try:
            with DBContextManager() as cursor:
                cursor.execute("""
                SELECT * FROM trade_cycles
                WHERE status NOT IN ('completed', 'canceled', 'failed')
                ORDER BY created_at DESC
                """)
                
                cycle_records = cursor.fetchall()
                
                # Obtenir les noms des colonnes
                column_names = [desc[0] for desc in cursor.description]
                
                with self.cycles_lock:
                    self.active_cycles = {}
                    for record in cycle_records:
                        # Convertir le tuple en dictionnaire
                        cycle_data = dict(zip(column_names, record))
                        
                        # Convertir les données SQL en objet TradeCycle
                        cycle = TradeCycle(
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
                        
                        self.active_cycles[cycle.id] = cycle
                
                logger.info(f"✅ {len(self.active_cycles)} cycles actifs chargés depuis la base de données")
        
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur lors du chargement des cycles actifs: {str(e)
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except Exception as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise}")
    
    def create_cycle(self, symbol: str, strategy: str, side: Union[OrderSide, str], 
                price: float, quantity: float, pocket: Optional[str] = None,
                target_price: Optional[float] = None, stop_price: Optional[float] = None,
                trailing_delta: Optional[float] = None) -> Optional[TradeCycle]:
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if strategy is not None and not isinstance(strategy, str):
            raise TypeError(f"strategy doit être une chaîne, pas {type(strategy).__name__}")
        if price is not None and not isinstance(price, (int, float)):
            raise TypeError(f"price doit être un nombre, pas {type(price).__name__}")
        if quantity is not None and not isinstance(quantity, (int, float)):
            raise TypeError(f"quantity doit être un nombre, pas {type(quantity).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if strategy is not None and not isinstance(strategy, str):
            raise TypeError(f"strategy doit être une chaîne, pas {type(strategy).__name__}")
        if price is not None and not isinstance(price, (int, float)):
            raise TypeError(f"price doit être un nombre, pas {type(price).__name__}")
        if quantity is not None and not isinstance(quantity, (int, float)):
            raise TypeError(f"quantity doit être un nombre, pas {type(quantity).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if strategy is not None and not isinstance(strategy, str):
            raise TypeError(f"strategy doit être une chaîne, pas {type(strategy).__name__}")
        if price is not None and not isinstance(price, (int, float)):
            raise TypeError(f"price doit être un nombre, pas {type(price).__name__}")
        if quantity is not None and not isinstance(quantity, (int, float)):
            raise TypeError(f"quantity doit être un nombre, pas {type(quantity).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if strategy is not None and not isinstance(strategy, str):
            raise TypeError(f"strategy doit être une chaîne, pas {type(strategy).__name__}")
        if price is not None and not isinstance(price, (int, float)):
            raise TypeError(f"price doit être un nombre, pas {type(price).__name__}")
        if quantity is not None and not isinstance(quantity, (int, float)):
            raise TypeError(f"quantity doit être un nombre, pas {type(quantity).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if strategy is not None and not isinstance(strategy, str):
            raise TypeError(f"strategy doit être une chaîne, pas {type(strategy).__name__}")
        if price is not None and not isinstance(price, (int, float)):
            raise TypeError(f"price doit être un nombre, pas {type(price).__name__}")
        if quantity is not None and not isinstance(quantity, (int, float)):
            raise TypeError(f"quantity doit être un nombre, pas {type(quantity).__name__}")
        """
        Crée un nouveau cycle de trading et exécute l'ordre d'entrée.
    
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            strategy: Nom de la stratégie
            side: Côté de l'ordre d'entrée (BUY ou SELL)
            price: Prix d'entrée
            quantity: Quantité à trader
            pocket: Nom de la poche de capital
            target_price: Prix cible pour la sortie
            stop_price: Prix de stop-loss
            trailing_delta: Delta pour le trailing stop
        
        Returns:
            Cycle créé ou None en cas d'erreur
        """
        try:
            # Convertir side en OrderSide s'il s'agit d'une chaîne
            if isinstance(side, str):
                side = OrderSide(side)

            # Générer un ID unique pour le cycle
            cycle_id = f"cycle_{uuid.uuid4().hex[:16]}"
            now = datetime.now()
        
            # Créer l'objet cycle
            cycle = TradeCycle(
                id=cycle_id,
                symbol=symbol,
                strategy=strategy,
                status=CycleStatus.INITIATING,
                entry_price=None,  # Sera mis à jour après l'exécution
                quantity=quantity,
                target_price=target_price,
                stop_price=stop_price,
                trailing_delta=trailing_delta,
                created_at=now,
                updated_at=now,
                pocket=pocket,
                demo=self.demo_mode
            )
        
            # Enregistrer le cycle en base de données
            self._save_cycle_to_db(cycle)
        
            # Créer l'ordre d'entrée
            entry_order = TradeOrder(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                client_order_id=f"entry_{cycle_id}",
                strategy=strategy,
                demo=self.demo_mode
            )
        
            # Exécuter l'ordre d'entrée
            logger.info(f"🔄 Exécution de l'ordre d'entrée pour le cycle {cycle_id}")
            execution = self.binance_executor.execute_order(entry_order)
        
            # Mettre à jour le cycle avec les informations de l'ordre
            with self.cycles_lock:
                cycle.entry_order_id = execution.order_id
                cycle.entry_price = execution.price
                cycle.status = CycleStatus.ACTIVE_BUY if side == OrderSide.BUY else CycleStatus.ACTIVE_SELL
                cycle.updated_at = datetime.now()
            
                # Stocker le cycle dans la mémoire
                self.active_cycles[cycle_id] = cycle
        
            # Enregistrer l'exécution et le cycle mis à jour en base de données
            self._save_execution_to_db(execution, cycle_id)
            self._save_cycle_to_db(cycle)
        
f"✅ Cycle {cycle_id} créé avec succès: {side.value if hasattr(side, 'value') else side} {quantity} {symbol} @ "
f"{execution.price}"
            return cycle
    
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur lors de la création du cycle: {str(e)
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except Exception as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise}")
            return None
    
    def close_cycle(self, cycle_id: str, exit_price: Optional[float] = None) -> bool:
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        if exit_price is not None and not isinstance(exit_price, (int, float)):
            raise TypeError(f"exit_price doit être un nombre, pas {type(exit_price).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        if exit_price is not None and not isinstance(exit_price, (int, float)):
            raise TypeError(f"exit_price doit être un nombre, pas {type(exit_price).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        if exit_price is not None and not isinstance(exit_price, (int, float)):
            raise TypeError(f"exit_price doit être un nombre, pas {type(exit_price).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        if exit_price is not None and not isinstance(exit_price, (int, float)):
            raise TypeError(f"exit_price doit être un nombre, pas {type(exit_price).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        if exit_price is not None and not isinstance(exit_price, (int, float)):
            raise TypeError(f"exit_price doit être un nombre, pas {type(exit_price).__name__}")
        """
        Ferme un cycle de trading en exécutant l'ordre de sortie.
        
        Args:
            cycle_id: ID du cycle à fermer
            exit_price: Prix de sortie (optionnel, sinon au marché)
            
        Returns:
            True si la fermeture a réussi, False sinon
        """
        try:
            # Récupérer le cycle
            with self.cycles_lock:
                if cycle_id not in self.active_cycles:
                    logger.warning(f"⚠️ Cycle {cycle_id} non trouvé dans les cycles actifs")
                    return False
                
                cycle = self.active_cycles[cycle_id]
            
            # Vérifier que le cycle peut être fermé
            if cycle.status not in [CycleStatus.WAITING_BUY, CycleStatus.ACTIVE_BUY, 
                                   CycleStatus.WAITING_SELL, CycleStatus.ACTIVE_SELL]:
                logger.warning(f"⚠️ Impossible de fermer le cycle {cycle_id} avec le statut {cycle.status}")
                return False
            
            # Déterminer le côté de l'ordre de sortie (inverse de l'entrée)
            if cycle.status in [CycleStatus.WAITING_BUY, CycleStatus.ACTIVE_BUY]:
                exit_side = OrderSide.SELL
            else:
                exit_side = OrderSide.BUY
            
            # Créer l'ordre de sortie
            exit_order = TradeOrder(
                symbol=cycle.symbol,
                side=exit_side,
                quantity=cycle.quantity,
                price=exit_price,  # None pour un ordre au marché
                client_order_id=f"exit_{cycle_id}",
                strategy=cycle.strategy,
                demo=cycle.demo
            )
            
            # Exécuter l'ordre de sortie
            logger.info(f"🔄 Exécution de l'ordre de sortie pour le cycle {cycle_id}")
            execution = self.binance_executor.execute_order(exit_order)
            
            # Calculer le profit/perte
            entry_value = cycle.entry_price * cycle.quantity
            exit_value = execution.price * execution.quantity
            
            if exit_side == OrderSide.SELL:
                # Si on vend, profit = sortie - entrée
                profit_loss = exit_value - entry_value
            else:
                # Si on achète (pour clôturer une vente), profit = entrée - sortie
                profit_loss = entry_value - exit_value
            
            # Calculer le pourcentage de profit/perte
            profit_loss_percent = (profit_loss / entry_value) * 100
            
            # Mettre à jour le cycle
            with self.cycles_lock:
                cycle.exit_order_id = execution.order_id
                cycle.exit_price = execution.price
                cycle.status = CycleStatus.COMPLETED
                cycle.profit_loss = profit_loss
                cycle.profit_loss_percent = profit_loss_percent
                cycle.completed_at = datetime.now()
                cycle.updated_at = datetime.now()
            
            # Enregistrer l'exécution et le cycle mis à jour en base de données
            self._save_execution_to_db(execution, cycle_id)
            self._save_cycle_to_db(cycle)
            
            # Supprimer le cycle des cycles actifs
            with self.cycles_lock:
                self.active_cycles.pop(cycle_id, None)
            
            logger.info(f"✅ Cycle {cycle_id} fermé avec succès: P&L = {profit_loss:.2f} ({profit_loss_percent:.2f}%)")
            return True
        
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur lors de la fermeture du cycle {cycle_id}: {str(e)
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except Exception as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise}")
            return False
    
    def cancel_cycle(self, cycle_id: str, reason: str = "Annulation manuelle") -> bool:
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        if reason is not None and not isinstance(reason, str):
            raise TypeError(f"reason doit être une chaîne, pas {type(reason).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        if reason is not None and not isinstance(reason, str):
            raise TypeError(f"reason doit être une chaîne, pas {type(reason).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        if reason is not None and not isinstance(reason, str):
            raise TypeError(f"reason doit être une chaîne, pas {type(reason).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        if reason is not None and not isinstance(reason, str):
            raise TypeError(f"reason doit être une chaîne, pas {type(reason).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        if reason is not None and not isinstance(reason, str):
            raise TypeError(f"reason doit être une chaîne, pas {type(reason).__name__}")
        """
        Annule un cycle de trading.
        Si un ordre est actif, il est annulé sur Binance.
        
        Args:
            cycle_id: ID du cycle à annuler
            reason: Raison de l'annulation
            
        Returns:
            True si l'annulation a réussi, False sinon
        """
        try:
            # Récupérer le cycle
            with self.cycles_lock:
                if cycle_id not in self.active_cycles:
                    logger.warning(f"⚠️ Cycle {cycle_id} non trouvé dans les cycles actifs")
                    return False
                
                cycle = self.active_cycles[cycle_id]
            
            # Vérifier si des ordres doivent être annulés
            if cycle.status in [CycleStatus.ACTIVE_BUY, CycleStatus.ACTIVE_SELL]:
                # Déterminer l'ordre à annuler
                order_id = cycle.entry_order_id if cycle.exit_order_id is None else cycle.exit_order_id
                
                if order_id:
                    # Annuler l'ordre sur Binance
                    logger.info(f"🔄 Annulation de l'ordre {order_id} pour le cycle {cycle_id}")
                    self.binance_executor.cancel_order(cycle.symbol, order_id)
            
            # Mettre à jour le cycle
            with self.cycles_lock:
                cycle.status = CycleStatus.CANCELED
                cycle.updated_at = datetime.now()
                # Ajouter la raison comme commentaire (pourrait être ajouté à la DB)
                if not hasattr(cycle, 'metadata'):
                    cycle.metadata = {}
                cycle.metadata['cancel_reason'] = reason
            
            # Enregistrer le cycle mis à jour en base de données
            self._save_cycle_to_db(cycle)
            
            # Supprimer le cycle des cycles actifs
            with self.cycles_lock:
                self.active_cycles.pop(cycle_id, None)
            
            logger.info(f"✅ Cycle {cycle_id} annulé: {reason}")
            return True
        
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur lors de l'annulation du cycle {cycle_id}: {str(e)
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except Exception as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise}")
            return False
    
    def update_stop_loss(self, cycle_id: str, new_stop_price: float) -> bool:
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        if new_stop_price is not None and not isinstance(new_stop_price, (int, float)):
            raise TypeError(f"new_stop_price doit être un nombre, pas {type(new_stop_price).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        if new_stop_price is not None and not isinstance(new_stop_price, (int, float)):
            raise TypeError(f"new_stop_price doit être un nombre, pas {type(new_stop_price).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        if new_stop_price is not None and not isinstance(new_stop_price, (int, float)):
            raise TypeError(f"new_stop_price doit être un nombre, pas {type(new_stop_price).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        if new_stop_price is not None and not isinstance(new_stop_price, (int, float)):
            raise TypeError(f"new_stop_price doit être un nombre, pas {type(new_stop_price).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        if new_stop_price is not None and not isinstance(new_stop_price, (int, float)):
            raise TypeError(f"new_stop_price doit être un nombre, pas {type(new_stop_price).__name__}")
        """
        Met à jour le stop-loss d'un cycle.
        
        Args:
            cycle_id: ID du cycle
            new_stop_price: Nouveau prix de stop-loss
            
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        try:
            # Récupérer le cycle
            with self.cycles_lock:
                if cycle_id not in self.active_cycles:
                    logger.warning(f"⚠️ Cycle {cycle_id} non trouvé dans les cycles actifs")
                    return False
                
                cycle = self.active_cycles[cycle_id]
                
                # Mettre à jour le stop-loss
                cycle.stop_price = new_stop_price
                cycle.updated_at = datetime.now()
            
            # Enregistrer le cycle mis à jour en base de données
            self._save_cycle_to_db(cycle)
            
            logger.info(f"✅ Stop-loss mis à jour pour le cycle {cycle_id}: {new_stop_price}")
            return True
        
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur lors de la mise à jour du stop-loss pour le cycle {cycle_id}: {str(e)
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except Exception as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except (ValueError, TypeError) as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise}")
            return False
    
    def get_cycle(self, cycle_id: str) -> Optional[TradeCycle]:
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        # Validation des paramètres
        if cycle_id is not None and not isinstance(cycle_id, str):
            raise TypeError(f"cycle_id doit être une chaîne, pas {type(cycle_id).__name__}")
        """
        Récupère un cycle par son ID.
        
        Args:
            cycle_id: ID du cycle
            
        Returns:
            Cycle ou None si non trouvé
        """
        with self.cycles_lock:
            return self.active_cycles.get(cycle_id)
    
    def get_active_cycles(self, symbol: Optional[str] = None, strategy: Optional[str] = None) -> List[TradeCycle]:
        """
        Récupère les cycles actifs, avec filtrage optionnel.
        
        Args:
            symbol: Filtrer par symbole (optionnel)
            strategy: Filtrer par stratégie (optionnel)
            
        Returns:
            Liste des cycles actifs filtrés
        """
        with self.cycles_lock:
            cycles = list(self.active_cycles.values())
        
        # Filtrer par symbole
        if symbol:
            cycles = [c for c in cycles if c.symbol == symbol]
        
        # Filtrer par stratégie
        if strategy:
            cycles = [c for c in cycles if c.strategy == strategy]
        
        return cycles
    
    def check_stop_losses(self, symbol: str, current_price: float) -> None:
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if current_price is not None and not isinstance(current_price, (int, float)):
            raise TypeError(f"current_price doit être un nombre, pas {type(current_price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if current_price is not None and not isinstance(current_price, (int, float)):
            raise TypeError(f"current_price doit être un nombre, pas {type(current_price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if current_price is not None and not isinstance(current_price, (int, float)):
            raise TypeError(f"current_price doit être un nombre, pas {type(current_price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if current_price is not None and not isinstance(current_price, (int, float)):
            raise TypeError(f"current_price doit être un nombre, pas {type(current_price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if current_price is not None and not isinstance(current_price, (int, float)):
            raise TypeError(f"current_price doit être un nombre, pas {type(current_price).__name__}")
        """
        Vérifie les stop-loss pour un symbole donné et ferme les cycles si nécessaire.
        
        Args:
            symbol: Symbole à vérifier
            current_price: Prix actuel
        """
        cycles_to_close = []
        
        # Identifier les cycles à fermer
        with self.cycles_lock:
            for cycle_id, cycle in self.active_cycles.items():
                if cycle.symbol != symbol or cycle.stop_price is None:
                    continue
                
                # Si c'est un achat, fermer si le prix passe sous le stop
                if cycle.status in [CycleStatus.ACTIVE_BUY, CycleStatus.WAITING_SELL] and current_price <= cycle.stop_price:
                    cycles_to_close.append(cycle_id)
                
                # Si c'est une vente, fermer si le prix passe au-dessus du stop
                elif cycle.status in [CycleStatus.ACTIVE_SELL, CycleStatus.WAITING_BUY] and current_price >= cycle.stop_price:
                    cycles_to_close.append(cycle_id)
        
        # Fermer les cycles identifiés
        for cycle_id in cycles_to_close:
            logger.info(f"🔴 Stop-loss déclenché pour le cycle {cycle_id} au prix {current_price}")
            self.close_cycle(cycle_id)
    
    def update_trailing_stops(self, symbol: str, current_price: float) -> None:
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if current_price is not None and not isinstance(current_price, (int, float)):
            raise TypeError(f"current_price doit être un nombre, pas {type(current_price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if current_price is not None and not isinstance(current_price, (int, float)):
            raise TypeError(f"current_price doit être un nombre, pas {type(current_price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if current_price is not None and not isinstance(current_price, (int, float)):
            raise TypeError(f"current_price doit être un nombre, pas {type(current_price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if current_price is not None and not isinstance(current_price, (int, float)):
            raise TypeError(f"current_price doit être un nombre, pas {type(current_price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if current_price is not None and not isinstance(current_price, (int, float)):
            raise TypeError(f"current_price doit être un nombre, pas {type(current_price).__name__}")
        """
        Met à jour les stops trailing pour un symbole donné.
        
        Args:
            symbol: Symbole à mettre à jour
            current_price: Prix actuel
        """
        with self.cycles_lock:
            for cycle_id, cycle in self.active_cycles.items():
                if cycle.symbol != symbol or cycle.trailing_delta is None:
                    continue
                
                # Pour les cycles d'achat
                if cycle.status in [CycleStatus.ACTIVE_BUY, CycleStatus.WAITING_SELL]:
                    # Si le prix a monté au-dessus du dernier maximum connu
                    if not hasattr(cycle, 'max_price') or current_price > cycle.max_price:
                        # Mise à jour du prix maximum
                        cycle.max_price = current_price
                        
                        # Calcul du nouveau stop-loss trailing
                        new_stop = current_price * (1 - cycle.trailing_delta / 100)
                        
                        # Mise à jour du stop-loss si plus haut que l'ancien
                        if cycle.stop_price is None or new_stop > cycle.stop_price:
                            cycle.stop_price = new_stop
                            cycle.updated_at = datetime.now()
                            logger.info(f"🔄 Trailing stop mis à jour pour le cycle {cycle_id}: {new_stop}")
                            
                            # Enregistrer la mise à jour en DB
                            self._save_cycle_to_db(cycle)
                
                # Pour les cycles de vente
                elif cycle.status in [CycleStatus.ACTIVE_SELL, CycleStatus.WAITING_BUY]:
                    # Si le prix a baissé en-dessous du dernier minimum connu
                    if not hasattr(cycle, 'min_price') or current_price < cycle.min_price:
                        # Mise à jour du prix minimum
                        cycle.min_price = current_price
                        
                        # Calcul du nouveau stop-loss trailing
                        new_stop = current_price * (1 + cycle.trailing_delta / 100)
                        
                        # Mise à jour du stop-loss si plus bas que l'ancien
                        if cycle.stop_price is None or new_stop < cycle.stop_price:
                            cycle.stop_price = new_stop
                            cycle.updated_at = datetime.now()
                            logger.info(f"🔄 Trailing stop mis à jour pour le cycle {cycle_id}: {new_stop}")
                            
                            # Enregistrer la mise à jour en DB
                            self._save_cycle_to_db(cycle)
    
    def check_target_prices(self, symbol: str, current_price: float) -> None:
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if current_price is not None and not isinstance(current_price, (int, float)):
            raise TypeError(f"current_price doit être un nombre, pas {type(current_price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if current_price is not None and not isinstance(current_price, (int, float)):
            raise TypeError(f"current_price doit être un nombre, pas {type(current_price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if current_price is not None and not isinstance(current_price, (int, float)):
            raise TypeError(f"current_price doit être un nombre, pas {type(current_price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if current_price is not None and not isinstance(current_price, (int, float)):
            raise TypeError(f"current_price doit être un nombre, pas {type(current_price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if current_price is not None and not isinstance(current_price, (int, float)):
            raise TypeError(f"current_price doit être un nombre, pas {type(current_price).__name__}")
        """
        Vérifie les prix cibles pour un symbole donné et ferme les cycles si nécessaire.
        
        Args:
            symbol: Symbole à vérifier
            current_price: Prix actuel
        """
        cycles_to_close = []
        
        # Identifier les cycles à fermer
        with self.cycles_lock:
            for cycle_id, cycle in self.active_cycles.items():
                if cycle.symbol != symbol or cycle.target_price is None:
                    continue
                
                # Si c'est un achat, fermer si le prix atteint ou dépasse la cible
                if cycle.status in [CycleStatus.ACTIVE_BUY, CycleStatus.WAITING_SELL] and current_price >= cycle.target_price:
                    cycles_to_close.append(cycle_id)
                
                # Si c'est une vente, fermer si le prix atteint ou passe sous la cible
                elif cycle.status in [CycleStatus.ACTIVE_SELL, CycleStatus.WAITING_BUY] and current_price <= cycle.target_price:
                    cycles_to_close.append(cycle_id)
        
        # Fermer les cycles identifiés
        for cycle_id in cycles_to_close:
            logger.info(f"🎯 Prix cible atteint pour le cycle {cycle_id} au prix {current_price}")
            self.close_cycle(cycle_id)
    
    def process_price_update(self, symbol: str, price: float) -> None:
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if price is not None and not isinstance(price, (int, float)):
            raise TypeError(f"price doit être un nombre, pas {type(price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if price is not None and not isinstance(price, (int, float)):
            raise TypeError(f"price doit être un nombre, pas {type(price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if price is not None and not isinstance(price, (int, float)):
            raise TypeError(f"price doit être un nombre, pas {type(price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if price is not None and not isinstance(price, (int, float)):
            raise TypeError(f"price doit être un nombre, pas {type(price).__name__}")
        # Validation des paramètres
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        if price is not None and not isinstance(price, (int, float)):
            raise TypeError(f"price doit être un nombre, pas {type(price).__name__}")
        """
        Traite une mise à jour de prix pour un symbole.
        Vérifie les stops, targets et met à jour les trailing stops.
        
        Args:
            symbol: Symbole mis à jour
            price: Nouveau prix
        """
        # Vérifier les stop-loss
        self.check_stop_losses(symbol, price)
        
        # Mettre à jour les trailing stops
        self.update_trailing_stops(symbol, price)
        
        # Vérifier les prix cibles
        self.check_target_prices(symbol, price)
    
    def close(self) -> None:
        """
        Ferme proprement le gestionnaire de cycles.
        """
        logger.info("Fermeture du gestionnaire de cycles...")
        
        # Rien de spécial à fermer, le pool de connexions est géré globalement
        logger.info("✅ Gestionnaire de cycles fermé")
            
    def _save_cycle_to_db(self, cycle: TradeCycle) -> bool:
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
                
                # Convertir l'enum en chaîne
                status = cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status)
                
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
                
                # Convertir l'enum en chaîne
                status = cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status)
                
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
        
            # Exécuter la requête
            with DBContextManager() as cursor:
                cursor.execute(query, params)
        
            logger.debug(f"✅ Cycle {cycle.id} enregistré en base de données")
            return True
    
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur lors de l'enregistrement du cycle en base de données: {str(e)
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except Exception as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except Exception as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except Exception as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise}")
            return False