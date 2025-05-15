"""
Gestionnaire des poches de capital.
Gère la répartition du capital en différentes poches pour le trading.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import time

# Importer les modules partagés
import sys
import os
import json                                     # NEW
from shared.src.redis_client import RedisClient # NEW
from urllib.parse import urljoin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import POCKET_CONFIG
from shared.src.schemas import PocketSummary

from portfolio.src.models import DBManager, SharedCache

# Configuration du logging
logger = logging.getLogger(__name__)

class PocketError(Exception):
    """Exception de base pour les erreurs de poches."""
    pass

class InsufficientFundsError(PocketError):
    """Exception levée lorsqu'une poche n'a pas assez de fonds."""
    pass

class PocketNotFoundError(PocketError):
    """Exception levée lorsqu'une poche n'existe pas."""
    pass

class PocketManager:
    """
    Gestionnaire des poches de capital.
    Gère la répartition du capital entre les poches active, tampon et sécurité.
    """
    
    def __init__(self, db_manager: DBManager = None):
        """
        Initialise le gestionnaire de poches.
        
        Args:
            db_manager: Gestionnaire de base de données préexistant (optionnel)
        """
        self.db = db_manager or DBManager()
        
        # Configuration des poches
        self.pocket_config = POCKET_CONFIG
        
        # Initialiser les poches si nécessaire
        self._ensure_pockets_exist()
        
        logger.info("✅ PocketManager initialisé")
    
    def _ensure_pockets_exist(self) -> None:
        """
        S'assure que les poches existent dans la base de données.
        Crée les poches si elles n'existent pas.
        """
        # Vérifier les poches existantes
        query = "SELECT pocket_type FROM capital_pockets"
        result = self.db.execute_query(query, fetch_all=True)
        
        existing_pockets = set(row['pocket_type'] for row in result) if result else set()
        
        # Poches requises
        required_pockets = {'active', 'buffer', 'safety'}
        
        # Déterminer les poches manquantes
        missing_pockets = required_pockets - existing_pockets
        
        if missing_pockets:
            logger.info(f"Création des poches manquantes: {', '.join(missing_pockets)}")
            
            # Calculer la valeur totale du portefeuille (ou utiliser une valeur par défaut)
            total_value = self._get_total_portfolio_value() or 1000.0  # Valeur par défaut
            
            # Préparer l'insertion
            values = []
            
            for pocket_type in missing_pockets:
                allocation = self.pocket_config.get(pocket_type, 0.0)
                current_value = total_value * allocation
                
                values.append((
                    pocket_type,
                    allocation * 100,  # Stocker en pourcentage
                    current_value,
                    0.0,  # Aucune valeur utilisée initialement
                    current_value,  # Disponible = courant
                    0  # Aucun cycle actif initialement
                ))
            
            # Exécuter l'insertion par lots
            query = """
            INSERT INTO capital_pockets
            (pocket_type, allocation_percent, current_value, used_value, available_value, active_cycles)
            VALUES %s
            """
            self.db.execute_batch(query, values)
            
            logger.info(f"✅ {len(missing_pockets)} poches créées")
            
            # Invalider le cache
            SharedCache.clear('pockets')
    
    def _get_total_portfolio_value(self) -> Optional[float]:
        """
        Calcule la valeur totale du portefeuille.
        
        Returns:
            Valeur totale ou None en cas d'erreur
        """
        # Essayer d'utiliser le cache
        cached_value = SharedCache.get('total_portfolio_value', max_age=10)
        if cached_value is not None:
            return cached_value
            
        query = """
        WITH latest_balances AS (
            SELECT 
                asset,
                MAX(timestamp) as latest_timestamp
            FROM 
                portfolio_balances
            GROUP BY 
                asset
        )
        SELECT 
            SUM(pb.value_usdc) as total_value
        FROM 
            portfolio_balances pb
        JOIN 
            latest_balances lb ON pb.asset = lb.asset AND pb.timestamp = lb.latest_timestamp
        WHERE
            pb.value_usdc IS NOT NULL
        """
        
        result = self.db.execute_query(query, fetch_one=True)
        
        total_value = None
        if result and result['total_value'] is not None:
            total_value = float(result['total_value'])
            # Mettre en cache
            SharedCache.set('total_portfolio_value', total_value)
        
        return total_value
    
    def get_pockets(self) -> List[PocketSummary]:
        """
        Récupère l'état actuel des poches.
        Utilise un cache pour améliorer les performances.
        
        Returns:
            Liste des poches de capital
        """
        # Vérifier le cache
        cached_pockets = SharedCache.get('pockets', max_age=5)
        if cached_pockets:
            return cached_pockets
            
        query = """
        SELECT 
            pocket_type,
            allocation_percent,
            current_value,
            used_value,
            available_value,
            active_cycles
        FROM 
            capital_pockets
        ORDER BY 
            CASE pocket_type
                WHEN 'active' THEN 1
                WHEN 'buffer' THEN 2
                WHEN 'safety' THEN 3
                ELSE 4
            END
        """
        
        result = self.db.execute_query(query, fetch_all=True)
        
        if not result:
            return []
        
        # Convertir en objets PocketSummary
        pockets = []
        for row in result:
            pocket = PocketSummary(
                pocket_type=row['pocket_type'],
                allocation_percent=float(row['allocation_percent']),
                current_value=float(row['current_value']),
                used_value=float(row['used_value']),
                available_value=float(row['available_value']),
                active_cycles=row['active_cycles']
            )
            pockets.append(pocket)
        
        # Mettre en cache
        SharedCache.set('pockets', pockets)
        
        return pockets
    def check_pocket_synchronization(self) -> bool:
        """
        Vérifie si les poches sont synchronisées avec les cycles **réellement confirmés**.

        Returns:
            True si la synchronisation est correcte, False sinon
        """
        try:
            # Récupérer les poches
            pockets_data = self._make_request_with_retry(
                urljoin(self.portfolio_api_url, "/pockets")
            )

            if not pockets_data:
                logger.error("Impossible de récupérer les données des poches")
                return False

            # Récupérer les cycles actifs
            trader_api_url = os.getenv("TRADER_API_URL", "http://trader:5002")
            active_cycles_data = self._make_request_with_retry(
                urljoin(trader_api_url, "/orders?status=active")
            )

            if not active_cycles_data:
                logger.error("Impossible de récupérer les cycles actifs")
                return False

            # ✅ Ne garder que les cycles confirmés (si le champ existe)
            confirmed_cycles = [c for c in active_cycles_data if c.get("confirmed") is True]

            active_cycle_count = len(confirmed_cycles)
            pocket_cycle_count = sum(p.get("active_cycles", 0) for p in pockets_data)

            trade_used_value = sum(
                float(c.get("quantity", 0)) * float(c.get("entry_price", 0))
                for c in confirmed_cycles
            )

            pocket_used_value = sum(float(p.get("used_value", 0)) for p in pockets_data)

            cycles_synced = abs(active_cycle_count - pocket_cycle_count) <= 1
            amount_diff_percent = abs(trade_used_value - pocket_used_value) / trade_used_value * 100 if trade_used_value > 0 else 0
            amounts_synced = amount_diff_percent <= 5.0

            if not cycles_synced or not amounts_synced:
                logger.warning(
                    f"⚠️ Désynchronisation détectée: {active_cycle_count} cycles confirmés vs {pocket_cycle_count} dans les poches. "
                    f"Montant utilisé: {trade_used_value:.2f} vs {pocket_used_value:.2f} (diff: {amount_diff_percent:.2f}%)"
                )

                reconcile_result = self._make_request_with_retry(
                    urljoin(self.portfolio_api_url, "/pockets/reconcile"),
                    method="POST"
                )

                if reconcile_result:
                    logger.info("✅ Réconciliation des poches demandée")
                    self.last_cache_update = 0
                    self._refresh_cache()
                    return False
                else:
                    logger.error("❌ Échec de la demande de réconciliation")
                    return False

            logger.info("✅ Poches correctement synchronisées (cycles confirmés seulement)")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification de synchronisation: {str(e)}")
            return False

        # --- Redis cache --------------------------------------------------
    def _publish_pockets_state(self) -> None:
        """
        Sauvegarde l'état complet des poches dans Redis et notifie les abonnés.
        """
        pockets = self.get_pockets()                 # List[ PocketSummary ]
        redis   = RedisClient()

        snapshot_key = "roottrading:pockets:snapshot"
        channel      = "roottrading:pockets.update"

        # Sérialiser les poches (dict) et stocker 30 s
        redis.set(snapshot_key,
                  json.dumps([p.dict() for p in pockets]),
                  ex=30)

        # Notifier les consommateurs (Coordinator) qu’un nouvel état est dispo
        redis.publish(channel, json.dumps({"ts": int(time.time())}))
        logger.debug("📡 Snapshot poches publié dans Redis")

    def update_pockets_allocation(self, total_value: float) -> bool:
        """
        Met à jour la répartition des poches en fonction de la valeur totale du portefeuille.
        
        Args:
            total_value: Valeur totale du portefeuille
            
        Returns:
            True si la mise à jour a réussi, False sinon
            
        Raises:
            ValueError: Si la valeur totale est négative
        """
        try:
            logger.info(f"Début update_pockets_allocation avec total_value={total_value}")
        
            if total_value < 0:
                raise ValueError(f"Valeur totale du portefeuille négative ({total_value})")
            
            if total_value == 0:
                logger.warning(f"⚠️ Valeur totale du portefeuille nulle, utilisation d'une valeur minimale")
                total_value = 100.0  # Valeur par défaut
                logger.info(f"Nouvelle valeur après correction: {total_value}")
            
            # Mettre en cache la valeur totale
            SharedCache.set('total_portfolio_value', total_value)
        
            # Récupérer l'état actuel des poches
            pockets = self.get_pockets()
        
            logger.info(f"Poches récupérées: {len(pockets)} poches")
        
            if not pockets:
                logger.warning("⚠️ Aucune poche trouvée")
                return False
            
            # Préparer la mise à jour par lots
            updates = []
            
            # Pour chaque poche, calculer les nouvelles valeurs
            for pocket in pockets:
                # Récupérer l'allocation configurée
                allocation = self.pocket_config.get(pocket.pocket_type, 0.0)
                logger.info(f"Poche {pocket.pocket_type}: allocation={allocation}")
                
                # Calculer la nouvelle valeur
                new_current_value = total_value * allocation
                
                # La valeur utilisée reste inchangée
                used_value = pocket.used_value
                
                # Calculer la nouvelle valeur disponible
                new_available_value = max(0, new_current_value - used_value)
                
                # Ajouter à la liste des mises à jour
                updates.append({
                    'pocket_type': pocket.pocket_type,
                    'allocation_percent': allocation * 100,
                    'current_value': new_current_value,
                    'available_value': new_available_value
                })
            
            # Exécuter les mises à jour en une seule transaction
            with self.db.conn.cursor() as cursor:
                for update in updates:
                    cursor.execute(
                        """
                        UPDATE capital_pockets
                        SET 
                            allocation_percent = %(allocation_percent)s,
                            current_value = %(current_value)s,
                            available_value = %(available_value)s,
                            updated_at = NOW()
                        WHERE 
                            pocket_type = %(pocket_type)s
                        """,
                        update
                    )
                
                self.db.conn.commit()
            
            # Invalider le cache
            SharedCache.clear('pockets')
            
            logger.info(f"✅ Allocation des poches mise à jour (total: {total_value:.2f})")
            
            return True
        
        except ValueError as e:
            # Erreur de validation
            logger.warning(f"⚠️ Erreur de validation: {str(e)}")
            # Utiliser la valeur minimale
            if 'négative' in str(e) or 'nulle' in str(e):
                try:
                    return self.update_pockets_allocation(100.0)
                except Exception as inner_e:
                    logger.error(f"❌ Échec de l'utilisation de la valeur par défaut: {str(inner_e)}")
                    return False
            return False
        except Exception as e:
            import traceback
            logger.error(f"❌ Exception dans update_pockets_allocation: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def reserve_funds(self, pocket_type: str, amount: float, cycle_id: str) -> bool:
        """
        Réserve des fonds dans une poche pour un cycle de trading.
        
        Args:
            pocket_type: Type de poche ('active', 'buffer', 'safety')
            amount: Montant à réserver
            cycle_id: ID du cycle de trading
            
        Returns:
            True si la réservation a réussi, False sinon
            
        Raises:
            InsufficientFundsError: Si la poche n'a pas assez de fonds
            PocketNotFoundError: Si la poche n'existe pas
        """
        if amount <= 0:
            logger.warning(f"⚠️ Montant invalide: {amount}")
            return False
        
        try:
            # Verrouiller la ligne pour éviter les conditions de course
            query = """
            SELECT 
                pocket_type,
                available_value,
                active_cycles
            FROM 
                capital_pockets
            WHERE 
                pocket_type = %s
            FOR UPDATE
            """
            
            # Démarrer une transaction explicite
            self.db.execute_query("BEGIN")
            
            result = self.db.execute_query(query, (pocket_type,), fetch_one=True, commit=False)
            
            if not result:
                # Annuler la transaction
                self.db.execute_query("ROLLBACK")
                raise PocketNotFoundError(f"Poche non trouvée: {pocket_type}")
            
            available_value = float(result['available_value'])
            active_cycles = int(result['active_cycles'])
            
            if available_value < amount:
                # Annuler la transaction
                self.db.execute_query("ROLLBACK")
                raise InsufficientFundsError(
                    f"Fonds insuffisants dans la poche {pocket_type}: {available_value} < {amount}"
                )
            
            # Mettre à jour la poche
            update_query = """
            UPDATE capital_pockets
            SET 
                used_value = used_value + %s,
                available_value = available_value - %s,
                active_cycles = active_cycles + 1,
                updated_at = NOW()
            WHERE 
                pocket_type = %s
            """
            
            update_result = self.db.execute_query(update_query, (amount, amount, pocket_type), commit=False)
            
            if update_result is None:
                # Annuler la transaction
                self.db.execute_query("ROLLBACK")
                logger.error(f"❌ Échec de la mise à jour de la poche {pocket_type}")
                return False
            
            # Enregistrer l'opération dans la table des transactions de poches (journal)
            journal_query = """
            INSERT INTO pocket_transactions
            (pocket_type, transaction_type, amount, cycle_id, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT DO NOTHING
            """
            
            try:
                self.db.execute_query(
                    journal_query, 
                    (pocket_type, 'reserve', amount, cycle_id),
                    commit=False
                )
            except Exception as e:
                # Si la table n'existe pas, on l'ignore
                logger.debug(f"Note: Table pocket_transactions peut ne pas exister: {str(e)}")
            
            # Valider la transaction
            self.db.execute_query("COMMIT")
            
            # Invalider le cache
            SharedCache.clear('pockets')
            
            logger.info(f"✅ {amount:.2f} réservés dans la poche {pocket_type} pour le cycle {cycle_id}")
            
            return True
            
        except PocketNotFoundError as e:
            logger.warning(f"⚠️ {str(e)}")
            return False
        except InsufficientFundsError as e:
            logger.warning(f"⚠️ {str(e)}")
            return False
        except Exception as e:
            # Annuler la transaction en cas d'erreur
            try:
                self.db.execute_query("ROLLBACK")
            except:
                pass
                
            logger.error(f"❌ Erreur lors de la réservation des fonds: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def release_funds(self, pocket_type: str, amount: float, cycle_id: str) -> bool:
        """
        Libère des fonds réservés dans une poche.
        
        Args:
            pocket_type: Type de poche ('active', 'buffer', 'safety')
            amount: Montant à libérer
            cycle_id: ID du cycle de trading
            
        Returns:
            True si la libération a réussi, False sinon
        """
        if amount <= 0:
            logger.warning(f"⚠️ Montant invalide: {amount}")
            return False
        
        try:
            # Verrouiller la ligne pour éviter les conditions de course
            query = """
            SELECT 
                pocket_type,
                used_value,
                active_cycles
            FROM 
                capital_pockets
            WHERE 
                pocket_type = %s
            FOR UPDATE
            """
            
            # Démarrer une transaction explicite
            self.db.execute_query("BEGIN")
            
            result = self.db.execute_query(query, (pocket_type,), fetch_one=True, commit=False)
            
            if not result:
                # Annuler la transaction
                self.db.execute_query("ROLLBACK")
                raise PocketNotFoundError(f"Poche non trouvée: {pocket_type}")
            
            used_value = float(result['used_value'])
            active_cycles = int(result['active_cycles'])
            
            # Vérifier si le montant à libérer n'est pas trop grand
            if amount > used_value:
                logger.warning(f"⚠️ Tentative de libérer plus que ce qui est utilisé: {amount} > {used_value}")
                # Limiter le montant à la valeur utilisée
                amount = used_value
            
            # Mettre à jour la poche
            query = """
            UPDATE capital_pockets
            SET 
                used_value = GREATEST(0, used_value - %s),
                available_value = available_value + %s,
                active_cycles = GREATEST(0, active_cycles - 1),
                updated_at = NOW()
            WHERE 
                pocket_type = %s
            """
            
            result = self.db.execute_query(query, (amount, amount, pocket_type), commit=False)
            
            if result is None:
                # Annuler la transaction
                self.db.execute_query("ROLLBACK")
                logger.error(f"❌ Échec de la libération des fonds de la poche {pocket_type}")
                return False
            
            # Enregistrer l'opération dans la table des transactions de poches (journal)
            journal_query = """
            INSERT INTO pocket_transactions
            (pocket_type, transaction_type, amount, cycle_id, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT DO NOTHING
            """
            
            try:
                self.db.execute_query(
                    journal_query, 
                    (pocket_type, 'release', amount, cycle_id),
                    commit=False
                )
            except Exception as e:
                # Si la table n'existe pas, on l'ignore
                logger.debug(f"Note: Table pocket_transactions peut ne pas exister: {str(e)}")
            
            # Valider la transaction
            self.db.execute_query("COMMIT")
            
            # Invalider le cache
            SharedCache.clear('pockets')
            
            logger.info(f"✅ {amount:.2f} libérés dans la poche {pocket_type} pour le cycle {cycle_id}")
            
            return True
            
        except PocketNotFoundError as e:
            logger.warning(f"⚠️ {str(e)}")
            return False
        except Exception as e:
            # Annuler la transaction en cas d'erreur
            try:
                self.db.execute_query("ROLLBACK")
            except:
                pass
                
            logger.error(f"❌ Erreur lors de la libération des fonds: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_available_funds(self, pocket_type: str) -> Optional[float]:
        """
        Récupère les fonds disponibles dans une poche.
        
        Args:
            pocket_type: Type de poche ('active', 'buffer', 'safety')
            
        Returns:
            Montant disponible ou None en cas d'erreur
        """
        # Essayer d'utiliser le cache
        cached_pockets = SharedCache.get('pockets', max_age=5)
        if cached_pockets:
            for pocket in cached_pockets:
                if pocket.pocket_type == pocket_type:
                    return pocket.available_value
        
        query = """
        SELECT available_value
        FROM capital_pockets
        WHERE pocket_type = %s
        """
        
        result = self.db.execute_query(query, (pocket_type,), fetch_one=True)
        
        if not result:
            logger.warning(f"⚠️ Poche non trouvée: {pocket_type}")
            return None
        
        return float(result['available_value'])
    
    def recalculate_active_cycles(self) -> bool:
        """
        Recalcule le nombre de cycles actifs pour chaque poche.
        Utile pour la réconciliation des données.
        
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        try:
            # Mise à jour des cycles actifs pour les poches ayant des cycles
            query = """
            WITH pocket_cycles AS (
                SELECT 
                    pocket,
                    COUNT(*) as cycle_count
                FROM 
                    trade_cycles
                WHERE 
                    status NOT IN ('completed', 'canceled', 'failed')
                    AND pocket IS NOT NULL
                GROUP BY 
                    pocket
            )
            UPDATE capital_pockets cp
            SET 
                active_cycles = COALESCE(pc.cycle_count, 0),
                updated_at = NOW()
            FROM 
                pocket_cycles pc
            WHERE 
                cp.pocket_type = pc.pocket
            """
            
            result = self.db.execute_query(query, commit=True)
            
            # Pour les poches sans cycles actifs
            zero_query = """
            UPDATE capital_pockets
            SET 
                active_cycles = 0,
                updated_at = NOW()
            WHERE 
                pocket_type NOT IN (
                    SELECT DISTINCT pocket
                    FROM trade_cycles
                    WHERE status NOT IN ('completed', 'canceled', 'failed')
                    AND pocket IS NOT NULL
                )
            """
            
            self.db.execute_query(zero_query, commit=True)
            
            # Invalider le cache
            SharedCache.clear('pockets')
            
            logger.info("✅ Nombre de cycles actifs recalculé pour toutes les poches")
            
            return result is not None
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du recalcul des cycles actifs: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def sync_with_trades(self) -> bool:
        """
        Synchronise les poches avec les trades actifs **réellement exécutés** (ordres FILLED).
        Recalcule les valeurs utilisées et disponibles.

        Returns:
            True si la synchronisation a réussi, False sinon
        """
        try:
            query = """
            WITH trade_values AS (
                SELECT 
                    tc.pocket,
                    SUM(te.price * te.quantity) as used_value
                FROM 
                    trade_cycles tc
                JOIN 
                    trade_executions te ON tc.entry_order_id = te.order_id
                WHERE 
                    tc.status NOT IN ('completed', 'canceled', 'failed')
                    AND te.status = 'FILLED'
                    AND tc.pocket IS NOT NULL
                GROUP BY 
                    tc.pocket
            )
            SELECT 
                cp.pocket_type,
                cp.current_value,
                COALESCE(tv.used_value, 0) as calculated_used_value
            FROM 
                capital_pockets cp
            LEFT JOIN 
                trade_values tv ON cp.pocket_type = tv.pocket
            """

            result = self.db.execute_query(query, fetch_all=True)

            if not result:
                logger.warning("⚠️ Aucune poche trouvée pour la synchronisation")
                return False

            self.db.execute_query("BEGIN")
            success = True

            for row in result:
                pocket_type = row['pocket_type']
                current_value = float(row['current_value'])
                used_value = float(row['calculated_used_value'])
                available_value = max(0, current_value - used_value)

                update_query = """
                UPDATE capital_pockets
                SET 
                    used_value = %s,
                    available_value = %s,
                    updated_at = NOW()
                WHERE 
                    pocket_type = %s
                """

                update_result = self.db.execute_query(
                    update_query, 
                    (used_value, available_value, pocket_type), 
                    commit=False
                )

                if update_result is None:
                    logger.error(f"❌ Échec de la mise à jour de la poche {pocket_type}")
                    success = False
                    break

            if success:
                self.db.execute_query("COMMIT")
                self.recalculate_active_cycles()
                SharedCache.clear('pockets')
                logger.info("✅ Poches synchronisées avec les trades actifs exécutés")
            else:
                self.db.execute_query("ROLLBACK")
                logger.warning("⚠️ Synchronisation annulée en raison d'erreurs")

            return success

        except Exception as e:
            try:
                self.db.execute_query("ROLLBACK")
            except:
                pass
            logger.error(f"❌ Erreur lors de la synchronisation avec les trades: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    
    def close(self) -> None:
        """
        Ferme la connexion à la base de données.
        """
        if self.db:
            self.db.close()