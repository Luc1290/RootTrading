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
        
        # Démarrer le thread de nettoyage des réservations orphelines
        self._start_cleanup_thread()
        
        logger.info("✅ PocketManager initialisé")
    
    def _start_cleanup_thread(self):
        """Démarre un thread de nettoyage périodique des réservations orphelines."""
        import threading
        
        def cleanup_routine():
            while True:
                try:
                    # Nettoyer les réservations orphelines toutes les 30 minutes
                    time.sleep(1800)
                    self._cleanup_orphan_reservations()
                except Exception as e:
                    logger.error(f"❌ Erreur dans le thread de nettoyage: {str(e)}")
        
        cleanup_thread = threading.Thread(target=cleanup_routine, daemon=True)
        cleanup_thread.start()
        logger.info("🧹 Thread de nettoyage des réservations démarré")
    
    def _cleanup_orphan_reservations(self):
        """Nettoie les réservations orphelines dans Redis et la DB."""
        try:
            redis = RedisClient()
            cleaned_count = 0
            
            # Parcourir tous les types de poches et assets
            for pocket_type in ['active', 'buffer', 'safety']:
                for asset in ['USDC', 'BTC', 'ETH', 'BNB', 'SUI']:
                    redis_key = f"pocket:{pocket_type}:{asset}:reservations"
                    all_reservations = redis.hgetall(redis_key)
                    
                    for res_id, res_data in all_reservations.items():
                        try:
                            reservation = json.loads(res_data)
                            # Si la réservation a plus de 24h, la considérer comme orpheline
                            age_hours = (time.time() - reservation.get('timestamp', 0)) / 3600
                            
                            if age_hours > 24:
                                # Vérifier si le cycle existe encore
                                cycle_check = """
                                SELECT status FROM trade_cycles WHERE id = %s
                                """
                                result = self.db.execute_query(cycle_check, (reservation['cycle_id'],), fetch_one=True)
                                
                                # Si le cycle n'existe pas ou est terminé, libérer la réservation
                                if not result or result['status'] in ['completed', 'canceled', 'failed']:
                                    # Libérer les fonds
                                    self.release_funds(
                                        pocket_type=reservation['pocket_type'],
                                        amount=reservation['amount'],
                                        cycle_id=reservation['cycle_id'],
                                        asset=reservation['asset']
                                    )
                                    logger.info(f"🧹 Réservation orpheline nettoyée: {res_id} (âge: {age_hours:.1f}h)")
                                    cleaned_count += 1
                        except Exception as e:
                            logger.error(f"Erreur lors du nettoyage de la réservation {res_id}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"✅ {cleaned_count} réservations orphelines nettoyées")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du nettoyage des réservations orphelines: {str(e)}")
    
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
            asset,
            allocation_percent,
            current_value,
            used_value,
            available_value,
            active_cycles
        FROM 
            capital_pockets
        ORDER BY 
            asset,
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
                asset=row['asset'],
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
                  expiration=30)

        # Notifier les consommateurs (Coordinator) qu’un nouvel état est dispo
        redis.publish(channel, json.dumps({"ts": int(time.time())}))
        logger.debug("📡 Snapshot poches publié dans Redis")
        
        # Publier sur Kafka pour une meilleure intégration
        try:
            from shared.src.kafka_client import KafkaManager
            kafka = KafkaManager()
            
            event = {
                "type": "pocket.updated",
                "timestamp": int(time.time()),
                "pockets": [p.dict() for p in pockets]
            }
            
            kafka.send_message("portfolio.pockets", json.dumps(event))
            logger.debug("📨 Événement pocket.updated publié sur Kafka")
            
        except Exception as e:
            logger.warning(f"⚠️ Impossible de publier sur Kafka: {e}")

    def update_pockets_allocation(self, total_value: float) -> bool:
        """
        Met à jour la répartition des poches en synchronisant avec les soldes réels du portfolio.
        Version multi-assets qui synchronise chaque actif individuellement.
        
        Args:
            total_value: Valeur totale du portefeuille en USDC
            
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        try:
            logger.info(f"🔄 Synchronisation des poches avec les soldes du portfolio (valeur totale: {total_value:.2f} USDC)")
            
            # Mettre en cache la valeur totale
            SharedCache.set('total_portfolio_value', total_value)
            
            # Récupérer les derniers soldes depuis portfolio_balances
            query = """
            SELECT asset, free, locked, total
            FROM portfolio_balances
            WHERE timestamp = (SELECT MAX(timestamp) FROM portfolio_balances)
            """
            
            balances = self.db.execute_query(query, fetch_all=True)
            
            if not balances:
                logger.error("❌ Aucun solde trouvé dans portfolio_balances")
                return False
            
            # Synchroniser chaque actif
            for balance in balances:
                asset = balance['asset']
                total_balance = float(balance['total'])  # Utiliser le total (free + locked)
                
                # Répartir selon les pourcentages configurés
                # 80% active, 10% buffer, 10% safety
                allocations = {
                    'active': total_balance * 0.80,
                    'buffer': total_balance * 0.10,
                    'safety': total_balance * 0.10
                }
                
                # Mettre à jour chaque poche pour cet actif
                for pocket_type, allocated_amount in allocations.items():
                    update_query = """
                    UPDATE capital_pockets
                    SET 
                        current_value = %s,
                        available_value = %s - used_value,
                        updated_at = NOW()
                    WHERE 
                        pocket_type = %s 
                        AND asset = %s
                    """
                    
                    self.db.execute_query(
                        update_query, 
                        (allocated_amount, allocated_amount, pocket_type, asset),
                        commit=True
                    )
                
                logger.info(f"✅ Poches synchronisées pour {asset}: {total_balance:.8f} réparti (80/10/10)")
            
            # Invalider le cache
            SharedCache.clear('pockets')
            
            # Publier le nouvel état dans Redis
            self._publish_pockets_state()
            
            logger.info("✅ Synchronisation des poches terminée avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la synchronisation des poches: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def reserve_funds(self, pocket_type: str, amount: float, cycle_id: str, asset: str = "USDC") -> bool:
        """
        Réserve des fonds dans une poche pour un cycle de trading.
        Utilise Redis WATCH/MULTI/EXEC pour garantir l'atomicité.
        
        Args:
            pocket_type: Type de poche ('active', 'buffer', 'safety')
            amount: Montant à réserver
            cycle_id: ID du cycle de trading
            asset: Actif de la poche (USDC, BTC, ETH, etc.)
            
        Returns:
            True si la réservation a réussi, False sinon
            
        Raises:
            InsufficientFundsError: Si la poche n'a pas assez de fonds
            PocketNotFoundError: Si la poche n'existe pas
        """
        if amount <= 0:
            logger.warning(f"⚠️ Montant invalide: {amount}")
            return False
        
        # Générer un ID unique pour cette réservation
        import uuid
        reservation_id = f"res_{uuid.uuid4().hex[:16]}"
        
        # Utiliser Redis pour gérer la réservation de manière atomique
        redis = RedisClient()
        redis_key = f"pocket:{pocket_type}:{asset}:reservations"
        
        # Nombre maximal de tentatives en cas de conflit
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Verrouiller la ligne pour éviter les conditions de course
                query = """
                SELECT 
                    pocket_type,
                    asset,
                    available_value,
                    active_cycles
                FROM 
                    capital_pockets
                WHERE 
                    pocket_type = %s AND asset = %s
                FOR UPDATE
                """
                
                # Démarrer une transaction explicite
                self.db.execute_query("BEGIN")
                
                result = self.db.execute_query(query, (pocket_type, asset), fetch_one=True, commit=False)
                
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
                
                # Enregistrer la réservation dans Redis avec TTL de 24h
                reservation_data = {
                    "reservation_id": reservation_id,
                    "cycle_id": cycle_id,
                    "amount": amount,
                    "timestamp": int(time.time()),
                    "pocket_type": pocket_type,
                    "asset": asset
                }
                
                # Utiliser une transaction Redis pour garantir l'atomicité
                pipe = redis.pipeline()
                pipe.hset(redis_key, reservation_id, json.dumps(reservation_data))
                pipe.expire(redis_key, 86400)  # TTL de 24h
                pipe.execute()
                
                # Mettre à jour la poche
                update_query = """
                UPDATE capital_pockets
                SET 
                    used_value = used_value + %s,
                    available_value = available_value - %s,
                    active_cycles = active_cycles + 1,
                    updated_at = NOW()
                WHERE 
                    pocket_type = %s AND asset = %s
                """
                
                update_result = self.db.execute_query(update_query, (amount, amount, pocket_type, asset), commit=False)
                
                if update_result is None:
                    # Annuler la transaction et nettoyer Redis
                    self.db.execute_query("ROLLBACK")
                    redis.hdel(redis_key, reservation_id)
                    logger.error(f"❌ Échec de la mise à jour de la poche {pocket_type}")
                    return False
                
                # Enregistrer l'opération dans la table des transactions de poches (journal)
                journal_query = """
                INSERT INTO pocket_transactions
                (pocket_type, asset, transaction_type, amount, cycle_id, reservation_id, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT DO NOTHING
                """
                
                try:
                    self.db.execute_query(
                        journal_query, 
                        (pocket_type, asset, 'reserve', amount, cycle_id, reservation_id),
                        commit=False
                    )
                except Exception as e:
                    # Si la table n'existe pas, on l'ignore
                    logger.debug(f"Note: Table pocket_transactions peut ne pas exister: {str(e)}")
                
                # Valider la transaction
                self.db.execute_query("COMMIT")
                
                # Invalider le cache
                SharedCache.clear('pockets')
                
                # Publier l'événement de réservation dans Redis
                self._publish_pockets_state()
                
                logger.info(f"✅ {amount:.8f} {asset} réservés dans la poche {pocket_type} pour le cycle {cycle_id} (ID: {reservation_id})")
                
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
                
                # Si c'est une erreur de concurrence, réessayer
                if "could not serialize access" in str(e).lower():
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"🔄 Tentative {retry_count}/{max_retries} après conflit de concurrence")
                        time.sleep(0.1 * retry_count)  # Backoff exponentiel
                        continue
                
                return False
        
        # Si on arrive ici, toutes les tentatives ont échoué
        logger.error(f"❌ Impossible de réserver les fonds après {max_retries} tentatives")
        return False
    
    def release_funds(self, pocket_type: str, amount: float, cycle_id: str, asset: str = "USDC") -> bool:
        """
        Libère des fonds réservés dans une poche.
        Utilise Redis pour retrouver la réservation originale.
        
        Args:
            pocket_type: Type de poche ('active', 'buffer', 'safety')
            amount: Montant à libérer
            cycle_id: ID du cycle de trading
            asset: Actif de la poche (USDC, BTC, ETH, etc.)
            
        Returns:
            True si la libération a réussi, False sinon
        """
        if amount <= 0:
            logger.warning(f"⚠️ Montant invalide: {amount}")
            return False
        
        try:
            # Démarrer une transaction explicite
            self.db.execute_query("BEGIN")
            
            # ÉTAPE 1: Vérifier dans Redis si une réservation existe pour ce cycle
            redis = RedisClient()
            redis_key = f"pocket:{pocket_type}:{asset}:reservations"
            
            # Rechercher la réservation dans Redis
            reservation_found = None
            all_reservations = redis.hgetall(redis_key)
            
            for res_id, res_data in all_reservations.items():
                try:
                    reservation = json.loads(res_data)
                    if reservation.get('cycle_id') == cycle_id:
                        reservation_found = reservation
                        break
                except:
                    continue
            
            # Si pas trouvé dans Redis, vérifier dans la DB
            journal_check = """
            SELECT SUM(amount) AS reserved, MAX(reservation_id) as reservation_id
            FROM pocket_transactions
            WHERE transaction_type = 'reserve' AND cycle_id = %s
            """
            reserve_result = self.db.execute_query(journal_check, (cycle_id,), fetch_one=True, commit=False)
            
            reserved_amount = 0.0
            reservation_id = None
            
            if reservation_found:
                reserved_amount = reservation_found['amount']
                reservation_id = reservation_found['reservation_id']
                logger.debug(f"✅ Réservation trouvée dans Redis: {reservation_id}")
            elif reserve_result and reserve_result['reserved']:
                reserved_amount = float(reserve_result['reserved'])
                reservation_id = reserve_result.get('reservation_id')
                logger.debug(f"✅ Réservation trouvée dans DB: {reservation_id}")
            
            if reserved_amount <= 0:
                logger.warning(f"⛔ Impossible de libérer le cycle {cycle_id}, aucune réservation détectée")
                self.db.execute_query("ROLLBACK")
                return False
            
            # Vérifier les montants déjà libérés pour ce cycle
            release_check = """
            SELECT COALESCE(SUM(amount), 0) AS released
            FROM pocket_transactions
            WHERE transaction_type = 'release' AND cycle_id = %s
            """
            release_result = self.db.execute_query(release_check, (cycle_id,), fetch_one=True, commit=False)
            
            already_released = 0.0
            if release_result and release_result['released']:
                already_released = float(release_result['released'])
            
            remaining_to_release = reserved_amount - already_released
            
            if amount > remaining_to_release:
                logger.warning(f"⚠️ Tentative de libérer {amount}, mais seulement {remaining_to_release} reste à libérer pour le cycle {cycle_id}")
                if remaining_to_release <= 0:
                    logger.warning(f"⛔ Cycle {cycle_id} déjà entièrement libéré")
                    self.db.execute_query("ROLLBACK")
                    return False
                # Ajuster le montant
                amount = remaining_to_release
            
            # ÉTAPE 2: Verrouiller la ligne pour éviter les conditions de course
            query = """
            SELECT 
                pocket_type,
                asset,
                used_value,
                active_cycles
            FROM 
                capital_pockets
            WHERE 
                pocket_type = %s AND asset = %s
            FOR UPDATE
            """
            
            result = self.db.execute_query(query, (pocket_type, asset), fetch_one=True, commit=False)
            
            if not result:
                # Annuler la transaction
                self.db.execute_query("ROLLBACK")
                raise PocketNotFoundError(f"Poche non trouvée: {pocket_type}")
            
            used_value = float(result['used_value'])
            active_cycles = int(result['active_cycles'])
            
            # Vérifier si le montant à libérer n'est pas trop grand par rapport à la poche
            if amount > used_value:
                logger.warning(f"⚠️ Tentative de libérer plus que ce qui est utilisé dans la poche: {amount} > {used_value}")
                # Limiter le montant à la valeur utilisée
                amount = min(amount, used_value)
            
            # Mettre à jour la poche
            query = """
            UPDATE capital_pockets
            SET 
                used_value = GREATEST(0, used_value - %s),
                available_value = available_value + %s,
                active_cycles = GREATEST(0, active_cycles - 1),
                updated_at = NOW()
            WHERE 
                pocket_type = %s AND asset = %s
            """
            
            result = self.db.execute_query(query, (amount, amount, pocket_type, asset), commit=False)
            
            if result is None:
                # Annuler la transaction
                self.db.execute_query("ROLLBACK")
                logger.error(f"❌ Échec de la libération des fonds de la poche {pocket_type}")
                return False
            
            # Enregistrer l'opération dans la table des transactions de poches (journal)
            journal_query = """
            INSERT INTO pocket_transactions
            (pocket_type, asset, transaction_type, amount, cycle_id, reservation_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT DO NOTHING
            """
            
            try:
                self.db.execute_query(
                    journal_query, 
                    (pocket_type, asset, 'release', amount, cycle_id, reservation_id),
                    commit=False
                )
            except Exception as e:
                # Si la table n'existe pas, on l'ignore
                logger.debug(f"Note: Table pocket_transactions peut ne pas exister: {str(e)}")
            
            # Valider la transaction
            self.db.execute_query("COMMIT")
            
            # Supprimer la réservation de Redis si elle existe
            if reservation_found and reservation_id:
                redis.hdel(redis_key, reservation_id)
                logger.debug(f"🗑️ Réservation {reservation_id} supprimée de Redis")
            
            # Invalider le cache
            SharedCache.clear('pockets')
            
            # Publier l'événement de libération dans Redis
            self._publish_pockets_state()
            
            logger.info(f"✅ {amount:.8f} {asset} libérés dans la poche {pocket_type} pour le cycle {cycle_id}")
            
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
        Recalcule les valeurs utilisées et disponibles pour chaque actif.

        Returns:
            True si la synchronisation a réussi, False sinon
        """
        try:
            # Récupérer les valeurs utilisées par poche ET par actif
            query = """
            WITH trade_values AS (
                SELECT 
                    tc.pocket,
                    tc.symbol,
                    CASE 
                        WHEN tc.symbol LIKE '%USDC' THEN 'USDC'
                        WHEN tc.symbol LIKE '%BTC' THEN 'BTC'
                        WHEN tc.symbol LIKE '%ETH' THEN 'ETH'
                        WHEN tc.symbol LIKE '%BNB' THEN 'BNB'
                        ELSE 'USDC'
                    END as asset,
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
                    tc.pocket, tc.symbol
            )
            SELECT 
                cp.pocket_type,
                cp.asset,
                cp.current_value,
                COALESCE(SUM(tv.used_value), 0) as calculated_used_value
            FROM 
                capital_pockets cp
            LEFT JOIN 
                trade_values tv ON cp.pocket_type = tv.pocket AND cp.asset = tv.asset
            GROUP BY
                cp.pocket_type, cp.asset, cp.current_value
            """

            result = self.db.execute_query(query, fetch_all=True)

            if not result:
                logger.warning("⚠️ Aucune poche trouvée pour la synchronisation")
                return False

            self.db.execute_query("BEGIN")
            success = True

            for row in result:
                pocket_type = row['pocket_type']
                asset = row['asset']
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
                    AND asset = %s
                """

                update_result = self.db.execute_query(
                    update_query, 
                    (used_value, available_value, pocket_type, asset), 
                    commit=False
                )

                if update_result is None:
                    logger.error(f"❌ Échec de la mise à jour de la poche {pocket_type} pour {asset}")
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