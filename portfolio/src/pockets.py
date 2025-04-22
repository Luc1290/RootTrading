"""
Gestionnaire des poches de capital.
Gère la répartition du capital en différentes poches pour le trading.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import POCKET_CONFIG
from shared.src.schemas import PocketSummary

from portfolio.src.models import DBManager

# Configuration du logging
logger = logging.getLogger(__name__)

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
            query = """
            INSERT INTO capital_pockets
            (pocket_type, allocation_percent, current_value, used_value, available_value, active_cycles)
            VALUES (%s, %s, %s, %s, %s, 0)
            """
            
            params_list = []
            
            for pocket_type in missing_pockets:
                allocation = self.pocket_config.get(pocket_type, 0.0)
                current_value = total_value * allocation
                
                params = (
                    pocket_type,
                    allocation * 100,  # Stocker en pourcentage
                    current_value,
                    0.0,  # Aucune valeur utilisée initialement
                    current_value  # Disponible = courant
                )
                params_list.append(params)
            
            # Exécuter l'insertion
            self.db.execute_many(query, params_list)
            
            logger.info(f"✅ {len(missing_pockets)} poches créées")
    
    def _get_total_portfolio_value(self) -> Optional[float]:
        """
        Calcule la valeur totale du portefeuille.
        
        Returns:
            Valeur totale ou None en cas d'erreur
        """
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
        """
        
        result = self.db.execute_query(query, fetch_one=True)
        
        if result and result['total_value'] is not None:
            return float(result['total_value'])
        
        return None
    
    def get_pockets(self) -> List[PocketSummary]:
        """
        Récupère l'état actuel des poches.
        
        Returns:
            Liste des poches de capital
        """
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
        
        return pockets
    
    def update_pockets_allocation(self, total_value: float) -> bool:
        """
        Met à jour la répartition des poches en fonction de la valeur totale du portefeuille.
        
        Args:
            total_value: Valeur totale du portefeuille
            
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        if total_value <= 0:
            logger.warning("⚠️ Valeur totale du portefeuille invalide")
            return False
        
        # Récupérer l'état actuel des poches
        pockets = self.get_pockets()
        
        if not pockets:
            logger.warning("⚠️ Aucune poche trouvée")
            return False
        
        # Calculer les nouvelles valeurs
        updates = []
        
        for pocket in pockets:
            # Récupérer l'allocation configurée
            allocation = self.pocket_config.get(pocket.pocket_type, 0.0)
            
            # Calculer la nouvelle valeur
            new_current_value = total_value * allocation
            
            # La valeur utilisée reste inchangée
            used_value = pocket.used_value
            
            # Calculer la nouvelle valeur disponible
            new_available_value = max(0, new_current_value - used_value)
            
            updates.append({
                'pocket_type': pocket.pocket_type,
                'allocation_percent': allocation * 100,
                'current_value': new_current_value,
                'used_value': used_value,
                'available_value': new_available_value
            })
        
        # Mettre à jour les poches
        query = """
        UPDATE capital_pockets
        SET 
            allocation_percent = %(allocation_percent)s,
            current_value = %(current_value)s,
            available_value = %(available_value)s,
            updated_at = NOW()
        WHERE 
            pocket_type = %(pocket_type)s
        """
        
        success = all(
            self.db.execute_query(query, update, commit=True) is not None
            for update in updates
        )
        
        if success:
            logger.info(f"✅ Allocation des poches mise à jour (total: {total_value:.2f})")
        
        return success
    
    def reserve_funds(self, pocket_type: str, amount: float, cycle_id: str) -> bool:
        """
        Réserve des fonds dans une poche pour un cycle de trading.
        
        Args:
            pocket_type: Type de poche ('active', 'buffer', 'safety')
            amount: Montant à réserver
            cycle_id: ID du cycle de trading
            
        Returns:
            True si la réservation a réussi, False sinon
        """
        if amount <= 0:
            logger.warning(f"⚠️ Montant invalide: {amount}")
            return False
        
        # Vérifier si la poche existe et a suffisamment de fonds
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
        
        result = self.db.execute_query(query, (pocket_type,), fetch_one=True, commit=False)
        
        if not result:
            logger.warning(f"⚠️ Poche non trouvée: {pocket_type}")
            return False
        
        available_value = float(result['available_value'])
        active_cycles = int(result['active_cycles'])
        
        if available_value < amount:
            logger.warning(f"⚠️ Fonds insuffisants dans la poche {pocket_type}: {available_value} < {amount}")
            return False
        
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
        
        update_result = self.db.execute_query(update_query, (amount, amount, pocket_type), commit=True)
        
        if update_result is None:
            logger.error(f"❌ Échec de la mise à jour de la poche {pocket_type}")
            return False
        
        # Enregistrer la réservation (pourrait être étendu avec une table dédiée)
        logger.info(f"✅ {amount:.2f} réservés dans la poche {pocket_type} pour le cycle {cycle_id}")
        
        return True
    
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
        
        result = self.db.execute_query(query, (amount, amount, pocket_type), commit=True)
        
        if result is None:
            logger.error(f"❌ Échec de la libération des fonds de la poche {pocket_type}")
            return False
        
        # Enregistrer la libération
        logger.info(f"✅ {amount:.2f} libérés dans la poche {pocket_type} pour le cycle {cycle_id}")
        
        return True
    
    def get_available_funds(self, pocket_type: str) -> Optional[float]:
        """
        Récupère les fonds disponibles dans une poche.
        
        Args:
            pocket_type: Type de poche ('active', 'buffer', 'safety')
            
        Returns:
            Montant disponible ou None en cas d'erreur
        """
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
        
        logger.info("✅ Nombre de cycles actifs recalculé pour toutes les poches")
        
        return result is not None
    
    def sync_with_trades(self) -> bool:
        """
        Synchronise les poches avec les trades actifs.
        Recalcule les valeurs utilisées et disponibles.
        
        Returns:
            True si la synchronisation a réussi, False sinon
        """
        # Obtenir la valeur totale utilisée par poche
        query = """
        WITH trade_values AS (
            SELECT 
                pocket,
                SUM(quantity * entry_price) as used_value
            FROM 
                trade_cycles
            WHERE 
                status NOT IN ('completed', 'canceled', 'failed')
                AND pocket IS NOT NULL
            GROUP BY 
                pocket
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
        
        # Mettre à jour les valeurs utilisées et disponibles
        updates = []
        
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
            
            update_result = self.db.execute_query(update_query, (used_value, available_value, pocket_type), commit=True)
            
            if update_result is None:
                logger.error(f"❌ Échec de la mise à jour de la poche {pocket_type}")
                return False
        
        # Mettre à jour le nombre de cycles actifs
        self.recalculate_active_cycles()
        
        logger.info("✅ Poches synchronisées avec les trades actifs")
        
        return True
    
    def close(self) -> None:
        """
        Ferme la connexion à la base de données.
        """
        if self.db:
            self.db.close()