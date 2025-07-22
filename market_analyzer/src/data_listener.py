"""
Market Data Listener
Écoute les nouvelles données dans market_data et déclenche les calculs.
Système trigger-based pour traitement temps réel.
"""

import logging
import asyncio
import asyncpg
import json
from datetime import datetime
from typing import Dict, Any, Optional
import sys
import os

# Ajouter les chemins pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import get_db_config
from .indicator_processor import IndicatorProcessor


logger = logging.getLogger(__name__)

class DataListener:
    """
    Écoute les changements dans market_data et déclenche les calculs automatiquement.
    Utilise PostgreSQL LISTEN/NOTIFY pour traitement temps réel.
    """
    
    def __init__(self):
        self.db_pool = None
        self.listen_conn = None
        self.running = False
        self.processed_count = 0
        self.indicator_processor = IndicatorProcessor()
        
        logger.info("📡 DataListener initialisé")

    async def initialize(self):
        """Initialise les connexions et le moteur de calcul."""
        try:
            db_config = get_db_config()
            
            # Pool principal pour les requêtes
            self.db_pool = await asyncpg.create_pool(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                min_size=2,
                max_size=10
            )
            
            # Connexion dédiée pour LISTEN
            self.listen_conn = await asyncpg.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password']
            )
            
            # Initialiser le processeur d'indicateurs
            await self.indicator_processor.initialize()
            
            # Créer le trigger si nécessaire
            await self._setup_database_trigger()
            
            logger.info("✅ DataListener connecté et prêt")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation DataListener: {e}")
            raise

    async def _setup_database_trigger(self):
        """Configure le trigger PostgreSQL pour notifier les nouveaux inserts."""
        
        # Fonction trigger qui envoie une notification
        trigger_function = """
            CREATE OR REPLACE FUNCTION notify_market_data_change()
            RETURNS TRIGGER AS $$
            BEGIN
                -- Envoyer notification avec les détails de la nouvelle donnée
                PERFORM pg_notify(
                    'market_data_change',
                    json_build_object(
                        'symbol', NEW.symbol,
                        'timeframe', NEW.timeframe,
                        'time', extract(epoch from NEW.time)::bigint,
                        'action', TG_OP
                    )::text
                );
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """
        
        # Trigger qui appelle la fonction sur INSERT/UPDATE
        trigger_definition = """
            DROP TRIGGER IF EXISTS market_data_change_trigger ON market_data;
            CREATE TRIGGER market_data_change_trigger
                AFTER INSERT OR UPDATE ON market_data
                FOR EACH ROW
                EXECUTE FUNCTION notify_market_data_change();
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(trigger_function)
                await conn.execute(trigger_definition)
                
            logger.info("✅ Trigger PostgreSQL configuré pour market_data")
            
        except Exception as e:
            logger.error(f"❌ Erreur configuration trigger: {e}")
            raise

    async def start_listening(self):
        """Démarre l'écoute des notifications."""
        self.running = True
        logger.info("🎧 Démarrage de l'écoute des changements market_data...")
        
        try:
            # S'abonner aux notifications
            await self.listen_conn.add_listener('market_data_change', self._handle_notification)
            
            logger.info("✅ Écoute active - en attente de nouvelles données...")
            
            # Boucle principale d'écoute
            while self.running:
                try:
                    # Attendre les notifications (bloquant)
                    await asyncio.sleep(0.1)  # Petite pause pour éviter de surcharger
                    
                except asyncio.CancelledError:
                    logger.info("🛑 Écoute interrompue")
                    break
                except Exception as e:
                    logger.error(f"❌ Erreur dans la boucle d'écoute: {e}")
                    await asyncio.sleep(1)  # Attendre avant de retry
                    
        except Exception as e:
            logger.error(f"❌ Erreur critique dans l'écoute: {e}")
        finally:
            await self._cleanup()

    async def _handle_notification(self, connection, pid, channel, payload):
        """
        Gestionnaire appelé quand une notification est reçue.
        
        Args:
            payload: JSON avec symbol, timeframe, time, action
        """
        try:
            # Parser la notification
            data = json.loads(payload)
            symbol = data['symbol']
            timeframe = data['timeframe']
            timestamp = datetime.fromtimestamp(data['time'])
            action = data['action']
            
            logger.debug(f"📬 Notification reçue: {action} {symbol} {timeframe} @ {timestamp}")
            
            # Ne traiter que les INSERT (nouvelles données)
            if action == 'INSERT':
                await self._process_new_data(symbol, timeframe, timestamp)
                self.processed_count += 1
                
                if self.processed_count % 10 == 0:
                    logger.info(f"📊 {self.processed_count} analyses complétées")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement notification: {e}")
            logger.error(f"Payload: {payload}")

    async def _process_new_data(self, symbol: str, timeframe: str, timestamp: datetime):
        """
        Traite une nouvelle donnée en lançant les calculs.
        
        Args:
            symbol: Symbole (ex: BTCUSDC)
            timeframe: Timeframe (ex: 1m, 5m)
            timestamp: Timestamp de la nouvelle donnée
        """
        try:
            # Vérifier si on a déjà analysé cette donnée
            if await self._is_already_analyzed(symbol, timeframe, timestamp):
                logger.debug(f"⏭️ Déjà analysé: {symbol} {timeframe} @ {timestamp}")
                return
            
            # Appeler le processeur d'indicateurs
            await self.indicator_processor.process_new_data(symbol, timeframe, timestamp)
            
            logger.debug(f"✅ Traitement terminé: {symbol} {timeframe}")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement données: {e}")

    async def _is_already_analyzed(self, symbol: str, timeframe: str, timestamp: datetime) -> bool:
        """Vérifie si cette donnée a déjà été analysée."""
        
        query = """
            SELECT 1 FROM analyzer_data 
            WHERE symbol = $1 AND timeframe = $2 AND time = $3
            LIMIT 1
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(query, symbol, timeframe, timestamp)
                return result is not None
                
        except Exception as e:
            logger.error(f"❌ Erreur vérification analyse: {e}")
            return False

    async def process_historical_batch(self, symbol: str = None, timeframe: str = None, limit: int = 1000):
        """
        Traite un lot de données historiques non analysées.
        Utile pour rattraper les données manquantes.
        
        Args:
            symbol: Symbole spécifique (optionnel)
            timeframe: Timeframe spécifique (optionnel)  
            limit: Nombre maximum de données à traiter
        """
        logger.info(f"🔄 Démarrage traitement historique (limit: {limit})")
        
        # Requête pour trouver les données non analysées
        base_query = """
            SELECT md.symbol, md.timeframe, md.time
            FROM market_data md
            LEFT JOIN analyzer_data ad ON (
                md.symbol = ad.symbol AND 
                md.timeframe = ad.timeframe AND 
                md.time = ad.time
            )
            WHERE ad.time IS NULL
        """
        
        conditions = []
        params = []
        
        if symbol:
            conditions.append(f"AND md.symbol = ${len(params) + 1}")
            params.append(symbol)
            
        if timeframe:
            conditions.append(f"AND md.timeframe = ${len(params) + 1}")
            params.append(timeframe)
        
        query = base_query + " ".join(conditions) + f"""
            ORDER BY md.time DESC
            LIMIT ${len(params) + 1}
        """
        params.append(limit)
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
            logger.info(f"📊 {len(rows)} données historiques à traiter")
            
            processed = 0
            for row in rows:
                try:
                    await self.indicator_processor.process_new_data(
                        row['symbol'], 
                        row['timeframe'], 
                        row['time']
                    )
                    processed += 1
                    
                    if processed % 50 == 0:
                        logger.info(f"📈 Historique traité: {processed}/{len(rows)}")
                        
                except Exception as e:
                    logger.error(f"❌ Erreur traitement historique {row['symbol']} {row['timeframe']}: {e}")
                    continue
            
            logger.info(f"✅ Traitement historique terminé: {processed}/{len(rows)} succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement historique: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du listener."""
        
        # Compter les données market_data vs analyzer_data
        stats_query = """
            SELECT 
                (SELECT COUNT(*) FROM market_data) as total_market_data,
                (SELECT COUNT(*) FROM analyzer_data) as total_analyzer_data,
                (SELECT COUNT(DISTINCT symbol) FROM market_data) as symbols_count,
                (SELECT COUNT(DISTINCT timeframe) FROM market_data) as timeframes_count
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(stats_query)
                
                coverage_percent = 0
                if row['total_market_data'] > 0:
                    coverage_percent = (row['total_analyzer_data'] / row['total_market_data']) * 100
                
                return {
                    'running': self.running,
                    'processed_count': self.processed_count,
                    'total_market_data': row['total_market_data'],
                    'total_analyzer_data': row['total_analyzer_data'],
                    'coverage_percent': round(coverage_percent, 2),
                    'symbols_count': row['symbols_count'],
                    'timeframes_count': row['timeframes_count'],
                    'missing_analyses': row['total_market_data'] - row['total_analyzer_data']
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {
                'running': self.running,
                'processed_count': self.processed_count,
                'error': str(e)
            }

    async def _cleanup(self):
        """Nettoie les ressources."""
        try:
            if self.listen_conn:
                await self.listen_conn.remove_listener('market_data_change', self._handle_notification)
                await self.listen_conn.close()
                
            await self.indicator_processor.close()
            
            if self.db_pool:
                await self.db_pool.close()
                
            logger.info("🧹 DataListener nettoyé")
            
        except Exception as e:
            logger.error(f"❌ Erreur nettoyage: {e}")

    async def stop(self):
        """Arrête l'écoute."""
        logger.info("🛑 Arrêt du DataListener...")
        self.running = False
        await self._cleanup()