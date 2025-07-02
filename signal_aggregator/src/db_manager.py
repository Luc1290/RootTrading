"""
Gestionnaire de base de données pour le Signal Aggregator.
Permet de lire les données enrichies stockées par le Gateway/Dispatcher.
"""
import logging
import asyncio
import asyncpg
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from shared.src.config import get_db_config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Gestionnaire de base de données pour accéder aux données enrichies"""
    
    def __init__(self):
        self.db_pool: Optional[asyncpg.Pool] = None
        self.db_connection: Optional[asyncpg.Connection] = None  # Connexion unique dédiée
        self.running = False
        self._db_lock = asyncio.Lock()  # Verrou pour sérialiser les accès DB
        self._db_disabled = False  # Flag pour désactiver la DB en cas de problème persistant
    
        
    def initialize_sync(self):
        """Initialise la connexion à la base de données de manière synchrone - DEPRECATED"""
        # Cette méthode ne fait plus rien - l'initialisation se fait de manière async
        logger.info("📋 Initialisation DB différée - sera effectuée lors du premier accès async")
        return True
    
    async def _async_initialize(self):
        """Partie asynchrone de l'initialisation - Connexion unique"""
        try:
            db_config = get_db_config()
            
            # Créer une connexion unique dédiée au lieu d'un pool
            self.db_connection = await asyncpg.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                command_timeout=10,
                server_settings={
                    'application_name': 'signal_aggregator_dedicated'
                }
            )
            logger.info("✅ Connexion unique base de données Signal Aggregator initialisée")
            self.running = True
            return True
        except Exception as e:
            logger.error(f"❌ Erreur initialisation connexion unique Signal Aggregator: {e}")
            self.db_connection = None
            return False
    
    async def initialize(self):
        """Version async pour compatibilité"""
        return await self._async_initialize()
    
    async def _reinitialize_connection(self):
        """Réinitialise la connexion unique en cas de problème"""
        try:
            # Fermer l'ancienne connexion si elle existe
            if self.db_connection:
                try:
                    await self.db_connection.close()
                except Exception as close_error:
                    logger.warning(f"Erreur fermeture ancienne connexion: {close_error}")
                finally:
                    self.db_connection = None
            
            # Réinitialiser l'état
            self.running = False
            
            # Attendre un peu avant de recréer
            await asyncio.sleep(0.2)
            
            # Recréer la connexion
            success = await self._async_initialize()
            if success:
                logger.info("✅ Connexion réinitialisée avec succès")
            else:
                logger.error("❌ Échec réinitialisation connexion")
                
        except Exception as e:
            logger.error(f"❌ Erreur réinitialisation connexion: {e}")
            self.running = False
            self.db_connection = None
    
    async def close(self):
        """Ferme les connexions"""
        self.running = False
        if self.db_connection:
            await self.db_connection.close()
            logger.info("🔌 Connexion base fermée (Signal Aggregator)")
        if self.db_pool:  # Compatibility
            await self.db_pool.close()
    
    async def get_enriched_market_data(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 100,
        include_indicators: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Récupère les données de marché enrichies depuis la base de données.
        
        Args:
            symbol: Symbole à récupérer (ex: SOLUSDC)
            interval: Timeframe (1m, 5m, 15m, etc.)
            limit: Nombre de points à récupérer
            include_indicators: Inclure les indicateurs techniques
            
        Returns:
            Liste des données OHLCV + indicateurs techniques
        """
        # Vérification préliminaire : si la DB est désactivée, retourner immédiatement
        if self._db_disabled:
            logger.debug(f"📴 DB désactivée pour {symbol}, utilisation Redis directement")
            return []
        
        try:
            # Vérifier si on a une boucle d'événements active
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                logger.warning("⚠️ Pas de boucle d'événements active, désactivation DB pour cette session")
                self._db_disabled = True
                return []
            
            # Initialiser automatiquement si pas encore fait
            if not self.db_connection or not self.running:
                logger.info("🔄 Initialisation automatique de la base de données...")
                init_success = await self._async_initialize()
                if not init_success:
                    logger.warning("Base de données non disponible, désactivation pour cette session")
                    self._db_disabled = True
                    return []
        except Exception as e:
            logger.warning(f"⚠️ Erreur vérification préliminaire DB, désactivation: {e}")
            self._db_disabled = True
            return []
        
        # Utiliser un verrou pour sérialiser les accès et éviter la concurrence
        async with self._db_lock:
            try:
                # Tentative unique avec le verrou - plus de retry nécessaire pour concurrence
                return await self._fetch_enriched_data_attempt(symbol, interval, limit, include_indicators)
            except Exception as e:
                error_msg = str(e).lower()
                
                # Pour les erreurs de connexion, on peut encore faire un retry
                connection_errors = [
                    "connection was closed",
                    "connection does not exist", 
                    "connection lost",
                    "connection terminated"
                ]
                
                if any(err in error_msg for err in connection_errors):
                    logger.warning(f"⚠️ Erreur connexion DB pour {symbol}, tentative de réinitialisation...")
                    try:
                        await self._reinitialize_connection()
                        # Un seul retry après réinitialisation
                        return await self._fetch_enriched_data_attempt(symbol, interval, limit, include_indicators)
                    except Exception as retry_error:
                        logger.error(f"❌ Échec retry après réinitialisation pour {symbol}: {retry_error}")
                
                logger.error(f"❌ Erreur récupération données enrichies pour {symbol}: {e}")
                return []
    
    async def _fetch_enriched_data_attempt(
        self,
        symbol: str,
        interval: str,
        limit: int,
        include_indicators: bool
    ) -> List[Dict[str, Any]]:
        """Tentative unique de récupération des données enrichies"""
        try:
            # Construction de la requête selon les besoins
            if include_indicators:
                query = """
                    SELECT 
                        time,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        -- Indicateurs techniques enrichis
                        rsi_14,
                        ema_12,
                        ema_26,
                        ema_50,
                        sma_20,
                        sma_50,
                        macd_line,
                        macd_signal,
                        macd_histogram,
                        bb_upper,
                        bb_middle,
                        bb_lower,
                        bb_position,
                        bb_width,
                        atr_14,
                        momentum_10,
                        volume_ratio,
                        avg_volume_20,
                        enhanced,
                        ultra_enriched
                    FROM market_data
                    WHERE symbol = $1
                        AND enhanced = true
                    ORDER BY time DESC
                    LIMIT $2
                """
                params = [symbol, limit]
            else:
                query = """
                    SELECT 
                        time,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM market_data
                    WHERE symbol = $1
                    ORDER BY time DESC
                    LIMIT $2
                """
                params = [symbol, limit]
            
            # Utilisation directe de la connexion unique dédiée
            try:
                rows = await self.db_connection.fetch(query, *params)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout requête DB pour {symbol}")
                return []
            
            if not rows:
                logger.debug(f"Aucune donnée enrichie trouvée pour {symbol}")
                return []
            
            # Convertir en format dict et inverser l'ordre (chronologique)
            data = []
            for row in reversed(rows):  # Inverser pour ordre chronologique
                record = {
                        'timestamp': row['time'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume'])
                }
                
                # Ajouter les indicateurs si demandés
                if include_indicators:
                    # RSI
                    if row['rsi_14'] is not None:
                        record['rsi_14'] = float(row['rsi_14'])
                    
                    # EMAs
                    for ema_period in [12, 26, 50]:
                        ema_key = f'ema_{ema_period}'
                        if row[ema_key] is not None:
                            record[ema_key] = float(row[ema_key])
                    
                    # SMAs
                    for sma_period in [20, 50]:
                        sma_key = f'sma_{sma_period}'
                        if row[sma_key] is not None:
                            record[sma_key] = float(row[sma_key])
                    
                    # MACD
                    for macd_field in ['macd_line', 'macd_signal', 'macd_histogram']:
                        if row[macd_field] is not None:
                            record[macd_field] = float(row[macd_field])
                    
                    # Bollinger Bands
                    for bb_field in ['bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_width']:
                        if row[bb_field] is not None:
                            record[bb_field] = float(row[bb_field])
                    
                    # Autres indicateurs
                    for indicator in ['atr_14', 'momentum_10', 'volume_ratio', 'avg_volume_20']:
                        if row[indicator] is not None:
                            record[indicator] = float(row[indicator])
                    
                    # Métadonnées
                    record['enhanced'] = bool(row['enhanced']) if row['enhanced'] is not None else False
                    record['ultra_enriched'] = bool(row['ultra_enriched']) if row['ultra_enriched'] is not None else False
                
                data.append(record)
            
            logger.debug(f"📊 Récupéré {len(data)} points enrichis pour {symbol} depuis la DB")
            return data
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération données enrichies pour {symbol}: {e}")
            return []
    
    async def get_latest_enriched_data(
        self,
        symbol: str,
        timeframe: str = "1m"
    ) -> Optional[Dict[str, Any]]:
        """
        Récupère le dernier point de données enrichi pour un symbole.
        
        Args:
            symbol: Symbole à récupérer
            timeframe: Timeframe (pour future extension)
            
        Returns:
            Dernier point de données ou None
        """
        data = await self.get_enriched_market_data(
            symbol=symbol,
            interval=timeframe,
            limit=1,
            include_indicators=True
        )
        
        return data[0] if data else None
    
    async def check_data_freshness(self, symbol: str, max_age_minutes: int = 5) -> bool:
        """
        Vérifie si les données enrichies sont récentes.
        
        Args:
            symbol: Symbole à vérifier
            max_age_minutes: Age maximum acceptable en minutes
            
        Returns:
            True si les données sont récentes
        """
        if not self.db_pool or not self.running:
            return False
        
        try:
            query = """
                SELECT time
                FROM market_data
                WHERE symbol = $1 AND enhanced = true
                ORDER BY time DESC
                LIMIT 1
            """
            
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(query, symbol)
                
                if not row:
                    return False
                
                last_update = row['time']
                age = datetime.utcnow() - last_update.replace(tzinfo=None)
                
                is_fresh = age.total_seconds() < (max_age_minutes * 60)
                
                if not is_fresh:
                    logger.warning(f"⚠️ Données enrichies {symbol} anciennes: {age.total_seconds():.0f}s")
                
                return is_fresh
                
        except Exception as e:
            logger.error(f"❌ Erreur vérification fraîcheur données {symbol}: {e}")
            return False
    
    async def get_available_symbols(self) -> List[str]:
        """
        Récupère la liste des symboles disponibles avec données enrichies.
        
        Returns:
            Liste des symboles
        """
        if not self.db_pool or not self.running:
            return []
        
        try:
            query = """
                SELECT DISTINCT symbol
                FROM market_data
                WHERE enhanced = true
                    AND time >= NOW() - INTERVAL '1 hour'
                ORDER BY symbol
            """
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query)
                symbols = [row['symbol'] for row in rows]
                
                logger.debug(f"📋 Symboles disponibles avec données enrichies: {symbols}")
                return symbols
                
        except Exception as e:
            logger.error(f"❌ Erreur récupération symboles disponibles: {e}")
            return []
    
    def get_enriched_market_data_sync(
        self,
        symbol: str,
        interval: str = "1m", 
        limit: int = 100,
        include_indicators: bool = True
    ) -> List[Dict[str, Any]]:
        """Version synchrone pour récupérer les données enrichies - DEPRECATED"""
        logger.warning("❗ get_enriched_market_data_sync deprecated - utilisez la version async")
        return []