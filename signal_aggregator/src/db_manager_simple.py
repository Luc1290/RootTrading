import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional
import logging
from shared.config import get_db_config

logger = logging.getLogger(__name__)

class SimpleDatabaseManager:
    """Database manager simple et thread-safe utilisant psycopg2 synchrone"""
    
    def __init__(self):
        self.connection = None
        self._db_disabled = False
    
    def initialize(self):
        """Initialisation synchrone simple"""
        try:
            db_config = get_db_config()
            
            self.connection = psycopg2.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                cursor_factory=psycopg2.extras.RealDictCursor
            )
            
            logger.info("✅ Connexion DB synchrone Signal Aggregator initialisée")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation DB synchrone: {e}")
            self._db_disabled = True
            return False
    
    def get_enriched_market_data(
        self, 
        symbol: str, 
        interval: str = '1m', 
        limit: int = 100,
        include_indicators: bool = True
    ) -> List[Dict]:
        """Récupère les données enrichies de manière synchrone"""
        
        if self._db_disabled:
            return []
        
        if not self.connection:
            if not self.initialize():
                return []
        
        try:
            cursor = self.connection.cursor()
            
            if include_indicators:
                query = """
                    SELECT 
                        time, open, high, low, close, volume,
                        rsi_14, ema_12, ema_26, ema_50, sma_20, sma_50,
                        macd_line, macd_signal, macd_histogram,
                        bb_upper, bb_middle, bb_lower, bb_position, bb_width,
                        atr_14, momentum_10, volume_ratio, avg_volume_20,
                        enhanced, ultra_enriched
                    FROM market_data
                    WHERE symbol = %s AND enhanced = true
                    ORDER BY time DESC
                    LIMIT %s
                """
            else:
                query = """
                    SELECT time, open, high, low, close, volume
                    FROM market_data
                    WHERE symbol = %s
                    ORDER BY time DESC
                    LIMIT %s
                """
            
            cursor.execute(query, (symbol, limit))
            rows = cursor.fetchall()
            cursor.close()
            
            if not rows:
                logger.debug(f"Aucune donnée enrichie trouvée pour {symbol}")
                return []
            
            # Convertir en liste de dicts et inverser l'ordre
            result = []
            for row in reversed(rows):
                result.append(dict(row))
            
            logger.debug(f"✅ Récupéré {len(result)} enregistrements enrichis pour {symbol}")
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Gestion des erreurs de connexion
            if any(err in error_msg for err in ['connection', 'server', 'closed']):
                logger.warning(f"⚠️ Erreur connexion DB pour {symbol}, tentative de reconnexion...")
                try:
                    self.connection.close()
                except:
                    pass
                self.connection = None
                
                # Retry une fois
                if self.initialize():
                    return self.get_enriched_market_data(symbol, interval, limit, include_indicators)
            
            logger.error(f"❌ Erreur récupération données enrichies pour {symbol}: {e}")
            return []
    
    def close(self):
        """Ferme la connexion proprement"""
        if self.connection:
            try:
                self.connection.close()
                logger.info("✅ Connexion DB fermée proprement")
            except Exception as e:
                logger.warning(f"⚠️ Erreur fermeture connexion: {e}")
            finally:
                self.connection = None