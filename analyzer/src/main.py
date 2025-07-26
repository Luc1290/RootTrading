"""
Point d'entrée principal pour le microservice Analyzer.
Charge les données depuis la DB, exécute les stratégies et publie les signaux.
"""

import asyncio
import logging
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import redis.asyncio as redis

# Ajouter les répertoires nécessaires au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))  # Pour shared
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))     # Pour strategies
sys.path.append(os.path.dirname(__file__))  # Pour les modules src locaux

from strategy_loader import StrategyLoader
from multiproc_manager import MultiProcessManager
from redis_subscriber import RedisPublisher

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalyzerService:
    """Service principal de l'analyzer."""
    
    def __init__(self):
        self.strategy_loader = StrategyLoader()
        self.multiproc_manager = MultiProcessManager()
        self.redis_publisher = RedisPublisher(redis_url='redis://redis:6379')
        
        # Configuration base de données
        self.db_config = {
            'host': os.getenv('DB_HOST', 'db'),  # Nom du service Docker
            'port': os.getenv('DB_PORT', 5432),
            'database': os.getenv('DB_NAME', 'trading'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        }
        
        # Configuration Redis
        self.redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')  # Nom du service Docker
        
        # Symboles et timeframes (seront récupérés depuis la DB au démarrage)
        self.symbols = []
        self.timeframes = []
        
        # Intervalle d'analyse
        self.analysis_interval = int(os.getenv('ANALYSIS_INTERVAL', 60))  # secondes
        
    async def connect_db(self):
        """Établit la connexion à la base de données."""
        try:
            self.db_connection = psycopg2.connect(**self.db_config)
            logger.info("Connexion à la base de données établie")
        except Exception as e:
            logger.error(f"Erreur connexion DB: {e}")
            raise
            
    async def connect_redis(self):
        """Établit la connexion Redis."""
        try:
            await self.redis_publisher.connect()
            logger.info("Connexion Redis établie")
        except Exception as e:
            logger.error(f"Erreur connexion Redis: {e}")
            raise
            
    def load_available_symbols_and_timeframes(self):
        """Récupère tous les symboles et timeframes disponibles depuis la DB."""
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Récupérer tous les symboles disponibles
                cursor.execute("SELECT DISTINCT symbol FROM market_data ORDER BY symbol")
                symbols_result = cursor.fetchall()
                self.symbols = [row['symbol'] for row in symbols_result]
                
                # Récupérer tous les timeframes disponibles
                cursor.execute("SELECT DISTINCT timeframe FROM market_data ORDER BY timeframe")
                timeframes_result = cursor.fetchall()
                self.timeframes = [row['timeframe'] for row in timeframes_result]
                
                logger.info(f"Symboles chargés ({len(self.symbols)}): {', '.join(self.symbols)}")
                logger.info(f"Timeframes chargés ({len(self.timeframes)}): {', '.join(self.timeframes)}")
                
        except Exception as e:
            logger.error(f"Erreur chargement symboles/timeframes: {e}")
            # Fallback vers valeurs par défaut
            self.symbols = ['BTCUSDC', 'ETHUSDC', 'SOLUSDC']
            self.timeframes = ['1m', '3m', '5m', '15m']
            logger.warning(f"Utilisation des valeurs par défaut: {self.symbols} / {self.timeframes}")
            
    def fetch_latest_data(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Récupère les dernières données d'analyse pour un symbole et timeframe.
        
        Args:
            symbol: Symbole à analyser
            timeframe: Timeframe à analyser
            
        Returns:
            Dict contenant les données OHLCV et tous les indicateurs
        """
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Récupération des données d'analyse les plus récentes
                cursor.execute("""
                    SELECT * FROM analyzer_data 
                    WHERE symbol = %s AND timeframe = %s 
                    ORDER BY time DESC 
                    LIMIT 1
                """, (symbol, timeframe))
                
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"Aucune donnée trouvée pour {symbol} {timeframe}")
                    return None
                    
                # Récupération des données OHLCV pour le contexte
                cursor.execute("""
                    SELECT open, high, low, close, volume, quote_asset_volume
                    FROM market_data 
                    WHERE symbol = %s AND timeframe = %s 
                    ORDER BY time DESC 
                    LIMIT 20
                """, (symbol, timeframe))
                
                ohlcv_rows = cursor.fetchall()
                
                # Construction du dictionnaire de données
                data = {
                    'open': [float(r['open']) for r in reversed(ohlcv_rows)],
                    'high': [float(r['high']) for r in reversed(ohlcv_rows)],
                    'low': [float(r['low']) for r in reversed(ohlcv_rows)],
                    'close': [float(r['close']) for r in reversed(ohlcv_rows)],
                    'volume': [float(r['volume']) for r in reversed(ohlcv_rows)],
                    'quote_volume': [float(r['quote_asset_volume']) for r in reversed(ohlcv_rows)]
                }
                
                # Conversion des indicateurs en dictionnaire avec conversion robuste
                indicators = {}
                for key, value in row.items():
                    if key not in ['time', 'symbol', 'timeframe', 'analysis_timestamp', 'analyzer_version']:
                        # Conversion ultra-robuste des valeurs
                        try:
                            if value is None or value == '':
                                # Valeur nulle ou vide
                                indicators[key] = None
                            elif isinstance(value, (int, float)):
                                # Déjà numérique
                                indicators[key] = float(value)
                            elif isinstance(value, str):
                                # Chaîne de caractères
                                value_stripped = value.strip()
                                if value_stripped == '' or value_stripped.lower() in ['null', 'none', 'nan']:
                                    indicators[key] = None
                                elif value_stripped.lower() in ['true', 'false']:
                                    indicators[key] = value_stripped.lower() == 'true'
                                else:
                                    # Essayer de convertir en float
                                    try:
                                        indicators[key] = float(value_stripped)
                                    except ValueError:
                                        # Si ça échoue, garder comme string (pour les valeurs comme "bullish", "bearish", etc.)
                                        indicators[key] = value_stripped
                            else:
                                # Autres types (bool, etc.)
                                indicators[key] = value
                        except Exception as e:
                            logger.warning(f"Erreur conversion indicateur {key}={value}: {e}")
                            indicators[key] = value
                            
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'timestamp': row['time'],
                    'data': data,
                    'indicators': indicators
                }
                
        except Exception as e:
            logger.error(f"Erreur récupération données {symbol} {timeframe}: {e}")
            return None
            
    async def analyze_symbol_timeframe(self, symbol: str, timeframe: str):
        """
        Analyse un symbole sur un timeframe donné avec toutes les stratégies.
        
        Args:
            symbol: Symbole à analyser
            timeframe: Timeframe à analyser
        """
        logger.info(f"Analyse de {symbol} {timeframe}")
        
        # Récupération des données
        market_data = self.fetch_latest_data(symbol, timeframe)
        if not market_data:
            return
            
        # Exécution des stratégies
        strategies = self.strategy_loader.get_all_strategies()
        signals = []
        
        for strategy_name, strategy_class in strategies.items():
            try:
                # Instanciation de la stratégie
                strategy = strategy_class(
                    symbol=symbol,
                    data=market_data['data'],
                    indicators=market_data['indicators']
                )
                
                # Génération du signal
                signal = strategy.generate_signal()
                
                if signal['side']:  # Si un signal est généré
                    signal_data = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'strategy': strategy_name,
                        'timestamp': datetime.utcnow().isoformat(),
                        'side': signal['side'],
                        'confidence': signal['confidence'],
                        'strength': signal['strength'],
                        'reason': signal['reason'],
                        'metadata': signal['metadata']
                    }
                    signals.append(signal_data)
                    
                    logger.info(f"Signal généré: {strategy_name} {symbol} {timeframe} "
                              f"{signal['side']} confidence={signal['confidence']:.2f}")
                    
            except Exception as e:
                logger.error(f"Erreur stratégie {strategy_name} pour {symbol} {timeframe}: {e}")
                
        # Publication des signaux
        if signals:
            await self.redis_publisher.publish_signals(signals)
            
    async def run_analysis_cycle(self):
        """Exécute un cycle d'analyse complet pour tous les symboles et timeframes."""
        logger.info("Début du cycle d'analyse")
        
        tasks = []
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                task = self.analyze_symbol_timeframe(symbol, timeframe)
                tasks.append(task)
                
        # Exécution en parallèle
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Cycle d'analyse terminé")
        
    async def start(self):
        """Démarre le service analyzer."""
        logger.info("Démarrage du service Analyzer")
        
        # Connexions
        await self.connect_db()
        await self.connect_redis()
        
        # Chargement des symboles et timeframes depuis la DB
        self.load_available_symbols_and_timeframes()
        
        # Chargement des stratégies
        self.strategy_loader.load_strategies()
        logger.info(f"Stratégies chargées: {list(self.strategy_loader.get_all_strategies().keys())}")
        
        # Boucle principale
        try:
            while True:
                start_time = datetime.utcnow()
                
                await self.run_analysis_cycle()
                
                # Calcul du temps d'attente
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                wait_time = max(0, self.analysis_interval - elapsed)
                
                if wait_time > 0:
                    logger.info(f"Attente de {wait_time:.1f}s avant le prochain cycle")
                    await asyncio.sleep(wait_time)
                    
        except KeyboardInterrupt:
            logger.info("Arrêt du service demandé")
        except Exception as e:
            logger.error(f"Erreur dans la boucle principale: {e}")
            raise
        finally:
            await self.cleanup()
            
    async def cleanup(self):
        """Nettoie les ressources."""
        if hasattr(self, 'db_connection'):
            self.db_connection.close()
            logger.info("Connexion DB fermée")
            
        if self.redis_publisher:
            await self.redis_publisher.disconnect()
            logger.info("Connexion Redis fermée")


async def main():
    """Fonction principale."""
    service = AnalyzerService()
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())