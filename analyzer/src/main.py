"""
Point d'entrée principal pour le microservice Analyzer.
Charge les données depuis la DB, exécute les stratégies et publie les signaux.
"""

import asyncio
import logging
import json
import os
import sys
import time
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
log_level = logging.DEBUG if os.getenv('DEBUG_LOGS', 'false').lower() == 'true' else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Logger spécialisé pour les statistiques
stats_logger = logging.getLogger('analyzer.stats')


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
        
        # Statistiques globales
        self.cycle_stats = {
            'total_analyses': 0,
            'total_signals': 0,
            'strategies_executed': 0,
            'cycle_count': 0
        }
        
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
                # Récupérer toutes les combinaisons symbol/timeframe qui ont des données récentes
                cursor.execute("""
                    SELECT DISTINCT symbol, timeframe 
                    FROM market_data 
                    WHERE time >= NOW() - INTERVAL '24 hours'
                    ORDER BY symbol, timeframe
                """)
                combinations = cursor.fetchall()
                
                # Extraire symboles et timeframes uniques
                self.symbols = sorted(list(set(row['symbol'] for row in combinations)))
                self.timeframes = sorted(list(set(row['timeframe'] for row in combinations)))
                
                # Stocker les combinaisons valides pour éviter les requêtes vides
                self.valid_combinations = [(row['symbol'], row['timeframe']) for row in combinations]
                
                logger.info(f"Symboles chargés ({len(self.symbols)}): {', '.join(self.symbols)}")
                logger.info(f"Timeframes chargés ({len(self.timeframes)}): {', '.join(self.timeframes)}")
                logger.info(f"Combinaisons valides: {len(self.valid_combinations)}")
                
        except Exception as e:
            logger.error(f"Erreur chargement symboles/timeframes: {e}")
            # Fallback vers valeurs par défaut
            self.symbols = ['BTCUSDC', 'ETHUSDC', 'SOLUSDC']
            self.timeframes = ['1m', '3m', '5m', '15m']
            self.valid_combinations = [(s, t) for s in self.symbols for t in self.timeframes]
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
                    'quote_volume': [float(r['quote_asset_volume']) for r in reversed(ohlcv_rows)],
                    # Compatibilité avec les stratégies qui cherchent 'ohlcv'
                    'ohlcv': ohlcv_rows  # Liste des rows complètes pour compatibilité
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
            
    def _log_strategy_debug(self, strategy_name: str, indicators: Dict, symbol: str, timeframe: str):
        """Log les indicateurs clés pour debug selon la stratégie."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
            
        strategy_logger = logging.getLogger(f'analyzer.{strategy_name}')
        
        try:
            # Indicateurs de base toujours intéressants
            debug_info = []
            
            # Prix et moyennes mobiles
            if 'ema_12' in indicators and indicators['ema_12'] is not None:
                debug_info.append(f"EMA12: {float(indicators['ema_12']):.4f}")
            if 'ema_26' in indicators and indicators['ema_26'] is not None:
                debug_info.append(f"EMA26: {float(indicators['ema_26']):.4f}")
                
            # Oscillateurs principaux
            if 'rsi_14' in indicators and indicators['rsi_14'] is not None:
                debug_info.append(f"RSI: {float(indicators['rsi_14']):.1f}")
            if 'williams_r' in indicators and indicators['williams_r'] is not None:
                debug_info.append(f"WilliamsR: {float(indicators['williams_r']):.1f}")
            if 'macd_line' in indicators and indicators['macd_line'] is not None:
                debug_info.append(f"MACD: {float(indicators['macd_line']):.4f}")
                
            # Indicateurs spécifiques selon la stratégie
            if 'VWAP' in strategy_name and 'vwap_10' in indicators and indicators['vwap_10'] is not None:
                debug_info.append(f"VWAP: {float(indicators['vwap_10']):.4f}")
            if 'ZScore' in strategy_name and 'bb_position' in indicators and indicators['bb_position'] is not None:
                debug_info.append(f"BB_Pos: {float(indicators['bb_position']):.3f}")
            if 'Support' in strategy_name or 'Resistance' in strategy_name:
                if 'nearest_support' in indicators and indicators['nearest_support'] is not None:
                    debug_info.append(f"Support: {float(indicators['nearest_support']):.4f}")
                if 'nearest_resistance' in indicators and indicators['nearest_resistance'] is not None:
                    debug_info.append(f"Resistance: {float(indicators['nearest_resistance']):.4f}")
                    
            # Volume et momentum
            if 'volume_ratio' in indicators and indicators['volume_ratio'] is not None:
                debug_info.append(f"Vol: {float(indicators['volume_ratio']):.1f}x")
            if 'momentum_score' in indicators and indicators['momentum_score'] is not None:
                debug_info.append(f"Mom: {float(indicators['momentum_score']):.2f}")
                
            if debug_info:
                strategy_logger.debug(f"{symbol} {timeframe} - {', '.join(debug_info[:6])}")  # Limiter à 6 indicateurs
                
        except Exception as e:
            logger.warning(f"Erreur log debug pour {strategy_name}: {e}")
            
    async def analyze_symbol_timeframe(self, symbol: str, timeframe: str):
        """
        Analyse un symbole sur un timeframe donné avec toutes les stratégies.
        
        Args:
            symbol: Symbole à analyser
            timeframe: Timeframe à analyser
        """
        start_time = time.time()
        
        # Récupération des données
        market_data = self.fetch_latest_data(symbol, timeframe)
        if not market_data:
            return
            
        # Exécution des stratégies
        strategies = self.strategy_loader.get_all_strategies()
        signals = []
        strategies_executed = 0
        strategies_with_signals = 0
        no_signal_reasons = []
        
        logger.info(f"Analyse de {symbol} {timeframe} ({len(strategies)} stratégies)")
        
        for strategy_name, strategy_class in strategies.items():
            strategy_start = time.time()
            strategy_logger = logging.getLogger(f'analyzer.{strategy_name}')
            
            try:
                # Log des indicateurs clés pour debug
                self._log_strategy_debug(strategy_name, market_data['indicators'], symbol, timeframe)
                
                # Instanciation de la stratégie
                strategy = strategy_class(
                    symbol=symbol,
                    data=market_data['data'],
                    indicators=market_data['indicators']
                )
                
                # Génération du signal
                signal = strategy.generate_signal()
                strategies_executed += 1
                
                # Timing par stratégie (seulement en DEBUG)
                strategy_time = time.time() - strategy_start
                if logger.isEnabledFor(logging.DEBUG) and strategy_time > 0.01:  # > 10ms
                    strategy_logger.debug(f"{symbol} {timeframe} - Temps d'exécution: {strategy_time*1000:.1f}ms")
                
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
                    strategies_with_signals += 1
                    
                    strategy_logger.info(f"Signal {signal['side']} généré (confidence={signal['confidence']:.2f}): {signal['reason'][:80]}")
                    
                else:
                    # Logger pourquoi aucun signal n'a été généré
                    reason = signal.get('reason', 'Raison inconnue')
                    strategy_logger.debug(f"Pas de signal: {reason[:60]}")
                    no_signal_reasons.append(f"{strategy_name}: {reason[:40]}")
                    
            except Exception as e:
                logger.error(f"Erreur stratégie {strategy_name} pour {symbol} {timeframe}: {e}")
                no_signal_reasons.append(f"{strategy_name}: ERREUR")
                
        # Statistiques de l'analyse
        analysis_time = time.time() - start_time
        success_rate = (strategies_with_signals / strategies_executed * 100) if strategies_executed > 0 else 0
        
        logger.info(f"{symbol} {timeframe}: {strategies_with_signals}/{strategies_executed} stratégies ont généré des signaux "
                   f"({success_rate:.1f}%) - {analysis_time*1000:.0f}ms")
        
        # Log des raisons de non-génération en DEBUG
        if logger.isEnabledFor(logging.DEBUG) and no_signal_reasons:
            logger.debug(f"{symbol} {timeframe} - Principales raisons sans signal: {'; '.join(no_signal_reasons[:3])}")
                
        # Mise à jour des statistiques globales
        self.cycle_stats['total_analyses'] += 1
        self.cycle_stats['total_signals'] += len(signals)
        self.cycle_stats['strategies_executed'] += strategies_executed
        
        # Publication des signaux
        if signals:
            await self.redis_publisher.publish_signals(signals)
            
    async def run_analysis_cycle(self):
        """Exécute un cycle d'analyse complet pour tous les symboles et timeframes."""
        cycle_start_time = time.time()
        self.cycle_stats['cycle_count'] += 1
        
        # Reset des compteurs pour ce cycle
        cycle_signals_before = self.cycle_stats['total_signals']
        cycle_analyses_before = self.cycle_stats['total_analyses']
        
        # Utiliser les combinaisons valides au lieu du produit cartésien
        combinations_to_analyze = getattr(self, 'valid_combinations', [])
        if not combinations_to_analyze:
            # Fallback si pas initialisé
            combinations_to_analyze = [(s, t) for s in self.symbols for t in self.timeframes]
            
        logger.info(f"Début du cycle d'analyse #{self.cycle_stats['cycle_count']} "
                   f"({len(combinations_to_analyze)} combinaisons valides)")
        
        tasks = []
        for symbol, timeframe in combinations_to_analyze:
            task = self.analyze_symbol_timeframe(symbol, timeframe)
            tasks.append(task)
                
        # Exécution en parallèle
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Traitement des exceptions
        exceptions_count = 0
        for result in results:
            if isinstance(result, Exception):
                exceptions_count += 1
                logger.error(f"Exception dans l'analyse: {result}")
        
        # Statistiques du cycle
        cycle_time = time.time() - cycle_start_time
        cycle_signals = self.cycle_stats['total_signals'] - cycle_signals_before
        cycle_analyses = self.cycle_stats['total_analyses'] - cycle_analyses_before
        
        # Calcul des statistiques
        overall_success_rate = (self.cycle_stats['total_signals'] / self.cycle_stats['total_analyses'] * 100) if self.cycle_stats['total_analyses'] > 0 else 0
        cycle_success_rate = (cycle_signals / cycle_analyses * 100) if cycle_analyses > 0 else 0
        avg_time_per_analysis = (cycle_time / cycle_analyses * 1000) if cycle_analyses > 0 else 0
        
        # Log principal des statistiques du cycle
        stats_logger.info(f"Cycle #{self.cycle_stats['cycle_count']} terminé: "
                         f"{cycle_signals} signaux générés sur {cycle_analyses} analyses "
                         f"({cycle_success_rate:.1f}%) - {cycle_time:.1f}s "
                         f"(moy: {avg_time_per_analysis:.0f}ms/analyse)")
        
        # Log des statistiques globales (moins fréquent)
        if self.cycle_stats['cycle_count'] % 5 == 0:  # Toutes les 5 cycles
            stats_logger.info(f"Statistiques globales: {self.cycle_stats['total_signals']} signaux totaux, "
                             f"{self.cycle_stats['total_analyses']} analyses totales "
                             f"(taux global: {overall_success_rate:.1f}%)")
        
        # Log des exceptions si présentes
        if exceptions_count > 0:
            logger.warning(f"Cycle terminé avec {exceptions_count} exceptions")
            
        logger.info("Cycle d'analyse terminé")
        
    async def listen_for_analysis_triggers(self):
        """Écoute les notifications Redis pour déclencher les analyses."""
        redis_client = None
        try:
            redis_client = redis.from_url(self.redis_url)
            pubsub = redis_client.pubsub()
            await pubsub.subscribe('analyzer_trigger')
            
            logger.info("🎧 Écoute des triggers d'analyse activée - Attente des nouvelles données...")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        notification = json.loads(message['data'])
                        if notification.get('event') == 'analyzer_data_ready':
                            symbol = notification.get('symbol')
                            timeframe = notification.get('timeframe')
                            
                            logger.debug(f"📬 Trigger reçu: {symbol} {timeframe}")
                            
                            # Lancer cycle d'analyse (toujours complet pour simplicité)
                            await self.run_analysis_cycle()
                            
                    except Exception as e:
                        logger.error(f"❌ Erreur traitement trigger: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Erreur écoute Redis: {e}")
        finally:
            if redis_client:
                await redis_client.close()
        
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
        
        # Boucle principale : écoute événements Redis
        try:
            await self.listen_for_analysis_triggers()
                    
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