"""
Point d'entrée principal pour le microservice Analyzer.
Démarre le gestionnaire d'analyse multiprocessus pour traiter les données de marché et générer des signaux.
"""
import argparse
import logging
import signal
import sys
import time
import os
import threading
import psutil
from flask import Flask, jsonify, request

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import SYMBOLS, LOG_LEVEL
from shared.src.redis_client import RedisClient

from analyzer.src.optimized_analyzer import OptimizedAnalyzer
from analyzer.src.strategy_loader import StrategyLoader
from analyzer.src.redis_subscriber import RedisSubscriber

# Configuration du logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('analyzer.log')
    ]
)
logger = logging.getLogger("analyzer")

# Classe principale pour gérer le service Analyzer
class AnalyzerService:
    """
    Service principal Analyzer qui gère l'API REST et le cycle de vie du gestionnaire d'analyse.
    """
    
    def __init__(self, symbols=None, port=5012):
        """
        Initialise le service Analyzer optimisé.
        
        Args:
            symbols: Liste des symboles à analyser
            port: Port pour l'API REST
        """
        self.symbols = symbols or SYMBOLS
        self.port = port
        self.start_time = time.time()
        self.running = False
        
        # Composants de l'analyzer optimisé
        self.strategy_loader = None
        self.optimized_analyzer = None
        self.redis_subscriber = None
        
        self.process = psutil.Process(os.getpid())
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """
        Configure les routes de l'API Flask.
        """
        self.app.route('/health', methods=['GET'])(self.health_check)
        self.app.route('/diagnostic', methods=['GET'])(self.diagnostic)
        self.app.route('/strategies', methods=['GET'])(self.list_strategies)
        self.app.route('/api/indicators/<symbol>', methods=['GET'])(self.get_indicators)
    
    def health_check(self):
        """
        Point de terminaison pour vérifier l'état du service.
        """
        return jsonify({
            "status": "healthy" if self.running else "stopped",
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "symbols": self.symbols
        })
    
    def diagnostic(self):
        """
        Point de terminaison pour le diagnostic du service optimisé.
        """
        if not self.optimized_analyzer:
            return jsonify({
                "status": "stopped",
                "timestamp": time.time(),
                "uptime": time.time() - self.start_time,
                "error": "Optimized analyzer not running"
            }), 503
            
        # État de l'analyzer optimisé
        analyzer_status = {
            "running": self.running,
            "strategy_loader_active": self.strategy_loader is not None,
            "redis_subscriber_active": self.redis_subscriber is not None,
            "strategies_count": self.strategy_loader.get_strategy_count() if self.strategy_loader else 0
        }
        
        # Construire la réponse
        diagnostic_info = {
            "status": "operational" if self.running else "stopped",
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "architecture": "optimized_db_first",
            "analyzer": analyzer_status,
            "symbols": self.symbols,
            "memory_usage_mb": round(self.process.memory_info().rss / (1024 * 1024), 2),
            "cpu_percent": self.process.cpu_percent(interval=0.1),
            "thread_count": threading.active_count()
        }
        
        return jsonify(diagnostic_info)
    
    def list_strategies(self):
        """
        Liste toutes les stratégies ultra-précises chargées.
        """
        if not self.strategy_loader:
            return jsonify({
                "error": "Strategy loader not running"
            }), 503
            
        try:
            strategies = self.strategy_loader.get_strategy_list()
            return jsonify({
                "strategies": strategies,
                "total_count": self.strategy_loader.get_strategy_count(),
                "architecture": "ultra_precise_db_first"
            })
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stratégies: {str(e)}")
            return jsonify({
                "error": f"Failed to retrieve strategies: {str(e)}"
            }), 500
    
    def get_indicators(self, symbol):
        """
        Retourne les indicateurs techniques depuis la DB pour un symbole donné.
        Architecture DB-first optimisée.
        """
        if not self.optimized_analyzer:
            return jsonify({
                "error": "Optimized analyzer not running"
            }), 503
        
        # Vérifier que le symbole est supporté
        if symbol not in self.symbols:
            return jsonify({
                "error": f"Symbol {symbol} not supported",
                "supported_symbols": self.symbols
            }), 400
        
        try:
            import asyncio
            from analyzer.src.indicators.db_indicators import db_indicators
            
            # Récupérer les indicateurs depuis la DB
            async def get_db_indicators():
                return await db_indicators.get_enriched_market_data(symbol, limit=1)
            
            # Exécuter l'appel asynchrone
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                df = loop.run_until_complete(get_db_indicators())
            finally:
                loop.close()
            
            if df is None or df.empty:
                return jsonify({
                    "error": f"No enriched data found for {symbol}",
                    "data_source": "database"
                }), 404
            
            # Extraire les indicateurs de la dernière chandelle
            latest = df.iloc[-1]
            
            indicators_data = {
                "symbol": symbol,
                "timestamp": latest.name.isoformat(),
                "data_source": "postgresql_enriched",
                "architecture": "db_first_optimized",
                "ohlcv": {
                    "open": float(latest['open']),
                    "high": float(latest['high']),
                    "low": float(latest['low']),
                    "close": float(latest['close']),
                    "volume": float(latest['volume'])
                },
                "indicators": {
                    "rsi_14": float(latest.get('rsi_14', 0)),
                    "atr_14": float(latest.get('atr_14', 0)),
                    "adx_14": float(latest.get('adx_14', 0)),
                    "macd_line": float(latest.get('macd_line', 0)),
                    "macd_signal": float(latest.get('macd_signal', 0)),
                    "macd_histogram": float(latest.get('macd_histogram', 0)),
                    "bb_upper": float(latest.get('bb_upper', 0)),
                    "bb_middle": float(latest.get('bb_middle', 0)),
                    "bb_lower": float(latest.get('bb_lower', 0)),
                    "ema_12": float(latest.get('ema_12', 0)),
                    "ema_26": float(latest.get('ema_26', 0)),
                    "ema_50": float(latest.get('ema_50', 0)),
                    "sma_20": float(latest.get('sma_20', 0)),
                    "sma_50": float(latest.get('sma_50', 0)),
                    "stoch_k": float(latest.get('stoch_k', 0)),
                    "stoch_d": float(latest.get('stoch_d', 0)),
                    "obv": float(latest.get('obv', 0))
                }
            }
            
            return jsonify(indicators_data)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des indicateurs DB pour {symbol}: {str(e)}")
            return jsonify({
                "error": f"Failed to retrieve DB indicators for {symbol}: {str(e)}"
            }), 500
    
    def start(self):
        """
        Démarre le service Analyzer optimisé.
        """
        if self.running:
            logger.warning("Le service est déjà en cours d'exécution")
            return
            
        self.running = True
        
        logger.info("🚀 Démarrage du service Analyzer RootTrading OPTIMISÉ...")
        logger.info(f"Configuration: {len(self.symbols)} symboles, architecture DB-first")
        
        try:
            # 1. Charger les stratégies ultra-précises
            self.strategy_loader = StrategyLoader(symbols=self.symbols)
            logger.info(f"✅ {self.strategy_loader.get_strategy_count()} stratégies ultra-précises chargées")
            
            # 2. Créer l'analyzer optimisé
            self.optimized_analyzer = OptimizedAnalyzer(
                strategy_loader=self.strategy_loader,
                max_workers=4
            )
            logger.info("✅ Analyzer optimisé initialisé")
            
            # 3. Démarrer le subscriber Redis avec callback d'analyse
            self.redis_subscriber = RedisSubscriber(symbols=self.symbols)
            self.redis_subscriber.start_listening(self._process_market_data)
            logger.info("✅ Subscriber Redis démarré")
            
            logger.info("✅ Service Analyzer optimisé démarré et en attente de données")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du démarrage de l'analyzer optimisé: {str(e)}")
            self.running = False
            raise
    
    def _process_market_data(self, data: dict):
        """
        Callback pour traiter les données de marché avec l'analyzer optimisé.
        """
        try:
            if not data or not data.get('symbol'):
                return
                
            symbol = data['symbol']
            
            # Analyser avec les stratégies en utilisant les données DB
            import asyncio
            
            async def analyze_data():
                return await self.optimized_analyzer._analyze_symbol_from_db(symbol)
            
            # Exécuter l'analyse
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                signals = loop.run_until_complete(analyze_data())
            finally:
                loop.close()
            
            # Publier les signaux via Redis
            if signals:
                for signal_dict in signals:
                    try:
                        # Convertir en StrategySignal et publier
                        from shared.src.schemas import StrategySignal, TradeSide, SignalStrength
                        
                        strategy_signal = StrategySignal(
                            strategy=signal_dict['strategy'],
                            symbol=signal_dict['symbol'],
                            side=signal_dict['side'],
                            price=signal_dict['price'],
                            confidence=signal_dict['confidence'],
                            timestamp=signal_dict.get('timestamp'),
                            strength=signal_dict.get('strength', SignalStrength.MEDIUM)
                        )
                        
                        self.redis_subscriber.publish_signal(strategy_signal)
                        
                    except Exception as e:
                        logger.error(f"❌ Erreur publication signal pour {symbol}: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Erreur traitement données marché: {e}")
    
    def start_api(self, debug=False):
        """
        Démarre l'API REST dans un thread séparé.
        
        Args:
            debug: Activer le mode debug pour Flask
        """
        api_thread = threading.Thread(
            target=lambda: self.app.run(host='0.0.0.0', port=self.port, debug=debug, use_reloader=False, threaded=True),
            daemon=True
        )
        api_thread.start()
        logger.info(f"✅ API REST démarrée sur le port {self.port}")
        return api_thread
    
    def stop(self):
        """
        Arrête proprement le service Analyzer optimisé.
        """
        if not self.running:
            return
            
        logger.info("Arrêt du service Analyzer optimisé...")
        self.running = False
        
        # Arrêter les composants proprement
        if self.redis_subscriber:
            self.redis_subscriber.stop()
            self.redis_subscriber = None
            
        if self.optimized_analyzer:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.optimized_analyzer.close())
            finally:
                loop.close()
            self.optimized_analyzer = None
            
        self.strategy_loader = None
            
        logger.info("Service Analyzer optimisé terminé")
        

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description='Analyzer de trading RootTrading OPTIMISÉ')
    parser.add_argument(
        '--symbols', 
        type=str, 
        default=None, 
        help='Liste de symboles séparés par des virgules (ex: BTCUSDC,ETHUSDC)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=5012, 
        help='Port pour l\'API REST'
    )
    parser.add_argument(
        '--no-api', 
        action='store_true', 
        help='Désactive l\'API REST'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Active le mode debug'
    )
    return parser.parse_args()


def main():
    """Fonction principale du service Analyzer OPTIMISÉ."""
    # Parser les arguments
    args = parse_arguments()
    
    # Configurer les symboles
    symbols = args.symbols.split(',') if args.symbols else SYMBOLS
    
    # Gestionnaire de signaux pour l'arrêt propre
    service = AnalyzerService(
        symbols=symbols,
        port=args.port
    )
    
    # Configurer les gestionnaires de signaux
    def signal_handler(sig, frame):
        logger.info(f"Signal {sig} reçu, arrêt en cours...")
        service.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Démarrer le service
        service.start()
        
        # Démarrer l'API REST si activée
        if not args.no_api:
            service.start_api(debug=args.debug)
        
        # Boucle principale
        while service.running:
            time.sleep(1.0)
        
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le service Analyzer: {str(e)}")
    finally:
        service.stop()


if __name__ == "__main__":
    main()