"""
Point d'entrée principal pour le microservice PnL Tracker.
Suit et analyse les performances de trading.
"""
import logging
import signal
import sys
import time
import os
import threading
import argparse
from typing import Dict, Any

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import LOG_LEVEL
from pnl_tracker.src.pnl_logger import PnLLogger
from pnl_tracker.src.strategy_tuner import StrategyTuner

# Configuration du logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pnl_tracker.log')
    ]
)
logger = logging.getLogger("pnl_tracker")

# Variables pour le contrôle du service
pnl_logger = None
strategy_tuner = None
running = True

def signal_handler(signum, frame):
    """
    Gestionnaire de signal pour l'arrêt propre.
    
    Args:
        signum: Numéro du signal
        frame: Frame actuelle
    """
    global running
    logger.info(f"Signal {signum} reçu, arrêt en cours...")
    running = False

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description='PnL Tracker et Strategy Tuner')
    parser.add_argument(
        '--export-dir', 
        type=str, 
        default='./exports',
        help='Répertoire pour l\'export des données'
    )
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default='./tuning_results',
        help='Répertoire pour les résultats d\'optimisation'
    )
    parser.add_argument(
        '--update-interval', 
        type=int, 
        default=3600,
        help='Intervalle de mise à jour des statistiques (en secondes)'
    )
    parser.add_argument(
        '--no-tuning', 
        action='store_true', 
        help='Désactive l\'optimisation des stratégies'
    )
    return parser.parse_args()

def run_strategy_optimization(tuner: StrategyTuner) -> None:
    """
    Exécute une optimisation périodique des stratégies.
    
    Args:
        tuner: Instance du tuner de stratégies
    """
    # Liste des stratégies à optimiser
    strategies_to_tune = [
        {"name": "rsi", "param_ranges": {"window": (7, 21), "overbought": (65, 85), "oversold": (15, 35)}},
        {"name": "bollinger", "param_ranges": {"window": (10, 30), "num_std": (1.5, 3.0)}},
        {"name": "ema_cross", "param_ranges": {"short_window": (3, 10), "long_window": (15, 30)}},
    ]
    
    # Liste des symboles pour l'optimisation
    symbols = ["BTCUSDC", "ETHUSDC"]
    
    try:
        logger.info("Démarrage de l'optimisation des stratégies...")
        
        for symbol in symbols:
            for strategy in strategies_to_tune:
                try:
                    logger.info(f"Optimisation de {strategy['name']} pour {symbol}...")
                    
                    results = tuner.optimize_strategy_params(
                        strategy_name=strategy["name"],
                        symbol=symbol,
                        param_ranges=strategy["param_ranges"],
                        days=90  # Optimiser sur 90 jours de données
                    )
                    
                    if results.get("success", False):
                        logger.info(f"✅ Optimisation réussie pour {strategy['name']} ({symbol})")
                        logger.info(f"Paramètres optimisés: {results['optimized_params']}")
                        logger.info(f"PnL: {results['backtest_results']['pnl']:.2f}%, "
                                  f"Win Rate: {results['backtest_results']['win_rate']:.2f}%")
                    else:
                        logger.warning(f"⚠️ Échec de l'optimisation pour {strategy['name']} ({symbol}): {results.get('error', 'Erreur inconnue')}")
                
                except Exception as e:
                    logger.error(f"❌ Erreur lors de l'optimisation de {strategy['name']} pour {symbol}: {str(e)}")
        
        logger.info("Optimisation des stratégies terminée")
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'optimisation des stratégies: {str(e)}")

def main():
    """
    Fonction principale du service PnL Tracker.
    """
    global pnl_logger, strategy_tuner, running
    
    # Parser les arguments
    args = parse_arguments()
    
    # Configurer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 Démarrage du service PnL Tracker RootTrading...")
    
    # Créer les répertoires s'ils n'existent pas
    os.makedirs(args.export_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Variables pour tracker les exports
    last_daily_export_date = None
    last_hourly_export_hour = None
    
    try:
        # Initialiser le logger PnL
        pnl_logger = PnLLogger(export_dir=args.export_dir)
        
        # Gérer les erreurs potentielles lors du premier calcul
        try:
            pnl_logger.update_stats()
            logger.info("✅ Statistiques initiales calculées")
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors du calcul des statistiques initiales: {str(e)}")
            logger.info("Continuons avec des statistiques vides")
        
        # Démarrer le thread de mise à jour des statistiques
        try:
            pnl_logger.start_update_thread(interval=args.update_interval)
            logger.info(f"✅ Thread de mise à jour démarré (intervalle: {args.update_interval}s)")
        except Exception as e:
            logger.error(f"❌ Erreur lors du démarrage du thread de mise à jour: {str(e)}")
        
        # Initialiser le tuner de stratégies si activé
        if not args.no_tuning:
            try:
                strategy_tuner = StrategyTuner(results_dir=args.results_dir)
                logger.info("✅ Strategy Tuner initialisé")
                
                # Planifier une optimisation toutes les 24 heures
                last_optimization_time = 0
                optimization_interval = 24 * 3600  # 24 heures
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'initialisation du Strategy Tuner: {str(e)}")
                strategy_tuner = None
        else:
            logger.info("Optimisation des stratégies désactivée")
        
        logger.info("✅ Service PnL Tracker démarré")
        
        # Tenter d'exporter des statistiques pour tester la fonctionnalité
        try:
            export_path = pnl_logger.export_stats_to_csv()
            if export_path:
                logger.info(f"Test d'exportation réussi: {export_path}")
            else:
                logger.warning("⚠️ Test d'exportation: aucun fichier généré")
        except Exception as e:
            logger.error(f"❌ Erreur lors du test d'exportation: {str(e)}")
        
        # Boucle principale pour garder le service actif
        while running:
            time.sleep(1)
            
            # Exporter les statistiques périodiquement (tous les jours à minuit)
            current_time = time.time()
            current_datetime = time.localtime(current_time)
            current_date = time.strftime("%Y-%m-%d", current_datetime)
            
            # Export horaire (toutes les heures)
            current_hour = current_datetime.tm_hour
            if current_datetime.tm_min == 0 and last_hourly_export_hour != current_hour:
                try:
                    logger.info(f"📊 Export horaire des statistiques PnL ({current_hour}h)")
                    
                    # Nom de fichier avec l'heure
                    timestamp = time.strftime("%Y%m%d_%H", current_datetime)
                    hourly_filename = f"pnl_stats_hourly_{timestamp}.xlsx"
                    
                    # Exporter les statistiques
                    export_path = pnl_logger.export_stats_to_csv(filename=hourly_filename)
                    if export_path:
                        logger.info(f"✅ Export horaire réussi: {export_path}")
                    
                    # Marquer l'export comme effectué pour cette heure
                    last_hourly_export_hour = current_hour
                    
                except Exception as e:
                    logger.error(f"❌ Erreur lors de l'export horaire: {str(e)}")
            
            # À minuit, exporter les statistiques complètes (une seule fois par jour)
            if current_datetime.tm_hour == 0 and current_datetime.tm_min == 0 and last_daily_export_date != current_date:
                try:
                    logger.info(f"🕐 Déclenchement de l'export quotidien pour {current_date}")
                    
                    # Exporter les statistiques
                    export_path = pnl_logger.export_stats_to_csv()
                    if export_path:
                        logger.info(f"✅ Statistiques PnL exportées vers: {export_path}")
                    
                    # Exporter l'historique des trades
                    try:
                        history_path = pnl_logger.export_trade_history(days=90)
                        if history_path:
                            logger.info(f"✅ Historique des trades exporté vers: {history_path}")
                    except Exception as inner_e:
                        logger.error(f"❌ Erreur lors de l'exportation de l'historique: {str(inner_e)}")
                    
                    # Marquer l'export comme effectué pour aujourd'hui
                    last_daily_export_date = current_date
                    logger.info(f"✅ Export quotidien terminé pour {current_date}")
                    
                except Exception as e:
                    logger.error(f"❌ Erreur lors de l'exportation périodique: {str(e)}")
            
            # Exécuter l'optimisation des stratégies périodiquement
            if strategy_tuner and not args.no_tuning and current_time - last_optimization_time > optimization_interval:
                try:
                    # Lancer l'optimisation dans un thread séparé
                    optimization_thread = threading.Thread(
                        target=run_strategy_optimization,
                        args=(strategy_tuner,),
                        daemon=True
                    )
                    optimization_thread.start()
                    
                    # Mettre à jour le timestamp d'optimisation
                    last_optimization_time = current_time
                except Exception as e:
                    logger.error(f"❌ Erreur lors du lancement de l'optimisation: {str(e)}")
    
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le service PnL Tracker: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Arrêter le logger PnL
        if pnl_logger:
            try:
                pnl_logger.stop_update_thread()
                pnl_logger.close()
                logger.info("Service PnL Tracker arrêté proprement")
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'arrêt du PnL Logger: {str(e)}")
        
        logger.info("Service PnL Tracker terminé")

if __name__ == "__main__":
    main()