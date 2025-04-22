"""
Point d'entrée principal pour le microservice Portfolio.
Démarre l'API REST et les tâches en arrière-plan.
"""
import argparse
import logging
import signal
import sys
import time
import os
import threading
from datetime import datetime, timedelta
import asyncio

import uvicorn
from fastapi import FastAPI

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import SYMBOLS, LOG_LEVEL
from shared.src.redis_client import RedisClient

from portfolio.src.api import app
from portfolio.src.models import PortfolioModel, DBManager
from portfolio.src.pockets import PocketManager

# Configuration du logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('portfolio.log')
    ]
)
logger = logging.getLogger("portfolio")

# Événement d'arrêt global
stop_event = threading.Event()

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description='Portfolio RootTrading')
    parser.add_argument(
        '--host', 
        type=str, 
        default='0.0.0.0', 
        help='Hôte pour l\'API REST'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000, 
        help='Port pour l\'API REST'
    )
    parser.add_argument(
        '--no-sync', 
        action='store_true', 
        help='Désactive la synchronisation automatique'
    )
    parser.add_argument(
        '--sync-interval', 
        type=int, 
        default=300,  # 5 minutes
        help='Intervalle de synchronisation en secondes'
    )
    return parser.parse_args()

def handle_market_data(channel: str, data: dict):
    """
    Traite les données de marché reçues de Redis.
    Met à jour les prix du portefeuille.
    
    Args:
        channel: Canal Redis d'où proviennent les données
        data: Données de marché
    """
    if not data.get('is_closed', False):
        return  # Ne traiter que les chandeliers fermés
    
    symbol = data.get('symbol')
    price = data.get('close')
    
    if not symbol or price is None:
        return
    
    logger.debug(f"Mise à jour du prix: {symbol} @ {price}")
    
    # Mettre à jour le prix dans la base de données (si nécessaire)
    try:
        db = DBManager()
        
        # Insérer dans market_data
        query = """
        INSERT INTO market_data (time, symbol, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (time, symbol) DO UPDATE
        SET open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
        """
        
        params = (
            datetime.fromtimestamp(data['start_time'] / 1000),
            symbol,
            data['open'],
            data['high'],
            data['low'],
            data['close'],
            data['volume']
        )
        
        db.execute_query(query, params, commit=True)
        db.close()
    
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour du prix: {str(e)}")

def handle_balance_update(channel: str, data: dict):
    """
    Traite les mises à jour de solde reçues de Redis.
    Met à jour les soldes du portefeuille.
    
    Args:
        channel: Canal Redis d'où proviennent les données
        data: Données de solde
    """
    if not isinstance(data, list):
        logger.warning(f"Format de solde invalide: {data}")
        return
    
    try:
        from shared.src.schemas import AssetBalance
        
        # Convertir en AssetBalance
        balances = []
        for balance_data in data:
            if not isinstance(balance_data, dict):
                continue
            
            balance = AssetBalance(
                asset=balance_data.get('asset'),
                free=float(balance_data.get('disponible', 0)),
                locked=float(balance_data.get('en_ordre', 0)),
                total=float(balance_data.get('total', 0)),
                value_usdc=float(balance_data.get('eur_value', 0))  # Utiliser le champ eur_value comme value_usdc
            )
            balances.append(balance)
        
        if not balances:
            return
        
        # Mettre à jour les soldes
        portfolio = PortfolioModel()
        portfolio.update_balances(balances)
        
        # Mettre à jour les poches
        total_value = sum(b.value_usdc or 0 for b in balances)
        if total_value > 0:
            pocket_manager = PocketManager()
            pocket_manager.update_pockets_allocation(total_value)
            pocket_manager.close()
        
        portfolio.close()
        
        logger.info(f"Soldes mis à jour: {len(balances)} actifs, valeur totale: {total_value:.2f} USDC")
    
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour des soldes: {str(e)}")

def sync_task(interval: int):
    """
    Tâche périodique pour synchroniser les poches avec les trades actifs.
    
    Args:
        interval: Intervalle entre les synchronisations en secondes
    """
    logger.info(f"Démarrage de la tâche de synchronisation (intervalle: {interval}s)")
    
    while not stop_event.is_set():
        try:
            # Attendre l'intervalle ou l'événement d'arrêt
            if stop_event.wait(timeout=interval):
                break
            
            # Synchroniser les poches
            logger.info("Synchronisation des poches...")
            pocket_manager = PocketManager()
            pocket_manager.sync_with_trades()
            pocket_manager.close()
            
            # Mettre à jour les statistiques quotidiennes
            logger.info("Mise à jour des statistiques...")
            db = DBManager()
            db.execute_query(
                "CALL update_daily_stats(%s)",
                (datetime.now().date(),),
                commit=True
            )
            db.close()
            
            logger.info("Synchronisation terminée")
        
        except Exception as e:
            logger.error(f"Erreur lors de la synchronisation: {str(e)}")
            time.sleep(10)  # Pause en cas d'erreur
    
    logger.info("Tâche de synchronisation arrêtée")

def subscribe_to_redis_channels():
    """
    S'abonne aux canaux Redis pour recevoir les données de marché et les soldes.
    """
    try:
        redis_client = RedisClient()
        
        # S'abonner aux données de marché
        market_channels = [f"roottrading:market:data:{symbol.lower()}" for symbol in SYMBOLS]
        redis_client.subscribe(market_channels, handle_market_data)
        logger.info(f"Abonné aux canaux de données de marché: {len(market_channels)} canaux")
        
        # S'abonner aux mises à jour de solde
        balance_channel = "roottrading:account:balances"
        redis_client.subscribe(balance_channel, handle_balance_update)
        logger.info(f"Abonné au canal des soldes: {balance_channel}")
        
        return redis_client
    
    except Exception as e:
        logger.error(f"Erreur lors de l'abonnement aux canaux Redis: {str(e)}")
        return None

def shutdown_handler(signum, frame):
    """
    Gestionnaire de signal pour l'arrêt propre.
    
    Args:
        signum: Numéro du signal
        frame: Frame actuelle
    """
    logger.info(f"Signal {signum} reçu, arrêt en cours...")
    stop_event.set()

def main():
    """
    Fonction principale qui démarre le service Portfolio.
    """
    # Parser les arguments
    args = parse_arguments()
    
    # Configurer les gestionnaires de signaux
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    logger.info("🚀 Démarrage du service Portfolio RootTrading...")
    
    # S'abonner aux canaux Redis
    redis_client = subscribe_to_redis_channels()
    
    # Démarrer la tâche de synchronisation si activée
    sync_thread = None
    if not args.no_sync:
        sync_thread = threading.Thread(
            target=sync_task,
            args=(args.sync_interval,),
            daemon=True
        )
        sync_thread.start()
    
    try:
        # Démarrer l'API FastAPI
        logger.info(f"Démarrage de l'API REST sur {args.host}:{args.port}...")
        uvicorn.run(app, host=args.host, port=args.port)
    
    except KeyboardInterrupt:
        logger.info("Interruption clavier détectée")
    except Exception as e:
        logger.error(f"Erreur critique: {str(e)}")
    finally:
        # Arrêter proprement
        logger.info("Arrêt du service Portfolio...")
        stop_event.set()
        
        # Attendre l'arrêt de la tâche de synchronisation
        if sync_thread and sync_thread.is_alive():
            sync_thread.join(timeout=5.0)
        
        # Fermer la connexion Redis
        if redis_client:
            redis_client.close()
        
        logger.info("Service Portfolio arrêté")

if __name__ == "__main__":
    main()