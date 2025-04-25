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
import traceback
import requests
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import SYMBOLS, LOG_LEVEL, BINANCE_API_KEY, BINANCE_SECRET_KEY
from shared.src.redis_client import RedisClient

from portfolio.src.api import app
from portfolio.src.models import PortfolioModel, DBManager
from portfolio.src.pockets import PocketManager
from portfolio.src.binance_account_manager import BinanceAccountManager

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
# Cache global des prix actuels
current_prices = {}

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
        '--binance-sync-interval', 
        type=int, 
        default=300,  # 5 minutes
        help='Intervalle de synchronisation Binance en secondes'
    )
    parser.add_argument(
        '--db-sync-interval', 
        type=int, 
        default=300,  # 5 minutes
        help='Intervalle de synchronisation DB en secondes'
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
    global current_prices
    
    symbol = data.get('symbol')
    price = data.get('close')
    
    if not symbol or price is None:
        return
    
    # Mettre à jour le cache de prix
    if symbol:
        current_prices[symbol] = price
    
    # Ne traiter que les chandeliers fermés pour la mise à jour de la base de données
    if not data.get('is_closed', False):
        return
    
    logger.info(f"Mise à jour du prix: {symbol} @ {price}")
    
    # Mettre à jour le prix dans la base de données
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
        
        timestamp = datetime.fromtimestamp(data['start_time'] / 1000)
        
        params = (
            timestamp,
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
        logger.error(traceback.format_exc())

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
            
            # Adapter le format en fonction de la source des données
            # Format Binance: {"asset": "BTC", "free": "0.1", "locked": "0.0"}
            # Format interne: {"asset": "BTC", "disponible": "0.1", "en_ordre": "0.0", "total": "0.1", "eur_value": "4000.0"}
            
            asset = balance_data.get('asset')
            
            # Vérifier si c'est le format Binance ou le format interne
            if 'free' in balance_data:
                free = float(balance_data.get('free', 0))
                locked = float(balance_data.get('locked', 0))
                total = free + locked
                value_usdc = balance_data.get('value_usdc')
            else:
                free = float(balance_data.get('disponible', 0))
                locked = float(balance_data.get('en_ordre', 0))
                total = float(balance_data.get('total', 0))
                value_usdc = float(balance_data.get('eur_value', 0))  # Utiliser le champ eur_value comme value_usdc
                
            balance = AssetBalance(
                asset=asset,
                free=free,
                locked=locked,
                total=total,
                value_usdc=value_usdc
            )
            balances.append(balance)
        
        if not balances:
            logger.warning("Aucune balance à mettre à jour reçue")
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
        logger.error(traceback.format_exc())

def get_current_prices() -> Dict[str, float]:
    """
    Récupère les prix actuels des actifs.
    Utilise d'abord le cache, puis tente de récupérer depuis la base de données.
    
    Returns:
        Dictionnaire {symbole: prix}
    """
    global current_prices
    
    # Si cache vide, essayer de remplir depuis la base de données
    if not current_prices:
        try:
            db = DBManager()
            # Récupérer les derniers prix pour chaque symbole
            query = """
            WITH latest_data AS (
                SELECT 
                    symbol, 
                    MAX(time) as latest_time
                FROM 
                    market_data
                GROUP BY 
                    symbol
            )
            SELECT 
                m.symbol, 
                m.close
            FROM 
                market_data m
            JOIN 
                latest_data l ON m.symbol = l.symbol AND m.time = l.latest_time
            """
            
            result = db.execute_query(query, fetch_all=True)
            db.close()
            
            if result:
                # Mettre à jour le cache
                for row in result:
                    symbol = row['symbol']
                    price = float(row['close'])
                    current_prices[symbol] = price
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des prix actuels: {str(e)}")
    
    return current_prices

def update_balances_from_binance():
    """
    Récupère et met à jour les balances depuis Binance.
    """
    logger.info("Mise à jour des balances depuis Binance...")
    
    try:
        # Vérifier si les clés API sont configurées
        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            logger.warning("Clés API Binance non configurées, impossible de récupérer les balances")
            return False
        
        # Créer le gestionnaire Binance
        account_manager = BinanceAccountManager(
            api_key=BINANCE_API_KEY,
            api_secret=BINANCE_SECRET_KEY
        )
        
        # Récupérer les balances
        binance_balances = account_manager.get_balances()
        
        if not binance_balances:
            logger.warning("Aucune balance récupérée depuis Binance")
            return False
        
        # Convertir en AssetBalance
        from shared.src.schemas import AssetBalance
        asset_balances = []
        
        # Prix actuel des actifs pour l'évaluation en USDC
        prices = get_current_prices()
        
        # Si on n'a pas les prix, essayer de les récupérer via l'API Binance
        if not prices:
            try:
                prices = account_manager.get_ticker_prices()
                # Mettre à jour le cache global
                for symbol, price in prices.items():
                    if symbol.endswith('USDC'):
                        base_asset = symbol[:-4]  # Enlever 'USDC'
                        current_prices[symbol] = price
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des prix Binance: {str(e)}")
        
        for balance in binance_balances:
            asset = balance["asset"]
            
            # Calculer la valeur en USDC
            value_usdc = None
            if asset == "USDC":
                value_usdc = balance["total"]
            elif f"{asset}USDC" in prices:
                value_usdc = balance["total"] * prices[f"{asset}USDC"]
            
            asset_balance = AssetBalance(
                asset=asset,
                free=balance["free"],
                locked=balance["locked"],
                total=balance["total"],
                value_usdc=value_usdc
            )
            asset_balances.append(asset_balance)
        
        # Mettre à jour les balances dans la base de données
        portfolio = PortfolioModel()
        success = portfolio.update_balances(asset_balances)
        portfolio.close()
        
        if not success:
            logger.error("Échec de la mise à jour des balances dans la base de données")
            return False
        
        # Publier sur Redis pour informer les autres services
        try:
            redis_client = RedisClient()
            redis_client.publish("roottrading:account:balances", asset_balances)
            redis_client.close()
        except Exception as e:
            logger.error(f"Erreur lors de la publication des balances sur Redis: {str(e)}")
        
        logger.info(f"✅ {len(asset_balances)} balances mises à jour depuis Binance")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la mise à jour des balances depuis Binance: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def binance_sync_task(interval: int):
    """
    Tâche périodique pour synchroniser les balances avec Binance.
    
    Args:
        interval: Intervalle entre les synchronisations en secondes
    """
    logger.info(f"Démarrage de la tâche de synchronisation Binance (intervalle: {interval}s)")
    
    # Première synchronisation au démarrage
    try:
        update_balances_from_binance()
    except Exception as e:
        logger.error(f"Erreur lors de la synchronisation initiale avec Binance: {str(e)}")
    
    while not stop_event.is_set():
        try:
            # Attendre l'intervalle ou l'événement d'arrêt
            if stop_event.wait(timeout=interval):
                break
            
            # Mettre à jour les balances
            update_balances_from_binance()
            
        except Exception as e:
            logger.error(f"Erreur lors de la synchronisation avec Binance: {str(e)}")
            logger.error(traceback.format_exc())
            time.sleep(60)  # Pause plus longue en cas d'erreur
    
    logger.info("Tâche de synchronisation Binance arrêtée")

def db_sync_task(interval: int):
    """
    Tâche périodique pour synchroniser les poches avec les trades actifs dans la base de données.
    
    Args:
        interval: Intervalle entre les synchronisations en secondes
    """
    logger.info(f"Démarrage de la tâche de synchronisation DB (intervalle: {interval}s)")
    
    while not stop_event.is_set():
        try:
            # Attendre l'intervalle ou l'événement d'arrêt
            if stop_event.wait(timeout=interval):
                break
            
            # Synchroniser les poches avec les trades actifs
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
            
            logger.info("Synchronisation DB terminée")
        
        except Exception as e:
            logger.error(f"Erreur lors de la synchronisation DB: {str(e)}")
            logger.error(traceback.format_exc())
            time.sleep(60)  # Pause plus longue en cas d'erreur
    
    logger.info("Tâche de synchronisation DB arrêtée")

def create_tables_if_needed():
    """
    Vérifie et crée les tables nécessaires si elles n'existent pas déjà.
    """
    try:
        db = DBManager()
        
        # Vérifier si la table portfolio_balances existe
        check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'portfolio_balances'
        );
        """
        
        result = db.execute_query(check_query, fetch_one=True)
        
        if not result or not result.get('exists', False):
            logger.warning("Table portfolio_balances non trouvée, création en cours...")
            
            # Créer la table
            create_query = """
            CREATE TABLE IF NOT EXISTS portfolio_balances (
                id SERIAL PRIMARY KEY,
                asset VARCHAR(10) NOT NULL,
                free NUMERIC(24, 8) NOT NULL,
                locked NUMERIC(24, 8) NOT NULL,
                total NUMERIC(24, 8) NOT NULL,
                value_usdc NUMERIC(24, 8),
                timestamp TIMESTAMP NOT NULL DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS portfolio_balances_asset_idx ON portfolio_balances(asset);
            CREATE INDEX IF NOT EXISTS portfolio_balances_timestamp_idx ON portfolio_balances(timestamp);
            """
            
            db.execute_query(create_query, commit=True)
            logger.info("Table portfolio_balances créée avec succès")
        
        # Vérifier si la table capital_pockets existe
        check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'capital_pockets'
        );
        """
        
        result = db.execute_query(check_query, fetch_one=True)
        
        if not result or not result.get('exists', False):
            logger.warning("Table capital_pockets non trouvée, création en cours...")
            
            # Créer la table
            create_query = """
            CREATE TABLE IF NOT EXISTS capital_pockets (
                id SERIAL PRIMARY KEY,
                pocket_type VARCHAR(20) NOT NULL CHECK (pocket_type IN ('active', 'buffer', 'safety')),
                allocation_percent NUMERIC(5, 2) NOT NULL,
                current_value NUMERIC(24, 8) NOT NULL,
                used_value NUMERIC(24, 8) NOT NULL,
                available_value NUMERIC(24, 8) NOT NULL,
                active_cycles INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
            """
            
            db.execute_query(create_query, commit=True)
            logger.info("Table capital_pockets créée avec succès")
        
        db.close()
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de la vérification/création des tables: {str(e)}")
        logger.error(traceback.format_exc())
        return False

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
        logger.error(traceback.format_exc())
        return None

def initialize_default_balances():
    """
    Initialise les balances avec des valeurs par défaut si aucune balance n'existe.
    """
    try:
        db = DBManager()
        portfolio = PortfolioModel(db_manager=db)
        
        # Vérifier si des balances existent
        balances = portfolio.get_latest_balances()
        
        # Si aucune balance n'existe, créer des valeurs par défaut
        if not balances:
            logger.info("Aucune balance trouvée, initialisation avec des valeurs par défaut...")
            from shared.src.schemas import AssetBalance
            default_balance = [
                AssetBalance(
                    asset="USDC",
                    free=100.0,
                    locked=0.0,
                    total=100.0,
                    value_usdc=100.0
                )
            ]
            portfolio.update_balances(default_balance)
            
            # Initialiser les poches avec cette valeur
            pocket_manager = PocketManager(db_manager=db)
            pocket_manager.update_pockets_allocation(100.0)
            pocket_manager.close()
            
            logger.info("✅ Balances et poches initialisées avec des valeurs par défaut")
        else:
            logger.info(f"✅ {len(balances)} balances existantes trouvées")
        
        portfolio.close()
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des balances: {str(e)}")
        logger.error(traceback.format_exc())
        return False

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
    
    # Vérifier et créer les tables si nécessaire
    if not create_tables_if_needed():
        logger.error("❌ Impossible de créer les tables nécessaires, arrêt du service")
        return
    
    # Initialiser les balances par défaut si nécessaire
    if not initialize_default_balances():
        logger.warning("⚠️ Initialisation des balances par défaut échouée, poursuite avec prudence")
    
    # S'abonner aux canaux Redis
    redis_client = subscribe_to_redis_channels()
    
    threads = []
    
    # Démarrer la tâche de synchronisation Binance
    if not args.no_sync:
        binance_thread = threading.Thread(
            target=binance_sync_task,
            args=(args.binance_sync_interval,),
            daemon=True,
            name="binance-sync"
        )
        binance_thread.start()
        threads.append(binance_thread)
        
        # Démarrer la tâche de synchronisation DB
        db_thread = threading.Thread(
            target=db_sync_task,
            args=(args.db_sync_interval,),
            daemon=True,
            name="db-sync"
        )
        db_thread.start()
        threads.append(db_thread)
    
    try:
        # Démarrer l'API FastAPI
        logger.info(f"Démarrage de l'API REST sur {args.host}:{args.port}...")
        uvicorn.run(app, host=args.host, port=args.port)
    
    except KeyboardInterrupt:
        logger.info("Interruption clavier détectée")
    except Exception as e:
        logger.error(f"Erreur critique: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Arrêter proprement
        logger.info("Arrêt du service Portfolio...")
        stop_event.set()
        
        # Attendre l'arrêt des threads de synchronisation
        for thread in threads:
            if thread.is_alive():
                logger.info(f"Attente de l'arrêt du thread {thread.name}...")
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(f"Le thread {thread.name} ne s'est pas arrêté proprement")
        
        # Fermer la connexion Redis
        if redis_client:
            redis_client.close()
        
        logger.info("Service Portfolio arrêté")

if __name__ == "__main__":
    main()