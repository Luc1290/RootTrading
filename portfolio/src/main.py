"""
Point d'entrée principal pour le microservice Portfolio.
Démarre l'API REST et les tâches en arrière-plan.
Version optimisée avec meilleure gestion des erreurs et des caches.
"""
import argparse
import asyncio
import logging
import signal
import sys
import time
import os
import threading
from datetime import datetime, timedelta
import traceback
import requests
from typing import Dict, List, Optional, Union
import json
import concurrent.futures

import uvicorn
from fastapi import FastAPI
from contextlib import contextmanager

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from portfolio.src import binance_account_manager
from shared.src import redis_client
from shared.src.config import SYMBOLS, LOG_LEVEL, BINANCE_API_KEY, BINANCE_SECRET_KEY
from shared.src.redis_client import RedisClient

from portfolio.src.api import app
from portfolio.src.models import PortfolioModel, DBManager, SharedCache
from portfolio.src.pockets import PocketManager
from portfolio.src.binance_account_manager import BinanceAccountManager, BinanceApiError

# Configuration du logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s',
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
# Mutex pour les opérations sensibles
cache_lock = threading.RLock()

def notify_balance_update(balances, total_value):
    """
    Notifie les autres services d'une mise à jour des balances via Redis.
    
    Args:
        balances: Liste des balances mises à jour
        total_value: Valeur totale du portefeuille
    """
    try:
        # Créer un message de notification avec timestamp pour versioning
        notification = {
            "event": "balance_updated",
            "timestamp": int(time.time() * 1000),
            "total_value": total_value,
            "balance_count": len(balances),
            "updated_at": datetime.now().isoformat()
        }
        
        # Publier sur un canal dédié aux notifications
        redis = RedisClient()
        redis.publish("roottrading:notification:balance_updated", notification)
        logger.info(f"✅ Notification de mise à jour des balances envoyée via Redis")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'envoi de la notification Redis: {str(e)}")
        logger.error(traceback.format_exc())

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
    parser.add_argument(
        '--redis-enabled',
        action='store_true',
        default=True,
        help='Active l\'utilisation de Redis'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Active le mode debug'
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
    
    # Mettre à jour le cache de prix avec lock pour thread-safety
    with cache_lock:
        if symbol:
            current_prices[symbol] = price
    
    # Ne traiter que les chandeliers fermés pour la mise à jour de la base de données
    if not data.get('is_closed', False):
        return
    
    logger.debug(f"Mise à jour du prix: {symbol} @ {price}")
    
    # Mettre à jour le prix dans la base de données
    # N'effectuer cette opération que périodiquement pour réduire la charge DB
    now = datetime.now()
    minute = now.minute
    
    # N'insérer dans la base de données que toutes les 5 minutes ou pour les chandeliers de 15min+
    timeframe = data.get('timeframe', '1m')
    if timeframe in ['15m', '30m', '1h', '4h', '1d'] or minute % 5 == 0:
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
            logger.error(f"❌ Erreur lors de la mise à jour du prix: {str(e)}")
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
            logger.warning("⚠️ Aucune balance à mettre à jour reçue")
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
        
        logger.info(f"✅ Soldes mis à jour: {len(balances)} actifs, valeur totale: {total_value:.2f} USDC")
        
        # Envoyer une notification de mise à jour
        notify_balance_update(balances, total_value)
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la mise à jour des soldes: {str(e)}")
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
    with cache_lock:
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
                logger.error(f"❌ Erreur lors de la récupération des prix actuels: {str(e)}")
    
    return current_prices

async def update_balances_from_binance():
    """
    Récupère et met à jour les balances depuis Binance.
    Version améliorée avec meilleure gestion d'erreur et retry.
    
    Returns:
        True si la mise à jour a réussi, False sinon
    """
    logger.info("Mise à jour des balances depuis Binance...")
    
    try:
        # Vérifier si les clés API sont configurées
        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            logger.warning("⚠️ Clés API Binance non configurées, impossible de récupérer les balances")
            return False
        
        logger.info(f"Clés API Binance disponibles: API_KEY={BINANCE_API_KEY[:3]}...")

        # Créer le gestionnaire Binance
        account_manager = BinanceAccountManager(
            api_key=BINANCE_API_KEY,
            api_secret=BINANCE_SECRET_KEY
        )
        
        # Récupérer les balances avec retry intégré dans la classe
        try:
            binance_balances = account_manager.get_balances()
            if not binance_balances:
                logger.warning("⚠️ Aucune balance reçue depuis Binance")
                return False
        except BinanceApiError as e:
            logger.error(f"❌ Erreur API Binance: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur inattendue: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
        logger.info(f"✅ {len(binance_balances)} balances récupérées depuis Binance")
        
        # Convertir en AssetBalance
        from shared.src.schemas import AssetBalance
        asset_balances = []
        
        # Prix actuel des actifs pour l'évaluation en USDC
        prices = get_current_prices()
        
        # Si on n'a pas les prix ou pas assez, essayer de les récupérer via l'API Binance
        if not prices or len(prices) < 20:
            try:
                prices = account_manager.get_ticker_prices()
                # Mettre à jour le cache global avec lock pour thread-safety
                with cache_lock:
                    for symbol, price in prices.items():
                        if symbol.endswith('USDC'):
                            current_prices[symbol] = price
            except Exception as e:
                logger.error(f"❌ Erreur lors de la récupération des prix Binance: {str(e)}")
        
        # Utiliser la méthode optimisée pour calculer toutes les valeurs USDC
        try:
            valued_balances = account_manager.calculate_asset_values()
            
            # Convertir en objets AssetBalance
            for balance in valued_balances:
                asset_balance = AssetBalance(
                    asset=balance["asset"],
                    free=balance["free"],
                    locked=balance["locked"],
                    total=balance["total"],
                    value_usdc=balance.get("value_usdc")
                )
                asset_balances.append(asset_balance)
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul des valeurs en USDC: {str(e)}")
            
            # Fallback: conversion basique sans valeurs USDC
            for balance in binance_balances:
                asset_balance = AssetBalance(
                    asset=balance["asset"],
                    free=balance["free"],
                    locked=balance["locked"],
                    total=balance["total"],
                    value_usdc=None
                )
                asset_balances.append(asset_balance)
        
        # Mettre à jour les balances dans la base de données
        db = DBManager()
        portfolio = PortfolioModel(db_manager=db)
        success = portfolio.update_balances(asset_balances)
        
        if not success:
            logger.error("❌ Échec de la mise à jour des balances dans la base de données")
            portfolio.close()
            db.close()
            return False
        
        # Récupérer le résumé pour la notification
        summary = portfolio.get_portfolio_summary()
        total_value = summary.total_value
        
        # Mise à jour des poches
        try:
            pocket_manager = PocketManager(db_manager=db)
            pocket_success = pocket_manager.update_pockets_allocation(total_value)
            if not pocket_success:
                logger.warning("⚠️ Problème lors de la mise à jour des poches")
            pocket_manager.close()
        except Exception as e:
            logger.error(f"❌ Erreur lors de la mise à jour des poches: {str(e)}")
        
        portfolio.close()
        db.close()

        # Publier sur Redis
        try:
            redis = RedisClient()
            
            # ✅ Ici on publie sur Redis en live (pubsub)
            redis.publish("roottrading:account:balances", asset_balances)

            # ✅ Ici on écrit durablement dans Redis
            redis.hset('account:balances', mapping={b.asset: b.total for b in asset_balances})
            total_balance = sum(b.value_usdc or 0 for b in asset_balances)
            redis.set('account:total_balance', total_balance)
            logger.info(f"✅ Balances enregistrées durablement dans Redis. Total estimé: {total_balance:.2f} USDC")
            
            # Envoyer une notification de mise à jour
            notify_balance_update(asset_balances, total_balance)
            
            # Fermer la connexion Redis
            redis.close()
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'enregistrement dans Redis: {str(e)}")
        
        return True

    except Exception as e:
        logger.error(f"❌ Erreur non gérée lors de la mise à jour des balances: {str(e)}")
        logger.error(traceback.format_exc())
        return False

async def binance_sync_worker():
    """
    Worker asynchrone pour la synchronisation avec Binance.
    Cette version utilise asyncio pour gérer les opérations asynchrones.
    """
    logger.info("🔄 Worker de synchronisation Binance démarré")
    
    while not stop_event.is_set():
        try:
            # Mettre à jour les balances
            success = await update_balances_from_binance()
            
            if success:
                logger.info("✅ Synchronisation Binance réussie")
                # Attendre l'intervalle normal
                await asyncio.sleep(300)  # 5 minutes par défaut
            else:
                # Attendre moins longtemps en cas d'échec
                logger.warning("⚠️ Échec de synchronisation Binance, nouvelle tentative dans 60 secondes")
                await asyncio.sleep(60)
                
        except asyncio.CancelledError:
            logger.info("🛑 Worker de synchronisation Binance annulé")
            break
        except Exception as e:
            logger.error(f"❌ Erreur dans le worker Binance: {str(e)}")
            logger.error(traceback.format_exc())
            await asyncio.sleep(60)
    
    logger.info("🛑 Worker de synchronisation Binance arrêté")

def binance_sync_task(interval: int):
    """
    Tâche périodique pour synchroniser les balances avec Binance.
    Version améliorée avec exécution asynchrone.
    
    Args:
        interval: Intervalle entre les synchronisations en secondes
    """
    logger.info(f"Démarrage de la tâche de synchronisation Binance (intervalle: {interval}s)")
    
    # Créer et configurer une nouvelle boucle d'événements
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Première synchronisation au démarrage
    try:
        loop.run_until_complete(update_balances_from_binance())
    except Exception as e:
        logger.error(f"❌ Erreur lors de la synchronisation initiale avec Binance: {str(e)}")
    
    # Créer la tâche de synchronisation périodique
    worker_task = None
    
    try:
        # Démarrer le worker asynchrone
        worker_task = loop.create_task(binance_sync_worker())
        
        # Exécuter la boucle d'événements jusqu'à ce que stop_event soit défini
        while not stop_event.is_set():
            loop.run_until_complete(asyncio.sleep(1))
            
    except KeyboardInterrupt:
        logger.info("Interruption clavier détectée dans la tâche Binance")
    except Exception as e:
        logger.error(f"❌ Erreur dans la tâche de synchronisation Binance: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Annuler la tâche worker si elle existe
        if worker_task and not worker_task.done():
            worker_task.cancel()
            try:
                loop.run_until_complete(worker_task)
            except asyncio.CancelledError:
                pass
            
        # Fermer la boucle d'événements
        loop.close()
        
    logger.info("Tâche de synchronisation Binance arrêtée")

async def db_sync_worker():
    """
    Worker asynchrone pour la synchronisation avec la base de données.
    """
    logger.info("🔄 Worker de synchronisation DB démarré")
    sync_failures = 0
    
    while not stop_event.is_set():
        try:
            # Synchroniser les poches avec les trades actifs
            logger.info("Synchronisation des poches...")
            pocket_manager = PocketManager()
            sync_success = pocket_manager.sync_with_trades()
            
            # Mettre à jour les statistiques quotidiennes
            logger.info("Mise à jour des statistiques...")
            db = DBManager()
            db.execute_query(
                "CALL update_daily_stats(%s)",
                (datetime.now().date(),),
                commit=True
            )
            db.close()
            
            # Récupérer la valeur totale du portefeuille pour envoyer la notification
            portfolio = PortfolioModel()
            summary = portfolio.get_portfolio_summary()
            total_value = summary.total_value
            
            # Notifier de la synchronisation
            if sync_success:
                sync_failures = 0
                logger.info("✅ Synchronisation DB terminée avec succès")
                
                # Envoyer notification de mise à jour
                notify_balance_update(summary.balances, total_value)
                
                # Attendre l'intervalle normal
                await asyncio.sleep(300)  # 5 minutes par défaut
            else:
                sync_failures += 1
                logger.warning(f"⚠️ Échec de la synchronisation DB (tentative {sync_failures})")
                
                # Si plusieurs échecs consécutifs, essayer une réconciliation complète
                if sync_failures >= 3:
                    logger.warning(f"⚠️ {sync_failures} échecs consécutifs, tentative de réconciliation complète...")
                    try:
                        pocket_manager.recalculate_active_cycles()
                        pocket_manager.update_pockets_allocation(total_value)
                        logger.info("✅ Réconciliation complète effectuée")
                        sync_failures = 0
                    except Exception as e:
                        logger.error(f"❌ Échec de la réconciliation complète: {str(e)}")
                
                # Attendre moins longtemps en cas d'échec
                await asyncio.sleep(60)
            
            pocket_manager.close()
            portfolio.close()
            
        except asyncio.CancelledError:
            logger.info("🛑 Worker de synchronisation DB annulé")
            break
        except Exception as e:
            logger.error(f"❌ Erreur dans le worker DB: {str(e)}")
            logger.error(traceback.format_exc())
            sync_failures += 1
            await asyncio.sleep(60)
    
    logger.info("🛑 Worker de synchronisation DB arrêté")

def db_sync_task(interval: int):
    """
    Tâche périodique pour synchroniser les poches avec les trades actifs dans la base de données.
    Version améliorée avec exécution asynchrone.
    
    Args:
        interval: Intervalle entre les synchronisations en secondes
    """
    logger.info(f"Démarrage de la tâche de synchronisation DB (intervalle: {interval}s)")
    
    # Créer et configurer une nouvelle boucle d'événements
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Première synchronisation au démarrage
    try:
        # Synchroniser les poches avec les trades actifs
        pocket_manager = PocketManager()
        sync_success = pocket_manager.sync_with_trades()
        pocket_manager.close()
        
        if sync_success:
            logger.info("✅ Synchronisation DB initiale réussie")
        else:
            logger.warning("⚠️ Échec de la synchronisation DB initiale")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la synchronisation DB initiale: {str(e)}")
    
    # Créer la tâche de synchronisation périodique
    worker_task = None
    
    try:
        # Démarrer le worker asynchrone
        worker_task = loop.create_task(db_sync_worker())
        
        # Exécuter la boucle d'événements jusqu'à ce que stop_event soit défini
        while not stop_event.is_set():
            loop.run_until_complete(asyncio.sleep(1))
            
    except KeyboardInterrupt:
        logger.info("Interruption clavier détectée dans la tâche DB")
    except Exception as e:
        logger.error(f"❌ Erreur dans la tâche de synchronisation DB: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Annuler la tâche worker si elle existe
        if worker_task and not worker_task.done():
            worker_task.cancel()
            try:
                loop.run_until_complete(worker_task)
            except asyncio.CancelledError:
                pass
            
        # Fermer la boucle d'événements
        loop.close()
        
    logger.info("Tâche de synchronisation DB arrêtée")

def create_tables_if_needed():
    """
    Vérifie et crée les tables nécessaires si elles n'existent pas déjà.
    Version optimisée avec transaction unique et gestion des tables supplémentaires.
    """
    try:
        db = DBManager()
        
        # Démarrer une transaction unique
        db.execute_query("BEGIN")
        
        # Liste des tables à vérifier et créer
        tables = {
            'portfolio_balances': """
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
                CREATE INDEX IF NOT EXISTS portfolio_balances_asset_timestamp_idx ON portfolio_balances(asset, timestamp);
            """,
            
            'capital_pockets': """
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
                
                CREATE INDEX IF NOT EXISTS capital_pockets_type_idx ON capital_pockets(pocket_type);
            """,
            
            'pocket_transactions': """
                CREATE TABLE IF NOT EXISTS pocket_transactions (
                    id SERIAL PRIMARY KEY,
                    pocket_type VARCHAR(20) NOT NULL,
                    transaction_type VARCHAR(20) NOT NULL,
                    amount NUMERIC(24, 8) NOT NULL,
                    cycle_id VARCHAR(36) NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    CONSTRAINT unique_pocket_cycle_transaction UNIQUE (pocket_type, cycle_id, transaction_type)
                );
                
                CREATE INDEX IF NOT EXISTS pocket_transactions_cycle_idx ON pocket_transactions(cycle_id);
                CREATE INDEX IF NOT EXISTS pocket_transactions_type_idx ON pocket_transactions(transaction_type);
            """,
            
            'market_data': """
                CREATE TABLE IF NOT EXISTS market_data (
                    time TIMESTAMP NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    open NUMERIC(24, 8) NOT NULL,
                    high NUMERIC(24, 8) NOT NULL,
                    low NUMERIC(24, 8) NOT NULL,
                    close NUMERIC(24, 8) NOT NULL,
                    volume NUMERIC(24, 8) NOT NULL,
                    PRIMARY KEY (time, symbol)
                );
                
                CREATE INDEX IF NOT EXISTS market_data_symbol_idx ON market_data(symbol);
                CREATE INDEX IF NOT EXISTS market_data_time_idx ON market_data(time);
            """
        }
        
        # Vérifier et créer chaque table
        for table_name, create_query in tables.items():
            check_query = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{table_name}'
            );
            """
            
            result = db.execute_query(check_query, fetch_one=True, commit=False)
            
            if not result or not result.get('exists', False):
                logger.warning(f"Table {table_name} non trouvée, création en cours...")
                
                # Créer la table
                db.execute_query(create_query, commit=False)
                logger.info(f"✅ Table {table_name} créée avec succès")
        
        # Valider toutes les créations de tables
        db.execute_query("COMMIT")
        
        # Vérifier si la procédure update_daily_stats existe
        check_proc_query = """
        SELECT EXISTS (
            SELECT FROM pg_proc 
            WHERE proname = 'update_daily_stats'
        );
        """
        
        result = db.execute_query(check_proc_query, fetch_one=True)
        
        if not result or not result.get('exists', False):
            logger.warning("Procédure update_daily_stats non trouvée, création en cours...")
            
            # Créer la procédure
            create_proc_query = """
            CREATE OR REPLACE PROCEDURE update_daily_stats(p_date DATE)
            LANGUAGE plpgsql
            AS $$
            BEGIN
                -- Insérer ou mettre à jour les statistiques quotidiennes
                -- Code de la procédure ici
                
                -- Exemple simple (à adapter selon les besoins réels)
                INSERT INTO daily_stats (date, total_value, trade_count)
                SELECT 
                    p_date, 
                    (SELECT SUM(value_usdc) FROM portfolio_balances WHERE DATE(timestamp) = p_date),
                    (SELECT COUNT(*) FROM trade_cycles WHERE DATE(created_at) = p_date)
                ON CONFLICT (date) DO UPDATE
                SET 
                    total_value = EXCLUDED.total_value,
                    trade_count = EXCLUDED.trade_count;
                    
                -- Assurez-vous que la table daily_stats existe
                EXCEPTION WHEN undefined_table THEN
                    -- Créer la table si elle n'existe pas
                    CREATE TABLE daily_stats (
                        date DATE PRIMARY KEY,
                        total_value NUMERIC(24, 8),
                        trade_count INTEGER,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                    
                    -- Réessayer
                    INSERT INTO daily_stats (date, total_value, trade_count)
                    SELECT 
                        p_date, 
                        (SELECT SUM(value_usdc) FROM portfolio_balances WHERE DATE(timestamp) = p_date),
                        (SELECT COUNT(*) FROM trade_cycles WHERE DATE(created_at) = p_date);
            END;
            $$;
            """
            
            db.execute_query(create_proc_query, commit=True)
            logger.info("✅ Procédure update_daily_stats créée avec succès")
        
        db.close()
        return True
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la vérification/création des tables: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Essayer d'annuler la transaction en cas d'erreur
        try:
            db.execute_query("ROLLBACK")
        except:
            pass
            
        if db:
            db.close()
            
        return False

def subscribe_to_redis_channels():
    """
    S'abonne aux canaux Redis pour recevoir les données de marché et les soldes.
    
    Returns:
        Client Redis ou None en cas d'erreur
    """
    try:
        redis_client = RedisClient()
        
        # S'abonner aux données de marché
        market_channels = [f"roottrading:market:data:{symbol.lower()}" for symbol in SYMBOLS]
        redis_client.subscribe(market_channels, handle_market_data)
        logger.info(f"✅ Abonné aux canaux de données de marché: {len(market_channels)} canaux")
        
        # S'abonner aux mises à jour de solde
        balance_channel = "roottrading:account:balances"
        redis_client.subscribe(balance_channel, handle_balance_update)
        logger.info(f"✅ Abonné au canal des soldes: {balance_channel}")
        
        return redis_client
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'abonnement aux canaux Redis: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def initialize_default_balances():
    """
    Initialise les balances avec des valeurs par défaut si aucune balance n'existe.
    
    Returns:
        True si l'initialisation a réussi ou n'était pas nécessaire, False sinon
    """
    try:
        db = DBManager()
        portfolio = PortfolioModel(db_manager=db)
        
        # Vérifier si des balances existent
        balances = portfolio.get_latest_balances()
        
        # Si aucune balance n'existe, créer des valeurs par défaut
        if not balances:
            logger.info("⚠️ Aucune balance trouvée, initialisation avec des valeurs par défaut...")
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
        logger.error(f"❌ Erreur lors de l'initialisation des balances: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def initialize_system():
    """
    Initialise le système: tables DB, valeurs par défaut, etc.
    
    Returns:
        True si l'initialisation a réussi, False sinon
    """
    # Vérifier et créer les tables si nécessaire
    if not create_tables_if_needed():
        logger.error("❌ Impossible de créer les tables nécessaires")
        return False
    
    # Initialiser les balances par défaut si nécessaire
    if not initialize_default_balances():
        logger.warning("⚠️ Initialisation des balances par défaut échouée, poursuite avec prudence")
    
    # Vérifier la connexion Redis si activée
    try:
        redis = RedisClient()
        redis.ping()
        redis.close()
        logger.info("✅ Connexion Redis vérifiée")
    except Exception as e:
        logger.warning(f"⚠️ Redis non disponible: {str(e)}")
    
    return True

async def startup_tasks():
    """
    Tâches à exécuter au démarrage de l'API.
    Cette fonction est appelée par le lifespan de FastAPI.
    """
    # Synchronisation initiale
    try:
        success = await update_balances_from_binance()
        if success:
            logger.info("✅ Synchronisation Binance initiale réussie")
        else:
            logger.warning("⚠️ Échec de la synchronisation Binance initiale")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la synchronisation initiale: {str(e)}")
    
    # Démarrer les tâches périodiques
    asyncio.create_task(periodic_balance_update())
    logger.info("✅ Tâches de démarrage terminées")

async def periodic_balance_update():
    """
    Met à jour périodiquement les soldes et les poches.
    Version améliorée avec gestion des erreurs et backoff.
    """
    update_count = 0
    error_count = 0
    
    while True:
        try:
            # Vérifier si on doit s'arrêter
            if stop_event.is_set():
                logger.info("🛑 Arrêt de la mise à jour périodique")
                break
                
            # Réduire la fréquence à 30 secondes pour être plus réactif
            update_count += 1
            
            # À chaque 2ème mise à jour (60 secondes), faire une mise à jour complète
            if update_count % 2 == 0:
                logger.info("Mise à jour périodique complète des balances...")
                await update_balances_from_binance()
            
            # Synchroniser les poches
            db = DBManager()
            pocket_manager = PocketManager(db_manager=db)
            pocket_manager.sync_with_trades()
            
            # Récupérer les données pour la notification
            portfolio = PortfolioModel(db_manager=db)
            summary = portfolio.get_portfolio_summary()
            total_value = summary.total_value
            
            # Envoyer une notification de mise à jour même sans changement
            # Cela permet au coordinator de rester informé
            notify_balance_update(summary.balances, total_value)
            
            portfolio.close()
            pocket_manager.close()
            db.close()
            
            logger.info("✅ Mise à jour périodique terminée")
            
            # Réinitialiser le compteur d'erreurs après un succès
            error_count = 0
            
        except asyncio.CancelledError:
            logger.info("🛑 Tâche périodique annulée")
            break
        except Exception as e:
            error_count += 1
            logger.error(f"❌ Erreur lors de la mise à jour périodique (#{error_count}): {str(e)}")
            logger.error(traceback.format_exc())
            
            # Backoff exponentiel en cas d'erreurs répétées
            if error_count > 3:
                wait_time = min(30 * (2 ** (error_count - 3)), 300)  # Max 5 minutes
                logger.warning(f"⚠️ Trop d'erreurs consécutives, attente de {wait_time}s avant nouvelle tentative")
                await asyncio.sleep(wait_time)
        
        # Attendre 30 secondes avant la prochaine mise à jour
        await asyncio.sleep(30)

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
    Version améliorée avec meilleure gestion d'erreur et des vérifications régulières.
    """
    # Parser les arguments
    args = parse_arguments()
    
    # Si mode debug, augmenter le niveau de log
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🐞 Mode debug activé")
    
    # Configurer les gestionnaires de signaux
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    logger.info("🚀 Démarrage du service Portfolio RootTrading...")
    
    # Initialiser le système
    if not initialize_system():
        logger.error("❌ Échec de l'initialisation du système, arrêt du service")
        return
    
    # S'abonner aux canaux Redis si activé
    redis_client = None
    if args.redis_enabled:
        redis_client = subscribe_to_redis_channels()
    
    # Configurer le lifespan de FastAPI pour les tâches de démarrage/arrêt
    app.add_event_handler("startup", startup_tasks)
    
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
        
        # Configuration de uvicorn avec options supplémentaires
        uvicorn_config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            workers=1,
            limit_concurrency=100,  # Limiter la concurrence
            timeout_keep_alive=65  # Fermer les connexions inactives après 65s
        )
        
        # Démarrer le serveur
        server = uvicorn.Server(uvicorn_config)
        server.run()
    
    except KeyboardInterrupt:
        logger.info("Interruption clavier détectée")
    except Exception as e:
        logger.error(f"❌ Erreur critique: {str(e)}")
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
                    logger.warning(f"⚠️ Le thread {thread.name} ne s'est pas arrêté proprement")
        
        # Fermer la connexion Redis
        if redis_client:
            redis_client.close()
            
        # Vider explicitement les caches
        SharedCache.clear()
        
        logger.info("✅ Service Portfolio arrêté avec succès")

if __name__ == "__main__":
    main()