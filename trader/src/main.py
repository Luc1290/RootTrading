"""
Point d'entrée principal pour le microservice Trader.
Démarre le gestionnaire d'ordres et expose une API REST pour les interactions manuelles.
"""
import argparse
import logging
import signal
import sys
import time
import os
import json
import threading
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, Response
import psutil

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import SYMBOLS, LOG_LEVEL, TRADING_MODE
from shared.src.enums import OrderSide

from trader.src.order_manager import OrderManager
process = psutil.Process(os.getpid())

# Configuration du logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trader.log')
    ]
)
logger = logging.getLogger("trader")

# Créer l'application Flask
app = Flask(__name__)

# Instance globale du gestionnaire d'ordres
order_manager = None

# Fonction pour initialiser le gestionnaire d'ordres
def init_order_manager(symbols):
    global order_manager
    order_manager = OrderManager(symbols=symbols)
    return order_manager

@app.route('/health', methods=['GET'])
def health_check():
    """
    Point de terminaison pour vérifier l'état du service.
    """
    return jsonify({
        "status": "ok",
        "timestamp": time.time(),
        "mode": TRADING_MODE,
        "symbols": SYMBOLS
    })

@app.route('/diagnostic', methods=['GET'])
def diagnostic():
    """
    Point de terminaison pour le diagnostic du service.
    Fournit des informations complètes sur l'état du système.
    """
    global order_manager
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    # Récupérer les cycles actifs
    active_cycles = order_manager.get_active_cycles()
    
    # Collecter les informations sur les derniers prix
    last_prices = order_manager.last_prices.copy()
    
    # Vérifier l'état des connexions
    redis_status = "connected" if hasattr(order_manager.redis_client, 'connection') else "disconnected"
    
    # Vérifier l'état d'exécution des threads
    thread_alive = order_manager.processing_thread and order_manager.processing_thread.is_alive()
    
    # Construire la réponse
    diagnostic_info = {
        "status": "operational",
        "timestamp": time.time(),
        "mode": TRADING_MODE,
        "symbols": SYMBOLS,
        "cycles": {
            "active_count": len(active_cycles),
            "cycles": [
                {
                    "id": cycle.id,
                    "symbol": cycle.symbol,
                    "strategy": cycle.strategy,
                    "status": cycle.status.value,
                    "entry_price": cycle.entry_price,
                    "current_price": last_prices.get(cycle.symbol),
                    "quantity": cycle.quantity,
                    "pl_percent": cycle.profit_loss_percent if hasattr(cycle, 'profit_loss_percent') else None,
                    "created_at": cycle.created_at.isoformat() if cycle.created_at else None
                } for cycle in active_cycles
            ]
        },
        "market_data": {
            "prices": {symbol: price for symbol, price in last_prices.items()},
            "update_time": order_manager.last_price_update if hasattr(order_manager, 'last_price_update') else None
        },
        "connections": {
            "redis": redis_status,
            "processing_thread": "running" if thread_alive else "stopped",
            "binance": order_manager.binance_executor.demo_mode and "demo" or "live"
        },
        "queue_size": order_manager.signal_queue.qsize() if hasattr(order_manager, 'signal_queue') else None,
        "memory_usage_mb": round(process.memory_info().rss / (1024 * 1024), 2) if 'process' in globals() else None
    }
    
    return jsonify(diagnostic_info)

@app.route('/orders', methods=['GET'])
def get_orders():
    """
    Récupère les ordres actifs.
    """
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    active_orders = order_manager.get_active_orders()
    return jsonify(active_orders)

@app.route('/order', methods=['POST'])
def create_order():
    """
    Crée un ordre manuel.
    
    Exemple de requête:
    {
        "symbol": "BTCUSDC",
        "side": "BUY",
        "quantity": 0.001,
        "price": 50000  # optionnel
    }
    """
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        data = request.json
        
        # Valider les paramètres requis
        if not all(k in data for k in ["symbol", "side", "quantity"]):
            return jsonify({"error": "Paramètres manquants. Requis: symbol, side, quantity"}), 400
        
        symbol = data["symbol"]
        side = OrderSide(data["side"])  # Conversion en enum
        quantity = float(data["quantity"])
        price = float(data["price"]) if "price" in data else None
        
        # Créer l'ordre
        result = order_manager.create_manual_order(symbol, side, quantity, price)
        
        # Vérifier si c'est un ID ou un message d'erreur
        if result.startswith("cycle_"):
            return jsonify({"order_id": result, "status": "created"}), 201
        else:
            return jsonify({"error": result}), 400
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création de l'ordre: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/order/<order_id>', methods=['DELETE'])
def cancel_order(order_id):
    """
    Annule un ordre existant.
    """
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        # Récupérer les paramètres optionnels
        reason = request.args.get('reason', 'Annulation manuelle via API')
        
        # Annuler le cycle
        success = order_manager.cycle_manager.cancel_cycle(order_id, reason)
        
        if success:
            return jsonify({"status": "canceled", "order_id": order_id}), 200
        else:
            return jsonify({"error": f"Impossible d'annuler l'ordre {order_id}"}), 400
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'annulation de l'ordre: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/close/<cycle_id>', methods=['POST'])
def close_cycle(cycle_id):
    """
    Ferme un cycle de trading.
    
    Exemple de requête:
    {
        "price": 50000  # optionnel
    }
    """
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        data = request.json or {}
        price = float(data["price"]) if "price" in data else None
        
        # Fermer le cycle
        success = order_manager.cycle_manager.close_cycle(cycle_id, price)
        
        if success:
            return jsonify({"status": "closed", "cycle_id": cycle_id}), 200
        else:
            return jsonify({"error": f"Impossible de fermer le cycle {cycle_id}"}), 400
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la fermeture du cycle: {str(e)}")
        return jsonify({"error": str(e)}), 500

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description='Trader RootTrading')
    parser.add_argument(
        '--port', 
        type=int, 
        default=5002, 
        help='Port pour l\'API REST'
    )
    parser.add_argument(
        '--symbols', 
        type=str, 
        default=None, 
        help='Liste de symboles séparés par des virgules (ex: BTCUSDC,ETHUSDC)'
    )
    parser.add_argument(
        '--no-api', 
        action='store_true', 
        help='Désactive l\'API REST'
    )
    return parser.parse_args()

def signal_handler(sig, frame):
    """
    Gestionnaire de signaux pour l'arrêt propre.
    
    Args:
        sig: Type de signal reçu
        frame: Frame actuelle
    """
    logger.info(f"Signal {sig} reçu, arrêt en cours...")
    if order_manager:
        order_manager.stop()
    sys.exit(0)

def main():
    """Fonction principale du service Trader."""
    # Parser les arguments
    args = parse_arguments()
    
    # Configurer les symboles
    symbols = args.symbols.split(',') if args.symbols else SYMBOLS
    
    logger.info("🚀 Démarrage du service Trader RootTrading...")
    logger.info(f"Configuration: {len(symbols)} symboles, "
               f"mode {'DÉMO' if TRADING_MODE.lower() == 'demo' else 'RÉEL'}")
    
    # Configurer les gestionnaires de signaux pour l'arrêt propre
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialiser le gestionnaire d'ordres
        manager = init_order_manager(symbols)
        manager.start()
        
        # Démarrer l'API REST si activée
        if not args.no_api:
            # Démarrer l'API dans un thread séparé
            api_thread = threading.Thread(
                target=lambda: app.run(host='0.0.0.0', port=args.port),
                daemon=True
            )
            api_thread.start()
            logger.info(f"✅ API REST démarrée sur le port {args.port}")
            
            # Maintenir le programme en vie
            while True:
                time.sleep(1)
        else:
            # En mode sans API, simplement attendre
            logger.info("✅ Mode sans API, en attente des signaux")
            while True:
                time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le service Trader: {str(e)}")
    finally:
        # Arrêter le gestionnaire d'ordres proprement
        if order_manager:
            logger.info("Arrêt du service Trader...")
            order_manager.stop()
        
        logger.info("Service Trader terminé")

if __name__ == "__main__":
    main()