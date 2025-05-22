"""
Définition des routes API pour l'application trader.
Sépare les routes de leur implémentation pour plus de clarté.
"""
import logging
import time
import json
import hmac
import hashlib
from flask import request, jsonify, Blueprint, current_app
import requests

from shared.src.config import BINANCE_SECRET_KEY, TRADING_MODE
from shared.src.enums import OrderSide
from shared.src.db_pool import get_db_metrics

# Configuration du logging
logger = logging.getLogger(__name__)

# Créer un Blueprint pour organiser les routes
routes_bp = Blueprint('api_routes', __name__)

# ============================================================================
# Routes de base
# ============================================================================

@routes_bp.route('/health', methods=['GET'])
def health_check():
    """
    Point de terminaison pour vérifier l'état du service.
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - order_manager.start_time if hasattr(order_manager, 'start_time') else 0,
        "mode": TRADING_MODE,
        "symbols": order_manager.symbols
    })

@routes_bp.route('/diagnostic', methods=['GET'])
def diagnostic():
    """
    Point de terminaison pour le diagnostic du service.
    Fournit des informations complètes sur l'état du système.
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    # Récupérer les cycles actifs
    active_cycles = order_manager.get_active_cycles()
    
    # Collecter les informations sur les derniers prix
    last_prices = order_manager.last_prices.copy() if hasattr(order_manager, 'last_prices') else {}
    
    # Vérifier l'état des connexions
    redis_status = "connected" if hasattr(order_manager.signal_processor.redis_client, 'connection') else "disconnected"
    
    # Vérifier l'état d'exécution des threads
    thread_alive = hasattr(order_manager.signal_processor, 'processing_thread') and order_manager.signal_processor.processing_thread.is_alive()
    
    # Construire la réponse
    diagnostic_info = {
        "status": "operational",
        "timestamp": time.time(),
        "mode": TRADING_MODE,
        "symbols": order_manager.symbols,
        "cycles": {
            "active_count": len(active_cycles),
            "cycles": [
                {
                    "id": cycle.id,
                    "symbol": cycle.symbol,
                    "strategy": cycle.strategy,
                    "status": cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status),
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
        "signal_stats": order_manager.signal_processor.get_stats() if hasattr(order_manager.signal_processor, 'get_stats') else {},
        "memory_usage_mb": current_app.config.get('MEMORY_USAGE', 0)
    }
    
    # Ajouter des statistiques sur la file d'attente
    if hasattr(order_manager.signal_processor, 'signal_queue'):
        diagnostic_info["queue_size"] = order_manager.signal_processor.signal_queue.qsize()
    
    # AJOUT DES MÉTRIQUES DE BASE DE DONNÉES
    try:        
        diagnostic_info["database"] = get_db_metrics()
        
        # Ajouter les informations de réconciliation
        if hasattr(order_manager, 'reconciliation_service'):
            diagnostic_info["reconciliation"] = order_manager.reconciliation_service.get_stats()
            
    except Exception as e:
        logger.warning(f"Impossible de récupérer les métriques de base de données: {str(e)}")
        diagnostic_info["database"] = {"status": "unavailable", "error": str(e)}
    
    return jsonify(diagnostic_info)

# ============================================================================
# Routes des ordres
# ============================================================================

@routes_bp.route('/orders', methods=['GET'])
def get_orders():
    """
    Récupère les ordres actifs.
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    active_orders = order_manager.get_active_orders()
    return jsonify(active_orders)

@routes_bp.route('/order', methods=['POST'])
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
    order_manager = current_app.config['ORDER_MANAGER']
    
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
        if isinstance(result, str) and result.startswith("cycle_"):
            return jsonify({"order_id": result, "status": "created"}), 201
        else:
            return jsonify({"error": str(result)}), 400
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création de l'ordre: {str(e)}")
        return jsonify({"error": str(e)}), 500

@routes_bp.route('/order/<order_id>', methods=['DELETE'])
def cancel_order(order_id):
    """
    Annule un ordre existant.
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
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

@routes_bp.route('/close/<cycle_id>', methods=['POST'])
def close_cycle(cycle_id):
    """
    Ferme un cycle de trading.
    
    Exemple de requête:
    {
        "price": 50000  # optionnel
    }
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
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

# ============================================================================
# Routes de configuration
# ============================================================================

@routes_bp.route('/config/pause', methods=['POST'])
def pause_trading():
    """
    Met en pause les transactions pour un symbole ou une stratégie spécifique.
    
    Exemple de requête:
    {
        "symbol": "BTCUSDC",  # optionnel
        "strategy": "Bollinger_Strategy",  # optionnel
        "duration": 3600  # durée en secondes, optionnel
    }
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        data = request.json or {}
        
        symbol = data.get("symbol")
        strategy = data.get("strategy")
        duration = data.get("duration", 0)  # 0 = pause indéfinie
        
        # Appliquer la pause
        if hasattr(order_manager, 'pause_symbol') and symbol:
            order_manager.pause_symbol(symbol, duration)
            logger.info(f"⏸️ Trading en pause pour le symbole {symbol}")
            
        if hasattr(order_manager, 'pause_strategy') and strategy:
            order_manager.pause_strategy(strategy, duration)
            logger.info(f"⏸️ Trading en pause pour la stratégie {strategy}")
            
        if not symbol and not strategy and hasattr(order_manager, 'pause_all'):
            order_manager.pause_all(duration)
            logger.info("⏸️ Trading en pause pour tous les symboles et stratégies")
        
        return jsonify({
            "status": "paused",
            "symbol": symbol,
            "strategy": strategy,
            "duration": duration,
            "until": time.time() + duration if duration > 0 else None
        })
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la mise en pause: {str(e)}")
        return jsonify({"error": str(e)}), 500

@routes_bp.route('/config/resume', methods=['POST'])
def resume_trading():
    """
    Reprend les transactions pour un symbole ou une stratégie spécifique.
    
    Exemple de requête:
    {
        "symbol": "BTCUSDC",  # optionnel
        "strategy": "Bollinger_Strategy"  # optionnel
    }
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        data = request.json or {}
        
        symbol = data.get("symbol")
        strategy = data.get("strategy")
        
        # Reprendre le trading
        if hasattr(order_manager, 'resume_symbol') and symbol:
            order_manager.resume_symbol(symbol)
            logger.info(f"▶️ Trading repris pour le symbole {symbol}")
            
        if hasattr(order_manager, 'resume_strategy') and strategy:
            order_manager.resume_strategy(strategy)
            logger.info(f"▶️ Trading repris pour la stratégie {strategy}")
            
        if not symbol and not strategy and hasattr(order_manager, 'resume_all'):
            order_manager.resume_all()
            logger.info("▶️ Trading repris pour tous les symboles et stratégies")
        
        return jsonify({
            "status": "resumed",
            "symbol": symbol,
            "strategy": strategy
        })
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la reprise: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ============================================================================
# Routes de réconciliation
# ============================================================================

@routes_bp.route('/reconcile', methods=['POST'])
def reconcile_cycles():
    """
    Force une réconciliation des cycles avec l'état sur Binance.
    
    Exemple de requête:
    {
        "force": true  # optionnel, défaut: true
    }
    
    Returns:
        Résultat de la réconciliation au format JSON
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        # Récupérer les options de la requête
        data = request.json or {}
        force = data.get('force', True)
        
        # Forcer une réconciliation
        logger.info(f"Réconciliation manuelle déclenchée (force={force})")
        
        # Appeler le service de réconciliation
        order_manager.reconciliation_service.reconcile_all_cycles(force=force)
        
        # Récupérer les statistiques après la réconciliation
        stats = order_manager.reconciliation_service.get_stats()
        
        return jsonify({
            "success": True,
            "message": f"Réconciliation effectuée: {stats['cycles_reconciled']}/{stats['cycles_checked']} cycles mis à jour",
            "stats": stats
        })
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la réconciliation manuelle: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        }), 500

@routes_bp.route('/reconcile/status', methods=['GET'])
def get_reconciliation_status():
    """
    Récupère l'état de la dernière réconciliation.
    
    Returns:
        Statistiques de réconciliation au format JSON
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        # Récupérer les statistiques
        stats = order_manager.reconciliation_service.get_stats()
        
        return jsonify({
            "success": True,
            "stats": stats
        })
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération du statut de réconciliation: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        }), 500

# ============================================================================
# Routes de diagnostic
# ============================================================================

@routes_bp.route('/diag/binance', methods=['GET'])
def diagnostic_binance():
    """
    Effectue un diagnostic de la connexion Binance.
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        # Extraire les premières/dernières lettres de la clé pour vérification sans exposer toute la clé
        api_key = order_manager.binance_executor.api_key
        api_key_preview = f"{api_key[:4]}...{api_key[-4:]}" if api_key else "Non définie"
        
        # Vérifier si la clé API a un format valide
        api_key_valid = bool(api_key and len(api_key) > 8)
        
        # Tester une requête public (ne nécessite pas d'authentification)
        ping_success = False
        ping_error = None
        try:
            ping_url = f"{order_manager.binance_executor.BASE_URL}/api/v3/ping"
            response = requests.get(ping_url, timeout=5)
            ping_success = response.status_code == 200
        except Exception as e:
            ping_error = str(e)
        
        # Tester une requête privée (nécessite authentification)
        auth_success = False
        auth_error = None
        try:
            # Utiliser les utilitaires de l'exécuteur pour tester l'authentification
            utils = order_manager.binance_executor.utils
            timestamp = int(time.time() * 1000)
            params = {"timestamp": timestamp}
            signature = utils.generate_signature(params)
            
            # Construire l'URL
            account_url = f"{order_manager.binance_executor.BASE_URL}/api/v3/account"
            params["signature"] = signature
            
            # Effectuer la requête
            headers = {"X-MBX-APIKEY": api_key}
            response = request.session.get(account_url, params=params, headers=headers, timeout=5)
            auth_success = response.status_code == 200
            if not auth_success:
                auth_error = response.text
        except Exception as e:
            auth_error = str(e)
        
        return jsonify({
            "api_key_preview": api_key_preview,
            "api_key_valid": api_key_valid,
            "ping_test": {
                "success": ping_success,
                "error": ping_error
            },
            "auth_test": {
                "success": auth_success,
                "error": auth_error
            },
            "demo_mode": order_manager.binance_executor.demo_mode
        })
    
    except Exception as e:
        logger.error(f"❌ Erreur lors du diagnostic Binance: {str(e)}")
        return jsonify({"error": str(e)}), 500

@routes_bp.route('/api/diagnostic/signature', methods=['GET'])
def diagnostic_binance_signature():
    """
    Endpoint Flask de diagnostic pour tester la génération de signature Binance.
    """
    timestamp = int(time.time() * 1000)
    test_params = {
        "symbol": "BTCUSDC",
        "side": "BUY",
        "type": "LIMIT",
        "quantity": "0.00100",
        "price": "30000.00",
        "timeInForce": "GTC",
        "timestamp": timestamp
    }
    query_string = "&".join([f"{key}={test_params[key]}" for key in sorted(test_params.keys())])
    signature = hmac.new(
        BINANCE_SECRET_KEY.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    return jsonify({
        "query_string": query_string,
        "signature": signature,
        "timestamp": timestamp,
        "message": "Signature générée localement (Flask version)"
    })

# ============================================================================
# Routes de contraintes et symboles
# ============================================================================

@routes_bp.route('/constraints/<symbol>', methods=['GET'])
def get_symbol_constraints(symbol):
    """
    Récupère les contraintes de trading pour un symbole donné.
    """
    try:
        order_manager = current_app.config.get('ORDER_MANAGER')
        if not order_manager:
            return jsonify({"error": "OrderManager non initialisé"}), 500
        
        constraints = order_manager.binance_executor.symbol_constraints
        
        # Récupérer les contraintes
        constraints_data = {
            "symbol": symbol,
            "min_qty": constraints.get_min_qty(symbol),
            "step_size": constraints.get_step_size(symbol),
            "min_notional": constraints.get_min_notional(symbol),
            "price_precision": constraints.get_price_precision(symbol),
            "has_real_time_data": symbol in constraints.symbol_info
        }
        
        # Ajouter les données en temps réel si disponibles
        if symbol in constraints.symbol_info:
            constraints_data["real_time_data"] = constraints.symbol_info[symbol]
        
        return jsonify(constraints_data)
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des contraintes pour {symbol}: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ============================================================================
# Routes de filtrage de marché
# ============================================================================

@routes_bp.route('/test', methods=['GET'])
def test_route():
    """Route de test simple pour vérifier l'enregistrement."""
    return {"status": "ok", "message": "Route de test fonctionne"}

@routes_bp.route('/market/filter/<symbol>', methods=['GET'])
def get_market_filter(symbol):
    """
    Récupère les informations de filtrage de marché pour un symbole donné.
    Utilisé par le coordinator pour adapter les stratégies de trading.
    
    Args:
        symbol: Symbole de trading (ex: ETHBTC, BTCUSDC)
        
    Returns:
        JSON avec les informations de filtrage:
        - mode: Mode de trading ('ride' ou 'react')
        - action: Action recommandée ('normal_trading', 'reduced_trading', 'no_trading')
        - trend_strength: Force de la tendance (0.0 à 1.0)
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        # Valider le symbole
        symbol = symbol.upper()
        if symbol not in order_manager.symbols:
            return jsonify({
                "error": f"Symbole {symbol} non supporté",
                "supported_symbols": order_manager.symbols
            }), 400
        
        # Récupérer le prix actuel
        current_price = order_manager.last_prices.get(symbol, 0)
        
        # Calculer des métriques de base pour le filtrage
        # Pour l'instant, logique simple - peut être étendue avec des indicateurs plus complexes
        mode = 'react'  # Mode par défaut
        action = 'normal_trading'  # Action par défaut
        trend_strength = 0.0  # Force de tendance neutre par défaut
        
        # Logique de base basée sur les cycles actifs pour ce symbole
        active_cycles = [cycle for cycle in order_manager.get_active_cycles() if cycle.symbol == symbol]
        active_count = len(active_cycles)
        
        # Ajuster l'action selon le nombre de cycles actifs
        if active_count >= 3:
            action = 'reduced_trading'  # Réduire le trading si trop de cycles actifs
        elif active_count >= 5:
            action = 'no_trading'  # Arrêter le trading si trop de cycles
        
        # Ajuster le mode selon la volatilité récente (placeholder pour logique future)
        if current_price > 0:
            # Pour l'instant, mode 'react' par défaut
            # Peut être étendu avec des calculs de volatilité, RSI, etc.
            mode = 'react'
        
        return jsonify({
            "symbol": symbol,
            "mode": mode,
            "action": action,
            "trend_strength": trend_strength,
            "current_price": current_price,
            "active_cycles_count": active_count,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération du filtrage pour {symbol}: {str(e)}")
        return jsonify({
            "error": f"Erreur: {str(e)}"
        }), 500

def register_routes(app, order_manager):
    """
    Enregistre les routes dans l'application Flask.
    
    Args:
        app: Application Flask
        order_manager: Gestionnaire d'ordres
    """
    # Stocker le gestionnaire d'ordres dans la configuration de l'application
    app.config['ORDER_MANAGER'] = order_manager
    
    # Stocker l'utilisation mémoire initiale
    import psutil
    app.config['MEMORY_USAGE'] = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    
    # Enregistrer le blueprint
    app.register_blueprint(routes_bp)
    
    logger.info("✅ Routes API enregistrées")