"""
Définition des routes API pour l'application trader.
Sépare les routes de leur implémentation pour plus de clarté.
"""
import logging
import time
import json
import hmac
import hashlib
from datetime import datetime
from flask import request, jsonify, Blueprint, current_app
import requests

from shared.src.config import BINANCE_SECRET_KEY, TRADING_MODE
from shared.src.enums import OrderSide, CycleStatus, OrderStatus
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
    
    # Architecture REST uniquement - plus de signal processor
    redis_status = "disabled"  # Plus utilisé depuis l'architecture REST
    
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
            "binance": order_manager.binance_executor.demo_mode and "demo" or "live"
        },
        "signal_stats": {},  # Plus de signal processor
        "memory_usage_mb": current_app.config.get('MEMORY_USAGE', 0)
    }
    
    # Plus de file d'attente de signaux - architecture REST
    
    # AJOUT DES MÉTRIQUES DE BASE DE DONNÉES
    try:        
        diagnostic_info["database"] = get_db_metrics()              
            
    except Exception as e:
        logger.warning(f"Impossible de récupérer les métriques de base de données: {str(e)}")
        diagnostic_info["database"] = {"status": "unavailable", "error": str(e)}
    
    return jsonify(diagnostic_info)

# ============================================================================
# Routes des prix
# ============================================================================

@routes_bp.route('/prices', methods=['GET'])
def get_prices():
    """
    Récupère les prix actuels pour les symboles demandés.
    Query params:
        - symbols: Symboles séparés par des virgules (ex: BTCUSDC,ETHUSDC)
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    symbols_param = request.args.get('symbols', '')
    if not symbols_param:
        return jsonify({"error": "Paramètre 'symbols' requis"}), 400
    
    symbols = [s.strip() for s in symbols_param.split(',')]
    
    # Récupérer les prix depuis le price monitor
    prices = {}
    for symbol in symbols:
        if hasattr(order_manager, 'price_monitor') and order_manager.price_monitor:
            with order_manager.price_monitor.price_lock:
                price = order_manager.price_monitor.last_prices.get(symbol)
                if price:
                    prices[symbol] = price
    
    if not prices:
        return jsonify({"error": f"Aucun prix disponible pour {symbols}"}), 404
    
    return jsonify(prices)

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

@routes_bp.route('/cycles', methods=['GET'])
def get_cycles():
    """
    API centralisée pour récupérer les cycles actifs.
    SEULE SOURCE DE VÉRITÉ pour tous les services.
    
    Query params:
        - symbol: Filtrer par symbole (optionnel)
        - status: Filtrer par statut (optionnel) 
        - confirmed: Filtrer par confirmation (optionnel, défaut: true)
        - include_completed: Inclure les cycles terminés (optionnel, défaut: false)
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        # Récupérer les paramètres de requête
        symbol = request.args.get('symbol')
        status = request.args.get('status')
        confirmed = request.args.get('confirmed', 'true').lower() == 'true'
        include_completed = request.args.get('include_completed', 'false').lower() == 'true'
        
        # Récupérer les cycles depuis le cycle manager (qui sont déjà synchronisés)
        if include_completed:
            # Si on veut tous les cycles, on utilise le repository
            cycles = order_manager.cycle_manager.repository.get_all_cycles()
        else:
            # Sinon on utilise les cycles actifs du cycle manager (déjà filtrés)
            cycles = order_manager.cycle_manager.get_active_cycles()
        
        # Filtrer selon les critères
        filtered_cycles = []
        for cycle in cycles:
            # Si on utilise get_active_cycles(), pas besoin de refiltrer par statut
            if not include_completed and cycle.status in [CycleStatus.COMPLETED, CycleStatus.CANCELED, CycleStatus.FAILED]:
                continue
                
            # Filtrer par confirmation
            if confirmed and not cycle.confirmed:
                continue
                
            # Filtrer par symbole
            if symbol and cycle.symbol != symbol:
                continue
                
            # Filtrer par statut spécifique
            if status and cycle.status.value != status:
                continue
                
            filtered_cycles.append(cycle)
        
        # Convertir en dictionnaires pour la réponse JSON
        result = []
        for cycle in filtered_cycles:
            result.append({
                "id": cycle.id,
                "symbol": cycle.symbol,
                "strategy": cycle.strategy,
                "status": cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status),
                "confirmed": cycle.confirmed,
                "entry_price": cycle.entry_price,
                "quantity": cycle.quantity,
                "stop_price": cycle.stop_price,
                "entry_order_id": cycle.entry_order_id,
                "exit_order_id": cycle.exit_order_id,
                "created_at": cycle.created_at.isoformat() if cycle.created_at else None,
                "updated_at": cycle.updated_at.isoformat() if cycle.updated_at else None
            })
        
        return jsonify({
            "success": True,
            "count": len(result),
            "cycles": result
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des cycles: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@routes_bp.route('/order', methods=['POST'])
def create_order():
    """
    Crée un ordre manuel.
    
    Exemple de requête:
    {
        "symbol": "BTCUSDC",
        "side": "BUY",  # ou "SELL"
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
        strategy = data.get("strategy", "Manual")  # Récupérer la stratégie ou utiliser "Manual" par défaut
        stop_price = float(data["stop_price"]) if "stop_price" in data else None
        
        # Créer l'ordre avec la stratégie et stop price seulement (plus de target avec TrailingStop pur)
        result = order_manager.create_manual_order(symbol, side, quantity, price, strategy, 
                                                  stop_price=stop_price)
        
        # Vérifier si c'est un ID ou un message d'erreur
        if isinstance(result, str) and result.startswith("cycle_"):
            return jsonify({"order_id": result, "status": "created"}), 201
        elif result is None:
            return jsonify({"error": "Ordre non créé - vérifiez les soldes et paramètres"}), 422
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
        
        # Vérifier d'abord si le cycle existe
        cycle = order_manager.cycle_manager.get_cycle(cycle_id)
        if not cycle:
            logger.warning(f"⚠️ Tentative de fermeture d'un cycle inexistant: {cycle_id}")
            return jsonify({
                "error": f"Cycle {cycle_id} not found",
                "cycle_id": cycle_id,
                "status": "not_found"
            }), 404
        
        # Vérifier si le cycle est déjà fermé
        if cycle.status in [CycleStatus.COMPLETED, CycleStatus.CANCELED, CycleStatus.FAILED]:
            logger.info(f"✅ Cycle {cycle_id} déjà fermé avec le statut {cycle.status}")
            return jsonify({
                "status": "already_closed", 
                "cycle_id": cycle_id,
                "cycle_status": cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status)
            }), 200
        
        # Fermer le cycle avec ordre MARKET pour fermeture immédiate
        success = order_manager.cycle_manager.close_cycle(cycle_id, price, force_market=True)
        
        if success:
            return jsonify({"status": "closed", "cycle_id": cycle_id}), 200
        else:
            return jsonify({"error": f"Impossible de fermer le cycle {cycle_id}"}), 400
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la fermeture du cycle: {str(e)}")
        return jsonify({"error": str(e)}), 500

@routes_bp.route('/close_accounting/<cycle_id>', methods=['POST'])
def close_cycle_accounting(cycle_id):
    """
    Ferme un cycle de manière comptable sans ordre réel.
    
    Exemple de requête:
    {
        "price": 141.46,  # prix obligatoire pour le P&L
        "reason": "Retournement BUY"  # optionnel
    }
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        data = request.json or {}
        price = float(data.get("price")) if data.get("price") else None
        reason = data.get("reason", "Fermeture comptable via API")
        
        if not price:
            return jsonify({"error": "Prix obligatoire pour fermeture comptable"}), 400
        
        # Vérifier d'abord si le cycle existe
        cycle = order_manager.cycle_manager.get_cycle(cycle_id)
        if not cycle:
            logger.warning(f"⚠️ Tentative de fermeture comptable d'un cycle inexistant: {cycle_id}")
            return jsonify({
                "error": f"Cycle {cycle_id} not found",
                "cycle_id": cycle_id,
                "status": "not_found"
            }), 404
        
        # Vérifier si le cycle est déjà fermé
        if cycle.status in [CycleStatus.COMPLETED, CycleStatus.CANCELED, CycleStatus.FAILED]:
            logger.info(f"✅ Cycle {cycle_id} déjà fermé avec le statut {cycle.status}")
            return jsonify({
                "status": "already_closed", 
                "cycle_id": cycle_id,
                "cycle_status": cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status)
            }), 200
        
        # Fermer comptablement
        success = order_manager.cycle_manager.close_cycle_accounting(cycle_id, price, reason)
        
        if success:
            return jsonify({"status": "closed_accounting", "cycle_id": cycle_id}), 200
        else:
            return jsonify({"error": f"Impossible de fermer comptablement le cycle {cycle_id}"}), 400
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la fermeture comptable: {str(e)}")
        return jsonify({"error": str(e)}), 500

@routes_bp.route('/reinforce', methods=['POST'])
def reinforce_cycle():
    """
    Renforce un cycle existant (DCA - Dollar Cost Averaging).
    
    Exemple de requête:
    {
        "cycle_id": "cycle_12345",
        "symbol": "BTCUSDC",
        "side": "BUY",
        "quantity": 0.001,
        "price": 50000,  # optionnel
        "metadata": {
            "reinforce_reason": "Prix favorable",
            "confidence": 0.8
        }
    }
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        data = request.json
        
        # Valider les paramètres requis
        if not all(k in data for k in ["cycle_id", "symbol", "side", "quantity"]):
            return jsonify({"error": "Paramètres manquants. Requis: cycle_id, symbol, side, quantity"}), 400
        
        cycle_id = data["cycle_id"]
        symbol = data["symbol"]
        side = OrderSide(data["side"])
        quantity = float(data["quantity"])
        price = float(data["price"]) if "price" in data else None
        metadata = data.get("metadata", {})
        
        # Récupérer le cycle existant
        cycle = order_manager.cycle_manager.get_cycle(cycle_id)
        if not cycle:
            return jsonify({"error": f"Cycle {cycle_id} non trouvé"}), 404
        
        # Vérifier que le cycle est dans un état approprié
        if cycle.status not in [CycleStatus.WAITING_SELL, CycleStatus.ACTIVE_BUY, CycleStatus.ACTIVE_SELL]:
            status_str = cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status)
            return jsonify({
                "error": f"Le cycle {cycle_id} n'est pas dans un état permettant le renforcement (état actuel: {status_str})"
            }), 400
        
        # Vérifier que les paramètres correspondent au cycle
        if cycle.symbol != symbol:
            return jsonify({"error": f"Le symbole {symbol} ne correspond pas au cycle (symbole du cycle: {cycle.symbol})"}), 400
        
        # Normaliser les sides pour la comparaison
        cycle_side_str = cycle.side.value if hasattr(cycle.side, 'value') else str(cycle.side)
        side_str = side.value if hasattr(side, 'value') else str(side)
        
        if cycle_side_str != side_str:
            return jsonify({"error": f"Le side {side_str} ne correspond pas au cycle (side du cycle: {cycle_side_str})"}), 400
        
        # Pour un renforcement, on doit créer un ordre directement via Binance
        # et mettre à jour le cycle existant
        try:
            # Préparer l'ordre pour Binance avec les mêmes paramètres que les ordres normaux
            from shared.src.schemas import TradeOrder
            
            # Créer l'ordre avec les bons paramètres
            reinforce_order = TradeOrder(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                # Limiter à 36 caractères pour Binance
                client_order_id=f"rf_{cycle_id[-12:]}_{int(time.time() % 1000000)}",
                strategy=f"{cycle.strategy}_DCA"
            )
            
            # Exécuter directement via BinanceExecutor
            execution = order_manager.binance_executor.execute_order(reinforce_order)
            
            if execution and (execution.status == OrderStatus.FILLED or execution.status == OrderStatus.NEW):
                order_id = execution.order_id
                executed_price = execution.price if execution.status == OrderStatus.FILLED else price
            
            # Mettre à jour le cycle avec les nouvelles informations
            # Calculer le nouveau prix moyen et la nouvelle quantité
                old_value = cycle.entry_price * cycle.quantity
                new_value = (executed_price or price) * quantity
                new_quantity = cycle.quantity + quantity
                new_avg_price = (old_value + new_value) / new_quantity if new_quantity > 0 else 0
            
                # Mettre à jour le cycle dans la base de données
                order_manager.cycle_manager.update_cycle_reinforcement(
                    cycle_id=cycle_id,
                    additional_quantity=quantity,
                    new_avg_price=new_avg_price,
                    reinforce_order_id=order_id,
                    metadata=metadata
                )
                
                logger.info(f"✅ Cycle {cycle_id} renforcé avec {quantity} unités. Nouveau prix moyen: {new_avg_price}")
                
                return jsonify({
                    "success": True,
                    "cycle_id": cycle_id,
                    "reinforce_order_id": order_id,
                    "original_entry_price": cycle.entry_price,
                    "original_quantity": cycle.quantity,
                    "additional_quantity": quantity,
                    "new_avg_price": new_avg_price,
                    "new_total_quantity": new_quantity
                }), 200
            else:
                error_msg = "Ordre non exécuté"
                if execution:
                    error_msg = f"Ordre en statut {execution.status.value if hasattr(execution.status, 'value') else execution.status}"
                return jsonify({
                    "error": f"Échec de la création de l'ordre de renforcement: {error_msg}"
                }), 400
                
        except Exception as reinforce_error:
            logger.error(f"❌ Erreur lors de l'exécution de l'ordre de renforcement: {str(reinforce_error)}")
            return jsonify({"error": f"Erreur d'exécution: {str(reinforce_error)}"}), 500
    
    except Exception as e:
        logger.error(f"❌ Erreur lors du renforcement du cycle: {str(e)}")
        return jsonify({"error": str(e)}), 500

@routes_bp.route('/cycles/<cycle_id>', methods=['GET'])
def get_cycle_details(cycle_id):
    """
    Récupère les détails d'un cycle spécifique.
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        cycle = order_manager.cycle_manager.get_cycle(cycle_id)
        
        if not cycle:
            return jsonify({"error": f"Cycle {cycle_id} non trouvé"}), 404
        
        # Construire la réponse avec les détails du cycle
        return jsonify({
            "status": "success",
            "data": {
                "id": cycle.id,
                "symbol": cycle.symbol,
                "strategy": cycle.strategy,
                "status": cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status),
                "side": cycle.side.value if hasattr(cycle.side, 'value') else str(cycle.side),
                "entry_price": cycle.entry_price,
                "exit_price": cycle.exit_price,
                "quantity": cycle.quantity,
                "stop_price": cycle.stop_price,
                "profit_loss": cycle.profit_loss,
                "profit_loss_percent": cycle.profit_loss_percent,
                "created_at": cycle.created_at.isoformat() if cycle.created_at else None,
                "updated_at": cycle.updated_at.isoformat() if cycle.updated_at else None,
                "completed_at": cycle.completed_at.isoformat() if cycle.completed_at else None,
                "metadata": cycle.metadata
            }
        })
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération du cycle: {str(e)}")
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
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la réconciliation manuelle: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        }), 500

@routes_bp.route('/cycles/reload', methods=['POST'])
def reload_active_cycles():
    """Force le rechargement de tous les cycles actifs depuis la base de données."""
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        # Forcer le rechargement des cycles actifs
        logger.info("🔄 Rechargement forcé des cycles actifs depuis la base de données")
        order_manager.cycle_manager._load_active_cycles_from_db()
        
        # Récupérer le nombre de cycles chargés
        with order_manager.cycle_manager.cycles_lock:
            cycles_count = len(order_manager.cycle_manager.active_cycles)
            cycle_ids = list(order_manager.cycle_manager.active_cycles.keys())
        
        logger.info(f"✅ {cycles_count} cycles actifs rechargés en mémoire")
        
        return jsonify({
            "success": True,
            "message": f"{cycles_count} cycles actifs rechargés",
            "cycles_count": cycles_count,
            "cycle_ids": cycle_ids[:10]  # Retourner seulement les 10 premiers IDs
        })
    
    except Exception as e:
        logger.error(f"❌ Erreur lors du rechargement des cycles: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        }), 500

@routes_bp.route('/cycles/cleanup', methods=['POST'])
def cleanup_stuck_cycles():
    """
    Nettoie les cycles bloqués (active_sell sans ordre de sortie, etc.).
    
    Body JSON optionnel:
    {
        "timeout_minutes": 30,  # Délai avant de considérer un cycle comme bloqué (défaut: 30)
        "dry_run": false       # Si true, simule le nettoyage sans modifier les données
    }
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        # Récupérer les paramètres de la requête
        data = request.json or {}
        timeout_minutes = data.get('timeout_minutes', 30)
        dry_run = data.get('dry_run', False)
        
        logger.info(f"🧹 Nettoyage des cycles bloqués (timeout: {timeout_minutes}m, dry_run: {dry_run})")
        
        # Récupérer tous les cycles actifs
        cycles = order_manager.cycle_manager.repository.get_active_cycles()
        
        cleaned_cycles = []
        now = datetime.now()
        
        for cycle in cycles:
            # Vérifier les cycles bloqués depuis trop longtemps (sans exit_order_id car géré par StopManager)
            if cycle.status in [CycleStatus.ACTIVE_SELL, CycleStatus.WAITING_SELL]:
                time_since_update = now - cycle.updated_at
                
                if time_since_update.total_seconds() > timeout_minutes * 60:
                    cycle_info = {
                        "id": cycle.id,
                        "symbol": cycle.symbol,
                        "status": cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status),
                        "created_at": cycle.created_at.isoformat(),
                        "updated_at": cycle.updated_at.isoformat(),
                        "stuck_duration_minutes": time_since_update.total_seconds() / 60
                    }
                    
                    if not dry_run:
                        # Marquer le cycle comme échoué
                        logger.info(f"🧹 Nettoyage du cycle bloqué {cycle.id}")
                        cycle.status = CycleStatus.FAILED
                        cycle.updated_at = now
                        if not hasattr(cycle, 'metadata'):
                            cycle.metadata = {}
                        cycle.metadata['cancel_reason'] = f"Nettoyé après {time_since_update.total_seconds()/60:.1f} minutes bloqué en {cycle_info['status']}"
                        
                        # Sauvegarder le cycle
                        order_manager.cycle_manager.repository.save_cycle(cycle)
                        
                        # Publier l'événement
                        order_manager.cycle_manager._publish_cycle_event(cycle, "failed")
                        
                        cycle_info["action"] = "cleaned"
                    else:
                        cycle_info["action"] = "would_clean"
                    
                    cleaned_cycles.append(cycle_info)
        
        return jsonify({
            "success": True,
            "dry_run": dry_run,
            "timeout_minutes": timeout_minutes,
            "total_active_cycles": len(cycles),
            "cleaned_cycles_count": len(cleaned_cycles),
            "cleaned_cycles": cleaned_cycles,
            "message": f"{'Simulation de' if dry_run else ''} Nettoyage terminé: {len(cleaned_cycles)} cycles {'identifiés' if dry_run else 'nettoyés'}"
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage des cycles: {str(e)}")
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

@routes_bp.route('/binance/cancel-order', methods=['POST'])
def cancel_binance_order():
    """Annule directement un ordre sur Binance par son order_id."""
    try:
        order_manager = current_app.config.get('ORDER_MANAGER')
        if not order_manager:
            return jsonify({"error": "OrderManager non initialisé"}), 500
            
        data = request.json
        order_id = data.get('order_id')
        symbol = data.get('symbol')
        
        if not all([order_id, symbol]):
            return jsonify({
                "success": False,
                "message": "Paramètres manquants: order_id et symbol requis"
            }), 400
        
        # Annuler directement sur Binance
        success = order_manager.binance_executor.cancel_order(symbol, order_id)
        
        if success:
            logger.info(f"✅ Ordre {order_id} annulé directement sur Binance")
            return jsonify({
                "success": True,
                "message": f"Ordre {order_id} annulé avec succès",
                "order_id": order_id,
                "symbol": symbol
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Impossible d'annuler l'ordre {order_id}"
            }), 400
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'annulation directe de l'ordre: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        }), 500

@routes_bp.route('/balance/check', methods=['POST'])
def check_balance_for_order():
    """Vérifie si le solde est suffisant pour un ordre donné."""
    try:
        order_manager = current_app.config.get('ORDER_MANAGER')
        if not order_manager:
            return jsonify({"error": "OrderManager non initialisé"}), 500
            
        data = request.json
        symbol = data.get('symbol')
        side = data.get('side')
        quantity = float(data.get('quantity', 0))
        price = float(data.get('price', 0))
        
        if not all([symbol, side, quantity, price]):
            return jsonify({
                "success": False,
                "message": "Paramètres manquants: symbol, side, quantity, price requis"
            }), 400
        
        # Seulement vérifier pour les ordres BUY
        if side.upper() != 'BUY':
            return jsonify({
                "success": True,
                "message": "Vérification non nécessaire pour les ordres SELL",
                "sufficient": True
            })
        
        # Extraire la quote currency
        quote_currency = symbol.replace("BTC", "").replace("ETH", "").replace("BNB", "").replace("SUI", "")
        if not quote_currency:  # Pour les paires comme ETHBTC
            quote_currency = "BTC" if "BTC" in symbol and symbol != "BTCUSDC" else "USDC"
        
        # Récupérer les soldes
        balances = order_manager.binance_executor.utils.fetch_account_balances(
            order_manager.binance_executor.time_offset
        )
        
        available_balance = balances.get(quote_currency, {}).get('free', 0)
        total_cost = price * quantity * 1.001  # Ajouter 0.1% pour les frais
        
        sufficient = available_balance >= total_cost
        
        response = {
            "success": True,
            "symbol": symbol,
            "quote_currency": quote_currency,
            "available_balance": available_balance,
            "required_balance": total_cost,
            "sufficient": sufficient,
            "message": "Solde suffisant" if sufficient else f"Solde insuffisant: {available_balance:.8f} {quote_currency} < {total_cost:.8f} requis"
        }
        
        # Si insuffisant, calculer la quantité maximale possible
        if not sufficient:
            adjusted_quantity = (available_balance * 0.99) / price
            min_quantity = order_manager.binance_executor.symbol_constraints.get_min_qty(symbol)
            
            if adjusted_quantity >= min_quantity:
                response["suggested_quantity"] = adjusted_quantity
                response["suggested_cost"] = adjusted_quantity * price * 1.001
            else:
                response["suggested_quantity"] = 0
                response["message"] += f" (minimum requis: {min_quantity} {symbol.replace(quote_currency, '')})"
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la vérification du solde: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        }), 500

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
    
@routes_bp.route('/price/<symbol>', methods=['GET'])
def get_current_price(symbol):
    """
    Récupère le prix actuel d'un symbole.
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    try:
        # Récupérer le prix depuis le moniteur de prix
        price = order_manager.price_monitor.get_last_price(symbol)
        
        if price:
            return jsonify({
                "success": True,
                "symbol": symbol,
                "price": price,
                "timestamp": time.time()
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Prix non disponible pour {symbol}"
            }), 404
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération du prix: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        }), 500

@routes_bp.route('/balance/<asset>', methods=['GET'])
def get_asset_balance(asset):
    """
    Récupère le solde d'un actif spécifique depuis Binance.
    Utilisé par le coordinator pour vérifier les balances directement.
    
    Args:
        asset: Actif à vérifier (BTC, ETH, USDC, etc.)
        
    Returns:
        JSON avec le solde de l'actif:
        - free: Solde libre
        - locked: Solde bloqué
        - total: Solde total
    """
    order_manager = current_app.config['ORDER_MANAGER']
    
    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500
    
    try:
        # Valider l'actif
        asset = asset.upper()
        
        # Récupérer les balances depuis Binance
        balances = order_manager.binance_executor.utils.fetch_account_balances(
            order_manager.binance_executor.time_offset
        )
        
        if asset not in balances:
            return jsonify({
                "success": False,
                "asset": asset,
                "free": 0.0,
                "locked": 0.0,
                "total": 0.0,
                "message": f"Actif {asset} non trouvé ou solde zéro"
            })
        
        balance_info = balances[asset]
        free_balance = float(balance_info.get('free', 0))
        locked_balance = float(balance_info.get('locked', 0))
        total_balance = free_balance + locked_balance
        
        return jsonify({
            "success": True,
            "asset": asset,
            "free": free_balance,
            "locked": locked_balance,
            "total": total_balance,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération du solde {asset}: {str(e)}")
        return jsonify({
            "success": False,
            "asset": asset,
            "message": f"Erreur: {str(e)}"
        }), 500

