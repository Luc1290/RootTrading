"""
Point d'entrée principal pour le microservice Risk Manager.
Surveille et gère les risques du système de trading.
"""
import logging
import signal
import sys
import time
import os
import threading
import psutil
from typing import Dict, Any
from flask import Flask, jsonify

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import LOG_LEVEL, SYMBOLS
from risk_manager.src.checker import RuleChecker

# Configuration du logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('risk_manager.log')
    ]
)
logger = logging.getLogger("risk_manager")

# Suivi de l'utilisation des ressources
process = psutil.Process(os.getpid())
start_time = time.time()

# Créer l'application Flask
app = Flask(__name__)

# Variables pour le contrôle du service
rule_checker = None
running = True

@app.route('/health', methods=['GET'])
def health_check():
    """
    Point de terminaison pour vérifier l'état du service.
    """
    global rule_checker
    
    status = "ok"
    if rule_checker and hasattr(rule_checker, 'connection_failures'):
        if rule_checker.connection_failures > 3:
            status = "degraded"
    
    return jsonify({
        "status": status,
        "timestamp": time.time(),
        "uptime": time.time() - start_time,
        "symbols": SYMBOLS
    })

@app.route('/diagnostic', methods=['GET'])
def diagnostic():
    """
    Point de terminaison pour le diagnostic du service.
    """
    global rule_checker, start_time, process
    
    # Récupérer l'état du vérificateur de règles
    rules_status = {
        "running": False,
        "connection_failures": 0,
        "active_rules": 0,
        "total_rules": 0,
        "triggered_rules": 0
    }
    
    system_state = {}
    
    if rule_checker:
        summary = rule_checker.get_status_summary()
        rules_status = {
            "running": rule_checker.check_thread and rule_checker.check_thread.is_alive(),
            "connection_failures": rule_checker.connection_failures,
            "active_rules": summary["active_rules"],
            "total_rules": summary["total_rules"],
            "triggered_rules": summary["triggered_rules"],
            "last_check": summary["last_check"]
        }
        system_state = summary.get("system_state", {})
    
    # Construire la réponse
    diagnostic_info = {
        "status": "operational" if rules_status["running"] else "stopped",
        "timestamp": time.time(),
        "uptime": time.time() - start_time,
        "rules": rules_status,
        "system_state": system_state,
        "symbols": SYMBOLS,
        "memory_usage_mb": round(process.memory_info().rss / (1024 * 1024), 2),
        "cpu_percent": process.cpu_percent(interval=0.1),
        "thread_count": threading.active_count()
    }
    
    return jsonify(diagnostic_info)

@app.route('/rules', methods=['GET'])
def get_rules():
    """
    Récupère la liste des règles et leur état.
    """
    global rule_checker
    
    if not rule_checker:
        return jsonify({"error": "RuleChecker non initialisé"}), 500
    
    rules = []
    for rule in rule_checker.rules:
        rule_info = {
            "name": rule["name"],
            "description": rule.get("description", ""),
            "type": rule["type"],
            "scope": rule["scope"],
            "enabled": rule.get("enabled", True),
            "triggered": rule["name"] in rule_checker.triggered_rules
        }
        
        if rule["scope"] == "symbol" and "symbol" in rule:
            rule_info["symbol"] = rule["symbol"]
        
        if rule["scope"] == "strategy" and "strategy" in rule:
            rule_info["strategy"] = rule["strategy"]
        
        rules.append(rule_info)
    
    return jsonify(rules)

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

def main():
    """
    Fonction principale du service RiskManager.
    """
    global rule_checker, running
    
    # Configurer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Obtenir le chemin du fichier de règles
    rules_file = os.getenv("RISK_RULES_FILE", "risk_manager/src/rules.yaml")
    portfolio_api_url = os.getenv("PORTFOLIO_API_URL", "http://portfolio:8000")
    trader_api_url = os.getenv("TRADER_API_URL", "http://trader:5002")
    
    # Configurer le port API
    api_port = int(os.getenv("RISK_API_PORT", "5007"))
    
    logger.info("🚀 Démarrage du service Risk Manager RootTrading...")
    
    try:
        # Initialiser le vérificateur de règles
        rule_checker = RuleChecker(
            rules_file=rules_file,
            portfolio_api_url=portfolio_api_url,
            trader_api_url=trader_api_url
        )
        
        # Démarrer le vérificateur
        check_interval = int(os.getenv("RISK_CHECK_INTERVAL", "60"))  # Par défaut: toutes les 60 secondes
        rule_checker.check_interval = check_interval
        rule_checker.start()
        
        logger.info(f"✅ Vérification des règles démarrée (intervalle: {check_interval}s)")
        
        # Démarrer l'API dans un thread séparé
        api_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=api_port, debug=False),
            daemon=True
        )
        api_thread.start()
        logger.info(f"✅ API REST démarrée sur le port {api_port}")
        
        # Boucle principale pour garder le service actif
        while running:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le service Risk Manager: {str(e)}")
    finally:
        # Arrêter le vérificateur de règles
        if rule_checker:
            rule_checker.stop()
        
        logger.info("Service Risk Manager terminé")

if __name__ == "__main__":
    main()