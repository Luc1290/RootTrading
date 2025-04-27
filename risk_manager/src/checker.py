"""
Module de vérification des règles de gestion des risques.
Évalue les risques et applique les règles définies.
"""
import logging
import yaml
import threading
import time
import json
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Set

# Importer les modules partagés
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.redis_client import RedisClient
from shared.src.config import SYMBOLS

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('risk_manager.log')
    ]
)
logger = logging.getLogger(__name__)

class RuleChecker:
    """
    Gestionnaire des règles de gestion des risques.
    Charge les règles depuis un fichier YAML et les applique pour évaluer les risques.
    """
    
    def __init__(self, rules_file: str = None, portfolio_api_url: str = "http://portfolio:8000",
                 trader_api_url: str = "http://trader:5002"):
        """
        Initialise le gestionnaire de règles.
        
        Args:
            rules_file: Chemin vers le fichier de règles YAML (optionnel)
            portfolio_api_url: URL de l'API du service Portfolio
            trader_api_url: URL de l'API du service Trader
        """
        self.rules_file = rules_file or os.path.join(os.path.dirname(__file__), "rules.yaml")
        self.portfolio_api_url = portfolio_api_url
        self.trader_api_url = trader_api_url
        
        # Charger les règles
        self.rules = self._load_rules()
        
        # Client Redis pour les notifications
        self.redis_client = RedisClient()
        
        # État du système et règles déclenchées
        self.system_state: Dict[str, Any] = {}
        self.triggered_rules: Set[str] = set()
        
        # Thread de vérification
        self.check_thread = None
        self.stop_event = threading.Event()
        
        # Intervalle de vérification (en secondes)
        self.check_interval = 60
        
        # Compteur de tentatives de connexion
        self.connection_failures = 0
        self.max_connection_failures = 10
        
        logger.info(f"✅ RuleChecker initialisé avec {len(self.rules)} règles")
    
    def _load_rules(self) -> List[Dict[str, Any]]:
        """
        Charge les règles depuis le fichier YAML.
        
        Returns:
            Liste des règles
        """
        if not os.path.exists(self.rules_file):
            logger.warning(f"⚠️ Fichier de règles non trouvé: {self.rules_file}, utilisation des règles par défaut")
            return self._get_default_rules()
        
        try:
            with open(self.rules_file, 'r') as f:
                rules_data = yaml.safe_load(f)
            
            if not isinstance(rules_data, dict) or "rules" not in rules_data:
                logger.error("❌ Format de fichier de règles invalide")
                return self._get_default_rules()
            
            rules = rules_data.get("rules", [])
            
            # Valider chaque règle
            valid_rules = []
            for rule in rules:
                if self._validate_rule(rule):
                    valid_rules.append(rule)
                else:
                    logger.warning(f"⚠️ Règle invalide ignorée: {rule.get('name', 'unnamed')}")
            
            logger.info(f"✅ {len(valid_rules)} règles chargées depuis {self.rules_file}")
            return valid_rules
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des règles: {str(e)}")
            return self._get_default_rules()
    
    def _get_default_rules(self) -> List[Dict[str, Any]]:
        """
        Retourne un ensemble de règles par défaut.
        
        Returns:
            Liste des règles par défaut
        """
        default_rules = [
            {
                "name": "max_concurrent_trades",
                "description": "Limite le nombre maximum de trades actifs simultanés",
                "type": "exposure",
                "scope": "global",
                "condition": "active_trades > {max_trades}",
                "parameters": {"max_trades": 10},
                "action": "pause_new_trades",
                "severity": "warning",
                "enabled": True
            },
            {
                "name": "max_daily_loss",
                "description": "Arrête le trading si la perte quotidienne dépasse un seuil",
                "type": "drawdown",
                "scope": "global",
                "condition": "daily_pnl_percent < {max_loss_percent}",
                "parameters": {"max_loss_percent": -5.0},
                "action": "disable_trading",
                "severity": "critical",
                "enabled": True
            },
            {
                "name": "max_symbol_exposure",
                "description": "Limite l'exposition maximale par symbole",
                "type": "exposure",
                "scope": "symbol",
                "condition": "symbol_exposure_percent > {max_percent}",
                "parameters": {"max_percent": 25.0},
                "action": "pause_symbol",
                "severity": "warning",
                "enabled": True
            },
            {
                "name": "btc_volatility_alert",
                "description": "Alerte en cas de forte volatilité sur BTC",
                "type": "volatility",
                "scope": "symbol",
                "symbol": "BTCUSDC",
                "condition": "symbol_volatility_1h > {volatility_threshold}",
                "parameters": {"volatility_threshold": 4.0},
                "action": "alert_only",
                "severity": "info",
                "enabled": True
            },
            {
                "name": "max_daily_trades",
                "description": "Limite le nombre de nouveaux trades par jour",
                "type": "frequency",
                "scope": "global",
                "condition": "daily_trades > {max_trades}",
                "parameters": {"max_trades": 30},
                "action": "pause_new_trades",
                "severity": "warning",
                "enabled": True
            }
        ]
        
        logger.info(f"✅ {len(default_rules)} règles par défaut chargées")
        return default_rules
    
    def _validate_rule(self, rule: Dict[str, Any]) -> bool:
        """
        Valide une règle.
        
        Args:
            rule: Règle à valider
            
        Returns:
            True si la règle est valide, False sinon
        """
        # Vérifier les champs obligatoires
        required_fields = ["name", "type", "scope", "condition", "action", "enabled"]
        for field in required_fields:
            if field not in rule:
                logger.warning(f"⚠️ Champ obligatoire manquant dans la règle: {field}")
                return False
        
        # Vérifier que le champ 'symbol' est présent si le scope est 'symbol' ou 'strategy'
        if rule["scope"] in ["symbol", "strategy"] and "symbol" not in rule:
            logger.warning(f"⚠️ Champ 'symbol' requis pour le scope '{rule['scope']}'")
            return False
        
        # Vérifier que le champ 'strategy' est présent si le scope est 'strategy'
        if rule["scope"] == "strategy" and "strategy" not in rule:
            logger.warning(f"⚠️ Champ 'strategy' requis pour le scope 'strategy'")
            return False
        
        # Vérifier que le champ 'parameters' est présent si la condition contient des placeholders
        if "{" in rule["condition"] and "}" in rule["condition"] and "parameters" not in rule:
            logger.warning(f"⚠️ Champ 'parameters' requis pour les conditions avec placeholders")
            return False
        
        return True
    
    def _reload_rules(self) -> None:
        """
        Recharge les règles depuis le fichier YAML.
        """
        new_rules = self._load_rules()
        
        # Comparer avec les règles actuelles
        if new_rules != self.rules:
            old_count = len(self.rules)
            self.rules = new_rules
            logger.info(f"🔄 Règles rechargées: {old_count} -> {len(self.rules)} règles")
    
    def _initialize_default_state(self) -> Dict[str, Any]:
        """
        Initialise un état par défaut avec des valeurs sécuritaires.
        
        Returns:
            État par défaut
        """
        state = {
            "timestamp": datetime.now().isoformat(),
            "active_trades": 0,
            "daily_pnl": 0.0,
            "daily_pnl_percent": 0.0,
            "daily_trades": 0,
            "total_value": 1000.0,  # Valeur initiale par défaut (1000 USDC)
            "symbols": {},
            "strategy_consecutive_losses": 0  # Valeur par défaut
        }
        
        # Ajouter des valeurs par défaut pour chaque symbole
        for symbol in SYMBOLS:
            state["symbols"][symbol] = {
                "active_trades": 0,
                "exposure": 0.0,
                "exposure_percent": 0.0,
                "volatility_1h": 1.0,  # Valeur par défaut pour la volatilité
                "volatility_24h": 2.0,
                "price_change_1h": 0.0,
                "price_change_24h": 0.0
            }
            
            # Définir des alias pour simplifier l'évaluation des règles
            state[f"{symbol}_exposure"] = 0.0
            state[f"{symbol}_exposure_percent"] = 0.0
            state[f"{symbol}_volatility_1h"] = 1.0
            state[f"{symbol}_volatility_24h"] = 2.0
            state[f"{symbol}_active_trades"] = 0
            state[f"{symbol}_price_change_1h"] = 0.0
            state[f"{symbol}_price_change_24h"] = 0.0
        
        return state
    
    def _get_system_state(self) -> Dict[str, Any]:
        """
        Récupère l'état actuel du système.
        
        Returns:
            État du système
        """
        # Initialiser avec des valeurs par défaut
        state = self._initialize_default_state()
        
        try:
            # Récupérer les données du portfolio
            portfolio_response = requests.get(f"{self.portfolio_api_url}/summary", timeout=5)
            if portfolio_response.status_code == 200:
                portfolio_data = portfolio_response.json()
                state["total_value"] = portfolio_data.get("total_value", state["total_value"])
                state["active_trades"] = portfolio_data.get("active_trades", state["active_trades"])
                state["performance_24h"] = portfolio_data.get("performance_24h", 0.0)
                
                # Calculer le PnL quotidien
                state["daily_pnl_percent"] = state["performance_24h"]
                if state["daily_pnl_percent"] is None:
                    state["daily_pnl_percent"] = 0.0
                if state["total_value"] > 0:
                    state["daily_pnl"] = state["total_value"] * state["daily_pnl_percent"] / 100
                
                # Récupérer les balances
                balances = portfolio_data.get("balances", [])
                state["balances"] = {b["asset"]: b for b in balances}
                
                # Réinitialiser le compteur d'échecs de connexion
                self.connection_failures = 0
            
            # Récupérer les données des trades actifs
            trades_response = requests.get(f"{self.trader_api_url}/orders", timeout=5)
            if trades_response.status_code == 200:
                active_orders = trades_response.json()
                
                # Organiser par symbole
                symbol_trades = {}
                for order in active_orders:
                    symbol = order.get("symbol", "unknown")
                    if symbol not in symbol_trades:
                        symbol_trades[symbol] = []
                    symbol_trades[symbol].append(order)
                
                # Calculer l'exposition par symbole
                for symbol, trades in symbol_trades.items():
                    if symbol in state["symbols"]:
                        exposure = sum(float(t.get("quantity", 0)) * float(t.get("entry_price", 0)) for t in trades)
                        state["symbols"][symbol]["active_trades"] = len(trades)
                        state["symbols"][symbol]["exposure"] = exposure
                        
                        # Calculer le pourcentage d'exposition
                        if state["total_value"] > 0:
                            state["symbols"][symbol]["exposure_percent"] = (exposure / state["total_value"]) * 100
                
                # Réinitialiser le compteur d'échecs de connexion
                self.connection_failures = 0
            
            # Récupérer les statistiques de trading quotidiennes
            stats_response = requests.get(f"{self.portfolio_api_url}/performance/daily?limit=1", timeout=5)
            if stats_response.status_code == 200:
                stats_data = stats_response.json()
                if "data" in stats_data and len(stats_data["data"]) > 0:
                    daily_stats = stats_data["data"][0]
                    state["daily_trades"] = daily_stats.get("total_trades", state["daily_trades"])
                    
                    # Si le PnL n'a pas été calculé plus haut, l'extraire des stats
                    if state["daily_pnl"] == 0:
                        state["daily_pnl"] = daily_stats.get("daily_profit_loss", state["daily_pnl"])
                
                # Réinitialiser le compteur d'échecs de connexion
                self.connection_failures = 0
            
            # Pour chaque symbole configuré, s'assurer que les variables sont définies
            for symbol in SYMBOLS:
                if symbol not in state["symbols"]:
                    state["symbols"][symbol] = {
                        "active_trades": 0,
                        "exposure": 0.0,
                        "exposure_percent": 0.0,
                        "volatility_1h": 1.0,
                        "volatility_24h": 2.0,
                        "price_change_1h": 0.0,
                        "price_change_24h": 0.0
                    }
                else:
                    # S'assurer que toutes les clés sont présentes
                    default_keys = ["active_trades", "exposure", "exposure_percent", 
                                   "volatility_1h", "volatility_24h", "price_change_1h", "price_change_24h"]
                    for key in default_keys:
                        if key not in state["symbols"][symbol]:
                            state["symbols"][symbol][key] = 0.0 if "percent" in key or "price" in key or "exposure" in key else 0
                
                # Mettre à jour la volatilité (à calculer ou récupérer)
                if "volatility_1h" not in state["symbols"][symbol]:
                    state["symbols"][symbol]["volatility_1h"] = self._calculate_volatility(symbol, "1h")
                
                if "volatility_24h" not in state["symbols"][symbol]:
                    state["symbols"][symbol]["volatility_24h"] = self._calculate_volatility(symbol, "24h")
                
                # Définir des alias pour simplifier l'évaluation des règles
                state[f"{symbol}_exposure"] = state["symbols"][symbol]["exposure"]
                state[f"{symbol}_exposure_percent"] = state["symbols"][symbol]["exposure_percent"]
                state[f"{symbol}_volatility_1h"] = state["symbols"][symbol]["volatility_1h"]
                state[f"{symbol}_volatility_24h"] = state["symbols"][symbol]["volatility_24h"]
                state[f"{symbol}_active_trades"] = state["symbols"][symbol]["active_trades"]
                state[f"{symbol}_price_change_1h"] = state["symbols"][symbol].get("price_change_1h", 0.0)
                state[f"{symbol}_price_change_24h"] = state["symbols"][symbol].get("price_change_24h", 0.0)
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération de l'état du système: {str(e)}")
            self.connection_failures += 1
            
            # Si trop d'échecs consécutifs, mettre une alerte dans les logs
            if self.connection_failures >= self.max_connection_failures:
                logger.warning(f"⚠️ {self.connection_failures} échecs de connexion consécutifs. Utilisation de valeurs par défaut.")
        
        return state
    
    def _calculate_volatility(self, symbol: str, period: str) -> float:
        """
        Calcule la volatilité d'un symbole sur une période donnée.
        
        Args:
            symbol: Symbole
            period: Période ('1h', '24h', etc.)
            
        Returns:
            Volatilité (en pourcentage)
        """
        # TODO: Implémenter le calcul de volatilité en utilisant les données historiques
        # Pour l'instant, utiliser des valeurs par défaut pour les tests
        if period == "1h":
            return 1.5  # 1.5% de volatilité par heure
        elif period == "24h":
            return 4.0  # 4% de volatilité par jour
        else:
            return 0.0
    
    def _evaluate_condition(self, condition: str, parameters: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """
        Évalue une condition de règle de manière sécurisée.
        
        Args:
            condition: Condition à évaluer
            parameters: Paramètres de la condition
            state: État du système
            
        Returns:
            True si la condition est remplie, False sinon
        """
        try:
            # Remplacer les placeholders par les valeurs de paramètres
            formatted_condition = condition
            for param_name, param_value in parameters.items():
                placeholder = "{" + param_name + "}"
                formatted_condition = formatted_condition.replace(placeholder, str(param_value))
        
            # Créer un environnement d'évaluation restreint avec l'état du système
            safe_dict = {k: v for k, v in state.items()}
        
            # Ajouter des fonctions mathématiques sécurisées
            safe_dict.update({
                'abs': abs,
                'max': max,
                'min': min,
                'round': round,
                'sum': sum
            })
        
            # Remplacer les opérateurs logiques textuels par leurs équivalents Python
            formatted_condition = formatted_condition.replace(" and ", " and ")
            formatted_condition = formatted_condition.replace(" or ", " or ")
            formatted_condition = formatted_condition.replace(" not ", " not ")
        
            # Évaluer la condition de manière restreinte
            return eval(formatted_condition, {"__builtins__": {}}, safe_dict)
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'évaluation de la condition '{condition}': {str(e)}")
            return False
    
    def _execute_action(self, action: str, rule: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """
        Exécute l'action d'une règle.
        
        Args:
            action: Action à exécuter
            rule: Règle concernée
            state: État du système
            
        Returns:
            True si l'action a été exécutée avec succès, False sinon
        """
        try:
            rule_name = rule["name"]
            
            # Log de l'action
            logger.warning(f"⚠️ Règle déclenchée: {rule_name} -> {action}")
            
            # Notifier via Redis
            notification = {
                "rule_name": rule_name,
                "rule_type": rule["type"],
                "action": action,
                "severity": rule.get("severity", "warning"),
                "description": rule.get("description", ""),
                "timestamp": datetime.now().isoformat(),
                "state": {k: state[k] for k in ["active_trades", "daily_pnl", "daily_pnl_percent"] 
                         if k in state}
            }
            
            # Ajouter des infos spécifiques selon le scope
            if rule["scope"] == "symbol" and "symbol" in rule:
                symbol = rule["symbol"]
                notification["symbol"] = symbol
                if symbol in state["symbols"]:
                    notification["symbol_state"] = state["symbols"][symbol]
            
            # Publier la notification
            try:
                self.redis_client.publish("roottrading:alerts", notification)
            except Exception as e:
                logger.error(f"❌ Erreur lors de la publication de la notification: {str(e)}")
            
            # Ajouter aux règles déclenchées
            self.triggered_rules.add(rule_name)
            
            # Exécuter l'action spécifique
            if action == "alert_only":
                # Rien à faire de plus, juste l'alerte
                return True
            
            # Pour les autres actions, essayez de les exécuter mais ne bloquez pas le fonctionnement
            # en cas d'échec (pour augmenter la résilience)
            try:
                if action == "pause_new_trades":
                    # Appeler l'API pour mettre en pause les nouveaux trades
                    if rule["scope"] == "global":
                        # Pause globale
                        response = requests.post(f"{self.trader_api_url}/config/pause", timeout=5)
                        return response.status_code == 200
                    
                    elif rule["scope"] == "symbol" and "symbol" in rule:
                        # Pause spécifique à un symbole
                        symbol = rule["symbol"]
                        response = requests.post(
                            f"{self.trader_api_url}/config/pause",
                            json={"symbol": symbol},
                            timeout=5
                        )
                        return response.status_code == 200
                    
                    elif rule["scope"] == "strategy" and "strategy" in rule:
                        # Pause spécifique à une stratégie
                        strategy = rule["strategy"]
                        response = requests.post(
                            f"{self.trader_api_url}/config/pause",
                            json={"strategy": strategy},
                            timeout=5
                        )
                        return response.status_code == 200
                
                elif action == "disable_trading":
                    # Appeler l'API pour désactiver le trading
                    response = requests.post(f"{self.trader_api_url}/config/disable", timeout=5)
                    return response.status_code == 200
                
                elif action == "pause_symbol" and rule["scope"] == "symbol" and "symbol" in rule:
                    # Pause spécifique à un symbole
                    symbol = rule["symbol"]
                    response = requests.post(
                        f"{self.trader_api_url}/config/pause",
                        json={"symbol": symbol},
                        timeout=5
                    )
                    return response.status_code == 200
                
                elif action == "close_symbol_positions" and rule["scope"] == "symbol" and "symbol" in rule:
                    # Fermer toutes les positions d'un symbole
                    symbol = rule["symbol"]
                    response = requests.post(
                        f"{self.trader_api_url}/close_all",
                        json={"symbol": symbol},
                        timeout=5
                    )
                    return response.status_code == 200
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'exécution de l'action '{action}': {str(e)}")
                return False
                
            # Action non reconnue ou non implémentée
            logger.warning(f"⚠️ Action '{action}' non implémentée")
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution de l'action '{action}': {str(e)}")
            return False
    
    def check_rules(self) -> List[Dict[str, Any]]:
        """
        Vérifie toutes les règles actives et exécute les actions nécessaires.
        
        Returns:
            Liste des règles déclenchées
        """
        # Recharger les règles (pour prendre en compte les modifications)
        self._reload_rules()
        
        # Récupérer l'état actuel du système
        state = self._get_system_state()
        self.system_state = state
        
        # Réinitialiser les règles déclenchées pour cette vérification
        triggered_rules_now = []
        
        # Vérifier chaque règle
        for rule in self.rules:
            # Ignorer les règles désactivées
            if not rule.get("enabled", True):
                continue
            
            # Vérifier si la condition est remplie
            try:
                condition = rule["condition"]
                parameters = rule.get("parameters", {})
                
                if self._evaluate_condition(condition, parameters, state):
                    # Condition remplie, exécuter l'action
                    action = rule["action"]
                    success = self._execute_action(action, rule, state)
                    
                    # Stocker les informations sur la règle déclenchée
                    triggered = {
                        "name": rule["name"],
                        "type": rule["type"],
                        "scope": rule["scope"],
                        "action": action,
                        "success": success,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    if rule["scope"] == "symbol" and "symbol" in rule:
                        triggered["symbol"] = rule["symbol"]
                    
                    if rule["scope"] == "strategy" and "strategy" in rule:
                        triggered["strategy"] = rule["strategy"]
                    
                    triggered_rules_now.append(triggered)
            except Exception as e:
                logger.error(f"❌ Erreur lors de la vérification de la règle {rule['name']}: {str(e)}")
        
        return triggered_rules_now
    
    def _check_loop(self) -> None:
        """
        Boucle de vérification périodique des règles.
        """
        logger.info("🔍 Démarrage de la vérification des règles...")
        
        while not self.stop_event.is_set():
            try:
                # Vérifier les règles
                triggered = self.check_rules()
                
                if triggered:
                    logger.warning(f"⚠️ {len(triggered)} règles déclenchées")
                else:
                    logger.info("✅ Aucune règle déclenchée")
                
                # Attendre jusqu'à la prochaine vérification
                for _ in range(self.check_interval):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
            
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle de vérification: {str(e)}")
                time.sleep(10)  # Pause en cas d'erreur
        
        logger.info("✅ Vérification des règles arrêtée")
    
    def start(self) -> None:
        """
        Démarre la vérification des règles en arrière-plan.
        """
        if self.check_thread and self.check_thread.is_alive():
            logger.warning("⚠️ Vérification des règles déjà en cours")
            return
        
        self.stop_event.clear()
        self.check_thread = threading.Thread(target=self._check_loop, daemon=True)
        self.check_thread.start()
        
        logger.info("✅ Vérification des règles démarrée")
    
    def stop(self) -> None:
        """
        Arrête la vérification des règles.
        """
        if not self.check_thread or not self.check_thread.is_alive():
            return
        
        logger.info("🛑 Arrêt de la vérification des règles...")
        self.stop_event.set()
        
        if self.check_thread:
            self.check_thread.join(timeout=self.check_interval + 5)
            if self.check_thread.is_alive():
                logger.warning("⚠️ Le thread de vérification ne s'est pas arrêté proprement")
        
        # Fermer le client Redis
        self.redis_client.close()
        
        logger.info("✅ Vérification des règles arrêtée")
    
    def get_system_state(self) -> Dict[str, Any]:
        """
        Récupère l'état actuel du système.
        
        Returns:
            État du système
        """
        return self.system_state
    
    def get_triggered_rules(self) -> List[str]:
        """
        Récupère la liste des règles déclenchées.
        
        Returns:
            Liste des règles déclenchées
        """
        return list(self.triggered_rules)
    
    def get_rule_details(self, rule_name: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les détails d'une règle.
        
        Args:
            rule_name: Nom de la règle
            
        Returns:
            Détails de la règle ou None si non trouvée
        """
        for rule in self.rules:
            if rule["name"] == rule_name:
                return rule
        return None
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Récupère un résumé de l'état du RuleChecker.
        
        Returns:
            Résumé de l'état
        """
        return {
            "active_rules": len([r for r in self.rules if r.get("enabled", True)]),
            "total_rules": len(self.rules),
            "triggered_rules": len(self.triggered_rules),
            "connection_failures": self.connection_failures,
            "system_state": {
                "active_trades": self.system_state.get("active_trades", 0),
                "total_value": self.system_state.get("total_value", 0),
                "daily_pnl_percent": self.system_state.get("daily_pnl_percent", 0),
                "symbols": {
                    symbol: {
                        "active_trades": self.system_state.get("symbols", {}).get(symbol, {}).get("active_trades", 0),
                        "exposure_percent": self.system_state.get("symbols", {}).get(symbol, {}).get("exposure_percent", 0)
                    } for symbol in SYMBOLS
                }
            },
            "last_check": datetime.now().isoformat()
        }

# Point d'entrée pour les tests
if __name__ == "__main__":
    # Initialiser et démarrer le vérificateur de règles
    checker = RuleChecker()
    checker.start()
    
    try:
        # Rester en vie jusqu'à Ctrl+C
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        # Arrêter proprement
        checker.stop()
        logger.info("Programme terminé")