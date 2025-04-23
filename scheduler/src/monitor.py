"""
Module de surveillance des services.
Vérifie l'état de santé des différents services et déclenche des alertes si nécessaire.
"""
import logging
import requests
import threading
import time
from typing import Dict, Any, List, Set, Optional
from datetime import datetime, timedelta
import json
import subprocess
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceMonitor:
    """
    Monitore l'état de santé des services de la plateforme RootTrading.
    Effectue des vérifications périodiques et déclenche des alertes si nécessaire.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialise le moniteur de services.
        
        Args:
            config_file: Chemin vers le fichier de configuration (optionnel)
        """
        # Charger la configuration
        self.config = self._load_config(config_file)
        
        # Liste des services à surveiller
        self.services = self.config.get("services", {})
        
        # État des services
        self.service_states: Dict[str, Dict[str, Any]] = {}
        
        # Thread de surveillance
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # Alertes envoyées (pour éviter la duplication)
        self.sent_alerts: Set[str] = set()
        
        # Intervalle de vérification (en secondes)
        self.check_interval = self.config.get("check_interval", 60)
        
        # Notification Discord (optionnel)
        self.discord_webhook = self.config.get("discord_webhook")
        
        logger.info(f"✅ ServiceMonitor initialisé avec {len(self.services)} services")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Charge la configuration depuis un fichier ou utilise les valeurs par défaut.
        
        Args:
            config_file: Chemin vers le fichier de configuration
            
        Returns:
            Configuration chargée
        """
        default_config = {
            "services": {
                "gateway": {
                    "url": "http://gateway:5000/health",
                    "timeout": 5,
                    "restart_command": "docker-compose restart gateway",
                    "critical": True
                },
                "analyzer": {
                    "url": "http://analyzer:5001/health",
                    "timeout": 5,
                    "restart_command": "docker-compose restart analyzer",
                    "critical": True
                },
                "trader": {
                    "url": "http://trader:5002/health",
                    "timeout": 5,
                    "restart_command": "docker-compose restart trader",
                    "critical": True
                },
                "portfolio": {
                    "url": "http://portfolio:8000/health",
                    "timeout": 5,
                    "restart_command": "docker-compose restart portfolio",
                    "critical": True
                },
                "frontend": {
                    "url": "http://frontend:3000/health",
                    "timeout": 5,
                    "restart_command": "docker-compose restart frontend",
                    "critical": False
                }
            },
            "check_interval": 60,  # Secondes
            "max_restarts": 3,     # Maximum de redémarrages par 24h
            "discord_webhook": None
        }
        
        if not config_file:
            return default_config
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Fusionner avec les valeurs par défaut
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict) and isinstance(config[key], dict):
                    # Pour les dictionnaires imbriqués comme "services"
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
            
            return config
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            return default_config
    
    def check_service(self, service_name: str, service_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vérifie l'état de santé d'un service.
        
        Args:
            service_name: Nom du service
            service_config: Configuration du service
            
        Returns:
            État de santé du service
        """
        url = service_config.get("url")
        timeout = service_config.get("timeout", 5)
        
        status = {
            "name": service_name,
            "status": "unknown",
            "response_time": 0,
            "last_check": datetime.now().isoformat(),
            "message": ""
        }
        
        if not url:
            status["status"] = "error"
            status["message"] = "URL non définie"
            return status
        
        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout)
            response_time = time.time() - start_time
            
            status["response_time"] = response_time
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    service_status = data.get("status", "").lower()
                    
                    if service_status == "ok":
                        status["status"] = "healthy"
                        status["message"] = f"Service en ligne, temps de réponse: {response_time:.3f}s"
                    else:
                        status["status"] = "degraded"
                        status["message"] = f"Service répond mais statut: {service_status}"
                        
                except ValueError:
                    status["status"] = "degraded"
                    status["message"] = "Réponse non-JSON"
            else:
                status["status"] = "error"
                status["message"] = f"Code HTTP: {response.status_code}"
        
        except requests.exceptions.Timeout:
            status["status"] = "error"
            status["message"] = f"Timeout après {timeout}s"
        
        except requests.exceptions.ConnectionError:
            status["status"] = "error"
            status["message"] = "Impossible de se connecter"
        
        except Exception as e:
            status["status"] = "error"
            status["message"] = f"Erreur: {str(e)}"
        
        return status
    
    def should_restart_service(self, service_name: str, service_state: Dict[str, Any]) -> bool:
        """
        Détermine si un service doit être redémarré.
        
        Args:
            service_name: Nom du service
            service_state: État actuel du service
            
        Returns:
            True si le service doit être redémarré, False sinon
        """
        # Si le service est en bonne santé, pas besoin de le redémarrer
        if service_state["status"] == "healthy":
            return False
        
        # Vérifier le nombre de redémarrages récents
        restarts = service_state.get("restarts", [])
        recent_restarts = [r for r in restarts if 
                           datetime.fromisoformat(r) > datetime.now() - timedelta(hours=24)]
        
        max_restarts = self.config.get("max_restarts", 3)
        
        # Si trop de redémarrages récents, éviter un nouveau redémarrage
        if len(recent_restarts) >= max_restarts:
            logger.warning(f"⚠️ Trop de redémarrages récents pour {service_name} ({len(recent_restarts)}/{max_restarts})")
            return False
        
        # Si service critique en erreur, le redémarrer
        if self.services[service_name].get("critical", False) and service_state["status"] == "error":
            return True
        
        # Si service dégradé pendant trop longtemps, le redémarrer
        if service_state["status"] == "degraded":
            # Check for consecutive degraded states
            degraded_count = service_state.get("consecutive_degraded", 0)
            if degraded_count >= 3:  # Redémarrer après 3 vérifications dégradées consécutives
                return True
        
        return False
    
    def restart_service(self, service_name: str) -> bool:
        """
        Signale qu'un service devrait être redémarré, mais ne tente pas de le faire directement.
    
        Args:
            service_name: Nom du service à redémarrer
        
        Returns:
            Toujours False car nous ne redémarrons pas réellement le service
        """
        logger.info(f"🔄 Service {service_name} nécessite un redémarrage")
    
        # Enregistrer que nous avons détecté un problème
        now = datetime.now().isoformat()
        if service_name in self.service_states:
            if "restarts" not in self.service_states[service_name]:
                self.service_states[service_name]["restarts"] = []
        
            self.service_states[service_name]["restarts"].append(now)
            self.service_states[service_name]["last_restart_request"] = now
    
        # Envoyer une alerte pour le besoin de redémarrage
        self.send_alert(f"Le service {service_name} nécessite un redémarrage (vérifier manuellement)", "warning")
    
        return False
    
    def send_alert(self, message: str, level: str = "warning") -> bool:
        """
        Envoie une alerte.
        
        Args:
            message: Message d'alerte
            level: Niveau d'alerte ('info', 'warning', 'critical')
            
        Returns:
            True si l'alerte a été envoyée, False sinon
        """
        # Éviter les alertes dupliquées
        alert_key = f"{level}:{message}"
        if alert_key in self.sent_alerts:
            return False
        
        logger.warning(f"⚠️ ALERTE ({level}): {message}")
        
        # Enregistrer l'alerte pour éviter les duplications
        self.sent_alerts.add(alert_key)
        
        # Si le webhook Discord est configuré, envoyer une notification
        if self.discord_webhook:
            try:
                # Déterminer la couleur selon le niveau
                color = {
                    "info": 0x3498db,       # Bleu
                    "warning": 0xf1c40f,    # Jaune
                    "critical": 0xe74c3c    # Rouge
                }.get(level, 0x95a5a6)      # Gris par défaut
                
                # Préparer le payload pour Discord
                payload = {
                    "embeds": [{
                        "title": f"Alerte RootTrading - {level.upper()}",
                        "description": message,
                        "color": color,
                        "timestamp": datetime.now().isoformat()
                    }]
                }
                
                # Envoyer la notification
                response = requests.post(
                    self.discord_webhook, 
                    json=payload, 
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 204:
                    logger.info("✅ Notification Discord envoyée")
                    return True
                else:
                    logger.warning(f"⚠️ Échec de l'envoi de la notification Discord: {response.status_code}")
                    return False
            
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'envoi de la notification Discord: {str(e)}")
                return False
        
        return True
    
    def monitor_services(self) -> None:
        """
        Boucle principale de surveillance des services.
        """
        logger.info("🔍 Démarrage de la surveillance des services...")
        
        while not self.stop_event.is_set():
            for service_name, service_config in self.services.items():
                try:
                    # Vérifier l'état du service
                    status = self.check_service(service_name, service_config)
                    
                    # Mettre à jour l'état
                    if service_name not in self.service_states:
                        self.service_states[service_name] = status
                    else:
                        # Mettre à jour le compteur de dégradations consécutives
                        if status["status"] == "degraded":
                            consec_degraded = self.service_states[service_name].get("consecutive_degraded", 0)
                            self.service_states[service_name]["consecutive_degraded"] = consec_degraded + 1
                        else:
                            self.service_states[service_name]["consecutive_degraded"] = 0
                        
                        # Mettre à jour le reste de l'état
                        self.service_states[service_name].update(status)
                    
                    # Vérifier si le service doit être redémarré
                    if self.should_restart_service(service_name, self.service_states[service_name]):
                        self.restart_service(service_name)
                    
                    # Envoyer des alertes si nécessaire
                    if status["status"] == "error":
                        self.send_alert(
                            f"Service {service_name} en erreur: {status['message']}",
                            "critical" if service_config.get("critical", False) else "warning"
                        )
                    
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la vérification du service {service_name}: {str(e)}")
            
            # Attendre jusqu'à la prochaine vérification
            for _ in range(self.check_interval):
                if self.stop_event.is_set():
                    break
                time.sleep(1)
    
    def start(self) -> None:
        """
        Démarre la surveillance des services.
        """
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Surveillance déjà en cours")
            return
        
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self.monitor_services, daemon=True)
        self.monitor_thread.start()
        
        logger.info("✅ Surveillance des services démarrée")
    
    def stop(self) -> None:
        """
        Arrête la surveillance des services.
        """
        if not self.monitor_thread or not self.monitor_thread.is_alive():
            return
        
        logger.info("Arrêt de la surveillance des services...")
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.check_interval + 5)
            if self.monitor_thread.is_alive():
                logger.warning("⚠️ Le thread de surveillance ne s'est pas arrêté proprement")
        
        logger.info("✅ Surveillance des services arrêtée")
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Récupère l'état actuel de tous les services.
        
        Returns:
            État de tous les services
        """
        return {
            "services": self.service_states,
            "last_update": datetime.now().isoformat(),
            "uptime": None,  # TODO: Implémenter le calcul d'uptime
            "total_restarts": sum(len(s.get("restarts", [])) for s in self.service_states.values())
        }

# Point d'entrée pour les tests
if __name__ == "__main__":
    # Initialiser et démarrer le moniteur
    monitor = ServiceMonitor()
    monitor.start()
    
    try:
        # Rester en vie jusqu'à Ctrl+C
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        # Arrêter proprement
        monitor.stop()
        logger.info("Programme terminé")