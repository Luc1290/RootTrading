"""
Module pour surveiller et corriger automatiquement la synchronisation des cycles.
Solution DÉFINITIVE au problème de désynchronisation du coordinator.
"""
import logging
import threading
import time
from typing import Dict, List, Optional
import requests
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class CycleSyncMonitor:
    """
    Moniteur de synchronisation des cycles.
    S'assure que le coordinator reste toujours synchronisé avec la base de données.
    """
    
    def __init__(self, trader_api_url: str, check_interval: int = 30):
        """
        Initialise le moniteur de synchronisation.
        
        Args:
            trader_api_url: URL de l'API du trader
            check_interval: Intervalle de vérification en secondes (par défaut 30s)
        """
        self.trader_api_url = trader_api_url
        self.check_interval = check_interval
        self.running = False
        self.monitor_thread = None
        
        # Cache des cycles connus
        self.known_cycles = set()
        
        # Statistiques
        self.stats = {
            "checks_performed": 0,
            "discrepancies_found": 0,
            "cycles_cleaned": 0,
            "last_check": None,
            "last_discrepancy": None
        }
        
        logger.info(f"✅ CycleSyncMonitor initialisé - Intervalle: {check_interval}s")
    
    def start(self):
        """Démarre le moniteur de synchronisation."""
        if self.running:
            logger.warning("Le moniteur est déjà en cours d'exécution")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("✅ Moniteur de synchronisation démarré")
    
    def stop(self):
        """Arrête le moniteur de synchronisation."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("✅ Moniteur de synchronisation arrêté")
    
    def _monitor_loop(self):
        """Boucle principale du moniteur."""
        while self.running:
            try:
                self._check_synchronization()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle de monitoring: {str(e)}")
                time.sleep(self.check_interval)
    
    def _check_synchronization(self):
        """
        Vérifie la synchronisation entre les cycles en mémoire et la DB.
        """
        self.stats["checks_performed"] += 1
        self.stats["last_check"] = time.time()
        
        try:
            # Récupérer les cycles actifs depuis l'API centralisée
            response = requests.get(
                urljoin(self.trader_api_url, "/cycles"),
                params={"confirmed": "true", "include_completed": "false"},
                timeout=5
            )
            
            if response.status_code != 200:
                logger.warning(f"Impossible de récupérer les cycles: {response.status_code}")
                return
                
            result = response.json()
            if not result.get('success'):
                logger.warning(f"Réponse invalide de l'API cycles")
                return
                
            db_cycles = result.get('cycles', [])
            
            # Les cycles retournés par l'API /cycles sont déjà filtrés comme actifs
            db_cycle_ids = {cycle['id'] for cycle in db_cycles}
            
            # Détecter les cycles fantômes (dans notre cache mais pas dans la DB)
            phantom_cycles = self.known_cycles - db_cycle_ids
            
            if phantom_cycles:
                logger.warning(f"⚠️ {len(phantom_cycles)} cycles fantômes détectés")
                self.stats["discrepancies_found"] += 1
                self.stats["last_discrepancy"] = time.time()
                
                # Nettoyer les cycles fantômes de notre cache
                for cycle_id in phantom_cycles:
                    self.known_cycles.remove(cycle_id)
                    self.stats["cycles_cleaned"] += 1
                    logger.info(f"🧹 Cycle fantôme retiré du cache: {cycle_id}")
            
            # Mettre à jour notre cache avec les cycles de la DB
            self.known_cycles = db_cycle_ids
            
            # Logger périodiquement l'état
            if self.stats["checks_performed"] % 10 == 0:
                logger.info(f"📊 Sync OK - Cycles actifs: {len(self.known_cycles)} - "
                          f"Vérifications: {self.stats['checks_performed']} - "
                          f"Nettoyages: {self.stats['cycles_cleaned']}")
                          
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification de sync: {str(e)}")
    
    def force_sync(self) -> bool:
        """
        Force une synchronisation immédiate.
        
        Returns:
            True si la synchronisation a réussi, False sinon
        """
        logger.info("🔄 Synchronisation forcée demandée")
        try:
            self._check_synchronization()
            return True
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sync forcée: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """
        Retourne les statistiques du moniteur.
        
        Returns:
            Dictionnaire des statistiques
        """
        return {
            **self.stats,
            "active_cycles": len(self.known_cycles),
            "running": self.running
        }