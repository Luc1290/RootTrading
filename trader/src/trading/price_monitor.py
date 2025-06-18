"""
Moniteur de prix pour le trader.
S'occupe de surveiller les prix en temps réel et de notifier les changements.
"""
import logging
import threading
import time
from typing import Dict, Callable, List, Optional
from datetime import datetime

from shared.src.redis_client import RedisClient

# Configuration du logging
logger = logging.getLogger(__name__)

class PriceMonitor:
    """
    Moniteur de prix.
    S'abonne aux mises à jour de prix et les transmet aux composants intéressés.
    """
    
    def __init__(self, symbols: List[str], price_update_callback: Callable[[str, float], None]):
        """
        Initialise le moniteur de prix.
        
        Args:
            symbols: Liste des symboles à surveiller
            price_update_callback: Fonction de rappel pour les mises à jour de prix
        """
        self.symbols = symbols
        self.price_update_callback = price_update_callback
        self.redis_client = RedisClient()
        
        # Configuration des canaux Redis pour les mises à jour de prix
        self.price_channels = [f"roottrading:market:data:{symbol.lower()}" for symbol in self.symbols]
        
        # Dictionnaire des derniers prix
        self.last_prices: Dict[str, float] = {}
        self.last_update_times: Dict[str, datetime] = {}
        
        # Mutex pour protéger les accès aux prix
        self.price_lock = threading.RLock()
        
        # Thread de vérification pour les timeouts
        self.check_thread = None
        self.running = False
        
        logger.info(f"✅ PriceMonitor initialisé pour {len(self.symbols)} symboles")
    
    def _process_price_update(self, channel: str, data: Dict) -> None:
        """
        Traite une mise à jour de prix depuis Redis.
        
        Args:
            channel: Canal Redis
            data: Données de prix
        """
        try:
            # Vérifier que c'est bien une donnée de prix
            if not isinstance(data, dict) or 'symbol' not in data or 'close' not in data:
                return
            
            # Ne traiter que les chandeliers fermés si spécifié
            if data.get('is_closed', True) is False and data.get('type') == 'kline':
                return
            
            symbol = data.get('symbol', '').upper()
            price = float(data.get('close', 0))
            
            # Ignorer les prix invalides
            if price <= 0 or symbol not in self.symbols:
                return
            
            # Mettre à jour le prix
            with self.price_lock:
                old_price = self.last_prices.get(symbol)
                self.last_prices[symbol] = price
                self.last_update_times[symbol] = datetime.now()
            
            # Calculer le pourcentage de changement
            if old_price:
                change_pct = (price - old_price) / old_price * 100
                log_level = logging.INFO if abs(change_pct) >= 0.5 else logging.DEBUG
                logger.log(log_level, f"📊 Prix {symbol}: {price:.2f} ({change_pct:+.2f}%)")
            else:
                logger.info(f"📊 Premier prix {symbol}: {price:.2f}")
            
            # Appeler le callback
            self.price_update_callback(symbol, price)
            
        except Exception as e:
            # Gestion robuste de l'erreur
            error_msg = str(e) if hasattr(e, '__str__') else repr(e)
            logger.error(f"❌ Erreur lors du traitement d'une mise à jour de prix: {error_msg}", exc_info=True)
    
    def _check_price_timeouts(self) -> None:
        """
        Vérifie les timeouts des prix et générer des alertes si nécessaire.
        """
        while self.running:
            try:
                now = datetime.now()
                alerts = []
                
                with self.price_lock:
                    for symbol in self.symbols:
                        last_time = self.last_update_times.get(symbol)
                        if last_time:
                            # Vérifier si le dernier prix date de plus de 5 minutes
                            elapsed = (now - last_time).total_seconds()
                            if elapsed > 300:  # 5 minutes
                                alerts.append((symbol, elapsed))
                
                # Générer des alertes en dehors du lock
                for symbol, elapsed in alerts:
                    logger.warning(f"⚠️ Aucune mise à jour de prix pour {symbol} depuis {elapsed:.0f} secondes")
                
                # Vérifier toutes les 60 secondes
                time.sleep(60)
                
            except Exception as e:
                # Gestion robuste de l'erreur
                error_msg = str(e) if hasattr(e, '__str__') else repr(e)
                logger.error(f"❌ Erreur dans la vérification des timeouts: {error_msg}", exc_info=True)
                time.sleep(10)  # En cas d'erreur, attendre 10 secondes
    
    def start(self) -> None:
        """
        Démarre le moniteur de prix.
        """
        if self.running:
            logger.warning("Le moniteur de prix est déjà en cours d'exécution")
            return
        
        # Démarrer la surveillance
        self.running = True
        
        # S'abonner aux canaux Redis et sauvegarder le client_id
        self.client_id = self.redis_client.subscribe(self.price_channels, self._process_price_update)
        logger.info(f"✅ Abonné aux canaux de prix: {', '.join(self.price_channels)}")
        
        # Démarrer le thread de vérification des timeouts
        self.check_thread = threading.Thread(
            target=self._check_price_timeouts,
            daemon=True,
            name="PriceTimeoutChecker"
        )
        self.check_thread.start()
        
        logger.info("✅ Moniteur de prix démarré")
    
    def stop(self) -> None:
        """
        Arrête le moniteur de prix.
        """
        if not self.running:
            return
        
        logger.info("Arrêt du moniteur de prix...")
        self.running = False
        
        # Se désabonner des canaux Redis
        if hasattr(self, 'client_id') and self.client_id:
            self.redis_client.unsubscribe(self.client_id)
        self.redis_client.close()
        
        # Attendre la fin du thread de vérification
        if self.check_thread and self.check_thread.is_alive():
            self.check_thread.join(timeout=5.0)
        
        logger.info("✅ Moniteur de prix arrêté")
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """
        Récupère le dernier prix connu pour un symbole.
        
        Args:
            symbol: Symbole à récupérer
            
        Returns:
            Dernier prix ou None si non disponible
        """
        with self.price_lock:
            return self.last_prices.get(symbol)
    
    def get_all_prices(self) -> Dict[str, float]:
        """
        Récupère tous les derniers prix connus.
        
        Returns:
            Dictionnaire des derniers prix par symbole
        """
        with self.price_lock:
            return self.last_prices.copy()