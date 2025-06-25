"""
Module de gestion des signaux de trading - Version refactorisée.
Utilise les modules externes pour une meilleure modularité.
"""
import logging
import json
import threading
import time
import uuid
from typing import Dict, Any, Optional, List
import queue

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.redis_client import RedisClient
from shared.src.config import TRADING_MODE
from shared.src.enums import OrderSide, SignalStrength, CycleStatus
from shared.src.schemas import StrategySignal, TradeOrder

# Importer les nouveaux modules
from coordinator.src.service_client import ServiceClient
from coordinator.src.market_filter import MarketFilter
from coordinator.src.signal_processor import SignalProcessor
from coordinator.src.allocation_manager import AllocationManager
from coordinator.src.cycle_sync_monitor import CycleSyncMonitor
from coordinator.src.smart_cycle_manager import SmartCycleManager, CycleAction

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignalHandler:
    """
    Gestionnaire des signaux de trading - Version simplifiée.
    Délègue les responsabilités aux modules spécialisés.
    """
    
    def __init__(self, trader_api_url: str = "http://trader:5002", 
                 portfolio_api_url: str = "http://portfolio:8000"):
        """
        Initialise le gestionnaire de signaux.
        
        Args:
            trader_api_url: URL de l'API du service Trader
            portfolio_api_url: URL de l'API du service Portfolio
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialiser les clients et modules
        self.service_client = ServiceClient(trader_api_url, portfolio_api_url)
        self.market_filter = MarketFilter()
        self.signal_processor = SignalProcessor(self.service_client)
        self.allocation_manager = AllocationManager()
        
        # Redis pour recevoir les signaux
        self.redis_client = RedisClient()
        
        # Smart Cycle Manager pour les décisions intelligentes
        self.smart_cycle_manager = SmartCycleManager(trader_api_url, self.redis_client)
        
        # Moniteur de synchronisation des cycles
        self.cycle_sync_monitor = CycleSyncMonitor(trader_api_url)
        
        # File d'attente pour traiter les signaux
        self.signal_queue = queue.Queue(maxsize=1000)
        
        # État du service
        self.running = False
        self.signal_thread = None
        
        # Statistiques
        self.stats = {
            "signals_received": 0,
            "signals_processed": 0,
            "cycles_created": 0,
            "errors": 0,
            "start_time": time.time()
        }
        
        # Configuration
        self.min_amount_usdc = 50.0
        self.max_amount_usdc = 500.0
        
        self.logger.info(f"✅ SignalHandler initialisé en mode {TRADING_MODE}")
        
    def _process_signal(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Callback pour traiter les signaux reçus via Redis.
        
        Args:
            channel: Canal Redis
            data: Données du signal
        """
        try:
            self.stats["signals_received"] += 1
            
            # Parser le signal
            signal = StrategySignal(**data)
            
            # IMPORTANT: Vérifier si le signal vient du signal_aggregator
            # Si oui, il a déjà été filtré et validé
            if "filtered" in channel or signal.strategy.startswith("Aggregated_"):
                self.logger.info(f"📨 Signal filtré reçu: {signal.strategy} {signal.side} {signal.symbol}")
                # Traiter directement sans re-filtrage
                self._queue_signal_for_processing(signal)
            else:
                # Signal brut - appliquer nos filtres locaux
                self.logger.debug(f"Signal brut reçu: {signal.strategy} {signal.side} {signal.symbol}")
                
                # Appliquer le filtre de marché
                if not self.market_filter.should_filter_signal(signal):
                    self._queue_signal_for_processing(signal)
                else:
                    self.logger.info(f"Signal filtré par MarketFilter: {signal.symbol}")
                    
        except Exception as e:
            self.logger.error(f"Erreur traitement signal: {str(e)}")
            self.stats["errors"] += 1
            
    def _queue_signal_for_processing(self, signal: StrategySignal) -> None:
        """
        Ajoute un signal à la file de traitement.
        
        Args:
            signal: Signal à traiter
        """
        try:
            self.signal_queue.put(signal, block=False)
            self.logger.debug(f"Signal ajouté à la file: {signal.symbol}")
        except queue.Full:
            self.logger.warning("File de signaux pleine, signal ignoré")
            
    def _signal_processing_loop(self) -> None:
        """
        Boucle principale de traitement des signaux.
        """
        self.logger.info("🚀 Démarrage de la boucle de traitement des signaux")
        
        while self.running:
            try:
                # Récupérer un signal de la file (timeout de 1 seconde)
                signal = self.signal_queue.get(timeout=1.0)
                
                # Traiter le signal
                self._process_single_signal(signal)
                
            except queue.Empty:
                # Pas de signal, continuer
                continue
            except Exception as e:
                self.logger.error(f"Erreur dans la boucle de traitement: {str(e)}")
                self.stats["errors"] += 1
                time.sleep(0.1)
                
        self.logger.info("Arrêt de la boucle de traitement des signaux")
        
    def _process_single_signal(self, signal: StrategySignal) -> None:
        """
        Traite un signal individuel.
        
        Args:
            signal: Signal à traiter
        """
        try:
            self.logger.info(f"⚡ Traitement du signal {signal.strategy} {signal.side} {signal.symbol}")
            
            # 1. Valider le signal basique
            is_valid, reason = self.signal_processor.validate_signal(signal)
            if not is_valid:
                self.logger.warning(f"Signal invalide: {reason}")
                return
                
            # 2. Récupérer les données nécessaires pour SmartCycleManager
            current_price = self._get_current_price(signal.symbol)
            if not current_price:
                self.logger.error(f"Impossible d'obtenir le prix pour {signal.symbol}")
                return
                
            # 3. Récupérer toutes les balances et vérifier la faisabilité
            base_asset = self._get_base_asset(signal.symbol)
            quote_asset = self._get_quote_asset(signal.symbol)
            
            all_balances = self.service_client.get_all_balances()
            if not all_balances:
                self.logger.error("Impossible d'obtenir les balances")
                return
                
            # Vérifier la faisabilité du trade avec allocation dynamique
            feasibility = self.allocation_manager.check_trade_feasibility(
                signal=signal,
                balances=all_balances,
                base_asset=base_asset,
                quote_asset=quote_asset
            )
            
            if not feasibility['sufficient']:
                self.logger.warning(f"❌ Trade non faisable: {feasibility['reason']}")
                return
                
            available_balance = feasibility['constraining_balance']
                
            # 4. Récupérer les cycles existants
            existing_cycles = self.service_client.get_active_cycles(signal.symbol)
            
            # 5. Utiliser SmartCycleManager pour décider de l'action
            decision = self.smart_cycle_manager.analyze_signal(
                signal=signal,
                current_price=current_price,
                available_balance=available_balance,
                existing_cycles=existing_cycles
            )
            
            if not decision:
                self.logger.info("SmartCycleManager n'a pas pris de décision")
                return
                
            # 6. Exécuter la décision
            success = self._execute_smart_decision(decision, signal)
            
            if success:
                self.stats["signals_processed"] += 1
                self.logger.info(f"✅ Signal traité avec succès: {decision.action.value}")
            else:
                self.logger.error(f"❌ Échec du traitement: {decision.action.value}")
                
        except Exception as e:
            self.logger.error(f"Erreur traitement signal: {str(e)}")
            self.stats["errors"] += 1
            
    def _execute_smart_decision(self, decision, signal: StrategySignal) -> bool:
        """
        Exécute une décision du SmartCycleManager.
        
        Args:
            decision: Décision à exécuter
            signal: Signal original
            
        Returns:
            True si succès
        """
        try:
            if decision.action == CycleAction.CREATE_NEW:
                # Utiliser la fonction originale pour compatibilité DB
                cycle_id = self._create_trade_cycle(signal)
                return cycle_id is not None
                
            elif decision.action == CycleAction.REINFORCE:
                return self._reinforce_cycle(decision)
                
            elif decision.action == CycleAction.REDUCE:
                self.logger.warning("Réduction de position pas encore implémentée")
                return False
                
            elif decision.action == CycleAction.CLOSE:
                return self._close_cycle(decision)
                
            elif decision.action == CycleAction.WAIT:
                self.logger.info(f"Action WAIT: {decision.reason}")
                return True
                
            else:
                self.logger.warning(f"Action inconnue: {decision.action}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur exécution décision: {str(e)}")
            return False
            
    def _create_trade_cycle(self, signal: StrategySignal) -> Optional[str]:
        """
        Crée un cycle de trading à partir d'un signal.
        Version conservée pour compatibilité avec la DB.

        Args:
            signal: Signal de trading validé

        Returns:
            ID du cycle créé ou None en cas d'échec
        """
        try:
            # Calculer le montant à trader avec allocation manager
            base_asset = self._get_base_asset(signal.symbol)
            quote_asset = self._get_quote_asset(signal.symbol)
            
            # Récupérer toutes les balances
            all_balances = self.service_client.get_all_balances()
            if not all_balances:
                self.logger.error("Impossible d'obtenir les balances")
                return None
                
            # Vérifier la faisabilité et obtenir le montant optimal
            feasibility = self.allocation_manager.check_trade_feasibility(
                signal=signal,
                balances=all_balances,
                base_asset=base_asset,
                quote_asset=quote_asset
            )
            
            if not feasibility['sufficient']:
                self.logger.warning(f"❌ Trade non faisable: {feasibility['reason']}")
                return None
                
            trade_amount = feasibility['trade_amount']
            
            # Appliquer le multiplicateur de taille suggéré (danger level)
            if signal.metadata:
                size_multiplier = signal.metadata.get('suggested_size_multiplier', 1.0)
                if size_multiplier < 1.0:
                    original_amount = trade_amount
                    trade_amount *= size_multiplier
                    self.logger.info(f"📉 Taille réduite de {original_amount:.2f} à {trade_amount:.2f} {quote_asset} "
                                   f"(multiplicateur: {size_multiplier:.1%}, danger: {signal.metadata.get('danger_level', 'N/A')})")
            
            # Convertir le montant en quantité
            if signal.symbol.endswith("BTC"):
                # Pour les paires BTC, calcul spécial
                quantity = trade_amount / signal.price
                self.logger.debug(f"📊 Calcul quantité {signal.symbol}: {trade_amount:.6f} BTC / {signal.price:.6f} = {quantity:.6f}")
            else:
                # Pour les paires USDC, calcul direct
                quantity = trade_amount / signal.price

            # Préparer les données de l'ordre avec format original
            order_data = {
                "symbol": signal.symbol,
                "side": signal.side.value,
                "quantity": quantity,
                "price": signal.price,
                "strategy": signal.strategy,
                "timestamp": int(time.time() * 1000)  # Timestamp en millisecondes
            }
            
            # Ajouter les paramètres de stop si disponibles
            if signal.metadata:
                stop_price = signal.metadata.get('stop_price')
                trailing_delta = signal.metadata.get('trailing_delta')
                
                if stop_price:
                    order_data["stop_price"] = stop_price
                    
                if trailing_delta:
                    order_data["trailing_delta"] = trailing_delta

            # Créer le cycle via l'API du Trader avec le format exact attendu
            self.logger.info(f"Envoi de la requête au Trader: {order_data}")
            
            order_id = self.service_client.create_order(order_data)
            
            if order_id:
                self.stats["cycles_created"] += 1
                self.logger.info(f"✅ Cycle de trading créé: {order_id} ({signal.side} {signal.symbol})")
                return order_id
            else:
                self.logger.error("❌ Échec de la création du cycle: aucune réponse du Trader")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la création du cycle: {str(e)}")
            return None
            
    def _reinforce_cycle(self, decision) -> bool:
        """
        Renforce un cycle existant (DCA).
        
        Args:
            decision: Décision de renforcement
            
        Returns:
            True si succès
        """
        try:
            # Récupérer le cycle existant pour avoir le bon side
            existing_cycles = self.service_client.get_active_cycles(decision.symbol)
            target_cycle = None
            
            for cycle in existing_cycles:
                if cycle.get('id') == decision.existing_cycle_id:
                    target_cycle = cycle
                    break
                    
            if not target_cycle:
                self.logger.error(f"Cycle {decision.existing_cycle_id} non trouvé")
                return False
                
            # Déterminer le side du cycle (position actuelle, pas le prochain ordre)
            status = target_cycle.get('status', '')
            if status in ['waiting_sell', 'active_sell']:
                cycle_side = 'BUY'  # Position BUY ouverte
            elif status in ['waiting_buy', 'active_buy']:
                cycle_side = 'SELL'  # Position SELL ouverte
            else:
                self.logger.error(f"Statut de cycle invalide pour renforcement: {status}")
                return False
                
            # Calculer la quantité 
            quantity = decision.amount
            if decision.price_target and decision.price_target > 0:
                quantity = decision.amount / decision.price_target
                
            # Préparer les métadonnées
            metadata = {
                "reinforce_reason": decision.reason,
                "confidence": decision.confidence,
                "smart_cycle": True
            }
            
            # Appeler l'API de renforcement
            result = self.service_client.reinforce_cycle(
                cycle_id=decision.existing_cycle_id,
                symbol=decision.symbol,
                side=cycle_side,
                quantity=quantity,
                price=decision.price_target,
                metadata=metadata
            )
            
            if result.get('success'):
                self.logger.info(f"✅ Cycle {decision.existing_cycle_id} renforcé avec {quantity:.8f} unités")
                return True
            else:
                error_msg = result.get('error', 'Erreur inconnue')
                self.logger.error(f"❌ Échec du renforcement: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erreur renforcement cycle: {str(e)}")
            return False
        
    def _close_cycle(self, decision) -> bool:
        """
        Ferme un cycle existant.
        
        Args:
            decision: Décision de fermeture
            
        Returns:
            True si succès
        """
        # TODO: Implémenter la fermeture via l'API du trader
        self.logger.warning("Fermeture pas encore implémentée")
        return False
        
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Récupère le prix actuel d'un symbole.
        
        Args:
            symbol: Symbole à récupérer
            
        Returns:
            Prix actuel ou None
        """
        prices = self.service_client.get_current_prices([symbol])
        return prices.get(symbol)
        
    def _get_quote_asset(self, symbol: str) -> str:
        """
        Extrait l'actif de quote d'un symbole.
        
        Args:
            symbol: Symbole (ex: BTCUSDC)
            
        Returns:
            Actif de quote (ex: USDC)
        """
        # Pour la plupart des symboles, c'est USDC
        if symbol.endswith('USDC'):
            return 'USDC'
        elif symbol.endswith('USDT'):
            return 'USDT'
        elif symbol.endswith('BTC'):
            return 'BTC'
        elif symbol.endswith('ETH'):
            return 'ETH'
        else:
            # Par défaut, supposer USDC
            return 'USDC'
            
    def _get_base_asset(self, symbol: str) -> str:
        """
        Extrait l'actif de base d'un symbole.
        
        Args:
            symbol: Symbole (ex: BTCUSDC)
            
        Returns:
            Actif de base (ex: BTC)
        """
        # Retirer le quote asset pour obtenir le base asset
        quote_asset = self._get_quote_asset(symbol)
        if symbol.endswith(quote_asset):
            return symbol[:-len(quote_asset)]
        else:
            # Fallback: prendre tout sauf les 4 derniers caractères
            return symbol[:-4]
        
    def start(self) -> None:
        """
        Démarre le gestionnaire de signaux.
        """
        self.logger.info("🚀 Démarrage du SignalHandler")
        self.running = True
        
        # Démarrer le cycle sync monitor
        self.cycle_sync_monitor.start()
        
        # S'abonner aux canaux Redis pour les signaux
        signal_channels = [
            "roottrading:signals:filtered",  # Signaux filtrés du signal_aggregator (SEULE SOURCE)
        ]
        
        for channel in signal_channels:
            self.redis_client.subscribe(channel, self._process_signal)
            self.logger.info(f"📡 Abonné au canal: {channel}")
            
        # S'abonner aux événements de cycle pour maintenir le sync monitor
        cycle_channels = {
            "roottrading:trade:cycle:created": self.handle_cycle_created,
            "roottrading:trade:cycle:completed": self.handle_cycle_completed,
            "roottrading:trade:cycle:canceled": self.handle_cycle_canceled,
            "roottrading:trade:cycle:failed": self.handle_cycle_failed,
            "roottrading:trade:order:failed": self.handle_order_failed
        }
        
        for channel, handler in cycle_channels.items():
            self.redis_client.subscribe(channel, handler)
            self.logger.info(f"📡 Abonné aux événements: {channel}")
            
        # Démarrer le thread de traitement
        self.signal_thread = threading.Thread(
            target=self._signal_processing_loop,
            name="SignalProcessor"
        )
        self.signal_thread.daemon = True
        self.signal_thread.start()
        
        self.logger.info("✅ SignalHandler démarré")
        
    def stop(self) -> None:
        """
        Arrête le gestionnaire de signaux.
        """
        self.logger.info("Arrêt du SignalHandler...")
        self.running = False
        
        # Arrêter le cycle sync monitor
        if self.cycle_sync_monitor:
            self.cycle_sync_monitor.stop()
            
        # Attendre l'arrêt du thread
        if self.signal_thread and self.signal_thread.is_alive():
            self.signal_thread.join(timeout=5.0)
            
        # Se désabonner de Redis
        self.redis_client.unsubscribe_all()
        
        self.logger.info("SignalHandler arrêté")
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du handler.
        
        Returns:
            Dict des statistiques
        """
        return {
            **self.stats,
            "uptime": time.time() - self.stats["start_time"],
            "queue_size": self.signal_queue.qsize(),
            "service_health": self.service_client.get_service_health(),
            "signal_processor_metrics": self.signal_processor.get_metrics(),
            "market_filters": self.market_filter.get_filter_status()
        }
        
    # === Gestionnaires d'événements Redis ===
    
    def handle_order_failed(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite l'échec d'un ordre.
        
        Args:
            channel: Canal Redis
            data: Données de l'événement
        """
        order_id = data.get('order_id')
        reason = data.get('reason', 'Raison inconnue')
        self.logger.warning(f"❌ Ordre échoué: {order_id} - {reason}")
        self.stats["errors"] += 1
        
    def handle_cycle_created(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite la création d'un cycle.
        
        Args:
            channel: Canal Redis
            data: Données de l'événement
        """
        cycle_id = data.get('cycle_id')
        symbol = data.get('symbol', 'Unknown')
        self.logger.info(f"✅ Cycle créé: {cycle_id} pour {symbol}")
        self.stats["cycles_created"] += 1
        
    def handle_cycle_completed(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite la completion d'un cycle.
        
        Args:
            channel: Canal Redis
            data: Données de l'événement
        """
        cycle_id = data.get('cycle_id')
        profit = data.get('profit', 0)
        
        self.logger.info(f"✅ Cycle terminé: {cycle_id} - Profit: {profit}")
        
        # Mettre à jour le cache du sync monitor
        if hasattr(self, 'cycle_sync_monitor') and self.cycle_sync_monitor:
            self.cycle_sync_monitor.remove_cycle_from_cache(cycle_id)
            self.logger.debug(f"🔄 Cycle {cycle_id} retiré du cache du sync monitor")
            
    def handle_cycle_canceled(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite l'annulation d'un cycle.
        
        Args:
            channel: Canal Redis
            data: Données de l'événement
        """
        cycle_id = data.get('cycle_id')
        self.logger.info(f"🚫 Cycle annulé: {cycle_id}")
        
        # Mettre à jour le cache du sync monitor
        if hasattr(self, 'cycle_sync_monitor') and self.cycle_sync_monitor:
            self.cycle_sync_monitor.remove_cycle_from_cache(cycle_id)
            self.logger.debug(f"🔄 Cycle {cycle_id} retiré du cache du sync monitor")
            
    def handle_cycle_failed(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite l'échec d'un cycle.
        
        Args:
            channel: Canal Redis
            data: Données de l'événement
        """
        cycle_id = data.get('cycle_id')
        self.logger.info(f"❌ Cycle échoué: {cycle_id}")
        
        # Mettre à jour le cache du sync monitor
        if hasattr(self, 'cycle_sync_monitor') and self.cycle_sync_monitor:
            self.cycle_sync_monitor.remove_cycle_from_cache(cycle_id)
            self.logger.debug(f"🔄 Cycle {cycle_id} retiré du cache du sync monitor")