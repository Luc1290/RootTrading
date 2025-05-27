"""
Module de gestion des signaux de trading.
Reçoit les signaux, les valide, et coordonne la création des cycles de trading.
"""
import logging
import json
import requests
import threading
import time
from typing import Dict, Any, Optional, List
import queue

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.redis_client import RedisClient
from shared.src.config import TRADING_MODE
from shared.src.enums import OrderSide, SignalStrength, CycleStatus
from coordinator.src.cycle_sync_monitor import CycleSyncMonitor
from shared.src.schemas import StrategySignal, TradeOrder

from coordinator.src.pocket_checker import PocketChecker

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SignalHandler:
    """
    Gestionnaire des signaux de trading.
    Reçoit les signaux, les valide, et coordonne la création des cycles de trading.
    """
    
    def __init__(self, trader_api_url: str = "http://trader:5002", 
                 portfolio_api_url: str = "http://portfolio:8000"):
        """
        Initialise le gestionnaire de signaux.
        
        Args:
            trader_api_url: URL de l'API du service Trader
            portfolio_api_url: URL de l'API du service Portfolio
        """
        self.trader_api_url = trader_api_url
        self.portfolio_api_url = portfolio_api_url
        
        # Client Redis pour les communications
        self.redis_client = RedisClient()
        self.redis_client.subscribe("roottrading:order:failed", self.handle_order_failed)
        
        # S'abonner aux événements de cycles pour rester synchronisé
        self.redis_client.subscribe("roottrading:cycle:created", self.handle_cycle_created)
        self.redis_client.subscribe("roottrading:cycle:closed", self.handle_cycle_closed)
        self.redis_client.subscribe("roottrading:cycle:canceled", self.handle_cycle_canceled)
        self.redis_client.subscribe("roottrading:cycle:failed", self.handle_cycle_failed)
        
        # Canal Redis pour les signaux
        self.signal_channel = "roottrading:analyze:signal"
        
        # File d'attente thread-safe pour les signaux
        self.signal_queue = queue.Queue()
        
        # Thread pour le traitement des signaux
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # Gestionnaire de poches
        self.pocket_checker = PocketChecker(portfolio_api_url)
        
        # Moniteur de synchronisation des cycles (solution définitive)
        self.sync_monitor = CycleSyncMonitor(
            trader_api_url=trader_api_url,
            check_interval=30  # Vérification toutes les 30 secondes
        )
        
        # Cache des prix actuels
        self.price_cache = {}
        
        # Configuration du mode de trading
        self.demo_mode = TRADING_MODE.lower() == 'demo'
        
        # Stratégies spéciales pour le filtrage
        self.filter_strategies = ['Ride_or_React_Strategy']
        self.market_filters = {}  # {symbol: {filter_data}}

        # Circuit breakers pour éviter les appels répétés à des services en échec
        self.trader_circuit = CircuitBreaker()
        self.portfolio_circuit = CircuitBreaker()
        
        # S'abonner aux notifications de mise à jour du portfolio (nouvelle fonctionnalité)
        self.redis_client.subscribe(
            "roottrading:notification:balance_updated", 
            self._handle_portfolio_update
        )
        
        # === NOUVEAU: Gestion intelligente des signaux multiples ===
        # Agrégation des signaux par symbole
        self.signal_aggregator = {}  # {symbol: {timestamp: [signals]}}
        self.aggregation_window = 1.0  # Réduit à 1 seconde pour plus de réactivité
        self.aggregator_lock = threading.RLock()
        
        # Cache des cycles actifs pour éviter les doublons
        self.active_cycles_cache = {}  # {symbol: {side: cycle_count}}
        self.cache_update_time = 0
        self.cache_ttl = 30  # TTL du cache en secondes
        
        # Configuration des limites
        self.max_cycles_per_symbol_side = 3  # Max 3 BUY ou 3 SELL par symbole
        self.contradiction_threshold = 0.7  # Seuil pour décider en cas de contradiction
        
        # NOUVEAU: Historique récent des signaux pour détecter les patterns
        self.recent_signals_history = {}  # {symbol: [(signal, timestamp)]}
        self.history_window = 30.0  # Garder 30 secondes d'historique
        
        # Thread pour l'agrégation périodique
        self.aggregation_thread = None
        
        logger.info(f"✅ SignalHandler initialisé en mode {'DÉMO' if self.demo_mode else 'RÉEL'}")
    
    def _handle_portfolio_update(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Gère les notifications de mise à jour du portfolio.
        Déclenche une mise à jour du cache et une réallocation.
        
        Args:
            channel: Canal Redis
            data: Données de notification
        """
        try:
            logger.info(f"📢 Notification de mise à jour du portfolio reçue: {data}")
            
            # Invalider le cache des poches
            if hasattr(self, 'pocket_checker') and self.pocket_checker:
                self.pocket_checker.last_cache_update = 0
                
                # Déclencher une réallocation des fonds
                logger.info("Réallocation des fonds suite à notification...")
                self.pocket_checker.reallocate_funds()
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de la notification: {str(e)}")
    
    def _process_signal(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Callback pour traiter les signaux reçus de Redis.
        Ajoute les signaux à la file d'attente pour traitement.
    
        Args:
            channel: Canal Redis d'où provient le signal
            data: Données du signal
        """
        try:
            # Convertir les chaînes de caractères en énumérations avant de créer le signal
            if 'side' in data and isinstance(data['side'], str):
                try:
                    data['side'] = OrderSide(data['side'])
                except ValueError:
                    logger.error(f"Valeur d'énumération invalide pour side: {data['side']}")
                    return
                
            if 'strength' in data and isinstance(data['strength'], str):
                try:
                    data['strength'] = SignalStrength(data['strength'])
                except ValueError:
                    logger.error(f"Valeur d'énumération invalide pour strength: {data['strength']}")
                    # Utiliser une valeur par défaut au lieu d'abandonner
                    data['strength'] = SignalStrength.MODERATE
        
            # Valider le signal
            signal = StrategySignal(**data)
        
            # Traiter les signaux de filtrage séparément
            if signal.strategy in self.filter_strategies:
                self._update_market_filters(signal)
                return
        
            # NOUVEAU: Ajouter le signal à l'agrégateur au lieu de la file directe
            self._add_signal_to_aggregator(signal)
        
            # Mettre à jour le cache des prix
            self.price_cache[signal.symbol] = signal.price
        
            # Logger les métadonnées pour debug
            if signal.metadata and ('target_price' in signal.metadata or 'stop_price' in signal.metadata):
                logger.info(f"📨 Signal reçu: {signal.side} {signal.symbol} @ {signal.price} ({signal.strategy}) - Target: {signal.metadata.get('target_price', 'N/A')}, Stop: {signal.metadata.get('stop_price', 'N/A')}")
            else:
                logger.info(f"📨 Signal reçu: {signal.side} {signal.symbol} @ {signal.price} ({signal.strategy})")
    
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement du signal: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _add_signal_to_aggregator(self, signal: StrategySignal) -> None:
        """
        Ajoute un signal à l'agrégateur avec logique intelligente.
        
        Args:
            signal: Signal à ajouter
        """
        with self.aggregator_lock:
            current_time = time.time()
            
            # Ajouter à l'historique
            if signal.symbol not in self.recent_signals_history:
                self.recent_signals_history[signal.symbol] = []
            
            # Nettoyer l'historique ancien
            self.recent_signals_history[signal.symbol] = [
                (s, t) for s, t in self.recent_signals_history[signal.symbol]
                if current_time - t < self.history_window
            ]
            
            # Ajouter le nouveau signal
            self.recent_signals_history[signal.symbol].append((signal, current_time))
            
            # Vérifier s'il y a des signaux récents contradictoires
            recent_signals = [s for s, t in self.recent_signals_history[signal.symbol] 
                            if current_time - t < 3.0]  # Signaux des 3 dernières secondes
            
            contradictory = any(s.side != signal.side for s in recent_signals[:-1])
            
            # Si signal contradictoire récent OU plusieurs signaux proches, agréger
            if contradictory or len(recent_signals) > 1:
                # Agrégation nécessaire
                if signal.symbol not in self.signal_aggregator:
                    self.signal_aggregator[signal.symbol] = {}
                
                time_key = int(current_time / self.aggregation_window) * self.aggregation_window
                
                if time_key not in self.signal_aggregator[signal.symbol]:
                    self.signal_aggregator[signal.symbol][time_key] = []
                
                self.signal_aggregator[signal.symbol][time_key].append(signal)
                logger.info(f"📊 Signal ajouté à l'agrégation ({len(recent_signals)} signaux récents)")
            else:
                # Signal isolé, traiter immédiatement
                logger.info(f"⚡ Signal isolé, traitement immédiat")
                self._process_single_signal(signal)
    
    def _process_single_signal(self, signal: StrategySignal) -> None:
        """
        Traite immédiatement un signal isolé en vérifiant les positions existantes.
        
        Args:
            signal: Signal à traiter
        """
        # Rafraîchir le cache des positions
        self._refresh_active_cycles_cache()
        
        # Convertir signal.side en string si c'est un enum
        signal_side_str = signal.side.value if hasattr(signal.side, 'value') else str(signal.side)
        
        # Vérifier les positions existantes pour ce symbole
        opposite_side = "SELL" if signal_side_str == "BUY" else "BUY"
        existing_opposite = self.active_cycles_cache.get(signal.symbol, {}).get(opposite_side, 0)
        existing_same = self.active_cycles_cache.get(signal.symbol, {}).get(signal_side_str, 0)
        
        # Cas 1: Signal opposé à des positions existantes
        if existing_opposite > 0:
            logger.warning(f"⚠️ Signal {signal.side} contradictoire avec {existing_opposite} "
                         f"positions {opposite_side} ouvertes sur {signal.symbol}")
            
            # Options possibles selon la force du signal
            if signal.strength == SignalStrength.VERY_STRONG:
                logger.info(f"🔄 Signal TRÈS FORT - Tentative de fermeture des positions opposées")
                # Fermer les positions opposées si le signal est très fort
                if self._close_opposite_positions(signal.symbol, opposite_side):
                    logger.info(f"✅ Positions {opposite_side} fermées, nouveau signal {signal.side} autorisé")
                    # Continuer avec le nouveau signal après fermeture
                else:
                    logger.warning(f"❌ Impossible de fermer les positions opposées, signal ignoré")
                    return
            elif signal.strength == SignalStrength.STRONG and existing_opposite == 1:
                # Si seulement 1 position opposée et signal fort, on peut considérer la fermeture
                logger.info(f"🤔 Signal FORT avec 1 position opposée - Évaluation...")
                # Pour l'instant on bloque, mais on pourrait être plus flexible
                return
            else:
                logger.info(f"❌ Signal contradictoire ignoré (positions opposées actives)")
                return
        
        # Cas 2: Vérifier si on peut ajouter à la position existante
        if existing_same >= self.max_cycles_per_symbol_side:
            logger.warning(f"❌ Limite de {self.max_cycles_per_symbol_side} cycles {signal.side} "
                         f"atteinte pour {signal.symbol}")
            return
        
        # Cas 3: Vérifier l'anti-spam
        if signal.symbol in self.recent_signals_history:
            same_side_recent = [
                s for s, t in self.recent_signals_history[signal.symbol]
                if s.side == signal.side and time.time() - t < 10.0
            ]
            
            if len(same_side_recent) > 2:
                logger.warning(f"⚠️ Trop de signaux {signal.side} récents pour {signal.symbol}")
                return
        
        # Cas 4: Signal renforcant une position existante
        if existing_same > 0 and signal.strength >= SignalStrength.STRONG:
            logger.info(f"💪 Signal renforçant {existing_same} position(s) {signal.side} existante(s)")
            # On laisse passer pour pyramider la position
        
        # Ajouter à la file de traitement
        self.signal_queue.put(signal)
        logger.info(f"✅ Signal ajouté à la file: {signal.side} {signal.symbol} "
                   f"(Positions: {existing_same} same, {existing_opposite} opposite)")
    
    def _close_opposite_positions(self, symbol: str, side: str) -> bool:
        """
        Ferme les positions du côté opposé pour un symbole donné.
        
        Args:
            symbol: Symbole concerné
            side: Côté des positions à fermer (BUY ou SELL)
            
        Returns:
            True si toutes les positions ont été fermées avec succès
        """
        try:
            # Récupérer les cycles actifs depuis l'API centralisée
            response = self._make_request_with_retry(
                f"{self.trader_api_url}/cycles",
                method="GET",
                params={"symbol": symbol, "confirmed": "true", "include_completed": "false"},
                timeout=5.0
            )
            
            if not response or not response.get('success'):
                logger.error("Impossible de récupérer les cycles pour fermeture")
                return False
            
            # Extraire les cycles depuis la réponse
            cycles = response.get('cycles', [])
            
            # Filtrer les cycles du côté concerné
            cycles_to_close = []
            for cycle in cycles:
                # Déterminer le côté en fonction du statut
                status = cycle.get('status', '')
                if (side == 'BUY' and status in ['waiting_buy', 'active_buy']) or \
                   (side == 'SELL' and status in ['waiting_sell', 'active_sell']):
                    cycles_to_close.append(cycle)
            
            if not cycles_to_close:
                logger.info(f"Aucun cycle {side} à fermer pour {symbol}")
                return True
            
            # Fermer chaque cycle
            success_count = 0
            for cycle in cycles_to_close:
                cycle_id = cycle.get('id')
                if not cycle_id:
                    continue
                
                # Appeler l'API pour fermer le cycle
                close_response = self._make_request_with_retry(
                    f"{self.trader_api_url}/close/{cycle_id}",
                    method="POST",
                    json_data={},
                    timeout=10.0
                )
                
                if close_response:
                    logger.info(f"✅ Cycle {cycle_id} fermé avec succès")
                    success_count += 1
                else:
                    logger.error(f"❌ Échec de fermeture du cycle {cycle_id}")
            
            # Mettre à jour le cache
            self.cache_update_time = 0  # Forcer le rafraîchissement
            
            return success_count == len(cycles_to_close)
            
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture des positions: {str(e)}")
            return False
    
    def _refresh_active_cycles_cache(self) -> None:
        """
        Rafraîchit le cache des cycles actifs depuis l'API centralisée du Trader.
        SEULE SOURCE DE VÉRITÉ pour les cycles.
        """
        try:
            if time.time() - self.cache_update_time < self.cache_ttl:
                return  # Cache encore valide
            
            # Récupérer les cycles actifs depuis l'API centralisée
            response = self._make_request_with_retry(
                f"{self.trader_api_url}/cycles",
                method="GET",
                params={"confirmed": "true", "include_completed": "false"},
                timeout=5.0
            )
            
            if not response or not response.get('success'):
                logger.warning("Impossible de récupérer les cycles actifs depuis l'API centralisée")
                # En cas d'échec, vider le cache pour éviter d'utiliser des données obsolètes
                self.active_cycles_cache = {}
                self.cache_update_time = time.time()
                return
            
            # Réinitialiser le cache
            self.active_cycles_cache = {}
            
            # Extraire les cycles depuis la réponse
            cycles = response.get('cycles', [])
            
            # Compter les cycles par symbole et côté
            for cycle in cycles:
                symbol = cycle.get('symbol')
                
                # Déterminer le côté en fonction du statut
                status = cycle.get('status', '')
                if status in ['waiting_buy', 'active_buy']:
                    side = 'BUY'
                elif status in ['waiting_sell', 'active_sell']:
                    side = 'SELL'
                else:
                    continue  # Ignorer les autres statuts
                
                if symbol:
                    if symbol not in self.active_cycles_cache:
                        self.active_cycles_cache[symbol] = {'BUY': 0, 'SELL': 0}
                    
                    self.active_cycles_cache[symbol][side] = self.active_cycles_cache[symbol].get(side, 0) + 1
            
            self.cache_update_time = time.time()
            logger.debug(f"Cache des cycles actifs mis à jour depuis l'API centralisée: {self.active_cycles_cache}")
            
        except Exception as e:
            logger.error(f"Erreur lors du rafraîchissement du cache: {str(e)}")
    
    def _can_create_new_cycle(self, symbol: str, side: str) -> bool:
        """
        Vérifie s'il est possible de créer un nouveau cycle.
        
        Args:
            symbol: Symbole du trade
            side: Côté du trade (BUY/SELL)
            
        Returns:
            True si on peut créer un nouveau cycle
        """
        # Ne pas bloquer si on ne peut pas récupérer les cycles
        # (mieux vaut laisser passer que bloquer tout)
        if not self.active_cycles_cache:
            logger.warning("Cache vide, autorisation du cycle par défaut")
            return True
        
        if symbol not in self.active_cycles_cache:
            return True  # Aucun cycle actif pour ce symbole
        
        current_count = self.active_cycles_cache[symbol].get(side, 0)
        
        # Permettre plus de cycles en cas de signaux très forts
        if current_count >= self.max_cycles_per_symbol_side:
            logger.warning(f"Limite atteinte: {current_count} cycles {side} actifs pour {symbol}")
            return False
        
        return True
    
    def _analyze_aggregated_signals(self) -> None:
        """
        Analyse les signaux agrégés et décide lesquels traiter.
        Cette méthode s'exécute périodiquement dans un thread séparé.
        """
        while not self.stop_event.is_set():
            try:
                time.sleep(self.aggregation_window)
                
                with self.aggregator_lock:
                    current_time = time.time()
                    
                    # Traiter chaque symbole
                    for symbol in list(self.signal_aggregator.keys()):
                        # Traiter chaque fenêtre temporelle
                        for time_key in list(self.signal_aggregator[symbol].keys()):
                            # Si la fenêtre est complète (assez ancienne)
                            if current_time - time_key >= self.aggregation_window:
                                signals = self.signal_aggregator[symbol].pop(time_key)
                                
                                if signals:
                                    # Analyser et traiter les signaux groupés
                                    self._process_aggregated_signals(symbol, signals)
            
            except Exception as e:
                logger.error(f"Erreur dans l'analyse des signaux agrégés: {str(e)}")
                time.sleep(1)
    
    def _process_aggregated_signals(self, symbol: str, signals: List[StrategySignal]) -> None:
        """
        Traite un groupe de signaux pour un symbole donné.
        
        Args:
            symbol: Symbole concerné
            signals: Liste des signaux à analyser
        """
        if not signals:
            return
        
        logger.info(f"🔍 Analyse de {len(signals)} signaux pour {symbol}")
        
        # Séparer les signaux par côté
        buy_signals = [s for s in signals if s.side == OrderSide.BUY]
        sell_signals = [s for s in signals if s.side == OrderSide.SELL]
        
        # Cas 1: Signaux contradictoires
        if buy_signals and sell_signals:
            logger.warning(f"⚠️ Signaux contradictoires détectés pour {symbol}: "
                         f"{len(buy_signals)} BUY vs {len(sell_signals)} SELL")
            
            # Calculer les scores moyens
            buy_score = self._calculate_signal_score(buy_signals)
            sell_score = self._calculate_signal_score(sell_signals)
            
            # Si la différence est significative, suivre le plus fort
            if abs(buy_score - sell_score) > self.contradiction_threshold:
                if buy_score > sell_score:
                    self._process_buy_signals(symbol, buy_signals)
                else:
                    self._process_sell_signals(symbol, sell_signals)
            else:
                logger.info(f"🤷 Signaux trop contradictoires, aucune action pour {symbol}")
        
        # Cas 2: Signaux unanimes BUY
        elif buy_signals:
            self._process_buy_signals(symbol, buy_signals)
        
        # Cas 3: Signaux unanimes SELL
        elif sell_signals:
            self._process_sell_signals(symbol, sell_signals)
    
    def _calculate_signal_score(self, signals: List[StrategySignal]) -> float:
        """
        Calcule un score pondéré pour un groupe de signaux.
        
        Args:
            signals: Liste de signaux
            
        Returns:
            Score moyen pondéré
        """
        if not signals:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for signal in signals:
            # Pondération basée sur la force et la confiance
            strength_weight = {
                SignalStrength.WEAK: 0.25,
                SignalStrength.MODERATE: 0.5,
                SignalStrength.STRONG: 0.75,
                SignalStrength.VERY_STRONG: 1.0
            }.get(signal.strength, 0.5)
            
            weight = strength_weight * signal.confidence
            total_score += weight
            total_weight += 1
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _process_buy_signals(self, symbol: str, signals: List[StrategySignal]) -> None:
        """
        Traite un groupe de signaux BUY.
        
        Args:
            symbol: Symbole
            signals: Signaux BUY
        """
        # Vérifier si on peut créer un nouveau cycle
        if not self._can_create_new_cycle(symbol, "BUY"):
            logger.warning(f"❌ Impossible de créer plus de cycles BUY pour {symbol}")
            return
        
        # Choisir le meilleur signal ou créer un signal composite
        best_signal = self._select_best_signal(signals)
        
        # Si plus de 3 stratégies sont d'accord et signal très fort, possibilité de double position
        if len(signals) >= 3 and best_signal.strength == SignalStrength.VERY_STRONG:
            logger.info(f"🚀 Signal de consensus fort détecté pour {symbol} ({len(signals)} stratégies)")
            # On pourrait créer 2 positions ici si vraiment fort
        
        # Ajouter à la file de traitement normale
        self.signal_queue.put(best_signal)
    
    def _process_sell_signals(self, symbol: str, signals: List[StrategySignal]) -> None:
        """
        Traite un groupe de signaux SELL.
        
        Args:
            symbol: Symbole
            signals: Signaux SELL
        """
        # Vérifier si on peut créer un nouveau cycle
        if not self._can_create_new_cycle(symbol, "SELL"):
            logger.warning(f"❌ Impossible de créer plus de cycles SELL pour {symbol}")
            return
        
        # Choisir le meilleur signal
        best_signal = self._select_best_signal(signals)
        
        # Ajouter à la file de traitement normale
        self.signal_queue.put(best_signal)
    
    def _select_best_signal(self, signals: List[StrategySignal]) -> StrategySignal:
        """
        Sélectionne le meilleur signal parmi une liste.
        
        Args:
            signals: Liste de signaux
            
        Returns:
            Meilleur signal
        """
        if not signals:
            return None
        
        # Trier par force puis par confiance
        sorted_signals = sorted(
            signals,
            key=lambda s: (s.strength.value if hasattr(s.strength, 'value') else str(s.strength), s.confidence),
            reverse=True
        )
        
        best = sorted_signals[0]
        
        # Enrichir les métadonnées avec les infos d'agrégation
        if not best.metadata:
            best.metadata = {}
        
        best.metadata['aggregated_count'] = len(signals)
        best.metadata['strategies'] = [s.strategy for s in signals]
        best.metadata['consensus_score'] = self._calculate_signal_score(signals)
        
        return best
    
    def _update_market_filters(self, signal: StrategySignal) -> None:
        """
        Met à jour les filtres de marché basés sur des stratégies spéciales comme Ride or React.
        Version améliorée avec meilleure gestion de l'obsolescence.
        
        Args:
            signal: Signal de la stratégie de filtrage
        """
        if signal.strategy in self.filter_strategies:
            # Vérifier que les métadonnées sont présentes
            if not signal.metadata:
                logger.warning(f"Signal de filtrage sans métadonnées reçu pour {signal.symbol}, ignoré")
                return
            
            # Stocker les informations de mode dans le dictionnaire de filtres
            mode = signal.metadata.get('mode', 'react')
            action = signal.metadata.get('action', 'normal_trading')
            
            # Vérifier si les données sont cohérentes
            if mode not in ['ride', 'react', 'neutral']:
                logger.warning(f"Mode de filtrage inconnu: {mode}, utilisation de 'react' par défaut")
                mode = 'react'
            
            # Mapper wait_for_reversal vers no_trading
            if action == 'wait_for_reversal':
                action = 'no_trading'
            
            if action not in ['normal_trading', 'no_trading', 'buy_only', 'sell_only']:
                logger.warning(f"Action de filtrage inconnue: {action}, utilisation de 'normal_trading' par défaut")
                action = 'normal_trading'
            
            # Mise à jour du filtre avec les nouvelles données
            self.market_filters[signal.symbol] = {
                'mode': mode,
                'action': action,
                'updated_at': time.time(),
                'is_obsolete': False,
                'source': signal.strategy
            }
            
            # Si des infos supplémentaires sont disponibles, les stocker aussi
            if 'trend_strength' in signal.metadata:
                self.market_filters[signal.symbol]['strength'] = float(signal.metadata['trend_strength'])
            
            logger.info(f"🔍 Filtre de marché mis à jour pour {signal.symbol}: "
                    f"mode={mode}, action={action}")
            
            # Publier la mise à jour sur Redis pour informer les autres composants
            try:
                from shared.src.redis_client import RedisClient
                redis_client = RedisClient()
                redis_client.publish("roottrading:market:filters", {
                    "symbol": signal.symbol,
                    "mode": mode,
                    "action": action,
                    "updated_at": time.time(),
                    "source": signal.strategy
                })
            except Exception as e:
                logger.warning(f"⚠️ Impossible de publier la mise à jour de filtre sur Redis: {str(e)}")
    
    def _should_filter_signal(self, signal: StrategySignal) -> bool:
        """
        Détermine si un signal doit être filtré en fonction des conditions de marché.
        Version améliorée avec gestion de l'obsolescence des filtres.
        
        Args:
            signal: Signal à évaluer
            
        Returns:
            True si le signal doit être filtré (ignoré), False sinon
        """
        # Vérifier si nous avons des informations de filtrage pour ce symbole
        if signal.symbol not in self.market_filters:
            # Aucune information de filtrage, essayer de récupérer des données récentes
            self._refresh_market_filter(signal.symbol)
            return False  # Ne pas filtrer si pas de données
        
        filter_info = self.market_filters[signal.symbol]
        
        # Vérifier si les informations de filtrage sont récentes
        # Réduire à 15 minutes (900 secondes) au lieu de 30 minutes
        max_age = 900  # 15 minutes
        if time.time() - filter_info.get('updated_at', 0) > max_age:
            logger.warning(f"Informations de filtrage obsolètes pour {signal.symbol}, tentative de rafraîchissement")
            
            # Essayer de rafraîchir les données de filtrage
            refreshed = self._refresh_market_filter(signal.symbol)
            
            if not refreshed:
                # Si le rafraîchissement échoue, utiliser un mode de fallback basé sur la force du signal
                logger.warning(f"Impossible de rafraîchir les informations de filtrage pour {signal.symbol}, utilisation du mode de secours")
                
                # En mode de secours, n'ignorer que les signaux très faibles
                if signal.strength == SignalStrength.WEAK:
                    logger.info(f"Signal {signal.side} ignoré en mode de secours (force insuffisante)")
                    return True
                
                # Laisser passer les autres signaux
                return False
            
            # Récupérer les informations rafraîchies
            filter_info = self.market_filters[signal.symbol]
        
        # En mode "ride", filtrer certains signaux contre-tendance
        if filter_info.get('mode') == 'ride':
            # Si dans une tendance haussière forte, filtrer les signaux SELL (sauf très forts)
            if signal.side == OrderSide.SELL and signal.strength != SignalStrength.VERY_STRONG:
                logger.info(f"🔍 Signal {signal.side} filtré: marché en mode RIDE pour {signal.symbol}")
                return True
        # En mode "react", aucun filtrage supplémentaire n'est nécessaire
        
        # Si une action spécifique est recommandée
        if 'action' in filter_info:
            action = filter_info.get('action')
            
            # Si l'action est "no_trading", filtrer tous les signaux
            if action == 'no_trading':
                logger.info(f"🔍 Signal {signal.side} filtré: action 'no_trading' active pour {signal.symbol}")
                return True
            
            # Si l'action est "buy_only", filtrer les signaux de vente
            elif action == 'buy_only' and signal.side == OrderSide.SELL:
                logger.info(f"🔍 Signal {signal.side} filtré: seuls les achats sont autorisés pour {signal.symbol}")
                return True
            
            # Si l'action est "sell_only", filtrer les signaux d'achat
            elif action == 'sell_only' and signal.side == OrderSide.BUY:
                logger.info(f"🔍 Signal {signal.side} filtré: seules les ventes sont autorisées pour {signal.symbol}")
                return True
        
        # Si aucune condition de filtrage n'a été rencontrée
        return False
    
    def _refresh_market_filter(self, symbol: str) -> bool:
        """
        Tente de rafraîchir les informations de filtrage pour un symbole.
        
        Args:
            symbol: Symbole pour lequel rafraîchir les données
            
        Returns:
            True si le rafraîchissement a réussi, False sinon
        """
        try:
            # Vérifier si le circuit breaker est ouvert
            if not self.trader_circuit.can_execute():
                logger.warning(f"Circuit breaker actif, impossible de rafraîchir les filtres")
                return False
            
            # Récupérer les dernières données de marché
            url = f"{self.trader_api_url}/market/filter/{symbol}"
            filter_data = self._make_request_with_retry(url, timeout=2.0)
            
            if not filter_data:
                logger.warning(f"Aucune donnée de filtrage disponible pour {symbol}")
                return False
            
            # Mettre à jour le filtre avec les nouvelles données
            self.market_filters[symbol] = {
                'mode': filter_data.get('mode', 'react'),  # Mode par défaut: react
                'action': filter_data.get('action', 'normal_trading'),
                'strength': filter_data.get('trend_strength', 0.0),
                'updated_at': time.time()  # Mettre à jour le timestamp
            }
            
            logger.info(f"✅ Informations de filtrage rafraîchies pour {symbol}: mode={self.market_filters[symbol]['mode']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du rafraîchissement des filtres pour {symbol}: {str(e)}")
            
            # En cas d'échec, marquer le filtre comme obsolète mais ne pas le supprimer complètement
            if symbol in self.market_filters:
                # Conserver les anciennes données mais les marquer comme explicitement obsolètes
                self.market_filters[symbol]['is_obsolete'] = True
            
            return False
    
    def _get_quote_asset(self, symbol: str) -> str:
        """
        Détermine l'actif de cotation (quote asset) pour un symbole.
        
        Args:
            symbol: Symbole de trading (ex: BTCUSDC, ETHBTC)
            
        Returns:
            L'actif de cotation (USDC, BTC, ETH, etc.)
        """
        # Pour les paires communes
        if symbol.endswith('USDC'):
            return 'USDC'
        elif symbol.endswith('BTC'):
            return 'BTC'
        elif symbol.endswith('ETH'):
            return 'ETH'
        elif symbol.endswith('BNB'):
            return 'BNB'
        else:
            # Par défaut, on suppose USDC
            logger.warning(f"Impossible de déterminer l'actif de quote pour {symbol}, utilisation de USDC par défaut")
            return 'USDC'
    
    def _calculate_trade_amount(self, signal: StrategySignal) -> tuple[float, str]:
        """
        Calcule le montant à trader basé sur le signal et l'actif.
        
        Args:
            signal: Signal de trading
            
        Returns:
            Tuple (montant, actif) - ex: (100.0, 'USDC') ou (0.001, 'BTC')
        """
        quote_asset = self._get_quote_asset(signal.symbol)
        
        # Valeurs par défaut selon l'actif
        default_amounts = {
            'USDC': 100.0,     # 100 USDC
            'BTC': 0.001,      # 0.001 BTC (~110 USDC au prix actuel)
            'ETH': 0.04,       # 0.04 ETH (~100 USDC au prix actuel)
            'BNB': 0.2         # 0.2 BNB (~100 USDC au prix actuel)
        }
        
        default_amount = default_amounts.get(quote_asset, 100.0)
        
        # Ajuster en fonction de la force du signal
        if signal.strength == SignalStrength.WEAK:
            amount = default_amount * 0.5
        elif signal.strength == SignalStrength.MODERATE:
            amount = default_amount * 0.8
        elif signal.strength == SignalStrength.STRONG:
            amount = default_amount * 1.0
        elif signal.strength == SignalStrength.VERY_STRONG:
            amount = default_amount * 1.2
        else:
            amount = default_amount
        
        return amount, quote_asset
    
    def _make_request_with_retry(self, url, method="GET", json_data=None, params=None, max_retries=3, timeout=5.0):
        """
        Effectue une requête HTTP avec mécanisme de retry.
        
        Args:
            url: URL de la requête
            method: Méthode HTTP (GET, POST, DELETE)
            json_data: Données JSON pour POST
            params: Paramètres de requête
            max_retries: Nombre maximum de tentatives
            timeout: Timeout en secondes
            
        Returns:
            Réponse JSON ou None en cas d'échec
        """
        retry_count = 0
        last_exception = None
        
        while retry_count < max_retries:
            try:
                if method == "GET":
                    response = requests.get(url, params=params, timeout=timeout)
                elif method == "POST":
                    response = requests.post(url, json=json_data, params=params, timeout=timeout)
                elif method == "DELETE":
                    response = requests.delete(url, params=params, timeout=timeout)
                else:
                    raise ValueError(f"Méthode non supportée: {method}")
                    
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                last_exception = e
                retry_count += 1
                wait_time = 0.5 * (2 ** retry_count)  # Backoff exponentiel
                logger.warning(f"Tentative {retry_count}/{max_retries} échouée: {str(e)}. Nouvelle tentative dans {wait_time}s")
                time.sleep(wait_time)
        
        logger.error(f"Échec après {max_retries} tentatives: {str(last_exception)}")
        return None
    
    def _create_trade_cycle(self, signal: StrategySignal) -> Optional[str]:
        """
        Crée un cycle de trading à partir d'un signal.

        Args:
            signal: Signal de trading validé

        Returns:
            ID du cycle créé ou None en cas d'échec
        """
        # Vérifier le circuit breaker pour le portfolio
        if not self.portfolio_circuit.can_execute():
            logger.warning(f"Circuit ouvert pour le service Portfolio, signal ignoré")
            return None

        try:
            # Calculer le montant à trader
            trade_amount, quote_asset = self._calculate_trade_amount(signal)
    
            # Déterminer la poche à utiliser en fonction de l'actif
            try:
                pocket_type = self.pocket_checker.determine_best_pocket(trade_amount, asset=quote_asset)
                # Appel au portfolio réussi
                self.portfolio_circuit.record_success()
            except Exception as e:
                self.portfolio_circuit.record_failure()
                logger.error(f"❌ Erreur lors de l'interaction avec Portfolio: {str(e)}")
                return None
    
            if not pocket_type:
                logger.warning(f"❌ Aucune poche disponible pour un trade de {trade_amount:.2f} USDC")
                return None
    
            # Convertir le montant en quantité
            # Pour les paires non-USDC, il faut d'abord convertir le montant USDC
            if signal.symbol.endswith("BTC"):
                # Pour ETHBTC, il faut convertir le montant USDC en BTC d'abord
                # Récupérer le prix de BTC/USDC pour la conversion
                btc_price = self._get_btc_price()
                if btc_price:
                    # Convertir le montant USDC en BTC
                    btc_amount = trade_amount / btc_price
                    # Ensuite calculer la quantité d'ETH
                    quantity = btc_amount / signal.price
                else:
                    logger.error("❌ Impossible de récupérer le prix BTC/USDC pour la conversion")
                    return None
            else:
                # Pour les paires USDC, calcul direct
                quantity = trade_amount / signal.price
    
            # Calculer le stop-loss et take-profit
            stop_price = signal.metadata.get('stop_price')
            target_price = signal.metadata.get('target_price')
    
            # Préparer la requête pour le Trader
            # Important: Convertir les enums en chaînes explicitement
            order_data = {
                "symbol": signal.symbol,
                "side": signal.side.value if hasattr(signal.side, 'value') else str(signal.side),
                "quantity": quantity,
                "price": signal.price,
                "strategy": signal.strategy,
                "timestamp": int(time.time() * 1000)  # Un timestamp actuel en millisecondes
            }
            
            # Ajouter les prix cibles si disponibles
            if target_price:
                order_data["target_price"] = target_price
            if stop_price:
                order_data["stop_price"] = stop_price
    
            # Réserver les fonds dans la poche
            temp_cycle_id = f"temp_{int(time.time())}"
            
            # Calculer le montant en USDC à réserver
            if quote_asset == 'BTC':
                # Pour les paires BTC, convertir le montant BTC en USDC pour la réservation
                btc_price = self._get_btc_price()
                if btc_price:
                    reserve_amount_usdc = trade_amount * btc_price
                    logger.info(f"Conversion pour réservation: {trade_amount:.6f} BTC = {reserve_amount_usdc:.2f} USDC (prix BTC: {btc_price:.2f})")
                else:
                    logger.error("❌ Impossible de récupérer le prix BTC/USDC pour la réservation")
                    return None
            elif quote_asset == 'ETH':
                # Pour les paires ETH, convertir le montant ETH en USDC
                eth_price = self._get_eth_price()  # À implémenter si nécessaire
                if eth_price:
                    reserve_amount_usdc = trade_amount * eth_price
                else:
                    logger.error("❌ Impossible de récupérer le prix ETH/USDC pour la réservation")
                    return None
            else:
                # Pour les paires USDC, pas de conversion nécessaire
                reserve_amount_usdc = trade_amount
            
            try:
                reserved = self.pocket_checker.reserve_funds(reserve_amount_usdc, temp_cycle_id, pocket_type)
                # Autre appel au portfolio réussi
                self.portfolio_circuit.record_success()
            except Exception as e:
                self.portfolio_circuit.record_failure()
                logger.error(f"❌ Erreur lors de la réservation des fonds: {str(e)}")
                return None
    
            if not reserved:
                logger.error(f"❌ Échec de réservation des fonds pour le trade")
                return None
    
            # Vérifier le circuit breaker pour le Trader
            if not self.trader_circuit.can_execute():
                logger.warning(f"Circuit ouvert pour le service Trader, libération des fonds réservés")
                self.pocket_checker.release_funds(reserve_amount_usdc, temp_cycle_id, pocket_type)
                return None
    
            # Créer le cycle via l'API du Trader avec retry
            try:
                logger.info(f"Envoi de la requête au Trader: {order_data}")
                result = self._make_request_with_retry(
                    f"{self.trader_api_url}/order",
                    method="POST",
                    json_data=order_data,
                    timeout=10.0  # Timeout plus long pour la création de l'ordre
                )
                
                if not result:
                    logger.error("❌ Échec de la création du cycle: aucune réponse du Trader")
                    # Libérer les fonds réservés
                    self.pocket_checker.release_funds(reserve_amount_usdc, temp_cycle_id, pocket_type)
                    return None
                
                cycle_id = result.get('order_id')
        
                # Appel au trader réussi
                self.trader_circuit.record_success()
        
                if not cycle_id:
                    logger.error("❌ Réponse invalide du Trader: pas d'ID de cycle")
                    # Libérer les fonds réservés
                    self.pocket_checker.release_funds(trade_amount, temp_cycle_id, pocket_type)
                    return None
        
                # Mettre à jour la réservation avec l'ID réel du cycle
                try:
                    # Stocker le mapping dans Redis pour la réconciliation future
                    try:
                        # Utiliser la méthode set avec expiration du RedisClientPool
                        self.redis_client.set(f"cycle_mapper:{temp_cycle_id}", cycle_id, expiration=3600)  # 1h TTL
                        logger.debug(f"🔗 Mapping sauvé: {temp_cycle_id} → {cycle_id}")
                    except Exception as redis_e:
                        logger.warning(f"⚠️ Impossible de sauver le mapping Redis: {redis_e}")
                    
                    # Libérer l'ID temporaire et réserver avec l'ID final
                    self.pocket_checker.release_funds(reserve_amount_usdc, temp_cycle_id, pocket_type)
                    self.pocket_checker.reserve_funds(reserve_amount_usdc, cycle_id, pocket_type)
                    self.portfolio_circuit.record_success()
                except Exception as e:
                    self.portfolio_circuit.record_failure()
                    logger.error(f"❌ Erreur lors de la mise à jour de la réservation: {str(e)}")
                    # Tenter d'annuler le cycle créé
                    try:
                        self._make_request_with_retry(
                            f"{self.trader_api_url}/order/{cycle_id}",
                            method="DELETE"
                        )
                    except:
                        pass
                    return None
        
                logger.info(f"✅ Cycle de trading créé: {cycle_id} ({signal.side} {signal.symbol})")
                return cycle_id
            
            except requests.RequestException as e:
                self.trader_circuit.record_failure()
                logger.error(f"❌ Erreur lors de la création du cycle: {str(e)}")
                # Libérer les fonds réservés
                self.pocket_checker.release_funds(reserve_amount_usdc, temp_cycle_id, pocket_type)
                return None
    
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création du cycle de trading: {str(e)}")
            return None
    
    def _signal_processing_loop(self) -> None:
        """
        Boucle de traitement des signaux de trading.
        Cette méthode s'exécute dans un thread séparé.
        """
        logger.info("Démarrage de la boucle de traitement des signaux")
        
        while not self.stop_event.is_set():
            try:
                # Récupérer un signal de la file d'attente avec timeout
                try:
                    signal = self.signal_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Vérifier si le signal doit être filtré
                if self._should_filter_signal(signal):
                    self.signal_queue.task_done()
                    continue
                
                # Vérifier la force du signal
                if signal.strength in [SignalStrength.WEAK]:
                    logger.info(f"⚠️ Signal ignoré: trop faible ({signal.strength})")
                    self.signal_queue.task_done()
                    continue
                
                # Créer un cycle de trading
                cycle_id = self._create_trade_cycle(signal)
                
                if cycle_id:
                    logger.info(f"✅ Trade exécuté pour le signal {signal.strategy} sur {signal.symbol}")
                else:
                    logger.warning(f"⚠️ Échec d'exécution du trade pour le signal {signal.strategy}")
                
                # Marquer la tâche comme terminée
                self.signal_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle de traitement des signaux: {str(e)}")
                time.sleep(1)  # Pause pour éviter une boucle d'erreur infinie
        
        logger.info("Boucle de traitement des signaux arrêtée")
    
    def start(self) -> None:
        """
        Démarre le gestionnaire de signaux.
        """
        logger.info("🚀 Démarrage du gestionnaire de signaux...")
        
        # S'abonner au canal des signaux
        self.redis_client.subscribe(self.signal_channel, self._process_signal)
        
        # Démarrer le thread de traitement des signaux
        self.stop_event.clear()
        self.processing_thread = threading.Thread(
            target=self._signal_processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        # NOUVEAU: Démarrer le thread d'agrégation des signaux
        self.aggregation_thread = threading.Thread(
            target=self._analyze_aggregated_signals,
            daemon=True,
            name="SignalAggregator"
        )
        self.aggregation_thread.start()
        
        # Démarrer le moniteur de synchronisation
        self.sync_monitor.start()
        
        logger.info("✅ Gestionnaire de signaux démarré")
    
    def stop(self) -> None:
        """
        Arrête le gestionnaire de signaux.
        """
        logger.info("Arrêt du gestionnaire de signaux...")
        
        # Signaler l'arrêt aux threads
        self.stop_event.set()
        
        # Attendre que le thread de traitement se termine
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # NOUVEAU: Attendre que le thread d'agrégation se termine
        if self.aggregation_thread and self.aggregation_thread.is_alive():
            self.aggregation_thread.join(timeout=5.0)
        
        # Arrêter le moniteur de synchronisation
        self.sync_monitor.stop()
        
        # Se désabonner du canal Redis
        self.redis_client.unsubscribe()
        
        logger.info("✅ Gestionnaire de signaux arrêté")

    def handle_order_failed(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite les notifications d'échec d'ordre.
    
        Args:
            channel: Canal Redis d'où provient la notification
            data: Données de la notification
        """
        try:
            cycle_id = data.get("cycle_id")
            symbol = data.get("symbol")
            reason = data.get("reason", "Raison inconnue")
        
            if not cycle_id:
                logger.warning("❌ Message d'échec d'ordre reçu sans cycle_id")
                return
            
            logger.info(f"⚠️ Ordre échoué pour le cycle {cycle_id}: {reason}")
        
            # Déterminer si c'est un cycle temporaire ou confirmé
            if cycle_id.startswith("temp_"):
                # Cycle temporaire, libérer les fonds
                amount = data.get("amount", 0)
                if amount > 0:
                    self.pocket_checker.release_funds(amount, cycle_id, "active")
                    logger.info(f"✅ {amount} USDC libérés pour le cycle temporaire {cycle_id} après échec")
            else:
                # Cycle confirmé, annuler le cycle via l'API Trader
                try:
                    self._make_request_with_retry(
                        f"{self.trader_api_url}/order/{cycle_id}",
                        method="DELETE"
                    )
                    logger.info(f"✅ Cycle {cycle_id} annulé après échec d'ordre")
                except Exception as e:
                    logger.error(f"❌ Erreur lors de l'annulation du cycle {cycle_id}: {str(e)}")
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de l'échec d'ordre: {str(e)}")
    
    def handle_cycle_created(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite la création d'un cycle pour maintenir la synchronisation.
        """
        cycle_id = data.get('cycle_id')
        logger.debug(f"📌 Cycle créé: {cycle_id}")
        # La réservation est déjà faite, on note juste l'événement
        
    def handle_cycle_closed(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite la fermeture d'un cycle et force une réconciliation des poches.
        """
        cycle_id = data.get('cycle_id')
        symbol = data.get('symbol')
        profit_loss = data.get('profit_loss', 0)
        
        logger.info(f"💰 Cycle fermé: {cycle_id} ({symbol}) - P&L: {profit_loss:.2f}")
        
        # Forcer une réconciliation pour mettre à jour les poches
        self.pocket_checker.force_refresh()
    
    def _get_btc_price(self) -> Optional[float]:
        """
        Récupère le prix actuel de BTC/USDC.
        
        Returns:
            Prix de BTC en USDC ou None en cas d'échec
        """
        try:
            # Récupérer le prix depuis le service trader via son API
            url = f"{self.trader_api_url}/price/BTCUSDC"
            response = self._make_request_with_retry(url, timeout=2.0)
            
            if response and 'price' in response:
                btc_price = float(response['price'])
                logger.debug(f"Prix BTC/USDC récupéré: {btc_price}")
                return btc_price
            
            logger.warning("Impossible de récupérer le prix BTC/USDC")
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du prix BTC: {str(e)}")
            return None
    
    def _get_eth_price(self) -> Optional[float]:
        """
        Récupère le prix actuel de ETH/USDC.
        
        Returns:
            Prix de ETH en USDC ou None en cas d'échec
        """
        try:
            # Récupérer le prix depuis le service trader via son API
            url = f"{self.trader_api_url}/price/ETHUSDC"
            response = self._make_request_with_retry(url, timeout=2.0)
            
            if response and 'price' in response:
                eth_price = float(response['price'])
                logger.debug(f"Prix ETH/USDC récupéré: {eth_price}")
                return eth_price
            
            logger.warning("Impossible de récupérer le prix ETH/USDC")
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du prix ETH: {str(e)}")
            return None
        
    def handle_cycle_canceled(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite l'annulation d'un cycle et libère les fonds.
        """
        cycle_id = data.get('cycle_id')
        logger.info(f"🚫 Cycle annulé: {cycle_id}")
        
        # Forcer une réconciliation pour libérer les fonds
        self.pocket_checker.force_refresh()
        
    def handle_cycle_failed(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite l'échec d'un cycle.
        """
        cycle_id = data.get('cycle_id')
        logger.info(f"❌ Cycle échoué: {cycle_id}")
        
        # Forcer une réconciliation
        self.pocket_checker.force_refresh()

class CircuitBreaker:
    """Circuit breaker pour éviter les appels répétés à des services en échec."""
    
    def __init__(self, max_failures=3, reset_timeout=60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.open_since = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def record_success(self):
        """Enregistre un succès et réinitialise le circuit."""
        self.failure_count = 0
        self.state = "CLOSED"
        self.open_since = None
    
    def record_failure(self):
        """Enregistre un échec et ouvre le circuit si nécessaire."""
        self.failure_count += 1
        if self.failure_count >= self.max_failures:
            self.state = "OPEN"
            self.open_since = time.time()
    
    def can_execute(self):
        """Vérifie si une opération peut être exécutée."""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            # Vérifier si le temps de reset est écoulé
            if time.time() - self.open_since > self.reset_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        
        # HALF_OPEN: permettre un essai
        return True