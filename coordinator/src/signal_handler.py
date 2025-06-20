"""
Module de gestion des signaux de trading.
Reçoit les signaux, les valide, et coordonne la création des cycles de trading.
"""
import logging
import json
import requests
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
from coordinator.src.cycle_sync_monitor import CycleSyncMonitor
from coordinator.src.smart_cycle_manager import SmartCycleManager, CycleAction, SmartCycleDecision
from shared.src.schemas import StrategySignal, TradeOrder

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
        self.logger = logging.getLogger(__name__)
        
        # Client Redis pour les communications
        self.redis_client = RedisClient()
        self.redis_client.subscribe("roottrading:order:failed", self.handle_order_failed)
        
        # S'abonner aux événements de cycles pour rester synchronisé
        self.redis_client.subscribe("roottrading:cycle:created", self.handle_cycle_created)
        self.redis_client.subscribe("roottrading:cycle:completed", self.handle_cycle_completed)
        self.redis_client.subscribe("roottrading:cycle:canceled", self.handle_cycle_canceled)
        self.redis_client.subscribe("roottrading:cycle:failed", self.handle_cycle_failed)
        
        # Canal Redis pour les signaux
        # CHANGEMENT: Écouter les signaux filtrés au lieu des signaux bruts
        self.signal_channel = "roottrading:signals:filtered"
        
        # File d'attente thread-safe pour les signaux
        self.signal_queue = queue.Queue()
        
        # Thread pour le traitement des signaux
        self.processing_thread = None
        self.stop_event = threading.Event()

        # Moniteur de synchronisation des cycles (solution définitive)
        self.sync_monitor = CycleSyncMonitor(
            trader_api_url=trader_api_url,
            check_interval=10  # Vérification toutes les 10 secondes pour plus de réactivité
        )
        
        # SmartCycleManager pour la gestion intelligente des cycles
        self.smart_cycle_manager = SmartCycleManager()
        
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
        self.max_cycles_per_symbol_side = 10  # AUGMENTÉ: Max 10 positions par side pour plus de granularité
        self.contradiction_threshold = 0.7  # Seuil pour décider en cas de contradiction
        
        # NOUVEAU: Historique récent des signaux pour détecter les patterns
        self.recent_signals_history = {}  # {symbol: [(signal, timestamp)]}
        self.history_window = 30.0  # Garder 30 secondes d'historique
        
        # Thread pour l'agrégation périodique
        self.aggregation_thread = None
        
        logger.info(f"✅ SignalHandler initialisé en mode {'DÉMO' if self.demo_mode else 'RÉEL'}")
    
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
            if signal.metadata and 'stop_price' in signal.metadata:
                logger.info(f"📨 Signal reçu: {signal.side} {signal.symbol} @ {signal.price} ({signal.strategy}) - Stop: {signal.metadata.get('stop_price', 'N/A')}")
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
        
        # Mapper signal vers position réelle
        signal_position = "LONG" if signal_side_str == "LONG" else "SHORT"
        opposite_position = "SHORT" if signal_position == "LONG" else "LONG"
        
        # NOUVEAU: Filtre stratégique - Évaluer si le signal mérite d'être traité
        if not self._should_process_signal_strategically(signal, signal_position, opposite_position):
            return
        
        # Vérifier les positions existantes pour ce symbole
        existing_opposite = self.active_cycles_cache.get(signal.symbol, {}).get(opposite_position, 0)
        existing_same = self.active_cycles_cache.get(signal.symbol, {}).get(signal_position, 0)
        
        # Cas 1: Signal opposé à des positions existantes
        # NOTE: Si _should_process_signal_strategically a retourné True,
        # c'est qu'on a décidé d'accepter le retournement (signal très fort)
        # Dans ce cas, on devrait fermer les positions opposées, pas bloquer le signal
        if existing_opposite > 0:
            # Calculer la force du signal pour décider
            signal_strength_score = self._get_signal_strength_score(signal)
            
            if signal_strength_score >= 0.85:  # Signal très fort accepté par le filtre stratégique
                logger.info(f"🔄 RETOURNEMENT ACCEPTÉ: Signal {signal_position} très fort ({signal_strength_score:.2f}) "
                           f"va fermer {existing_opposite} positions {opposite_position} sur {signal.symbol}")
                # Fermer les positions opposées avant de créer la nouvelle
                self._close_opposite_positions(signal.symbol, opposite_position)
                # Le signal continuera pour créer la nouvelle position
            else:
                # Signal pas assez fort, on bloque
                logger.warning(f"🚫 BLOQUÉ: Signal {signal_position} contradictoire avec {existing_opposite} "
                             f"positions {opposite_position} sur {signal.symbol} - Force insuffisante: {signal_strength_score:.2f}")
                return
        
        # Cas 2: Vérifier si on peut ajouter à la position existante
        if existing_same >= self.max_cycles_per_symbol_side:
            logger.warning(f"❌ Limite de {self.max_cycles_per_symbol_side} cycles {signal.side} "
                         f"atteinte pour {signal.symbol}")
            return
        
        # Cas 3: Vérifier l'anti-spam (exemption pour les signaux agrégés)
        if signal.symbol in self.recent_signals_history and not signal.strategy.startswith("Aggregated_"):
            same_side_recent = [
                s for s, t in self.recent_signals_history[signal.symbol]
                if s.side == signal.side and time.time() - t < 10.0
            ]
            
            if len(same_side_recent) > 2:
                logger.warning(f"⚠️ Trop de signaux {signal.side} récents pour {signal.symbol}")
                return
        
        # Cas 4: Signal renforcant une position existante
        if existing_same > 0 and signal.strength and signal.strength >= SignalStrength.STRONG:
            logger.info(f"💪 Signal renforçant {existing_same} position(s) {signal_position} existante(s)")
            # On laisse passer pour pyramider la position
        
        # Ajouter à la file de traitement
        self.signal_queue.put(signal)
        logger.info(f"✅ Signal ajouté à la file: {signal_position} {signal.symbol} "
                   f"(Positions: {existing_same} same, {existing_opposite} opposite)")
    
    def _should_process_signal_strategically(self, signal: StrategySignal, signal_position: str, opposite_position: str) -> bool:
        """
        Évalue si un signal mérite d'être traité selon le contexte stratégique du portfolio.
        
        Cette méthode implémente la philosophie "Conviction & Cohérence":
        - Ignorer les signaux contradictoires faibles
        - Accepter les signaux qui confirment la tendance
        - N'accepter les retournements que s'ils sont très forts
        
        Args:
            signal: Signal à évaluer
            signal_position: Position que le signal veut prendre (LONG/SHORT)
            opposite_position: Position opposée (SHORT/LONG)
            
        Returns:
            True si le signal doit être traité, False s'il doit être ignoré
        """
        # Récupérer les positions existantes
        existing_opposite = self.active_cycles_cache.get(signal.symbol, {}).get(opposite_position, 0)
        existing_same = self.active_cycles_cache.get(signal.symbol, {}).get(signal_position, 0)
        
        # Cas 1: Signal confirmant - toujours accepter (sauf si limite atteinte)
        if existing_same > 0 and existing_opposite == 0:
            logger.info(f"✅ Signal confirmant la tendance {signal_position} sur {signal.symbol}")
            return True
        
        # Cas 2: Pas de position - accepter tous les signaux
        if existing_same == 0 and existing_opposite == 0:
            logger.info(f"✅ Nouvelle position {signal_position} sur {signal.symbol}")
            return True
        
        # Cas 3: Signal contradictoire - appliquer filtre stratégique
        if existing_opposite > 0:
            # Calculer la "conviction" du portefeuille sur la position opposée
            portfolio_conviction = self._calculate_portfolio_conviction(signal.symbol, opposite_position)
            
            # Évaluer la force du signal contradictoire
            signal_strength_score = self._get_signal_strength_score(signal)
            
            # Décision stratégique
            if signal_strength_score < 0.7:  # Signal faible à modéré
                logger.info(f"🚫 Signal {signal_position} {signal.symbol} IGNORÉ - "
                           f"Trop faible (score: {signal_strength_score:.2f}) pour contredire "
                           f"{existing_opposite} positions {opposite_position}")
                return False
                
            elif signal_strength_score < 0.85:  # Signal fort
                if portfolio_conviction > 0.6:
                    logger.info(f"🚫 Signal {signal_position} {signal.symbol} IGNORÉ - "
                               f"Conviction portfolio trop forte ({portfolio_conviction:.2f}) "
                               f"sur positions {opposite_position}")
                    return False
                else:
                    logger.info(f"⚠️ Signal {signal_position} fort accepté - "
                               f"Conviction portfolio faible sur {opposite_position}")
                    return True
                    
            else:  # Signal très fort (>= 0.85)
                logger.info(f"🔄 Signal {signal_position} TRÈS FORT ({signal_strength_score:.2f}) - "
                           f"Retournement stratégique accepté malgré {existing_opposite} positions {opposite_position}")
                return True
        
        # Cas 4: Autres cas (ne devrait pas arriver)
        return True
    
    def _calculate_portfolio_conviction(self, symbol: str, position: str) -> float:
        """
        Calcule la conviction du portfolio sur une position.
        Plus la valeur est élevée, plus le portfolio est "convaincu" de sa position.
        
        Args:
            symbol: Symbole concerné
            position: Position à évaluer (LONG/SHORT)
            
        Returns:
            Score de conviction entre 0 et 1
        """
        # Facteurs de conviction:
        # 1. Nombre de cycles actifs
        active_cycles = self.active_cycles_cache.get(symbol, {}).get(position, 0)
        cycles_factor = min(active_cycles / 3.0, 1.0)  # Normalisé sur 3 max
        
        # 2. Performance récente (à implémenter avec les données de PnL)
        # Pour l'instant on utilise une valeur fixe
        performance_factor = 0.5
        
        # 3. Durée des positions (plus c'est long, plus on est convaincu)
        # Pour l'instant on utilise une valeur fixe
        duration_factor = 0.6
        
        # 4. Cohérence des signaux récents
        recent_coherence = self._calculate_recent_signal_coherence(symbol, position)
        
        # Moyenne pondérée
        conviction = (
            cycles_factor * 0.4 +
            performance_factor * 0.2 +
            duration_factor * 0.2 +
            recent_coherence * 0.2
        )
        
        return conviction
    
    def _calculate_recent_signal_coherence(self, symbol: str, position: str) -> float:
        """
        Calcule la cohérence des signaux récents pour une position.
        
        Args:
            symbol: Symbole concerné
            position: Position à évaluer (LONG/SHORT)
            
        Returns:
            Score de cohérence entre 0 et 1
        """
        if symbol not in self.recent_signals_history:
            return 0.5  # Neutre si pas d'historique
        
        current_time = time.time()
        recent_signals = [
            s for s, t in self.recent_signals_history[symbol]
            if current_time - t < 60.0  # Dernière minute
        ]
        
        if not recent_signals:
            return 0.5
        
        # Compter les signaux cohérents avec la position
        expected_side = OrderSide.LONG if position == "LONG" else OrderSide.SHORT
        coherent_signals = sum(1 for s in recent_signals if s.side == expected_side)
        
        coherence = coherent_signals / len(recent_signals)
        return coherence
    
    def _get_signal_strength_score(self, signal: StrategySignal) -> float:
        """
        Convertit la force du signal en score numérique.
        
        Args:
            signal: Signal à évaluer
            
        Returns:
            Score entre 0 et 1
        """
        # Mapping des forces de signal
        strength_scores = {
            SignalStrength.VERY_WEAK: 0.2,
            SignalStrength.WEAK: 0.4,
            SignalStrength.MODERATE: 0.6,
            SignalStrength.STRONG: 0.8,
            SignalStrength.VERY_STRONG: 1.0
        }
        
        base_score = strength_scores.get(signal.strength, 0.5)
        
        # Bonus progressif pour les signaux agrégés selon le nombre de stratégies
        if signal.strategy.startswith("Aggregated_"):
            try:
                num_strategies = int(signal.strategy.split("_")[1])
                if num_strategies == 1:
                    # Petit bonus pour une seule stratégie
                    base_score = min(base_score + 0.05, 1.0)
                elif num_strategies == 2:
                    # Bonus moyen pour 2 stratégies
                    base_score = min(base_score + 0.10, 1.0)
                elif num_strategies >= 3:
                    # Bonus fort pour 3+ stratégies (consensus fort)
                    base_score = min(base_score + 0.15, 1.0)
            except (IndexError, ValueError):
                # Si on ne peut pas parser le nombre, bonus par défaut
                base_score = min(base_score + 0.10, 1.0)
        
        # Bonus/malus selon la confidence
        if hasattr(signal, 'confidence') and signal.confidence is not None:
            confidence_factor = signal.confidence
            base_score = base_score * 0.7 + confidence_factor * 0.3
        
        return base_score
    
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
                
                # Déterminer le côté de la POSITION RÉELLE (pas du prochain ordre)
                status = cycle.get('status', '')
                if status in ['waiting_buy', 'active_buy']:
                    # waiting_buy = position SHORT (va racheter pour fermer)
                    side = 'SHORT'
                elif status in ['waiting_sell', 'active_sell']:
                    # waiting_sell = position LONG (va vendre pour fermer)
                    side = 'LONG'
                else:
                    continue  # Ignorer les autres statuts
                
                if symbol:
                    if symbol not in self.active_cycles_cache:
                        self.active_cycles_cache[symbol] = {'LONG': 0, 'SHORT': 0}
                    
                    self.active_cycles_cache[symbol][side] = self.active_cycles_cache[symbol].get(side, 0) + 1
            
            self.cache_update_time = time.time()
            logger.debug(f"Cache des cycles actifs mis à jour depuis l'API centralisée: {self.active_cycles_cache}")
            
        except Exception as e:
            logger.error(f"Erreur lors du rafraîchissement du cache: {str(e)}")
    
    def _close_opposite_positions(self, symbol: str, side: str) -> bool:
        """
        Ferme toutes les positions opposées pour permettre un retournement de marché.
        
        Args:
            symbol: Symbole concerné (ex: BTCUSDC)
            side: Position à fermer (LONG ou SHORT)
            
        Returns:
            True si toutes les positions ont été fermées avec succès
        """
        try:
            logger.info(f"🔄 Fermeture des positions {side} sur {symbol} pour retournement")
            
            # Récupérer les cycles actifs pour ce symbole
            response = self._make_request_with_retry(
                f"{self.trader_api_url}/cycles",
                method="GET",
                params={
                    "symbol": symbol,
                    "confirmed": "true",
                    "include_completed": "false"
                },
                timeout=5.0
            )
            
            if not response or not response.get('success'):
                logger.error(f"Impossible de récupérer les cycles {side} pour {symbol}")
                return False
            
            # Extraire les cycles
            cycles = response.get('cycles', [])
            cycles_to_close = []
            
            # Filtrer les cycles de la position à fermer
            for cycle in cycles:
                status = cycle.get('status', '')
                cycle_position = None
                
                if status in ['waiting_sell', 'active_sell']:
                    cycle_position = 'LONG'  # Position longue en attente/en cours de vente
                elif status in ['waiting_buy', 'active_buy']:
                    cycle_position = 'SHORT'  # Position courte en attente/en cours de rachat
                
                if cycle_position == side:
                    cycles_to_close.append(cycle)
            
            if not cycles_to_close:
                logger.info(f"Aucune position {side} à fermer pour {symbol}")
                return True
            
            # Fermer chaque cycle
            success_count = 0
            for cycle in cycles_to_close:
                cycle_id = cycle.get('id')
                if not cycle_id:
                    continue
                
                logger.info(f"📤 Fermeture du cycle {cycle_id} ({side})")
                
                # Appeler l'API pour fermer le cycle au marché
                close_response = self._make_request_with_retry(
                    f"{self.trader_api_url}/close/{cycle_id}",
                    method="POST",
                    json_data={"reason": "market_reversal"},
                    timeout=10.0
                )
                
                if close_response and (close_response.get('success') or close_response.get('status') in ['closed', 'completed']):
                    success_count += 1
                    logger.info(f"✅ Cycle {cycle_id} fermé avec succès")
                else:
                    logger.error(f"❌ Échec de fermeture du cycle {cycle_id}")
            
            # Rafraîchir le cache après fermeture
            self._refresh_active_cycles_cache()
            
            success = success_count == len(cycles_to_close)
            if success:
                logger.info(f"✅ Toutes les positions {side} fermées ({success_count}/{len(cycles_to_close)})")
            else:
                logger.warning(f"⚠️ Fermeture partielle: {success_count}/{len(cycles_to_close)} positions fermées")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la fermeture des positions {side}: {str(e)}")
            return False
    
    def _can_create_new_cycle(self, symbol: str, side: str) -> bool:
        """
        Vérifie s'il est possible de créer un nouveau cycle.
        
        Args:
            symbol: Symbole du trade
            side: Côté du trade (LONG/SHORT)
            
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
        LONG_signals = [s for s in signals if s.side == OrderSide.LONG]
        SHORT_signals = [s for s in signals if s.side == OrderSide.SHORT]

        # Cas 1: Signaux contradictoires
        if LONG_signals and SHORT_signals:
            logger.warning(f"⚠️ Signaux contradictoires détectés pour {symbol}: "
                         f"{len(LONG_signals)} LONG vs {len(SHORT_signals)} SHORT")
            
            # Calculer les scores moyens
            long_score = self._calculate_signal_score(LONG_signals)
            short_score = self._calculate_signal_score(SHORT_signals)
            
            # Si la différence est significative, suivre le plus fort
            if abs(long_score - short_score) > self.contradiction_threshold:
                if long_score > short_score:
                    self._process_long_signals(symbol, LONG_signals)
                else:
                    self._process_short_signals(symbol, SHORT_signals)
            else:
                logger.info(f"🤷 Signaux trop contradictoires, aucune action pour {symbol}")
        
        # Cas 2: Signaux unanimes LONG
        elif LONG_signals:
            self._process_long_signals(symbol, LONG_signals)

        # Cas 3: Signaux unanimes SHORT
        elif SHORT_signals:
            self._process_short_signals(symbol, SHORT_signals)

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
    
    def _process_long_signals(self, symbol: str, signals: List[StrategySignal]) -> None:
        """
        Traite un groupe de signaux LONG.
        
        Args:
            symbol: Symbole
            signals: Signaux LONG
        """
        # Vérifier si on peut créer un nouveau cycle
        if not self._can_create_new_cycle(symbol, "LONG"):
            logger.warning(f"❌ Impossible de créer plus de cycles LONG pour {symbol}")
            return
        
        # Choisir le meilleur signal ou créer un signal composite
        best_signal = self._select_best_signal(signals)
        
        # Si plus de 3 stratégies sont d'accord et signal très fort, possibilité de double position
        if len(signals) >= 3 and best_signal.strength == SignalStrength.VERY_STRONG:
            logger.info(f"🚀 Signal de consensus fort détecté pour {symbol} ({len(signals)} stratégies)")
            # On pourrait créer 2 positions ici si vraiment fort
        
        # Ajouter à la file de traitement normale
        self.signal_queue.put(best_signal)

    def _process_short_signals(self, symbol: str, signals: List[StrategySignal]) -> None:
        """
        Traite un groupe de signaux SHORT.
        
        Args:
            symbol: Symbole
            signals: Signaux SHORT
        """
        # Vérifier si on peut créer un nouveau cycle
        if not self._can_create_new_cycle(symbol, "SHORT"):
            logger.warning(f"❌ Impossible de créer plus de cycles SHORT pour {symbol}")
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
        # NOUVEAU: Les signaux agrégés sont exemptés du filtrage (conflits déjà résolus)
        if signal.strategy.startswith("Aggregated_"):
            logger.info(f"✅ Signal agrégé exempté du filtrage: {signal.strategy}")
            return False
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
            # Si dans une tendance haussière forte, filtrer les signaux SHORT (sauf très forts)
            if signal.side == OrderSide.SHORT and signal.strength != SignalStrength.VERY_STRONG:
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
            elif action == 'buy_only' and signal.side == OrderSide.SHORT:
                logger.info(f"🔍 Signal {signal.side} filtré: seuls les achats sont autorisés pour {signal.symbol}")
                return True
            
            # Si l'action est "sell_only", filtrer les signaux d'achat
            elif action == 'sell_only' and signal.side == OrderSide.LONG:
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
    
    def _get_base_asset(self, symbol: str) -> str:
        """
        Détermine l'actif de base (base asset) pour un symbole.
        
        Args:
            symbol: Symbole de trading (ex: BTCUSDC, ETHBTC)
            
        Returns:
            L'actif de base (BTC, ETH, etc.)
        """
        # Pour les paires communes
        if symbol.endswith('USDC'):
            return symbol[:-4]  # BTCUSDC -> BTC, ETHUSDC -> ETH
        elif symbol.endswith('BTC'):
            return symbol[:-3]  # ETHBTC -> ETH
        elif symbol.endswith('ETH'):
            return symbol[:-3]  # XRPETH -> XRP
        elif symbol.endswith('BNB'):
            return symbol[:-3]  # ADABNB -> ADA
        else:
            # Par défaut, supposer que les 3 premiers caractères sont l'actif de base
            return symbol[:3]
    
    def _calculate_trade_amount(self, signal: StrategySignal, available_balance: Optional[float] = None) -> tuple[float, str]:
        """
        Calcule le montant à trader basé sur le signal, la balance disponible et la performance.
        
        Args:
            signal: Signal de trading
            available_balance: Balance disponible (optionnel, sinon récupérée du portfolio)
            
        Returns:
            Tuple (montant, actif) - ex: (100.0, 'USDC') ou (0.001, 'BTC')
        """
        quote_asset = self._get_quote_asset(signal.symbol)
        
        # Si la balance n'est pas fournie, la récupérer depuis le portfolio
        if available_balance is None:
            try:
                portfolio_url = f"http://portfolio:8000/balance/{quote_asset}"
                response = self._make_request_with_retry(portfolio_url)
                available_balance = response.get('available', 0) if response else 0
            except Exception as e:
                self.logger.warning(f"Impossible de récupérer la balance {quote_asset}: {e}")
                available_balance = 0
        
        # Pourcentages d'allocation par force de signal
        allocation_percentages = {
            SignalStrength.WEAK: float(os.getenv('ALLOCATION_WEAK_PCT', 2.0)),      # 2% du capital
            SignalStrength.MODERATE: float(os.getenv('ALLOCATION_MODERATE_PCT', 5.0)), # 5% du capital  
            SignalStrength.STRONG: float(os.getenv('ALLOCATION_STRONG_PCT', 8.0)),     # 8% du capital
            SignalStrength.VERY_STRONG: float(os.getenv('ALLOCATION_VERY_STRONG_PCT', 12.0)) # 12% du capital
        }
        
        # Calculer le montant basé sur le pourcentage de la balance
        base_percentage = allocation_percentages.get(signal.strength, 5.0)
        calculated_amount = available_balance * (base_percentage / 100.0)
        
        # Montants minimums et maximums par devise
        min_amounts = {
            'USDC': float(os.getenv('MIN_TRADE_USDC', 10.0)),
            'BTC': float(os.getenv('MIN_TRADE_BTC', 0.0001)),
            'ETH': float(os.getenv('MIN_TRADE_ETH', 0.003)),
            'BNB': float(os.getenv('MIN_TRADE_BNB', 0.02))
        }
        
        max_amounts = {
            'USDC': float(os.getenv('MAX_TRADE_USDC', 250.0)),
            'BTC': float(os.getenv('MAX_TRADE_BTC', 0.005)),
            'ETH': float(os.getenv('MAX_TRADE_ETH', 0.13)),
            'BNB': float(os.getenv('MAX_TRADE_BNB', 2.0))
        }
        
        min_amount = min_amounts.get(quote_asset, 10.0)
        max_amount = max_amounts.get(quote_asset, 100.0)
        
        # Appliquer les limites
        final_amount = max(min_amount, min(calculated_amount, max_amount))
        
        self.logger.info(f"Allocation dynamique {signal.symbol}: {base_percentage}% de {available_balance:.8f} {quote_asset} = {final_amount:.6f} {quote_asset}")
        
        return final_amount, quote_asset
    
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
                # Headers par défaut pour toutes les requêtes
                headers = {}
                
                if method == "GET":
                    response = requests.get(url, params=params, timeout=timeout, headers=headers)
                elif method == "POST":
                    # Ajouter explicitement le Content-Type pour POST avec JSON
                    if json_data is not None:
                        headers['Content-Type'] = 'application/json'
                    response = requests.post(url, json=json_data, params=params, timeout=timeout, headers=headers)
                elif method == "DELETE":
                    response = requests.delete(url, params=params, timeout=timeout, headers=headers)
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
    
            # Déterminer l'actif et le montant nécessaire selon le côté de l'ordre
            base_asset = self._get_base_asset(signal.symbol)
            
            if signal.side == OrderSide.LONG:
                # LONG : On achète donc on a besoin de l'actif de cotation (quote asset)
                required_asset = quote_asset
                required_amount = trade_amount
                logger.info(f"LONG {signal.symbol}: Besoin de {required_amount:.6f} {required_asset}")
            else:  # OrderSide.SHORT
                # SHORT : On vend donc on a besoin de l'actif de base (base asset)
                required_asset = base_asset
                # Calculer la quantité d'actif de base nécessaire
                if signal.symbol.endswith("BTC"):
                    # Pour ETHBTC par exemple, calculer la quantité d'ETH
                    quantity = trade_amount / signal.price
                else:
                    # Pour les paires USDC, calculer la quantité d'actif de base
                    quantity = trade_amount / signal.price
                required_amount = quantity
                logger.info(f"SHORT {signal.symbol}: Besoin de {required_amount:.6f} {required_asset}")

            # Vérifier directement les balances Binance
            try:
                if not self._check_binance_balance(required_asset, required_amount):
                    logger.warning(f"❌ Solde {required_asset} insuffisant sur Binance pour le trade")
                    return None
            except Exception as e:
                logger.error(f"❌ Erreur lors de la vérification des soldes Binance: {str(e)}")
                return None
    
            # Convertir le montant en quantité
            if signal.symbol.endswith("BTC"):
                # Pour ETHBTC, trade_amount est déjà en BTC (ex: 0.00025 BTC)
                # Calculer directement la quantité d'ETH : BTC_amount / prix_ETHBTC
                quantity = trade_amount / signal.price
                logger.debug(f"📊 Calcul quantité ETHBTC: {trade_amount:.6f} BTC / {signal.price:.6f} = {quantity:.6f} ETH")
            else:
                # Pour les paires USDC, calcul direct
                quantity = trade_amount / signal.price
    
            # Calculer le stop-loss et trailing delta
            stop_price = signal.metadata.get('stop_price')
            trailing_delta = signal.metadata.get('trailing_delta')
    
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
            
            # Ajouter les paramètres de stop si disponibles
            if stop_price:
                order_data["stop_price"] = stop_price
            if trailing_delta:
                order_data["trailing_delta"] = trailing_delta
    
            # Plus de réservation de poches - on vérifie directement les balances Binance
    
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
                    return None
                
                cycle_id = result.get('order_id')
        
                # Appel au trader réussi
                self.trader_circuit.record_success()
        
                if not cycle_id:
                    logger.error("❌ Réponse invalide du Trader: pas d'ID de cycle")
                    return None
        
                logger.info(f"✅ Cycle de trading créé: {cycle_id} ({signal.side} {signal.symbol})")
                return cycle_id
            
            except requests.RequestException as e:
                self.trader_circuit.record_failure()
                logger.error(f"❌ Erreur lors de la création du cycle: {str(e)}")
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
                
                # Vérifier la force du signal (exemption pour les signaux agrégés)
                if signal.strength and signal.strength in [SignalStrength.WEAK] and not signal.strategy.startswith("Aggregated_"):
                    logger.info(f"⚠️ Signal ignoré: trop faible ({signal.strength})")
                    self.signal_queue.task_done()
                    continue
                
                # NOUVEAU: Utiliser le SmartCycleManager pour décider de l'action
                decision = self._process_signal_with_smart_manager(signal)
                
                if decision and decision.action != CycleAction.WAIT:
                    success = self._execute_smart_decision(decision)
                    if success:
                        logger.info(f"✅ Action {decision.action.value} exécutée: {decision.reason}")
                    else:
                        logger.warning(f"⚠️ Échec d'exécution de l'action {decision.action.value}")
                else:
                    logger.info(f"💤 Aucune action requise pour {signal.symbol}: {decision.reason if decision else 'Signal non traité'}")
                
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
        self.pubsub_client_id = self.redis_client.subscribe(self.signal_channel, self._process_signal)
        
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
        if hasattr(self, 'pubsub_client_id'):
            self.redis_client.unsubscribe(self.pubsub_client_id)
        
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
        
    def handle_cycle_completed(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite la fermeture d'un cycle et force une réconciliation des poches.
        """
        cycle_id = data.get('cycle_id')
        symbol = data.get('symbol')
        profit_loss = data.get('profit_loss', 0)
        
        logger.info(f"💰 Cycle fermé: {cycle_id} ({symbol}) - P&L: {profit_loss:.2f}")
        
        # Mettre à jour le cache du sync monitor
        if hasattr(self, 'sync_monitor') and self.sync_monitor:
            self.sync_monitor.remove_cycle_from_cache(cycle_id)
            logger.debug(f"🔄 Cycle {cycle_id} retiré du cache du sync monitor")
    
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
    
    def _check_binance_balance(self, asset: str, required_amount: float) -> bool:
        """
        Vérifie si on a assez de solde d'un actif sur Binance.
        
        Args:
            asset: Actif à vérifier (BTC, ETH, USDC, etc.)
            required_amount: Montant requis
            
        Returns:
            True si le solde est suffisant, False sinon
        """
        try:
            # Récupérer les balances depuis le trader
            url = f"{self.trader_api_url}/balance/{asset}"
            response = self._make_request_with_retry(url, timeout=2.0)
            
            if not response:
                logger.error(f"Impossible de récupérer le solde {asset}")
                return False
            
            available_balance = float(response.get('free', 0))
            
            # Ajouter une marge de sécurité de 1% pour les frais
            required_with_margin = required_amount * 1.01
            
            logger.info(f"Vérification solde {asset}: {available_balance:.8f} disponible, {required_with_margin:.8f} requis")
            
            if available_balance >= required_with_margin:
                return True
            else:
                logger.warning(f"❌ Solde {asset} insuffisant: {available_balance:.8f} < {required_with_margin:.8f}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du solde {asset}: {str(e)}")
            return False
    
    def _check_all_required_balances(self, signal: StrategySignal) -> Dict[str, Any]:
        """
        Vérifie toutes les balances nécessaires pour exécuter un trade.
        
        Args:
            signal: Signal de trading
            
        Returns:
            Dict avec 'sufficient', 'constraining_balance', 'reason', 'details'
        """
        try:
            base_asset = self._get_base_asset(signal.symbol)
            quote_asset = self._get_quote_asset(signal.symbol)
            
            # Récupérer les balances des deux actifs
            balances = {}
            for asset in [base_asset, quote_asset]:
                try:
                    # Vérifier d'abord Binance directement
                    binance_response = self._make_request_with_retry(
                        f"{self.trader_api_url}/balance/{asset}", timeout=2.0
                    )
                    if binance_response:
                        balances[asset] = {
                            'binance_free': float(binance_response.get('free', 0)),
                            'binance_total': float(binance_response.get('total', 0))
                        }
                    else:
                        balances[asset] = {'binance_free': 0, 'binance_total': 0}
                        
                    # Aussi récupérer du portfolio pour comparaison
                    portfolio_response = self._make_request_with_retry(
                        f"http://portfolio:8000/balance/{asset}", timeout=2.0
                    )
                    if portfolio_response:
                        balances[asset]['portfolio_available'] = float(portfolio_response.get('available', 0))
                    else:
                        balances[asset]['portfolio_available'] = 0
                        
                except Exception as e:
                    self.logger.warning(f"Erreur récupération balance {asset}: {e}")
                    balances[asset] = {'binance_free': 0, 'binance_total': 0, 'portfolio_available': 0}
            
            # Déterminer quelle balance est critique selon le type de trade
            if signal.side == OrderSide.LONG:
                # LONG: On achète, on a besoin de quote_asset (ex: BTC pour ETHBTC)
                critical_asset = quote_asset
                critical_balance = balances[quote_asset]['binance_free']
                
                # Calculer le montant réel qui sera tradé en passant la balance
                trade_amount, _ = self._calculate_trade_amount(signal, critical_balance)
                estimated_cost = trade_amount
                if critical_balance < estimated_cost:
                    return {
                        'sufficient': False,
                        'constraining_balance': critical_balance,
                        'reason': f"Solde {critical_asset} insuffisant: {critical_balance:.8f} < ~{estimated_cost:.8f}",
                        'details': balances
                    }
                
            else:  # OrderSide.SHORT
                # SHORT: On vend, on a besoin de base_asset (ex: ETH pour ETHBTC)
                critical_asset = base_asset
                critical_balance = balances[base_asset]['binance_free']
                
                # Estimer la quantité nécessaire (approximation)
                estimated_quantity = 0.01  # Quantité de base minimale
                if critical_balance < estimated_quantity:
                    return {
                        'sufficient': False,
                        'constraining_balance': critical_balance,
                        'reason': f"Solde {critical_asset} insuffisant: {critical_balance:.8f} < ~{estimated_quantity:.8f}",
                        'details': balances
                    }
            
            # Si on arrive ici, les balances sont suffisantes
            # NOUVEAU: Utiliser toujours les balances Binance réelles pour être cohérent
            if signal.side == OrderSide.LONG:
                # LONG: on a besoin de quote_asset, utiliser sa balance Binance réelle
                constraining_balance = balances[quote_asset]['binance_free'] * 0.95  # 5% de marge de sécurité
                self.logger.info(f"💡 LONG {signal.symbol}: balance contraignante basée sur {quote_asset}: "
                               f"{balances[quote_asset]['binance_free']:.6f} * 0.95 = {constraining_balance:.6f} {quote_asset}")
            else:  # OrderSide.SHORT
                # SHORT: on a besoin de base_asset, calculer l'équivalent en quote_asset
                available_base = balances[base_asset]['binance_free']
                # Convertir la quantité de base disponible en valeur quote (avec marge de sécurité)
                constraining_balance = available_base * signal.price * 0.9  # 10% de marge
                self.logger.info(f"💡 SHORT {signal.symbol}: balance contraignante basée sur {base_asset}: "
                               f"{available_base:.6f} * {signal.price:.6f} * 0.9 = {constraining_balance:.6f} {quote_asset}")
            
            self.logger.info(f"✅ Balances suffisantes pour {signal.side} {signal.symbol}: "
                           f"{base_asset}={balances[base_asset]['binance_free']:.6f}, "
                           f"{quote_asset}={balances[quote_asset]['binance_free']:.6f}")
            
            return {
                'sufficient': True,
                'constraining_balance': constraining_balance,
                'reason': 'Balances suffisantes',
                'details': balances
            }
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la vérification des balances: {str(e)}")
            return {
                'sufficient': False,
                'constraining_balance': 0,
                'reason': f"Erreur vérification: {str(e)}",
                'details': {}
            }
        
    def handle_cycle_canceled(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite l'annulation d'un cycle et libère les fonds.
        """
        cycle_id = data.get('cycle_id')
        logger.info(f"🚫 Cycle annulé: {cycle_id}")
        
        # Mettre à jour le cache du sync monitor
        if hasattr(self, 'sync_monitor') and self.sync_monitor:
            self.sync_monitor.remove_cycle_from_cache(cycle_id)
            logger.debug(f"🔄 Cycle {cycle_id} retiré du cache du sync monitor")

    def handle_cycle_failed(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite l'échec d'un cycle.
        """
        cycle_id = data.get('cycle_id')
        logger.info(f"❌ Cycle échoué: {cycle_id}")
        
        # Mettre à jour le cache du sync monitor
        if hasattr(self, 'sync_monitor') and self.sync_monitor:
            self.sync_monitor.remove_cycle_from_cache(cycle_id)
            logger.debug(f"🔄 Cycle {cycle_id} retiré du cache du sync monitor")
    
    def _handle_portfolio_update(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite les notifications de mise à jour du portfolio.
        
        Args:
            channel: Canal Redis d'où provient la notification
            data: Données de la notification (balances mises à jour)
        """
        try:
            logger.info(f"💰 Mise à jour du portfolio reçue")
            
            # Invalider le cache des cycles actifs pour forcer un rafraîchissement
            # lors du prochain signal
            self.cache_update_time = 0
            
            # Logger les changements de balance si disponibles
            if 'balances' in data:
                for asset, balance in data['balances'].items():
                    logger.debug(f"  {asset}: {balance.get('free', 0):.8f} libre, {balance.get('locked', 0):.8f} verrouillé")
                    
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de la mise à jour du portfolio: {str(e)}")
    
    def _process_signal_with_smart_manager(self, signal: StrategySignal) -> Optional['SmartCycleDecision']:
        """
        Traite un signal avec le SmartCycleManager pour prendre une décision intelligente.
        
        Args:
            signal: Signal reçu
            
        Returns:
            SmartCycleDecision ou None
        """
        try:
            # Récupérer le prix actuel
            current_price = signal.price
            
            # NOUVEAU: Vérifier toutes les balances nécessaires pour ce trade
            balance_check = self._check_all_required_balances(signal)
            if not balance_check['sufficient']:
                self.logger.warning(f"❌ Balances insuffisantes pour {signal.side} {signal.symbol}: {balance_check['reason']}")
                return None
            
            # Utiliser la balance contraignante comme available_balance
            available_balance = balance_check['constraining_balance']
            
            # Récupérer les cycles existants
            existing_cycles = []
            try:
                cycles_response = self._make_request_with_retry(f"{self.trader_api_url}/cycles?symbol={signal.symbol}")
                if cycles_response and 'cycles' in cycles_response:
                    existing_cycles = cycles_response['cycles']
            except Exception as e:
                self.logger.warning(f"Impossible de récupérer les cycles existants: {e}")
            
            # Demander au SmartCycleManager de prendre une décision
            decision = self.smart_cycle_manager.analyze_signal(
                signal=signal,
                current_price=current_price,
                available_balance=available_balance,
                existing_cycles=existing_cycles
            )
            
            self.logger.info(f"🧠 SmartCycleManager décision: {decision.action.value} - {decision.reason}")
            return decision
            
        except Exception as e:
            self.logger.error(f"❌ Erreur dans _process_signal_with_smart_manager: {str(e)}")
            return None
    
    def _execute_smart_decision(self, decision) -> bool:
        """
        Exécute la décision prise par le SmartCycleManager.
        
        Args:
            decision: SmartCycleDecision à exécuter
            
        Returns:
            True si succès, False sinon
        """
        try:
            if decision.action == CycleAction.CREATE_NEW:
                return self._create_new_smart_cycle(decision, getattr(decision, 'signal', None))
            
            elif decision.action == CycleAction.REINFORCE:
                return self._reinforce_existing_cycle(decision)
            
            elif decision.action == CycleAction.REDUCE:
                return self._reduce_cycle_position(decision)
            
            elif decision.action == CycleAction.CLOSE:
                return self._close_cycle_completely(decision)
            
            else:
                self.logger.warning(f"Action non supportée: {decision.action}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de l'exécution de la décision: {str(e)}")
            return False
    
    def _create_new_smart_cycle(self, decision, signal: StrategySignal = None) -> bool:
        """
        Crée un nouveau cycle basé sur une décision SmartCycleManager.
        
        Args:
            decision: Décision de création
            
        Returns:
            True si succès, False sinon
        """
        try:
            # Récupérer le signal depuis la décision
            original_signal = decision.signal if decision.signal else signal
            
            # Déterminer le side basé sur la side désirée
            # Si on veut une position LONG → signal LONG (acheter pour avoir l'actif)
            # Si on veut une position SHORT → signal SHORT (vendre pour ne plus avoir l'actif)
            if original_signal and hasattr(original_signal, 'side'):
                side = original_signal.side.value if hasattr(original_signal.side, 'value') else str(original_signal.side)
            else:
                side = "LONG"  # Par défaut
            
            # Préparer les données du cycle
            order_data = {
                "symbol": decision.symbol,
                "side": side,
                "quantity": decision.amount / decision.price_target if decision.price_target else decision.amount,
                "price": decision.price_target or 0,
                "strategy": f"SmartCycle_{decision.confidence:.0%}",
                "timestamp": int(time.time() * 1000),
                "metadata": {
                    "smart_cycle": True,
                    "reason": decision.reason,
                    "confidence": decision.confidence
                }
            }
            
            # Envoyer au trader
            result = self._make_request_with_retry(
                f"{self.trader_api_url}/order",
                method="POST",
                json_data=order_data,
                timeout=10.0
            )
            
            if result and result.get('order_id'):
                self.logger.info(f"✅ Nouveau SmartCycle créé: {result['order_id']}")
                return True
            else:
                self.logger.error(f"❌ Échec création SmartCycle: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erreur création SmartCycle: {str(e)}")
            return False
    
    def _reinforce_existing_cycle(self, decision) -> bool:
        """
        Renforce un cycle existant (DCA).
        
        Args:
            decision: Décision de renforcement
            
        Returns:
            True si succès, False sinon
        """
        # TODO: Implémenter le renforcement de cycle
        self.logger.warning(f"⚠️ Renforcement de cycle pas encore implémenté: {decision.existing_cycle_id}")
        return False
    
    def _reduce_cycle_position(self, decision) -> bool:
        """
        Réduit partiellement une position.
        
        
        Args:
            decision: Décision de réduction
            
        Returns:
            True si succès, False sinon
        """
        # 
        # : Implémenter la vente partielle
        self.logger.warning(f"⚠️ Vente partielle pas encore implémentée: {decision.existing_cycle_id}")
        return False
    
    def _close_cycle_completely(self, decision) -> bool:
        """
        Ferme complètement un cycle.
        
        Args:
            decision: Décision de fermeture
            
        Returns:
            True si succès, False sinon
        """
        try:
            # Envoyer la demande de fermeture au trader
            result = self._make_request_with_retry(
                f"{self.trader_api_url}/close/{decision.existing_cycle_id}",
                method="POST",
                json_data={"reason": decision.reason},
                timeout=10.0
            )
            
            if result and result.get('success'):
                self.logger.info(f"✅ Cycle {decision.existing_cycle_id} fermé: {decision.reason}")
                return True
            else:
                self.logger.error(f"❌ Échec fermeture cycle: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erreur fermeture cycle: {str(e)}")
            return False

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