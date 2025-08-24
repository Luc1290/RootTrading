"""
Buffer intelligent pour les signaux individuels.
Regroupe les signaux par symbole et timeframe pendant une fenêtre temporelle.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class IntelligentSignalBuffer:
    """
    Buffer qui regroupe intelligemment les signaux individuels en batches contextuels.
    
    Contrairement à l'ancien système qui attendait tous les signaux d'un cycle,
    ce buffer regroupe dynamiquement les signaux par contexte ET gère la synchronisation multi-timeframes.
    """
    
    def __init__(self, 
                 buffer_timeout: float = 5.0,      # Timeout en secondes
                 max_buffer_size: int = 50,        # Taille max du buffer
                 min_batch_size: int = 1,          # Taille min pour traiter un batch
                 sync_window: float = 3.0,         # Fenêtre de sync multi-TF en secondes
                 enable_mtf_sync: bool = True):    # Activer la sync multi-timeframes
        """
        Args:
            buffer_timeout: Temps max d'attente avant de traiter le buffer
            max_buffer_size: Nombre max de signaux avant traitement forcé
            min_batch_size: Nombre min de signaux pour former un batch
            sync_window: Fenêtre temporelle pour synchroniser les timeframes
            enable_mtf_sync: Activer la synchronisation multi-timeframes
        """
        self.buffer_timeout = buffer_timeout
        self.max_buffer_size = max_buffer_size
        self.min_batch_size = min_batch_size
        self.sync_window = sync_window
        self.enable_mtf_sync = enable_mtf_sync
        
        # Buffer principal groupé par (symbole, timeframe)
        self.signal_buffer: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        
        # Buffer multi-timeframes groupé par symbole seulement
        self.mtf_buffer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Timestamps pour gérer les timeouts
        self.first_signal_time: Dict[tuple, datetime] = {}
        self.first_mtf_signal_time: Dict[str, datetime] = {}
        
        # Lock pour thread safety
        self.buffer_lock = asyncio.Lock()
        
        # Callback pour traiter les batches
        self.batch_processor = None
        
        # Task pour gérer les timeouts
        self.timeout_task = None
        
        # Timeframes par ordre de priorité décisionnelle
        # ARCHITECTURE: 3m/5m = decision makers, 15m = context validator, 1m = timing tool
        self.timeframe_priority = {
            '1d': 1000, '4h': 400, '1h': 200, 
            '15m': 150,  # Context validator - fort pour régime
            '5m': 50,    # Core decision maker - poids élevé  
            '3m': 45,    # Core decision maker - poids élevé
            '1m': 10     # Timing tool seulement - influence réduite pour éviter bruit
        }
        
        # Pondération pour calcul de consensus (influence réelle sur décisions)
        self.decision_weights = {
            '15m': 0.25,  # 25% - Validation contexte/régime (peut être overridé par pump 1m)
            '5m': 0.35,   # 35% - Cœur décisionnel  
            '3m': 0.25,   # 25% - Cœur décisionnel
            '1m': 0.15    # 15% - PUMP DETECTOR - peut overrider si signal explosif
        }
        
        # Seuils pour détection de pump/dump sur 1m
        self.pump_detection_thresholds = {
            'min_confidence': 0.85,  # Confidence minimale pour détecter un pump
            'volume_boost': 2.0,     # Volume multiplié par 2+ (si disponible)
            'momentum_boost': 1.5    # Momentum élevé (si disponible)
        }
        
        # Statistiques (inclut tracking des pumps détectés)
        self.stats = {
            'signals_buffered': 0,
            'batches_processed': 0,
            'timeout_triggers': 0,
            'size_triggers': 0,
            'mtf_sync_triggers': 0,
            'mtf_batches_processed': 0,
            'pumps_detected': 0,
            'pump_overrides': 0
        }
        
    def set_batch_processor(self, processor_callback):
        """Définit la fonction de callback pour traiter les batches."""
        self.batch_processor = processor_callback
        
    def _detect_pump_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Détecte si un signal 1m est un pump/dump explosif qui doit override les autres TF.
        
        Args:
            signal: Signal 1m à analyser
            
        Returns:
            bool: True si c'est un pump détecté
        """
        if signal.get('timeframe') != '1m':
            return False
            
        confidence = signal.get('confidence', 0.0)
        metadata = signal.get('metadata', {})
        
        # Critère 1: Confidence très élevée (signal très fort)
        if confidence < self.pump_detection_thresholds['min_confidence']:
            return False
            
        # Critère 2: Volume explosif (si disponible)
        volume_ratio = metadata.get('volume_ratio', 1.0)
        has_volume_boost = volume_ratio >= self.pump_detection_thresholds['volume_boost']
        
        # Critère 3: Momentum explosif (si disponible) 
        momentum_score = metadata.get('momentum_score', 1.0)
        has_momentum_boost = momentum_score >= self.pump_detection_thresholds['momentum_boost']
        
        # Critère 4: Confluence très élevée (si disponible)
        confluence_score = metadata.get('confluence_score', 0.5)
        has_confluence = confluence_score >= 0.8
        
        # Critères additionnels spécifiques pumps
        breakout_strength = metadata.get('breakout_strength', 0.0)
        has_breakout = breakout_strength >= 0.7
        
        # DÉTECTION PUMP: Confidence très haute + au moins 2 critères explosifs
        explosive_criteria_count = sum([
            has_volume_boost,
            has_momentum_boost, 
            has_confluence,
            has_breakout
        ])
        
        is_pump = explosive_criteria_count >= 2
        
        if is_pump:
            logger.info(f"🚀 PUMP DÉTECTÉ {signal.get('symbol')} {signal.get('side')}: "
                       f"confidence={confidence:.2f}, volume={volume_ratio:.1f}x, "
                       f"momentum={momentum_score:.1f}x, confluence={confluence_score:.2f}, "
                       f"breakout={breakout_strength:.2f}")
            self.stats['pumps_detected'] += 1
            
        return is_pump
        
    async def add_signal(self, signal: Dict[str, Any]) -> None:
        """
        Ajoute un signal au buffer et déclenche le traitement si nécessaire.
        
        Args:
            signal: Signal à ajouter
        """
        async with self.buffer_lock:
            symbol = signal.get('symbol', 'UNKNOWN')
            timeframe = signal.get('timeframe', '5m')
            context_key = (symbol, timeframe)
            
            # Ajouter au buffer spécifique par timeframe
            self.signal_buffer[context_key].append(signal)
            self.stats['signals_buffered'] += 1
            
            # Ajouter au buffer multi-timeframes si activé
            if self.enable_mtf_sync:
                self.mtf_buffer[symbol].append(signal)
                
                # Marquer le timestamp du premier signal MTF pour ce symbole
                if symbol not in self.first_mtf_signal_time:
                    self.first_mtf_signal_time[symbol] = datetime.utcnow()
            
            # Marquer le timestamp du premier signal pour ce contexte
            if context_key not in self.first_signal_time:
                self.first_signal_time[context_key] = datetime.utcnow()
                
            # Démarrer le task de timeout si pas déjà actif
            if not self.timeout_task or self.timeout_task.done():
                self.timeout_task = asyncio.create_task(self._timeout_monitor())
                
            logger.debug(f"Signal ajouté: {symbol} {timeframe} "
                        f"(TF buffer: {len(self.signal_buffer[context_key])}, "
                        f"MTF buffer: {len(self.mtf_buffer[symbol]) if self.enable_mtf_sync else 'disabled'})")
            
            # Si MTF activé, SEULEMENT MTF (pas de traitement par TF individuel)
            if self.enable_mtf_sync:
                await self._check_mtf_processing(symbol)
            else:
                await self._check_immediate_processing(context_key)
            
    async def _check_immediate_processing(self, context_key: tuple) -> None:
        """Vérifie s'il faut traiter immédiatement un contexte."""
        buffer_size = len(self.signal_buffer[context_key])
        
        # Déclencher si le buffer est trop grand
        if buffer_size >= self.max_buffer_size:
            logger.info(f"Déclenchement par taille: {buffer_size} signaux pour {context_key}")
            await self._process_context(context_key, trigger="size")
            self.stats['size_triggers'] += 1
            
    async def _check_mtf_processing(self, symbol: str) -> None:
        """Vérifie s'il faut traiter immédiatement le buffer multi-timeframes."""
        if symbol not in self.mtf_buffer:
            return
            
        mtf_signals = self.mtf_buffer[symbol]
        buffer_size = len(mtf_signals)
        
        # Déclencher immédiatement si trop de signaux
        if buffer_size >= self.max_buffer_size:
            logger.info(f"Déclenchement MTF par taille: {buffer_size} signaux pour {symbol}")
            await self._process_mtf_symbol(symbol, trigger="size")
            self.stats['size_triggers'] += 1
            return
            
        # Analyser la diversité des timeframes
        timeframes_present = set(s.get('timeframe', '5m') for s in mtf_signals)
        directions_present = set(s.get('side', 'UNKNOWN') for s in mtf_signals)
        
        # Conditions de déclenchement intelligent :
        
        # 1. Si on a des signaux de TOUS les timeframes principaux (3m+5m+15m) dans la même direction
        expected_timeframes = {'3m', '5m', '15m'}  # Timeframes principaux de l'analyzer
        has_all_main_timeframes = expected_timeframes.issubset(timeframes_present)
        
        if has_all_main_timeframes and len(directions_present) == 1:
            logger.info(f"Déclenchement MTF par timeframes complets: {len(timeframes_present)} TFs "
                       f"({list(timeframes_present)}), 1 direction pour {symbol}")
            await self._process_mtf_symbol(symbol, trigger="complete_timeframes")
            self.stats['mtf_sync_triggers'] += 1
            return
            
        # 2. Fallback: Si on a 3+ timeframes différents dans la même direction
        elif len(timeframes_present) >= 3 and len(directions_present) == 1:
            logger.info(f"Déclenchement MTF par diversité TF: {len(timeframes_present)} TFs, "
                       f"1 direction pour {symbol}")
            await self._process_mtf_symbol(symbol, trigger="timeframe_diversity") 
            self.stats['mtf_sync_triggers'] += 1
            return
            
        # 3. GESTION DES CONFLITS : Signaux opposés sur différents timeframes
        if len(directions_present) >= 2 and len(timeframes_present) >= 2:
            # Analyser la hiérarchie des conflits
            conflict_analysis = self._analyze_mtf_conflicts(mtf_signals)
            
            if conflict_analysis['should_process']:
                logger.info(f"Déclenchement MTF par conflit résolvable: "
                           f"{conflict_analysis['resolution']} pour {symbol}")
                await self._process_mtf_symbol(symbol, trigger="conflict_resolution")
                self.stats['mtf_sync_triggers'] += 1
                return
            else:
                logger.debug(f"Conflit MTF non résolvable pour {symbol}, attente de plus de signaux")
            
        # 4. Si on a un signal de timeframe élevé (1h+) avec confirmation courte
        high_tf_signals = [s for s in mtf_signals 
                          if self.timeframe_priority.get(s.get('timeframe', '5m'), 0) >= 100]
        
        if high_tf_signals and buffer_size >= 2:
            logger.info(f"Déclenchement MTF par TF élevé: {len(high_tf_signals)} signaux 1h+ "
                       f"avec {buffer_size-len(high_tf_signals)} confirmations pour {symbol}")
            await self._process_mtf_symbol(symbol, trigger="high_timeframe")
            self.stats['mtf_sync_triggers'] += 1
            return
            
        # 5. Si fenêtre de sync expirée et assez de signaux
        first_time = self.first_mtf_signal_time.get(symbol)
        if first_time:
            elapsed = (datetime.utcnow() - first_time).total_seconds()
            if elapsed >= self.sync_window and buffer_size >= self.min_batch_size:
                logger.info(f"Déclenchement MTF par timeout sync: {elapsed:.1f}s, "
                           f"{buffer_size} signaux pour {symbol}")
                await self._process_mtf_symbol(symbol, trigger="sync_timeout")
                self.stats['mtf_sync_triggers'] += 1
            
    async def _timeout_monitor(self) -> None:
        """Monitor qui vérifie périodiquement les timeouts."""
        while True:
            try:
                await asyncio.sleep(1.0)  # Vérifier chaque seconde
                
                async with self.buffer_lock:
                    now = datetime.utcnow()
                    
                    # Si MTF activé, NE PAS traiter les timeframes individuels en timeout
                    if not self.enable_mtf_sync:
                        contexts_to_process = []
                        
                        for context_key, first_time in self.first_signal_time.items():
                            if context_key in self.signal_buffer and self.signal_buffer[context_key]:
                                elapsed = (now - first_time).total_seconds()
                                buffer_size = len(self.signal_buffer[context_key])
                                
                                # Déclencher par timeout
                                if elapsed >= self.buffer_timeout and buffer_size >= self.min_batch_size:
                                    contexts_to_process.append(context_key)
                                    
                        # Traiter les contextes en timeout (seulement si MTF désactivé)
                        for context_key in contexts_to_process:
                            logger.info(f"Déclenchement par timeout: {context_key}")
                            await self._process_context(context_key, trigger="timeout")
                            self.stats['timeout_triggers'] += 1
                    else:
                        # Mode MTF : traiter seulement les timeouts MTF
                        mtf_symbols_to_process = []
                        
                        for symbol, first_time in self.first_mtf_signal_time.items():
                            if symbol in self.mtf_buffer and self.mtf_buffer[symbol]:
                                elapsed = (now - first_time).total_seconds()
                                buffer_size = len(self.mtf_buffer[symbol])
                                
                                # Déclencher par timeout MTF si on a assez de signaux
                                if elapsed >= self.buffer_timeout and buffer_size >= self.min_batch_size:
                                    mtf_symbols_to_process.append(symbol)
                        
                        # Traiter les MTF timeouts
                        for symbol in mtf_symbols_to_process:
                            logger.info(f"Déclenchement MTF par timeout global: {symbol}")
                            await self._process_mtf_symbol(symbol, trigger="timeout_global")
                            if symbol in self.first_mtf_signal_time:
                                del self.first_mtf_signal_time[symbol]
                            self.stats['timeout_triggers'] += 1
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur dans timeout_monitor: {e}")
                
    async def _process_context(self, context_key: tuple, trigger: str = "manual") -> None:
        """
        Traite tous les signaux d'un contexte spécifique.
        
        Args:
            context_key: Clé du contexte (symbole, timeframe)
            trigger: Raison du déclenchement
        """
        if context_key not in self.signal_buffer or not self.signal_buffer[context_key]:
            return
            
        # Extraire les signaux
        signals = self.signal_buffer[context_key].copy()
        
        # Nettoyer le buffer
        del self.signal_buffer[context_key]
        if context_key in self.first_signal_time:
            del self.first_signal_time[context_key]
            
        logger.info(f"Traitement contexte {context_key}: {len(signals)} signaux (trigger: {trigger})")
        
        # Traiter le batch si on a un processor
        if self.batch_processor and signals:
            try:
                await self.batch_processor(signals, context_key)
                self.stats['batches_processed'] += 1
            except Exception as e:
                logger.error(f"Erreur traitement batch {context_key}: {e}")
                
    async def _process_mtf_symbol(self, symbol: str, trigger: str = "manual") -> None:
        """
        Traite tous les signaux multi-timeframes d'un symbole.
        
        Args:
            symbol: Symbole à traiter
            trigger: Raison du déclenchement
        """
        if symbol not in self.mtf_buffer or not self.mtf_buffer[symbol]:
            return
            
        # Extraire les signaux
        signals = self.mtf_buffer[symbol].copy()
        
        # Nettoyer le buffer MTF
        del self.mtf_buffer[symbol]
        if symbol in self.first_mtf_signal_time:
            del self.first_mtf_signal_time[symbol]
            
        # Nettoyer aussi les buffers individuels de ce symbole
        contexts_to_clean = []
        for context_key in self.signal_buffer.keys():
            if context_key[0] == symbol:  # Premier élément = symbole
                contexts_to_clean.append(context_key)
                
        for context_key in contexts_to_clean:
            if context_key in self.signal_buffer:
                del self.signal_buffer[context_key]
            if context_key in self.first_signal_time:
                del self.first_signal_time[context_key]
                
        # Trier par priorité de timeframe (plus élevé = plus important)
        signals.sort(key=lambda s: self.timeframe_priority.get(s.get('timeframe', '5m'), 0), 
                    reverse=True)
                    
        # Analyser la composition
        timeframes = [s.get('timeframe', '5m') for s in signals]
        sides = [s.get('side', 'UNKNOWN') for s in signals]
        strategies = [s.get('strategy', 'Unknown') for s in signals]
        
        logger.info(f"Traitement MTF {symbol}: {len(signals)} signaux "
                   f"(TFs: {list(set(timeframes))}, Sides: {list(set(sides))}, "
                   f"Trigger: {trigger})")
        
        # Analyser les conflits potentiels avant traitement
        unique_sides = set(s.get('side', 'UNKNOWN') for s in signals)
        final_signals = signals
        original_signal_count = len(signals)  # Conserver le nombre original
        
        if len(unique_sides) > 1:
            # Il y a un conflit, analyser et filtrer
            conflict_analysis = self._analyze_mtf_conflicts(signals)
            
            if conflict_analysis['should_process'] and conflict_analysis['winning_side']:
                # Filtrer pour ne garder que les signaux de la direction gagnante
                winning_side = conflict_analysis['winning_side']
                final_signals = [s for s in signals if s.get('side') == winning_side]
                
                # Ajouter le nombre original dans les métadonnées de chaque signal
                for signal in final_signals:
                    if 'metadata' not in signal:
                        signal['metadata'] = {}
                    signal['metadata']['original_signal_count'] = original_signal_count
                    signal['metadata']['mtf_conflict_resolved'] = True
                    signal['metadata']['conflict_resolution'] = conflict_analysis['resolution']
                
                logger.info(f"Conflit MTF résolu pour {symbol}: {conflict_analysis['resolution']} "
                           f"→ {len(final_signals)} signaux {winning_side} retenus sur {original_signal_count}")
            else:
                logger.warning(f"Conflit MTF non résolvable pour {symbol}, "
                              f"abandon du traitement : {conflict_analysis['resolution']}")
                return  # Ne pas traiter si conflit non résolvable
        
        # Traiter le batch MTF si on a un processor
        if self.batch_processor and final_signals:
            try:
                # Créer une clé spéciale pour le batch MTF
                mtf_context_key = (symbol, "multi_timeframe")
                await self.batch_processor(final_signals, mtf_context_key)
                self.stats['mtf_batches_processed'] += 1
            except Exception as e:
                logger.error(f"Erreur traitement batch MTF {symbol}: {e}")
                
    def _analyze_mtf_conflicts(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyse les conflits entre signaux multi-timeframes et détermine la résolution.
        
        Args:
            signals: Liste des signaux en conflit
            
        Returns:
            Dict avec l'analyse du conflit et la stratégie de résolution
        """
        # Grouper par direction
        buy_signals = [s for s in signals if s.get('side') == 'BUY']
        sell_signals = [s for s in signals if s.get('side') == 'SELL']
        
        # Calculer le score de chaque groupe avec pondération décisionnelle intelligente
        def calculate_group_score(group_signals):
            if not group_signals:
                return 0.0
                
            total_weighted_score = 0
            total_weight = 0
            
            for signal in group_signals:
                timeframe = signal.get('timeframe', '5m')
                confidence = signal.get('confidence', 0.5)
                
                # Utiliser les poids décisionnels au lieu des priorités brutes
                decision_weight = self.decision_weights.get(timeframe, 0.01)
                
                # Règles spéciales selon rôle du timeframe
                if timeframe == '1m':
                    # 1m = timing tool : fort impact SEULEMENT si très confiant OU opposé fort consensus
                    if confidence >= 0.8:
                        adjusted_weight = decision_weight * 2  # Boost si très confiant
                    elif len(group_signals) == 1:  # Seul signal 1m contre consensus
                        adjusted_weight = decision_weight * 0.3  # Réduire influence isolée
                    else:
                        adjusted_weight = decision_weight
                        
                elif timeframe in ['3m', '5m']:
                    # 3m/5m = decision makers : poids normal, boost si consensus interne
                    core_signals = [s for s in group_signals if s.get('timeframe') in ['3m', '5m']]
                    if len(core_signals) >= 2:
                        adjusted_weight = decision_weight * 1.3  # Boost consensus 3m+5m
                    else:
                        adjusted_weight = decision_weight
                        
                elif timeframe == '15m':
                    # 15m = context validator : poids stable, crucial pour direction
                    adjusted_weight = decision_weight
                    
                else:
                    # Timeframes plus élevés : poids maximal
                    adjusted_weight = decision_weight * 2
                
                # Score final pondéré
                signal_score = confidence * adjusted_weight
                total_weighted_score += signal_score
                total_weight += adjusted_weight
                
            # Retourner score moyen pondéré avec bonus de consensus
            avg_score = total_weighted_score / max(total_weight, 0.01)
            
            # Bonus si multiple signaux (consensus interne)
            if len(group_signals) >= 2:
                consensus_bonus = min(0.2, (len(group_signals) - 1) * 0.1)
                avg_score += consensus_bonus
                
            return avg_score
        
        buy_score = calculate_group_score(buy_signals)
        sell_score = calculate_group_score(sell_signals)
        
        # Analyser la composition des timeframes
        buy_timeframes = [s.get('timeframe', '5m') for s in buy_signals]
        sell_timeframes = [s.get('timeframe', '5m') for s in sell_signals]
        
        buy_high_tf = len([tf for tf in buy_timeframes 
                          if self.timeframe_priority.get(tf, 0) >= 100])
        sell_high_tf = len([tf for tf in sell_timeframes 
                           if self.timeframe_priority.get(tf, 0) >= 100])
        
        # Analyser la composition par rôles de timeframes
        buy_core_signals = [s for s in buy_signals if s.get('timeframe') in ['3m', '5m']]
        sell_core_signals = [s for s in sell_signals if s.get('timeframe') in ['3m', '5m']]
        buy_timing_signals = [s for s in buy_signals if s.get('timeframe') == '1m']
        sell_timing_signals = [s for s in sell_signals if s.get('timeframe') == '1m']
        buy_context_signals = [s for s in buy_signals if s.get('timeframe') == '15m']
        sell_context_signals = [s for s in sell_signals if s.get('timeframe') == '15m']
        
        # Règles de résolution des conflits HIÉRARCHIQUES
        resolution_rules = []
        should_process = False
        winning_side = None
        
        # RÈGLE 0 (PRIORITÉ ABSOLUE): DÉTECTION PUMP 1m - override tout autre signal
        buy_pumps = [s for s in buy_timing_signals if self._detect_pump_signal(s)]
        sell_pumps = [s for s in sell_timing_signals if self._detect_pump_signal(s)]
        
        if buy_pumps and not sell_pumps:
            resolution_rules.append(f"🚀 PUMP BUY détecté - override tous les autres timeframes ({len(buy_pumps)} pump(s))")
            winning_side = "BUY"
            should_process = True
            self.stats['pump_overrides'] += 1
        elif sell_pumps and not buy_pumps:
            resolution_rules.append(f"📉 DUMP SELL détecté - override tous les autres timeframes ({len(sell_pumps)} dump(s))")
            winning_side = "SELL" 
            should_process = True
            self.stats['pump_overrides'] += 1
        elif buy_pumps and sell_pumps:
            # Conflit de pumps simultanés - prendre le plus fort
            buy_pump_strength = sum(s.get('confidence', 0) for s in buy_pumps)
            sell_pump_strength = sum(s.get('confidence', 0) for s in sell_pumps)
            if buy_pump_strength > sell_pump_strength:
                resolution_rules.append(f"🚀 PUMP BUY plus fort que DUMP SELL ({buy_pump_strength:.2f} vs {sell_pump_strength:.2f})")
                winning_side = "BUY"
                should_process = True
                self.stats['pump_overrides'] += 1
            else:
                resolution_rules.append(f"📉 DUMP SELL plus fort que PUMP BUY ({sell_pump_strength:.2f} vs {buy_pump_strength:.2f})")
                winning_side = "SELL"
                should_process = True
                self.stats['pump_overrides'] += 1
        
        # Règle 1: DOMINANCE CONTEXTE 15m (régime/direction long terme) - seulement si pas de pump
        elif buy_context_signals and not sell_context_signals:
            resolution_rules.append("BUY dominé par contexte 15m seul")
            winning_side = "BUY"
            should_process = True
        elif sell_context_signals and not buy_context_signals:
            resolution_rules.append("SELL dominé par contexte 15m seul")
            winning_side = "SELL"
            should_process = True
            
        # Règle 2: CONSENSUS DÉCISIONNEL 3m+5m (cœur de la décision)
        elif len(buy_core_signals) >= 2 and len(sell_core_signals) == 0:
            resolution_rules.append(f"BUY consensus décisionnel fort ({len(buy_core_signals)} signaux 3m+5m)")
            winning_side = "BUY"
            should_process = True
        elif len(sell_core_signals) >= 2 and len(buy_core_signals) == 0:
            resolution_rules.append(f"SELL consensus décisionnel fort ({len(sell_core_signals)} signaux 3m+5m)")
            winning_side = "SELL"
            should_process = True
            
        # Règle 3: SCORE PONDÉRÉ avec seuils adaptés aux nouveaux poids
        else:
            max_score = max(buy_score, sell_score, 0.01)
            score_diff = abs(buy_score - sell_score)
            relative_diff = score_diff / max_score
            
            # Seuils ajustés pour les nouveaux poids décisionnels (plus bas car scores plus petits)
            if score_diff > 0.05 or relative_diff > 0.15:  # 5% absolu OU 15% relatif
                if buy_score > sell_score:
                    resolution_rules.append(f"BUY score pondéré supérieur ({buy_score:.3f} vs {sell_score:.3f})")
                    winning_side = "BUY"
                    should_process = True
                else:
                    resolution_rules.append(f"SELL score pondéré supérieur ({sell_score:.3f} vs {buy_score:.3f})")
                    winning_side = "SELL"
                    should_process = True
            # Règle 4: GESTION SPÉCIALE 1m (timing seulement)  
            elif buy_timing_signals and not sell_timing_signals and len(buy_core_signals) >= 1:
                # 1m BUY + au moins 1 signal décisionnel → mais vérifier qu'il n'est pas isolé
                if len(buy_signals) >= 2:  # 1m pas seul
                    resolution_rules.append(f"BUY avec support timing 1m + décisionnel")
                    winning_side = "BUY"
                    should_process = True
                else:
                    resolution_rules.append("Signal 1m isolé ignoré - attendre confirmation")
            elif sell_timing_signals and not buy_timing_signals and len(sell_core_signals) >= 1:
                if len(sell_signals) >= 2:
                    resolution_rules.append(f"SELL avec support timing 1m + décisionnel")
                    winning_side = "SELL"
                    should_process = True
                else:
                    resolution_rules.append("Signal 1m isolé ignoré - attendre confirmation")
                    
            # Règle 5: Majorité décisionnelle dans les core timeframes
            elif len(buy_core_signals) > len(sell_core_signals) and len(buy_core_signals) >= 1:
                resolution_rules.append(f"BUY majorité décisionnelle ({len(buy_core_signals)} vs {len(sell_core_signals)} core signals)")
                winning_side = "BUY"
                should_process = True
            elif len(sell_core_signals) > len(buy_core_signals) and len(sell_core_signals) >= 1:
                resolution_rules.append(f"SELL majorité décisionnelle ({len(sell_core_signals)} vs {len(buy_core_signals)} core signals)")
                winning_side = "SELL"
                should_process = True
            
            # Règle 6: Fallback - confidence moyenne si pas d'autre résolution
            else:
                avg_buy_conf = sum(s.get('confidence', 0.5) for s in buy_signals) / len(buy_signals) if buy_signals else 0
                avg_sell_conf = sum(s.get('confidence', 0.5) for s in sell_signals) / len(sell_signals) if sell_signals else 0
                
                conf_diff = abs(avg_buy_conf - avg_sell_conf)
                if conf_diff > 0.08:  # 8% d'écart suffit 
                    if avg_buy_conf > avg_sell_conf:
                        resolution_rules.append(f"BUY confidence supérieure ({avg_buy_conf:.2f} vs {avg_sell_conf:.2f})")
                        winning_side = "BUY"
                        should_process = True
                    else:
                        resolution_rules.append(f"SELL confidence supérieure ({avg_sell_conf:.2f} vs {avg_buy_conf:.2f})")
                        winning_side = "SELL"
                        should_process = True
                else:
                    resolution_rules.append("Conflit équilibré - attendre signaux supplémentaires pour départager")
        
        # Décision finale : pas de traitement si conflit équilibré
        if not should_process:
            resolution_rules.append("Attente de signaux supplémentaires pour départager")
        
        return {
            'should_process': should_process,
            'winning_side': winning_side,
            'resolution': '; '.join(resolution_rules),
            'buy_signals_count': len(buy_signals),
            'sell_signals_count': len(sell_signals),
            'buy_score': buy_score,
            'sell_score': sell_score,
            # Analyse par rôles de timeframes
            'buy_core_signals': len(buy_core_signals),
            'sell_core_signals': len(sell_core_signals),
            'buy_timing_signals': len(buy_timing_signals),
            'sell_timing_signals': len(sell_timing_signals),
            'buy_context_signals': len(buy_context_signals),
            'sell_context_signals': len(sell_context_signals),
            'conflict_type': 'balanced' if not should_process else 'resolvable',
            'decision_logic': 'hierarchical_timeframe_weighting'
        }
                
    async def force_flush_all(self) -> None:
        """Force le traitement de tous les signaux en buffer."""
        async with self.buffer_lock:
            contexts_to_process = list(self.signal_buffer.keys())
            
        for context_key in contexts_to_process:
            await self._process_context(context_key, trigger="flush")
            
        logger.info(f"Flush forcé: {len(contexts_to_process)} contextes traités")
        
    async def get_buffer_status(self) -> Dict[str, Any]:
        """Retourne le statut actuel du buffer."""
        async with self.buffer_lock:
            total_buffered = sum(len(signals) for signals in self.signal_buffer.values())
            contexts_count = len(self.signal_buffer)
            
            context_details = {}
            for (symbol, timeframe), signals in self.signal_buffer.items():
                first_time = self.first_signal_time.get((symbol, timeframe))
                elapsed = (datetime.utcnow() - first_time).total_seconds() if first_time else 0
                
                context_details[f"{symbol}_{timeframe}"] = {
                    'signal_count': len(signals),
                    'elapsed_seconds': elapsed,
                    'will_timeout_soon': elapsed >= (self.buffer_timeout - 1.0)
                }
                
        return {
            'total_buffered_signals': total_buffered,
            'active_contexts': contexts_count,
            'context_details': context_details,
            'stats': self.stats.copy()
        }
        
    async def cleanup(self) -> None:
        """Nettoie les ressources du buffer."""
        if self.timeout_task and not self.timeout_task.done():
            self.timeout_task.cancel()
            try:
                await self.timeout_task
            except asyncio.CancelledError:
                pass
                
        await self.force_flush_all()
        logger.info("Buffer nettoyé")