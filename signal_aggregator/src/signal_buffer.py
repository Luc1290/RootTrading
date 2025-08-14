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
        
        # Timeframes par ordre de priorité (plus élevé = plus important)
        self.timeframe_priority = {
            '1d': 1000, '4h': 400, '1h': 100, 
            '15m': 15, '5m': 5, '3m': 3, '1m': 1
        }
        
        # Statistiques
        self.stats = {
            'signals_buffered': 0,
            'batches_processed': 0,
            'timeout_triggers': 0,
            'size_triggers': 0,
            'mtf_sync_triggers': 0,
            'mtf_batches_processed': 0
        }
        
    def set_batch_processor(self, processor_callback):
        """Définit la fonction de callback pour traiter les batches."""
        self.batch_processor = processor_callback
        
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
            
            # Vérifier le traitement immédiat (priorité au MTF si activé)
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
        
        # 1. Si on a des signaux de 3+ timeframes différents dans la même direction
        if len(timeframes_present) >= 3 and len(directions_present) == 1:
            logger.info(f"Déclenchement MTF par diversité TF: {len(timeframes_present)} TFs, "
                       f"1 direction pour {symbol}")
            await self._process_mtf_symbol(symbol, trigger="timeframe_diversity")
            self.stats['mtf_sync_triggers'] += 1
            return
            
        # 2. GESTION DES CONFLITS : Signaux opposés sur différents timeframes
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
            
        # 3. Si on a un signal de timeframe élevé (1h+) avec confirmation courte
        high_tf_signals = [s for s in mtf_signals 
                          if self.timeframe_priority.get(s.get('timeframe', '5m'), 0) >= 100]
        
        if high_tf_signals and buffer_size >= 2:
            logger.info(f"Déclenchement MTF par TF élevé: {len(high_tf_signals)} signaux 1h+ "
                       f"avec {buffer_size-len(high_tf_signals)} confirmations pour {symbol}")
            await self._process_mtf_symbol(symbol, trigger="high_timeframe")
            self.stats['mtf_sync_triggers'] += 1
            return
            
        # 4. Si fenêtre de sync expirée et assez de signaux
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
                    contexts_to_process = []
                    
                    for context_key, first_time in self.first_signal_time.items():
                        if context_key in self.signal_buffer and self.signal_buffer[context_key]:
                            elapsed = (now - first_time).total_seconds()
                            buffer_size = len(self.signal_buffer[context_key])
                            
                            # Déclencher par timeout
                            if elapsed >= self.buffer_timeout and buffer_size >= self.min_batch_size:
                                contexts_to_process.append(context_key)
                                
                    # Traiter les contextes en timeout
                    for context_key in contexts_to_process:
                        logger.info(f"Déclenchement par timeout: {context_key}")
                        await self._process_context(context_key, trigger="timeout")
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
        
        if len(unique_sides) > 1:
            # Il y a un conflit, analyser et filtrer
            conflict_analysis = self._analyze_mtf_conflicts(signals)
            
            if conflict_analysis['should_process'] and conflict_analysis['winning_side']:
                # Filtrer pour ne garder que les signaux de la direction gagnante
                winning_side = conflict_analysis['winning_side']
                final_signals = [s for s in signals if s.get('side') == winning_side]
                
                logger.info(f"Conflit MTF résolu pour {symbol}: {conflict_analysis['resolution']} "
                           f"→ {len(final_signals)} signaux {winning_side} retenus sur {len(signals)}")
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
        
        # Calculer le score de chaque groupe basé sur le timeframe et la confidence
        def calculate_group_score(group_signals):
            if not group_signals:
                return 0.0
                
            total_score = 0
            for signal in group_signals:
                # Score basé sur la priorité du timeframe
                tf_score = self.timeframe_priority.get(signal.get('timeframe', '5m'), 1)
                
                # Score basé sur la confidence du signal
                confidence = signal.get('confidence', 0.5)
                
                # Facteur de pondération : timeframe plus élevé = plus important
                if tf_score >= 400:  # 4h+
                    weight = 3.0
                elif tf_score >= 100:  # 1h+
                    weight = 2.5
                elif tf_score >= 15:  # 15m+
                    weight = 2.0
                else:  # < 15m
                    weight = 1.0
                    
                # FIX: Normaliser tf_score différemment pour avoir des scores lisibles
                normalized_tf_score = tf_score / 100.0  # Au lieu de /1000, utiliser /100
                signal_score = normalized_tf_score * confidence * weight
                total_score += signal_score
                
            return total_score / len(group_signals)  # Score moyen pondéré
        
        buy_score = calculate_group_score(buy_signals)
        sell_score = calculate_group_score(sell_signals)
        
        # Analyser la composition des timeframes
        buy_timeframes = [s.get('timeframe', '5m') for s in buy_signals]
        sell_timeframes = [s.get('timeframe', '5m') for s in sell_signals]
        
        buy_high_tf = len([tf for tf in buy_timeframes 
                          if self.timeframe_priority.get(tf, 0) >= 100])
        sell_high_tf = len([tf for tf in sell_timeframes 
                           if self.timeframe_priority.get(tf, 0) >= 100])
        
        # Règles de résolution des conflits
        resolution_rules = []
        should_process = False
        winning_side = None
        
        # Règle 1: Dominance claire d'un timeframe élevé
        if buy_high_tf >= 1 and sell_high_tf == 0:
            resolution_rules.append("BUY dominé par timeframes élevés")
            winning_side = "BUY"
            should_process = True
        elif sell_high_tf >= 1 and buy_high_tf == 0:
            resolution_rules.append("SELL dominé par timeframes élevés")
            winning_side = "SELL"
            should_process = True
            
        # Règle 2: Score pondéré significativement différent
        else:
            max_score = max(buy_score, sell_score, 0.01)  # Éviter division par 0
            score_diff = abs(buy_score - sell_score)
            relative_diff = score_diff / max_score
            
            if score_diff > 0.1 or relative_diff > 0.2:  # 10% d'écart absolu OU 20% relatif
                if buy_score > sell_score:
                    resolution_rules.append(f"BUY score supérieur ({buy_score:.2f} vs {sell_score:.2f})")
                    winning_side = "BUY"
                    should_process = True
                else:
                    resolution_rules.append(f"SELL score supérieur ({sell_score:.2f} vs {buy_score:.2f})")
                    winning_side = "SELL"
                    should_process = True
            
            # Règle 3: Majorité claire d'une direction (2:1 minimum)
            elif len(buy_signals) >= 2 * len(sell_signals):
                resolution_rules.append(f"BUY majorité claire ({len(buy_signals)} vs {len(sell_signals)})")
                winning_side = "BUY"
                should_process = True
            elif len(sell_signals) >= 2 * len(buy_signals):
                resolution_rules.append(f"SELL majorité claire ({len(sell_signals)} vs {len(buy_signals)})")
                winning_side = "SELL"
                should_process = True
            
            # Règle 4: Confidence moyenne très différente (>25% d'écart)
            else:
                avg_buy_conf = sum(s.get('confidence', 0.5) for s in buy_signals) / len(buy_signals) if buy_signals else 0
                avg_sell_conf = sum(s.get('confidence', 0.5) for s in sell_signals) / len(sell_signals) if sell_signals else 0
                
                conf_diff = abs(avg_buy_conf - avg_sell_conf)
                if conf_diff > 0.25:
                    if avg_buy_conf > avg_sell_conf:
                        resolution_rules.append(f"BUY confidence supérieure ({avg_buy_conf:.2f} vs {avg_sell_conf:.2f})")
                        winning_side = "BUY"
                        should_process = True
                    else:
                        resolution_rules.append(f"SELL confidence supérieure ({avg_sell_conf:.2f} vs {avg_buy_conf:.2f})")
                        winning_side = "SELL"
                        should_process = True
                else:
                    resolution_rules.append("Conflit non résolvable - scores équilibrés")
        
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
            'buy_high_tf_count': buy_high_tf,
            'sell_high_tf_count': sell_high_tf,
            'conflict_type': 'balanced' if not should_process else 'resolvable'
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