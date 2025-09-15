"""
Buffer intelligent pour les signaux individuels.
Regroupe les signaux par symbole et timeframe pendant une fenêtre temporelle.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from signal_aggregator.src.strategy_classification import get_strategy_family

logger = logging.getLogger(__name__)


class IntelligentSignalBuffer:
    """
    Buffer qui regroupe intelligemment les signaux individuels en batches contextuels.
    
    Contrairement à l'ancien système qui attendait tous les signaux d'un cycle,
    ce buffer regroupe dynamiquement les signaux par contexte ET gère la synchronisation multi-timeframes.
    """
    
    def __init__(self, 
                 buffer_timeout: float = 5.0,      # Timeout en secondes
                 max_buffer_size: int = 1000,        # Taille max du buffer
                 min_batch_size: int = 1,          # Taille min pour traiter un batch
                 sync_window: float = 3.0,         # Fenêtre de sync multi-TF en secondes
                 enable_mtf_sync: bool = True,     # Activer la sync multi-timeframes
                 wave_timeout: float = 10.0):      # Timeout pour fin de vague en secondes
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
        
        # NOUVEAU: Buffer par SYMBOLE SEULEMENT (pas par timeframe/direction)
        # On regroupe TOUS les signaux d'un symbole pour choisir UN gagnant final
        self.symbol_wave_buffer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Timestamps pour détecter la fin de vague (dernier signal reçu par symbole)
        self.last_signal_time: Dict[str, datetime] = {}
        self.first_signal_time: Dict[str, datetime] = {}
        
        # Timeout pour détecter fin de vague (configurable, défaut 10 secondes)
        self.wave_timeout = wave_timeout
        
        # ANCIEN: Conservé pour compatibilité mais plus utilisé
        self.signal_buffer: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        self.mtf_buffer: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        self.first_mtf_signal_time: Dict[tuple, datetime] = {}
        
        # Lock pour thread safety
        self.buffer_lock = asyncio.Lock()
        
        # Callback pour traiter les batches
        self.batch_processor = None
        
        # Task pour gérer les timeouts
        self.timeout_task = None
        
        # Timeframes par ordre de priorité décisionnelle
        # ARCHITECTURE: 3m/5m = decision makers, 15m = context validator, 1m = timing tool
        self.timeframe_priority = {
            '1d': 1000, 
            '1h': 200, 
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
        
        # Statistiques (inclut nouvelles métriques de vague)
        self.stats = {
            'signals_buffered': 0,
            'batches_processed': 0,
            'timeout_triggers': 0,
            'size_triggers': 0,
            'wave_completed': 0,
            'wave_conflicts_resolved': 0,
            'wave_no_conflicts': 0,
            'waves_discarded_tie': 0,
            'mtf_sync_triggers': 0,  # Conservé pour compatibilité
            'mtf_batches_processed': 0,  # Conservé pour compatibilité
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
        
    async def _wave_timeout_monitor(self) -> None:
        """NOUVEAU: Monitor qui détecte les fins de vague par symbole."""
        while True:
            try:
                await asyncio.sleep(1.0)  # Vérifier chaque seconde
                
                async with self.buffer_lock:
                    now = datetime.utcnow()
                    symbols_to_process = []
                    
                    for symbol, last_time in self.last_signal_time.items():
                        if symbol in self.symbol_wave_buffer and self.symbol_wave_buffer[symbol]:
                            elapsed = (now - last_time).total_seconds()
                            wave_size = len(self.symbol_wave_buffer[symbol])
                            
                            # Fin de vague détectée si timeout atteint ET au moins 1 signal
                            if elapsed >= self.wave_timeout and wave_size >= 1:
                                symbols_to_process.append(symbol)
                    
                    # Traiter les vagues complètes
                    for symbol in symbols_to_process:
                        wave_size = len(self.symbol_wave_buffer[symbol])
                        elapsed = (now - self.last_signal_time[symbol]).total_seconds()
                        logger.info(f"🌊 FIN DE VAGUE détectée {symbol}: {wave_size} signaux, {elapsed:.1f}s timeout")
                        await self._process_symbol_wave(symbol, trigger="wave_timeout")
                        self.stats['timeout_triggers'] += 1
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur dans wave_timeout_monitor: {e}")
                
    async def _process_symbol_wave(self, symbol: str, trigger: str = "manual") -> None:
        """
        NOUVEAU: Traite une vague complète de signaux pour un symbole.
        Résout les conflits BUY vs SELL et choisit UN signal gagnant final.
        
        Args:
            symbol: Symbole à traiter
            trigger: Raison du déclenchement
        """
        if symbol not in self.symbol_wave_buffer or not self.symbol_wave_buffer[symbol]:
            return
            
        # Extraire tous les signaux de la vague
        wave_signals = self.symbol_wave_buffer[symbol].copy()
        
        # Nettoyer les buffers
        del self.symbol_wave_buffer[symbol]
        if symbol in self.first_signal_time:
            del self.first_signal_time[symbol]
        if symbol in self.last_signal_time:
            del self.last_signal_time[symbol]
            
        wave_size = len(wave_signals)
        logger.info(f"🏆 TRAITEMENT VAGUE {symbol}: {wave_size} signaux (trigger: {trigger})")
        
        # Analyser la composition de la vague
        buy_signals = [s for s in wave_signals if s.get('side') == 'BUY']
        sell_signals = [s for s in wave_signals if s.get('side') == 'SELL']
        
        buy_count = len(buy_signals)
        sell_count = len(sell_signals)
        
        logger.info(f"🎢 Vague {symbol}: {buy_count} BUY vs {sell_count} SELL")
        
        if buy_count == 0 and sell_count == 0:
            logger.warning(f"⚠️ Vague {symbol} vide, abandon")
            return
            
        # Choisir les signaux gagnants (maintenant une liste)
        winning_signals = await self._choose_winning_signal(symbol, buy_signals, sell_signals)
        
        if winning_signals:
            # Envoyer les signaux gagnants au batch processor
            if self.batch_processor:
                try:
                    # Créer une clé spéciale pour les signaux de vague
                    wave_context_key = (symbol, "wave_winner")
                    await self.batch_processor(winning_signals, wave_context_key)
                    self.stats['wave_completed'] += 1
                    
                    # Pas de log ici car on ne sait pas encore si la validation va réussir
                except Exception as e:
                    logger.error(f"❌ Erreur envoi signaux gagnants {symbol}: {e}")
        else:
            logger.info(f"🚫 Vague {symbol}: aucun gagnant, signal ignoré")
            
        self.stats['batches_processed'] += 1
        
    async def _choose_winning_signal(self, symbol: str, buy_signals: List[Dict[str, Any]], 
                                    sell_signals: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Choisit les signaux gagnants entre BUY et SELL selon les critères :
        - 40% nombre de stratégies
        - 40% score de consensus moyen
        - 20% timeframe le plus élevé
        
        Args:
            symbol: Symbole analysé
            buy_signals: Liste des signaux BUY
            sell_signals: Liste des signaux SELL
            
        Returns:
            Liste des signaux gagnants (même direction) ou None si égalité parfaite
        """
        buy_count = len(buy_signals)
        sell_count = len(sell_signals)
        
        wave_total = buy_count + sell_count  # PATCH E: taille totale de la vague
        
        # Si une seule direction, pas de conflit
        if buy_count > 0 and sell_count == 0:
            logger.info(f"✅ {symbol}: BUY gagne par défaut ({buy_count} signaux)")
            self.stats['wave_no_conflicts'] += 1
            # NOUVEAU: Retourner les signaux BUY originaux directement
            return self._prepare_winning_signals(buy_signals, 1.0, 0.0, wave_total)
            
        if sell_count > 0 and buy_count == 0:
            logger.info(f"✅ {symbol}: SELL gagne par défaut ({sell_count} signaux)")
            self.stats['wave_no_conflicts'] += 1
            # NOUVEAU: Retourner les signaux SELL originaux directement
            return self._prepare_winning_signals(sell_signals, 1.0, 0.0, wave_total)
            
        # Conflit détecté : appliquer les critères de choix
        logger.warning(f"🤼 CONFLIT {symbol}: {buy_count} BUY vs {sell_count} SELL")
        self.stats['wave_conflicts_resolved'] += 1
        
        buy_score = self._calculate_signal_strength(buy_signals)
        sell_score = self._calculate_signal_strength(sell_signals)
        
        logger.info(f"📊 Scores {symbol}: BUY={buy_score:.3f} vs SELL={sell_score:.3f}")
        
        # Seuil adaptatif selon le nombre de signaux
        min_diff = self._get_min_diff_threshold(buy_count, sell_count)
        
        if abs(buy_score - sell_score) < min_diff:
            logger.warning(f"⚖️ {symbol}: Égalité quasi parfaite ({buy_score:.3f} vs {sell_score:.3f}), signal ignoré")
            self.stats['waves_discarded_tie'] += 1
            return None
            
        # Déterminer le gagnant et retourner les signaux originaux
        if buy_score > sell_score:
            logger.info(f"🟢 {symbol}: BUY gagne ({buy_score:.3f} vs {sell_score:.3f})")
            # NOUVEAU: Retourner les signaux BUY originaux pour validation de consensus
            return self._prepare_winning_signals(buy_signals, buy_score, sell_score, wave_total)
        else:
            logger.info(f"🔴 {symbol}: SELL gagne ({sell_score:.3f} vs {buy_score:.3f})")
            # NOUVEAU: Retourner les signaux SELL originaux pour validation de consensus
            return self._prepare_winning_signals(sell_signals, sell_score, buy_score, wave_total)
    
    def _get_min_diff_threshold(self, buy_count: int, sell_count: int) -> float:
        """
        Calcule le seuil de différence minimum adaptatif selon le nombre de signaux.
        
        Plus on a de signaux, plus on peut être confiant dans une petite différence.
        Moins on a de signaux, plus on exige une différence importante.
        
        Args:
            buy_count: Nombre de signaux BUY
            sell_count: Nombre de signaux SELL
            
        Returns:
            Seuil de différence minimum (entre 0.03 et 0.08)
        """
        total = buy_count + sell_count
        
        if total >= 20:  # Beaucoup de signaux = forte conviction statistique
            return 0.03   # 3% suffisant avec beaucoup de données
        elif total >= 10:  # Nombre moyen de signaux
            return 0.05   # 5% standard
        else:  # Peu de signaux = besoin de plus de certitude
            return 0.05   # PATCH: 8%→5% pour récupérer plus de vagues
            
    def _calculate_signal_strength(self, signals: List[Dict[str, Any]]) -> float:
        """
        Calcule le score de force d'un groupe de signaux selon les critères pondérés.
        
        PATCH C: Harmonisé 30/30/25/15 + plancher de qualité
        
        Args:
            signals: Liste des signaux du même côté
            
        Returns:
            Score de force (0-1)
        """
        if not signals:
            return 0.0
            
        # Critère 1: Nombre de stratégies (30% du score)
        strategies_count = len(signals)
        # Normaliser sur base de 10 stratégies max (valeur réaliste)
        strategies_score = min(1.0, strategies_count / 10.0)
        
        # Critère 2: Score de consensus moyen (30% du score)
        confidences = [s.get('confidence', 0.5) for s in signals]
        avg_confidence = sum(confidences) / len(confidences)
        
        # PATCH C: Plancher de qualité - pénaliser les signaux tièdes
        quality_floor = 0.55
        if avg_confidence < quality_floor:
            # Atténue le score global si qualité médiocre
            penalty = (quality_floor - avg_confidence) * 0.4
        else:
            penalty = 0.0
        
        # Critère 3: Diversité des familles de stratégies (25% du score)
        unique_families = set()
        for signal in signals:
            strategy = signal.get('strategy', 'Unknown')
            family = get_strategy_family(strategy)
            if family != 'unknown':
                unique_families.add(family)
        
        # Normaliser sur 5 familles principales
        family_diversity_score = len(unique_families) / 5.0
        
        # Critère 4: Timeframe le plus élevé (15% du score)
        timeframes = [s.get('timeframe', '5m') for s in signals]
        max_tf_priority = max([self.timeframe_priority.get(tf, 50) for tf in timeframes])
        # Normaliser sur base de 1000 (1d)
        timeframe_score = min(1.0, max_tf_priority / 1000.0)
        
        # Score final pondéré (30/30/25/15) avec pénalité qualité
        final_score = (
            strategies_score * 0.30 +
            avg_confidence   * 0.30 +
            family_diversity_score * 0.25 +
            timeframe_score  * 0.15
        ) - penalty
                      
        return max(0.0, final_score)  # Ne pas aller en négatif
        
    def _prepare_winning_signals(self, winning_signals: List[Dict[str, Any]], 
                                winning_score: float, losing_score: float, wave_total: int = None) -> List[Dict[str, Any]]:
        """
        NOUVEAU: Prépare les signaux gagnants en ajoutant les métadonnées de résolution de conflit.
        Contrairement à _create_consensus_signal, cette méthode préserve les signaux originaux
        pour que le consensus adaptatif puisse analyser les vraies familles de stratégies.
        
        Args:
            winning_signals: Liste des signaux gagnants (même direction)
            winning_score: Score du côté gagnant
            losing_score: Score du côté perdant
            
        Returns:
            Liste des signaux gagnants enrichis avec métadonnées de vague
        """
        if not winning_signals:
            return []
            
        # Enrichir chaque signal gagnant avec les métadonnées de vague
        enriched_signals = []
        
        for signal in winning_signals:
            # Copier le signal original
            enriched_signal = signal.copy()
            
            # Ajouter les métadonnées de résolution de vague
            if 'metadata' not in enriched_signal:
                enriched_signal['metadata'] = {}
                
            # PATCH F: Propager la volatilité vers adaptive_consensus
            vr = signal.get('volatility_regime') or (signal.get('metadata') or {}).get('volatility_regime')
            if vr:
                enriched_signal['metadata']['volatility_regime'] = vr
            
            # Enrichir avec les informations de vague
            enriched_signal['metadata'].update({
                'wave_resolution': {
                    'is_wave_winner': True,
                    'winning_score': winning_score,
                    'losing_score': losing_score,
                    'conflict_resolved': losing_score > 0,  # True si il y avait un conflit
                    'total_wave_signals': wave_total or len(winning_signals),  # PATCH E: vraie taille vague
                    'resolution_timestamp': datetime.utcnow().isoformat()
                }
            })
            
            enriched_signals.append(enriched_signal)
            
        logger.info(f"📋 Préparé {len(enriched_signals)} signaux gagnants avec métadonnées de vague")
        return enriched_signals
        
    def _create_consensus_signal(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Crée un signal de consensus à partir d'un groupe de signaux de même direction.
        
        Args:
            signals: Liste des signaux de même direction
            
        Returns:
            Signal de consensus
        """
        if not signals:
            return {}
            
        # Prendre les infos du signal le plus fort
        best_signal = max(signals, key=lambda s: s.get('confidence', 0.0))
        
        symbol = best_signal.get('symbol', 'UNKNOWN')
        side = best_signal.get('side', 'UNKNOWN')
        
        # Calculer les métadonnees du consensus
        strategies = [s.get('strategy', 'Unknown') for s in signals]
        timeframes = list(set(s.get('timeframe', '5m') for s in signals))
        confidences = [s.get('confidence', 0.5) for s in signals]
        
        avg_confidence = sum(confidences) / len(confidences)
        max_confidence = max(confidences)
        
        # Créer le signal de consensus
        consensus_signal = {
            'strategy': 'WAVE_CONSENSUS',
            'symbol': symbol,
            'side': side,
            'timestamp': datetime.utcnow().isoformat(),
            'confidence': min(1.0, (avg_confidence + max_confidence) / 2),  # Mix moyenne et max
            'timeframe': max(timeframes, key=lambda tf: self.timeframe_priority.get(tf, 0)),  # TF le plus élevé
            'metadata': {
                'type': 'WAVE_CONSENSUS',
                'wave_size': len(signals),
                'strategies_count': len(strategies),
                'strategies': strategies,
                'timeframes': sorted(timeframes),
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'strength_score': self._calculate_signal_strength(signals),
                'original_signals': signals
            }
        }
        
        return consensus_signal
        
    def _has_simultaneous_opposite_processing(self, symbol: str, side: str) -> bool:
        """
        Vérifie s'il y a un traitement simultané de la direction opposée.
        Plus strict - utilisé pour les déclenchements normaux.
        """
        opposite_side = 'SELL' if side == 'BUY' else 'BUY'
        opposite_key = (symbol, opposite_side)
        
        # Vérifier si l'opposé a des signaux en buffer ET dans une fenêtre critique
        if opposite_key in self.mtf_buffer and self.mtf_buffer[opposite_key]:
            if opposite_key in self.first_mtf_signal_time:
                opposite_first_time = self.first_mtf_signal_time[opposite_key]
                our_first_time = self.first_mtf_signal_time.get((symbol, side))
                
                if our_first_time:
                    time_diff = abs((our_first_time - opposite_first_time).total_seconds())
                    # Considérer comme simultané si dans les 3 secondes
                    return time_diff < 3.0
        return False
        
    def _has_critical_opposite_processing(self, symbol: str, side: str) -> bool:
        """
        Vérifie s'il y a un traitement critique de la direction opposée.
        Moins strict - utilisé pour les timeouts où on peut être plus permissif.
        """
        opposite_side = 'SELL' if side == 'BUY' else 'BUY'
        opposite_key = (symbol, opposite_side)
        
        # Seulement bloquer si l'opposé a beaucoup de signaux OU est très récent
        if opposite_key in self.mtf_buffer and self.mtf_buffer[opposite_key]:
            opposite_buffer_size = len(self.mtf_buffer[opposite_key])
            
            # Bloquer si l'opposé a plus de 3 signaux (forte conviction)
            if opposite_buffer_size > 3:
                return True
                
            # Ou si très récent (moins d'1 seconde)
            if opposite_key in self.first_mtf_signal_time:
                opposite_first_time = self.first_mtf_signal_time[opposite_key]
                time_since_opposite = (datetime.utcnow() - opposite_first_time).total_seconds()
                return time_since_opposite < 1.0
        
        return False
        
    async def add_signal(self, signal: Dict[str, Any]) -> None:
        """
        NOUVEAU: Ajoute un signal au buffer de vague par symbole.
        Détecte automatiquement la fin de vague avec timeout intelligent.
        
        Args:
            signal: Signal à ajouter
        """
        async with self.buffer_lock:
            symbol = signal.get('symbol', 'UNKNOWN')
            side = signal.get('side')
            now = datetime.utcnow()
            
            # PATCH D: Exclure side=None dès l'entrée (veto/filtres)
            if side is None:
                logger.debug(f"🚫 Signal veto/filtre ignoré pour {symbol} (side=None)")
                return
            
            # Ajouter au buffer de vague pour ce symbole
            self.symbol_wave_buffer[symbol].append(signal)
            self.stats['signals_buffered'] += 1
            
            # Mettre à jour les timestamps
            if symbol not in self.first_signal_time:
                self.first_signal_time[symbol] = now
                logger.info(f"🌊 Début de vague détecté pour {symbol}")
            
            # TOUJOURS mettre à jour le dernier signal (reset du timeout)
            self.last_signal_time[symbol] = now
            
            # Démarrer le monitor de timeout si pas actif
            if not self.timeout_task or self.timeout_task.done():
                self.timeout_task = asyncio.create_task(self._wave_timeout_monitor())
                
            side = signal.get('side', 'UNKNOWN')
            timeframe = signal.get('timeframe', 'UNKNOWN')
            wave_size = len(self.symbol_wave_buffer[symbol])
            
            logger.debug(f"➕ Signal ajouté {symbol} {timeframe} {side} (vague: {wave_size} signaux)")
            
            # Vérifier si taille max atteinte (sécurité)
            if wave_size >= self.max_buffer_size:
                logger.warning(f"🚨 Vague {symbol} trop grande ({wave_size}), traitement forcé")
                await self._process_symbol_wave(symbol, trigger="size_limit")
                self.stats['size_triggers'] += 1
            
    async def _check_immediate_processing(self, context_key: tuple) -> None:
        """Vérifie s'il faut traiter immédiatement un contexte."""
        buffer_size = len(self.signal_buffer[context_key])
        
        # Déclencher si le buffer est trop grand
        if buffer_size >= self.max_buffer_size:
            logger.info(f"Déclenchement par taille: {buffer_size} signaux pour {context_key}")
            await self._process_context(context_key, trigger="size")
            self.stats['size_triggers'] += 1
            
    async def _check_mtf_processing(self, mtf_key: tuple) -> None:
        """Vérifie s'il faut traiter immédiatement le buffer multi-timeframes."""
        if mtf_key not in self.mtf_buffer:
            return
            
        mtf_signals = self.mtf_buffer[mtf_key]
        buffer_size = len(mtf_signals)
        symbol, side = mtf_key
        
        # 🚨 NOUVEAU: Vérifier qu'il n'y a pas de traitement simultané de l'opposé
        opposite_side = 'SELL' if side == 'BUY' else 'BUY'
        opposite_key = (symbol, opposite_side)
        if opposite_key in self.first_mtf_signal_time:
            # Vérifier si l'opposé est en attente dans une fenêtre critique
            opposite_first_time = self.first_mtf_signal_time[opposite_key]
            our_first_time = self.first_mtf_signal_time.get(mtf_key)
            
            if our_first_time and opposite_first_time:
                time_diff = abs((our_first_time - opposite_first_time).total_seconds())
                # Si les deux directions ont commencé dans les 2 secondes, traiter en priorité
                if time_diff < 2.0:
                    opposite_buffer_size = len(self.mtf_buffer.get(opposite_key, []))
                    if opposite_buffer_size > 0:
                        logger.warning(f"⚠️ CONFLIT MTF détecté {symbol}: {side}({buffer_size}) vs {opposite_side}({opposite_buffer_size})")
                        # Prioriser le plus fort ou le plus ancien
                        if buffer_size < opposite_buffer_size or (buffer_size == opposite_buffer_size and our_first_time > opposite_first_time):
                            logger.info(f"🗑️ ABANDON MTF {symbol} {side}: l'opposé est plus fort/ancien")
                            return  # Abandonner le traitement de ce côté
        
        # Déclencher immédiatement si trop de signaux
        if buffer_size >= self.max_buffer_size:
            logger.info(f"Déclenchement MTF par taille: {buffer_size} signaux {side} pour {symbol}")
            await self._process_mtf_symbol(mtf_key, trigger="size")
            self.stats['size_triggers'] += 1
            return
            
        # Analyser la diversité des timeframes (pas besoin de directions, déjà séparé)
        timeframes_present = set(s.get('timeframe', '5m') for s in mtf_signals)
        # Toutes les directions sont identiques maintenant car séparées par clé
        
        # Conditions de déclenchement intelligent :
        
        # 1. Si on a des signaux de TOUS les timeframes principaux (3m+5m+15m) dans la même direction
        expected_timeframes = {'3m', '5m', '15m'}  # Timeframes principaux de l'analyzer
        has_all_main_timeframes = expected_timeframes.issubset(timeframes_present)
        
        if has_all_main_timeframes:
            logger.info(f"Déclenchement MTF par timeframes complets: {len(timeframes_present)} TFs "
                       f"({list(timeframes_present)}), direction {side} pour {symbol}")
            await self._process_mtf_symbol(mtf_key, trigger="complete_timeframes")
            self.stats['mtf_sync_triggers'] += 1
            return
            
        # 2. Fallback: Si on a 3+ timeframes différents ET aucun conflit détecté
        elif len(timeframes_present) >= 3:
            # Vérifier une dernière fois qu'il n'y a pas de traitement simultané de l'opposé
            if not self._has_simultaneous_opposite_processing(symbol, side):
                logger.info(f"Déclenchement MTF par diversité TF: {len(timeframes_present)} TFs, "
                           f"direction {side} pour {symbol}")
                await self._process_mtf_symbol(mtf_key, trigger="timeframe_diversity") 
                self.stats['mtf_sync_triggers'] += 1
                return
            else:
                logger.warning(f"🚫 Déclenchement MTF bloqué {symbol} {side}: traitement opposé simultané")
            
        # 3. Plus de conflits possibles car BUY/SELL sont séparés!
        # Les conflits sont maintenant impossibles dans le même buffer
            
        # 4. Si on a un signal de timeframe élevé (1h+) avec confirmation courte ET aucun conflit
        high_tf_signals = [s for s in mtf_signals 
                          if self.timeframe_priority.get(s.get('timeframe', '5m'), 0) >= 100]
        
        if high_tf_signals and buffer_size >= 2 and not self._has_simultaneous_opposite_processing(symbol, side):
            logger.info(f"Déclenchement MTF par TF élevé: {len(high_tf_signals)} signaux 1h+ "
                       f"avec {buffer_size-len(high_tf_signals)} confirmations {side} pour {symbol}")
            await self._process_mtf_symbol(mtf_key, trigger="high_timeframe")
            self.stats['mtf_sync_triggers'] += 1
            return
            
        # 5. Si fenêtre de sync expirée et assez de signaux ET aucun conflit critique
        first_time = self.first_mtf_signal_time.get(mtf_key)
        if first_time:
            elapsed = (datetime.utcnow() - first_time).total_seconds()
            if elapsed >= self.sync_window and buffer_size >= self.min_batch_size:
                # Vérifier les conflits avant le timeout (mais être plus permissif)
                can_process_timeout = not self._has_critical_opposite_processing(symbol, side)
                if can_process_timeout:
                    logger.info(f"Déclenchement MTF par timeout sync: {elapsed:.1f}s, "
                               f"{buffer_size} signaux {side} pour {symbol}")
                    await self._process_mtf_symbol(mtf_key, trigger="sync_timeout")
                    self.stats['mtf_sync_triggers'] += 1
                else:
                    logger.warning(f"⚠️ Timeout MTF retardé {symbol} {side}: conflit critique avec opposé")
            
    async def _timeout_monitor(self) -> None:
        """ANCIEN: Conservé pour compatibilité mais redirige vers le nouveau système."""
        return await self._wave_timeout_monitor()
                
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
                
    async def _process_mtf_symbol(self, mtf_key: tuple, trigger: str = "manual") -> None:
        """
        Traite tous les signaux multi-timeframes d'un symbole/direction.
        
        Args:
            mtf_key: Tuple (symbol, side) à traiter
            trigger: Raison du déclenchement
        """
        if mtf_key not in self.mtf_buffer or not self.mtf_buffer[mtf_key]:
            return
            
        symbol, side = mtf_key
        
        # Extraire les signaux
        signals = self.mtf_buffer[mtf_key].copy()
        
        # Nettoyer le buffer MTF
        del self.mtf_buffer[mtf_key]
        if mtf_key in self.first_mtf_signal_time:
            del self.first_mtf_signal_time[mtf_key]
            
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
        
        logger.info(f"Traitement MTF {symbol} {side}: {len(signals)} signaux "
                   f"(TFs: {list(set(timeframes))}, Trigger: {trigger})")
        
        # Plus de conflits possibles car BUY/SELL sont déjà séparés!
        final_signals = signals
        original_signal_count = len(signals)
        
        # Traiter le batch MTF si on a un processor
        if self.batch_processor and final_signals:
            try:
                # Créer une clé spéciale pour le batch MTF incluant la direction
                mtf_context_key = (symbol, f"multi_timeframe_{side}")
                await self.batch_processor(final_signals, mtf_context_key)
                self.stats['mtf_batches_processed'] += 1
            except Exception as e:
                logger.error(f"Erreur traitement batch MTF {symbol} {side}: {e}")
                
    def _analyze_mtf_conflicts(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        OBSOLÈTE: Plus de conflits possibles car BUY/SELL sont séparés dès le buffer.
        Méthode conservée pour compatibilité mais ne devrait plus être appelée.
        """
        logger.warning("_analyze_mtf_conflicts appelée mais obsolète car BUY/SELL séparés!")
        return {
            'should_process': True,
            'winning_side': signals[0].get('side') if signals else None,
            'resolution': 'No conflict - signals separated by direction'
        }
                
    async def force_flush_all(self) -> None:
        """NOUVEAU: Force le traitement de toutes les vagues en buffer."""
        async with self.buffer_lock:
            symbols_to_process = list(self.symbol_wave_buffer.keys())
            
        for symbol in symbols_to_process:
            await self._process_symbol_wave(symbol, trigger="flush")
            
        logger.info(f"🌊 Flush forcé: {len(symbols_to_process)} vagues traitées")
        
    async def get_buffer_status(self) -> Dict[str, Any]:
        """NOUVEAU: Retourne le statut des vagues en cours."""
        async with self.buffer_lock:
            total_buffered = sum(len(signals) for signals in self.symbol_wave_buffer.values())
            waves_count = len(self.symbol_wave_buffer)
            now = datetime.utcnow()
            
            wave_details = {}
            for symbol, signals in self.symbol_wave_buffer.items():
                first_time = self.first_signal_time.get(symbol)
                last_time = self.last_signal_time.get(symbol)
                
                elapsed_since_first = (now - first_time).total_seconds() if first_time else 0
                elapsed_since_last = (now - last_time).total_seconds() if last_time else 0
                
                # Analyser la composition
                buy_count = sum(1 for s in signals if s.get('side') == 'BUY')
                sell_count = sum(1 for s in signals if s.get('side') == 'SELL')
                timeframes = list(set(s.get('timeframe', '5m') for s in signals))
                
                wave_details[symbol] = {
                    'total_signals': len(signals),
                    'buy_signals': buy_count,
                    'sell_signals': sell_count,
                    'has_conflict': buy_count > 0 and sell_count > 0,
                    'timeframes': sorted(timeframes),
                    'elapsed_since_first': elapsed_since_first,
                    'elapsed_since_last': elapsed_since_last,
                    'will_timeout_soon': elapsed_since_last >= (self.wave_timeout - 2.0)
                }
                
        return {
            'total_buffered_signals': total_buffered,
            'active_waves': waves_count,
            'wave_timeout_seconds': self.wave_timeout,
            'wave_details': wave_details,
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
        logger.info("🧹 Buffer de vagues nettoyé")