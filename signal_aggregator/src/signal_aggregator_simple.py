"""
Service principal d'agrégation des signaux - VERSION ULTRA-SIMPLIFIÉE.

Ce service orchestre :
1. Réception des signaux individuels depuis Redis  
2. Buffer intelligent avec consensus adaptatif SEULEMENT
3. Filtres critiques minimalistes (4 filtres max)
4. Envoi des signaux validés au coordinator

FINI: 23+ validators complexes, hiérarchies, vetos, etc.
NOUVEAU: Consensus adaptatif + filtres de sécurité seulement
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import redis.asyncio as redis
import json
from datetime import datetime, timedelta

from signal_buffer import IntelligentSignalBuffer
from signal_processor_simple import SimpleSignalProcessor

logger = logging.getLogger(__name__)


class SimpleSignalAggregatorService:
    """
    Service principal d'agrégation ultra-simplifié.
    Logic: Buffer + Consensus adaptatif + Filtres critiques + Done !
    """
    
    def __init__(self, context_manager, database_manager=None):
        """
        Initialise le service d'agrégation simplifié.
        
        Args:
            context_manager: Gestionnaire de contexte de marché  
            database_manager: Gestionnaire de base de données (optionnel)
        """
        self.context_manager = context_manager
        self.database_manager = database_manager
        
        # Composant principal simplifié
        self.signal_processor = SimpleSignalProcessor(context_manager, database_manager)
        
        # Buffer intelligent - OPTIMISÉ POUR RAPIDITÉ
        self.signal_buffer = IntelligentSignalBuffer(
            buffer_timeout=6.0,        # RÉDUIT: 6s pour crypto rapide
            max_buffer_size=30,        # 30 signaux max (réduit)
            min_batch_size=1,          # Minimum 1 signal
            sync_window=3.0,           # RÉDUIT: 3s pour sync multi-TF crypto
            enable_mtf_sync=True       
        )
        
        # Configuration Redis
        self.redis_client = None
        self.redis_url = 'redis://redis:6379'
        
        # Canaux Redis
        self.input_channel = 'analyzer:signals'
        self.output_channel = 'roottrading:signals:filtered'
        
        # Protection contradictions - SIMPLIFIÉE
        self.recent_signals: Dict[str, Dict[str, Any]] = {}
        self.contradiction_window = 90.0  # 90s de protection
        
        # Statistiques ultra-simples
        self.stats = {
            'signals_received': 0,
            'signals_validated': 0,
            'signals_sent': 0,
            'consensus_rejected': 0,
            'critical_filter_rejected': 0,
            'errors': 0
        }
        
    async def start(self):
        """Démarre le service d'agrégation simplifié."""
        logger.info("🚀 Démarrage service agrégation SIMPLIFIÉ...")
        
        # Connexion Redis
        await self._connect_redis()
        
        # Configuration du buffer
        self.signal_buffer.set_batch_processor(self._process_signal_batch_simple)
        
        # Démarrage des tâches
        tasks = [
            asyncio.create_task(self._listen_for_signals()),
            asyncio.create_task(self._health_monitor())
        ]
        
        logger.info("✅ Service d'agrégation simplifié démarré")
        await asyncio.gather(*tasks)
        
    async def _connect_redis(self):
        """Établit la connexion Redis."""
        try:
            self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("✅ Connexion Redis établie")
        except Exception as e:
            logger.error(f"❌ Erreur connexion Redis: {e}")
            raise
            
    async def _listen_for_signals(self):
        """Écoute les signaux depuis Redis."""
        logger.info(f"🎧 Écoute signaux depuis {self.input_channel}")
        
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe(self.input_channel)
        
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    await self._handle_signal_message(message['data'])
        except Exception as e:
            logger.error(f"❌ Erreur écoute signaux: {e}")
        finally:
            await pubsub.unsubscribe(self.input_channel)
            
    async def _handle_signal_message(self, message_data: str):
        """Traite un message de signal reçu."""
        try:
            self.stats['signals_received'] += 1
            
            # Traitement ultra-simple du signal
            processed_signal = await self.signal_processor.process_signal(message_data)
            if processed_signal:
                # Ajout au buffer pour consensus
                await self.signal_buffer.add_signal(processed_signal)
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement message: {e}")
            self.stats['errors'] += 1
            
    async def _process_signal_batch_simple(self, signals: list, context_key: tuple = None):
        """
        Traite un batch de signaux avec le système ultra-simplifié.
        REMPLACE TOUTE LA COMPLEXITÉ PAR: Groupe par symbol/side → Consensus → Filtres → Done
        
        Args:
            signals: Liste des signaux à traiter
            context_key: Clé de contexte (symbol, timeframe) ou (symbol, "multi_timeframe")
        """
        try:
            if context_key:
                symbol, timeframe = context_key
                logger.info(f"📦 Traitement batch {symbol} {timeframe}: {len(signals)} signaux")
            else:
                logger.info(f"📦 Traitement batch: {len(signals)} signaux")
            
            # Si MTF (multi_timeframe), les signaux sont déjà filtrés par direction
            if context_key and context_key[1] == "multi_timeframe":
                # Signaux MTF déjà résolus, traiter directement
                symbol = context_key[0]
                if signals:
                    side = signals[0]['side']  # Tous dans la même direction après résolution MTF
                    timeframe = signals[0].get('timeframe', '3m')
                    await self._validate_signal_group(signals)
            else:
                # Grouper par symbol + side (logique simple)
                signal_groups = {}
                for signal in signals:
                    key = f"{signal['symbol']}_{signal['side']}"
                    if key not in signal_groups:
                        signal_groups[key] = []
                    signal_groups[key].append(signal)
                    
                # Traiter chaque groupe
                for group_key, group_signals in signal_groups.items():
                    await self._validate_signal_group(group_signals)
                
        except Exception as e:
            logger.error(f"❌ Erreur traitement batch: {e}")
            self.stats['errors'] += 1
            
    async def _validate_signal_group(self, signals: list):
        """Valide un groupe de signaux (même symbol/side)."""
        if not signals:
            return
            
        # Extraction des infos communes
        first_signal = signals[0]
        symbol = first_signal['symbol']
        timeframe = first_signal.get('timeframe', '3m')
        side = first_signal['side']
        
        logger.info(f"🔍 Validation groupe {symbol} {side}: {len(signals)} signaux")
        
        # Validation avec système simplifié
        validated_signal = await self.signal_processor.validate_signal_group(
            signals, symbol, timeframe, side
        )
        
        if validated_signal:
            # Protection contre contradictions
            if self._check_recent_contradiction(symbol, side):
                logger.info(f"🚫 Signal {symbol} {side} bloqué: contradiction récente")
                return
                
            # Envoi du signal validé
            await self._send_validated_signal(validated_signal)
            self._track_recent_signal(symbol, side)
            
            self.stats['signals_validated'] += 1
            self.stats['signals_sent'] += 1
        else:
            logger.info(f"❌ Groupe {symbol} {side} rejeté")
            
    def _check_recent_contradiction(self, symbol: str, side: str) -> bool:
        """Vérifie les contradictions récentes pour éviter ping-pong."""
        if symbol not in self.recent_signals:
            return False
            
        recent = self.recent_signals[symbol]
        time_diff = (datetime.utcnow() - recent['timestamp']).total_seconds()
        
        # Si signal opposé récent dans la fenêtre de protection
        if recent['side'] != side and time_diff < self.contradiction_window:
            return True
            
        return False
        
    def _track_recent_signal(self, symbol: str, side: str):
        """Enregistre le signal récent pour éviter contradictions."""
        self.recent_signals[symbol] = {
            'side': side,
            'timestamp': datetime.utcnow()
        }
        
    async def _send_validated_signal(self, signal: Dict[str, Any]):
        """Envoie le signal validé vers le coordinator."""
        try:
            signal_json = json.dumps(signal, default=str)
            await self.redis_client.publish(self.output_channel, signal_json)
            
            strategies_count = signal.get('metadata', {}).get('strategies_count', 1)
            logger.info(f"📤 Signal envoyé: {signal['symbol']} {signal['side']} "
                       f"({strategies_count} stratégies)")
            
        except Exception as e:
            logger.error(f"❌ Erreur envoi signal: {e}")
            
    async def _health_monitor(self):
        """Monitore la santé du service."""
        while True:
            try:
                await asyncio.sleep(30)  # Check toutes les 30s
                
                # Log stats périodiques
                if self.stats['signals_received'] > 0:
                    success_rate = (self.stats['signals_validated'] / self.stats['signals_received']) * 100
                    logger.info(f"📊 Stats: {self.stats['signals_received']} reçus, "
                              f"{self.stats['signals_validated']} validés ({success_rate:.1f}%), "
                              f"{self.stats['errors']} erreurs")
                
                # Nettoyage des signaux anciens
                self._cleanup_old_signals()
                
            except Exception as e:
                logger.error(f"❌ Erreur health monitor: {e}")
                
    def _cleanup_old_signals(self):
        """Nettoie les anciens signaux de la mémoire."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.contradiction_window * 2)
        
        to_remove = []
        for symbol, signal_info in self.recent_signals.items():
            if signal_info['timestamp'] < cutoff_time:
                to_remove.append(symbol)
                
        for symbol in to_remove:
            del self.recent_signals[symbol]
            
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques complètes."""
        processor_stats = self.signal_processor.get_stats()
        
        return {
            'service_stats': self.stats,
            'processor_stats': processor_stats,
            'buffer_stats': self.signal_buffer.get_stats() if hasattr(self.signal_buffer, 'get_stats') else {},
            'active_symbols': len(self.recent_signals)
        }