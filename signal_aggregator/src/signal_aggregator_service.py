"""
Service principal d'agrégation des signaux - ARCHITECTURE PROPRE.

Ce service orchestre :
1. Réception des signaux individuels depuis Redis
2. Buffer intelligent avec synchronisation multi-timeframes
3. Consensus adaptatif par régime de marché
4. Validation hiérarchique avec Market Structure Validator (VETO)
5. Envoi des signaux validés au coordinator
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import redis.asyncio as redis
import json
from datetime import datetime, timedelta

from signal_buffer import IntelligentSignalBuffer
from signal_processor import SignalProcessor
from adaptive_consensus import AdaptiveConsensusAnalyzer

logger = logging.getLogger(__name__)


class SignalAggregatorService:
    """
    Service principal d'agrégation qui combine tous les nouveaux composants.
    """
    
    def __init__(self, validator_loader, context_manager, database_manager=None):
        """
        Initialise le service d'agrégation.
        
        Args:
            validator_loader: Chargeur de validators
            context_manager: Gestionnaire de contexte de marché  
            database_manager: Gestionnaire de base de données (optionnel)
        """
        self.validator_loader = validator_loader
        self.context_manager = context_manager
        self.database_manager = database_manager
        
        # Composants principaux
        self.signal_processor = SignalProcessor(validator_loader, context_manager, database_manager)
        self.adaptive_consensus = AdaptiveConsensusAnalyzer()
        
        # Buffer intelligent avec sync multi-timeframes
        self.signal_buffer = IntelligentSignalBuffer(
            buffer_timeout=15.0,       # 15 secondes max d'attente (assez pour collecter tous les signaux)
            max_buffer_size=50,        # 50 signaux max avant traitement forcé
            min_batch_size=1,          # Minimum 1 signal pour traiter
            sync_window=10.0,          # 10 secondes pour sync multi-TF (pour capturer tous les timeframes)
            enable_mtf_sync=True       # Activer la synchronisation multi-timeframes
        )
        
        # Configuration Redis
        self.redis_client = None
        self.redis_url = 'redis://redis:6379'
        
        # Canaux Redis
        self.input_channel = 'analyzer:signals'      # Signaux depuis l'analyzer
        self.output_channel = 'roottrading:signals:filtered'  # Signaux vers le coordinator (même canal que redis_handler)
        
        # Protection contre signaux contradictoires
        self.recent_signals: Dict[str, Dict[str, Any]] = {}  # {symbol: {'side': 'BUY', 'timestamp': datetime}}
        self.contradiction_window = 30.0  # 30 secondes de protection
        
        # Statistiques
        self.stats = {
            'signals_received': 0,
            'batches_processed': 0,
            'signals_sent': 0,
            'signals_blocked': 0,
            'consensus_rejected_insufficient': 0,  # Consensus rejetés car < 3 stratégies validées
            'errors': 0
        }
        
    async def start(self):
        """Démarre le service d'agrégation."""
        logger.info("Démarrage du service d'agrégation...")
        
        # Connexion Redis
        await self._connect_redis()
        
        # Configuration du buffer
        self.signal_buffer.set_batch_processor(self._process_signal_batch)
        
        # Démarrage des tâches
        tasks = [
            asyncio.create_task(self._listen_for_signals()),
            asyncio.create_task(self._health_monitor())
        ]
        
        logger.info("Service d'agrégation démarré")
        await asyncio.gather(*tasks)
        
    async def _connect_redis(self):
        """Établit la connexion Redis."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connexion Redis établie")
        except Exception as e:
            logger.error(f"Erreur connexion Redis: {e}")
            raise
            
    async def _listen_for_signals(self):
        """Écoute les signaux depuis Redis."""
        logger.info(f"Écoute des signaux sur le canal: {self.input_channel}")
        
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe(self.input_channel)
        
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    signal_data = message['data'].decode('utf-8')
                    await self._handle_incoming_signal(signal_data)
                    
        except Exception as e:
            logger.error(f"Erreur écoute Redis: {e}")
        finally:
            await pubsub.unsubscribe(self.input_channel)
            
    async def _handle_incoming_signal(self, signal_data: str):
        """
        Gère un signal entrant depuis l'analyzer.
        
        Args:
            signal_data: Données JSON du signal
        """
        try:
            signal = json.loads(signal_data)
            self.stats['signals_received'] += 1
            
            logger.debug(f"Signal reçu: {signal.get('strategy')} {signal.get('symbol')} "
                        f"{signal.get('timeframe')} {signal.get('side')}")
            
            # Ajouter au buffer intelligent (gère la sync multi-TF automatiquement)
            await self.signal_buffer.add_signal(signal)
            
        except Exception as e:
            logger.error(f"Erreur traitement signal entrant: {e}")
            self.stats['errors'] += 1
            
    async def _process_signal_batch(self, signals: list, context_key: tuple):
        """
        Traite un batch de signaux provenant du buffer intelligent.
        Cette méthode est appelée automatiquement par le buffer.
        
        Args:
            signals: Liste des signaux à traiter
            context_key: Clé du contexte (symbol, timeframe) ou (symbol, "multi_timeframe")
        """
        try:
            symbol, timeframe_or_mtf = context_key
            is_multi_timeframe = timeframe_or_mtf == "multi_timeframe"
            
            logger.info(f"Traitement batch: {len(signals)} signaux pour {symbol} "
                       f"({'Multi-TF' if is_multi_timeframe else timeframe_or_mtf})")
            
            # Obtenir le régime de marché pour le consensus adaptatif
            market_regime = await self._get_market_regime(symbol, signals)
            
            # Pour les signaux MTF post-conflit, ajuster la logique de consensus
            if is_multi_timeframe:
                # Récupérer le nombre original de stratégies depuis les métadonnées
                original_count = signals[0].get('metadata', {}).get('original_signal_count', len(signals)) if signals else 0
                
                # Si on a eu un conflit résolu (signaux filtrés), assouplir les critères
                if original_count > len(signals):
                    # On avait plus de stratégies au départ, le buffer a filtré après résolution de conflit
                    logger.info(f"Signaux MTF post-conflit: {original_count} stratégies originales → {len(signals)} après résolution")
                    
                    # Créer des signaux "virtuels" pour représenter le consensus original
                    # Cela permet au consensus adaptatif de voir qu'on avait assez de stratégies
                    enhanced_signals = signals.copy()
                    for sig in enhanced_signals:
                        sig['metadata'] = sig.get('metadata', {})
                        sig['metadata']['mtf_conflict_resolved'] = True
                        sig['metadata']['original_strategy_count'] = original_count
                    
                    # Analyser avec des critères assouplis pour MTF post-conflit
                    has_consensus, consensus_details = self.adaptive_consensus.analyze_adaptive_consensus_mtf(
                        enhanced_signals, market_regime, original_count
                    )
                else:
                    # Pas de conflit, analyse normale
                    has_consensus, consensus_details = self.adaptive_consensus.analyze_adaptive_consensus(
                        signals, market_regime
                    )
            else:
                # Analyse normale pour les signaux non-MTF
                has_consensus, consensus_details = self.adaptive_consensus.analyze_adaptive_consensus(
                    signals, market_regime
                )
            
            if not has_consensus:
                logger.info(f"Pas de consensus pour {symbol}: {consensus_details.get('reason', 'N/A')}")
                return
                
            logger.info(f"Consensus validé pour {symbol}: {consensus_details['total_strategies']} stratégies, "
                       f"adaptabilité: {consensus_details['avg_adaptability']:.2f}")
            
            # Valider chaque signal individuellement avec le système hiérarchique
            validated_signals = []
            
            for signal in signals:
                validated_signal = await self.signal_processor._validate_signal(signal)
                if validated_signal:
                    # Ajouter les métadonnées de consensus
                    validated_signal['metadata'].update({
                        'consensus_analysis': consensus_details,
                        'is_consensus': True,
                        'market_regime': market_regime
                    })
                    validated_signals.append(validated_signal)
                    
            if validated_signals:
                # VÉRIFICATION FINALE : Un vrai consensus doit avoir au minimum 3 stratégies validées
                if len(validated_signals) < 3:
                    self.stats['consensus_rejected_insufficient'] += 1
                    logger.warning(f"❌ Consensus rejeté pour {symbol}: seulement {len(validated_signals)} stratégie(s) "
                                 f"validée(s) sur {len(signals)} originales. Minimum requis: 3")
                    return
                
                # Créer un signal composite final
                composite_signal = self._create_composite_signal(signals, validated_signals, consensus_details)
                
                if composite_signal:
                    # Envoyer au coordinator
                    await self._send_to_coordinator(composite_signal)
                    
                    self.stats['batches_processed'] += 1
                    self.stats['signals_sent'] += 1
                    
                    logger.info(f"✅ Signal composite envoyé: {symbol} {composite_signal['side']} "
                               f"({len(validated_signals)} stratégies validées, confidence: {composite_signal['confidence']:.2f})")
            else:
                logger.info(f"Aucun signal validé dans le batch {symbol}")
                
        except Exception as e:
            logger.error(f"Erreur traitement batch: {e}")
            self.stats['errors'] += 1
            
    async def _get_market_regime(self, symbol: str, signals: list) -> str:
        """Obtient le régime de marché pour le consensus adaptatif."""
        try:
            # Utiliser le timeframe du premier signal pour récupérer le régime
            sample_signal = signals[0] if signals else None
            if sample_signal:
                timeframe = sample_signal.get('timeframe', '5m')
                context = self.context_manager.get_market_context(symbol, timeframe)
                return context.get('market_regime', 'UNKNOWN')
        except Exception as e:
            logger.warning(f"Impossible de récupérer le régime pour {symbol}: {e}")
            
        return 'UNKNOWN'
        
    def _create_composite_signal(self, original_signals: list, validated_signals: list, 
                                consensus_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Crée un signal composite représentant le consensus final."""
        if not validated_signals:
            return None
            
        # Utiliser le signal avec la meilleure confidence comme base
        best_signal = max(validated_signals, key=lambda s: s.get('confidence', 0))
        
        # Statistiques du consensus
        original_count = len(original_signals)
        validated_count = len(validated_signals)
        avg_confidence = sum(s.get('confidence', 0) for s in validated_signals) / validated_count
        
        # Créer le signal composite avec le nombre de stratégies VALIDÉES
        composite_signal = {
            **best_signal,
            'strategy': f"CONSENSUS_{validated_count}_STRATEGIES",
            'confidence': min(1.0, avg_confidence * 1.1),  # Bonus consensus +10%
            'metadata': {
                **best_signal.get('metadata', {}),
                'is_composite': True,
                'consensus_details': consensus_details,
                'original_strategies': original_count,  # Nombre initial avant validation
                'validated_strategies': validated_count,  # Nombre final utilisé pour le nom
                'validation_rate': validated_count / original_count,
                'original_strategies_list': [s.get('strategy') for s in original_signals],
                'validated_strategies_list': [s.get('strategy') for s in validated_signals]
            }
        }
        
        return composite_signal
        
    async def _send_to_coordinator(self, signal: Dict[str, Any]):
        """Envoie un signal validé au coordinator avec protection contre contradictions."""
        try:
            symbol = signal['symbol']
            side = signal['side']
            current_time = datetime.utcnow()
            
            # Vérifier s'il y a un signal récent contradictoire
            if symbol in self.recent_signals:
                recent_signal = self.recent_signals[symbol]
                time_diff = (current_time - recent_signal['timestamp']).total_seconds()
                
                if time_diff < self.contradiction_window:
                    recent_side = recent_signal['side']
                    
                    # Bloquer si signal opposé dans la fenêtre
                    if (side == 'BUY' and recent_side == 'SELL') or (side == 'SELL' and recent_side == 'BUY'):
                        logger.warning(f"🚫 Signal {side} {symbol} BLOQUÉ: signal {recent_side} envoyé il y a {time_diff:.1f}s")
                        self.stats['signals_blocked'] += 1
                        return
                    
                    # Même direction : remplacer le signal précédent
                    if side == recent_side:
                        logger.info(f"🔄 Signal {side} {symbol} remplace le précédent (il y a {time_diff:.1f}s)")
            
            # Envoyer le signal
            signal_json = json.dumps(signal, default=str)
            await self.redis_client.publish(self.output_channel, signal_json)
            
            # Enregistrer ce signal comme récent
            self.recent_signals[symbol] = {
                'side': side,
                'timestamp': current_time
            }
            
            logger.info(f"✅ Signal envoyé au coordinator: {symbol} {side} (confidence: {signal.get('confidence', 0):.2f})")
            
        except Exception as e:
            logger.error(f"Erreur envoi signal au coordinator: {e}")
            self.stats['errors'] += 1
            
    async def _health_monitor(self):
        """Monitor de santé du service."""
        while True:
            try:
                await asyncio.sleep(30)  # Toutes les 30 secondes
                
                # Statistiques du buffer
                buffer_status = await self.signal_buffer.get_buffer_status()
                
                # Nettoyer les anciens signaux (plus de contradiction_window)
                current_time = datetime.utcnow()
                to_remove = []
                for symbol, signal_data in self.recent_signals.items():
                    age = (current_time - signal_data['timestamp']).total_seconds()
                    if age > self.contradiction_window:
                        to_remove.append(symbol)
                
                for symbol in to_remove:
                    del self.recent_signals[symbol]
                
                # Log des statistiques
                logger.info(f"Stats agrégation - Reçus: {self.stats['signals_received']}, "
                           f"Traités: {self.stats['batches_processed']}, "
                           f"Envoyés: {self.stats['signals_sent']}, "
                           f"Bloqués: {self.stats['signals_blocked']}, "
                           f"Consensus insuffisants: {self.stats['consensus_rejected_insufficient']}, "
                           f"Buffer: {buffer_status['total_buffered_signals']} signaux")
                
                # Log des stats de validation
                validation_stats = self.signal_processor.get_stats()
                if validation_stats['signals_processed'] > 0:
                    logger.info(f"Stats validation - Taux veto: {validation_stats.get('veto_rate', 0):.1%}, "
                               f"Taux validation: {validation_stats.get('validation_rate', 0):.1%}")
                    
            except Exception as e:
                logger.error(f"Erreur health monitor: {e}")
                
    async def shutdown(self):
        """Arrêt propre du service."""
        logger.info("Arrêt du service d'agrégation...")
        
        # Vider le buffer
        await self.signal_buffer.force_flush_all()
        await self.signal_buffer.cleanup()
        
        # Fermer Redis
        if self.redis_client:
            await self.redis_client.close()
            
        logger.info("Service d'agrégation arrêté")
        
    def get_stats(self) -> Dict[str, Any]:
        """Retourne toutes les statistiques du service."""
        return {
            'aggregator_stats': self.stats,
            'validation_stats': self.signal_processor.get_stats(),
            'buffer_status': asyncio.create_task(self.signal_buffer.get_buffer_status()) if self.signal_buffer else None
        }