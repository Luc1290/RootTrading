"""
Module de traitement des signaux - VERSION ULTRA-SIMPLIFIÉE.
Remplace l'ancien système complexe par juste consensus adaptatif + filtres critiques.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
import json
from adaptive_consensus import AdaptiveConsensusAnalyzer
from critical_filters import CriticalFilters

logger = logging.getLogger(__name__)


class SimpleSignalProcessor:
    """
    Processeur ultra-simplifié pour la validation des signaux.
    Logic: Consensus adaptatif + quelques filtres critiques seulement.
    """
    
    def __init__(self, context_manager, database_manager=None):
        """
        Initialise le processeur simplifié.
        
        Args:
            context_manager: Gestionnaire de contexte de marché
            database_manager: Gestionnaire de base de données (optionnel)
        """
        self.context_manager = context_manager
        self.database_manager = database_manager
        
        # Systèmes simplifiés
        self.consensus_analyzer = AdaptiveConsensusAnalyzer()
        self.critical_filters = CriticalFilters()
        
        # SEUIL MINIMUM DE CONFIDENCE POUR SCALPING
        self.min_confidence_threshold = 0.6  # Rejeter signaux < 60% confidence
        
        # Cache de contexte pour optimisation
        self.context_cache = {}
        self.cache_ttl = 5  # 5 secondes
        
        # Statistiques détaillées pour debug et optimisation
        self.stats = {
            'signals_processed': 0,
            'signals_validated': 0,
            'consensus_rejected': 0,
            'critical_filter_rejected': 0,
            'low_confidence_rejected': 0,  # NOUVEAU: Track rejets par faible confidence
            'errors': 0,
            'rejections_by_regime': {},  # Par régime de marché
            'rejections_by_family': {},  # Par famille de stratégies
            'avg_strategies_per_consensus': 0,
            'avg_confidence_validated': 0,
            'consensus_strength_distribution': [],  # Pour analyser les seuils
            'wave_winner_signals': 0,  # Signaux post-vague
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    async def process_signal(self, signal_data: str) -> Optional[Dict[str, Any]]:
        """
        Traite un signal individuel reçu depuis Redis.
        VERSION SIMPLIFIÉE: Juste structure + passage au buffer.
        
        Args:
            signal_data: Données du signal au format JSON
            
        Returns:
            Signal parsé et validé structurellement ou None
        """
        try:
            # Parsing du message
            signal = json.loads(signal_data)
            self.stats['signals_processed'] += 1
            
            # Validation structure de base uniquement
            if not self._validate_signal_structure(signal):
                logger.debug(f"Signal rejeté: structure invalide")
                self.stats['errors'] += 1
                return None
            
            # NORMALISATION CONFIDENCE: Clamp entre 0 et 1
            conf = signal.get('confidence')
            if conf is not None:
                try:
                    conf_val = float(conf)
                except:
                    conf_val = 0.0
                # Clamp entre 0 et 1
                if conf_val < 0.0: 
                    conf_val = 0.0
                if conf_val > 1.0: 
                    conf_val = 1.0
                signal['confidence'] = conf_val
            else:
                # Confidence manquante = signal faible par défaut
                signal['confidence'] = 0.0
            
            # FILTRE CONFIDENCE MINIMUM POUR SCALPING
            if signal['confidence'] < self.min_confidence_threshold:
                logger.debug(f"Signal rejeté: confidence {signal['confidence']:.2f} < {self.min_confidence_threshold}")
                self.stats['low_confidence_rejected'] += 1
                return None
                
            # Ajouter timestamp de réception
            signal['received_at'] = datetime.utcnow().isoformat()
            
            return signal
            
        except json.JSONDecodeError as e:
            logger.error(f"Erreur parsing JSON signal: {e}")
            self.stats['errors'] += 1
            return None
        except Exception as e:
            logger.error(f"Erreur traitement signal: {e}")
            self.stats['errors'] += 1
            return None
    
    async def _get_cached_context(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Récupère le contexte avec cache pour optimisation."""
        cache_key = f"{symbol}_{timeframe}"
        now = datetime.utcnow()
        
        if cache_key in self.context_cache:
            cached_context, timestamp = self.context_cache[cache_key]
            if (now - timestamp).total_seconds() < self.cache_ttl:
                self.stats['cache_hits'] += 1
                return cached_context
        
        # Cache miss - récupérer le contexte
        context = self.context_manager.get_market_context(symbol, timeframe)
        if context:
            self.context_cache[cache_key] = (context, now)
        self.stats['cache_misses'] += 1
        return context
    
    def _normalize_consensus_strength(self, strength: float, market_regime: str) -> float:
        """Normalise le consensus_strength en confidence 0-1 basé sur les seuils réels."""
        # Seuils typiques selon adaptive_consensus.py
        if market_regime in ['TRENDING_BEAR', 'BREAKOUT_BEAR']:
            max_expected = 3.5  # Plus strict en bear
        else:
            max_expected = 2.5  # Normal
        
        # Normalisation avec saturation à 1.0
        normalized = min(1.0, strength / max_expected)
        return max(0.0, normalized)  # S'assurer que c'est positif
            
    async def validate_signal_group(self, signals: list, symbol: str, 
                                  timeframe: str, side: str) -> Optional[Dict[str, Any]]:
        """
        Valide un groupe de signaux avec le système simplifié.
        REMPLACE TOUTE LA LOGIQUE COMPLEXE DE VALIDATION.
        
        Args:
            signals: Liste des signaux du même symbole/direction
            symbol: Symbole tradé
            timeframe: Timeframe des signaux  
            side: Direction (BUY/SELL)
            
        Returns:
            Signal de consensus validé ou None si rejeté
        """
        try:
            if not signals:
                return None
                
            # Détecter si c'est un groupe post-vague
            is_wave_winner = any(
                s.get('metadata', {}).get('wave_resolution', {}).get('is_wave_winner', False)
                for s in signals
            )
            
            if is_wave_winner:
                self.stats['wave_winner_signals'] += 1
                logger.info(f"🏆 Validation post-vague pour {symbol} {side}")
            
            # Récupération du contexte de marché avec cache
            context = await self._get_cached_context(symbol, timeframe)
            if not context:
                logger.warning(f"Pas de contexte marché pour {symbol} {timeframe}")
                return None
                
            # ÉTAPE 1: Consensus adaptatif (principal système)
            market_regime = context.get('market_regime', 'UNKNOWN')
            logger.info(f"🔍 Market regime pour {symbol}: {market_regime}")
            logger.info(f"📊 Stratégies: {[s.get('strategy') for s in signals]}")
            
            # Adapter les critères si c'est un signal post-vague
            if is_wave_winner:
                # Assouplir légèrement les critères car déjà passé une sélection
                logger.info("🏆 Critères assouplis pour gagnant de vague")
            
            has_consensus, consensus_analysis = self.consensus_analyzer.analyze_adaptive_consensus(
                signals, market_regime, timeframe
            )
            
            logger.info(f"🔍 Consensus result: has_consensus={has_consensus}, analysis={consensus_analysis}")
            
            if not has_consensus:
                # Enregistrer stats détaillées du rejet
                if market_regime not in self.stats['rejections_by_regime']:
                    self.stats['rejections_by_regime'][market_regime] = 0
                self.stats['rejections_by_regime'][market_regime] += 1
                
                logger.info(f"❌ Consensus rejeté {symbol} {side}: {consensus_analysis.get('reason') if consensus_analysis else 'None'}")
                self.stats['consensus_rejected'] += 1
                return None
            
            # Enregistrer la strength pour statistiques
            consensus_strength = consensus_analysis.get('consensus_strength', 0)
            self.stats['consensus_strength_distribution'].append(consensus_strength)
                
            # ÉTAPE 2: Filtres critiques seulement (éviter les vrais dangers)
            filters_pass, filter_reason = self.critical_filters.apply_critical_filters(
                signals, context
            )
            
            if not filters_pass:
                logger.info(f"Filtres critiques rejetent {symbol} {side}: {filter_reason}")
                self.stats['critical_filter_rejected'] += 1
                return None
                
            # ÉTAPE 3: Sauvegarder les signaux individuels en base de données
            if self.database_manager:
                for signal in signals:
                    try:
                        # Préparer le signal individuel pour la base de données
                        individual_signal = {
                            'strategy': signal.get('strategy'),
                            'symbol': signal.get('symbol'),
                            'side': signal.get('side'),
                            'timestamp': signal.get('timestamp', datetime.utcnow().isoformat()),
                            'confidence': signal.get('confidence'),
                            'price': context.get('current_price', 0.0),
                            'metadata': {
                                'timeframe': signal.get('timeframe'),
                                'original_metadata': signal.get('metadata', {}),
                                'part_of_consensus': True,
                                'market_regime': consensus_analysis.get('regime', 'UNKNOWN')
                            }
                        }
                        # Stocker le signal individuel
                        self.database_manager.store_validated_signal(individual_signal)
                    except Exception as e:
                        logger.warning(f"Erreur sauvegarde signal individuel {signal.get('strategy')}: {e}")
            
            # ÉTAPE 4: Construire signal de consensus validé
            consensus_signal = self._build_consensus_signal(
                signals, symbol, timeframe, side, context, 
                consensus_analysis, filter_reason, is_wave_winner
            )
            
            # Mettre à jour les statistiques moyennes
            self.stats['signals_validated'] += 1
            self.stats['avg_strategies_per_consensus'] = (
                (self.stats['avg_strategies_per_consensus'] * (self.stats['signals_validated'] - 1) + len(signals)) / 
                self.stats['signals_validated']
            )
            confidence = consensus_signal.get('confidence', 0)
            self.stats['avg_confidence_validated'] = (
                (self.stats['avg_confidence_validated'] * (self.stats['signals_validated'] - 1) + confidence) / 
                self.stats['signals_validated']
            )
            
            # ÉTAPE 5: Sauvegarde du consensus
            if self.database_manager:
                try:
                    signal_id = self.database_manager.store_validated_signal(consensus_signal)
                    if signal_id:
                        # Ajouter le db_id dans les métadonnées pour que le coordinator puisse le trouver
                        consensus_signal['metadata']['db_id'] = signal_id
                        logger.debug(f"DB ID {signal_id} ajouté au signal consensus {symbol}")
                except Exception as e:
                    logger.error(f"Erreur sauvegarde consensus: {e}")
                    
            logger.info(f"✅ Signal consensus validé: {symbol} {side} "
                       f"({len(signals)} stratégies, score: {consensus_analysis.get('consensus_strength', 0):.2f})")
                       
            return consensus_signal
            
        except Exception as e:
            logger.error(f"Erreur validation groupe signaux: {e}")
            self.stats['errors'] += 1
            return None
            
    def _validate_signal_structure(self, signal: Dict[str, Any]) -> bool:
        """Valide la structure de base d'un signal."""
        required_fields = ['strategy', 'symbol', 'side', 'confidence', 'timeframe']
        
        for field in required_fields:
            if field not in signal:
                return False
                
        # Validation des valeurs
        if signal['side'] not in ['BUY', 'SELL']:
            return False
            
        try:
            confidence = float(signal['confidence'])
            if confidence < 0 or confidence > 1:
                return False
        except (ValueError, TypeError):
            return False
            
        return True
        
    def _build_consensus_signal(self, signals: list, symbol: str, timeframe: str, 
                               side: str, context: Dict[str, Any], 
                               consensus_analysis: Dict[str, Any],
                               filter_status: str, is_wave_winner: bool = False) -> Dict[str, Any]:
        """Construit le signal de consensus final."""
        
        # Calculs de base
        strategies_count = len(signals)
        avg_confidence = sum(float(s['confidence']) for s in signals) / strategies_count
        
        # Analyser la distribution des timeframes
        timeframe_distribution = {}
        for signal in signals:
            tf = signal.get('timeframe', timeframe)  # Utiliser le timeframe par défaut si manquant
            timeframe_distribution[tf] = timeframe_distribution.get(tf, 0) + 1
        
        dominant_timeframe = max(timeframe_distribution, key=timeframe_distribution.get) if timeframe_distribution else timeframe
        
        # Métadonnées des stratégies
        strategy_names = [s['strategy'] for s in signals]
        family_distribution = consensus_analysis.get('families_count', {})
        
        # Utiliser la normalisation améliorée
        market_regime = consensus_analysis.get('regime', context.get('market_regime', 'UNKNOWN'))
        normalized_confidence = self._normalize_consensus_strength(
            consensus_analysis.get('consensus_strength', 0), 
            market_regime
        )
        
        # Signal de consensus compatible avec StrategySignal
        consensus_signal = {
            'strategy': 'CONSENSUS',  # Champ requis pour StrategySignal
            'symbol': symbol,
            'side': side,
            'timestamp': datetime.utcnow().isoformat(),
            'price': context.get('current_price', 0.0),  # Prix actuel du marché
            'confidence': normalized_confidence,  # Normalisation améliorée basée sur le régime
            
            # Toutes les métadonnées dans le champ metadata
            'metadata': {
                'type': 'CONSENSUS',
                'timeframe': timeframe,
                'strategies_count': strategies_count,
                'strategy_names': strategy_names,
                'avg_confidence': avg_confidence,
                'consensus_strength': consensus_analysis.get('consensus_strength', 0),
                'market_regime': market_regime,
                'family_distribution': family_distribution,
                'filter_status': filter_status,
                
                # Distribution des timeframes
                'timeframe_distribution': timeframe_distribution,
                'dominant_timeframe': dominant_timeframe,
                'is_multi_timeframe': len(timeframe_distribution) > 1,
                
                # Signaux post-vague
                'is_wave_winner': is_wave_winner,
                
                # Contexte de marché
                'volume_ratio': context.get('volume_ratio'),
                'confluence_score': context.get('confluence_score'),
                'volatility_regime': context.get('volatility_regime'),
                
                # Debug
                'processor': 'SimpleSignalProcessor',
                'consensus_analysis': consensus_analysis,
                'original_signals': signals
            }
        }
        
        return consensus_signal
        
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques détaillées du processeur."""
        total_processed = self.stats['signals_processed']
        if total_processed > 0:
            success_rate = (self.stats['signals_validated'] / total_processed) * 100
        else:
            success_rate = 0
        
        # Statistiques de cache
        total_cache_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = (self.stats['cache_hits'] / total_cache_requests * 100) if total_cache_requests > 0 else 0
        
        # Statistiques de consensus strength
        consensus_strengths = self.stats['consensus_strength_distribution']
        consensus_stats = {}
        if consensus_strengths:
            consensus_stats = {
                'min': min(consensus_strengths),
                'max': max(consensus_strengths),
                'avg': sum(consensus_strengths) / len(consensus_strengths),
                'count': len(consensus_strengths)
            }
            
        return {
            **self.stats,
            'success_rate_percent': success_rate,
            'cache_hit_rate_percent': cache_hit_rate,
            'consensus_strength_stats': consensus_stats,
            'filter_config': self.critical_filters.get_filter_stats() if hasattr(self.critical_filters, 'get_filter_stats') else {},
            'performance': {
                'avg_strategies_per_consensus': round(self.stats['avg_strategies_per_consensus'], 2),
                'avg_confidence_validated': round(self.stats['avg_confidence_validated'], 3),
                'wave_winner_percentage': (self.stats['wave_winner_signals'] / max(1, self.stats['signals_validated'])) * 100
            }
        }
        
    def reset_stats(self):
        """Remet à zéro les statistiques."""
        for key in self.stats:
            if isinstance(self.stats[key], (int, float)):
                self.stats[key] = 0
            elif isinstance(self.stats[key], dict):
                self.stats[key] = {}
            elif isinstance(self.stats[key], list):
                self.stats[key] = []
        
        # Vider le cache de contexte aussi
        self.context_cache = {}