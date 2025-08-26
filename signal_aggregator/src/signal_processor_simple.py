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
        
        # Statistiques ultra-simples
        self.stats = {
            'signals_processed': 0,
            'signals_validated': 0,
            'consensus_rejected': 0,
            'critical_filter_rejected': 0,
            'errors': 0
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
                
            # Récupération du contexte de marché
            context = self.context_manager.get_market_context(symbol, timeframe)
            if not context:
                logger.warning(f"Pas de contexte marché pour {symbol} {timeframe}")
                return None
                
            # ÉTAPE 1: Consensus adaptatif (principal système)
            market_regime = context.get('market_regime', 'UNKNOWN')
            logger.info(f"🔍 Market regime pour {symbol}: {market_regime}")
            logger.info(f"📊 Stratégies: {[s.get('strategy') for s in signals]}")
            has_consensus, consensus_analysis = self.consensus_analyzer.analyze_adaptive_consensus(
                signals, market_regime, timeframe
            )
            
            logger.info(f"🔍 Consensus result: has_consensus={has_consensus}, analysis={consensus_analysis}")
            
            if not has_consensus:
                logger.info(f"❌ Consensus rejeté {symbol} {side}: {consensus_analysis.get('reason') if consensus_analysis else 'None'}")
                self.stats['consensus_rejected'] += 1
                return None
                
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
                consensus_analysis, filter_reason
            )
            
            self.stats['signals_validated'] += 1
            
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
                               filter_status: str) -> Dict[str, Any]:
        """Construit le signal de consensus final."""
        
        # Calculs de base
        strategies_count = len(signals)
        avg_confidence = sum(float(s['confidence']) for s in signals) / strategies_count
        
        # Métadonnées des stratégies
        strategy_names = [s['strategy'] for s in signals]
        family_distribution = consensus_analysis.get('families_count', {})
        
        # Signal de consensus compatible avec StrategySignal
        consensus_signal = {
            'strategy': 'CONSENSUS',  # Champ requis pour StrategySignal
            'symbol': symbol,
            'side': side,
            'timestamp': datetime.utcnow().isoformat(),
            'price': context.get('current_price', 0.0),  # Prix actuel du marché
            'confidence': min(1.0, consensus_analysis.get('consensus_strength', 0) / 5.0),  # Normaliser 0-1
            
            # Toutes les métadonnées dans le champ metadata
            'metadata': {
                'type': 'CONSENSUS',
                'timeframe': timeframe,
                'strategies_count': strategies_count,
                'strategy_names': strategy_names,
                'avg_confidence': avg_confidence,
                'consensus_strength': consensus_analysis.get('consensus_strength', 0),
                'market_regime': consensus_analysis.get('regime', 'UNKNOWN'),
                'family_distribution': family_distribution,
                'filter_status': filter_status,
                
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
        """Retourne les statistiques du processeur."""
        total_processed = self.stats['signals_processed']
        if total_processed > 0:
            success_rate = (self.stats['signals_validated'] / total_processed) * 100
        else:
            success_rate = 0
            
        return {
            **self.stats,
            'success_rate_percent': success_rate,
            'filter_config': self.critical_filters.get_filter_stats()
        }
        
    def reset_stats(self):
        """Remet à zéro les statistiques."""
        for key in self.stats:
            if isinstance(self.stats[key], (int, float)):
                self.stats[key] = 0