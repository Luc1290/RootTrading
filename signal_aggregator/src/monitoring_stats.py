"""
Système de monitoring et statistiques par régime et stratégie
"""
from typing import Dict, List, Optional, DefaultDict, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class SignalMonitoringStats:
    """
    Collecte et analyse les statistiques de signaux par régime et stratégie
    """
    
    def __init__(self, redis_client=None, history_size: int = 1000):
        self.redis = redis_client
        self.history_size = history_size
        
        # Compteurs par régime et stratégie
        self.accepted_signals: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.rejected_signals: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Historique détaillé des signaux
        self.signal_history: deque = deque(maxlen=history_size)
        
        # Statistiques de performance en temps réel
        self.hourly_stats: DefaultDict[str, Dict] = defaultdict(dict)
        
        # Charger les stats existantes
        if self.redis:
            self._load_stats()
    
    def _load_stats(self):
        """Charge les statistiques depuis Redis"""
        try:
            data = self.redis.get("signal_monitoring_stats")
            if data:
                # Redis peut retourner un dict ou une string selon la config
                if isinstance(data, str):
                    stats_data = json.loads(data)
                else:
                    stats_data = data
                
                # Reconstruire les compteurs
                for regime, strategies in stats_data.get('accepted', {}).items():
                    for strategy, count in strategies.items():
                        self.accepted_signals[regime][strategy] = count
                
                for regime, strategies in stats_data.get('rejected', {}).items():
                    for strategy, count in strategies.items():
                        self.rejected_signals[regime][strategy] = count
                
                logger.info(f"📊 Statistiques de monitoring chargées")
        except Exception as e:
            logger.error(f"Erreur chargement stats monitoring: {e}")
    
    def _save_stats(self):
        """Sauvegarde les statistiques dans Redis"""
        if not self.redis:
            return
            
        try:
            stats_data = {
                'accepted': {regime: dict(strategies) for regime, strategies in self.accepted_signals.items()},
                'rejected': {regime: dict(strategies) for regime, strategies in self.rejected_signals.items()},
                'last_update': datetime.now().isoformat()
            }
            
            self.redis.set("signal_monitoring_stats", json.dumps(stats_data))
            
            # Sauvegarde aussi l'historique récent (dernier 100 signaux)
            recent_history = list(self.signal_history)[-100:]
            self.redis.set("signal_history_recent", json.dumps(recent_history))
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde stats monitoring: {e}")
    
    def record_signal_decision(self, 
                             strategy: str, 
                             regime: str, 
                             symbol: str,
                             accepted: bool,
                             confidence: float,
                             reason: Optional[str] = None):
        """
        Enregistre une décision de signal
        
        Args:
            strategy: Nom de la stratégie
            regime: Régime de marché
            symbol: Symbole tradé
            accepted: True si accepté, False si rejeté
            confidence: Niveau de confiance
            reason: Raison du rejet (si applicable)
        """
        # Normaliser le nom du régime
        regime_name = regime if isinstance(regime, str) else getattr(regime, 'name', str(regime))
        
        # Compteurs principaux
        if accepted:
            self.accepted_signals[regime_name][strategy] += 1
        else:
            self.rejected_signals[regime_name][strategy] += 1
        
        # Ajouter à l'historique détaillé
        signal_record = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'regime': regime_name,
            'symbol': symbol,
            'accepted': accepted,
            'confidence': confidence,
            'reason': reason
        }
        
        self.signal_history.append(signal_record)
        
        # Log des tendances importantes
        total_for_strategy = self.get_total_signals_for_strategy(strategy)
        acceptance_rate = self.get_acceptance_rate(strategy, regime_name)
        
        if total_for_strategy > 0 and total_for_strategy % 20 == 0:  # Log tous les 20 signaux
            logger.info(f"📈 Stats {strategy} en {regime_name}: "
                       f"{acceptance_rate:.1%} acceptation sur {total_for_strategy} signaux")
        
        # Sauvegarde périodique
        if len(self.signal_history) % 50 == 0:
            self._save_stats()
    
    def get_acceptance_rate(self, strategy: str, regime: Optional[str] = None) -> float:
        """Calcule le taux d'acceptation pour une stratégie dans un régime"""
        if regime:
            accepted = self.accepted_signals[regime][strategy]
            rejected = self.rejected_signals[regime][strategy]
        else:
            # Tous régimes confondus
            accepted = sum(self.accepted_signals[r][strategy] for r in self.accepted_signals)
            rejected = sum(self.rejected_signals[r][strategy] for r in self.rejected_signals)
        
        total = accepted + rejected
        return accepted / total if total > 0 else 0.0
    
    def get_total_signals_for_strategy(self, strategy: str) -> int:
        """Retourne le nombre total de signaux pour une stratégie"""
        accepted = sum(self.accepted_signals[r][strategy] for r in self.accepted_signals)
        rejected = sum(self.rejected_signals[r][strategy] for r in self.rejected_signals)
        return accepted + rejected
    
    def get_regime_performance(self, regime: str) -> Dict[str, Any]:
        """Retourne les performances par stratégie pour un régime donné"""
        strategies_stats = {}
        
        for strategy in set(list(self.accepted_signals[regime].keys()) + 
                          list(self.rejected_signals[regime].keys())):
            accepted = self.accepted_signals[regime][strategy]
            rejected = self.rejected_signals[regime][strategy]
            total = accepted + rejected
            
            if total > 0:
                strategies_stats[strategy] = {
                    'accepted': accepted,
                    'rejected': rejected,
                    'total': total,
                    'acceptance_rate': accepted / total,
                    'signals_per_hour': self._calculate_signals_per_hour(strategy, regime)
                }
        
        return strategies_stats
    
    def _calculate_signals_per_hour(self, strategy: str, regime: str) -> float:
        """Calcule le nombre de signaux par heure pour une stratégie/régime"""
        # Analyser les dernières 24h
        cutoff = datetime.now() - timedelta(hours=24)
        
        recent_signals = [
            s for s in self.signal_history 
            if (s['strategy'] == strategy and 
                s['regime'] == regime and 
                datetime.fromisoformat(s['timestamp']) > cutoff)
        ]
        
        if not recent_signals:
            return 0.0
        
        # Calculer la durée réelle couverte
        timestamps = [datetime.fromisoformat(s['timestamp']) for s in recent_signals]
        duration_hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600
        
        return len(recent_signals) / max(1, duration_hours)
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Génère un rapport complet des statistiques"""
        report = {
            'summary': {
                'total_signals': len(self.signal_history),
                'total_accepted': sum(sum(strategies.values()) for strategies in self.accepted_signals.values()),
                'total_rejected': sum(sum(strategies.values()) for strategies in self.rejected_signals.values()),
                'overall_acceptance_rate': 0.0
            },
            'by_regime': {},
            'by_strategy': {},
            'trends': self._analyze_trends()
        }
        
        # Calcul du taux global
        total_accepted = report['summary']['total_accepted']
        total_rejected = report['summary']['total_rejected']
        total_all = total_accepted + total_rejected
        
        if total_all > 0:
            report['summary']['overall_acceptance_rate'] = total_accepted / total_all
        
        # Analyse par régime
        all_regimes = set(list(self.accepted_signals.keys()) + list(self.rejected_signals.keys()))
        for regime in all_regimes:
            report['by_regime'][regime] = self.get_regime_performance(regime)
        
        # Analyse par stratégie
        all_strategies: set[str] = set()
        for regime_strategies in self.accepted_signals.values():
            all_strategies.update(regime_strategies.keys())
        for regime_strategies in self.rejected_signals.values():
            all_strategies.update(regime_strategies.keys())
        
        for strategy in all_strategies:
            total_signals = self.get_total_signals_for_strategy(strategy)
            acceptance_rate = self.get_acceptance_rate(strategy)
            
            report['by_strategy'][strategy] = {
                'total_signals': total_signals,
                'acceptance_rate': acceptance_rate,
                'best_regime': self._find_best_regime_for_strategy(strategy),
                'worst_regime': self._find_worst_regime_for_strategy(strategy)
            }
        
        return report
    
    def _find_best_regime_for_strategy(self, strategy: str) -> Optional[str]:
        """Trouve le régime avec le meilleur taux d'acceptation pour une stratégie"""
        best_regime = None
        best_rate = -1.0
        
        for regime in self.accepted_signals:
            if strategy in self.accepted_signals[regime] or strategy in self.rejected_signals[regime]:
                rate = self.get_acceptance_rate(strategy, regime)
                total = (self.accepted_signals[regime][strategy] + 
                        self.rejected_signals[regime][strategy])
                
                if total >= 5 and rate > best_rate:  # Au moins 5 signaux
                    best_rate = rate
                    best_regime = regime
        
        return best_regime
    
    def _find_worst_regime_for_strategy(self, strategy: str) -> Optional[str]:
        """Trouve le régime avec le pire taux d'acceptation pour une stratégie"""
        worst_regime = None
        worst_rate = 2.0  # > 1
        
        for regime in self.accepted_signals:
            if strategy in self.accepted_signals[regime] or strategy in self.rejected_signals[regime]:
                rate = self.get_acceptance_rate(strategy, regime)
                total = (self.accepted_signals[regime][strategy] + 
                        self.rejected_signals[regime][strategy])
                
                if total >= 5 and rate < worst_rate:  # Au moins 5 signaux
                    worst_rate = rate
                    worst_regime = regime
        
        return worst_regime
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyse les tendances temporelles"""
        if len(self.signal_history) < 10:
            return {'status': 'insufficient_data'}
        
        # Analyser les dernières heures
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_6h = now - timedelta(hours=6)
        
        recent_signals = [s for s in self.signal_history if datetime.fromisoformat(s['timestamp']) > last_hour]
        moderate_signals = [s for s in self.signal_history if datetime.fromisoformat(s['timestamp']) > last_6h]
        
        return {
            'last_hour': {
                'total': len(recent_signals),
                'accepted': sum(1 for s in recent_signals if s['accepted']),
                'acceptance_rate': sum(1 for s in recent_signals if s['accepted']) / max(1, len(recent_signals))
            },
            'last_6h': {
                'total': len(moderate_signals),
                'accepted': sum(1 for s in moderate_signals if s['accepted']),
                'acceptance_rate': sum(1 for s in moderate_signals if s['accepted']) / max(1, len(moderate_signals))
            },
            'most_active_regime': self._get_most_active_regime(),
            'most_successful_strategy': self._get_most_successful_strategy()
        }
    
    def _get_most_active_regime(self) -> Optional[str]:
        """Trouve le régime le plus actif"""
        regime_counts: defaultdict[str, int] = defaultdict(int)
        
        for regime in self.accepted_signals:
            regime_counts[regime] += sum(self.accepted_signals[regime].values())
        for regime in self.rejected_signals:
            regime_counts[regime] += sum(self.rejected_signals[regime].values())
        
        return max(regime_counts, key=lambda x: regime_counts[x]) if regime_counts else None
    
    def _get_most_successful_strategy(self) -> Optional[str]:
        """Trouve la stratégie avec le meilleur taux d'acceptation"""
        all_strategies: set[str] = set()
        for regime_strategies in self.accepted_signals.values():
            all_strategies.update(regime_strategies.keys())
        for regime_strategies in self.rejected_signals.values():
            all_strategies.update(regime_strategies.keys())
        
        best_strategy = None
        best_rate = -1.0
        
        for strategy in all_strategies:
            total = self.get_total_signals_for_strategy(strategy)
            if total >= 10:  # Au moins 10 signaux
                rate = self.get_acceptance_rate(strategy)
                if rate > best_rate:
                    best_rate = rate
                    best_strategy = strategy
        
        return best_strategy