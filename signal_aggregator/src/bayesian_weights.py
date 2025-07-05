"""
Système de pondération bayésienne adaptative pour les stratégies
"""
import numpy as np
from typing import Dict, List, Optional
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

@dataclass
class StrategyPerformance:
    """Métriques de performance d'une stratégie"""
    wins: int = 0
    losses: int = 0
    total_signals: int = 0
    cumulative_return: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def win_rate(self) -> float:
        if self.total_signals == 0:
            return 0.5  # Prior neutre
        return self.wins / self.total_signals
    
    @property
    def profit_factor(self) -> float:
        if self.losses == 0:
            return 2.0 if self.wins > 0 else 1.0
        return self.wins / self.losses

class BayesianStrategyWeights:
    """
    Calcul des poids de stratégie basé sur l'approche bayésienne
    avec mise à jour continue basée sur les performances.
    Sauvegarde en PostgreSQL ET cache Redis.
    """
    
    def __init__(self, redis_client=None, db_pool=None, alpha: float = 1.0, beta: float = 1.0):
        """
        Args:
            redis_client: Client Redis pour cache
            db_pool: Pool de connexions PostgreSQL pour persistance
            alpha: Paramètre alpha de la distribution Beta (succès initiaux + 1)
            beta: Paramètre beta de la distribution Beta (échecs initiaux + 1)
        """
        self.redis = redis_client
        self.db_pool = db_pool
        self.alpha_prior = alpha
        self.beta_prior = beta
        self.performance_data: Dict[str, StrategyPerformance] = {}
        self.base_weight = 1.0
        
        # Charger les données historiques depuis DB puis Redis
        if self.db_pool:
            self._load_from_database()
        elif self.redis:
            self._load_performance_data()
    
    def _load_performance_data(self):
        """Charge les données de performance depuis Redis"""
        try:
            data = self.redis.get("strategy_performance_bayesian")
            if data:
                loaded_data = json.loads(data)
                for strategy, perf_dict in loaded_data.items():
                    self.performance_data[strategy] = StrategyPerformance(
                        wins=perf_dict.get('wins', 0),
                        losses=perf_dict.get('losses', 0),
                        total_signals=perf_dict.get('total_signals', 0),
                        cumulative_return=perf_dict.get('cumulative_return', 0.0),
                        last_update=datetime.fromisoformat(perf_dict.get('last_update', datetime.now().isoformat()))
                    )
                logger.info(f"📊 Chargé les performances bayésiennes pour {len(self.performance_data)} stratégies")
        except Exception as e:
            logger.error(f"Erreur chargement performances bayésiennes: {e}")
    
    def _save_performance_data(self):
        """Sauvegarde les données de performance dans Redis"""
        if not self.redis:
            return
            
        try:
            data_to_save = {}
            for strategy, perf in self.performance_data.items():
                data_to_save[strategy] = {
                    'wins': perf.wins,
                    'losses': perf.losses,
                    'total_signals': perf.total_signals,
                    'cumulative_return': perf.cumulative_return,
                    'last_update': perf.last_update.isoformat()
                }
            
            self.redis.set("strategy_performance_bayesian", json.dumps(data_to_save))
        except Exception as e:
            logger.error(f"Erreur sauvegarde performances bayésiennes: {e}")
    
    def update_performance(self, strategy: str, is_win: bool, return_pct: float = 0.0):
        """
        Met à jour les performances d'une stratégie
        
        Args:
            strategy: Nom de la stratégie
            is_win: True si le trade était gagnant
            return_pct: Pourcentage de retour du trade
        """
        if strategy not in self.performance_data:
            self.performance_data[strategy] = StrategyPerformance()
        
        perf = self.performance_data[strategy]
        perf.total_signals += 1
        
        if is_win:
            perf.wins += 1
        else:
            perf.losses += 1
        
        perf.cumulative_return += return_pct
        perf.last_update = datetime.now()
        
        # Sauvegarder périodiquement en DB ET Redis
        if perf.total_signals % 5 == 0:  # Plus fréquent pour DB
            if self.db_pool:
                self._save_to_database()
            self._save_performance_data()
        
        logger.debug(f"📈 Mise à jour {strategy}: win_rate={perf.win_rate:.2%}, total={perf.total_signals}")
    
    def get_bayesian_weight(self, strategy: str) -> float:
        """
        Calcule le poids bayésien d'une stratégie
        
        Utilise une distribution Beta avec mise à jour bayésienne:
        - Prior: Beta(α, β)
        - Posterior: Beta(α + succès, β + échecs)
        - Poids = E[θ] = (α + succès) / (α + β + total)
        """
        if strategy not in self.performance_data:
            # Retourner le prior
            return self.alpha_prior / (self.alpha_prior + self.beta_prior)
        
        perf = self.performance_data[strategy]
        
        # Paramètres de la distribution Beta posterior
        alpha_post = self.alpha_prior + perf.wins
        beta_post = self.beta_prior + perf.losses
        
        # Espérance de la distribution Beta (taux de succès estimé)
        expected_win_rate = alpha_post / (alpha_post + beta_post)
        
        # Facteur de confiance basé sur le nombre d'observations
        # Plus on a de données, plus on fait confiance à l'estimation
        confidence = 1 - math.exp(-perf.total_signals / 50)  # Converge vers 1
        
        # Ajustement basé sur le profit factor
        profit_adjustment = min(2.0, max(0.5, perf.profit_factor))
        
        # Poids final combinant win rate estimé, confiance et profit
        weight = expected_win_rate * confidence * profit_adjustment
        
        # Appliquer une transformation log-odds pour accentuer les différences
        # log(p/(1-p)) puis sigmoid pour ramener entre 0 et 1
        if 0 < weight < 1:
            log_odds = math.log(weight / (1 - weight))
            # Amplifier les différences
            log_odds *= 1.5
            # Retransformer en probabilité
            weight = 1 / (1 + math.exp(-log_odds))
        
        return max(0.1, min(2.0, weight))  # Limiter entre 0.1 et 2.0
    
    def get_all_weights(self, strategies: List[str]) -> Dict[str, float]:
        """
        Obtient les poids normalisés pour toutes les stratégies
        
        Args:
            strategies: Liste des noms de stratégies
            
        Returns:
            Dict {strategy_name: weight}
        """
        weights = {}
        
        for strategy in strategies:
            weights[strategy] = self.get_bayesian_weight(strategy)
        
        # Normaliser pour que la somme soit égale au nombre de stratégies
        # (pour maintenir la même échelle que les poids fixes)
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalization_factor = len(strategies) / total_weight
            weights = {k: v * normalization_factor for k, v in weights.items()}
        
        return weights
    
    def get_performance_summary(self) -> Dict[str, Dict]:
        """Retourne un résumé des performances de toutes les stratégies"""
        summary = {}
        
        for strategy, perf in self.performance_data.items():
            summary[strategy] = {
                'win_rate': perf.win_rate,
                'profit_factor': perf.profit_factor,
                'total_signals': perf.total_signals,
                'cumulative_return': perf.cumulative_return,
                'bayesian_weight': self.get_bayesian_weight(strategy),
                'last_update': perf.last_update.isoformat()
            }
        
        return summary
    
    def decay_old_performance(self, decay_days: int = 30, decay_factor: float = 0.95):
        """
        Applique un facteur de décroissance aux anciennes performances
        pour donner plus de poids aux performances récentes
        """
        current_time = datetime.now()
        
        for strategy, perf in self.performance_data.items():
            days_old = (current_time - perf.last_update).days
            
            if days_old > decay_days:
                # Appliquer le facteur de décroissance
                decay = decay_factor ** (days_old / decay_days)
                
                # Réduire l'impact des anciennes données
                perf.wins = int(perf.wins * decay)
                perf.losses = int(perf.losses * decay)
                perf.total_signals = perf.wins + perf.losses
                
                logger.debug(f"🔄 Décroissance appliquée à {strategy}: {days_old} jours, facteur {decay:.2f}")
        
        # Sauvegarder après décroissance
        if self.db_pool:
            self._save_to_database()
        self._save_performance_data()
    
    def _load_from_database(self):
        """Charge les performances depuis PostgreSQL"""
        try:
            import asyncio
            
            async def load_data():
                async with self.db_pool.acquire() as conn:
                    # Récupérer les performances depuis performance_stats
                    # Agrégation des données pour calculer wins/losses
                    query = """
                    SELECT 
                        strategy,
                        SUM(winning_trades) as total_wins,
                        SUM(losing_trades) as total_losses, 
                        SUM(total_trades) as total_signals,
                        AVG(profit_loss_percent) as avg_return,
                        MAX(end_date) as last_update
                    FROM performance_stats 
                    WHERE period = 'realtime'
                    GROUP BY strategy
                    """
                    rows = await conn.fetch(query)
                    
                    for row in rows:
                        self.performance_data[row['strategy']] = StrategyPerformance(
                            wins=row['total_wins'] or 0,
                            losses=row['total_losses'] or 0,
                            total_signals=row['total_signals'] or 0,
                            cumulative_return=row['avg_return'] or 0.0,
                            last_update=row['last_update'] or datetime.now()
                        )
                    
                    logger.info(f"📊 Chargé {len(rows)} stratégies depuis PostgreSQL")
            
            # Exécuter de manière synchrone
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(load_data())
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Erreur chargement DB: {e}")
            # Fallback sur Redis si DB échoue
            if self.redis:
                self._load_performance_data()
    
    def _save_to_database(self):
        """Sauvegarde les performances dans PostgreSQL"""
        try:
            import asyncio
            
            async def save_data():
                async with self.db_pool.acquire() as conn:
                    for strategy, perf in self.performance_data.items():
                        # Insérer/mettre à jour dans performance_stats
                        # Utiliser période 'realtime' pour les données bayésiennes
                        query = """
                        INSERT INTO performance_stats (
                            symbol, strategy, period, start_date, end_date,
                            total_trades, winning_trades, losing_trades,
                            profit_loss_percent, win_rate, created_at
                        ) VALUES (
                            'ALL', $1, 'realtime', CURRENT_DATE, CURRENT_DATE,
                            $2, $3, $4, $5, $6, NOW()
                        )
                        ON CONFLICT (symbol, strategy, period, start_date) 
                        DO UPDATE SET
                            total_trades = EXCLUDED.total_trades,
                            winning_trades = EXCLUDED.winning_trades,
                            losing_trades = EXCLUDED.losing_trades,
                            profit_loss_percent = EXCLUDED.profit_loss_percent,
                            win_rate = EXCLUDED.win_rate,
                            created_at = NOW()
                        """
                        
                        await conn.execute(
                            query,
                            strategy,
                            perf.total_signals,
                            perf.wins,
                            perf.losses,
                            perf.cumulative_return,
                            perf.win_rate * 100  # Convertir en pourcentage
                        )
                    
                    logger.debug(f"💾 Sauvegardé {len(self.performance_data)} stratégies en DB")
            
            # Exécuter de manière synchrone
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(save_data())
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Erreur sauvegarde DB: {e}")
            # Continuer avec Redis seulement
            pass