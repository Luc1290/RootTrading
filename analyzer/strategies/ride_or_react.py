"""
Stratégie adaptative "Ride or React" qui ajuste son comportement en fonction des tendances.
Permet de "laisser courir" lors des fortes tendances et d'être plus réactif pendant les phases de consolidation.
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from shared.src.config import get_strategy_param
from shared.src.enums import OrderSide, SignalStrength
from shared.src.schemas import StrategySignal

from .base_strategy import BaseStrategy

# Configuration du logging
logger = logging.getLogger(__name__)

class RideOrReactStrategy(BaseStrategy):
    """
    Stratégie adaptative qui permet de:
    - "Ride": Laisser courir les positions pendant les tendances fortes
    - "React": Être plus réactif pendant les phases de consolidation ou retournement
    
    Cette stratégie agit comme un filtre sur les autres signaux, plutôt que de générer ses propres signaux directement.
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        """
        Initialise la stratégie Ride or React.
        
        Args:
            symbol: Symbole de trading (ex: 'BTCUSDC')
            params: Paramètres spécifiques à la stratégie
        """
        super().__init__(symbol, params)
        
        # Seuils de variation pour les différentes timeframes
        self.thresholds = {
            '1h': self.params.get('threshold_1h', get_strategy_param('ride_or_react', 'thresholds.1h', 0.8)),
            '3h': self.params.get('threshold_3h', get_strategy_param('ride_or_react', 'thresholds.3h', 2.5)),
            '6h': self.params.get('threshold_6h', get_strategy_param('ride_or_react', 'thresholds.6h', 3.6)),
            '12h': self.params.get('threshold_12h', get_strategy_param('ride_or_react', 'thresholds.12h', 5.1)),
            '24h': self.params.get('threshold_24h', get_strategy_param('ride_or_react', 'thresholds.24h', 7.8))
        }
        
        # État actuel de la stratégie
        self.current_mode = "react"  # Par défaut, être réactif
        self.last_evaluation_time = None
        self.current_market_condition = None
        
        # Buffer de prix pour différentes timeframes
        self.price_history = {}
        self.timestamps = []
        
        logger.info(f"🔧 Stratégie Ride or React initialisée pour {symbol} "
                   f"(seuils: 1h={self.thresholds['1h']}%, 24h={self.thresholds['24h']}%)")
    
    @property
    def name(self) -> str:
        """Nom unique de la stratégie."""
        return "Ride_or_React_Strategy"
    
    def get_min_data_points(self) -> int:
        """
        Nombre minimum de points de données nécessaires.
        
        Returns:
            Nombre minimum de données requises
        """
        # Besoin d'au moins 24h de données (en fonction de l'intervalle)
        return 1440 // self.get_interval_minutes()
    
    def get_interval_minutes(self) -> int:
        """
        Détermine l'intervalle en minutes des données reçues.
        
        Returns:
            Intervalle en minutes
        """
        # Tenter de déterminer l'intervalle à partir des données
        df = self.get_data_as_dataframe()
        if df is not None and len(df) >= 2:
            # Calculer l'intervalle moyen entre deux points de données
            time_diffs = np.diff(df.index.astype(np.int64)) // 10**9 // 60  # Différence en minutes
            return int(np.median(time_diffs))
        
        # Valeur par défaut si impossible à déterminer
        return 1  # Supposer 1 minute par défaut
    
    def add_market_data(self, data: Dict[str, Any]) -> None:
        """
        Ajoute des données de marché au buffer et met à jour l'historique des prix.
        
        Args:
            data: Données de marché à ajouter
        """
        # Ajouter aux données générales
        super().add_market_data(data)
        
        # Ne traiter que les chandeliers fermés
        if not data.get('is_closed', False):
            return
        
        # Extraire les informations
        timestamp = data.get('start_time')
        price = data.get('close')
        
        if timestamp and price:
            # Convertir le timestamp en datetime
            dt = datetime.fromtimestamp(timestamp / 1000)
            
            # Stocker le prix et le timestamp
            self.timestamps.append(dt)
            self.price_history[dt] = price
            
            # Nettoyer les anciennes données (plus de 24h)
            cutoff = dt - timedelta(hours=24)
            self.timestamps = [ts for ts in self.timestamps if ts >= cutoff]
            self.price_history = {ts: p for ts, p in self.price_history.items() if ts >= cutoff}
    
    def _get_price_change(self, hours: int) -> Tuple[float, float]:
        """
        Calcule la variation de prix sur une période donnée.
        
        Args:
            hours: Nombre d'heures dans le passé
            
        Returns:
            Tuple (variation en pourcentage, prix actuel)
        """
        if not self.timestamps:
            return 0.0, 0.0
        
        # Obtenir le prix actuel
        current_time = max(self.timestamps)
        current_price = self.price_history[current_time]
        
        # Déterminer l'heure de référence
        reference_time = current_time - timedelta(hours=hours)
        
        # Trouver le prix le plus proche de l'heure de référence
        closest_time = min(self.timestamps, key=lambda x: abs((x - reference_time).total_seconds()))
        
        # Si le temps le plus proche est trop éloigné, retourner 0
        if abs((closest_time - reference_time).total_seconds()) > hours * 3600 * 0.5:  # Plus de la moitié de la période
            return 0.0, current_price
        
        reference_price = self.price_history[closest_time]
        
        # Calculer la variation en pourcentage
        percent_change = ((current_price - reference_price) / reference_price) * 100
        
        return percent_change, current_price
    
    def _evaluate_market_condition(self) -> Dict[str, Any]:
        """
        Évalue les conditions actuelles du marché sur plusieurs timeframes.
        
        Returns:
            Dictionnaire avec les conditions de marché
        """
        results = {}
        current_price = 0.0
        
        # Évaluer les variations sur différentes timeframes
        for hours, key in [(1, '1h'), (3, '3h'), (6, '6h'), (12, '12h'), (24, '24h')]:
            percent_change, price = self._get_price_change(hours)
            current_price = price  # Mettre à jour le prix actuel
            threshold = self.thresholds[key]
            
            results[f"{key}_change"] = percent_change
            results[f"{key}_threshold"] = threshold
            results[f"{key}_exceeded"] = percent_change >= threshold
        
        # Déterminer le mode global
        ride_conditions_met = sum(results[f"{key}_exceeded"] for key in ['1h', '3h', '6h', '12h', '24h'])
        
        # Si au moins deux timeframes montrent une tendance forte, passer en mode "ride"
        if ride_conditions_met >= 2:
            results["mode"] = "ride"
            results["action"] = "wait_for_reversal"
        else:
            results["mode"] = "react"
            results["action"] = "normal_trading"
        
        # Stocker le prix actuel
        results["current_price"] = current_price
        
        return results
    
    def generate_signal(self) -> Optional[StrategySignal]:
        """
        Évalue les conditions de marché et décide du mode (ride ou react).
        Cette stratégie ne génère pas directement de signaux d'achat/vente,
        mais peut être utilisée pour filtrer les signaux d'autres stratégies.
        
        Returns:
            Signal de mode (metadata contient le mode) ou None
        """
        # Vérifier si nous avons assez de données
        if len(self.timestamps) < 12:  # Au moins 12 points de données
            return None
        
        # Évaluer le marché au maximum toutes les 15 minutes
        now = datetime.now()
        if self.last_evaluation_time and (now - self.last_evaluation_time).total_seconds() < 900:
            return None
        
        # Évaluer les conditions de marché
        market_condition = self._evaluate_market_condition()
        self.current_market_condition = market_condition
        self.current_mode = market_condition["mode"]
        self.last_evaluation_time = now
        
        # Loguer le changement de mode si pertinent
        logger.info(f"🌊 [Ride or React] {self.symbol}: Mode={self.current_mode}, "
                   f"1h={market_condition['1h_change']:.2f}%, "
                   f"24h={market_condition['24h_change']:.2f}%")
        
        # Créer un signal "informatif" qui peut être utilisé par d'autres stratégies
        # Ce n'est pas un signal d'achat/vente, mais un signal de contexte du marché
        return self.create_signal(
            side=OrderSide.BUY if market_condition["mode"] == "ride" else OrderSide.SELL,
            price=market_condition["current_price"],
            confidence=0.8,
            metadata=market_condition
        )
    
    def should_filter_signal(self, signal: StrategySignal) -> bool:
        """
        Détermine si un signal d'une autre stratégie doit être filtré en fonction du mode actuel.
        
        Args:
            signal: Signal à évaluer
            
        Returns:
            True si le signal doit être filtré (ignoré), False sinon
        """
        if not self.current_market_condition:
            return False  # Pas assez d'informations pour filtrer
        
        # En mode "ride", filtrer les signaux de vente si le marché est en tendance haussière
        if self.current_mode == "ride":
            # Si le marché est en forte hausse et le signal est SELL, filtrer
            is_uptrend = self.current_market_condition.get('24h_change', 0) > 0
            
            if is_uptrend and signal.side == OrderSide.SELL:
                logger.info(f"🔍 [Ride or React] Filtrage d'un signal SELL en mode RIDE (tendance haussière)")
                return True
            
            # Si le marché est en forte baisse et le signal est BUY, filtrer
            if not is_uptrend and signal.side == OrderSide.BUY:
                logger.info(f"🔍 [Ride or React] Filtrage d'un signal BUY en mode RIDE (tendance baissière)")
                return True
        
        # En mode "react", ne pas filtrer les signaux
        return False