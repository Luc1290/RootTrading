"""
Module de détection de supports dynamiques pour améliorer les signaux BUY
Utilise les niveaux de prix historiques, volumes et rejets pour identifier les zones de support
"""
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class SupportDetector:
    """
    Détecteur de supports dynamiques basé sur :
    - Niveaux de prix historiques récurrents
    - Zones de volumes élevés (POC - Point of Control)
    - Rejets de prix (wicks longs vers le bas)
    - Moyennes mobiles comme supports dynamiques
    """
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.support_strength_threshold = 0.6  # Seuil de force du support
        
    def detect_support_levels(self, df: pd.DataFrame, current_price: float) -> Dict[str, any]:
        """
        Détecte les niveaux de support pour optimiser les entrées BUY
        
        Args:
            df: DataFrame avec données OHLCV
            current_price: Prix actuel
            
        Returns:
            Dict avec support_levels, nearest_support, support_strength
        """
        try:
            if len(df) < 20:
                return self._default_support_analysis(current_price)
            
            # Limiter aux dernières périodes
            df_recent = df.tail(self.lookback_periods).copy()
            
            # 1. Détecter les supports par niveaux de prix récurrents
            price_supports = self._find_price_level_supports(df_recent, current_price)
            
            # 2. Détecter les supports par volume (POC)
            volume_supports = self._find_volume_supports(df_recent, current_price)
            
            # 3. Détecter les supports par rejets (wicks)
            wick_supports = self._find_wick_supports(df_recent, current_price)
            
            # 4. Supports par moyennes mobiles
            ma_supports = self._find_moving_average_supports(df_recent, current_price)
            
            # Combiner tous les supports
            all_supports = price_supports + volume_supports + wick_supports + ma_supports
            
            # Trier par proximité avec le prix actuel
            all_supports.sort(key=lambda x: abs(x['level'] - current_price))
            
            # Trouver le support le plus proche en dessous du prix
            nearest_support = None
            support_distance = float('inf')
            
            for support in all_supports:
                if support['level'] < current_price:
                    distance = current_price - support['level']
                    if distance < support_distance:
                        nearest_support = support
                        support_distance = distance
            
            # Calculer la force du support le plus proche
            support_strength = 0.0
            if nearest_support:
                support_strength = nearest_support['strength']
                
            # Distance relative au support (en %)
            support_distance_pct = 0.0
            if nearest_support:
                support_distance_pct = (current_price - nearest_support['level']) / current_price * 100
            
            return {
                'support_levels': all_supports[:5],  # Top 5 supports
                'nearest_support': nearest_support,
                'support_strength': support_strength,
                'support_distance_pct': support_distance_pct,
                'is_near_support': support_distance_pct <= 2.0 and support_strength >= self.support_strength_threshold
            }
            
        except Exception as e:
            logger.error(f"Erreur détection supports: {e}")
            return self._default_support_analysis(current_price)
    
    def _find_price_level_supports(self, df: pd.DataFrame, current_price: float) -> List[Dict]:
        """Trouve les supports basés sur les niveaux de prix récurrents"""
        supports = []
        
        try:
            # Créer des bins de prix (résolution de 0.5%)
            price_range = df['high'].max() - df['low'].min()
            bin_size = price_range * 0.005  # 0.5% du range
            
            if bin_size <= 0:
                return supports
            
            # Compter les touches de prix par bin
            lows = df['low'].tolist()  # Plus compatible que .values
            price_bins = {}
            
            for low in lows:
                bin_center = round(low / bin_size) * bin_size
                if bin_center not in price_bins:
                    price_bins[bin_center] = {'count': 0, 'total_volume': 0}
                price_bins[bin_center]['count'] += 1
                
            # Identifier les niveaux avec plusieurs touches
            for price_level, data in price_bins.items():
                if data['count'] >= 3:  # Au moins 3 touches
                    strength = min(1.0, data['count'] / 10.0)  # Force basée sur le nombre de touches
                    
                    supports.append({
                        'level': price_level,
                        'strength': strength,
                        'type': 'price_level',
                        'touches': data['count']
                    })
                    
        except Exception as e:
            logger.error(f"Erreur détection supports prix: {e}")
            
        return supports
    
    def _find_volume_supports(self, df: pd.DataFrame, current_price: float) -> List[Dict]:
        """Trouve les supports basés sur les zones de volume élevé"""
        supports = []
        
        try:
            if 'volume' not in df.columns:
                return supports
                
            # Créer un profil de volume simplifié
            df_copy = df.copy()
            df_copy['price_mid'] = (df_copy['high'] + df_copy['low']) / 2
            
            # Diviser en bins de prix
            price_range = df_copy['high'].max() - df_copy['low'].min()
            bin_size = price_range * 0.01  # 1% du range
            
            if bin_size <= 0:
                return supports
            
            # Accumuler le volume par bin
            volume_profile = {}
            for _, row in df_copy.iterrows():
                bin_center = round(row['price_mid'] / bin_size) * bin_size
                if bin_center not in volume_profile:
                    volume_profile[bin_center] = 0
                volume_profile[bin_center] += row['volume']
            
            # Trouver les zones de volume élevé
            avg_volume = np.mean(list(volume_profile.values()))
            
            for price_level, volume in volume_profile.items():
                if volume > avg_volume * 1.5:  # 50% au-dessus de la moyenne
                    strength = min(1.0, volume / (avg_volume * 3))
                    
                    supports.append({
                        'level': price_level,
                        'strength': strength,
                        'type': 'volume_poc',
                        'volume': volume
                    })
                    
        except Exception as e:
            logger.error(f"Erreur détection supports volume: {e}")
            
        return supports
    
    def _find_wick_supports(self, df: pd.DataFrame, current_price: float) -> List[Dict]:
        """Trouve les supports basés sur les rejets (wicks longs vers le bas)"""
        supports = []
        
        try:
            df_copy = df.copy()
            
            # Calculer la taille des wicks inférieurs
            df_copy['body_size'] = abs(df_copy['close'] - df_copy['open'])
            df_copy['lower_wick'] = df_copy[['open', 'close']].min(axis=1) - df_copy['low']
            df_copy['wick_ratio'] = df_copy['lower_wick'] / df_copy['body_size']
            
            # Identifier les wicks significatifs (rejet fort)
            significant_wicks = df_copy[
                (df_copy['lower_wick'] > 0) & 
                (df_copy['wick_ratio'] > 1.5) &  # Wick > 1.5x le body
                (df_copy['lower_wick'] / df_copy['close'] > 0.005)  # Wick > 0.5% du prix
            ]
            
            # Grouper les wicks par niveau de prix
            wick_levels = {}
            for _, row in significant_wicks.iterrows():
                level = row['low']
                if level not in wick_levels:
                    wick_levels[level] = {'count': 0, 'avg_wick': 0}
                wick_levels[level]['count'] += 1
                wick_levels[level]['avg_wick'] += row['lower_wick']
            
            # Créer les supports
            for level, data in wick_levels.items():
                avg_wick = data['avg_wick'] / data['count']
                strength = min(1.0, (data['count'] * avg_wick) / (level * 0.02))  # Force relative
                
                supports.append({
                    'level': level,
                    'strength': strength,
                    'type': 'wick_rejection',
                    'rejections': data['count']
                })
                
        except Exception as e:
            logger.error(f"Erreur détection supports wicks: {e}")
            
        return supports
    
    def _find_moving_average_supports(self, df: pd.DataFrame, current_price: float) -> List[Dict]:
        """Trouve les supports basés sur les moyennes mobiles"""
        supports = []
        
        try:
            if len(df) < 50:
                return supports
            
            # Moyennes mobiles communes comme supports dynamiques
            ma_periods = [20, 50, 100, 200]
            
            for period in ma_periods:
                if len(df) >= period:
                    ma = df['close'].rolling(window=period).mean().iloc[-1]
                    
                    # Vérifier si la MA agit comme support
                    recent_touches = 0
                    for i in range(min(20, len(df))):  # 20 dernières bougies
                        low = df['low'].iloc[-(i+1)]
                        ma_at_time = df['close'].rolling(window=period).mean().iloc[-(i+1)]
                        
                        # Si le prix touche la MA sans la casser significativement
                        if abs(low - ma_at_time) / ma_at_time < 0.01 and df['close'].iloc[-(i+1)] > ma_at_time:
                            recent_touches += 1
                    
                    if recent_touches >= 2:  # Au moins 2 touches récentes
                        strength = min(1.0, recent_touches / 5.0)
                        
                        supports.append({
                            'level': ma,
                            'strength': strength,
                            'type': f'ma_{period}',
                            'touches': recent_touches
                        })
                        
        except Exception as e:
            logger.error(f"Erreur détection supports MA: {e}")
            
        return supports
    
    def _default_support_analysis(self, current_price: float) -> Dict:
        """Retourne une analyse de support par défaut"""
        return {
            'support_levels': [],
            'nearest_support': None,
            'support_strength': 0.0,
            'support_distance_pct': 100.0,
            'is_near_support': False
        }
    
    def should_boost_buy_signal(self, support_analysis: Dict, confidence: float) -> Tuple[bool, float]:
        """
        Détermine si un signal BUY doit être renforcé grâce aux supports
        
        Returns:
            (should_boost, confidence_adjustment)
        """
        try:
            if not support_analysis['is_near_support']:
                return False, 0.0
            
            support_strength = support_analysis['support_strength']
            distance_pct = support_analysis['support_distance_pct']
            
            # Boost si près d'un support fort
            if support_strength >= 0.8 and distance_pct <= 1.0:
                return True, 0.15  # +15% confidence
            elif support_strength >= 0.6 and distance_pct <= 1.5:
                return True, 0.10  # +10% confidence  
            elif support_strength >= 0.4 and distance_pct <= 2.0:
                return True, 0.05  # +5% confidence
                
            return False, 0.0
            
        except Exception as e:
            logger.error(f"Erreur boost signal BUY: {e}")
            return False, 0.0