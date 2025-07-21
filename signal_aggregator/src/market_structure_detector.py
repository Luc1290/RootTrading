#!/usr/bin/env python3
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import asyncio
from enum import Enum
from dataclasses import dataclass
from shared.src.technical_indicators import TechnicalIndicators
from .shared.redis_utils import RedisManager

logger = logging.getLogger(__name__)


class StructureType(Enum):
    """Types de structure de marché"""
    HIGHER_HIGHS_HIGHER_LOWS = "HH_HL"      # Tendance haussière forte
    HIGHER_HIGHS_LOWER_LOWS = "HH_LL"       # Divergence (faiblesse)
    LOWER_HIGHS_HIGHER_LOWS = "LH_HL"       # Consolidation/triangle
    LOWER_HIGHS_LOWER_LOWS = "LH_LL"        # Tendance baissière forte
    SIDEWAYS = "SIDEWAYS"                   # Range horizontal
    UNDEFINED = "UNDEFINED"


@dataclass 
class KeyLevel:
    """Niveau clé (support/résistance)"""
    price: float
    level_type: str        # 'support', 'resistance', 'pivot'
    strength: float        # Force du niveau (0-1)
    timeframe: str         # Timeframe d'origine
    touches: int           # Nombre de touches
    last_touch: datetime   # Dernière interaction
    volume_confirmation: bool
    broken: bool = False


@dataclass
class LiquidityZone:
    """Zone de liquidité"""
    price_min: float
    price_max: float
    zone_type: str         # 'buy_liquidity', 'sell_liquidity', 'stop_hunt'
    strength: float        # Force de la zone
    timeframe: str
    volume_cluster: float  # Volume agrégé dans la zone
    last_interaction: datetime


@dataclass
class FractalPattern:
    """Pattern fractal détecté"""
    pattern_type: str      # 'swing_high', 'swing_low', 'double_top', 'double_bottom', etc.
    price_levels: List[float]
    timeframe: str
    confidence: float      # Confiance dans le pattern
    completion_time: datetime
    target_levels: List[float]  # Niveaux cibles


@dataclass
class MarketStructureAnalysis:
    """Analyse complète de structure de marché"""
    structure_type: StructureType
    trend_strength: float          # Force de tendance (0-1)
    key_levels: List[KeyLevel]     # Niveaux S/R importants
    liquidity_zones: List[LiquidityZone]
    fractal_patterns: List[FractalPattern]
    structure_score: float         # Score global de structure (0-100)
    bias: str                      # 'bullish', 'bearish', 'neutral'
    next_targets: List[float]      # Prochains objectifs probables
    risk_zones: List[Tuple[float, float]]  # Zones de risque élevé


class MarketStructureDetector:
    """Détecteur avancé de structure de marché multi-timeframes"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.indicators = TechnicalIndicators()
        
        # Configuration des timeframes pour l'analyse de structure (adapté aux TF disponibles en DB)
        self.structure_timeframes = {
            '1m': {'weight': 0.10, 'lookback_periods': 100},   # Court terme - bruit
            '3m': {'weight': 0.20, 'lookback_periods': 150},   # Court terme - filtrage
            '5m': {'weight': 0.30, 'lookback_periods': 200},   # Moyen terme - principal
            '15m': {'weight': 0.35, 'lookback_periods': 300},  # Long terme - tendance
            # Timeframes indisponibles en DB: '1h', '4h', '1d' remplacés par TF existants
        }
        
        # Paramètres de détection
        self.swing_detection_window = 10  # Périodes pour détecter les swings
        self.min_swing_strength = 0.01   # 1% minimum pour considérer un swing
        self.support_resistance_threshold = 0.005  # 0.5% pour grouper les niveaux
        self.volume_confirmation_threshold = 1.2   # 20% volume au-dessus de la moyenne
        
    async def analyze_market_structure(self, symbol: str) -> MarketStructureAnalysis:
        """
        Analyse la structure de marché complète pour un symbole
        
        Returns:
            MarketStructureAnalysis avec tous les éléments structurels
        """
        try:
            # Vérifier le cache avec utilitaire partagé
            cache_key = f"market_structure:{symbol}"
            cached_data = RedisManager.get_cached_data(self.redis, cache_key)
            
            if cached_data:
                return self._deserialize_structure_analysis(cached_data)
            
            # Récupérer les données multi-timeframes
            timeframe_data = await self._get_multi_timeframe_data(symbol)
            
            if not timeframe_data:
                logger.warning(f"⚠️ Pas de données pour analyse structure {symbol}")
                return self._create_default_analysis(symbol)
            
            # 1. Détecter la structure générale (HH/HL/LH/LL)
            structure_type = self._detect_market_structure_type(timeframe_data)
            
            # 2. Identifier les niveaux clés (S/R)
            key_levels = self._identify_key_levels(timeframe_data)
            
            # 3. Détecter les zones de liquidité
            liquidity_zones = self._detect_liquidity_zones(timeframe_data)
            
            # 4. Analyser les patterns fractals
            fractal_patterns = self._analyze_fractal_patterns(timeframe_data)
            
            # 5. Calculer les métriques globales
            trend_strength = self._calculate_trend_strength(timeframe_data, structure_type)
            structure_score = self._calculate_structure_score(key_levels, liquidity_zones, fractal_patterns)
            bias = self._determine_market_bias(structure_type, key_levels, timeframe_data)
            
            # 6. Générer les cibles et zones de risque
            next_targets = self._calculate_next_targets(key_levels, structure_type, timeframe_data)
            risk_zones = self._identify_risk_zones(key_levels, liquidity_zones)
            
            analysis = MarketStructureAnalysis(
                structure_type=structure_type,
                trend_strength=trend_strength,
                key_levels=key_levels,
                liquidity_zones=liquidity_zones,
                fractal_patterns=fractal_patterns,
                structure_score=structure_score,
                bias=bias,
                next_targets=next_targets,
                risk_zones=risk_zones
            )
            
            # Mettre en cache pour 2 minutes
            cache_data = self._serialize_structure_analysis(analysis)
            self.redis.set(cache_key, json.dumps(cache_data), expiration=120)
            
            logger.info(f"📈 Structure {symbol}: {structure_type.value} | "
                       f"Bias: {bias} | Score: {structure_score:.1f} | "
                       f"Levels: {len(key_levels)} | Liquidity: {len(liquidity_zones)}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse structure pour {symbol}: {e}")
            return self._create_default_analysis(symbol)
    
    async def _get_multi_timeframe_data(self, symbol: str) -> Dict[str, List[Dict]]:
        """Récupère les données historiques multi-timeframes"""
        try:
            timeframe_data = {}
            
            for tf, config in self.structure_timeframes.items():
                # Essayer de récupérer les données historiques
                historical_data = await self._get_historical_data(symbol, tf, config['lookback_periods'])
                
                if historical_data and len(historical_data) >= 50:
                    timeframe_data[tf] = historical_data
                else:
                    logger.debug(f"Données insuffisantes pour {tf}: {len(historical_data) if historical_data else 0}")
            
            return timeframe_data
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération données multi-timeframes: {e}")
            return {}
    
    async def _get_historical_data(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        """Récupère les données historiques pour un timeframe"""
        try:
            # 1. Essayer Redis d'abord (données enrichies) avec utilitaire partagé
            key = f"historical:{symbol}:{timeframe}"
            data = RedisManager.get_cached_data(self.redis, key)
            
            if data:
                
                if isinstance(data, list) and len(data) >= limit // 2:
                    return data[-limit:]
                else:
                    logger.debug(f"Redis insuffisant pour {symbol}:{timeframe} ({len(data) if data else 0}/{limit})")
            
            # 2. Fallback vers la base de données
            logger.info(f"💾 Fallback DB pour {symbol}:{timeframe} (limit: {limit})")
            db_data = await self._get_historical_from_db(symbol, timeframe, limit)
            
            if db_data and len(db_data) >= 10:  # Au moins 10 points pour une analyse valide
                logger.info(f"✅ DB: {len(db_data)} points récupérés pour {symbol}:{timeframe}")
                return db_data[-limit:]  # Retourner les plus récents
            
            # 3. Données vraiment insuffisantes
            logger.warning(f"⚠️ Données insuffisantes pour {symbol}:{timeframe} (Redis: {len(historical) if historical else 0}, DB: {len(db_data) if db_data else 0})")
            return []
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération historique {timeframe}: {e}")
            return []
    
    async def _get_historical_from_db(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        """Récupère les données historiques depuis la base de données"""
        try:
            from shared.src.db_pool import fetch_all
            
            # Requête pour récupérer les données enrichies
            query = """
                SELECT 
                    EXTRACT(epoch FROM time) as timestamp,
                    time, symbol, open, high, low, close, volume,
                    rsi_14, ema_7, ema_26, ema_99, sma_20, sma_50,
                    macd_line, macd_signal, macd_histogram,
                    bb_upper, bb_middle, bb_lower, bb_position, bb_width,
                    atr_14, adx_14, plus_di, minus_di,
                    momentum_10, volume_ratio, avg_volume_20
                FROM market_data 
                WHERE symbol = %s AND timeframe = %s AND enhanced = true
                ORDER BY time DESC 
                LIMIT %s
            """
            
            # Exécuter la requête
            rows = fetch_all(query, (symbol, timeframe, limit), dict_result=True)
            
            if rows:
                # Convertir en format attendu (timestamp Unix)
                data = []
                for row in rows:
                    # Convertir les Decimal en float pour JSON
                    row_dict = {}
                    for key, value in row.items():
                        if value is not None:
                            if hasattr(value, '__float__'):  # Decimal, float, etc.
                                row_dict[key] = float(value)
                            else:
                                row_dict[key] = value
                        else:
                            row_dict[key] = value
                    data.append(row_dict)
                
                # Inverser pour avoir les plus anciens en premier
                return list(reversed(data))
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération DB pour {symbol}:{timeframe}: {e}")
            return []
    
    
    def _detect_market_structure_type(self, timeframe_data: Dict[str, List[Dict]]) -> StructureType:
        """Détecte le type de structure de marché global"""
        try:
            structure_votes = []
            
            for tf, data in timeframe_data.items():
                if len(data) < 20:
                    continue
                
                # Détecter les swings pour ce timeframe
                swings = self._detect_swing_points(data)
                
                if len(swings) < 4:
                    continue
                
                # Analyser la progression des highs et lows
                structure = self._analyze_swing_progression(swings)
                weight = self.structure_timeframes.get(tf, {}).get('weight', 1.0)
                
                structure_votes.append((structure, weight))
            
            if not structure_votes:
                return StructureType.UNDEFINED
            
            # Vote pondéré
            weighted_votes = {}
            for structure, weight in structure_votes:
                if structure not in weighted_votes:
                    weighted_votes[structure] = 0
                weighted_votes[structure] += weight
            
            # Retourner la structure avec le plus de votes
            return max(weighted_votes, key=lambda x: weighted_votes[x])
            
        except Exception as e:
            logger.error(f"❌ Erreur détection type structure: {e}")
            return StructureType.UNDEFINED
    
    def _detect_swing_points(self, data: List[Dict]) -> List[Dict]:
        """Détecte les points de swing (pivots) dans les données"""
        try:
            if len(data) < self.swing_detection_window * 2:
                return []
            
            swings = []
            window = self.swing_detection_window
            
            for i in range(window, len(data) - window):
                current_high = data[i]['high']
                current_low = data[i]['low']
                current_time = data[i]['timestamp']
                
                # Vérifier si c'est un swing high
                is_swing_high = True
                for j in range(i - window, i + window + 1):
                    if j != i and data[j]['high'] >= current_high:
                        is_swing_high = False
                        break
                
                if is_swing_high:
                    swings.append({
                        'type': 'high',
                        'price': current_high,
                        'timestamp': current_time,
                        'index': i
                    })
                
                # Vérifier si c'est un swing low
                is_swing_low = True
                for j in range(i - window, i + window + 1):
                    if j != i and data[j]['low'] <= current_low:
                        is_swing_low = False
                        break
                
                if is_swing_low:
                    swings.append({
                        'type': 'low',
                        'price': current_low,
                        'timestamp': current_time,
                        'index': i
                    })
            
            # Trier par timestamp
            swings.sort(key=lambda x: x['timestamp'])
            return swings
            
        except Exception as e:
            logger.error(f"❌ Erreur détection swing points: {e}")
            return []
    
    def _analyze_swing_progression(self, swings: List[Dict]) -> StructureType:
        """Analyse la progression des swings pour déterminer la structure"""
        try:
            if len(swings) < 4:
                return StructureType.UNDEFINED
            
            # Séparer les highs et lows
            highs = [s for s in swings if s['type'] == 'high']
            lows = [s for s in swings if s['type'] == 'low']
            
            if len(highs) < 2 or len(lows) < 2:
                return StructureType.UNDEFINED
            
            # Analyser la tendance des highs
            high_trend = self._calculate_price_trend([h['price'] for h in highs[-3:]])
            
            # Analyser la tendance des lows  
            low_trend = self._calculate_price_trend([low['price'] for low in lows[-3:]])
            
            # Déterminer la structure
            if high_trend > 0.01 and low_trend > 0.01:  # HH et HL
                return StructureType.HIGHER_HIGHS_HIGHER_LOWS
            elif high_trend > 0.01 and low_trend < -0.01:  # HH et LL
                return StructureType.HIGHER_HIGHS_LOWER_LOWS
            elif high_trend < -0.01 and low_trend > 0.01:  # LH et HL
                return StructureType.LOWER_HIGHS_HIGHER_LOWS
            elif high_trend < -0.01 and low_trend < -0.01:  # LH et LL
                return StructureType.LOWER_HIGHS_LOWER_LOWS
            else:
                return StructureType.SIDEWAYS
                
        except Exception as e:
            logger.error(f"❌ Erreur analyse progression swings: {e}")
            return StructureType.UNDEFINED
    
    def _calculate_price_trend(self, prices: List[float]) -> float:
        """Calcule la tendance des prix (pente normalisée)"""
        try:
            if len(prices) < 2:
                return 0.0
            
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # Normaliser par le prix moyen
            avg_price = np.mean(prices)
            return slope / avg_price if avg_price > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul tendance prix: {e}")
            return 0.0
    
    def _identify_key_levels(self, timeframe_data: Dict[str, List[Dict]]) -> List[KeyLevel]:
        """Identifie les niveaux clés de support/résistance"""
        try:
            all_levels = []
            
            for tf, data in timeframe_data.items():
                if len(data) < 20:
                    continue
                
                # Détecter les niveaux pour ce timeframe
                tf_levels = self._find_support_resistance_levels(data, tf)
                all_levels.extend(tf_levels)
            
            # Fusionner les niveaux proches
            merged_levels = self._merge_nearby_levels(all_levels)
            
            # Trier par force
            merged_levels.sort(key=lambda x: x.strength, reverse=True)
            
            # Retourner les 10 meilleurs niveaux
            return merged_levels[:10]
            
        except Exception as e:
            logger.error(f"❌ Erreur identification niveaux clés: {e}")
            return []
    
    def _find_support_resistance_levels(self, data: List[Dict], timeframe: str) -> List[KeyLevel]:
        """Trouve les niveaux S/R dans un dataset"""
        try:
            levels = []
            swings = self._detect_swing_points(data)
            
            # Grouper les swings par prix similaire
            price_clusters: Dict[float, List[Dict]] = {}
            threshold = self.support_resistance_threshold
            
            for swing in swings:
                price = swing['price']
                found_cluster = False
                
                for cluster_price in price_clusters:
                    if abs(price - cluster_price) / cluster_price < threshold:
                        price_clusters[cluster_price].append(swing)
                        found_cluster = True
                        break
                
                if not found_cluster:
                    price_clusters[price] = [swing]
            
            # Convertir les clusters en niveaux
            for cluster_price, swings_in_cluster in price_clusters.items():
                if len(swings_in_cluster) >= 2:  # Au moins 2 touches
                    level_type = 'resistance' if swings_in_cluster[0]['type'] == 'high' else 'support'
                    
                    # Calculer la force
                    touches = len(swings_in_cluster)
                    strength = min(1.0, touches / 5.0)  # Max force à 5 touches
                    
                    # Volume confirmation (simplifié)
                    volume_confirmation = touches >= 3
                    
                    level = KeyLevel(
                        price=cluster_price,
                        level_type=level_type,
                        strength=strength,
                        timeframe=timeframe,
                        touches=touches,
                        last_touch=datetime.fromtimestamp(swings_in_cluster[-1]['timestamp']),
                        volume_confirmation=volume_confirmation
                    )
                    
                    levels.append(level)
            
            return levels
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche niveaux S/R: {e}")
            return []
    
    def _merge_nearby_levels(self, levels: List[KeyLevel]) -> List[KeyLevel]:
        """Fusionne les niveaux proches entre timeframes"""
        try:
            if not levels:
                return []
            
            merged = []
            threshold = self.support_resistance_threshold
            
            # Trier par prix
            levels.sort(key=lambda x: x.price)
            
            current_group = [levels[0]]
            
            for level in levels[1:]:
                # Vérifier si le niveau est proche du groupe actuel
                group_avg_price = sum(level.price for level in current_group) / len(current_group)
                
                if abs(level.price - group_avg_price) / group_avg_price < threshold:
                    current_group.append(level)
                else:
                    # Fusionner le groupe actuel
                    merged_level = self._merge_level_group(current_group)
                    if merged_level is not None:
                        merged.append(merged_level)
                    current_group = [level]
            
            # Fusionner le dernier groupe
            if current_group:
                merged_level = self._merge_level_group(current_group)
                if merged_level is not None:
                    merged.append(merged_level)
            
            return merged
            
        except Exception as e:
            logger.error(f"❌ Erreur fusion niveaux: {e}")
            return levels  # Retourner les originaux en cas d'erreur
    
    def _merge_level_group(self, group: List[KeyLevel]) -> Optional[KeyLevel]:
        """Fusionne un groupe de niveaux similaires"""
        try:
            if not group:
                return None
            
            if len(group) == 1:
                return group[0]
            
            # Prix moyen pondéré par la force
            total_weight = sum(level.strength for level in group)
            weighted_price = sum(level.price * level.strength for level in group) / total_weight
            
            # Prendre le type le plus fréquent
            types = [level.level_type for level in group]
            level_type = max(set(types), key=types.count)
            
            # Force combinée
            combined_strength = min(1.0, sum(level.strength for level in group))
            
            # Timeframes combinés
            timeframes = list(set(level.timeframe for level in group))
            main_timeframe = min(timeframes, key=lambda tf: self.structure_timeframes.get(tf, {}).get('weight', 1.0))
            
            # Touches totales
            total_touches = sum(level.touches for level in group)
            
            # Dernière interaction
            last_touch = max(level.last_touch for level in group)
            
            # Volume confirmation si au moins un l'a
            volume_confirmation = any(level.volume_confirmation for level in group)
            
            return KeyLevel(
                price=weighted_price,
                level_type=level_type,
                strength=combined_strength,
                timeframe=main_timeframe,
                touches=total_touches,
                last_touch=last_touch,
                volume_confirmation=volume_confirmation
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur fusion groupe niveaux: {e}")
            return group[0] if group else None
    
    def _detect_liquidity_zones(self, timeframe_data: Dict[str, List[Dict]]) -> List[LiquidityZone]:
        """Détecte les zones de liquidité"""
        try:
            # Implémentation simplifiée pour les zones de liquidité
            # Dans une vraie implémentation, on analyserait les ordres de volume,
            # les gaps, les niveaux cassés rapidement, etc.
            
            liquidity_zones = []
            
            for tf, data in timeframe_data.items():
                if len(data) < 50:
                    continue
                
                # Détecter les zones de volume élevé
                volumes = [candle['volume'] for candle in data]
                avg_volume = np.mean(volumes)
                volume_threshold = avg_volume * 2  # 2x volume moyen
                
                high_volume_periods = []
                for i, candle in enumerate(data):
                    if candle['volume'] > volume_threshold:
                        high_volume_periods.append(i)
                
                # Grouper les périodes de volume élevé consécutives
                if high_volume_periods:
                    clusters = self._group_consecutive_periods(high_volume_periods)
                    
                    for cluster in clusters:
                        if len(cluster) >= 2:  # Au moins 2 périodes consécutives
                            start_idx = min(cluster)
                            end_idx = max(cluster)
                            
                            price_min = min(data[i]['low'] for i in cluster)
                            price_max = max(data[i]['high'] for i in cluster)
                            volume_cluster = sum(data[i]['volume'] for i in cluster)
                            
                            # Déterminer le type de zone
                            price_movement = data[end_idx]['close'] - data[start_idx]['open']
                            zone_type = 'buy_liquidity' if price_movement > 0 else 'sell_liquidity'
                            
                            strength = min(1.0, volume_cluster / (avg_volume * len(cluster) * 3))
                            
                            zone = LiquidityZone(
                                price_min=price_min,
                                price_max=price_max,
                                zone_type=zone_type,
                                strength=strength,
                                timeframe=tf,
                                volume_cluster=volume_cluster,
                                last_interaction=datetime.fromtimestamp(data[end_idx]['timestamp'])
                            )
                            
                            liquidity_zones.append(zone)
            
            # Trier par force
            liquidity_zones.sort(key=lambda x: x.strength, reverse=True)
            return liquidity_zones[:5]  # Top 5 zones
            
        except Exception as e:
            logger.error(f"❌ Erreur détection zones liquidité: {e}")
            return []
    
    def _group_consecutive_periods(self, periods: List[int]) -> List[List[int]]:
        """Groupe les périodes consécutives"""
        if not periods:
            return []
        
        groups = []
        current_group = [periods[0]]
        
        for i in range(1, len(periods)):
            if periods[i] == periods[i-1] + 1:
                current_group.append(periods[i])
            else:
                groups.append(current_group)
                current_group = [periods[i]]
        
        groups.append(current_group)
        return groups
    
    def _analyze_fractal_patterns(self, timeframe_data: Dict[str, List[Dict]]) -> List[FractalPattern]:
        """Analyse les patterns fractals"""
        try:
            # Implémentation simplifiée des patterns fractals
            patterns = []
            
            for tf, data in timeframe_data.items():
                swings = self._detect_swing_points(data)
                
                # Détecter des patterns simples (double top/bottom)
                detected_patterns = self._detect_double_patterns(swings, tf)
                patterns.extend(detected_patterns)
            
            return patterns[:3]  # Top 3 patterns
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse patterns fractals: {e}")
            return []
    
    def _detect_double_patterns(self, swings: List[Dict], timeframe: str) -> List[FractalPattern]:
        """Détecte les patterns double top/bottom"""
        try:
            patterns = []
            
            # Séparer highs et lows
            highs = [s for s in swings if s['type'] == 'high']
            lows = [s for s in swings if s['type'] == 'low']
            
            # Détecter double tops
            for i in range(len(highs) - 1):
                for j in range(i + 1, len(highs)):
                    price_diff = abs(highs[i]['price'] - highs[j]['price']) / highs[i]['price']
                    
                    if price_diff < 0.02:  # 2% tolerance
                        confidence = 1.0 - price_diff  # Plus c'est proche, plus confiant
                        
                        pattern = FractalPattern(
                            pattern_type='double_top',
                            price_levels=[highs[i]['price'], highs[j]['price']],
                            timeframe=timeframe,
                            confidence=confidence,
                            completion_time=datetime.fromtimestamp(highs[j]['timestamp']),
                            target_levels=[min(highs[i]['price'], highs[j]['price']) * 0.95]  # Target -5%
                        )
                        patterns.append(pattern)
            
            # Détecter double bottoms
            for i in range(len(lows) - 1):
                for j in range(i + 1, len(lows)):
                    price_diff = abs(lows[i]['price'] - lows[j]['price']) / lows[i]['price']
                    
                    if price_diff < 0.02:  # 2% tolerance
                        confidence = 1.0 - price_diff
                        
                        pattern = FractalPattern(
                            pattern_type='double_bottom',
                            price_levels=[lows[i]['price'], lows[j]['price']],
                            timeframe=timeframe,
                            confidence=confidence,
                            completion_time=datetime.fromtimestamp(lows[j]['timestamp']),
                            target_levels=[max(lows[i]['price'], lows[j]['price']) * 1.05]  # Target +5%
                        )
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Erreur détection double patterns: {e}")
            return []
    
    def _calculate_trend_strength(self, timeframe_data: Dict[str, List[Dict]], structure_type: StructureType) -> float:
        """Calcule la force de tendance globale"""
        try:
            if structure_type in [StructureType.HIGHER_HIGHS_HIGHER_LOWS, StructureType.LOWER_HIGHS_LOWER_LOWS]:
                return 0.8  # Tendances fortes
            elif structure_type in [StructureType.HIGHER_HIGHS_LOWER_LOWS, StructureType.LOWER_HIGHS_HIGHER_LOWS]:
                return 0.3  # Tendances faibles/divergences
            else:
                return 0.1  # Sideways
                
        except Exception as e:
            logger.error(f"❌ Erreur calcul force tendance: {e}")
            return 0.0
    
    def _calculate_structure_score(self, key_levels: List[KeyLevel], 
                                   liquidity_zones: List[LiquidityZone], 
                                   fractal_patterns: List[FractalPattern]) -> float:
        """Calcule le score global de structure"""
        try:
            score = 0.0
            
            # Score des niveaux clés
            level_score = sum(level.strength for level in key_levels[:5]) * 20
            score += level_score
            
            # Score des zones de liquidité
            liquidity_score = sum(zone.strength for zone in liquidity_zones[:3]) * 15
            score += liquidity_score
            
            # Score des patterns
            pattern_score = sum(pattern.confidence for pattern in fractal_patterns) * 10
            score += pattern_score
            
            return min(100.0, score)
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul score structure: {e}")
            return 0.0
    
    def _determine_market_bias(self, structure_type: StructureType, 
                               key_levels: List[KeyLevel], 
                               timeframe_data: Dict[str, List[Dict]]) -> str:
        """Détermine le biais de marché"""
        try:
            if structure_type == StructureType.HIGHER_HIGHS_HIGHER_LOWS:
                return 'bullish'
            elif structure_type == StructureType.LOWER_HIGHS_LOWER_LOWS:
                return 'bearish'
            else:
                # Analyser les niveaux pour déterminer le biais
                resistance_levels = [level for level in key_levels if level.level_type == 'resistance']
                support_levels = [level for level in key_levels if level.level_type == 'support']
                
                if len(support_levels) > len(resistance_levels):
                    return 'bullish'
                elif len(resistance_levels) > len(support_levels):
                    return 'bearish'
                else:
                    return 'neutral'
                    
        except Exception as e:
            logger.error(f"❌ Erreur détermination biais: {e}")
            return 'neutral'
    
    def _calculate_next_targets(self, key_levels: List[KeyLevel], 
                                structure_type: StructureType, 
                                timeframe_data: Dict[str, List[Dict]]) -> List[float]:
        """Calcule les prochains objectifs probables"""
        try:
            targets = []
            
            # Utiliser les niveaux clés comme targets
            for level in key_levels[:5]:
                targets.append(level.price)
            
            # Ajouter des targets basés sur la structure
            if timeframe_data:
                # Prendre le dernier prix disponible
                latest_data = list(timeframe_data.values())[0]
                if latest_data:
                    current_price = latest_data[-1]['close']
                    
                    if structure_type == StructureType.HIGHER_HIGHS_HIGHER_LOWS:
                        # Projeter des targets haussiers
                        targets.extend([current_price * 1.05, current_price * 1.10])
                    elif structure_type == StructureType.LOWER_HIGHS_LOWER_LOWS:
                        # Projeter des targets baissiers
                        targets.extend([current_price * 0.95, current_price * 0.90])
            
            # Éliminer les doublons et trier
            unique_targets = list(set(targets))
            unique_targets.sort()
            
            return unique_targets[:5]  # Max 5 targets
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul targets: {e}")
            return []
    
    def _identify_risk_zones(self, key_levels: List[KeyLevel], 
                             liquidity_zones: List[LiquidityZone]) -> List[Tuple[float, float]]:
        """Identifie les zones de risque élevé"""
        try:
            risk_zones = []
            
            # Zones autour des niveaux clés forts
            for level in key_levels[:3]:
                if level.strength > 0.7:
                    zone_width = level.price * 0.01  # 1% de part et d'autre
                    risk_zones.append((
                        level.price - zone_width,
                        level.price + zone_width
                    ))
            
            # Zones de liquidité comme zones de risque
            for zone in liquidity_zones[:2]:
                if zone.strength > 0.6:
                    risk_zones.append((zone.price_min, zone.price_max))
            
            return risk_zones
            
        except Exception as e:
            logger.error(f"❌ Erreur identification zones risque: {e}")
            return []
    
    def _create_default_analysis(self, symbol: str) -> MarketStructureAnalysis:
        """Crée une analyse par défaut"""
        return MarketStructureAnalysis(
            structure_type=StructureType.UNDEFINED,
            trend_strength=0.0,
            key_levels=[],
            liquidity_zones=[],
            fractal_patterns=[],
            structure_score=0.0,
            bias='neutral',
            next_targets=[],
            risk_zones=[]
        )
    
    def _serialize_structure_analysis(self, analysis: MarketStructureAnalysis) -> Dict:
        """Sérialise l'analyse pour le cache"""
        # Implémentation simplifiée pour le cache
        return {
            'structure_type': analysis.structure_type.value,
            'trend_strength': analysis.trend_strength,
            'structure_score': analysis.structure_score,
            'bias': analysis.bias,
            'next_targets': analysis.next_targets,
            'key_levels_count': len(analysis.key_levels),
            'liquidity_zones_count': len(analysis.liquidity_zones),
            'patterns_count': len(analysis.fractal_patterns)
        }
    
    def _deserialize_structure_analysis(self, data: Dict) -> MarketStructureAnalysis:
        """Désérialise l'analyse depuis le cache"""
        return MarketStructureAnalysis(
            structure_type=StructureType(data.get('structure_type', 'UNDEFINED')),
            trend_strength=data.get('trend_strength', 0.0),
            key_levels=[],  # Simplifié pour le cache
            liquidity_zones=[],
            fractal_patterns=[],
            structure_score=data.get('structure_score', 0.0),
            bias=data.get('bias', 'neutral'),
            next_targets=data.get('next_targets', []),
            risk_zones=[]
        )