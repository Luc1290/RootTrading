"""Universe Manager - Sélection dynamique des cryptos à trader"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from scipy import stats
import json

logger = logging.getLogger(__name__)

@dataclass
class PairScore:
    """Score de tradeabilité d'une paire"""
    symbol: str
    score: float
    atr_pct: float
    roc: float
    volume_ratio: float
    is_ranging: bool
    timestamp: datetime
    components: Dict[str, float] = field(default_factory=dict)

@dataclass
class PairState:
    """État d'une paire dans le système de sélection"""
    symbol: str
    is_selected: bool
    score_history: List[float] = field(default_factory=list)
    last_selected: Optional[datetime] = None
    last_deselected: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    consecutive_above_threshold: int = 0
    consecutive_below_threshold: int = 0

class UniverseManager:
    """Gestionnaire de l'univers tradable avec sélection dynamique"""
    
    def __init__(self, redis_client, db_pool=None, config: Optional[Dict] = None):
        self.redis = redis_client
        self.db_pool = db_pool
        self.config = config or self._default_config()
        
        # État des paires
        self.pair_states: Dict[str, PairState] = {}
        self.selected_universe: Set[str] = set()
        self.core_pairs: Set[str] = set(self.config['core_pairs'])
        
        # Cache des données de marché
        self.market_data_cache: Dict[str, Dict] = {}
        self.last_update = datetime.now()
        
        # Vérifier le pool DB
        if not self.db_pool:
            logger.warning("Pas de pool DB fourni, utilisation de Redis uniquement")
        
        logger.info(f"UniverseManager initialisé avec {len(self.core_pairs)} paires core")
    
    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            # Paires toujours actives (format USDC pour Binance)
            # BTC/ETH = bases incontournables, top liquidité
            # SOL = Solana, très liquide, mouvements rapides
            # XRP = Ripple, énorme volume, bonne volatilité
            # ADA = Cardano, stable et liquide
            # LINK = Chainlink, oracle leader, patterns fiables
            'core_pairs': ['BTCUSDC', 'ETHUSDC', 'SOLUSDC', 'XRPUSDC', 'ADAUSDC', 'LINKUSDC'],
            
            # Nombre max de satellites (en plus des core)
            'max_satellites': 6,  # 6 core + 6 satellites = 12 cryptos max

            # Seuils de score
            'score_threshold_enter': 0.2,  # Abaissé pour permettre plus de satellites
            'score_threshold_exit': -0.3,  # Abaissé aussi pour cohérence
            
            # Hystérésis (nombre de périodes)
            'periods_above_to_enter': 3,
            'periods_below_to_exit': 6,
            
            # Cool-down après désélection (minutes)
            'cooldown_minutes': 45,  # Augmenté pour éviter réactivation trop rapide
            
            # Poids des composants du score
            'weights': {
                'atr': 0.3,
                'roc': 0.3,
                'volume': 0.15,
                'trend': 0.25  # Nouveau: bonus pour tendance directionnelle
            },
            
            # Paramètres de calcul
            'atr_period': 14,
            'roc_period_fast': 15,
            'roc_period_slow': 30,
            'volume_period': 20,
            
            # Seuils de range
            'bb_squeeze_threshold': 0.02,  # Bollinger Band squeeze %
            'adx_range_threshold': 20,     # ADX < 20 = ranging
            
            # Hard risk (forçage de sortie)
            'hard_risk': {
                'enabled': True,
                'atr_spike_threshold': 0.5,  # % ATR
                'spread_multiplier': 3,      # x median spread
                'slippage_multiplier': 2     # x baseline slippage
            }
        }
    
    def update_market_data(self, symbol: str, data: Dict) -> None:
        """Met à jour les données de marché pour une paire"""
        self.market_data_cache[symbol] = {
            **data,
            'timestamp': datetime.now()
        }
    
    def calculate_score(self, symbol: str) -> PairScore:
        """Calcule le score de tradeabilité d'une paire en utilisant les données de la DB"""
        try:
            # Utiliser la DB si disponible (priorité car données déjà calculées)
            if self.db_pool:
                return self._calculate_score_from_db(symbol)
            else:
                # Fallback sur Redis/calculs manuels si pas de DB
                return self._calculate_score_from_redis(symbol)
                
        except Exception as e:
            logger.error(f"Erreur calcul score {symbol}: {e}")
            return PairScore(symbol, -1.0, 0, 0, 0, True, datetime.now())
    
    def _calculate_score_from_db(self, symbol: str) -> PairScore:
        """Calcule le score depuis les données analyzer_data de la DB"""
        try:
            # Utiliser le pool de connexions avec RealDictCursor
            from shared.src.db_pool import real_dict_cursor
            
            with real_dict_cursor() as cursor:
                # Récupérer les dernières données d'analyse pour ce symbole
                query = """
                    SELECT 
                        a.*,
                        m.close as current_price,
                        m.volume as current_volume
                    FROM analyzer_data a
                    JOIN market_data m ON (a.time = m.time AND a.symbol = m.symbol AND a.timeframe = m.timeframe)
                    WHERE a.symbol = %s 
                    AND a.timeframe = '3m'
                    ORDER BY a.time DESC
                    LIMIT 1
                """
                cursor.execute(query, (symbol,))
                data = cursor.fetchone()
                
                if not data:
                    logger.warning(f"Pas de données analyzer pour {symbol}")
                    return PairScore(symbol, -1.0, 0, 0, 0, True, datetime.now())
                
                # Extraire les métriques depuis analyzer_data
                atr_pct = float(data['natr']) if data['natr'] else 0  # Normalized ATR
                roc = (float(data['roc_10']) + float(data['roc_20'])) / 2 if data['roc_10'] and data['roc_20'] else 0
                volume_ratio = float(data['volume_ratio']) if data['volume_ratio'] else 1.0
                is_ranging = data['bb_squeeze'] if data['bb_squeeze'] is not None else False
                adx = float(data['adx_14']) if data['adx_14'] else 0
                
                # Score de tendance basé sur les données DB
                trend_score = 0.0
                
                # Utiliser ADX de la DB
                if adx > 25:
                    trend_score += 1.0
                elif adx > 20:
                    trend_score += 0.5
                
                # Utiliser le trend_strength de la DB
                if data['trend_strength'] == 'VERY_STRONG':
                    trend_score += 0.5
                elif data['trend_strength'] == 'STRONG':
                    trend_score += 0.3
                
                # Utiliser le regime de marché
                if data['market_regime'] in ['TRENDING_BULL', 'BREAKOUT_BULL']:
                    trend_score += 0.3
                elif data['market_regime'] in ['TRENDING_BEAR', 'BREAKOUT_BEAR']:
                    trend_score -= 0.3
                
                # Normaliser entre -1 et 1
                trend_score = max(-1, min(1, trend_score))
                
                # Calculer les z-scores pour normalisation
                all_scores = self._get_all_recent_scores()
                
                z_atr = self._calculate_zscore(atr_pct, 
                                              [s.atr_pct for s in all_scores])
                z_roc = self._calculate_zscore(roc, 
                                              [s.roc for s in all_scores])
                z_volume = self._calculate_zscore(volume_ratio, 
                                                 [s.volume_ratio for s in all_scores])
                
                # Score final pondéré avec trend bias
                score = (
                    z_atr * self.config['weights']['atr'] +
                    z_roc * self.config['weights']['roc'] +
                    z_volume * self.config['weights']['volume'] +
                    trend_score * self.config['weights']['trend']
                )
                
                # Pénalité si en range
                if is_ranging:
                    score -= 0.5
                
                # Bonus si confluence élevée dans la DB
                if data['confluence_score'] and float(data['confluence_score']) > 70:
                    score += 0.3
                
                # Bonus si volume context favorable
                if data['volume_context'] in ['BREAKOUT', 'PUMP_START']:
                    score += 0.2
                
                return PairScore(
                    symbol=symbol,
                    score=score,
                    atr_pct=atr_pct,
                    roc=roc,
                    volume_ratio=volume_ratio,
                    is_ranging=is_ranging,
                    timestamp=datetime.now(),
                    components={
                        'z_atr': z_atr,
                        'z_roc': z_roc,
                        'z_volume': z_volume,
                        'trend_bias': trend_score,
                        'adx': adx,
                        'range_penalty': -0.5 if is_ranging else 0,
                        'market_regime': data['market_regime'],
                        'confluence_score': float(data['confluence_score']) if data['confluence_score'] else 0
                    }
                )
                
        except Exception as e:
            logger.error(f"Erreur récupération données DB pour {symbol}: {e}")
            # Fallback sur calcul Redis si erreur DB
            return self._calculate_score_from_redis(symbol)
    
    def _calculate_score_from_redis(self, symbol: str) -> PairScore:
        """Calcule le score depuis Redis (méthode originale comme fallback)"""
        try:
            # Récupérer les données depuis Redis
            candles_key = f"candles:{symbol}:3m"
            candles_data = self.redis.get(candles_key)
            
            if not candles_data:
                logger.warning(f"Pas de données Redis pour {symbol}")
                return PairScore(symbol, -1.0, 0, 0, 0, True, datetime.now())
            
            candles = json.loads(candles_data)
            if len(candles) < 50:
                return PairScore(symbol, -1.0, 0, 0, 0, True, datetime.now())
            
            # Extraire les prix et volumes
            closes = [float(c['close']) for c in candles[-50:]]
            highs = [float(c['high']) for c in candles[-50:]]
            lows = [float(c['low']) for c in candles[-50:]]
            volumes = [float(c['volume']) for c in candles[-50:]]
            
            # 1. ATR% (volatilité normalisée)
            tr_list = []
            for i in range(1, len(closes)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                tr_list.append(tr)
            
            atr = np.mean(tr_list[-self.config['atr_period']:])
            atr_pct = (atr / closes[-1]) * 100
            
            # 2. ROC (momentum)
            roc_fast = ((closes[-1] - closes[-self.config['roc_period_fast']]) / 
                        closes[-self.config['roc_period_fast']]) * 100
            roc_slow = ((closes[-1] - closes[-self.config['roc_period_slow']]) / 
                        closes[-self.config['roc_period_slow']]) * 100
            roc = (roc_fast + roc_slow) / 2
            
            # 3. Volume ratio
            recent_volume = np.mean(volumes[-5:])
            avg_volume = np.mean(volumes[-self.config['volume_period']:])
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 4. Détection de range (Bollinger Bands squeeze)
            ma20 = np.mean(closes[-20:])
            std20 = np.std(closes[-20:])
            bb_width = (2 * std20) / ma20 if ma20 > 0 else 0
            is_ranging = bb_width < self.config['bb_squeeze_threshold']
            
            # 5. Calcul ADX pour trend bias (tendance directionnelle)
            adx = self._calculate_adx(highs, lows, closes, period=14)
            
            # 6. Calcul de la pente EMA (trend direction)
            ema_short = self._calculate_ema(closes, 9)
            ema_long = self._calculate_ema(closes, 21)
            if len(ema_short) >= 2 and len(ema_long) >= 2:
                # Pente normalisée de l'EMA courte
                ema_slope = ((ema_short[-1] - ema_short[-5]) / ema_short[-5]) * 100 if len(ema_short) > 5 else 0
                # Signal de croisement EMA
                ema_cross = 1.0 if ema_short[-1] > ema_long[-1] else -1.0
            else:
                ema_slope = 0
                ema_cross = 0
            
            # Score de tendance combiné
            trend_score = 0.0
            if adx > 25:  # Tendance forte
                trend_score += 1.0
            elif adx > 20:  # Tendance modérée
                trend_score += 0.5
            
            if abs(ema_slope) > 1.0:  # Pente significative
                trend_score += 0.5 * np.sign(ema_slope) * ema_cross
            
            # Normaliser entre -1 et 1
            trend_score = max(-1, min(1, trend_score))
            
            # Calculer les z-scores pour normalisation
            all_scores = self._get_all_recent_scores()
            
            z_atr = self._calculate_zscore(atr_pct, 
                                          [s.atr_pct for s in all_scores])
            z_roc = self._calculate_zscore(roc, 
                                          [s.roc for s in all_scores])
            z_volume = self._calculate_zscore(volume_ratio, 
                                             [s.volume_ratio for s in all_scores])
            
            # Score final pondéré avec trend bias
            score = (
                z_atr * self.config['weights']['atr'] +
                z_roc * self.config['weights']['roc'] +
                z_volume * self.config['weights']['volume'] +
                trend_score * self.config['weights']['trend']  # Bonus/malus de tendance
            )
            
            # Pénalité si en range
            if is_ranging:
                score -= 0.5
            
            return PairScore(
                symbol=symbol,
                score=score,
                atr_pct=atr_pct,
                roc=roc,
                volume_ratio=volume_ratio,
                is_ranging=is_ranging,
                timestamp=datetime.now(),
                components={
                    'z_atr': z_atr,
                    'z_roc': z_roc,
                    'z_volume': z_volume,
                    'trend_bias': trend_score,
                    'adx': adx,
                    'range_penalty': -0.5 if is_ranging else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Erreur calcul score {symbol}: {e}")
            return PairScore(symbol, -1.0, 0, 0, 0, True, datetime.now())
    
    def _calculate_zscore(self, value: float, population: List[float]) -> float:
        """Calcule le z-score d'une valeur par rapport à la population"""
        if len(population) < 2:
            return 0.0
        
        mean = np.mean(population)
        std = np.std(population, ddof=1)  # Sample standard deviation
        
        if std == 0:
            return 0.0
        
        return (value - mean) / std
    
    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calcule l'EMA (Exponential Moving Average)"""
        if len(prices) < period:
            return []
        
        ema = []
        multiplier = 2 / (period + 1)
        
        # Première valeur = SMA
        ema.append(np.mean(prices[:period]))
        
        # Calcul EMA pour le reste
        for i in range(period, len(prices)):
            ema_value = (prices[i] - ema[-1]) * multiplier + ema[-1]
            ema.append(ema_value)
        
        return ema
    
    def _calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calcule l'ADX (Average Directional Index)"""
        if len(highs) < period + 1:
            return 0.0
        
        try:
            # Calcul du True Range
            tr_list = []
            for i in range(1, len(closes)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                tr_list.append(tr)
            
            # Calcul +DM et -DM
            plus_dm = []
            minus_dm = []
            for i in range(1, len(highs)):
                high_diff = highs[i] - highs[i-1]
                low_diff = lows[i-1] - lows[i]
                
                if high_diff > low_diff and high_diff > 0:
                    plus_dm.append(high_diff)
                else:
                    plus_dm.append(0)
                
                if low_diff > high_diff and low_diff > 0:
                    minus_dm.append(low_diff)
                else:
                    minus_dm.append(0)
            
            # Moyennes lissées
            atr = np.mean(tr_list[-period:])
            plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
            minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr > 0 else 0
            
            # Calcul DX et ADX
            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx = 100 * abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
            
            # ADX est la moyenne du DX (simplifié ici)
            return dx
            
        except Exception as e:
            logger.debug(f"Erreur calcul ADX: {e}")
            return 0.0
    
    def _batch_calculate_scores_from_db(self, symbols: List[str]) -> Dict[str, PairScore]:
        """Calcule tous les scores depuis la DB en une seule requête"""
        scores = {}
        
        try:
            from shared.src.db_pool import real_dict_cursor
            
            with real_dict_cursor() as cursor:
                # Requête JOIN comme dans visualization - market_data + analyzer_data
                query = """
                    SELECT DISTINCT ON (md.symbol)
                        md.symbol,
                        md.close,
                        md.high,
                        md.low,
                        md.volume,
                        md.time,
                        -- Indicateurs depuis analyzer_data
                        ad.natr,
                        ad.roc_10,
                        ad.roc_20,
                        ad.volume_ratio,
                        ad.bb_squeeze,
                        ad.adx_14,
                        ad.trend_strength,
                        ad.market_regime,
                        ad.confluence_score,
                        ad.volume_context
                    FROM market_data md
                    LEFT JOIN analyzer_data ad ON (md.time = ad.time AND md.symbol = ad.symbol AND md.timeframe = ad.timeframe)
                    WHERE md.symbol = ANY(%s)
                    AND md.timeframe = '3m'
                    AND md.time > NOW() - INTERVAL '15 minutes'
                    ORDER BY md.symbol, md.time DESC
                """
                cursor.execute(query, (symbols,))
                results = cursor.fetchall()
                
                # Extraire toutes les métriques d'abord pour calcul de z-scores
                all_atr = []
                all_roc = []
                all_volume = []
                raw_data = []
                
                for data in results:
                    atr_pct = float(data['natr']) if data['natr'] else 0
                    roc = (float(data['roc_10']) + float(data['roc_20'])) / 2 if data['roc_10'] and data['roc_20'] else 0
                    volume_ratio = float(data['volume_ratio']) if data['volume_ratio'] else 1.0
                    
                    all_atr.append(atr_pct)
                    all_roc.append(roc)
                    all_volume.append(volume_ratio)
                    raw_data.append(data)
                
                # Traiter chaque résultat avec z-scores
                for i, data in enumerate(raw_data):
                    symbol = data['symbol']
                    
                    # Métriques
                    atr_pct = all_atr[i]
                    roc = all_roc[i]
                    volume_ratio = all_volume[i]
                    is_ranging = data['bb_squeeze'] if data['bb_squeeze'] is not None else False
                    adx = float(data['adx_14']) if data['adx_14'] else 0
                    
                    # Score de tendance
                    trend_score = self._calculate_trend_score_from_db(data)
                    
                    # Calculer les z-scores
                    z_atr = self._calculate_zscore(atr_pct, all_atr)
                    z_roc = self._calculate_zscore(roc, all_roc)
                    z_volume = self._calculate_zscore(volume_ratio, all_volume)
                    
                    # Score final avec z-scores
                    score = (
                        z_atr * self.config['weights']['atr'] +
                        z_roc * self.config['weights']['roc'] +
                        z_volume * self.config['weights']['volume'] +
                        trend_score * self.config['weights']['trend']
                    )
                    
                    # Ajustements
                    if is_ranging:
                        score -= 0.5
                    
                    if data['confluence_score'] and float(data['confluence_score']) > 70:
                        score += 0.3
                    
                    if data['volume_context'] in ['BREAKOUT', 'PUMP_START']:
                        score += 0.2
                    
                    scores[symbol] = PairScore(
                        symbol=symbol,
                        score=score,
                        atr_pct=atr_pct,
                        roc=roc,
                        volume_ratio=volume_ratio,
                        is_ranging=is_ranging,
                        timestamp=datetime.now(),
                        components={
                            'z_atr': z_atr,
                            'z_roc': z_roc,
                            'z_volume': z_volume,
                            'trend_bias': trend_score,
                            'adx': adx,
                            'range_penalty': -0.5 if is_ranging else 0,
                            'market_regime': data['market_regime'],
                            'confluence_score': float(data['confluence_score']) if data['confluence_score'] else 0
                        }
                    )
                    
                    # Mettre en cache
                    self.market_data_cache[symbol] = {
                        'score': scores[symbol],
                        'timestamp': datetime.now()
                    }
                
                # Pour les symboles sans données DB, utiliser le fallback
                for symbol in symbols:
                    if symbol not in scores:
                        scores[symbol] = self._calculate_score_from_redis(symbol)
                
        except Exception as e:
            logger.error(f"Erreur batch calculate scores: {e}")
            # Fallback complet sur Redis
            for symbol in symbols:
                scores[symbol] = self._calculate_score_from_redis(symbol)
        
        return scores
    
    def _calculate_trend_score_from_db(self, data: Dict) -> float:
        """Calcule le score de tendance depuis les données DB"""
        trend_score = 0.0
        
        # ADX
        adx = float(data['adx_14']) if data['adx_14'] else 0
        if adx > 25:
            trend_score += 1.0
        elif adx > 20:
            trend_score += 0.5
        
        # Trend strength
        if data['trend_strength'] == 'VERY_STRONG':
            trend_score += 0.5
        elif data['trend_strength'] == 'STRONG':
            trend_score += 0.3
        
        # Market regime
        if data['market_regime'] in ['TRENDING_BULL', 'BREAKOUT_BULL']:
            trend_score += 0.3
        elif data['market_regime'] in ['TRENDING_BEAR', 'BREAKOUT_BEAR']:
            trend_score -= 0.3
        
        return max(-1, min(1, trend_score))
    
    def _get_all_recent_scores(self) -> List[PairScore]:
        """Récupère les scores récents de toutes les paires"""
        scores = []
        
        # Récupérer la liste des symboles depuis Redis
        symbols_data = self.redis.get("trading:symbols")
        if not symbols_data:
            return scores
        
        symbols = json.loads(symbols_data)
        
        for symbol in symbols:
            if symbol in self.market_data_cache:
                cache_entry = self.market_data_cache[symbol]
                if (datetime.now() - cache_entry['timestamp']).seconds < 180:
                    scores.append(cache_entry.get('score'))
        
        return [s for s in scores if s is not None]
    
    def apply_hysteresis(self, symbol: str, current_score: float) -> bool:
        """Applique l'hystérésis pour éviter le ping-pong"""
        if symbol not in self.pair_states:
            self.pair_states[symbol] = PairState(symbol=symbol, is_selected=False)
        
        state = self.pair_states[symbol]
        
        # Ajouter le score à l'historique
        state.score_history.append(current_score)
        if len(state.score_history) > 10:
            state.score_history.pop(0)
        
        # Vérifier cool-down
        if state.cooldown_until and datetime.now() < state.cooldown_until:
            return False
        
        # Logique d'hystérésis
        if not state.is_selected:
            # Pour entrer : score > seuil pendant N périodes
            if current_score > self.config['score_threshold_enter']:
                state.consecutive_above_threshold += 1
                state.consecutive_below_threshold = 0
                
                if state.consecutive_above_threshold >= self.config['periods_above_to_enter']:
                    state.is_selected = True
                    state.last_selected = datetime.now()
                    state.consecutive_above_threshold = 0
                    return True
            else:
                state.consecutive_above_threshold = 0
        else:
            # Pour sortir : score < seuil pendant M périodes
            if current_score < self.config['score_threshold_exit']:
                state.consecutive_below_threshold += 1
                state.consecutive_above_threshold = 0
                
                if state.consecutive_below_threshold >= self.config['periods_below_to_exit']:
                    state.is_selected = False
                    state.last_deselected = datetime.now()
                    state.cooldown_until = datetime.now() + timedelta(
                        minutes=self.config['cooldown_minutes']
                    )
                    state.consecutive_below_threshold = 0
                    return False
            else:
                state.consecutive_below_threshold = 0
        
        return state.is_selected
    
    def update_universe(self) -> Tuple[Set[str], Dict[str, PairScore]]:
        """Met à jour l'univers tradable"""
        try:
            # Récupérer tous les symboles
            symbols_data = self.redis.get("trading:symbols")
            if not symbols_data:
                logger.warning("Pas de symboles configurés")
                return self.core_pairs, {}
            
            # symbols_data peut être une string JSON ou déjà une liste
            if isinstance(symbols_data, str):
                all_symbols = json.loads(symbols_data)
            else:
                all_symbols = symbols_data
            
            # Calculer les scores (fallback sur Redis si DB trop lente)
            scores = {}
            
            # TEMPORAIRE: Forcer DB car Redis n'a pas les données de marché
            if False and len(all_symbols) > 10 or not self.db_pool:
                logger.info(f"Utilisation Redis pour {len(all_symbols)} symboles (éviter requête DB lente)")
                for symbol in all_symbols:
                    try:
                        score = self._calculate_score_from_redis(symbol)
                        scores[symbol] = score
                        
                        # Mettre en cache
                        self.market_data_cache[symbol] = {
                            'score': score,
                            'timestamp': datetime.now()
                        }
                    except Exception as e:
                        logger.debug(f"Erreur calcul score Redis {symbol}: {e}")
                        scores[symbol] = PairScore(symbol, -1.0, 0, 0, 0, True, datetime.now())
            else:
                # Utiliser la DB seulement pour un petit nombre de symboles
                scores = self._batch_calculate_scores_from_db(all_symbols)
            
            # Enregistrer l'état actuel avant modification
            previous_universe = self.selected_universe.copy()
            
            # Toujours inclure les paires core
            selected = set(self.core_pairs)
            
            # Filtrer les satellites éligibles
            eligible_satellites = []
            for symbol, score in scores.items():
                if symbol not in self.core_pairs and score.score > -1:
                    # Appliquer hystérésis
                    if self.apply_hysteresis(symbol, score.score):
                        eligible_satellites.append((symbol, score.score))
            
            # Sélectionner les meilleurs satellites
            eligible_satellites.sort(key=lambda x: x[1], reverse=True)
            for symbol, _ in eligible_satellites[:self.config['max_satellites']]:
                selected.add(symbol)
            
            # Mettre à jour l'univers
            self.selected_universe = selected
            self.last_update = datetime.now()
            
            # Publier dans Redis
            self.redis.set(
                "universe:selected",
                json.dumps(list(selected)),
                expiration=180  # TTL 3 minutes
            )
            
            # Log les changements (utiliser previous_universe)
            added = selected - previous_universe
            removed = previous_universe - selected
            
            if added:
                logger.info(f"Paires ajoutées à l'univers: {added}")
            if removed:
                logger.info(f"Paires retirées de l'univers: {removed}")
            
            return selected, scores
            
        except Exception as e:
            logger.error(f"Erreur update_universe: {e}")
            return self.core_pairs, {}
    
    def is_pair_tradable(self, symbol: str) -> bool:
        """Vérifie si une paire peut ouvrir de nouvelles positions"""
        return symbol in self.selected_universe
    
    def check_hard_risk(self, symbol: str) -> bool:
        """Vérifie les conditions de risque extrême"""
        if not self.config['hard_risk']['enabled']:
            return False
        
        try:
            # Récupérer les métriques de risque depuis Redis
            risk_key = f"risk:{symbol}"
            risk_data = self.redis.get(risk_key)
            
            if not risk_data:
                return False
            
            risk = json.loads(risk_data)
            
            # Vérifier les conditions
            atr_spike = risk.get('atr_spike', 0) > self.config['hard_risk']['atr_spike_threshold']
            spread_high = risk.get('spread_ratio', 0) > self.config['hard_risk']['spread_multiplier']
            slippage_high = risk.get('slippage_ratio', 0) > self.config['hard_risk']['slippage_multiplier']
            
            # Toutes les conditions doivent être vraies
            return atr_spike and spread_high and slippage_high
            
        except Exception as e:
            logger.error(f"Erreur check_hard_risk {symbol}: {e}")
            return False
    
    def get_universe_stats(self) -> Dict:
        """Retourne les statistiques de l'univers"""
        stats = {
            'selected_count': len(self.selected_universe),
            'selected_symbols': list(self.selected_universe),
            'core_pairs': list(self.core_pairs),
            'satellites': list(self.selected_universe - self.core_pairs),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'pair_states': {}
        }
        
        for symbol, state in self.pair_states.items():
            stats['pair_states'][symbol] = {
                'is_selected': state.is_selected,
                'recent_scores': state.score_history[-5:] if state.score_history else [],
                'cooldown_until': state.cooldown_until.isoformat() if state.cooldown_until else None
            }
        
        return stats
    
    def force_pair_selection(self, symbol: str, duration_minutes: int = 60) -> None:
        """Force la sélection d'une paire pour test"""
        self.selected_universe.add(symbol)
        logger.info(f"Paire {symbol} forcée dans l'univers pour {duration_minutes} minutes")
        
        # Publier dans Redis
        self.redis.set(
            "universe:selected",
            json.dumps(list(self.selected_universe)),
            expiration=300
        )
    
