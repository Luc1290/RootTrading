"""
Gap Detector - Module de détection et remplissage intelligent des données manquantes
Optimise le rechargement après coupure en ne chargeant que les données manquantes
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncpg
from shared.src.config import get_db_config, SYMBOLS

logger = logging.getLogger(__name__)

class GapDetector:
    """
    Détecte les gaps de données dans la DB et ne charge que les données manquantes
    """
    
    def __init__(self):
        self.db_pool = None
        self.gap_report = {}
        
    async def initialize(self):
        """Initialize la connexion DB"""
        try:
            db_config = get_db_config()
            self.db_pool = await asyncpg.create_pool(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                min_size=2,
                max_size=10
            )
            logger.info("✅ GapDetector initialisé avec pool DB")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation GapDetector: {e}")
            raise
            
    async def close(self):
        """Ferme la connexion DB"""
        if self.db_pool:
            await self.db_pool.close()
            
    async def detect_gaps_for_symbol(self, 
                                   symbol: str, 
                                   timeframe: str,
                                   lookback_hours: int = 24) -> List[Tuple[datetime, datetime]]:
        """
        Détecte les gaps pour un symbole/timeframe spécifique.
        Nouvelle logique : détecte le gap principal depuis la dernière candle jusqu'à maintenant.
        
        Returns:
            Liste de tuples (gap_start, gap_end)
        """
        # Map timeframe vers secondes
        interval_seconds = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }.get(timeframe, 60)
        
        # Calculer la période à vérifier
        now = datetime.utcnow()
        start_time = now - timedelta(hours=lookback_hours)
        
        # NOUVELLE APPROCHE : Chercher la dernière candle et calculer le gap jusqu'à maintenant
        query = """
        WITH latest_data AS (
            -- Récupérer la dernière candle pour ce symbole/timeframe
            SELECT MAX(time) as last_candle_time
            FROM market_data
            WHERE symbol = $1 
            AND timeframe = $2
            AND time >= $3
        ),
        gap_calculation AS (
            SELECT 
                last_candle_time,
                $4::timestamp as current_time,
                EXTRACT(EPOCH FROM ($4::timestamp - last_candle_time)) as seconds_since_last,
                $5 as interval_seconds
            FROM latest_data
            WHERE last_candle_time IS NOT NULL
        )
        SELECT 
            last_candle_time + interval '1 second' * interval_seconds as gap_start,
            current_time as gap_end,
            (seconds_since_last / interval_seconds)::int as missing_candles,
            seconds_since_last
        FROM gap_calculation
        WHERE seconds_since_last > interval_seconds * 2  -- Plus de 2 intervalles de retard
        """
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                query, 
                symbol, 
                timeframe, 
                start_time,
                now,  # current_time 
                interval_seconds
            )
            
            gaps = []
            for row in rows:
                gap_start = row['gap_start']
                gap_end = row['gap_end']
                
                # Convertir en datetime si nécessaire
                if hasattr(gap_start, 'replace'):
                    gaps.append((gap_start, gap_end))
                else:
                    # Si c'est un autre type, essayer de le convertir
                    try:
                        if isinstance(gap_start, str):
                            gap_start = datetime.fromisoformat(gap_start.replace('Z', '+00:00'))
                            gap_end = datetime.fromisoformat(gap_end.replace('Z', '+00:00'))
                        gaps.append((gap_start, gap_end))
                    except Exception as convert_error:
                        logger.warning(f"Erreur conversion timestamp {gap_start}: {convert_error}")
                        continue
            
            if gaps:
                total_missing = sum(row['missing_candles'] for row in rows)
                total_hours = total_missing * interval_seconds / 3600
                logger.warning(f"📊 {symbol} {timeframe}: GAP PRINCIPAL détecté - "
                             f"{total_missing} candles manquantes ({total_hours:.1f}h)")
            else:
                logger.info(f"✅ {symbol} {timeframe}: Aucun gap détecté")
                
            return gaps
            
    async def detect_all_gaps(self, symbols: List[str] = None, lookback_hours: int = 24) -> Dict:
        """
        Détecte tous les gaps pour tous les symboles et timeframes
        
        Returns:
            Dict avec structure: {symbol: {timeframe: [(gap_start, gap_end), ...]}}
        """
        if symbols is None:
            symbols = SYMBOLS
            
        timeframes = ['1m', '5m', '15m', '1h', '4h']
        all_gaps = {}
        total_gaps = 0
        
        logger.info(f"🔍 Détection des gaps sur {lookback_hours}h pour {len(symbols)} symboles...")
        
        for symbol in symbols:
            all_gaps[symbol] = {}
            
            for timeframe in timeframes:
                try:
                    gaps = await self.detect_gaps_for_symbol(symbol, timeframe, lookback_hours)
                    if gaps:
                        all_gaps[symbol][timeframe] = gaps
                        total_gaps += len(gaps)
                except Exception as e:
                    logger.error(f"❌ Erreur détection gaps {symbol} {timeframe}: {e}")
                    
        self.gap_report = all_gaps
        logger.info(f"📊 Détection terminée: {total_gaps} gaps trouvés au total")
        
        return all_gaps
        
    async def get_last_complete_candle(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """
        Récupère le timestamp de la dernière candle complète en DB
        """
        query = """
        SELECT MAX(time) as last_time
        FROM market_data
        WHERE symbol = $1 
        AND timeframe = $2
        AND enhanced = true
        """
        
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query, symbol, timeframe)
            return row['last_time'] if row and row['last_time'] else None
            
    def calculate_fetch_periods(self, gaps: List[Tuple[datetime, datetime]], 
                               max_candles_per_request: int = 1000,
                               timeframe: str = '1m') -> List[Tuple[datetime, datetime]]:
        """
        Optimise les périodes de fetch en regroupant les gaps proches
        
        Args:
            gaps: Liste des gaps détectés
            max_candles_per_request: Limite Binance (1000)
            timeframe: Pour calculer la durée max par requête
            
        Returns:
            Liste optimisée de périodes à fetcher
        """
        if not gaps:
            return []
            
        # Calculer la durée max par requête selon le timeframe
        interval_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }.get(timeframe, 1)
        
        max_duration = timedelta(minutes=interval_minutes * max_candles_per_request)
        merge_threshold = timedelta(minutes=interval_minutes * 10)  # Fusionner si < 10 candles d'écart
        
        # Trier les gaps par ordre chronologique
        sorted_gaps = sorted(gaps, key=lambda x: x[0])
        
        fetch_periods = []
        current_start = sorted_gaps[0][0]
        current_end = sorted_gaps[0][1]
        
        for gap_start, gap_end in sorted_gaps[1:]:
            # S'assurer que nous avons des objets datetime
            if not isinstance(gap_start, datetime):
                logger.warning(f"Gap start n'est pas datetime: {type(gap_start)} - {gap_start}")
                continue
            if not isinstance(current_end, datetime):
                logger.warning(f"Current end n'est pas datetime: {type(current_end)} - {current_end}")
                continue
                
            # Si le gap suivant est proche, fusionner
            try:
                if gap_start - current_end < merge_threshold:
                    current_end = gap_end
                else:
                    # Sinon, ajouter la période courante et commencer une nouvelle
                    fetch_periods.append((current_start, current_end))
                    current_start = gap_start
                    current_end = gap_end
            except TypeError as e:
                logger.error(f"Erreur calcul durée gap: {e} - gap_start: {type(gap_start)}, current_end: {type(current_end)}")
                continue
                
        # Ajouter la dernière période
        fetch_periods.append((current_start, current_end))
        
        # Diviser les périodes trop longues
        final_periods = []
        for start, end in fetch_periods:
            try:
                # Vérifier les types avant calcul
                if not isinstance(start, datetime) or not isinstance(end, datetime):
                    logger.warning(f"Période invalide - start: {type(start)}, end: {type(end)}")
                    continue
                    
                duration = end - start
                if duration > max_duration:
                    # Diviser en sous-périodes
                    current = start
                    while current < end:
                        period_end = min(current + max_duration, end)
                        final_periods.append((current, period_end))
                        current = period_end + timedelta(seconds=1)
                else:
                    final_periods.append((start, end))
            except TypeError as e:
                logger.error(f"Erreur division période: {e} - start: {type(start)}, end: {type(end)}")
                continue
                
        logger.info(f"📈 Optimisé {len(gaps)} gaps en {len(final_periods)} requêtes pour {timeframe}")
        
        return final_periods
        
    def generate_gap_filling_plan(self, all_gaps: Dict) -> Dict:
        """
        Génère un plan optimisé de remplissage des gaps
        
        Returns:
            Dict avec structure: {symbol: {timeframe: [fetch_periods]}}
        """
        filling_plan = {}
        total_requests = 0
        
        for symbol, timeframe_gaps in all_gaps.items():
            if not timeframe_gaps:
                continue
                
            filling_plan[symbol] = {}
            
            for timeframe, gaps in timeframe_gaps.items():
                if gaps:
                    fetch_periods = self.calculate_fetch_periods(gaps, timeframe=timeframe)
                    filling_plan[symbol][timeframe] = fetch_periods
                    total_requests += len(fetch_periods)
                    
        logger.info(f"📋 Plan de remplissage: {total_requests} requêtes nécessaires")
        
        return filling_plan
        
    def estimate_fill_time(self, filling_plan: Dict, rate_limit_delay: float = 0.1) -> float:
        """
        Estime le temps nécessaire pour remplir tous les gaps
        """
        total_requests = sum(
            len(periods) 
            for symbol_plan in filling_plan.values() 
            for periods in symbol_plan.values()
        )
        
        # Temps estimé en secondes (avec délai entre requêtes)
        estimated_seconds = total_requests * rate_limit_delay
        
        return estimated_seconds