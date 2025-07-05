"""
Module pour récupérer les indicateurs techniques depuis la base de données
au lieu de les recalculer. Optimise les performances et évite la duplication.
"""
import asyncio
import asyncpg
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
import os
import sys

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from shared.src.db_pool import DBConnectionPool
from shared.src.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class DatabaseIndicators:
    """
    Gestionnaire optimisé pour récupérer les indicateurs depuis la DB
    et calculer seulement ce qui manque
    """
    
    def __init__(self):
        self.db_pool = DBConnectionPool.get_instance()
        self.tech_indicators = TechnicalIndicators()
        
    def get_enriched_market_data(self, symbol: str, timeframe: str = '1m', 
                                     limit: int = 200) -> Optional[pd.DataFrame]:
        """
        Récupère les données de marché avec TOUS les indicateurs de la DB
        
        Args:
            symbol: Symbole à analyser (ex: BTCUSDC)
            timeframe: Non utilisé pour l'instant (données 1m en DB)
            limit: Nombre de chandelles à récupérer
            
        Returns:
            DataFrame avec OHLCV + tous les indicateurs disponibles
        """
        try:
            query = """
            SELECT 
                time, symbol, open, high, low, close, volume,
                -- Indicateurs de base
                rsi_14, ema_12, ema_26, ema_50, sma_20, sma_50,
                -- MACD complet
                macd_line, macd_signal, macd_histogram,
                -- Bollinger Bands complet
                bb_upper, bb_middle, bb_lower, bb_position, bb_width,
                -- Indicateurs avancés
                atr_14, adx_14, plus_di, minus_di,
                stoch_k, stoch_d, stoch_rsi,
                williams_r, cci_20, mfi_14, vwap_10,
                roc_10, roc_20, obv,
                trend_angle, pivot_count,
                -- Métriques volume et momentum
                momentum_10, volume_ratio, avg_volume_20,
                enhanced, ultra_enriched
            FROM market_data 
            WHERE symbol = %s AND enhanced = true
            ORDER BY time DESC 
            LIMIT %s
            """
            
            from shared.src.db_pool import fetch_all
            rows = fetch_all(query, (symbol, limit), dict_result=True)
                
            if not rows or rows is None:
                logger.warning(f"Aucune donnée trouvée pour {symbol}")
                return None
                
            # Convertir en DataFrame et inverser l'ordre (plus ancien -> plus récent)
            df = pd.DataFrame(rows)
            
            # Vérifier que la colonne 'time' existe
            if 'time' not in df.columns:
                logger.error(f"Colonne 'time' manquante pour {symbol}")
                return None
                
            # Convertir les colonnes numériques de Decimal vers float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                             'rsi_14', 'ema_12', 'ema_26', 'ema_50', 'sma_20', 'sma_50',
                             'macd_line', 'macd_signal', 'macd_histogram',
                             'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_width',
                             'atr_14', 'adx_14', 'plus_di', 'minus_di',
                             'stoch_k', 'stoch_d', 'stoch_rsi',
                             'williams_r', 'cci_20', 'mfi_14', 'vwap_10',
                             'roc_10', 'roc_20', 'obv',
                             'trend_angle', 'pivot_count',
                             'momentum_10', 'volume_ratio', 'avg_volume_20']
            
            for col in numeric_columns:
                if col in df.columns:
                    # Convertir Decimal vers float explicitement
                    df[col] = df[col].apply(lambda x: float(x) if x is not None else np.nan)
                
            df = df.sort_values('time').reset_index(drop=True)
            
            # Convertir time en index datetime
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            logger.info(f"📊 Récupéré {len(df)} chandelles enrichies pour {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération données enrichies {symbol}: {e}")
            return None
    
    def get_available_indicators(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Vérifie quels indicateurs sont disponibles dans les données
        
        Returns:
            Dict indiquant la disponibilité de chaque indicateur
        """
        if df is None or df.empty:
            return {}
            
        indicators_status = {}
        
        # Liste des indicateurs à vérifier
        indicators_to_check = [
            'rsi_14', 'ema_12', 'ema_26', 'ema_50', 'sma_20', 'sma_50',
            'macd_line', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_width',
            'atr_14', 'adx_14', 'plus_di', 'minus_di',
            'stoch_k', 'stoch_d', 'stoch_rsi',
            'williams_r', 'cci_20', 'mfi_14', 'vwap_10',
            'roc_10', 'roc_20', 'obv', 'trend_angle', 'pivot_count',
            'momentum_10', 'volume_ratio', 'avg_volume_20'
        ]
        
        for indicator in indicators_to_check:
            if indicator in df.columns:
                # Vérifier qu'il y a des valeurs non-nulles récentes
                recent_values = df[indicator].tail(10)
                indicators_status[indicator] = not recent_values.isna().all()
            else:
                indicators_status[indicator] = False
                
        available_count = sum(indicators_status.values())
        total_count = len(indicators_status)
        
        logger.debug(f"📈 Indicateurs disponibles: {available_count}/{total_count}")
        
        return indicators_status
    
    def calculate_missing_indicators(self, df: pd.DataFrame, 
                                   missing_indicators: List[str]) -> Dict[str, np.ndarray]:
        """
        Calcule uniquement les indicateurs manquants
        
        Args:
            df: DataFrame avec données OHLCV
            missing_indicators: Liste des indicateurs à calculer
            
        Returns:
            Dict avec les indicateurs calculés
        """
        if df is None or df.empty:
            return {}
            
        calculated = {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            for indicator in missing_indicators:
                try:
                    if indicator == 'rsi_14':
                        val = self.tech_indicators.calculate_rsi(close, 14)
                        if val is not None:
                            calculated[indicator] = np.full(len(close), val)
                            
                    elif indicator.startswith('ema_'):
                        period = int(indicator.split('_')[1])
                        val = self.tech_indicators.calculate_ema(close, period)
                        if val is not None:
                            calculated[indicator] = np.full(len(close), val)
                            
                    elif indicator.startswith('sma_'):
                        period = int(indicator.split('_')[1])
                        val = self.tech_indicators.calculate_sma(close, period)
                        if val is not None:
                            calculated[indicator] = np.full(len(close), val)
                            
                    elif indicator in ['macd_line', 'macd_signal', 'macd_histogram']:
                        macd_data = self.tech_indicators.calculate_macd(close)
                        if macd_data and macd_data.get(indicator):
                            calculated[indicator] = np.full(len(close), macd_data[indicator])
                            
                    elif indicator.startswith('bb_'):
                        bb_data = self.tech_indicators.calculate_bollinger_bands(close, 20, 2.0)
                        if bb_data and bb_data.get(indicator):
                            calculated[indicator] = np.full(len(close), bb_data[indicator])
                            
                    elif indicator == 'atr_14':
                        val = self.tech_indicators.calculate_atr(high, low, close, 14)
                        if val is not None:
                            calculated[indicator] = np.full(len(close), val)
                            
                    elif indicator in ['adx_14', 'plus_di', 'minus_di']:
                        adx, plus_di, minus_di = self.tech_indicators.calculate_adx(high, low, close, 14)
                        if indicator == 'adx_14' and adx is not None:
                            calculated[indicator] = np.full(len(close), adx)
                        elif indicator == 'plus_di' and plus_di is not None:
                            calculated[indicator] = np.full(len(close), plus_di)
                        elif indicator == 'minus_di' and minus_di is not None:
                            calculated[indicator] = np.full(len(close), minus_di)
                            
                    elif indicator in ['stoch_k', 'stoch_d']:
                        stoch_k, stoch_d = self.tech_indicators.calculate_stochastic(high, low, close)
                        if indicator == 'stoch_k' and stoch_k is not None:
                            calculated[indicator] = np.full(len(close), stoch_k)
                        elif indicator == 'stoch_d' and stoch_d is not None:
                            calculated[indicator] = np.full(len(close), stoch_d)
                            
                    elif indicator.startswith('roc_'):
                        period = int(indicator.split('_')[1])
                        val = self.tech_indicators.calculate_roc(close, period)
                        if val is not None:
                            calculated[indicator] = np.full(len(close), val)
                            
                    elif indicator == 'obv':
                        val = self.tech_indicators.calculate_obv(close, volume)
                        if val is not None:
                            calculated[indicator] = np.full(len(close), val)
                            
                    else:
                        logger.debug(f"⚠️ Indicateur {indicator} non implémenté pour calcul fallback")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Erreur calcul {indicator}: {e}")
                    
            if calculated:
                logger.info(f"🔧 Calculé {len(calculated)} indicateurs manquants")
                
        except Exception as e:
            logger.error(f"❌ Erreur calcul indicateurs manquants: {e}")
            
        return calculated
    
    def get_optimized_indicators(self, symbol: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """
        Méthode principale : récupère les indicateurs de la DB et calcule ce qui manque
        
        Returns:
            DataFrame complet avec tous les indicateurs (DB + calculés)
        """
        # 1. Récupérer les données enrichies de la DB
        df = self.get_enriched_market_data(symbol, limit=limit)
        
        if df is None or df.empty:
            logger.warning(f"Impossible de récupérer les données pour {symbol}")
            return None
        
        # 2. Vérifier les indicateurs disponibles
        indicators_status = self.get_available_indicators(df)
        missing_indicators = [ind for ind, available in indicators_status.items() if not available]
        
        # 3. Calculer les indicateurs manquants si nécessaire
        if missing_indicators:
            logger.info(f"🔧 Calcul de {len(missing_indicators)} indicateurs manquants pour {symbol}")
            calculated_indicators = self.calculate_missing_indicators(df, missing_indicators)
            
            # Ajouter les indicateurs calculés au DataFrame
            for indicator, values in calculated_indicators.items():
                if len(values) == len(df):
                    df[indicator] = values
                    
        logger.info(f"✅ Dataset complet pour {symbol}: {len(df)} chandelles avec tous les indicateurs")
        
        return df
    
    def close(self):
        """Ferme les connexions"""
        if self.db_pool:
            self.db_pool.close()

# Instance globale pour utilisation dans l'analyzer
db_indicators = DatabaseIndicators()