"""
DataConverter - Conversion bidirectionnelle entre formats ROOT et Freqtrade.
Gère la conversion des données OHLCV et indicateurs entre dict (ROOT) et DataFrame (Freqtrade).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataConverter:
    """Convertit les données entre formats ROOT (dict) et Freqtrade (DataFrame)."""

    @staticmethod
    def root_to_dataframe(
        data: Dict[str, Any],
        indicators: Dict[str, Any],
        historical_data: Optional[List[Dict]] = None,
    ) -> pd.DataFrame:
        """
        Convertit données ROOT (dict) → DataFrame Freqtrade.

        Args:
            data: Dict OHLCV actuel depuis ROOT
            indicators: Dict indicateurs actuels depuis ROOT
            historical_data: Liste optionnelle de données historiques

        Returns:
            DataFrame avec colonnes Freqtrade standard (date, open, high, low, close, volume, + indicateurs)
        """
        try:
            # Si données historiques fournies, les utiliser
            if historical_data:
                df = pd.DataFrame(historical_data)

                # Normaliser les noms de colonnes
                column_mapping = {
                    "timestamp": "date",
                    "open_price": "open",
                    "high_price": "high",
                    "low_price": "low",
                    "close_price": "close",
                    "volume": "volume",
                }

                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns:
                        df.rename(columns={old_col: new_col}, inplace=True)

            else:
                # Créer DataFrame à partir d'un seul point de données
                df = pd.DataFrame(
                    [
                        {
                            "date": data.get("timestamp", datetime.now()),
                            "open": data.get("open_price", 0),
                            "high": data.get("high_price", 0),
                            "low": data.get("low_price", 0),
                            "close": data.get("close_price", 0),
                            "volume": data.get("volume", 0),
                        }
                    ]
                )

            # Convertir timestamp en datetime si nécessaire
            if "date" in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df["date"]):
                    df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)

            # Ajouter les indicateurs au DataFrame
            if indicators:
                for indicator_name, value in indicators.items():
                    if value is not None:
                        # Si value est une liste, l'assigner directement
                        if isinstance(value, (list, pd.Series)):
                            if len(value) == len(df):
                                df[indicator_name] = value
                        else:
                            # Sinon répéter la valeur pour toutes les lignes
                            df[indicator_name] = value

            return df

        except Exception as e:
            logger.error(f"Erreur conversion ROOT → DataFrame: {e}")
            # Retourner DataFrame vide avec structure minimale
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    @staticmethod
    def dataframe_to_root(
        df: pd.DataFrame, symbol: str
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Convertit DataFrame Freqtrade → format ROOT (data + indicators).

        Args:
            df: DataFrame Freqtrade
            symbol: Symbole de trading

        Returns:
            Tuple (data_dict, indicators_dict) au format ROOT
        """
        try:
            if df.empty:
                return {}, {}

            # Prendre la dernière ligne pour les données actuelles
            last_row = df.iloc[-1]

            # Colonnes OHLCV standard
            ohlcv_columns = {"open", "high", "low", "close", "volume"}

            # Construire dict data
            data = {
                "symbol": symbol,
                "timestamp": (
                    last_row.name
                    if isinstance(df.index, pd.DatetimeIndex)
                    else datetime.now()
                ),
                "open_price": float(last_row.get("open", 0)),
                "high_price": float(last_row.get("high", 0)),
                "low_price": float(last_row.get("low", 0)),
                "close_price": float(last_row.get("close", 0)),
                "volume": float(last_row.get("volume", 0)),
            }

            # Construire dict indicators (toutes les colonnes sauf OHLCV)
            indicators = {}
            for col in df.columns:
                if col not in ohlcv_columns:
                    value = last_row[col]
                    # Convertir NaN en None
                    if pd.isna(value):
                        indicators[col] = None
                    else:
                        indicators[col] = (
                            float(value)
                            if isinstance(value, (int, float, np.number))
                            else value
                        )

            return data, indicators

        except Exception as e:
            logger.error(f"Erreur conversion DataFrame → ROOT: {e}")
            return {}, {}

    @staticmethod
    def merge_historical_with_indicators(
        historical_df: pd.DataFrame, indicators: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Fusionne données historiques avec indicateurs calculés par ROOT.

        Args:
            historical_df: DataFrame avec données OHLCV historiques
            indicators: Dict indicateurs depuis ROOT

        Returns:
            DataFrame enrichi avec indicateurs
        """
        try:
            df = historical_df.copy()

            # Ajouter chaque indicateur
            for indicator_name, value in indicators.items():
                if value is not None:
                    if isinstance(value, (list, pd.Series)):
                        # Si c'est une série historique
                        if len(value) == len(df):
                            df[indicator_name] = value
                        else:
                            logger.warning(
                                f"Longueur indicateur {indicator_name} ({len(value)}) "
                                f"!= longueur données ({len(df)})"
                            )
                    else:
                        # Sinon valeur scalaire, répliquer sur toutes les lignes
                        df[indicator_name] = value

            return df

        except Exception as e:
            logger.error(f"Erreur fusion historique + indicateurs: {e}")
            return historical_df

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        """
        Valide qu'un DataFrame contient les colonnes minimales requises par Freqtrade.

        Args:
            df: DataFrame à valider

        Returns:
            True si valide, False sinon
        """
        required_columns = {"open", "high", "low", "close", "volume"}

        if df.empty:
            logger.error("DataFrame vide")
            return False

        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            logger.error(f"Colonnes manquantes: {missing_columns}")
            return False

        # Vérifier qu'il n'y a pas de valeurs nulles dans les colonnes OHLCV
        for col in required_columns:
            if df[col].isna().any():
                logger.warning(f"Colonne {col} contient des valeurs nulles")

        return True

    @staticmethod
    def resample_dataframe(df: pd.DataFrame, timeframe: str = "5m") -> pd.DataFrame:
        """
        Rééchantillonne un DataFrame vers une timeframe différente.

        Args:
            df: DataFrame source
            timeframe: Timeframe cible ('1m', '5m', '15m', '1h', '1d', etc.)

        Returns:
            DataFrame rééchantillonné
        """
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.error("Index doit être DatetimeIndex pour resample")
                return df

            # Mapping timeframe string → pandas offset
            timeframe_map = {
                "1m": "1T",
                "5m": "5T",
                "15m": "15T",
                "30m": "30T",
                "1h": "1H",
                "2h": "2H",
                "4h": "4H",
                "6h": "6H",
                "12h": "12H",
                "1d": "1D",
                "1w": "1W",
            }

            offset = timeframe_map.get(timeframe, timeframe)

            # Rééchantillonner OHLCV
            resampled = df.resample(offset).agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )

            # Pour les indicateurs, prendre la dernière valeur
            indicator_cols = [
                col
                for col in df.columns
                if col not in {"open", "high", "low", "close", "volume"}
            ]
            for col in indicator_cols:
                resampled[col] = df[col].resample(offset).last()

            return resampled.dropna()

        except Exception as e:
            logger.error(f"Erreur resample: {e}")
            return df
