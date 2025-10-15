"""
RootToFreqtradeAdapter - Convertit stratégies ROOT vers format Freqtrade.
Permet de backtester des stratégies ROOT avec l'engine Freqtrade.
"""

import sys
import os
from typing import Dict, Any, Optional, Type
import pandas as pd
import logging

# Ajouter le path pour imports ROOT
analyzer_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(analyzer_root)

from strategies.base_strategy import BaseStrategy
from .data_converter import DataConverter

# Import conditionnel Freqtrade (peut ne pas être installé)
try:
    from freqtrade.strategy import IStrategy
    import talib.abstract as ta

    FREQTRADE_AVAILABLE = True
except ImportError:
    FREQTRADE_AVAILABLE = False
    IStrategy = object  # Fallback pour éviter erreurs

logger = logging.getLogger(__name__)


class RootToFreqtradeAdapter:
    """
    Adaptateur ROOT → Freqtrade.
    Convertit une stratégie ROOT en stratégie Freqtrade pour backtesting.
    """

    def __init__(self, root_strategy_class: Type[BaseStrategy]):
        """
        Initialise l'adaptateur avec une classe de stratégie ROOT.

        Args:
            root_strategy_class: Classe héritant de BaseStrategy
        """
        if not FREQTRADE_AVAILABLE:
            raise ImportError(
                "Freqtrade n'est pas installé. " "Installez avec: pip install freqtrade"
            )

        self.root_strategy_class = root_strategy_class
        self.strategy_name = root_strategy_class.__name__

    def convert(self) -> Type[IStrategy]:
        """
        Convertit la stratégie ROOT en stratégie Freqtrade.

        Returns:
            Classe IStrategy compatible Freqtrade
        """
        root_strategy_class = self.root_strategy_class
        strategy_name = self.strategy_name

        class AdaptedFreqtradeStrategy(IStrategy):
            """Stratégie Freqtrade générée depuis stratégie ROOT."""

            # Métadonnées Freqtrade
            minimal_roi = {
                "0": 0.10,  # 10% ROI
                "30": 0.05,  # 5% après 30min
                "60": 0.02,  # 2% après 1h
                "120": 0.01,  # 1% après 2h
            }

            stoploss = -0.05  # -5% stop loss

            trailing_stop = True
            trailing_stop_positive = 0.01
            trailing_stop_positive_offset = 0.02
            trailing_only_offset_is_reached = True

            timeframe = "5m"

            # Désactiver confirmations supplémentaires
            use_exit_signal = True
            exit_profit_only = False
            ignore_roi_if_entry_signal = False

            # Process uniquement nouveaux chandeliers
            process_only_new_candles = True

            # Paramètres d'ordre
            order_types = {
                "entry": "limit",
                "exit": "limit",
                "stoploss": "market",
                "stoploss_on_exchange": False,
            }

            # Conserve référence à la classe ROOT originale
            _root_strategy_class = root_strategy_class

            def populate_indicators(
                self, dataframe: pd.DataFrame, metadata: dict
            ) -> pd.DataFrame:
                """
                Calcule les indicateurs nécessaires.
                Note: On suppose que ROOT utilise des indicateurs déjà calculés.
                Ici on ajoute des indicateurs de base si nécessaires.
                """
                # Ajouter indicateurs techniques de base si pas déjà présents

                # EMA
                for period in [7, 12, 26, 50, 99, 200]:
                    col_name = f"ema_{period}"
                    if col_name not in dataframe.columns:
                        dataframe[col_name] = ta.EMA(dataframe, timeperiod=period)

                # SMA
                for period in [20, 50, 100, 200]:
                    col_name = f"sma_{period}"
                    if col_name not in dataframe.columns:
                        dataframe[col_name] = ta.SMA(dataframe, timeperiod=period)

                # RSI
                if "rsi" not in dataframe.columns:
                    dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

                # MACD
                if "macd_line" not in dataframe.columns:
                    macd = ta.MACD(
                        dataframe, fastperiod=12, slowperiod=26, signalperiod=9
                    )
                    dataframe["macd_line"] = macd["macd"]
                    dataframe["macd_signal"] = macd["macdsignal"]
                    dataframe["macd_histogram"] = macd["macdhist"]

                # Bollinger Bands
                if "bb_upperband" not in dataframe.columns:
                    bollinger = ta.BBANDS(
                        dataframe, timeperiod=20, nbdevup=2, nbdevdn=2
                    )
                    dataframe["bb_upperband"] = bollinger["upperband"]
                    dataframe["bb_middleband"] = bollinger["middleband"]
                    dataframe["bb_lowerband"] = bollinger["lowerband"]

                # ATR
                if "atr" not in dataframe.columns:
                    dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

                # ADX
                if "adx" not in dataframe.columns:
                    dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

                # Volume moyens
                if "volume_ma_20" not in dataframe.columns:
                    dataframe["volume_ma_20"] = (
                        dataframe["volume"].rolling(window=20).mean()
                    )

                return dataframe

            def populate_entry_trend(
                self, dataframe: pd.DataFrame, metadata: dict
            ) -> pd.DataFrame:
                """
                Génère les signaux d'entrée en appelant la stratégie ROOT.
                """
                # Initialiser les colonnes de signal
                dataframe["enter_long"] = 0
                dataframe["enter_tag"] = ""

                # Pour chaque ligne (chandeliers), évaluer la stratégie ROOT
                for idx in range(len(dataframe)):
                    try:
                        # Convertir ligne actuelle en format ROOT
                        current_data, current_indicators = self._row_to_root_format(
                            dataframe.iloc[: idx + 1], metadata.get("pair", "UNKNOWN")
                        )

                        # Instancier stratégie ROOT
                        root_strategy = self._root_strategy_class(
                            symbol=metadata.get("pair", "UNKNOWN"),
                            data=current_data,
                            indicators=current_indicators,
                        )

                        # Générer signal
                        signal = root_strategy.generate_signal()

                        # Interpréter signal ROOT
                        if signal and signal.get("side") == "BUY":
                            confidence = signal.get("confidence", 0.5)

                            # Filtrer signaux faibles (confidence < 0.4)
                            if confidence >= 0.4:
                                dataframe.loc[dataframe.index[idx], "enter_long"] = 1
                                dataframe.loc[dataframe.index[idx], "enter_tag"] = (
                                    f"{strategy_name}_{signal.get('strength', 'moderate')}"
                                )

                    except Exception as e:
                        logger.debug(f"Erreur évaluation stratégie ROOT idx {idx}: {e}")
                        continue

                return dataframe

            def populate_exit_trend(
                self, dataframe: pd.DataFrame, metadata: dict
            ) -> pd.DataFrame:
                """
                Génère les signaux de sortie en appelant la stratégie ROOT.
                """
                # Initialiser colonnes de signal
                dataframe["exit_long"] = 0
                dataframe["exit_tag"] = ""

                # Pour chaque ligne, évaluer la stratégie ROOT
                for idx in range(len(dataframe)):
                    try:
                        # Convertir ligne actuelle en format ROOT
                        current_data, current_indicators = self._row_to_root_format(
                            dataframe.iloc[: idx + 1], metadata.get("pair", "UNKNOWN")
                        )

                        # Instancier stratégie ROOT
                        root_strategy = self._root_strategy_class(
                            symbol=metadata.get("pair", "UNKNOWN"),
                            data=current_data,
                            indicators=current_indicators,
                        )

                        # Générer signal
                        signal = root_strategy.generate_signal()

                        # Interpréter signal ROOT
                        if signal and signal.get("side") == "SELL":
                            confidence = signal.get("confidence", 0.5)

                            # Filtrer signaux faibles
                            if confidence >= 0.4:
                                dataframe.loc[dataframe.index[idx], "exit_long"] = 1
                                dataframe.loc[dataframe.index[idx], "exit_tag"] = (
                                    f"{strategy_name}_{signal.get('strength', 'moderate')}"
                                )

                    except Exception as e:
                        logger.debug(
                            f"Erreur évaluation stratégie ROOT exit idx {idx}: {e}"
                        )
                        continue

                return dataframe

            def _row_to_root_format(
                self, dataframe_slice: pd.DataFrame, symbol: str
            ) -> tuple[Dict[str, Any], Dict[str, Any]]:
                """
                Convertit une ligne de DataFrame en format ROOT (data + indicators).

                Args:
                    dataframe_slice: Slice du DataFrame jusqu'à la ligne actuelle
                    symbol: Symbole de trading

                Returns:
                    Tuple (data, indicators) format ROOT
                """
                return DataConverter.dataframe_to_root(dataframe_slice, symbol)

        # Assigner nom dynamique
        AdaptedFreqtradeStrategy.__name__ = f"{strategy_name}_Freqtrade"
        AdaptedFreqtradeStrategy.__qualname__ = f"{strategy_name}_Freqtrade"

        return AdaptedFreqtradeStrategy

    def export_to_file(self, output_path: str) -> None:
        """
        Exporte la stratégie convertie vers un fichier Python.

        Args:
            output_path: Chemin du fichier de sortie
        """
        try:
            strategy_class = self.convert()

            # Générer code Python
            code = f'''"""
Stratégie Freqtrade générée automatiquement depuis ROOT.
Stratégie source: {self.strategy_name}
Généré par RootToFreqtradeAdapter
"""

from freqtrade.strategy import IStrategy
import talib.abstract as ta
import pandas as pd


class {strategy_class.__name__}(IStrategy):
    """Stratégie {self.strategy_name} adaptée pour Freqtrade."""

    minimal_roi = {strategy_class.minimal_roi}
    stoploss = {strategy_class.stoploss}
    trailing_stop = {strategy_class.trailing_stop}
    timeframe = '{strategy_class.timeframe}'

    # TODO: Implémenter populate_indicators, populate_entry_trend, populate_exit_trend
    # Voir stratégie ROOT source: {self.root_strategy_class.__module__}
'''

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(code)

            logger.info(f"Stratégie exportée: {output_path}")

        except Exception as e:
            logger.error(f"Erreur export stratégie: {e}")
            raise
