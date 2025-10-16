"""
FreqtradeToRootAdapter - Convertit stratégies Freqtrade vers format ROOT.
Permet d'importer stratégies de la communauté Freqtrade dans l'écosystème ROOT.
"""

from .data_converter import DataConverter
from strategies.base_strategy import BaseStrategy
import logging
import os
import sys
from typing import Any, Dict, Optional, Type

import pandas as pd

# Ajouter le path pour imports ROOT
analyzer_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../"))
sys.path.append(analyzer_root)


# Import conditionnel Freqtrade
try:
    from freqtrade.strategy import IStrategy

    FREQTRADE_AVAILABLE = True
except ImportError:
    FREQTRADE_AVAILABLE = False
    IStrategy = object

logger = logging.getLogger(__name__)


class FreqtradeToRootAdapter:
    """
    Adaptateur Freqtrade → ROOT.
    Convertit une stratégie Freqtrade en stratégie ROOT pour utilisation en production.
    """

    def __init__(self, freqtrade_strategy_class: Type[IStrategy]):
        """
        Initialise l'adaptateur avec une classe de stratégie Freqtrade.

        Args:
            freqtrade_strategy_class: Classe héritant de IStrategy
        """
        if not FREQTRADE_AVAILABLE:
            raise ImportError(
                "Freqtrade n'est pas installé. "
                "Installez avec: pip install freqtrade")

        self.freqtrade_strategy_class = freqtrade_strategy_class
        self.strategy_name = freqtrade_strategy_class.__name__

        # Instancier la stratégie Freqtrade pour accéder à ses paramètres
        self.ft_strategy_instance = freqtrade_strategy_class()

    def convert(
        self, confidence_mapping: Optional[Dict[str, float]] = None
    ) -> Type[BaseStrategy]:
        """
        Convertit la stratégie Freqtrade en stratégie ROOT.

        Args:
            confidence_mapping: Mapping optionnel pour calculer confidence depuis tags/metadata
                               Ex: {'strong': 0.8, 'moderate': 0.6, 'weak': 0.4}

        Returns:
            Classe BaseStrategy compatible ROOT
        """
        ft_instance = self.ft_strategy_instance
        strategy_name = self.strategy_name

        # Mapping par défaut
        if confidence_mapping is None:
            confidence_mapping = {
                "very_strong": 0.85,
                "strong": 0.7,
                "moderate": 0.55,
                "weak": 0.4,
            }

        class AdaptedRootStrategy(BaseStrategy):
            """Stratégie ROOT générée depuis stratégie Freqtrade."""

            # Conserver référence à la stratégie Freqtrade originale
            _ft_strategy = ft_instance
            _confidence_map = confidence_mapping

            def __init__(
                self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]
            ):
                super().__init__(symbol, data, indicators)
                self.name = f"{strategy_name}_ROOT"

                # Métadonnées Freqtrade
                self.timeframe = getattr(self._ft_strategy, "timeframe", "5m")
                self.stoploss = getattr(self._ft_strategy, "stoploss", -0.05)
                self.minimal_roi = getattr(
                    self._ft_strategy, "minimal_roi", {})

            def generate_signal(self) -> Dict[str, Any]:
                """
                Génère un signal ROOT en appelant la stratégie Freqtrade.

                Returns:
                    Dict avec side, confidence, strength, reason, metadata
                """
                try:
                    # Valider données
                    if not self.validate_data():
                        return self._no_signal("Données invalides")

                    # Convertir données ROOT → DataFrame Freqtrade
                    df = DataConverter.root_to_dataframe(
                        data=self.data, indicators=self.indicators
                    )

                    if df.empty or len(df) < 2:
                        return self._no_signal("DataFrame insuffisant")

                    # Appeler populate_indicators
                    df = self._ft_strategy.populate_indicators(
                        df, {"pair": self.symbol}
                    )

                    # Appeler populate_entry_trend
                    df = self._ft_strategy.populate_entry_trend(
                        df, {"pair": self.symbol}
                    )

                    # Appeler populate_exit_trend
                    df = self._ft_strategy.populate_exit_trend(
                        df, {"pair": self.symbol}
                    )

                    # Extraire signal de la dernière ligne
                    last_row = df.iloc[-1]

                    # Analyser signal BUY
                    if last_row.get("enter_long", 0) == 1:
                        enter_tag = last_row.get("enter_tag", "")
                        confidence, strength = self._extract_confidence_strength(
                            enter_tag)

                        return {
                            "side": "BUY",
                            "confidence": confidence,
                            "strength": strength,
                            "reason": f"Freqtrade entry signal: {enter_tag}",
                            "metadata": {
                                "source": "freqtrade",
                                "strategy": strategy_name,
                                "enter_tag": enter_tag,
                                "timeframe": self.timeframe,
                                "stoploss": self.stoploss,
                            },
                        }

                    # Analyser signal SELL
                    if last_row.get("exit_long", 0) == 1:
                        exit_tag = last_row.get("exit_tag", "")
                        confidence, strength = self._extract_confidence_strength(
                            exit_tag)

                        return {
                            "side": "SELL",
                            "confidence": confidence,
                            "strength": strength,
                            "reason": f"Freqtrade exit signal: {exit_tag}",
                            "metadata": {
                                "source": "freqtrade",
                                "strategy": strategy_name,
                                "exit_tag": exit_tag,
                                "timeframe": self.timeframe,
                            },
                        }

                    # Analyser signal SHORT (si stratégie supporte)
                    if last_row.get("enter_short", 0) == 1:
                        # ROOT utilise uniquement SPOT (BUY/SELL), pas de SHORT
                        logger.debug("Signal SHORT ignoré (ROOT = SPOT only)")

                    # Pas de signal
                    return self._no_signal("Aucun signal Freqtrade")

                except Exception as e:
                    logger.error(
                        f"Erreur génération signal Freqtrade→ROOT: {e}")
                    return self._no_signal(f"Erreur: {str(e)}")

            def _extract_confidence_strength(
                    self, tag: str) -> tuple[float, str]:
                """
                Extrait confidence et strength depuis un tag Freqtrade.

                Args:
                    tag: Tag depuis enter_tag ou exit_tag

                Returns:
                    Tuple (confidence, strength)
                """
                # Convertir tag en lowercase pour matching
                tag_lower = tag.lower()

                # Chercher mots-clés dans le tag
                for keyword, confidence in sorted(
                        self._confidence_map.items(), key=lambda x: x[1], reverse=True):
                    if keyword in tag_lower:
                        strength = self.get_strength_from_confidence(
                            confidence)
                        return confidence, strength

                # Défaut: moderate
                return 0.55, "moderate"

            def _no_signal(self, reason: str) -> Dict[str, Any]:
                """Retourne un signal vide."""
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": reason,
                    "metadata": {
                        "source": "freqtrade",
                        "strategy": strategy_name},
                }

            def validate_data(self) -> bool:
                """Valide que les données nécessaires sont présentes."""
                if not self.data:
                    logger.warning(f"{self.name}: data manquant")
                    return False

                # Vérifier présence OHLCV
                required_fields = ["close_price", "volume"]
                for field in required_fields:
                    if field not in self.data or self.data[field] is None:
                        logger.warning(f"{self.name}: {field} manquant")
                        return False

                return True

        # Assigner nom dynamique
        AdaptedRootStrategy.__name__ = f"{strategy_name}_ROOT"
        AdaptedRootStrategy.__qualname__ = f"{strategy_name}_ROOT"

        return AdaptedRootStrategy

    def export_to_file(
            self,
            output_path: str,
            include_docstring: bool = True) -> None:
        """
        Exporte la stratégie convertie vers un fichier Python.

        Args:
            output_path: Chemin du fichier de sortie
            include_docstring: Inclure documentation originale
        """
        try:
            strategy_class = self.convert()

            # Générer code Python
            docstring = (
                f'"""\n{self.freqtrade_strategy_class.__doc__}\n"""'
                if include_docstring
                else ""
            )

            code = f'''"""
Stratégie ROOT générée automatiquement depuis Freqtrade.
Stratégie source: {self.strategy_name}
Généré par FreqtradeToRootAdapter
"""

from typing import Dict, Any
from strategies.base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class {strategy_class.__name__}(BaseStrategy):
    {docstring}

    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        self.name = "{strategy_class.__name__}"

    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal de trading.

        Note: Cette implémentation est un placeholder.
        Vous devez implémenter la logique depuis la stratégie Freqtrade originale.
        Voir: {self.freqtrade_strategy_class.__module__}
        """
        # TODO: Implémenter logique de signal
        return {{
            'side': None,
            'confidence': 0.0,
            'strength': 'weak',
            'reason': 'Non implémenté',
            'metadata': {{'source': 'freqtrade_adapter'}}
        }}
'''

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(code)

            logger.info(f"Stratégie exportée: {output_path}")

        except Exception as e:
            logger.error(f"Erreur export stratégie: {e}")
            raise

    def analyze_strategy(self) -> Dict[str, Any]:
        """
        Analyse la stratégie Freqtrade pour extraire métadonnées utiles.

        Returns:
            Dict avec informations sur la stratégie
        """
        try:
            ft_strategy = self.ft_strategy_instance

            info = {
                "name": self.strategy_name,
                "timeframe": getattr(ft_strategy, "timeframe", "unknown"),
                "stoploss": getattr(ft_strategy, "stoploss", None),
                "trailing_stop": getattr(ft_strategy, "trailing_stop", False),
                "minimal_roi": getattr(ft_strategy, "minimal_roi", {}),
                "use_exit_signal": getattr(ft_strategy, "use_exit_signal", True),
                "startup_candle_count": getattr(ft_strategy, "startup_candle_count", 0),
                "has_custom_indicators": hasattr(ft_strategy, "populate_indicators"),
                "has_entry_signal": hasattr(ft_strategy, "populate_entry_trend"),
                "has_exit_signal": hasattr(ft_strategy, "populate_exit_trend"),
                "supports_short": hasattr(ft_strategy, "can_short")
                and ft_strategy.can_short,
            }

            return info

        except Exception as e:
            logger.error(f"Erreur analyse stratégie: {e}")
            return {"error": str(e)}
