"""
CCI_Reversal_Strategy - Stratégie basée sur le CCI et les indicateurs pré-calculés.
"""

import logging
from typing import Any

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class CCI_Reversal_Strategy(BaseStrategy):
    """
    Stratégie utilisant le CCI et les indicateurs pré-calculés pour détecter les retournements.

    Signaux générés:
    - BUY: CCI en zone de survente avec conditions favorables
    - SELL: CCI en zone de surachat avec conditions favorables
    """

    def __init__(self, symbol: str, data: dict[str, Any], indicators: dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres CCI durcis pour crypto intraday
        self.oversold_level = -150  # Zone survente stricte (crypto adapté)
        self.overbought_level = 150  # Zone surachat stricte (crypto adapté)
        self.extreme_oversold = -200  # Extrême vraiment extrême
        self.extreme_overbought = 200  # Extrême vraiment extrême

        # Paramètres de validation temporelle - SIMPLIFIÉS (STATELESS)
        # Pas de persistance requise (crypto 3m rapide)
        self.min_cci_persistence = 0
        # self.cci_history supprimé pour rendre la stratégie stateless
        # self.max_history_size supprimé

        # Seuils adaptatifs selon volatilité - OPTIMISÉS winrate
        self.volatility_adjustment = {
            "low": 0.9,  # Moins sensible même en faible vol
            "normal": 1.0,  # Seuils standards
            "high": 1.3,  # Plus strict en haute volatilité
            "extreme": 1.5,  # Très strict en volatilité extrême
        }

    def _create_rejection_signal(
        self, reason: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Helper to create rejection signals."""
        base_metadata = {"strategy": self.name}
        if metadata:
            base_metadata.update(metadata)
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": reason,
            "metadata": base_metadata,
        }

    def _validate_cci_data(
        self, values: dict[str, Any]
    ) -> tuple[float | None, dict[str, Any] | None]:
        """Valide les données CCI. Retourne (cci_20, error_signal)."""
        cci_20_raw = values["cci_20"]
        if cci_20_raw is None:
            return None, self._create_rejection_signal("CCI non disponible", {})

        try:
            cci_20 = float(cci_20_raw)
            return cci_20, None
        except (ValueError, TypeError):
            return None, self._create_rejection_signal(
                f"CCI invalide: {cci_20_raw}", {}
            )

    def _validate_signal_requirements(
        self, signal_side: str, values: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Valide les exigences du signal. Retourne un signal de rejet ou None si valide."""
        # Validation momentum
        momentum_score_raw = values.get("momentum_score")
        momentum_score = 0.0
        if momentum_score_raw is not None:
            try:
                momentum_score = float(momentum_score_raw)
                if signal_side == "BUY" and momentum_score < 30:
                    return self._create_rejection_signal(
                        f"Rejet BUY: momentum encore trop faible ({momentum_score:.1f})",
                        {"momentum_score": momentum_score},
                    )
                if signal_side == "SELL" and momentum_score > 70:
                    return self._create_rejection_signal(
                        f"Rejet SELL: momentum encore trop fort ({momentum_score:.1f})",
                        {"momentum_score": momentum_score},
                    )
            except (ValueError, TypeError):
                pass

        # Validation confluence
        confluence_score_raw = values.get("confluence_score")
        if confluence_score_raw is not None:
            try:
                confluence_score = float(confluence_score_raw)
                if confluence_score < 30:
                    return self._create_rejection_signal(
                        f"Rejet: confluence trop faible ({confluence_score:.0f} < 30)",
                        {"confluence_score": confluence_score},
                    )
            except (ValueError, TypeError):
                pass

        # Validation RSI
        rsi_raw = values.get("rsi_14")
        if rsi_raw is not None:
            try:
                rsi = float(rsi_raw)
                if signal_side == "BUY" and rsi > 65:
                    return self._create_rejection_signal(
                        f"Rejet BUY: RSI trop haut ({rsi:.1f}) pour reversal",
                        {"rsi": rsi},
                    )
                if signal_side == "SELL" and rsi < 35:
                    return self._create_rejection_signal(
                        f"Rejet SELL: RSI trop bas ({rsi:.1f}) pour reversal",
                        {"rsi": rsi},
                    )
            except (ValueError, TypeError):
                pass

        # Validation volume
        volume_ratio_raw = values.get("volume_ratio")
        if volume_ratio_raw is not None:
            try:
                volume_ratio = float(volume_ratio_raw)
                if volume_ratio < 0.2:
                    return self._create_rejection_signal(
                        f"Rejet: volume trop faible ({volume_ratio:.2f}x < 0.2x)",
                        {"volume_ratio": volume_ratio},
                    )
            except (ValueError, TypeError):
                pass

        return None

    def _get_current_values(self) -> dict[str, float | None]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            "cci_20": self.indicators.get("cci_20"),
            "momentum_score": self.indicators.get("momentum_score"),
            "trend_strength": self.indicators.get("trend_strength"),
            "directional_bias": self.indicators.get("directional_bias"),
            "confluence_score": self.indicators.get("confluence_score"),
            "signal_strength": self.indicators.get("signal_strength"),
            "pattern_detected": self.indicators.get("pattern_detected"),
            "pattern_confidence": self.indicators.get("pattern_confidence"),
            "market_regime": self.indicators.get("market_regime"),
            "regime_strength": self.indicators.get("regime_strength"),
            "volatility_regime": self.indicators.get("volatility_regime"),
            # Ajout volume
            "volume_ratio": self.indicators.get("volume_ratio"),
            # Ajout RSI pour confirmation
            "rsi_14": self.indicators.get("rsi_14"),
        }

    # Méthodes d'historique CCI supprimées - stratégie stateless
    # Pas de persistance temporelle requise en crypto 3m

    def _get_adjusted_thresholds(self, volatility_regime: str) -> dict[str, float]:
        """Ajuste les seuils selon le régime de volatilité."""
        adjustment = self.volatility_adjustment.get(volatility_regime, 1.0)
        return {
            "oversold": self.oversold_level * adjustment,
            "overbought": self.overbought_level * adjustment,
            "extreme_oversold": self.extreme_oversold * adjustment,
            "extreme_overbought": self.extreme_overbought * adjustment,
        }

    def generate_signal(self) -> dict[str, Any]:
        """
        Génère un signal basé sur le CCI et les indicateurs pré-calculés.
        """
        if not self.validate_data():
            return self._create_rejection_signal("Données insuffisantes", {})

        values = self._get_current_values()

        # Validation des données CCI
        cci_20, error_signal = self._validate_cci_data(values)
        if error_signal:
            return error_signal

        signal_side = None
        reason = ""
        confidence_boost = 0.0

        # Historique CCI supprimé - stratégie stateless

        # Ajustement des seuils selon volatilité
        volatility_regime = values.get("volatility_regime", "normal")
        thresholds = self._get_adjusted_thresholds(str(volatility_regime))

        # Logique de signal directe (crypto 3m réactif)
        if cci_20 <= thresholds["oversold"]:
            signal_side = "BUY"
            if cci_20 <= thresholds["extreme_oversold"]:
                zone = "survente extrême"
                confidence_boost += 0.15  # Bonus réduit
            else:
                zone = "survente"
                confidence_boost += 0.08  # Bonus réduit
            reason = f"CCI ({cci_20:.1f}) en zone de {zone}"

        elif cci_20 >= thresholds["overbought"]:
            signal_side = "SELL"
            if cci_20 >= thresholds["extreme_overbought"]:
                zone = "surachat extrême"
                confidence_boost += 0.15  # Bonus réduit
            else:
                zone = "surachat"
                confidence_boost += 0.08  # Bonus réduit
            reason = f"CCI ({cci_20:.1f}) en zone de {zone}"

        if signal_side:
            # Validation des exigences du signal
            error_signal = self._validate_signal_requirements(signal_side, values)
            if error_signal:
                return error_signal

            base_confidence = 0.65  # Base harmonisée avec autres stratégies

            # Récupération momentum pour calcul de boost
            momentum_score_raw = values.get("momentum_score")
            momentum_score = 0.0
            if momentum_score_raw is not None:
                try:
                    momentum_score = float(momentum_score_raw)
                except (ValueError, TypeError):
                    momentum_score = 0.0

            if momentum_score != 0:
                # Confirmations momentum de retournement
                if (signal_side == "BUY" and momentum_score > 45) or (
                    signal_side == "SELL" and momentum_score < 55
                ):
                    confidence_boost += 0.12
                    reason += " avec momentum de retournement"
                elif (signal_side == "BUY" and momentum_score > 35) or (
                    signal_side == "SELL" and momentum_score < 65
                ):
                    confidence_boost += 0.08
                    reason += " avec début retournement"

            # Utilisation du trend_strength
            trend_strength_raw = values.get("trend_strength")
            if trend_strength_raw and str(trend_strength_raw).lower() in ["strong"]:
                confidence_boost += 0.1
                reason += f" et tendance {str(trend_strength_raw).lower()}"

            # Utilisation du directional_bias
            directional_bias = values.get("directional_bias")
            if directional_bias:
                bias_upper = str(directional_bias).upper()
                if (signal_side == "BUY" and bias_upper == "BULLISH") or (
                    signal_side == "SELL" and bias_upper == "BEARISH"
                ):
                    confidence_boost += 0.1
                    reason += " confirmé par bias directionnel"
                elif (signal_side == "BUY" and bias_upper == "BEARISH") or (
                    signal_side == "SELL" and bias_upper == "BULLISH"
                ):
                    confidence_boost -= 0.1  # Contradictoire

            # Utilisation du confluence_score avec niveaux multiples
            confluence_score_raw = values.get("confluence_score")
            confluence_score = 0.0
            if confluence_score_raw is not None:
                try:
                    confluence_score = float(confluence_score_raw)
                except (ValueError, TypeError):
                    confluence_score = 0.0

            # Confirmations confluence
            if confluence_score > 80:
                confidence_boost += 0.15
                reason += " avec très haute confluence"
            elif confluence_score > 65:
                confidence_boost += 0.08
                reason += " avec confluence solide"

            # Utilisation du pattern_detected et pattern_confidence avec
            # conversion sécurisée
            pattern_detected = values.get("pattern_detected")
            pattern_confidence_raw = values.get("pattern_confidence")
            pattern_confidence = 0.0
            if pattern_confidence_raw is not None:
                try:
                    pattern_confidence = float(pattern_confidence_raw)
                except (ValueError, TypeError):
                    pattern_confidence = 0.0

            if pattern_detected and pattern_confidence > 60:
                confidence_boost += 0.1
                reason += f" avec pattern {pattern_detected}"

            # Utilisation du market_regime
            market_regime = values.get("market_regime")
            regime_strength_raw = values.get("regime_strength")

            if (
                market_regime
                and regime_strength_raw
                and str(regime_strength_raw).upper() in ["STRONG"]
            ) and (
                (
                    signal_side == "BUY"
                    and market_regime in ["TRENDING_BULL", "BREAKOUT_BULL"]
                )
                or (
                    signal_side == "SELL"
                    and market_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]
                )
            ):
                confidence_boost += 0.1
                reason += f" en régime {market_regime}"

            # Validation RSI pour confirmation
            rsi_raw = values.get("rsi_14")
            if rsi_raw is not None:
                try:
                    rsi = float(rsi_raw)
                    # Confirmations RSI
                    if (signal_side == "BUY" and rsi < 35) or (
                        signal_side == "SELL" and rsi > 65
                    ):
                        confidence_boost += 0.10
                        reason += f" confirmé par RSI ({rsi:.1f})"
                except (ValueError, TypeError):
                    pass

            # Validation volume pour confirmation
            volume_ratio_raw = values.get("volume_ratio")
            if volume_ratio_raw is not None:
                try:
                    volume_ratio = float(volume_ratio_raw)
                    if volume_ratio > 1.5:
                        confidence_boost += 0.08
                        reason += " avec volume élevé"
                except (ValueError, TypeError):
                    pass

            # Ajustement final selon volatilité - OPTIMISÉ
            if volatility_regime == "low":
                confidence_boost += 0.12  # Excellents signaux en faible volatilité
            elif volatility_regime == "normal":
                confidence_boost += 0.05  # Bonus standard
            elif volatility_regime == "high":
                confidence_boost -= 0.02  # Pénalité minimale
            elif volatility_regime == "extreme":
                confidence_boost -= 0.05  # Pénalité modérée

            # Utilisation du signal_strength pré-calculé
            signal_strength_calc_raw = values.get("signal_strength")
            if signal_strength_calc_raw:
                strength_upper = str(signal_strength_calc_raw).upper()
                if strength_upper == "STRONG":
                    confidence_boost += 0.1
                    reason += " + signal fort"
                elif strength_upper == "MODERATE":
                    confidence_boost += 0.05
                    reason += " + signal modéré"

            # Calcul final avec plafond et clamp strict
            total_boost = min(confidence_boost, 0.30)  # Boost total réduit
            confidence = self.calculate_confidence(base_confidence, 1 + total_boost)
            # Clamp explicite [0,1]
            confidence = min(1.0, max(0.0, confidence))

            # Filtre final : confidence minimum élevé
            if confidence < 0.45:  # Seuil plus strict pour qualité
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Signal CCI {signal_side} détecté mais confidence insuffisante ({confidence:.2f} < 0.45)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "cci_20": cci_20,
                        "rejected_confidence": confidence,
                    },
                }

            strength = self.get_strength_from_confidence(confidence)

            return {
                "side": signal_side,
                "confidence": confidence,
                "strength": strength,
                "reason": reason,
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "cci_20": cci_20,
                    "zone": zone,
                    "momentum_score": momentum_score,
                    "trend_strength": trend_strength_raw,
                    "directional_bias": directional_bias,
                    "confluence_score": confluence_score,
                    "pattern_detected": pattern_detected,
                    "pattern_confidence": pattern_confidence,
                    "market_regime": market_regime,
                    "volatility_regime": volatility_regime,
                },
            }

        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": f"CCI neutre ({cci_20:.1f}) - pas de zone extrême",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "cci_20": cci_20,
            },
        }
