"""
MACD_Crossover_Strategy - Stratégie basée sur les croisements MACD.
"""

import logging
import math
from typing import Any

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MACD_Crossover_Strategy(BaseStrategy):
    """
    Stratégie utilisant les croisements MACD pour détecter les changements de momentum.

    MACD = EMA12 - EMA26, Signal = EMA9 du MACD, Histogram = MACD - Signal

    Signaux générés:
    - BUY: MACD croise au-dessus Signal + confirmations haussières
    - SELL: MACD croise en-dessous Signal + confirmations baissières
    """

    def __init__(self, symbol: str, data: dict[str, Any], indicators: dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres MACD ASSOUPLIS (moins de sur-filtrage)
        self.min_macd_distance = 0.005  # Distance durcie (0.002 → 0.005)
        self.histogram_threshold = 0.001  # Seuil histogram assoupli
        self.zero_line_bonus = 0.08  # Bonus maintenu
        # Paramètres de filtre de tendance
        self.trend_filter_enabled = True  # Activer le filtre de tendance globale
        self.contra_trend_penalty = 0.15  # Pénalité réduite (0.3 → 0.15)
        # FILTRES ASSOUPLIS pour plus de signaux
        self.min_confidence_threshold = 0.40  # Réduit (60% → 40%)
        self.strong_separation_threshold = 0.01  # Séparation forte réduite
        self.require_histogram_confirmation = False  # DÉSACTIVÉ (trop strict)
        self.min_confluence_bonus = 55  # Confluence en bonus (pas obligatoire)

    def _get_current_values(self) -> dict[str, float | None]:
        """Récupère les valeurs actuelles des indicateurs MACD."""
        return {
            # MACD complet
            "macd_line": self.indicators.get("macd_line"),
            "macd_signal": self.indicators.get("macd_signal"),
            "macd_histogram": self.indicators.get("macd_histogram"),
            "macd_zero_cross": self.indicators.get("macd_zero_cross"),
            "macd_signal_cross": self.indicators.get("macd_signal_cross"),
            "macd_trend": self.indicators.get("macd_trend"),
            # PPO (Percentage Price Oscillator - MACD normalisé)
            "ppo": self.indicators.get("ppo"),
            # EMA pour contexte (MACD = EMA12 - EMA26)
            "ema_12": self.indicators.get("ema_12"),
            "ema_26": self.indicators.get("ema_26"),
            "ema_50": self.indicators.get("ema_50"),
            # Trend et momentum pour confirmation
            "trend_strength": self.indicators.get("trend_strength"),
            "directional_bias": self.indicators.get("directional_bias"),
            "momentum_score": self.indicators.get("momentum_score"),
            # Oscillateurs pour confluence
            "rsi_14": self.indicators.get("rsi_14"),
            "stoch_k": self.indicators.get("stoch_k"),
            "stoch_d": self.indicators.get("stoch_d"),
            # Volume pour confirmation
            "volume_ratio": self.indicators.get("volume_ratio"),
            "volume_quality_score": self.indicators.get("volume_quality_score"),
            # Contexte marché
            "market_regime": self.indicators.get("market_regime"),
            "volatility_regime": self.indicators.get("volatility_regime"),
            # Confluence
            "signal_strength": self.indicators.get("signal_strength"),
            "confluence_score": self.indicators.get("confluence_score"),
            # Indicateurs de tendance globale
            "regime_strength": self.indicators.get("regime_strength"),
            "trend_alignment": self.indicators.get("trend_alignment"),
            "adx_14": self.indicators.get("adx_14"),
        }

    def _get_current_price(self) -> float | None:
        """Récupère le prix actuel depuis les données OHLCV."""
        try:
            if self.data and "close" in self.data and self.data["close"]:
                return float(self.data["close"][-1])
        except (IndexError, ValueError, TypeError):
            pass
        return None

    def _validate_macd_data(self, _is_valid) -> tuple[
        bool,
        dict[str, Any] | None,
        dict[str, Any] | None,
        float | None,
        float | None,
        float | None,
        float | None,
    ]:
        """Valide les données MACD. Returns (is_valid, error_response, values, macd_line, macd_signal, macd_histogram, macd_distance)."""
        if not self.validate_data():
            return (
                False,
                {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "Données insuffisantes",
                    "metadata": {"strategy": self.name},
                },
                None,
                None,
                None,
                None,
                None,
            )

        values = self._get_current_values()

        # Vérification des indicateurs MACD essentiels
        try:
            macd_line_val = values.get("macd_line")
            macd_signal_val = values.get("macd_signal")
            macd_histogram_val = values.get("macd_histogram")

            macd_line = (
                float(macd_line_val)
                if macd_line_val is not None and _is_valid(macd_line_val)
                else None
            )
            macd_signal = (
                float(macd_signal_val)
                if macd_signal_val is not None and _is_valid(macd_signal_val)
                else None
            )
            macd_histogram = (
                float(macd_histogram_val)
                if macd_histogram_val is not None and _is_valid(macd_histogram_val)
                else None
            )
        except (ValueError, TypeError) as e:
            return (
                False,
                {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Erreur conversion MACD: {e}",
                    "metadata": {"strategy": self.name},
                },
                None,
                None,
                None,
                None,
                None,
            )

        if not (_is_valid(macd_line) and _is_valid(macd_signal)):
            return (
                False,
                {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "MACD line ou signal invalides/NaN",
                    "metadata": {"strategy": self.name},
                },
                None,
                None,
                None,
                None,
                None,
            )

        if macd_line is None or macd_signal is None:
            return (
                False,
                {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "MACD line ou signal est None",
                    "metadata": {"strategy": self.name},
                },
                None,
                None,
                None,
                None,
                None,
            )

        macd_distance = abs(macd_line - macd_signal)
        if macd_distance < self.min_macd_distance:
            return (
                False,
                {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Rejet MACD: séparation insuffisante ({macd_distance:.4f} < {self.min_macd_distance})",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "macd_line": macd_line,
                        "macd_signal": macd_signal,
                        "distance": macd_distance,
                    },
                },
                None,
                None,
                None,
                None,
                None,
            )

        return True, None, values, macd_line, macd_signal, macd_histogram, macd_distance

    def _validate_macd_signal_requirements(
        self,
        signal_side: str,
        macd_histogram: float | None,
        rsi_14: float | None,
        is_strong_uptrend: bool,
        is_strong_downtrend: bool,
        _is_valid,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Valide les exigences pour un signal MACD. Returns (is_valid, rejection_response)."""
        validations = [
            ("histogram_buy", lambda: signal_side == "BUY" and macd_histogram is not None and _is_valid(macd_histogram) and macd_histogram < -0.001,
             lambda: f"Rejet MACD BUY: histogram contradictoire ({macd_histogram:.4f})"),
            ("histogram_sell", lambda: signal_side == "SELL" and macd_histogram is not None and _is_valid(macd_histogram) and macd_histogram > 0.001,
             lambda: f"Rejet MACD SELL: histogram contradictoire ({macd_histogram:.4f})"),
            ("downtrend_buy", lambda: signal_side == "BUY" and is_strong_downtrend,
             lambda: "Rejet MACD BUY: regime fortement baissier"),
            ("uptrend_sell", lambda: signal_side == "SELL" and is_strong_uptrend,
             lambda: "Rejet MACD SELL: regime fortement haussier"),
        ]

        for _, condition, message_fn in validations:
            if condition():
                return False, {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": message_fn(),
                    "metadata": {"strategy": self.name},
                }

        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if (signal_side == "BUY" and rsi >= 80) or (signal_side == "SELL" and rsi <= 20):
                    return False, {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet MACD {signal_side}: RSI {'surachat' if signal_side == 'BUY' else 'survente'} ({rsi:.1f})",
                        "metadata": {"strategy": self.name},
                    }
            except (ValueError, TypeError):
                pass

        return True, None

    def generate_signal(self) -> dict[str, Any]:
        """
        Génère un signal basé sur les croisements MACD.
        """

        # Helper pour valider les nombres (anti-NaN)
        def _is_valid(x):
            try:
                x = float(x) if x is not None else None
                return x is not None and not math.isnan(x)
            except (TypeError, ValueError):
                return False

        # VALIDATIONS PRÉLIMINAIRES GROUPÉES
        (
            is_valid,
            error_response,
            values,
            macd_line,
            macd_signal,
            macd_histogram,
            macd_distance,
        ) = self._validate_macd_data(_is_valid)
        if not is_valid:
            return error_response

        current_price = self._get_current_price()

        # Confluence maintenant OPTIONNELLE (bonus seulement)
        confluence_score = values.get("confluence_score")
        confluence_penalty = 0.0
        if confluence_score is not None and _is_valid(confluence_score):
            conf_val = float(confluence_score)
            if conf_val < 30:  # Très faible = pénalité
                confluence_penalty = -0.10
        elif not _is_valid(confluence_score):
            confluence_penalty = -0.05  # Pénalité légère si absent

        macd_above_signal = macd_line > macd_signal

        signal_side = None
        reason = ""
        base_confidence = 0.65  # Standardisé à 0.50 pour équité avec autres stratégies
        confidence_boost = 0.0
        cross_type = None

        # Filtre de tendance global AVANT de décider du signal
        market_regime = values.get("market_regime")
        trend_alignment = values.get("trend_alignment")
        regime_strength = values.get("regime_strength")
        adx_value = values.get("adx_14")

        # Déterminer la tendance principale
        is_strong_uptrend = False
        is_strong_downtrend = False

        if self.trend_filter_enabled and market_regime:
            market_regime_upper = str(market_regime).upper()
            if market_regime_upper in ["TRENDING_BULL", "BREAKOUT_BULL"]:
                is_strong_uptrend = True
            elif market_regime_upper in ["TRENDING_BEAR", "BREAKOUT_BEAR"]:
                is_strong_downtrend = True

        # Vérification supplémentaire avec trend_alignment (format 0-1 décimal)
        if trend_alignment is not None:
            try:
                alignment = float(trend_alignment)
                if alignment > 0.6:  # Forte tendance haussière (format 0-1)
                    is_strong_uptrend = True
                elif alignment < 0.4:  # Forte tendance baissière (format 0-1)
                    is_strong_downtrend = True
            except (ValueError, TypeError):
                pass

        # Conditions assouplies (moins de sur-filtrage)
        conditions_bonus = 0.0

        # Logique MACD assouplie (pénalités vs rejets)
        if macd_above_signal:
            # MACD au-dessus du signal - BUY
            signal_side = "BUY"
            cross_type = "bullish_cross"
            reason = f"MACD ({macd_line:.4f}) > Signal ({macd_signal:.4f})"
            confidence_boost += 0.10

            # Bonus conditions favorables
            if is_strong_uptrend:
                conditions_bonus += 0.15
                reason += " + tendance haussière forte"
            elif macd_line is not None and macd_line > 0:
                conditions_bonus += 0.08
                reason += " + MACD positif"

        else:
            # MACD en-dessous du signal - SELL
            signal_side = "SELL"
            cross_type = "bearish_cross"
            reason = f"MACD ({macd_line:.4f}) < Signal ({macd_signal:.4f})"
            confidence_boost += 0.10

            # Bonus conditions favorables
            if is_strong_downtrend:
                conditions_bonus += 0.15
                reason += " + tendance baissière forte"
            elif macd_line is not None and macd_line < 0:
                conditions_bonus += 0.08
                reason += " + MACD négatif"

        # VALIDATIONS SIGNAL REQUIREMENTS
        rsi_14 = values.get("rsi_14")
        is_valid_signal, rejection_response = self._validate_macd_signal_requirements(
            signal_side,
            macd_histogram,
            rsi_14,
            is_strong_uptrend,
            is_strong_downtrend,
            _is_valid,
        )
        if not is_valid_signal:
            return rejection_response

        # Appliquer bonus des conditions
        confidence_boost += conditions_bonus

        # Bonus selon la force de la séparation - SEUILS PLUS STRICTS
        separation_strength = (
            abs(macd_line - macd_signal)
            if macd_line is not None and macd_signal is not None
            else 0.0
        )
        if separation_strength >= self.strong_separation_threshold:  # 0.02
            confidence_boost += 0.18
            reason += f" - séparation TRÈS forte ({separation_strength:.4f})"
        elif separation_strength >= 0.01:
            confidence_boost += 0.12
            reason += f" - séparation forte ({separation_strength:.4f})"
        elif separation_strength >= 0.007:
            confidence_boost += 0.06
            reason += f" - séparation modérée ({separation_strength:.4f})"
        else:
            # Séparation trop faible - pénalité
            confidence_boost -= 0.05
            reason += f" ATTENTION: séparation faible ({separation_strength:.4f})"

        # Confirmation avec Histogram MACD - SEUILS PLUS STRICTS
        if macd_histogram is not None:
            if (
                signal_side == "BUY" and macd_histogram > self.histogram_threshold * 2
            ):  # Double seuil
                confidence_boost += 0.18
                reason += f" + histogram TRÈS positif ({macd_histogram:.4f})"
            elif (
                signal_side == "SELL" and macd_histogram < -self.histogram_threshold * 2
            ):  # Double seuil
                confidence_boost += 0.18
                reason += f" + histogram TRÈS négatif ({macd_histogram:.4f})"
            elif signal_side == "BUY" and macd_histogram > self.histogram_threshold:
                confidence_boost += 0.12
                reason += f" + histogram positif ({macd_histogram:.4f})"
            elif signal_side == "SELL" and macd_histogram < -self.histogram_threshold:
                confidence_boost += 0.12
                reason += f" + histogram négatif ({macd_histogram:.4f})"
            elif (signal_side == "BUY" and macd_histogram > 0) or (
                signal_side == "SELL" and macd_histogram < 0
            ):
                confidence_boost += 0.06  # Réduit de 0.10
                reason += f" + histogram favorable ({macd_histogram:.4f})"
            else:
                confidence_boost -= 0.10  # Pénalité augmentée
                reason += f" MAIS histogram CONTRADICTOIRE ({macd_histogram:.4f})"

        # Bonus si MACD dans la bonne zone par rapport à zéro
        if macd_line is not None:
            if signal_side == "BUY" and macd_line > 0:
                confidence_boost += self.zero_line_bonus
                reason += " + MACD au-dessus zéro"
            elif signal_side == "SELL" and macd_line < 0:
                confidence_boost += self.zero_line_bonus
                reason += " + MACD en-dessous zéro"
            elif signal_side == "BUY" and macd_line < -0.01:
                confidence_boost -= 0.05
                reason += " mais MACD très négatif"
            elif signal_side == "SELL" and macd_line > 0.01:
                confidence_boost -= 0.05
                reason += " mais MACD très positif"

        # Confirmation avec macd_trend pré-calculé
        macd_trend = values.get("macd_trend")
        if macd_trend and (
            (signal_side == "BUY" and macd_trend == "BULLISH")
            or (signal_side == "SELL" and macd_trend == "BEARISH")
        ):
            confidence_boost += 0.10
            reason += f" + trend MACD {macd_trend}"

        # Confirmation avec EMA (base du MACD)
        ema_12 = values.get("ema_12")
        ema_26 = values.get("ema_26")
        ema_50 = values.get("ema_50")

        if ema_12 is not None and ema_26 is not None:
            try:
                ema12_val = float(ema_12)
                ema26_val = float(ema_26)
                ema_cross_matches = (
                    signal_side == "BUY" and ema12_val > ema26_val
                ) or (signal_side == "SELL" and ema12_val < ema26_val)

                if ema_cross_matches:
                    confidence_boost += 0.10
                    reason += " + EMA confirme"
                else:
                    confidence_boost -= 0.05
                    reason += " mais EMA diverge"
            except (ValueError, TypeError):
                pass

        # Confirmation avec EMA 50 pour filtre de tendance
        if ema_50 is not None and current_price is not None:
            try:
                ema50_val = float(ema_50)
                if signal_side == "BUY" and current_price > ema50_val:
                    confidence_boost += 0.08
                    reason += " + prix > EMA50"
                elif signal_side == "SELL" and current_price < ema50_val:
                    confidence_boost += 0.08
                    reason += " + prix < EMA50"
            except (ValueError, TypeError):
                pass

        # Confirmation avec trend_strength (VARCHAR:
        # absent/weak/moderate/strong/very_strong)
        trend_strength = values.get("trend_strength")
        if trend_strength is not None:
            trend_str = str(trend_strength).lower()
            if trend_str in ["strong", "very_strong"]:
                confidence_boost += 0.12
                reason += f" + tendance {trend_str}"
            elif trend_str == "moderate":
                confidence_boost += 0.08
                reason += f" + tendance {trend_str}"

        # Confirmation avec directional_bias
        directional_bias = values.get("directional_bias")
        if directional_bias and (
            (signal_side == "BUY" and directional_bias == "BULLISH")
            or (signal_side == "SELL" and directional_bias == "BEARISH")
        ):
            confidence_boost += 0.10
            reason += f" + bias {directional_bias}"

        # Momentum score pour confluence
        momentum_score = values.get("momentum_score")
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                # Format 0-100, 50=neutre
                if (signal_side == "BUY" and momentum > 55) or (
                    signal_side == "SELL" and momentum < 45
                ):
                    confidence_boost += 0.08
                    reason += " + momentum favorable"
                elif (signal_side == "BUY" and momentum < 35) or (
                    signal_side == "SELL" and momentum > 65
                ):
                    confidence_boost -= 0.10
                    reason += " mais momentum défavorable"
            except (ValueError, TypeError):
                pass

        # Confirmation avec RSI (éviter zones extrêmes)
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if (signal_side == "BUY" and rsi < 70) or (
                    signal_side == "SELL" and rsi > 30
                ):
                    confidence_boost += 0.05
            except (ValueError, TypeError):
                pass

        # Stochastic pour confluence
        stoch_k = values.get("stoch_k")
        stoch_d = values.get("stoch_d")
        if stoch_k is not None and stoch_d is not None:
            try:
                k = float(stoch_k) if stoch_k is not None else 0.0
                d = float(stoch_d) if stoch_d is not None else 0.0
                stoch_cross = k > d

                if (signal_side == "BUY" and stoch_cross) or (
                    signal_side == "SELL" and not stoch_cross
                ):
                    confidence_boost += 0.08
                    reason += " + Stoch confirme"
            except (ValueError, TypeError):
                pass

        # Volume pour confirmation
        volume_ratio = values.get("volume_ratio")
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= 2.0:
                    confidence_boost += 0.15
                    reason += f" + volume exceptionnel ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.5:
                    confidence_boost += 0.08
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass

        # Market regime en BONUS (plus de bannissement)
        market_regime_val = values.get("market_regime")
        if market_regime_val:
            regime_upper = str(market_regime_val).upper()
            if regime_upper in [
                "TRENDING_BULL",
                "TRENDING_BEAR",
                "BREAKOUT_BULL",
                "BREAKOUT_BEAR",
            ]:
                confidence_boost += 0.12
                reason += f" + {regime_upper.lower()}"
            elif regime_upper == "RANGING":
                # MACD peut être utile en ranging pour oscillations
                confidence_boost += 0.05
                reason += " (ranging - oscillations)"
            elif regime_upper == "VOLATILE":
                confidence_boost -= 0.05  # Pénalité légère, pas bannissement
                reason += " (volatil)"
            elif regime_upper == "TRANSITION":
                confidence_boost += 0.02
                reason += " (transition)"

        # PPO pour confirmation (MACD normalisé)
        ppo = values.get("ppo")
        if ppo is not None:
            try:
                ppo_val = float(ppo)
                if (signal_side == "BUY" and ppo_val > 0) or (
                    signal_side == "SELL" and ppo_val < 0
                ):
                    confidence_boost += 0.05
                    reason += f" + PPO confirme ({ppo_val:.3f})"
            except (ValueError, TypeError):
                pass

        # Signal strength (VARCHAR: WEAK/MODERATE/STRONG)
        signal_strength_calc = values.get("signal_strength")
        if signal_strength_calc is not None:
            sig_str = str(signal_strength_calc).upper()
            if sig_str == "STRONG":
                confidence_boost += 0.10
                reason += " + signal fort"
            elif sig_str == "MODERATE":
                confidence_boost += 0.05
                reason += " + signal modéré"

        # Confluence OPTIONNELLE avec bonus
        if confluence_score is not None and _is_valid(confluence_score):
            try:
                confluence = float(confluence_score)
                if confluence > 80:
                    confidence_boost += 0.15
                    reason += f" + confluence PARFAITE ({confluence:.0f})"
                elif confluence > 70:
                    confidence_boost += 0.12
                    reason += f" + confluence excellente ({confluence:.0f})"
                elif confluence > self.min_confluence_bonus:
                    confidence_boost += 0.08
                    reason += f" + confluence bonne ({confluence:.0f})"
            except (ValueError, TypeError):
                pass

        # Appliquer pénalité confluence uniquement
        confidence_boost += confluence_penalty

        # CALCUL FINAL avec modèle standard
        confidence = max(
            0.0,
            min(1.0, self.calculate_confidence(base_confidence, 1 + confidence_boost)),
        )

        # Filtre final confidence
        if confidence < self.min_confidence_threshold:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"MACD {signal_side} rejeté - confidence insuffisante ({confidence:.2f} < {self.min_confidence_threshold})",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "rejected_signal": signal_side,
                    "rejected_confidence": confidence,
                    "min_required": self.min_confidence_threshold,
                    "separation_strength": separation_strength,
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
                "current_price": current_price,
                "macd_line": macd_line,
                "macd_signal": macd_signal,
                "macd_histogram": macd_histogram,
                "cross_type": cross_type,
                "macd_distance": macd_distance,
                "macd_trend": macd_trend,
                "ema_12": ema_12,
                "ema_26": ema_26,
                "ema_50": ema_50,
                "trend_strength": trend_strength,
                "directional_bias": directional_bias,
                "momentum_score": momentum_score,
                "rsi_14": rsi_14,
                "stoch_k": stoch_k,
                "stoch_d": stoch_d,
                "volume_ratio": volume_ratio,
                "market_regime": market_regime_val,
                "ppo": ppo,
                "confluence_score": confluence_score,
                "trend_alignment": trend_alignment,
                "regime_strength": regime_strength,
                "adx_14": adx_value,
            },
        }

    def validate_data(self) -> bool:
        """Valide que tous les indicateurs MACD requis sont présents."""
        if not super().validate_data():
            return False

        required = ["macd_line", "macd_signal"]

        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False

        return True
