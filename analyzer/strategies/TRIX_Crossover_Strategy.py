"""
TRIX_Crossover_Strategy - Stratégie basée sur TRIX simulé avec TEMA.
TRIX est un oscillateur basé sur le taux de changement pourcentuel d'une triple EMA lissée.
"""

import logging
from typing import Any

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class TRIX_Crossover_Strategy(BaseStrategy):
    """
    Stratégie utilisant TRIX simulé avec TEMA pour détecter les changements de momentum.

    TRIX (Triple Exponential Average) :
    - TRIX = (TEMA_today - TEMA_yesterday) / TEMA_yesterday * 10000
    - Indicateur de momentum basé sur la pente de TEMA
    - Filtre le bruit mieux que MACD grâce au triple lissage

    Comme TRIX n'est pas directement disponible, nous le simulons avec :
    - TEMA 12 périodes comme base
    - ROC sur TEMA pour approximer TRIX
    - DEMA 12 comme signal line (plus rapide que TEMA)

    Signaux générés:
    - BUY: TRIX simulé crosses above zero + confirmations haussières
    - SELL: TRIX simulé crosses below zero + confirmations baissières
    """

    def __init__(self, symbol: str, data: dict[str, Any], indicators: dict[str, Any]):
        super().__init__(symbol, data, indicators)

        # Paramètres TRIX simulé ASSOUPLIS basés sur données réelles DB
        # Seuil haussier 0.04% (médiane ROC)
        self.trix_bullish_threshold = 0.0004
        self.trix_bearish_threshold = -0.0004  # Seuil baissier -0.04%
        self.neutral_zone = 0.0002  # Zone neutre 0.02% (très petite)
        self.strong_trix_threshold = 0.01  # TRIX fort réduit à 1%
        self.extreme_trix_threshold = 0.03  # TRIX extrême réduit à 3%

        # Paramètres crossover signal line
        self.signal_line_crossover = True  # Utiliser DEMA comme signal line
        self.min_tema_dema_separation = (
            0.0005  # Séparation réduite à 0.05% avec fresh cross
        )

        # Paramètres momentum confirmation
        self.momentum_alignment_required = (
            False  # Désactiver l'exigence d'alignement strict
        )
        self.min_roc_confirmation = 0.001  # ROC minimum réduit à 0.1%

        # Paramètres filtrage
        self.min_trend_strength = "WEAK"  # Accepter même les trends faibles
        self.volume_confirmation_threshold = 1.2  # Volume confirmation assoupli
        self.strong_volume_threshold = 1.8  # Volume fort assoupli

    def _get_current_values(self) -> dict[str, float | None]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        # Calculer les valeurs précédentes depuis les données historiques
        tema_12_prev = None
        dema_12_prev = None
        roc_10_prev = None

        # Récupérer les valeurs précédentes depuis les listes de données
        if hasattr(self, "data") and self.data:
            close_list = self.data.get("close", [])
            if len(close_list) >= 2:
                # Simuler TEMA/DEMA précédents avec EMA approximative sur N-1
                try:
                    # Approximation: utiliser close[-2] pour estimer les MA
                    # précédentes
                    close_prev = float(close_list[-2])
                    close_curr = float(close_list[-1])

                    # Estimation grossière des MA précédentes (ratio de
                    # variation)
                    tema_curr = self.indicators.get("tema_12")
                    dema_curr = self.indicators.get("dema_12")
                    roc_curr = self.indicators.get("roc_10")

                    if tema_curr and close_curr and close_prev:
                        price_ratio = close_prev / close_curr
                        tema_12_prev = float(tema_curr) * price_ratio
                        dema_12_prev = (
                            float(dema_curr) * price_ratio if dema_curr else None
                        )

                    if (
                        roc_curr and len(close_list) >= 12
                    ):  # ROC(10) need 12 points for prev calc
                        # ROC précédent approximé: (close[-2] - close[-12]) /
                        # close[-12]
                        close_12_ago = float(close_list[-12])
                        roc_10_prev = (
                            (close_prev - close_12_ago) / close_12_ago
                            if close_12_ago != 0
                            else 0
                        )

                except (ValueError, TypeError, IndexError):
                    pass

        return {
            # Moyennes mobiles (base TRIX)
            "tema_12": self.indicators.get("tema_12"),
            "dema_12": self.indicators.get("dema_12"),
            "ema_12": self.indicators.get("ema_12"),
            "ema_26": self.indicators.get("ema_26"),
            "atr_14": self.indicators.get("atr_14"),
            # Valeurs précédentes calculées
            "tema_12_prev": tema_12_prev,
            "dema_12_prev": dema_12_prev,
            "roc_10_prev": roc_10_prev,
            # ROC et momentum (approximation TRIX)
            "roc_10": self.indicators.get("roc_10"),
            "roc_20": self.indicators.get("roc_20"),
            "momentum_10": self.indicators.get("momentum_10"),
            "momentum_score": self.indicators.get("momentum_score"),
            # Trend et direction
            "trend_strength": self.indicators.get("trend_strength"),
            "directional_bias": self.indicators.get("directional_bias"),
            "trend_alignment": self.indicators.get("trend_alignment"),
            "trend_angle": self.indicators.get("trend_angle"),
            # MACD (comparaison avec TRIX)
            "macd_line": self.indicators.get("macd_line"),
            "macd_signal": self.indicators.get("macd_signal"),
            "macd_histogram": self.indicators.get("macd_histogram"),
            "macd_zero_cross": self.indicators.get("macd_zero_cross"),
            # Volume et confirmation
            "volume_ratio": self.indicators.get("volume_ratio"),
            "relative_volume": self.indicators.get("relative_volume"),
            "volume_quality_score": self.indicators.get("volume_quality_score"),
            "trade_intensity": self.indicators.get("trade_intensity"),
            # Oscillateurs (confluence)
            "rsi_14": self.indicators.get("rsi_14"),
            "rsi_21": self.indicators.get("rsi_21"),
            "stoch_k": self.indicators.get("stoch_k"),
            "stoch_d": self.indicators.get("stoch_d"),
            # Market context
            "market_regime": self.indicators.get("market_regime"),
            "volatility_regime": self.indicators.get("volatility_regime"),
            "confluence_score": self.indicators.get("confluence_score"),
            "signal_strength": self.indicators.get("signal_strength"),
        }

    def _calculate_trix_proxy(self, values: dict[str, Any]) -> dict[str, Any]:
        """Calcule une approximation de TRIX basée sur TEMA et ROC."""
        tema_12 = values.get("tema_12")
        roc_10 = values.get("roc_10")
        momentum_10 = values.get("momentum_10")

        if tema_12 is None:
            return {"trix_value": None, "trix_direction": None, "trix_strength": 0}

        try:
            tema_val = float(tema_12)

            # Approximation TRIX avec ROC sur TEMA (priorité ROC)
            if roc_10 is not None:
                roc_val = float(roc_10)
                # ROC déjà en format décimal, utiliser directement
                trix_proxy = roc_val  # ROC en format décimal
            elif momentum_10 is not None:
                # Alternative avec momentum - convertir en format décimal
                # cohérent
                momentum_val = float(momentum_10)
                # Momentum normalisé pour être cohérent avec ROC (format
                # décimal)
                trix_proxy = momentum_val / 100.0  # Normalisation cohérente
            else:
                return {"trix_value": None, "trix_direction": None, "trix_strength": 0}

            # Direction TRIX avec ZONE NEUTRE élargie
            if abs(trix_proxy) < self.neutral_zone:
                # Zone neutre - pas de signal TRIX
                trix_direction = "neutral"
                trix_strength = 0.1
            elif trix_proxy >= self.extreme_trix_threshold:
                trix_direction = "extreme_bullish"  # Nouveau niveau
                trix_strength = 0.9  # Très fort
            elif trix_proxy >= self.strong_trix_threshold:
                trix_direction = "strong_bullish"
                trix_strength = 0.7  # Réduit de 0.8
            elif trix_proxy >= self.trix_bullish_threshold:
                trix_direction = "bullish"
                trix_strength = 0.4  # Réduit de 0.5
            elif trix_proxy <= -self.extreme_trix_threshold:
                trix_direction = "extreme_bearish"  # Nouveau niveau
                trix_strength = 0.9  # Très fort
            elif trix_proxy <= -self.strong_trix_threshold:
                trix_direction = "strong_bearish"
                trix_strength = 0.7  # Réduit de 0.8
            elif trix_proxy <= self.trix_bearish_threshold:
                trix_direction = "bearish"
                trix_strength = 0.4  # Réduit de 0.5
            else:
                trix_direction = "neutral"
                trix_strength = 0.1

            result = {
                "trix_value": trix_proxy,
                "trix_direction": trix_direction,
                "trix_strength": trix_strength,
                "tema_value": tema_val,
            }

        except (ValueError, TypeError):
            return {"trix_value": None, "trix_direction": None, "trix_strength": 0}
        else:
            return result

    def _detect_signal_line_crossover(self, values: dict[str, Any]) -> dict[str, Any]:
        """Détecte FRESH crossover entre TEMA (TRIX base) et DEMA (signal line)."""
        tema_12 = values.get("tema_12")
        dema_12 = values.get("dema_12")
        tema_12_prev = values.get("tema_12_prev")
        dema_12_prev = values.get("dema_12_prev")

        # Résultat par défaut
        default_result = {"is_crossover": False, "direction": None, "strength": 0}

        if None in (tema_12, dema_12):
            return default_result

        try:
            t = float(tema_12) if tema_12 is not None else 0.0
            d = float(dema_12) if dema_12 is not None else 0.0
            diff = (t - d) / (t or 1e-12)

            # Si on a les valeurs précédentes, détecter le fresh crossover
            if tema_12_prev is not None and dema_12_prev is not None:
                try:
                    tp = float(tema_12_prev)
                    dp = float(dema_12_prev)
                    diff_prev = (tp - dp) / (tp or 1e-12)

                    # Détection FRESH crossover (changement de signe récent)
                    crossed_up = (diff_prev <= 0) and (
                        diff > self.min_tema_dema_separation
                    )
                    crossed_down = (diff_prev >= 0) and (
                        diff < -self.min_tema_dema_separation
                    )

                    if crossed_up or crossed_down:
                        direction = "bullish" if crossed_up else "bearish"
                        return {
                            "is_crossover": True,
                            "direction": direction,
                            "strength": min(abs(diff) * 100, 1.0),
                            "tema_dema_diff": diff,
                        }
                    return {  # noqa: TRY300
                        "is_crossover": False,
                        "direction": "neutral",
                        "strength": 0.1,
                        "tema_dema_diff": diff,
                    }
                except (ValueError, TypeError):
                    pass  # Continue to fallback

            # Fallback: détection simple avec seuil renforcé si pas de valeurs précédentes
            if abs(diff) > self.min_tema_dema_separation * 2:
                direction = "bullish" if diff > 0 else "bearish"
                return {
                    "is_crossover": True,
                    "direction": direction,
                    "strength": min(abs(diff) * 100, 1.0),
                    "tema_dema_diff": diff,
                }
            return {  # noqa: TRY300
                "is_crossover": False,
                "direction": "neutral",
                "strength": 0.1,
                "tema_dema_diff": diff,
            }

        except (ValueError, TypeError):
            return default_result

    def _detect_momentum_alignment(
        self, values: dict[str, Any], trix_direction: str
    ) -> dict[str, Any]:
        """Détecte l'alignement du momentum avec TRIX."""
        momentum_score = values.get("momentum_score")
        directional_bias = values.get("directional_bias")
        trend_strength = values.get("trend_strength")

        alignment_score = 0.0
        alignment_indicators = []

        # Momentum score alignment
        if momentum_score is not None:
            try:
                momentum_val = float(momentum_score)
                # momentum_score RÉEL observé: 47.70-61.66, 99% entre 49.54-50.82
                # Seulement 0.14% > 52, donc seuils très fins nécessaires
                if trix_direction in ["bullish", "strong_bullish", "extreme_bullish"]:
                    # Pas de rejet strict - momentum quasi-normal autour de 50
                    # Top 10% (p90=50.25, donc 50.5 est rare)
                    if momentum_val > 50.5:
                        alignment_score += 0.20
                        alignment_indicators.append(
                            f"Momentum haussier ({momentum_val:.1f})"
                        )
                    elif momentum_val > 50.1:  # Au-dessus moyenne (50.04)
                        alignment_score += 0.12
                        alignment_indicators.append(
                            f"Momentum favorable ({momentum_val:.1f})"
                        )
                    elif momentum_val < 49.8:  # Sous moyenne mais pas critique
                        alignment_score += 0.05  # Petit bonus quand même
                        alignment_indicators.append(
                            f"Momentum neutre-bas ({momentum_val:.1f})"
                        )
                elif trix_direction in ["bearish", "strong_bearish", "extreme_bearish"]:
                    # Pas de rejet strict - momentum quasi-normal autour de 50
                    if (
                        momentum_val < 49.5
                    ):  # Bottom 10% (p10=49.84, donc 49.5 est rare)
                        alignment_score += 0.20
                        alignment_indicators.append(
                            f"Momentum baissier ({momentum_val:.1f})"
                        )
                    elif momentum_val < 49.9:  # En-dessous moyenne (50.04)
                        alignment_score += 0.12
                        alignment_indicators.append(
                            f"Momentum défavorable ({momentum_val:.1f})"
                        )
                    elif momentum_val > 50.2:  # Au-dessus moyenne mais pas critique
                        alignment_score += 0.05  # Petit bonus quand même
                        alignment_indicators.append(
                            f"Momentum neutre-haut ({momentum_val:.1f})"
                        )
            except (ValueError, TypeError):
                pass

        # Directional bias alignment - RÉDUIT
        if directional_bias:
            if (
                trix_direction in ["bullish", "strong_bullish", "extreme_bullish"]
                and directional_bias == "BULLISH"
            ):
                alignment_score += 0.18  # Réduit de 0.25
                alignment_indicators.append("Bias directionnel haussier")
            elif (
                trix_direction in ["bearish", "strong_bearish", "extreme_bearish"]
                and directional_bias == "BEARISH"
            ):
                alignment_score += 0.18  # Réduit de 0.25
                alignment_indicators.append("Bias directionnel baissier")

        # Trend strength confirmation - score neutre si WEAK
        if trend_strength is not None:
            # trend_strength: WEAK/MODERATE/STRONG/VERY_STRONG
            if trend_strength == "VERY_STRONG":
                alignment_score += 0.18
                alignment_indicators.append(f"Trend très forte ({trend_strength})")
            elif trend_strength == "STRONG":
                alignment_score += 0.14
                alignment_indicators.append(f"Trend forte ({trend_strength})")
            elif trend_strength == "MODERATE":
                alignment_score += 0.10
                alignment_indicators.append(f"Trend modérée ({trend_strength})")
            # WEAK ne donne plus de bonus (score neutre)

        # ROC confirmation
        roc_10 = values.get("roc_10")
        if roc_10 is not None:
            try:
                roc_val = float(roc_10)
                if (
                    trix_direction in ["bullish", "strong_bullish", "extreme_bullish"]
                    and roc_val > self.min_roc_confirmation
                ):
                    alignment_score += 0.12  # Réduit de 0.15
                    alignment_indicators.append(
                        f"ROC haussier ({roc_val:.2f}%)"
                    )  # ROC déjà en %
                elif (
                    trix_direction in ["bearish", "strong_bearish", "extreme_bearish"]
                    and roc_val < -self.min_roc_confirmation
                ):
                    alignment_score += 0.12  # Réduit de 0.15
                    alignment_indicators.append(
                        f"ROC baissier ({roc_val:.2f}%)"
                    )  # ROC déjà en %
            except (ValueError, TypeError):
                pass

        return {
            "is_aligned": alignment_score >= 0.25,  # Seuil relevé de 0.15 à 0.25
            "score": alignment_score,
            "indicators": alignment_indicators,
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

    def _validate_signal_requirements(
        self, signal_side: str, values: dict[str, Any], trix_direction: str
    ) -> dict[str, Any] | None:
        """Valide les exigences additionnelles pour le signal. Retourne un signal de rejet ou None si valide."""
        # ROC bounds validation
        roc_10 = values.get("roc_10")
        if roc_10 is not None:
            try:
                r = float(roc_10)
                if signal_side == "BUY" and not (-0.001 <= r <= 0.05):
                    return self._create_rejection_signal(
                        f"ROC {'trop faible' if r < -0.001 else 'trop fort'} ({r:.3%})",
                        {"roc_10": r},
                    )
                if signal_side == "SELL" and not (-0.05 <= r <= 0.001):
                    return self._create_rejection_signal(
                        f"ROC {'trop faible' if r > 0.001 else 'trop fort'} ({r:.3%})",
                        {"roc_10": r},
                    )
            except (ValueError, TypeError):
                pass

        # Pullback validation for BUY
        if signal_side == "BUY":
            ema_12 = values.get("ema_12")
            atr_14 = values.get("atr_14")
            close = (
                self.data.get("close", [None])[-1] if self.data.get("close") else None
            )

            if None not in (close, ema_12, atr_14):
                try:
                    close_val = float(close) if close is not None else 0.0
                    ema12_val = float(ema_12) if ema_12 is not None else 0.0
                    atr_val = float(atr_14) if atr_14 is not None else 0.0

                    overextended = (close_val > ema12_val * 1.01) or (
                        close_val - ema12_val > atr_val * 1.5
                    )

                    if overextended:
                        return self._create_rejection_signal(
                            "TRIX BUY mais prix sur-étiré: attente pullback",
                            {
                                "close": close_val,
                                "ema_12": ema12_val,
                                "atr_14": atr_val,
                            },
                        )
                except (ValueError, TypeError):
                    pass

        # Volume validation
        volume_ratio = values.get("volume_ratio")
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio < 0.5:
                    return self._create_rejection_signal(
                        f"Volume anémique ({vol_ratio:.2f}x)",
                        {"volume_ratio": vol_ratio},
                    )
            except (ValueError, TypeError):
                pass

        # Confluence validation
        confluence_score = values.get("confluence_score")
        if confluence_score is not None:
            try:
                conf_val = float(confluence_score)
                if conf_val < 25:
                    return self._create_rejection_signal(
                        f"Rejet TRIX: confluence trop faible ({conf_val:.1f})",
                        {
                            "confluence_score": conf_val,
                            "trix_direction": trix_direction,
                        },
                    )
            except (ValueError, TypeError):
                pass

        return None

    def generate_signal(self) -> dict[str, Any]:
        """
        Génère un signal basé sur TRIX crossover simulé.
        """
        if not self.validate_data():
            return self._create_rejection_signal("Données insuffisantes", {})

        values = self._get_current_values()

        # Calculer TRIX proxy
        trix_data = self._calculate_trix_proxy(values)
        if trix_data["trix_value"] is None or trix_data["trix_direction"] == "neutral":
            return self._create_rejection_signal(
                f"TRIX {'indisponible' if trix_data['trix_value'] is None else 'neutre'} ({trix_data.get('trix_value', 'N/A')})",
                {"trix_value": trix_data.get("trix_value")},
            )

        trix_direction = trix_data["trix_direction"]

        # Détection signal line crossover (optionnel)
        crossover_data = self._detect_signal_line_crossover(values)

        # Vérifier alignment du momentum
        alignment_data = self._detect_momentum_alignment(values, trix_direction)
        directional_bias = values.get("directional_bias")

        # Valider momentum alignment et directional bias
        momentum_not_aligned = (
            self.momentum_alignment_required and not alignment_data["is_aligned"]
        )
        bias_contradicts = directional_bias and (
            (
                trix_direction in ["bullish", "strong_bullish", "extreme_bullish"]
                and directional_bias == "BEARISH"
            )
            or (
                trix_direction in ["bearish", "strong_bearish", "extreme_bearish"]
                and directional_bias == "BULLISH"
            )
        )

        if momentum_not_aligned or bias_contradicts:
            reason = (
                f"TRIX {trix_direction} mais momentum pas aligné"
                if momentum_not_aligned
                else f"Rejet TRIX: bias contradictoire ({directional_bias})"
            )
            metadata = (
                {
                    "trix_direction": trix_direction,
                    "alignment_score": alignment_data["score"],
                }
                if momentum_not_aligned
                else {
                    "trix_direction": trix_direction,
                    "directional_bias": directional_bias,
                }
            )
            return self._create_rejection_signal(reason, metadata)

        # Déterminer le signal side et valider les exigences
        if trix_direction in ["bullish", "strong_bullish", "extreme_bullish"]:
            signal_side = "BUY"
        elif trix_direction in ["bearish", "strong_bearish", "extreme_bearish"]:
            signal_side = "SELL"
        else:
            signal_side = None

        if not signal_side:
            return self._create_rejection_signal(
                f"Direction TRIX indéterminée: {trix_direction}",
                {"trix_direction": trix_direction},
            )

        rejection = self._validate_signal_requirements(
            signal_side, values, trix_direction
        )
        if rejection:
            return rejection

        # Calculer confidence
        base_confidence = 0.65
        confidence_boost = 0.0

        confidence_boost += trix_data["trix_strength"] * 0.25
        confidence_boost += alignment_data["score"] * 0.20

        reason = f"TRIX {trix_direction} ({trix_data['trix_value']:.4f})"

        if alignment_data["indicators"]:
            reason += f" + {alignment_data['indicators'][0]}"

        if (
            self.signal_line_crossover
            and crossover_data["is_crossover"]
            and (
                (signal_side == "BUY" and crossover_data["direction"] == "bullish")
                or (signal_side == "SELL" and crossover_data["direction"] == "bearish")
            )
        ):
            confidence_boost += crossover_data["strength"] * 0.15
            reason += " + crossover signal line"

        volume_ratio = values.get("volume_ratio")
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.strong_volume_threshold:
                    confidence_boost += 0.12
                    reason += f" + volume fort ({vol_ratio:.1f}x)"
                elif vol_ratio >= self.volume_confirmation_threshold:
                    confidence_boost += 0.07
                    reason += f" + volume ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass

        macd_histogram = values.get("macd_histogram")
        if macd_histogram is not None:
            try:
                hist_val = float(macd_histogram)
                if (signal_side == "BUY" and hist_val > 0.001) or (
                    signal_side == "SELL" and hist_val < -0.001
                ):
                    confidence_boost += 0.06
                    reason += " + MACD histogram"
            except (ValueError, TypeError):
                pass

        confluence_score = values.get("confluence_score")
        if confluence_score is not None:
            try:
                conf_val = float(confluence_score)
                if conf_val > 75:
                    confidence_boost += 0.08
                    reason += " + haute confluence"
            except (ValueError, TypeError):
                pass

        confidence = self.calculate_confidence(base_confidence, 1.0 + confidence_boost)
        strength = self.get_strength_from_confidence(confidence)

        return {
            "side": signal_side,
            "confidence": confidence,
            "strength": strength,
            "reason": reason,
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "trix_value": trix_data["trix_value"],
                "trix_direction": trix_direction,
                "trix_strength": trix_data["trix_strength"],
                "tema_value": trix_data["tema_value"],
                "alignment_score": alignment_data["score"],
                "alignment_indicators": alignment_data["indicators"],
                "crossover_direction": crossover_data.get("direction"),
                "crossover_strength": crossover_data.get("strength"),
                "volume_ratio": volume_ratio,
                "confluence_score": confluence_score,
            },
        }

    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        required_indicators = ["tema_12", "roc_10", "momentum_score"]

        if not self.indicators:
            logger.warning(f"{self.name}: Aucun indicateur disponible")
            return False

        for indicator in required_indicators:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False

        return True
