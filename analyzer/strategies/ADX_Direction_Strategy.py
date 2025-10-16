"""
ADX_Direction_Strategy - Stratégie basée sur la force et direction de tendance ADX.
"""

import logging
import math
from typing import Any

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class ADX_Direction_Strategy(BaseStrategy):
    """
    Stratégie utilisant ADX, +DI et -DI pour identifier les tendances fortes.

    Signaux générés:
    - BUY: ADX > 25 avec +DI > -DI et momentum haussier
    - SELL: ADX > 25 avec -DI > +DI et momentum baissier
    """

    def __init__(self, symbol: str, data: dict[str, Any], indicators: dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Seuils ADX CRYPTO STRICTS - Anti-faux signaux
        # Tendance minimum CRYPTO (relevé de 15 à 22)
        self.adx_threshold = 22.0
        self.adx_strong = 30.0  # Tendance forte (relevé de 25 à 30)
        self.adx_extreme = 40.0  # Tendance très forte (relevé de 35 à 40)
        # Différence minimale DI STRICT (5.0 vs 2.0)
        self.di_diff_threshold = 5.0

        # NOUVEAUX FILTRES CRYPTO
        # Confluence pour bonus (pas obligatoire)
        self.min_confluence_bonus = 55
        # Momentum opposition forte (rejet seulement)
        self.min_momentum_alignment = 15
        self.required_confirmations = 2  # Confirmations minimum requises

        # Gestion des régimes de marché (pas de pénalité ranging - ADX détecte
        # les sorties)
        self.volatile_penalty = 0.10  # Pénalité marché volatil réduite

    def _get_current_values(self) -> dict[str, float | None]:
        """Récupère les valeurs actuelles des indicateurs ADX."""
        return {
            "adx_14": self.indicators.get("adx_14"),
            "plus_di": self.indicators.get("plus_di"),
            "minus_di": self.indicators.get("minus_di"),
            "dx": self.indicators.get("dx"),
            "adxr": self.indicators.get("adxr"),
            "trend_strength": self.indicators.get("trend_strength"),
            "directional_bias": self.indicators.get("directional_bias"),
            "trend_angle": self.indicators.get("trend_angle"),
            "momentum_score": self.indicators.get("momentum_score"),
            "signal_strength": self.indicators.get("signal_strength"),
            "confluence_score": self.indicators.get("confluence_score"),
            "market_regime": self.indicators.get("market_regime"),
        }

    def _create_rejection_signal(
        self, reason: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Helper pour créer un signal de rejet."""
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

    def _validate_adx_indicators(
        self, values: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, float | None]]:
        """
        Valide les indicateurs ADX. Retourne (None, validated_data) si succès,
        (rejection_signal, {}) si échec.
        """

        def _is_valid(x):
            try:
                x = float(x) if x is not None else None
                return x is not None and not math.isnan(x)
            except (TypeError, ValueError):
                return False

        # Vérification et conversion des indicateurs essentiels
        try:
            adx_val = values.get("adx_14")
            plus_di_val = values.get("plus_di")
            minus_di_val = values.get("minus_di")

            adx = float(adx_val) if _is_valid(adx_val) else None
            plus_di = float(plus_di_val) if _is_valid(plus_di_val) else None
            minus_di = float(minus_di_val) if _is_valid(minus_di_val) else None
        except (ValueError, TypeError) as e:
            return self._create_rejection_signal(f"Erreur conversion ADX: {e}"), {}

        # Validation combinée de None et validité
        if (
            not (_is_valid(adx) and _is_valid(plus_di) and _is_valid(minus_di))
            or adx is None
            or plus_di is None
            or minus_di is None
        ):
            return self._create_rejection_signal("Valeurs ADX/DI invalides ou NaN"), {}

        # Validation du seuil ADX
        if adx < self.adx_threshold:
            return (
                self._create_rejection_signal(
                    f"ADX insuffisant ({adx:.1f}) < {self.adx_threshold} - tendance trop faible",
                    {
                        "symbol": self.symbol,
                        "adx": adx,
                        "plus_di": plus_di,
                        "minus_di": minus_di,
                    },
                ),
                {},
            )

        # Calcul et validation de la différence entre DI
        di_diff = plus_di - minus_di
        di_diff_abs = abs(di_diff)

        # Vérifier écart DI suffisant ET non égal
        if di_diff_abs < self.di_diff_threshold or plus_di == minus_di:
            reason = (
                "DI égaux : direction neutre, pas de signal ADX"
                if plus_di == minus_di
                else f"DI écart insuffisant ({di_diff_abs:.1f}) < {self.di_diff_threshold} - direction incertaine"
            )
            return (
                self._create_rejection_signal(
                    reason,
                    {
                        "symbol": self.symbol,
                        "adx": adx,
                        "plus_di": plus_di,
                        "minus_di": minus_di,
                    },
                ),
                {},
            )

        return None, {
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "di_diff": di_diff,
            "di_diff_abs": di_diff_abs,
            "_is_valid": _is_valid,
        }

    def _validate_momentum_alignment(
        self,
        values: dict[str, Any],
        plus_di: float,
        minus_di: float,
        _is_valid: callable,
    ) -> dict[str, Any] | None:
        """
        Valide l'alignement momentum. Retourne None si OK, sinon signal de rejet.
        """
        momentum_score = values.get("momentum_score")
        if momentum_score is not None and _is_valid(momentum_score):
            try:
                momentum_val = float(momentum_score)
                momentum_center = 50

                adx_direction = "bullish" if plus_di > minus_di else "bearish"

                if adx_direction == "bullish" and momentum_val < (
                    momentum_center - self.min_momentum_alignment
                ):
                    return self._create_rejection_signal(
                        f"Momentum fortement opposé bullish: {momentum_val:.1f} contre ADX haussier",
                        {"rejected_reason": "momentum_strongly_opposed"},
                    )
                if adx_direction == "bearish" and momentum_val > (
                    momentum_center + self.min_momentum_alignment
                ):
                    return self._create_rejection_signal(
                        f"Momentum fortement opposé bearish: {momentum_val:.1f} contre ADX baissier",
                        {"rejected_reason": "momentum_strongly_opposed"},
                    )
            except (ValueError, TypeError):
                pass

        return None

    def generate_signal(self) -> dict[str, Any]:
        """
        Génère un signal basé sur ADX et les indicateurs directionnels.
        """
        if not self.validate_data():
            return self._create_rejection_signal("Données insuffisantes")

        values = self._get_current_values()

        # Valider les indicateurs ADX
        rejection_signal, validated_data = self._validate_adx_indicators(values)
        if rejection_signal:
            return rejection_signal

        # Extraire les données validées
        adx = validated_data["adx"]
        plus_di = validated_data["plus_di"]
        minus_di = validated_data["minus_di"]
        di_diff = validated_data["di_diff"]
        di_diff_abs = validated_data["di_diff_abs"]
        _is_valid = validated_data["_is_valid"]

        # Valider l'alignement momentum
        momentum_rejection = self._validate_momentum_alignment(
            values, plus_di, minus_di, _is_valid
        )
        if momentum_rejection:
            return momentum_rejection

        signal_side = None
        reason = ""
        base_confidence = 0.65
        confidence_boost = 0.0

        # Logique de signal basée sur la direction des DI
        if plus_di is not None and minus_di is not None and plus_di > minus_di:
            # Tendance haussière
            signal_side = "BUY"
            reason = f"ADX ({adx:.1f}) avec tendance haussière (+DI > -DI)"
        else:
            # Tendance baissière
            signal_side = "SELL"
            reason = f"ADX ({adx:.1f}) avec tendance baissière (-DI > +DI)"

        # SYSTÈME CONFIRMATIONS OBLIGATOIRES
        confirmations_count = 0
        confirmations_details = []

        # Ajustement confiance selon ADX (RÉDUIT pour éviter sur-confiance)
        if adx >= self.adx_extreme:
            confidence_boost += 0.15  # RÉDUIT de 0.18 à 0.15
            reason += " - tendance très forte"
            confirmations_count += 1
            confirmations_details.append("ADX_extreme")
        elif adx >= self.adx_strong:
            confidence_boost += 0.10  # RÉDUIT de 0.12 à 0.10
            reason += " - tendance forte"
            confirmations_count += 1
            confirmations_details.append("ADX_strong")
        else:
            confidence_boost += 0.04  # RÉDUIT de 0.06 à 0.04
            reason += " - tendance minimum"

        # Ajustement selon différence DI (PLUS STRICT)
        if di_diff_abs >= 15:  # Seuil relevé de 20 à 15
            confidence_boost += 0.12
            reason += f" (écart DI excellent: {di_diff_abs:.1f})"
            confirmations_count += 1
            confirmations_details.append("DI_gap_excellent")
        elif di_diff_abs >= 8:  # Seuil relevé de 10 à 8
            confidence_boost += 0.08
            reason += f" (écart DI bon: {di_diff_abs:.1f})"
            confirmations_count += 1
            confirmations_details.append("DI_gap_good")
        else:
            confidence_boost += 0.02  # RÉDUIT de 0.04 à 0.02
            reason += f" (écart DI minimum: {di_diff_abs:.1f})"

        # Utilisation des indicateurs complémentaires

        # Trend strength pré-calculé (COMPTAGE CONFIRMATIONS)
        trend_strength = values.get("trend_strength")
        if trend_strength:
            trend_str = str(trend_strength).lower()
            if trend_str in ["extreme", "very_strong"]:
                confidence_boost += 0.12
                confirmations_count += 1
                confirmations_details.append("trend_very_strong")
                reason += f" avec trend_strength {trend_str}"
            elif trend_str == "strong":
                confidence_boost += 0.08
                confirmations_count += 1
                confirmations_details.append("trend_strong")
                reason += f" avec trend_strength {trend_str}"

        # Directional bias confirmation avec normalisation
        directional_bias = values.get("directional_bias")
        if directional_bias:
            # Normaliser les variantes possibles
            bias_str = str(directional_bias).upper().strip()
            bias_bullish = bias_str in ["BULLISH", "BULL", "UP", "HAUSSIER", "POSITIVE"]
            bias_bearish = bias_str in [
                "BEARISH",
                "BEAR",
                "DOWN",
                "BAISSIER",
                "NEGATIVE",
            ]

            if (signal_side == "BUY" and bias_bullish) or (
                signal_side == "SELL" and bias_bearish
            ):
                confidence_boost += 0.10
                confirmations_count += 1
                confirmations_details.append("directional_bias_aligned")
                reason += " confirmé par bias directionnel"

        # Momentum score - bonus selon alignement (pas de variable morte)
        momentum_score = values.get("momentum_score")
        if momentum_score is not None and _is_valid(momentum_score):
            try:
                momentum_val = float(momentum_score)
                # Bonus progressif selon alignement momentum (seuils plus
                # tolérants)
                if momentum_val is not None and (
                    (signal_side == "BUY" and momentum_val > 60)
                    or (signal_side == "SELL" and momentum_val < 40)
                ):
                    confidence_boost += 0.12
                    confirmations_count += 1
                    confirmations_details.append("momentum_strong")
                    reason += f" avec momentum FORT ({momentum_val:.1f})"
                elif momentum_val is not None and (
                    (signal_side == "BUY" and momentum_val > 55)
                    or (signal_side == "SELL" and momentum_val < 45)
                ):
                    confidence_boost += 0.08
                    confirmations_count += 1
                    confirmations_details.append("momentum_aligned")
                    reason += f" avec momentum favorable ({momentum_val:.1f})"
                elif (signal_side == "BUY" and momentum_val > 50) or (
                    signal_side == "SELL" and momentum_val < 50
                ):
                    confidence_boost += 0.04
                    reason += f" avec momentum légèrement aligné ({momentum_val:.1f})"
            except (ValueError, TypeError):
                pass

        # ADXR pour confirmation de persistance (seuil plus souple)
        adxr = values.get("adxr")
        if adxr and _is_valid(adxr):
            try:
                adxr_val = float(adxr)
                adxr_threshold = max(self.adx_threshold - 2, 15)  # Seuil plus souple
                if adxr_val > adxr_threshold:
                    confidence_boost += 0.05
                    reason += " (ADXR confirme persistance)"
            except (ValueError, TypeError):
                pass

        # Signal strength (varchar: WEAK/MODERATE/STRONG)
        signal_strength_calc = values.get("signal_strength")
        if signal_strength_calc:
            signal_str = str(signal_strength_calc).upper()
            if signal_str == "STRONG":
                confidence_boost += 0.10
                reason += " + signal fort"
            elif signal_str == "MODERATE":
                confidence_boost += 0.05
                reason += " + signal modéré"

        # CONFLUENCE et REGIME d'abord - puis validation finale
        # (confluence peut ajouter des confirmations)

        # Confluence score maintenant optionnel - bonus seulement
        confluence_score = values.get("confluence_score", 0)
        if confluence_score and _is_valid(confluence_score):
            try:
                confluence = float(confluence_score)
                # Bonus confluence si présente et élevée
                if confluence > 80:
                    confidence_boost += 0.15
                    confirmations_count += 1
                    confirmations_details.append("confluence_excellent")
                    reason += f" + confluence excellente ({confluence:.0f})"
                elif confluence > 70:
                    confidence_boost += 0.10
                    confirmations_count += 1
                    confirmations_details.append("confluence_high")
                    reason += f" + confluence élevée ({confluence:.0f})"
                elif confluence > self.min_confluence_bonus:
                    confidence_boost += 0.05
                    reason += f" + confluence correcte ({confluence:.0f})"
                # Pas de pénalité si confluence faible - ADX prime
            except (ValueError, TypeError):
                pass

        # Gestion des régimes de marché (sans pénalité ranging - ADX détecte
        # les sorties)
        market_regime = values.get("market_regime")
        if market_regime:
            regime_str = str(market_regime).upper()
            if regime_str == "RANGING":
                # Pas de pénalité ranging - ADX fort peut indiquer sortie de range
                # Bonus léger si ADX monte en ranging (sortie potentielle)
                if adx > self.adx_strong:
                    confidence_boost += 0.05
                    reason += " (sortie de range potentielle)"
                else:
                    reason += " (marché ranging - surveiller)"
            elif regime_str == "VOLATILE":
                confidence_boost -= self.volatile_penalty
                reason += " (marché volatil)"
            elif regime_str in ["TRENDING_BULL", "TRENDING_BEAR"]:
                # Bonus si aligné avec la tendance
                if (signal_side == "BUY" and regime_str == "TRENDING_BULL") or (
                    signal_side == "SELL" and regime_str == "TRENDING_BEAR"
                ):
                    confidence_boost += 0.15
                    confirmations_count += 1
                    confirmations_details.append("regime_aligned")
                    reason += f" (aligné {regime_str.lower()})"
                else:
                    confidence_boost -= 0.05  # Pénalité réduite
                    reason += f" (contre-tendance {regime_str.lower()})"

        # VALIDATION FINALE CONFIRMATIONS - APRÈS confluence & regime
        if confirmations_count < self.required_confirmations:
            return self._create_rejection_signal(
                f"Confirmations insuffisantes ADX ({confirmations_count}/{self.required_confirmations}) - rejeté",
                {
                    "rejected_reason": "insufficient_confirmations",
                    "confirmations_count": confirmations_count,
                    "confirmations_details": confirmations_details,
                },
            )

        confidence = self.calculate_confidence(base_confidence, 1 + confidence_boost)
        strength = self.get_strength_from_confidence(confidence)

        return {
            "side": signal_side,
            "confidence": confidence,
            "strength": strength,
            "reason": reason,
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "adx": adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "di_difference": di_diff,
                "dx": values.get("dx"),
                "adxr": adxr,
                "trend_strength": trend_strength,
                "directional_bias": directional_bias,
                "momentum_score": momentum_score,
                "signal_strength_calc": signal_strength_calc,
                "confluence_score": confluence_score,
                "market_regime": values.get("market_regime"),
                "confirmations_count": confirmations_count,
                "confirmations_details": confirmations_details,
            },
        }

    def validate_data(self) -> bool:
        """Valide que tous les indicateurs ADX requis sont présents."""
        if not super().validate_data():
            return False

        required = ["adx_14", "plus_di", "minus_di"]

        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            indicator_val = self.indicators.get(indicator)
            if indicator_val is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False

        return True
