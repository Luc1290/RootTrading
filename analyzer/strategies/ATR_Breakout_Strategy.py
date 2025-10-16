"""
ATR_Breakout_Strategy - Stratégie basée sur les breakouts avec volatilité ATR.
"""

import logging
import math
from typing import Any

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class ATR_Breakout_Strategy(BaseStrategy):
    """
    Stratégie PURE BREAKOUT utilisant ATR pour détecter les cassures de niveaux.

    Signaux générés:
    - BUY: Prix casse résistance avec ATR élevé (>50 percentile) + momentum haussier
    - SELL: Prix casse support avec ATR élevé (>50 percentile) + momentum baissier

    Pas de reversion/rejet - uniquement continuation de breakout.
    """

    def __init__(self, symbol: str,
                 data: dict[str, Any], indicators: dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres ATR et volatilité DURCIS pour vrais breakouts
        # Multiplicateur ATR pour zone de breakout (réduit)
        self.atr_multiplier = 1.0
        self.volatility_threshold = (
            50.0  # Seuil volatilité 50% minimum (percentile 0-100)
        )
        # 3% de proximité maximum (plus réaliste)
        self.resistance_proximity = 0.03
        self.support_proximity = 0.03  # 3% de proximité maximum
        self.breakout_zone_ratio = 0.01  # 1% zone breakout (plus large)
        # Paramètres de tendance - ACTIVÉS pour filtrer
        self.trend_filter_enabled = True  # Filtre activé pour cohérence
        self.min_trend_strength = 0.20  # Force minimum de tendance pour breakout
        # ADX minimum pour breakout valide (assoupli)
        self.min_adx_breakout = 18

    def _get_current_values(self) -> dict[str, float | None]:
        """Récupère les valeurs actuelles des indicateurs ATR."""
        return {
            "atr_14": self.indicators.get("atr_14"),
            "atr_percentile": self.indicators.get("atr_percentile"),
            "natr": self.indicators.get("natr"),
            "volatility_regime": self.indicators.get("volatility_regime"),
            "atr_stop_long": self.indicators.get("atr_stop_long"),
            "atr_stop_short": self.indicators.get("atr_stop_short"),
            "nearest_support": self.indicators.get("nearest_support"),
            "nearest_resistance": self.indicators.get("nearest_resistance"),
            "support_strength": self.indicators.get("support_strength"),
            "resistance_strength": self.indicators.get("resistance_strength"),
            "break_probability": self.indicators.get("break_probability"),
            "bb_upper": self.indicators.get("bb_upper"),
            "bb_lower": self.indicators.get("bb_lower"),
            "bb_width": self.indicators.get("bb_width"),
            "bb_squeeze": self.indicators.get("bb_squeeze"),
            "bb_expansion": self.indicators.get("bb_expansion"),
            "momentum_score": self.indicators.get("momentum_score"),
            "signal_strength": self.indicators.get("signal_strength"),
            "confluence_score": self.indicators.get("confluence_score"),
            # Indicateurs de tendance (actifs seulement)
            "market_regime": self.indicators.get("market_regime"),
            "trend_alignment": self.indicators.get("trend_alignment"),
            # Utilisé pour validation tendance
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

    def _get_previous_price(self) -> float | None:
        """Récupère le prix précédent pour détecter les crossings."""
        try:
            if self.data and "close" in self.data and len(
                    self.data["close"]) >= 2:
                return float(self.data["close"][-2])
        except (IndexError, ValueError, TypeError):
            pass
        return None

    def generate_signal(self) -> dict[str, Any]:
        """
        Génère un signal basé sur ATR et les breakouts de volatilité.
        """
        if not self.validate_data():
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données insuffisantes",
                "metadata": {"strategy": self.name},
            }

        values = self._get_current_values()
        current_price = self._get_current_price()
        prev_price = self._get_previous_price()

        # Helper pour valider les nombres (anti-NaN)
        def _is_valid(x):
            try:
                x = float(x) if x is not None else None
                return x is not None and not math.isnan(x)
            except (TypeError, ValueError):
                return False

        # Vérification des indicateurs essentiels avec protection NaN
        try:
            atr_val = values.get("atr_14")
            atr_perc_val = values.get("atr_percentile")

            atr = float(atr_val) if atr_val is not None and _is_valid(
                atr_val) else None
            atr_percentile = (float(atr_perc_val) if atr_perc_val is not None and _is_valid(
                atr_perc_val) else None)
            volatility_regime = values.get("volatility_regime")
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion ATR: {e}",
                "metadata": {"strategy": self.name},
            }

        if not (_is_valid(atr) and _is_valid(current_price)):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Valeurs ATR ou prix invalides/NaN",
                "metadata": {"strategy": self.name},
            }

        # Normaliser atr_percentile si nécessaire (gestion échelle 0-1 vs
        # 0-100)
        if atr_percentile is not None:
            ap = float(atr_percentile)
            if ap <= 1.0:  # reçu en 0..1, convertir en 0..100
                atr_percentile = ap * 100.0
            else:
                atr_percentile = ap

        # Vérification de la volatilité - SEUIL DURCI pour vrais breakouts
        if atr_percentile is not None and atr_percentile < self.volatility_threshold:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Volatilité insuffisante ({atr_percentile:.2f}) < {self.volatility_threshold} - ATR trop faible pour breakout",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "atr": atr,
                    "atr_percentile": atr_percentile,
                    "current_price": current_price,
                },
            }

        # Récupération des niveaux de support/résistance
        nearest_resistance = values.get("nearest_resistance")
        nearest_support = values.get("nearest_support")
        resistance_strength = values.get("resistance_strength")
        support_strength = values.get("support_strength")

        signal_side = None
        reason = ""
        base_confidence = 0.65  # Standardisé à 0.65 pour équité avec autres stratégies
        confidence_boost = 0.0
        proximity_type = None

        # Filtre de tendance OBLIGATOIRE pour breakouts cohérents
        market_regime = values.get("market_regime")
        trend_alignment = values.get("trend_alignment")
        adx_value = values.get("adx_14")

        # Déterminer la tendance principale - OBLIGATOIRE pour breakouts
        is_uptrend = False
        is_downtrend = False
        trend_confirmed = False

        # Validation ADX pour confirmer force de tendance
        if adx_value is not None and _is_valid(adx_value):
            adx_val = float(adx_value)
            if adx_val < self.min_adx_breakout:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"ADX trop faible ({adx_val:.1f}) - pas de force pour breakout",
                    "metadata": {
                        "strategy": self.name},
                }

        if market_regime:
            if market_regime == "TRENDING_BULL":
                is_uptrend = True
                trend_confirmed = True
            elif market_regime == "TRENDING_BEAR":
                is_downtrend = True
                trend_confirmed = True
            elif market_regime == "RANGING":
                # En ranging strict, pas de breakout
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Marché en range ({market_regime}) - pas de breakout",
                    "metadata": {
                        "strategy": self.name},
                }
            elif market_regime == "TRANSITION":
                # Autoriser TRANSITION si ADX≥18 ou BB expansion
                transition_ok = False
                if adx_value is not None and _is_valid(
                        adx_value) and float(adx_value) >= 18:
                    transition_ok = True
                if values.get("bb_expansion") is True:
                    transition_ok = True
                if not transition_ok:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": "Transition sans signes de poussée (ADX/BB) - on attend",
                        "metadata": {
                            "strategy": self.name},
                    }

        # Vérification trend_alignment pour renforcer la détection
        if trend_alignment is not None and _is_valid(trend_alignment):
            try:
                align_val = float(trend_alignment)
                if align_val > 0.2:  # Tendance haussière confirmée
                    if not is_uptrend:  # Si pas déjà détecté par regime
                        is_uptrend = True
                        trend_confirmed = True
                elif align_val < -0.2:  # Tendance baissière confirmée
                    if not is_downtrend:
                        is_downtrend = True
                        trend_confirmed = True
            except (ValueError, TypeError):
                pass

        # Pas de tendance détectée = pas de breakout
        if not trend_confirmed:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Aucune tendance claire détectée - breakout impossible",
                "metadata": {
                    "strategy": self.name},
            }

        # PURE BREAKOUT - Zone élargie (ATR + ratio max)
        breakout_zone_atr = atr * self.atr_multiplier if atr is not None else 0.0  # Zone ATR
        breakout_zone_max = current_price * \
            self.breakout_zone_ratio if current_price is not None else 0.0  # 1% max du prix
        breakout_zone = max(
            breakout_zone_atr, breakout_zone_max
        )  # Prendre le plus large

        if nearest_resistance is not None:
            try:
                resistance_level = float(nearest_resistance)
                # Distance mesurée vs le niveau (plus logique)
                distance_to_resistance = (
                    (current_price - resistance_level) / resistance_level
                    if current_price is not None and resistance_level != 0
                    else 0.0
                )

                # Détection par crossing + proximité
                bull_cross = (
                    prev_price is not None
                    and current_price is not None
                    and prev_price <= resistance_level < current_price
                )

                # BREAKOUT HAUSSIER : Prix AU-DESSUS de la résistance + zone
                # ATR
                if is_uptrend and (
                    bull_cross
                    or (
                        distance_to_resistance > 0
                        and distance_to_resistance <= self.resistance_proximity
                    )
                ):
                    # Vérifier si on est dans la zone de breakout élargie
                    if current_price is not None and current_price > resistance_level and current_price <= (
                        resistance_level * (1 + breakout_zone / resistance_level) if resistance_level != 0 else 0.0
                    ):
                        signal_side = "BUY"
                        proximity_type = "resistance_breakout"
                        cross_msg = " [CROSS]" if bull_cross else ""
                        reason = f"BREAKOUT résistance {resistance_level:.2f} cassée{cross_msg} (Prix: {current_price:.2f}, ATR: {atr:.4f})"
                        confidence_boost += 0.20  # Confiance élevée pour vrai breakout

                    # Bonus si résistance forte
                    if resistance_strength is not None and signal_side:
                        res_str_val = resistance_strength
                        res_str = str(res_str_val).upper(
                        ) if res_str_val is not None else ""
                        if res_str == "MAJOR":
                            confidence_boost += 0.08
                            reason += " - résistance majeure"
                        elif res_str == "STRONG":
                            confidence_boost += 0.05
                            reason += " - résistance forte"
            except (ValueError, TypeError):
                pass

        if signal_side is None and nearest_support is not None:
            try:
                support_level = float(nearest_support)
                # Distance mesurée vs le niveau (plus logique)
                distance_to_support = (
                    (support_level - current_price) / support_level
                    if current_price is not None and support_level != 0
                    else 0.0
                )

                # Détection par crossing + proximité
                bear_cross = (
                    prev_price is not None
                    and current_price is not None
                    and prev_price >= support_level > current_price
                )

                # BREAKOUT BAISSIER : Prix EN-DESSOUS du support + zone ATR
                if is_downtrend and (
                    bear_cross
                    or (
                        distance_to_support > 0
                        and distance_to_support <= self.support_proximity
                    )
                ):
                    # Vérifier si on est dans la zone de breakout élargie
                    if current_price is not None and current_price < support_level and current_price >= (
                        support_level * (1 - breakout_zone / support_level) if support_level != 0 else 0.0
                    ):
                        signal_side = "SELL"
                        proximity_type = "support_breakout"
                        cross_msg = " [CROSS]" if bear_cross else ""
                        reason = f"BREAKOUT support {support_level:.2f} cassé{cross_msg} (Prix: {current_price:.2f}, ATR: {atr:.4f})"
                        confidence_boost += 0.20  # Confiance élevée pour vrai breakout

                    # Bonus si support fort
                    if support_strength is not None and signal_side:
                        sup_str_val = support_strength
                        sup_str = str(sup_str_val).upper(
                        ) if sup_str_val is not None else ""
                        if sup_str == "MAJOR":
                            confidence_boost += 0.08
                            reason += " - support majeur"
                        elif sup_str == "STRONG":
                            confidence_boost += 0.05
                            reason += " - support fort"
            except (ValueError, TypeError):
                pass

        # Fallback quand S/R manquent : breakout Bollinger + ATR
        if signal_side is None and (
            nearest_resistance is None or nearest_support is None
        ):
            bb_upper = values.get("bb_upper")
            bb_lower = values.get("bb_lower")
            if _is_valid(
                    atr_percentile) and atr_percentile is not None and atr_percentile >= 60:
                bb_upper_val = bb_upper
                bb_lower_val = bb_lower
                if (
                    bb_upper_val is not None
                    and current_price is not None
                    and current_price > float(bb_upper_val)
                    and is_uptrend
                ):
                    signal_side = "BUY"
                    reason = f"Breakout BB haute + ATR élevé ({atr_percentile:.0f}%)"
                    confidence_boost += 0.15
                elif (
                    bb_lower_val is not None
                    and current_price is not None
                    and current_price < float(bb_lower_val)
                    and is_downtrend
                ):
                    signal_side = "SELL"
                    reason = f"Breakdown BB basse + ATR élevé ({atr_percentile:.0f}%)"
                    confidence_boost += 0.15

        # Pas de breakout détecté
        if signal_side is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Pas de breakout actif - prix hors zones de cassure",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "current_price": current_price,
                    "resistance": nearest_resistance,
                    "support": nearest_support,
                    "atr": atr,
                },
            }

        # Ajustements de confiance selon l'ATR - SEUILS DURCIS
        if atr_percentile is not None:
            if atr_percentile >= 80:
                confidence_boost += 0.15
                reason += " - volatilité EXTRÊME parfaite pour breakout"
            elif atr_percentile >= 65:
                confidence_boost += 0.10
                reason += " - volatilité élevée favorable"
            elif atr_percentile >= 50:
                confidence_boost += 0.05
                reason += " - volatilité suffisante"
            # Pas de bonus si < 50% (déjà filtré avant)

        # Régime volatilité ATR avec seuils adaptatifs
        if volatility_regime is not None and atr_percentile is not None and _is_valid(
                atr_percentile):
            try:
                atr_percentile = float(atr_percentile)

                if signal_side == "BUY":
                    # BUY breakout ATR : volatilité expansion + proximité
                    # résistance = explosion haussière
                    if volatility_regime == "expanding":
                        if atr_percentile > 85:  # Expansion extrême + résistance
                            confidence_boost += 0.15
                            reason += f" + expansion extrême ({atr_percentile:.0f}%)"
                        elif atr_percentile > 70:  # Expansion forte
                            confidence_boost += 0.12
                            reason += f" + expansion forte ({atr_percentile:.0f}%)"
                        else:  # Expansion modérée
                            confidence_boost += 0.08
                            reason += f" + expansion modérée ({atr_percentile:.0f}%)"
                    elif volatility_regime == "high":
                        if (
                            atr_percentile > 90
                        ):  # Volatilité extrême = continuation explosive
                            confidence_boost += 0.12
                            reason += f" + volatilité extrême ({atr_percentile:.0f}%)"
                        elif atr_percentile > 75:  # Volatilité élevée favorable
                            confidence_boost += 0.10
                            reason += f" + volatilité élevée ({atr_percentile:.0f}%)"
                        else:  # Volatilité modérément élevée
                            confidence_boost += 0.06
                            reason += f" + volatilité favorable ({atr_percentile:.0f}%)"
                    elif volatility_regime == "normal":
                        if atr_percentile is not None and (
                            40 <= atr_percentile <= 60
                        ):  # Volatilité idéale pour breakout contrôlé
                            confidence_boost += 0.12
                            reason += f" + volatilité idéale contrôlée ({atr_percentile:.0f}%)"
                        else:
                            confidence_boost += 0.08
                            reason += f" + volatilité normale ({atr_percentile:.0f}%)"
                    elif volatility_regime == "low":
                        confidence_boost += (
                            0.05  # Faible volatilité = breakout moins puissant
                        )
                        reason += (
                            f" + volatilité faible limitée ({atr_percentile:.0f}%)"
                        )

                # SELL breakdown ATR : volatilité élevée + proximité support =
                # chute violente
                elif volatility_regime == "expanding":
                    if (
                        atr_percentile > 80
                    ):  # Expansion + support = cascade baissière
                        confidence_boost += 0.14
                        reason += f" + expansion cascade ({atr_percentile:.0f}%)"
                    else:  # Expansion modérée
                        confidence_boost += 0.10
                        reason += f" + expansion breakdown ({atr_percentile:.0f}%)"
                elif volatility_regime == "high":
                    if atr_percentile is not None and (
                        atr_percentile > 85
                    ):  # Volatilité extrême = panique/liquidation
                        confidence_boost += 0.12
                        reason += f" + volatilité extrême ({atr_percentile:.0f}%)"
                    else:  # Volatilité élevée
                        confidence_boost += 0.08
                        reason += f" + volatilité élevée ({atr_percentile:.0f}%)"
                elif volatility_regime == "normal":
                    confidence_boost += 0.10
                    reason += (
                        f" + volatilité normale breakdown ({atr_percentile:.0f}%)"
                    )
                elif volatility_regime == "low":
                    confidence_boost += (
                        0.08  # Volatilité faible = breakdown contrôlé
                    )
                    reason += (
                        f" + volatilité faible contrôlée ({atr_percentile:.0f}%)"
                    )

            except (ValueError, TypeError):
                pass

        # Bollinger Bands pour confirmation
        bb_upper = values.get("bb_upper")
        bb_lower = values.get("bb_lower")
        bb_squeeze = values.get("bb_squeeze")
        bb_expansion = values.get("bb_expansion")

        if signal_side == "BUY" and bb_upper is not None:
            try:
                bb_up = float(bb_upper)
                if current_price is not None and current_price >= bb_up * \
                        0.98:  # Proche de la bande haute
                    confidence_boost += 0.1
                    reason += " près BB haute"
            except (ValueError, TypeError):
                pass

        if signal_side == "SELL" and bb_lower is not None:
            try:
                bb_low = float(bb_lower)
                if current_price is not None and current_price <= bb_low * \
                        1.02:  # Proche de la bande basse
                    confidence_boost += 0.1
                    reason += " près BB basse"
            except (ValueError, TypeError):
                pass

        # Bollinger Squeeze = compression avant expansion
        if bb_squeeze is not None and bb_squeeze:
            confidence_boost += 0.08
            reason += " avec BB squeeze"

        # Bollinger Expansion = expansion après compression
        if bb_expansion is not None and bb_expansion:
            confidence_boost += 0.06
            reason += " avec BB expansion"

        # Break probability avec validation NaN
        break_probability = values.get("break_probability")
        if break_probability is not None and _is_valid(break_probability):
            try:
                break_prob_val = break_probability
                break_prob = float(
                    break_prob_val) if break_prob_val is not None else 0.0
                if break_prob > 0.6:
                    confidence_boost += 0.1
                    reason += f" (prob break: {break_prob:.2f})"
            except (ValueError, TypeError):
                pass

        # Momentum OBLIGATOIRE pour confirmer breakout - REJET si contraire
        momentum_score = values.get("momentum_score")
        if momentum_score is not None and _is_valid(momentum_score):
            try:
                momentum = float(momentum_score)
                # Rejet si momentum trop contraire au breakout (assoupli)
                if signal_side == "BUY" and momentum < 48:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet breakout haussier : momentum trop faible ({momentum:.0f})",
                        "metadata": {
                            "strategy": self.name,
                            "momentum_score": momentum},
                    }
                if signal_side == "SELL" and momentum > 52:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet breakout baissier : momentum trop fort ({momentum:.0f})",
                        "metadata": {
                            "strategy": self.name,
                            "momentum_score": momentum},
                    }

                # Bonus si momentum aligné
                if signal_side == "BUY" and momentum > 60:
                    confidence_boost += 0.15
                    reason += f" + momentum FORT haussier ({momentum:.0f})"
                elif signal_side == "SELL" and momentum < 40:
                    confidence_boost += 0.15
                    reason += f" + momentum FORT baissier ({momentum:.0f})"
            except (ValueError, TypeError):
                pass

        # Signal strength (VARCHAR: WEAK/MODERATE/STRONG)
        signal_strength_calc = values.get("signal_strength")
        if signal_strength_calc is not None:
            sig_str = str(signal_strength_calc).upper()
            if sig_str == "STRONG":
                confidence_boost += 0.06
                reason += " + signal fort"
            elif sig_str == "MODERATE":
                confidence_boost += 0.03
                reason += " + signal modéré"

        confluence_score = values.get("confluence_score")
        if confluence_score is not None and _is_valid(confluence_score):
            try:
                confluence = float(confluence_score)
                if confluence > 60:
                    confidence_boost += 0.08
                    reason += f" + confluence élevée ({confluence:.0f})"
                elif confluence > 45:
                    confidence_boost += 0.05
                    reason += f" + confluence modérée ({confluence:.0f})"
            except (ValueError, TypeError):
                pass

        # Clamp confiance dans [0,1] pour éviter explosion
        confidence = max(
            0.0, min(
                1.0, self.calculate_confidence(
                    base_confidence, 1 + confidence_boost)), )
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
                "atr": atr,
                "atr_percentile": atr_percentile,
                "volatility_regime": volatility_regime,
                "proximity_type": proximity_type,
                "resistance": nearest_resistance,
                "support": nearest_support,
                "resistance_strength": resistance_strength,
                "support_strength": support_strength,
                "break_probability": break_probability,
                "bb_squeeze": bb_squeeze,
                "momentum_score": momentum_score,
                "confluence_score": confluence_score,
                "market_regime": market_regime,
                "trend_alignment": trend_alignment,
            },
        }

    def validate_data(self) -> bool:
        """Valide que tous les indicateurs ATR requis sont présents."""
        if not super().validate_data():
            return False

        required = ["atr_14"]

        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(
                    f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False

        # Vérifier aussi qu'on a des données de prix
        if not self.data or "close" not in self.data or not self.data["close"]:
            logger.warning(f"{self.name}: Données de prix manquantes")
            return False

        return True
