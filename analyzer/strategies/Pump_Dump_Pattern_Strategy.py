"""
Pump_Dump_Pattern_Strategy - Stratégie de détection des patterns pump & dump.
Détecte les mouvements anormaux de prix avec volume exceptionnel, suivis de corrections.
"""

import logging
from typing import Any

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class Pump_Dump_Pattern_Strategy(BaseStrategy):
    """
    Stratégie détectant les patterns pump & dump pour surfer sur le momentum.

    Pattern Pump:
    - Hausse de prix rapide et massive (>2%)
    - Volume spike exceptionnel (>2x normal)
    - RSI momentum fort mais pas extrême
    - Momentum très élevé

    Pattern Dump:
    - Chute de prix rapide après distribution
    - Volume élevé de vente
    - RSI baissier et momentum négatif

    Signaux générés:
    - BUY: Détection d'un pump débutant (surfer sur la vague)
    - SELL: Détection d'un dump débutant (sortir avant la chute)
    """

    def __init__(self, symbol: str,
                 data: dict[str, Any], indicators: dict[str, Any]):
        super().__init__(symbol, data, indicators)

        # Seuils pour détection pump - MOMENTUM TRADING OPTIMISÉ (décimaux
        # convertis en %)
        # 1.8% hausse minimum (sera * 100 = 1.8%)
        self.pump_price_threshold = 0.018
        # 3.0% hausse extrême (sera * 100 = 3.0%)
        self.extreme_pump_threshold = 0.030
        # Volume 1.8x normal (confirmation momentum)
        self.pump_volume_multiplier = 1.8
        # Volume 3.5x normal (pump majeur)
        self.extreme_volume_multiplier = 3.5
        self.pump_rsi_threshold = 60  # RSI momentum positif (pas trop tard)
        self.extreme_rsi_threshold = 75  # RSI fort mais pas surachat extrême

        # Seuils pour détection dump - MOMENTUM TRADING OPTIMISÉ (décimaux
        # convertis en %)
        # 1.5% chute minimum (sera * 100 = -1.5%)
        self.dump_price_threshold = -0.015
        # 2.5% chute extrême (sera * 100 = -2.5%)
        self.extreme_dump_threshold = -0.025
        self.dump_rsi_threshold = 45  # RSI faiblesse (pas trop bas)
        # Momentum négatif (signal sortie)
        self.momentum_reversal_threshold = -0.3

        # Paramètres de validation momentum
        self.min_volatility_regime = 0.7  # Volatilité suffisante pour momentum
        self.min_trade_intensity = 1.3  # Activité trading confirmée

    def _create_rejection_signal(self, reason: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
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

    def _get_current_values(self) -> dict[str, float | None]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            # Oscillateurs momentum
            "rsi_14": self.indicators.get("rsi_14"),
            "rsi_21": self.indicators.get("rsi_21"),
            "williams_r": self.indicators.get("williams_r"),
            "roc_10": self.indicators.get("roc_10"),
            "roc_20": self.indicators.get("roc_20"),
            "momentum_10": self.indicators.get("momentum_10"),
            # Volume analysis
            "volume_spike_multiplier": self.indicators.get("volume_spike_multiplier"),
            "relative_volume": self.indicators.get("relative_volume"),
            "volume_quality_score": self.indicators.get("volume_quality_score"),
            "trade_intensity": self.indicators.get("trade_intensity"),
            "volume_pattern": self.indicators.get("volume_pattern"),
            # Volatilité et contexte
            "atr_14": self.indicators.get("atr_14"),
            "volatility_regime": self.indicators.get("volatility_regime"),
            "momentum_score": self.indicators.get("momentum_score"),
            "trend_strength": self.indicators.get("trend_strength"),
            "directional_bias": self.indicators.get("directional_bias"),
            # Pattern et confluence
            "pattern_detected": self.indicators.get("pattern_detected"),
            "pattern_confidence": self.indicators.get("pattern_confidence"),
            "signal_strength": self.indicators.get("signal_strength"),
            "confluence_score": self.indicators.get("confluence_score"),
            "anomaly_detected": self.indicators.get("anomaly_detected"),
        }

    def _detect_pump_pattern(self, values: dict[str, Any]) -> dict[str, Any]:
        """Détecte un pattern de pump (hausse anormale)."""
        pump_score = 0.0
        pump_indicators = []

        # Analyse ROC (Rate of Change) - DB format: pourcentages directs (ex:
        # 17.26 = 17.26%)
        roc_10 = values.get("roc_10")
        values.get("roc_20")

        if roc_10 is not None:
            try:
                roc_val = float(roc_10)
                # ROC en POURCENTAGE direct dans DB (ex: 17.26 pour 17.26%)
                # Comparaison directe avec seuils en pourcentage
                extreme_pump_pct = self.extreme_pump_threshold * 100  # 3.0%
                pump_pct = self.pump_price_threshold * 100  # 1.8%

                if roc_val >= extreme_pump_pct:
                    pump_score += 0.4
                    pump_indicators.append(f"ROC10 extrême ({roc_val:.1f}%)")
                elif roc_val >= pump_pct:
                    pump_score += 0.2
                    pump_indicators.append(f"ROC10 élevé ({roc_val:.1f}%)")
            except (ValueError, TypeError):
                pass

        # RSI surachat extrême
        rsi_14 = values.get("rsi_14")
        if rsi_14 is not None:
            try:
                rsi_val = float(rsi_14)
                if rsi_val >= self.extreme_rsi_threshold:
                    pump_score += 0.3
                    pump_indicators.append(f"RSI extrême ({rsi_val:.1f})")
                elif rsi_val >= self.pump_rsi_threshold:
                    pump_score += 0.15
                    pump_indicators.append(f"RSI surachat ({rsi_val:.1f})")
            except (ValueError, TypeError):
                pass

        # Volume spike critique
        volume_spike = values.get("volume_spike_multiplier")
        if volume_spike is not None:
            try:
                spike_val = float(volume_spike)
                if spike_val >= self.extreme_volume_multiplier:
                    pump_score += 0.35
                    pump_indicators.append(
                        f"Volume spike extrême ({spike_val:.1f}x)")
                elif spike_val >= self.pump_volume_multiplier:
                    pump_score += 0.2
                    pump_indicators.append(f"Volume spike ({spike_val:.1f}x)")
            except (ValueError, TypeError):
                pass

        # Williams %R confirmant surachat
        williams_r = values.get("williams_r")
        if williams_r is not None:
            try:
                wr_val = float(williams_r)
                if wr_val >= -10:  # Williams %R > -10 = surachat extrême
                    pump_score += 0.15
                    pump_indicators.append(
                        f"Williams%R surachat ({wr_val:.1f})")
            except (ValueError, TypeError):
                pass

        return {
            "is_pump": pump_score >= 0.50,  # Seuil pour détection précoce momentum
            "pump_score": pump_score,
            "indicators": pump_indicators,
        }

    def _detect_dump_pattern(self, values: dict[str, Any]) -> dict[str, Any]:
        """Détecte un pattern de dump (chute rapide avec volume de vente)."""
        dump_score = 0.0
        dump_indicators = []

        # ROC négatif indiquant chute
        roc_10 = values.get("roc_10")
        if roc_10 is not None:
            try:
                roc_val = float(roc_10)
                # ROC en POURCENTAGE direct dans DB (ex: -13.32 pour -13.32%)
                # Comparaison directe avec seuils en pourcentage
                extreme_dump_pct = self.extreme_dump_threshold * 100  # -2.5%
                dump_pct = self.dump_price_threshold * 100  # -1.5%

                if roc_val <= extreme_dump_pct:
                    dump_score += 0.3
                    dump_indicators.append(
                        f"ROC10 chute extrême ({roc_val:.1f}%)")
                elif roc_val <= dump_pct:
                    dump_score += 0.15
                    dump_indicators.append(f"ROC10 chute ({roc_val:.1f}%)")
            except (ValueError, TypeError):
                pass

        # RSI baissier (momentum de vente)
        rsi_14 = values.get("rsi_14")
        if rsi_14 is not None:
            try:
                rsi_val = float(rsi_14)
                # RSI qui décline rapidement indique faiblesse
                if rsi_val <= self.dump_rsi_threshold:
                    dump_score += 0.25
                    dump_indicators.append(
                        f"RSI faiblesse extrême ({rsi_val:.1f})")
                elif rsi_val <= 45:
                    dump_score += 0.15
                    dump_indicators.append(f"RSI baissier ({rsi_val:.1f})")
            except (ValueError, TypeError):
                pass

        # Momentum négatif fort
        momentum_score = values.get("momentum_score")
        if momentum_score is not None:
            try:
                momentum_val = float(momentum_score)
                if momentum_val <= self.momentum_reversal_threshold:
                    dump_score += 0.2
                    dump_indicators.append(
                        f"Momentum négatif fort ({momentum_val:.2f})"
                    )
            except (ValueError, TypeError):
                pass

        # Volume toujours élevé mais qualité décroissante
        relative_volume = values.get("relative_volume")
        volume_quality = values.get("volume_quality_score")

        if relative_volume is not None and volume_quality is not None:
            try:
                rel_vol = float(relative_volume)
                vol_qual = float(volume_quality)
                # Volume élevé de vente (distribution/panic)
                if rel_vol >= 2.0 and vol_qual <= 60:
                    dump_score += 0.15
                    dump_indicators.append(
                        f"Volume vente massive ({rel_vol:.1f}x, qualité {vol_qual:.0f})"
                    )
            except (ValueError, TypeError):
                pass

        return {
            "is_dump": dump_score >= 0.45,  # Seuil pour détection rapide faiblesse
            "dump_score": dump_score,
            "indicators": dump_indicators,
        }

    def generate_signal(self) -> dict[str, Any]:
        """
        Génère un signal basé sur la détection de patterns pump & dump.
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

        # Vérifications préalables - ASSOUPLIE (pumps/dumps créent leur propre
        # volatilité)
        volatility_regime = values.get("volatility_regime")
        # Suppression du filtre volatilité - les patterns pump/dump peuvent
        # survenir même en faible volatilité

        # Détection des patterns
        pump_analysis = self._detect_pump_pattern(values)
        dump_analysis = self._detect_dump_pattern(values)

        signal_side = None
        reason = ""
        confidence_boost = 0.0
        metadata: dict[str, Any] = {
            "strategy": self.name, "symbol": self.symbol}

        if pump_analysis["is_pump"]:
            # Signal BUY - Pump détecté, surfer sur la vague haussière
            signal_side = "BUY"
            # Afficher plus d'indicateurs si score élevé
            indicators_to_show = (
                pump_analysis["indicators"][:3]
                if pump_analysis["pump_score"] > 0.7
                else pump_analysis["indicators"][:2]
            )
            reason = f"Pump détecté ({pump_analysis['pump_score']:.2f}): {', '.join(indicators_to_show)}"
            confidence_boost = pump_analysis["pump_score"] * 0.9

            metadata.update(
                {
                    "pattern_type": "pump",
                    "pump_score": pump_analysis["pump_score"],
                    "pump_indicators": pump_analysis["indicators"],
                }
            )

        elif dump_analysis["is_dump"]:
            # Signal SELL - Dump détecté, sortir avant la chute
            signal_side = "SELL"
            # Afficher plus d'indicateurs si score élevé
            indicators_to_show = (
                dump_analysis["indicators"][:3]
                if dump_analysis["dump_score"] > 0.7
                else dump_analysis["indicators"][:2]
            )
            reason = f"Dump détecté ({dump_analysis['dump_score']:.2f}): {', '.join(indicators_to_show)}"
            confidence_boost = dump_analysis["dump_score"] * 0.8

            metadata.update(
                {
                    "pattern_type": "dump",
                    "dump_score": dump_analysis["dump_score"],
                    "dump_indicators": dump_analysis["indicators"],
                }
            )

        if signal_side:
            # VALIDATION DIRECTIONAL BIAS - REJET si contradictoire
            directional_bias = values.get("directional_bias")
            if directional_bias and (
                (signal_side == "BUY" and directional_bias == "BEARISH") or
                (signal_side == "SELL" and directional_bias == "BULLISH")):
                return self._create_rejection_signal(
                    f"Rejet {signal_side}: bias contradictoire ({directional_bias})",
                    {"directional_bias": directional_bias}
                )

            # VALIDATION MARKET REGIME - MALUS si défavorable
            market_regime = values.get("market_regime")
            if market_regime and (
                (signal_side == "BUY" and market_regime == "TRENDING_BEAR") or
                (signal_side == "SELL" and market_regime == "TRENDING_BULL")):
                confidence_boost -= 0.20  # Malus fort pour régime contradictoire
                reason += f" MAIS régime contradictoire ({market_regime})"

            # Confirmations supplémentaires
            base_confidence = (
                0.65  # Standardisé à 0.65 pour équité avec autres stratégies
            )

            # Trade intensity pour confirmer l'activité anormale
            trade_intensity = values.get("trade_intensity")
            if trade_intensity is not None:
                try:
                    intensity = float(trade_intensity)
                    if intensity >= self.min_trade_intensity:
                        confidence_boost += 0.1
                        reason += f" + intensité ({intensity:.1f})"
                except (ValueError, TypeError):
                    pass

            # Pattern confidence du système
            pattern_confidence = values.get("pattern_confidence")
            if pattern_confidence is not None:
                try:
                    pat_conf = float(pattern_confidence)
                    if pat_conf > 70:
                        confidence_boost += 0.1
                        reason += " + pattern confirmé"
                except (ValueError, TypeError):
                    pass

            # Anomaly detected pour confirmer mouvement anormal
            anomaly_detected = values.get("anomaly_detected")
            if anomaly_detected and str(anomaly_detected).lower() == "true":
                confidence_boost += 0.15
                reason += " + anomalie détectée"

            # Confluence score
            confluence_score = values.get("confluence_score")
            if confluence_score is not None:
                try:
                    conf_score = float(confluence_score)
                    if conf_score > 60:
                        confidence_boost += 0.1
                except (ValueError, TypeError):
                    pass

            # Filtre final de confidence pour momentum trading
            raw_confidence = base_confidence * (1.0 + confidence_boost)
            if raw_confidence < 0.45:  # Seuil assoupli 45% (momentum trading)
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Signal pump/dump rejeté - confidence insuffisante ({raw_confidence:.2f} < 0.45)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "rejected_signal": signal_side,
                        "raw_confidence": raw_confidence,
                        "pattern_type": metadata.get("pattern_type"),
                    },
                }

            confidence = self.calculate_confidence(
                base_confidence, 1.0 + confidence_boost
            )
            strength = self.get_strength_from_confidence(confidence)

            # Mise à jour des métadonnées
            metadata.update({"rsi_14": values.get("rsi_14"),
                             "roc_10": values.get("roc_10"),
                             "volume_spike_multiplier": values.get("volume_spike_multiplier"),
                             "relative_volume": values.get("relative_volume"),
                             "volatility_regime": volatility_regime,
                             "trade_intensity": trade_intensity,
                             "pattern_confidence": pattern_confidence,
                             "confluence_score": confluence_score,
                             "anomaly_detected": anomaly_detected,
                             })

            return {
                "side": signal_side,
                "confidence": confidence,
                "strength": strength,
                "reason": reason,
                "metadata": metadata,
            }

        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": f"Aucun pattern pump/dump détecté (pump: {pump_analysis['pump_score']:.2f}, dump: {dump_analysis['dump_score']:.2f})",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "pump_score": pump_analysis["pump_score"],
                "dump_score": dump_analysis["dump_score"],
                "volatility_regime": volatility_regime,
            },
        }

    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        # Indicateurs ESSENTIELS seulement (autres optionnels)
        required_indicators = ["rsi_14", "roc_10", "volume_spike_multiplier"]

        # Indicateurs optionnels mais utiles

        if not self.indicators:
            logger.warning(f"{self.name}: Aucun indicateur disponible")
            return False

        # Vérifier indicateurs essentiels
        missing_required = 0
        for indicator in required_indicators:
            if indicator not in self.indicators or self.indicators[indicator] is None:
                missing_required += 1
                logger.warning(
                    f"{self.name}: Indicateur essentiel manquant: {indicator}"
                )

        # Rejeter seulement si trop d'indicateurs essentiels manquent
        if missing_required > 1:  # Tolérer 1 indicateur manquant
            logger.warning(
                f"{self.name}: Trop d'indicateurs essentiels manquants ({missing_required})"
            )
            return False

        return True

    def _convert_volatility_to_score(self, volatility_regime: str) -> float:
        """Convertit un régime de volatilité en score numérique."""
        if not volatility_regime:
            return 2.0

        try:
            vol_lower = volatility_regime.lower()

            # Mapping des régimes vers les scores
            volatility_map = {
                "high": 3.0, "very_high": 3.0, "extreme": 3.0,
                "normal": 2.0, "moderate": 2.0, "average": 2.0,
                "low": 1.0, "very_low": 1.0, "minimal": 1.0,
            }

            if vol_lower in volatility_map:
                return volatility_map[vol_lower]

            # Essayer de convertir directement en float
            try:
                return float(volatility_regime)
            except (ValueError, TypeError):
                return 2.0  # Valeur par défaut

        except Exception:
            return 2.0
