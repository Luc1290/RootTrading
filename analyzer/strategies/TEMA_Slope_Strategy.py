"""
TEMA_Slope_Strategy - Stratégie basée sur la pente du TEMA (Triple Exponential Moving Average).
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class TEMA_Slope_Strategy(BaseStrategy):
    """
    Stratégie utilisant la pente du TEMA pour détecter les changements de momentum et tendance.

    Le TEMA (Triple Exponential Moving Average) est une moyenne mobile très réactive qui réduit le lag :
    - TEMA = 3*EMA1 - 3*EMA2 + EMA3
    - EMA1 = EMA du prix, EMA2 = EMA de EMA1, EMA3 = EMA de EMA2
    - Pente positive = momentum haussier
    - Pente négative = momentum baissier

    Signaux générés:
    - BUY: TEMA pente positive forte + prix au-dessus TEMA + confirmations
    - SELL: TEMA pente négative forte + prix en-dessous TEMA + confirmations
    """

    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres TEMA Slope ANTI-SPAM - STRICTEMENT RENFORCÉS
        self.min_slope_threshold = (
            0.0003  # Pente minimum 4x plus stricte (réduit le bruit)
        )
        self.strong_slope_threshold = (
            0.0018  # Pente forte 2x plus stricte (qualité > quantité)
        )
        self.very_strong_slope_threshold = (
            0.008  # Pente très forte plus stricte (signaux premium)
        )
        self.price_tema_alignment_bonus = 0.12  # Bonus réduit (éviter sur-optimisme)

        # NOUVEAUX FILTRES ANTI-SPAM
        self.min_confluence_required = 60  # Confluence minimum OBLIGATOIRE
        self.min_volume_required = 1.25  # Volume minimum DURCI
        self.min_confirmations = 2  # Confirmations minimum requises
        self.max_distance_tema = 0.015  # Distance max prix/TEMA (éviter divergences)

    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs TEMA et contexte."""
        return {
            # TEMA principal
            "tema_12": self.indicators.get("tema_12"),
            # Autres moyennes mobiles pour comparaison
            "ema_12": self.indicators.get("ema_12"),
            "ema_26": self.indicators.get("ema_26"),
            "ema_50": self.indicators.get("ema_50"),
            "dema_12": self.indicators.get("dema_12"),  # Double EMA pour comparaison
            "hull_20": self.indicators.get("hull_20"),  # Hull MA (aussi réactive)
            # Tendance et momentum
            "trend_strength": self.indicators.get("trend_strength"),
            "trend_angle": self.indicators.get("trend_angle"),
            "directional_bias": self.indicators.get("directional_bias"),
            "momentum_score": self.indicators.get("momentum_score"),
            # ADX pour force de tendance
            "adx_14": self.indicators.get("adx_14"),
            "plus_di": self.indicators.get("plus_di"),
            "minus_di": self.indicators.get("minus_di"),
            # MACD pour confirmation momentum
            "macd_line": self.indicators.get("macd_line"),
            "macd_signal": self.indicators.get("macd_signal"),
            "macd_histogram": self.indicators.get("macd_histogram"),
            "macd_trend": self.indicators.get("macd_trend"),
            # ROC pour momentum
            "roc_10": self.indicators.get("roc_10"),
            "momentum_10": self.indicators.get("momentum_10"),
            # RSI pour confluence
            "rsi_14": self.indicators.get("rsi_14"),
            "rsi_21": self.indicators.get("rsi_21"),
            # Volume pour confirmation
            "volume_ratio": self.indicators.get("volume_ratio"),
            "volume_quality_score": self.indicators.get("volume_quality_score"),
            "trade_intensity": self.indicators.get("trade_intensity"),
            # ATR pour volatilité
            "atr_14": self.indicators.get("atr_14"),
            "atr_percentile": self.indicators.get("atr_percentile"),
            "volatility_regime": self.indicators.get("volatility_regime"),
            # VWAP pour contexte
            "vwap_10": self.indicators.get("vwap_10"),
            "anchored_vwap": self.indicators.get("anchored_vwap"),
            # Support/Résistance
            "nearest_support": self.indicators.get("nearest_support"),
            "nearest_resistance": self.indicators.get("nearest_resistance"),
            "support_strength": self.indicators.get("support_strength"),
            "resistance_strength": self.indicators.get("resistance_strength"),
            # Market structure
            "market_regime": self.indicators.get("market_regime"),
            "regime_strength": self.indicators.get("regime_strength"),
            "trend_alignment": self.indicators.get("trend_alignment"),
            # Pattern et confluence
            "pattern_detected": self.indicators.get("pattern_detected"),
            "pattern_confidence": self.indicators.get("pattern_confidence"),
            "signal_strength": self.indicators.get("signal_strength"),
            "confluence_score": self.indicators.get("confluence_score"),
        }

    def _get_current_price(self) -> Optional[float]:
        """Récupère le prix actuel depuis les données OHLCV."""
        try:
            if self.data and "close" in self.data and self.data["close"]:
                return float(self.data["close"][-1])
        except (IndexError, ValueError, TypeError):
            pass
        return None

    def _get_recent_prices(self) -> Optional[list]:
        """Récupère les derniers prix pour calculer la pente TEMA."""
        try:
            if self.data and "close" in self.data and len(self.data["close"]) >= 3:
                return [float(p) for p in self.data["close"][-3:]]
        except (IndexError, ValueError, TypeError):
            pass
        return None

    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur la pente du TEMA.
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
        recent_prices = self._get_recent_prices()

        # Analyser le TEMA et sa pente
        if current_price is None or recent_prices is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix ou données récentes non disponibles",
                "metadata": {"strategy": self.name},
            }

        tema_analysis = self._analyze_tema_slope(values, current_price, recent_prices)
        if tema_analysis is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données TEMA non disponibles",
                "metadata": {"strategy": self.name},
            }

        # Vérifier les conditions de signal
        signal_condition = self._check_tema_signal_conditions(tema_analysis)
        if signal_condition is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": tema_analysis.get(
                    "rejection_reason", "Conditions TEMA pente non remplies"
                ),
                "metadata": {"strategy": self.name},
            }

        # Créer le signal avec confirmations
        if current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix actuel non disponible",
                "metadata": {"strategy": self.name},
            }

        return self._create_tema_slope_signal(
            values, current_price, tema_analysis, signal_condition
        )

    def _analyze_tema_slope(
        self, values: Dict[str, Any], current_price: float, recent_prices: list
    ) -> Optional[Dict[str, Any]]:
        """Analyse la pente du TEMA et sa relation avec le prix."""
        tema_12 = values.get("tema_12")

        if tema_12 is None:
            return None

        try:
            tema_val = float(tema_12)
        except (ValueError, TypeError):
            return None

        # Calculer une approximation de la pente TEMA
        # En l'absence de valeurs TEMA historiques, utiliser trend_angle ou approximer
        tema_slope = None
        slope_strength = "unknown"

        # Méthode 1: Utiliser trend_angle si disponible
        trend_angle = values.get("trend_angle")
        if trend_angle is not None:
            try:
                angle = float(trend_angle)
                tema_slope = angle  # Approximation
            except (ValueError, TypeError):
                pass

        # Méthode 2 AMÉLIORÉE: Fallback EMA plus stable que prix brut
        if tema_slope is None and recent_prices is not None and len(recent_prices) >= 3:
            try:
                # Essayer fallback EMA_12 d'abord (plus stable)
                ema_12 = values.get("ema_12")
                if ema_12 is not None:
                    ema_val = float(ema_12)
                    price_change = recent_prices[-1] - recent_prices[0]
                    ema_slope_approx = price_change / max(abs(ema_val), 1e-9)
                    tema_slope = ema_slope_approx
                else:
                    # Fallback prix brut si pas d'EMA
                    price_change = recent_prices[-1] - recent_prices[0]
                    price_slope = (
                        price_change / recent_prices[0] if recent_prices[0] != 0 else 0
                    )
                    price_tema_ratio = current_price / tema_val if tema_val != 0 else 1
                    tema_slope = price_slope * price_tema_ratio
            except (ValueError, TypeError, ZeroDivisionError, IndexError):
                tema_slope = 0

        if tema_slope is None:
            tema_slope = 0

        # Classifier la force de la pente
        abs_slope = abs(tema_slope)
        if abs_slope >= self.very_strong_slope_threshold:
            slope_strength = "very_strong"
        elif abs_slope >= self.strong_slope_threshold:
            slope_strength = "strong"
        elif abs_slope >= self.min_slope_threshold:
            slope_strength = "moderate"
        else:
            slope_strength = "weak"

        # Direction de la pente
        slope_direction = (
            "bullish" if tema_slope > 0 else "bearish" if tema_slope < 0 else "neutral"
        )

        # Relation prix/TEMA
        price_above_tema = current_price > tema_val
        price_tema_distance = (
            abs(current_price - tema_val) / tema_val if tema_val != 0 else 0
        )

        # Alignement prix/TEMA/pente
        alignment = None
        if slope_direction == "bullish" and price_above_tema:
            alignment = "bullish_aligned"
        elif slope_direction == "bearish" and not price_above_tema:
            alignment = "bearish_aligned"
        elif slope_direction == "bullish" and not price_above_tema:
            alignment = "bullish_divergent"
        elif slope_direction == "bearish" and price_above_tema:
            alignment = "bearish_divergent"
        else:
            alignment = "neutral"

        # Raisons de rejet potentielles
        rejection_reasons = []
        if slope_strength == "weak":
            rejection_reasons.append(f"Pente TEMA trop faible ({abs_slope:.6f})")
        if alignment in ["bullish_divergent", "bearish_divergent"]:
            rejection_reasons.append(f"Prix et pente TEMA divergents ({alignment})")

        return {
            "tema_value": tema_val,
            "tema_slope": tema_slope,
            "slope_strength": slope_strength,
            "slope_direction": slope_direction,
            "price_above_tema": price_above_tema,
            "price_tema_distance": price_tema_distance,
            "alignment": alignment,
            "rejection_reason": (
                "; ".join(rejection_reasons) if rejection_reasons else None
            ),
        }

    def _check_tema_signal_conditions(
        self, tema_analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Vérifie si les conditions de signal TEMA sont remplies."""
        slope_strength = tema_analysis["slope_strength"]
        slope_direction = tema_analysis["slope_direction"]
        alignment = tema_analysis["alignment"]

        # FILTRES ANTI-SPAM RENFORCÉS

        # Rejeter si pente trop faible (inchangé mais plus strict maintenant)
        if slope_strength == "weak":
            return None

        # Rejeter si prix/pente divergents (inchangé)
        if alignment in ["bullish_divergent", "bearish_divergent"]:
            return None

        # Rejeter si pente neutre (inchangé)
        if slope_direction == "neutral":
            return None

        # NOUVEAU: Rejeter si distance prix/TEMA trop importante (éviter faux signaux)
        if tema_analysis["price_tema_distance"] > self.max_distance_tema:
            return None

        # Déterminer le type de signal
        if alignment == "bullish_aligned":
            signal_side = "BUY"
        elif alignment == "bearish_aligned":
            signal_side = "SELL"
        else:
            return None

        return {
            "signal_side": signal_side,
            "slope_strength": slope_strength,
            "slope_direction": slope_direction,
            "alignment": alignment,
        }

    def _create_tema_slope_signal(
        self,
        values: Dict[str, Any],
        current_price: float,
        tema_analysis: Dict[str, Any],
        signal_condition: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Crée le signal TEMA slope avec confirmations."""
        signal_side = signal_condition["signal_side"]
        slope_strength = signal_condition["slope_strength"]
        slope_direction = signal_condition["slope_direction"]
        alignment = signal_condition["alignment"]

        base_confidence = 0.65  # Standardisé à 0.65 pour équité avec autres stratégies
        confidence_boost = 0.0

        tema_val = tema_analysis["tema_value"]
        tema_slope = tema_analysis["tema_slope"]
        price_tema_distance = tema_analysis["price_tema_distance"]

        # Construction de la raison
        reason = f"TEMA pente {slope_direction}: {tema_slope:.6f} ({slope_strength})"

        # VALIDATION OBLIGATOIRE CONFLUENCE ANTI-SPAM
        confluence_score = values.get("confluence_score", 0)
        if (
            not confluence_score
            or float(confluence_score) < self.min_confluence_required
        ):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Confluence insuffisante ({confluence_score}) < {self.min_confluence_required} - signal TEMA rejeté",
                "metadata": {
                    "strategy": self.name,
                    "rejected_reason": "low_confluence",
                },
            }

        # REJETS CRITIQUES - régime/bias contradictoires
        market_regime = values.get("market_regime")
        directional_bias = values.get("directional_bias")

        if signal_side == "BUY":
            if market_regime == "TRENDING_BEAR":
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "Rejet TEMA: BUY interdit en TRENDING_BEAR",
                    "metadata": {"strategy": self.name, "market_regime": market_regime},
                }
            if str(directional_bias).upper() == "BEARISH":
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "Rejet TEMA: bias contradictoire (BEARISH)",
                    "metadata": {
                        "strategy": self.name,
                        "directional_bias": directional_bias,
                    },
                }
        elif signal_side == "SELL":
            if market_regime == "TRENDING_BULL":
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "Rejet TEMA: SELL interdit en TRENDING_BULL",
                    "metadata": {"strategy": self.name, "market_regime": market_regime},
                }
            if str(directional_bias).upper() == "BULLISH":
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "Rejet TEMA: bias contradictoire (BULLISH)",
                    "metadata": {
                        "strategy": self.name,
                        "directional_bias": directional_bias,
                    },
                }

        # VALIDATION OBLIGATOIRE VOLUME ANTI-SPAM
        volume_ratio = values.get("volume_ratio", 0)
        if not volume_ratio or float(volume_ratio) < self.min_volume_required:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Volume insuffisant ({volume_ratio}) < {self.min_volume_required} - signal TEMA rejeté",
                "metadata": {"strategy": self.name, "rejected_reason": "low_volume"},
            }

        # Bonus selon la force de la pente (RÉDUITS pour éviter sur-confiance)
        if slope_strength == "very_strong":
            confidence_boost += 0.20  # Réduit de 0.25 à 0.20
            reason += " - momentum très fort"
        elif slope_strength == "strong":
            confidence_boost += 0.16  # Réduit de 0.20 à 0.16
            reason += " - momentum fort"
        else:  # moderate
            confidence_boost += 0.12  # Réduit de 0.15 à 0.12
            reason += " - momentum modéré"

        # Bonus selon l'alignement prix/TEMA
        if alignment in ["bullish_aligned", "bearish_aligned"]:
            confidence_boost += self.price_tema_alignment_bonus
            distance_text = f"distance {price_tema_distance:.3f}"
            reason += f" + prix/TEMA alignés ({distance_text})"

        # SYSTÈME DE CONFIRMATIONS OBLIGATOIRES ANTI-SPAM
        confirmations_count = 0
        required_confirmations = []

        # Confirmation avec autres moyennes mobiles (PREMIÈRE CONFIRMATION OBLIGATOIRE)
        ema_12 = values.get("ema_12")
        ema_26 = values.get("ema_26")
        ema_confirmed = False
        if ema_12 is not None and ema_26 is not None:
            try:
                ema12_val = float(ema_12)
                ema26_val = float(ema_26)
                ema_cross_aligned = (
                    signal_side == "BUY" and ema12_val > ema26_val
                ) or (signal_side == "SELL" and ema12_val < ema26_val)

                if ema_cross_aligned:
                    confirmations_count += 1
                    ema_confirmed = True
                    confidence_boost += 0.10  # Réduit de 0.12
                    reason += " + EMA confirme"
                else:
                    confidence_boost -= 0.08  # Pénalité plus forte
                    reason += " mais EMA diverge"
            except (ValueError, TypeError):
                pass

        required_confirmations.append(("EMA_Cross", ema_confirmed))

        # Confirmation avec Hull MA (autre MA réactive)
        hull_20 = values.get("hull_20")
        if hull_20 is not None:
            try:
                hull_val = float(hull_20)
                hull_aligned = (signal_side == "BUY" and current_price > hull_val) or (
                    signal_side == "SELL" and current_price < hull_val
                )

                if hull_aligned:
                    confidence_boost += 0.10
                    reason += " + Hull MA confirme"
            except (ValueError, TypeError):
                pass

        # Confirmation avec DEMA (Double EMA)
        dema_12 = values.get("dema_12")
        if dema_12 is not None:
            try:
                dema_val = float(dema_12)
                dema_aligned = (signal_side == "BUY" and current_price > dema_val) or (
                    signal_side == "SELL" and current_price < dema_val
                )

                if dema_aligned:
                    confidence_boost += 0.08
                    reason += " + DEMA confirme"
            except (ValueError, TypeError):
                pass

        # Confirmation avec momentum indicators (DEUXIÈME CONFIRMATION OBLIGATOIRE)
        momentum_score = values.get("momentum_score")
        momentum_confirmed = False
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                # BUY : momentum positif REQUIS (plus strict maintenant)
                if signal_side == "BUY":
                    if momentum > 65:
                        confirmations_count += 1
                        momentum_confirmed = True
                        confidence_boost += 0.15  # Réduit de 0.18
                        reason += f" + momentum très positif ({momentum:.1f})"
                    elif momentum > 60:  # Seuil relevé de 58 à 60
                        confirmations_count += 1
                        momentum_confirmed = True
                        confidence_boost += 0.10  # Réduit de 0.12
                        reason += f" + momentum positif ({momentum:.1f})"
                    elif momentum < 48:  # Pénalité plus forte
                        confidence_boost -= 0.15  # Plus pénalisant
                        reason += f" mais momentum insuffisant ({momentum:.1f})"
                # SELL : momentum négatif REQUIS (plus strict maintenant)
                elif signal_side == "SELL":
                    if momentum < 35:
                        confirmations_count += 1
                        momentum_confirmed = True
                        confidence_boost += 0.15  # Réduit de 0.18
                        reason += f" + momentum très négatif ({momentum:.1f})"
                    elif momentum < 40:  # Seuil relevé de 42 à 40
                        confirmations_count += 1
                        momentum_confirmed = True
                        confidence_boost += 0.10  # Réduit de 0.12
                        reason += f" + momentum négatif ({momentum:.1f})"
                    elif momentum > 52:  # Pénalité plus forte
                        confidence_boost -= 0.15  # Plus pénalisant
                        reason += f" mais momentum insuffisant ({momentum:.1f})"
            except (ValueError, TypeError):
                pass

        required_confirmations.append(("Momentum", momentum_confirmed))

        # CORRECTION: Confirmation avec ROC - seuils directionnels adaptatifs (format décimal)
        roc_10 = values.get("roc_10")
        if roc_10 is not None:
            try:
                roc = float(roc_10)
                # BUY : ROC positif avec différents niveaux (format décimal)
                if signal_side == "BUY":
                    if roc > 0.03:  # ROC > 3%
                        confidence_boost += (
                            0.15  # ROC très positif = excellent pour BUY
                        )
                        reason += f" + ROC très positif ({roc*100:.2f}%)"
                    elif roc > 0.015:  # ROC > 1.5%
                        confidence_boost += 0.12  # ROC positif fort
                        reason += f" + ROC positif fort ({roc*100:.2f}%)"
                    elif roc > 0.005:  # ROC > 0.5%
                        confidence_boost += 0.08  # ROC positif modéré
                        reason += f" + ROC positif ({roc*100:.2f}%)"
                    elif roc < -0.01:  # ROC < -1%
                        confidence_boost -= 0.08  # ROC négatif = défavorable pour BUY
                        reason += f" mais ROC négatif ({roc*100:.2f}%)"
                # SELL : ROC négatif avec différents niveaux (format décimal)
                elif signal_side == "SELL":
                    if roc < -0.03:  # ROC < -3%
                        confidence_boost += (
                            0.15  # ROC très négatif = excellent pour SELL
                        )
                        reason += f" + ROC très négatif ({roc*100:.2f}%)"
                    elif roc < -0.015:  # ROC < -1.5%
                        confidence_boost += 0.12  # ROC négatif fort
                        reason += f" + ROC négatif fort ({roc*100:.2f}%)"
                    elif roc < -0.005:  # ROC < -0.5%
                        confidence_boost += 0.08  # ROC négatif modéré
                        reason += f" + ROC négatif ({roc*100:.2f}%)"
                    elif roc > 0.01:  # ROC > 1%
                        confidence_boost -= 0.08  # ROC positif = défavorable pour SELL
                        reason += f" mais ROC positif ({roc*100:.2f}%)"
            except (ValueError, TypeError):
                pass

        # Confirmation avec MACD - COMPTE pour confirmations
        macd_line = values.get("macd_line")
        macd_signal = values.get("macd_signal")
        macd_aligned = False
        if macd_line is not None and macd_signal is not None:
            try:
                macd_val = float(macd_line)
                macd_sig = float(macd_signal)
                macd_aligned = (signal_side == "BUY" and macd_val > macd_sig) or (
                    signal_side == "SELL" and macd_val < macd_sig
                )

                if macd_aligned:
                    confirmations_count += 1  # COMPTE maintenant
                    confidence_boost += 0.10
                    reason += " + MACD confirme"
            except (ValueError, TypeError):
                pass

        # Confirmation avec directional bias
        directional_bias = values.get("directional_bias")
        if directional_bias is not None:
            try:
                bias_str = str(directional_bias).upper()
                if (signal_side == "BUY" and bias_str == "BULLISH") or (
                    signal_side == "SELL" and bias_str == "BEARISH"
                ):
                    confidence_boost += 0.12
                    reason += f" + bias {directional_bias}"
            except (AttributeError, TypeError):
                pass

        # Confirmation avec trend strength
        trend_strength = values.get("trend_strength")
        if trend_strength is not None:
            try:
                trend_strength_val = float(trend_strength)
                if trend_strength_val > 0.6:
                    confidence_boost += 0.12
                    reason += f" + tendance forte ({trend_strength_val:.2f})"
                elif trend_strength_val > 0.4:
                    confidence_boost += 0.08
                    reason += f" + tendance modérée ({trend_strength_val:.2f})"
            except (ValueError, TypeError):
                pass

        # Confirmation avec ADX - COMPTE pour confirmations
        adx_14 = values.get("adx_14")
        if adx_14 is not None:
            try:
                adx = float(adx_14)
                if adx >= 25:  # Tendance forte
                    confirmations_count += 1  # COMPTE maintenant
                    confidence_boost += 0.12
                    reason += " + ADX fort"
                elif adx > 20:
                    confidence_boost += 0.08
                    reason += " + ADX modéré"
            except (ValueError, TypeError):
                pass

        # CORRECTION: Confirmation avec RSI - zones directionnelles optimales
        rsi_14 = values.get("rsi_14")
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                # BUY : favoriser RSI en reprise depuis zone basse
                if signal_side == "BUY":
                    if 45 <= rsi <= 65:
                        confidence_boost += (
                            0.12  # Zone optimale BUY (momentum haussier)
                        )
                        reason += " + RSI haussier optimal"
                    elif 35 <= rsi <= 44:
                        confidence_boost += 0.08  # Sortie d'oversold = bon pour BUY
                        reason += " + RSI sortie oversold"
                    elif rsi >= 75:
                        confidence_boost -= 0.08  # Overbought = risqué pour BUY
                        reason += " mais RSI overbought"
                    elif rsi <= 30:
                        confidence_boost += 0.05  # Oversold peut être opportunité BUY
                        reason += " + RSI oversold (opportunité)"
                # SELL : favoriser RSI en déclin depuis zone haute
                elif signal_side == "SELL":
                    if 35 <= rsi <= 55:
                        confidence_boost += (
                            0.12  # Zone optimale SELL (momentum baissier)
                        )
                        reason += " + RSI baissier optimal"
                    elif 56 <= rsi <= 65:
                        confidence_boost += 0.08  # Entrée en overbought = bon pour SELL
                        reason += " + RSI entrée overbought"
                    elif rsi <= 25:
                        confidence_boost -= 0.08  # Oversold = risqué pour SELL
                        reason += " mais RSI oversold"
                    elif rsi >= 70:
                        confidence_boost += (
                            0.05  # Overbought peut être opportunité SELL
                        )
                        reason += " + RSI overbought (opportunité)"
            except (ValueError, TypeError):
                pass

        # VALIDATION FINALE CONFIRMATIONS ANTI-SPAM
        if confirmations_count < self.min_confirmations:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Confirmations insuffisantes ({confirmations_count}/{self.min_confirmations}) - signal TEMA rejeté",
                "metadata": {
                    "strategy": self.name,
                    "rejected_reason": "insufficient_confirmations",
                    "confirmations_count": confirmations_count,
                    "required_confirmations": required_confirmations,
                },
            }

        # Volume confirmation (DÉJÀ VALIDÉ PLUS HAUT - juste bonus ici)
        # volume_ratio déjà vérifié comme obligatoire, ici juste bonus additionnel
        try:
            vol_ratio = float(volume_ratio)
            if vol_ratio >= 1.4:  # Seuil relevé pour bonus
                confidence_boost += 0.10  # Réduit de 0.12
                reason += f" + volume élevé ({vol_ratio:.1f}x)"
            elif vol_ratio >= 1.25:  # Seuil relevé
                confidence_boost += 0.06  # Réduit de 0.08
                reason += f" + volume modéré ({vol_ratio:.1f}x)"
        except (ValueError, TypeError):
            pass

        # VWAP context
        vwap_10 = values.get("vwap_10")
        if vwap_10 is not None:
            try:
                vwap = float(vwap_10)
                vwap_aligned = (signal_side == "BUY" and current_price > vwap) or (
                    signal_side == "SELL" and current_price < vwap
                )

                if vwap_aligned:
                    confidence_boost += 0.08
                    reason += " + VWAP aligné"
            except (ValueError, TypeError):
                pass

        # Support/Resistance context
        if signal_side == "BUY":
            nearest_support = values.get("nearest_support")
            if nearest_support is not None:
                try:
                    support = float(nearest_support)
                    distance_to_support = abs(current_price - support) / current_price
                    if distance_to_support <= 0.02:
                        confidence_boost += 0.10
                        reason += " + près support"
                except (ValueError, TypeError):
                    pass
        else:  # SELL
            nearest_resistance = values.get("nearest_resistance")
            if nearest_resistance is not None:
                try:
                    resistance = float(nearest_resistance)
                    distance_to_resistance = (
                        abs(current_price - resistance) / current_price
                    )
                    if distance_to_resistance <= 0.02:
                        confidence_boost += 0.10
                        reason += " + près résistance"
                except (ValueError, TypeError):
                    pass

        # Market regime - seulement BONUS si aligné (rejets déjà traités)
        if (signal_side == "BUY" and market_regime == "TRENDING_BULL") or (
            signal_side == "SELL" and market_regime == "TRENDING_BEAR"
        ):
            confidence_boost += 0.10
            reason += f" + régime aligné ({market_regime})"
        elif market_regime == "RANGING":
            confidence_boost -= 0.05  # TEMA slope moins fiable en ranging
            reason += " (marché ranging)"

        # CORRECTION: Volatility context - adaptation selon force pente et direction
        volatility_regime = values.get("volatility_regime")
        if volatility_regime == "normal":
            confidence_boost += 0.08  # Volatilité normale toujours favorable
            reason += " + volatilité normale"
        elif volatility_regime == "low":
            # Basse volatilité : pentes fortes plus significatives
            if slope_strength in ["strong", "very_strong"]:
                confidence_boost += (
                    0.12  # Pente forte en basse vol = signal très fiable
                )
                reason += " + basse volatilité pente forte"
            else:
                confidence_boost += 0.03  # Pente faible en basse vol = signal faible
                reason += " + basse volatilité"
        elif volatility_regime == "high":
            # Haute volatilité : adapter selon direction et force
            if slope_strength == "very_strong":
                confidence_boost += (
                    0.10  # Pente très forte maîtrise la haute volatilité
                )
                reason += " + haute volatilité maîtrisée"
            elif slope_strength == "strong":
                confidence_boost += 0.05  # Pente forte acceptable en haute vol
                reason += " + haute volatilité acceptable"
            else:
                confidence_boost -= (
                    0.08  # Pente faible en haute vol = signal peu fiable
                )
                reason += " mais haute volatilité défavorable"
        elif volatility_regime == "extreme":
            # Volatilité extrême : très sélectif
            if (
                slope_strength == "very_strong"
                and abs(tema_slope) > self.very_strong_slope_threshold * 2
            ):
                confidence_boost += 0.08  # Pente exceptionnelle en vol extrême
                reason += " + volatilité extrême maîtrisée"
            else:
                confidence_boost -= 0.12  # Trop risqué en volatilité extrême
                reason += " mais volatilité extrême"

        # Confluence score (DÉJÀ VALIDÉ COMME OBLIGATOIRE - juste bonus ici)
        # confluence_score déjà vérifié comme obligatoire, ici juste bonus additionnel
        try:
            confluence = float(confluence_score)
            if confluence > 80:  # Seuil élevé pour bonus
                confidence_boost += 0.08  # Réduit de 0.10
                reason += f" + confluence excellente ({confluence:.0f})"
            elif confluence > 70:  # Seuil relevé
                confidence_boost += 0.05  # Réduit
                reason += f" + confluence élevée ({confluence:.0f})"
        except (ValueError, TypeError):
            pass

        # FILTRE FINAL CONFIANCE MINIMUM ANTI-SPAM + CLAMP EXPLICITE
        raw_confidence = self.calculate_confidence(
            base_confidence, 1 + confidence_boost
        )
        if raw_confidence < 0.55:  # Seuil minimum élevé pour TEMA
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Confiance finale insuffisante ({raw_confidence:.2f}) < 0.55 - signal TEMA rejeté",
                "metadata": {
                    "strategy": self.name,
                    "rejected_reason": "low_final_confidence",
                    "raw_confidence": raw_confidence,
                    "confirmations_count": confirmations_count,
                },
            }

        # Clamp explicite de la confiance pour homogénéité
        confidence = min(
            1.0,
            max(0.0, self.calculate_confidence(base_confidence, 1 + confidence_boost)),
        )
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
                "tema_value": tema_val,
                "tema_slope": tema_slope,
                "slope_strength": slope_strength,
                "slope_direction": slope_direction,
                "alignment": alignment,
                "price_tema_distance": price_tema_distance,
                "ema_12": values.get("ema_12"),
                "ema_26": values.get("ema_26"),
                "hull_20": values.get("hull_20"),
                "dema_12": values.get("dema_12"),
                "momentum_score": values.get("momentum_score"),
                "roc_10": values.get("roc_10"),
                "macd_line": values.get("macd_line"),
                "macd_signal": values.get("macd_signal"),
                "directional_bias": values.get("directional_bias"),
                "trend_strength": values.get("trend_strength"),
                "adx_14": values.get("adx_14"),
                "rsi_14": values.get("rsi_14"),
                "volume_ratio": values.get("volume_ratio"),
                "vwap_10": values.get("vwap_10"),
                "market_regime": values.get("market_regime"),
                "volatility_regime": values.get("volatility_regime"),
                "confluence_score": values.get("confluence_score"),
            },
        }

    def validate_data(self) -> bool:
        """Valide que les données TEMA nécessaires sont présentes - ANTI-SPAM RENFORCÉ."""
        if not super().validate_data():
            return False

        # INDICATEURS OBLIGATOIRES ANTI-SPAM (plus de TEMA_12 seulement)
        essential_indicators = [
            "tema_12",
            "confluence_score",
            "volume_ratio",
            "momentum_score",
        ]
        optional_but_preferred = ["ema_12", "ema_26", "trend_strength"]

        missing_essential = 0
        for indicator in essential_indicators:
            if indicator not in self.indicators or self.indicators[indicator] is None:
                missing_essential += 1
                logger.debug(f"{self.name}: Indicateur essentiel manquant: {indicator}")

        # Au moins 3/4 indicateurs essentiels requis
        if missing_essential > 1:
            logger.warning(
                f"{self.name}: Trop d'indicateurs essentiels manquants ({missing_essential}/4)"
            )
            return False

        # Vérifier qualité minimale des données critiques
        if (
            "confluence_score" in self.indicators
            and self.indicators["confluence_score"]
        ):
            try:
                conf_val = float(self.indicators["confluence_score"])
                if (
                    conf_val < 30
                ):  # Confluence trop faible = données de mauvaise qualité
                    logger.debug(
                        f"{self.name}: Confluence trop faible pour validation: {conf_val}"
                    )
                    return False
            except (ValueError, TypeError):
                pass

        return True
