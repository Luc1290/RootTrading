"""
Calculateur d'opportunités de trading manuel - VERSION OPTIMISÉE v2.0
Analyse multi-critères pour signaux BUY/WAIT/AVOID sur timeframe 1m/5m scalping
Optimisé pour SPOT avec target 1%+ en 5-30min
Utilise TOUS les 108+ indicateurs disponibles en DB
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OpportunityCalculator:
    """
    Calcule le score d'opportunité de trading basé sur 6 piliers OPTIMISÉS:
    - Trend Quality (25 pts)
    - Momentum Confluence (35 pts) ↑ Ajout Williams %R, CCI, ROC
    - Volume Validation (32 pts) ↑ Ajout OBV, Trade Intensity, Spikes
    - Price Action (20 pts) ↑ Ajout VWAP bands, Volume Profile POC
    - Consensus & Signals (10 pts)
    - Institutional Flow (20 pts) ★ NOUVEAU - Smart Money tracking

    Total: 142 points (au lieu de 100)
    Optimisé pour scalping SPOT 1m/5m avec détection institutionnelle
    Seuils calibrés pour réduire les faux signaux
    """

    @staticmethod
    def safe_float(value, default=0.0):
        """Convertir en float avec fallback"""
        return float(value) if value is not None else default

    def _check_quality_gates(self, ad: dict, current_price: float, atr_percent: float) -> tuple[bool, str]:
        """
        3 Quality Gates qui court-circuitent AVANT le scoring
        Élimine les setups médiocres dès le départ (SPOT = pas de short, exit rapide obligatoire)

        Returns:
            (gate_passed, reason)
        """

        # ============================================================
        # GATE A - R/R MINIMAL (CRITIQUE pour SPOT)
        # ============================================================
        nearest_support = self.safe_float(ad.get('nearest_support'))
        nearest_resistance = self.safe_float(ad.get('nearest_resistance'))
        bb_position = self.safe_float(ad.get('bb_position'), 0.5)

        # Calcul distances TP/SL pour scalping SPOT
        # Target: 1% minimum (scalping conservateur)
        tp1_dist = max(0.01, atr_percent * 1.2)   # Target minimum 1% OU 1.2x ATR

        # SL intelligent: utiliser nearest_support si disponible, sinon ATR
        if nearest_support > 0 and current_price > nearest_support:
            sl_dist = max(0.007, (current_price - nearest_support) / current_price)
        else:
            sl_dist = max(0.007, atr_percent * 0.7)   # Fallback ATR

        rr_ratio = tp1_dist / sl_dist if sl_dist > 0 else 0

        # R/R minimal strict pour SPOT
        if rr_ratio < 1.40:
            sl_basis = "support" if (nearest_support > 0 and current_price > nearest_support) else "ATR"
            return False, f"❌ Gate A (R/R): Ratio {rr_ratio:.2f} < 1.40 (reward insuffisant) | TP:{tp1_dist*100:.2f}% vs SL:{sl_dist*100:.2f}% (base:{sl_basis})"

        # Refuse si résistance < TP1 distance (impossible d'atteindre target)
        if nearest_resistance > 0:
            dist_to_resistance_pct = ((float(nearest_resistance) - float(current_price)) / float(current_price)) * 100

            # Gate CRITIQUE: résistance plus proche que notre target
            if dist_to_resistance_pct < (tp1_dist * 100):
                return False, f"❌ Gate A (Résistance): Résistance à {dist_to_resistance_pct:.2f}% < Target {tp1_dist*100:.2f}% → Impossible d'atteindre TP1"

            # Gate secondaire: résistance TRÈS proche + prix déjà haut + overbought
            rsi = self.safe_float(ad.get('rsi_14'), 50)
            mfi = self.safe_float(ad.get('mfi_14'), 50)

            if dist_to_resistance_pct < 0.3 and bb_position > 0.95 and (rsi > 70 or mfi > 75):
                return False, f"❌ Gate A (Résistance): Collé au plafond {dist_to_resistance_pct:.1f}% + BB {bb_position:.2f} + Overbought → Bloqué"

        # ============================================================
        # GATE B - VOLUME ABSOLU (confirmation obligatoire)
        # ============================================================
        vol_context = ad.get('volume_context')
        rel_volume = self.safe_float(ad.get('relative_volume'), 1.0)
        obv_osc = self.safe_float(ad.get('obv_oscillator'))
        vol_pattern = ad.get('volume_pattern')

        # Refuse DISTRIBUTION (vente institutionnelle forte)
        if vol_context == 'DISTRIBUTION' and rel_volume > 1.5:
            return False, f"❌ Gate B (Volume): Context DISTRIBUTION avec volume {rel_volume:.1f}x → Smart money sort massivement"

        # Refuse si volume relatif < 0.5x (vraiment mort)
        if rel_volume < 0.5:
            return False, f"❌ Gate B (Volume): Rel volume {rel_volume:.2f}x < 0.5x → Marché mort"

        # Refuse si OBV négatif + volume déclinant (momentum mort)
        if obv_osc < -200 and vol_pattern == 'DECLINING':
            return False, f"❌ Gate B (OBV): OBV {obv_osc:.0f} + volume DECLINING → Momentum mort"

        # ============================================================
        # GATE C - VWAP POSITION (référence institutionnelle)
        # ============================================================
        vwap = self.safe_float(ad.get('vwap_quote_10'))
        vwap_upper = self.safe_float(ad.get('vwap_upper_band'))

        # Refuse si prix > VWAP upper band ET résistance proche (overbought + plafonné)
        if vwap_upper > 0 and float(current_price) > float(vwap_upper):
            if nearest_resistance > 0:
                dist_to_resistance_pct = ((float(nearest_resistance) - float(current_price)) / float(current_price)) * 100
                if dist_to_resistance_pct < 1.0:  # Résistance < 1%
                    return False, f"❌ Gate C (VWAP): Prix > VWAP upper + résistance {dist_to_resistance_pct:.1f}% → Overbought plafonné"

        # ✅ Tous les gates passés
        return True, f"✅ Quality Gates OK | R/R:{rr_ratio:.2f} | Vol:{rel_volume:.1f}x | Context:{vol_context or 'N/A'}"

    def _check_higher_timeframe(self, higher_tf: Optional[dict]) -> tuple[bool, str]:
        """
        Vérifie l'alignement du timeframe supérieur (5m)
        Pour scalping 1m, le 5m doit confirmer la direction (pas de contre-tendance)

        Returns:
            (is_aligned, reason)
        """
        # Si pas de données 5m, on accepte (mode dégradé)
        if not higher_tf:
            return True, "✅ 5m: données indisponibles (mode dégradé)"

        rsi_5m = self.safe_float(higher_tf.get('rsi_14'), 50)
        macd_trend_5m = higher_tf.get('macd_trend')
        regime_5m = higher_tf.get('market_regime', 'UNKNOWN')
        plus_di_5m = self.safe_float(higher_tf.get('plus_di'))
        minus_di_5m = self.safe_float(higher_tf.get('minus_di'))

        # Critères d'alignement 5m (au moins 1 critère bullish + pas de régime baissier fort)
        is_bullish = (
            macd_trend_5m == 'BULLISH' or
            rsi_5m > 50 or
            plus_di_5m > minus_di_5m
        )

        is_bear_regime = regime_5m in ['TRENDING_BEAR', 'BREAKOUT_BEAR']

        # Rejet si contre-tendance forte (régime baissier + aucun indicateur bullish)
        if is_bear_regime and not is_bullish:
            return False, f"🟡 5m contre-tendance forte: Régime={regime_5m}, RSI={rsi_5m:.0f}, MACD={macd_trend_5m or 'N/A'}, +DI={plus_di_5m:.0f} vs -DI={minus_di_5m:.0f}"

        # Rejet si TOUS les indicateurs baissiers
        if not is_bullish:
            return False, f"🟡 5m tous indicateurs baissiers: RSI={rsi_5m:.0f}<50, MACD={macd_trend_5m or 'BEARISH'}, -DI>{plus_di_5m:.0f}"

        # ✅ Contexte 5m acceptable
        return True, f"✅ 5m aligné: Régime={regime_5m}, RSI={rsi_5m:.0f}, MACD={macd_trend_5m or 'N/A'}"

    def calculate_opportunity(
        self,
        symbol: str,
        current_price: float,
        analyzer_data: Optional[dict],
        signals_data: Optional[dict],
        higher_tf: Optional[dict] = None
    ) -> dict:
        """
        Calcule l'opportunité de trading pour un symbole

        Args:
            symbol: Symbole à analyser
            current_price: Prix actuel
            analyzer_data: Données techniques de l'analyzer (1m)
            signals_data: Données des signaux de trading
            higher_tf: Données techniques du timeframe supérieur (5m) pour validation contexte

        Returns:
            dict: Opportunité complète avec score, action, targets, etc.
        """
        if not analyzer_data:
            return {
                "symbol": symbol,
                "score": 0,
                "action": "AVOID",
                "reason": "Pas de données techniques disponibles"
            }

        ad = analyzer_data
        signals_count = int(signals_data.get('count', 0)) if signals_data else 0
        avg_confidence = self.safe_float(signals_data.get('avg_conf')) if signals_data else 0

        # Calculer ATR en pourcentage avec fallback intelligent
        atr_value = self.safe_float(ad.get('atr_14'))
        natr = self.safe_float(ad.get('natr'))  # Normalized ATR (déjà en %)

        if atr_value > 0 and current_price > 0:
            atr_percent = atr_value / current_price
        elif natr > 0:
            atr_percent = natr / 100.0  # natr est en pourcentage
        else:
            # Pas de données ATR fiables → ne pas trader
            return {
                "symbol": symbol,
                "score": 0,
                "action": "WAIT_DATA",
                "reason": "ATR/NATR indisponibles → volatilité non mesurable, setup invalide"
            }

        # ============================================================
        # QUALITY GATES - Vérifie mais ne bloque PAS le scoring (juste l'action)
        # ============================================================
        gate_passed, gate_reason = self._check_quality_gates(ad, current_price, atr_percent)

        # ============================================================
        # MULTI-TIMEFRAME GATE - Vérifier contexte 5m si disponible
        # ============================================================
        higher_tf_ok, higher_tf_reason = self._check_higher_timeframe(higher_tf)

        # ============================================================
        # 1. TREND QUALITY (25 points)
        # ============================================================
        trend_score = self._calculate_trend_score(ad)

        # ============================================================
        # 2. MOMENTUM CONFLUENCE (35 points) ↑ OPTIMISÉ
        # ============================================================
        momentum_score = self._calculate_momentum_score(ad)

        # ============================================================
        # 3. VOLUME VALIDATION (32 points) ↑ OPTIMISÉ
        # ============================================================
        volume_score = self._calculate_volume_score(ad)

        # ============================================================
        # 4. PRICE ACTION (20 points) ↑ OPTIMISÉ
        # ============================================================
        price_action_score = self._calculate_price_action_score(ad, current_price)

        # ============================================================
        # 5. CONSENSUS & SIGNALS (10 points)
        # ============================================================
        consensus_score = self._calculate_consensus_score(ad)

        # ============================================================
        # 6. INSTITUTIONAL FLOW (20 points) ★ NOUVEAU
        # ============================================================
        institutional_score = self._calculate_institutional_flow(ad, current_price)

        # ============================================================
        # SCORE TOTAL (142 points max)
        # ============================================================
        total_score = trend_score + momentum_score + volume_score + price_action_score + consensus_score + institutional_score

        # ============================================================
        # ESTIMATION DURÉE DE HOLD (scalping 5m)
        # ============================================================
        estimated_hold_time = self._estimate_hold_time(ad, atr_percent, momentum_score)

        # Détails des composantes
        score_details = {
            "trend": round(trend_score, 1),
            "momentum": round(momentum_score, 1),
            "volume": round(volume_score, 1),
            "price_action": round(price_action_score, 1),
            "consensus": round(consensus_score, 1),
            "institutional": round(institutional_score, 1)
        }

        # Explication détaillée de chaque pilier
        score_explanation = self._build_score_explanation(ad, score_details)

        # ============================================================
        # DÉTERMINATION ACTION
        # ============================================================
        # Si quality gate échoué, forcer WAIT_QUALITY_GATE (mais garder le score)
        if not gate_passed:
            action = "WAIT_QUALITY_GATE"
            reason = gate_reason
        # Si higher TF non aligné, forcer WAIT_HIGHER_TF
        elif not higher_tf_ok:
            action = "WAIT_HIGHER_TF"
            reason = higher_tf_reason
        else:
            action, reason = self._determine_action(
                ad, total_score, trend_score, momentum_score, volume_score,
                price_action_score, current_price, institutional_score
            )

        # Calcul zones et targets
        entry_min = current_price * 0.998
        entry_max = current_price * 1.002

        tp1 = current_price * (1 + max(0.01, atr_percent * 0.7))
        tp2 = current_price * (1 + max(0.015, atr_percent * 1.0))
        tp3 = current_price * (1 + max(0.02, atr_percent * 1.5))

        stop_loss = current_price * (1 - max(0.012, atr_percent * 1.2))

        # Taille position recommandée basée sur volatilité
        if atr_percent < 0.01:
            rec_min, rec_max = 5000, 10000  # Faible volatilité
        elif atr_percent < 0.02:
            rec_min, rec_max = 3000, 7000   # Volatilité moyenne
        else:
            rec_min, rec_max = 2000, 5000   # Haute volatilité

        # Préparer la réponse
        return {
            "symbol": symbol,
            "score": round(total_score, 1),
            "score_details": score_details,
            "score_explanation": score_explanation,
            "signals_count": signals_count,
            "avg_confidence": avg_confidence,
            "momentum_score": self.safe_float(ad.get('momentum_score')),
            "volume_ratio": self.safe_float(ad.get('volume_ratio'), 1.0),
            "market_regime": ad.get('market_regime', 'UNKNOWN'),
            "adx": self.safe_float(ad.get('adx_14')),
            "rsi": self.safe_float(ad.get('rsi_14'), 50),
            "mfi": self.safe_float(ad.get('mfi_14')),
            "volume_context": ad.get('volume_context'),
            "volume_quality_score": self.safe_float(ad.get('volume_quality_score')),
            "nearest_support": self.safe_float(ad.get('nearest_support')),
            "nearest_resistance": self.safe_float(ad.get('nearest_resistance')),
            "break_probability": self.safe_float(ad.get('break_probability')),
            "entry_zone": {
                "min": round(entry_min, 8),
                "max": round(entry_max, 8)
            },
            "targets": {
                "tp1": round(tp1, 8),
                "tp2": round(tp2, 8),
                "tp3": round(tp3, 8)
            },
            "stop_loss": round(stop_loss, 8),
            "recommended_size": {
                "min": rec_min,
                "max": rec_max
            },
            "action": action,
            "reason": reason,
            "estimated_hold_time": estimated_hold_time
        }

    def _calculate_trend_score(self, ad: dict) -> float:
        """Calcule le score de tendance (max 25 pts) - Seuils ajustés pour 1m"""
        trend_score = 0

        # ADX - Force de tendance (max 10 pts) - Seuils plus hauts pour 1m
        adx = self.safe_float(ad.get('adx_14'))
        if adx > 45:  # Plus strict (était 40)
            trend_score += 10
        elif adx > 35:  # Plus strict (était 30)
            trend_score += 8
        elif adx > 28:  # Plus strict (était 25)
            trend_score += 5
        elif adx > 22:  # Plus strict (était 20)
            trend_score += 2

        # Directional Movement - Alignement (max 8 pts) - Plus strict
        plus_di = self.safe_float(ad.get('plus_di'))
        minus_di = self.safe_float(ad.get('minus_di'))
        if plus_di > minus_di and plus_di > 28:  # Plus strict (était 25)
            trend_score += 8
        elif plus_di > minus_di and plus_di > 23:  # Plus strict (était 20)
            trend_score += 5
        elif plus_di > minus_di:
            trend_score += 3

        # Regime confidence (max 7 pts) - Plus strict sur confidence
        regime_conf = self.safe_float(ad.get('regime_confidence'))
        regime = ad.get('market_regime', 'UNKNOWN')
        if regime in ['TRENDING_BULL', 'BREAKOUT_BULL'] and regime_conf > 85:  # Plus strict (était 80)
            trend_score += 7
        elif regime in ['TRENDING_BULL', 'BREAKOUT_BULL'] and regime_conf > 70:  # Plus strict (était 60)
            trend_score += 4

        return trend_score

    def _calculate_momentum_score(self, ad: dict) -> float:
        """Calcule le score de momentum (max 35 pts) - OPTIMISÉ avec oscillateurs avancés"""
        momentum_score = 0

        # RSI - Position et alignement (max 8 pts)
        rsi_14 = self.safe_float(ad.get('rsi_14'), 50)
        rsi_21 = self.safe_float(ad.get('rsi_21'), 50)
        if 52 < rsi_14 < 68 and rsi_14 > rsi_21:
            momentum_score += 8
        elif 45 < rsi_14 < 58:
            momentum_score += 4
        elif rsi_14 < 28:  # Oversold = opportunité
            momentum_score += 3

        # Williams %R (max 6 pts) ★ NOUVEAU - Plus sensible que Stochastic
        williams_r = self.safe_float(ad.get('williams_r'))
        if -30 < williams_r < -10:  # Zone achat optimale
            momentum_score += 6
        elif -50 < williams_r < -30:  # Zone neutre positive
            momentum_score += 3
        elif williams_r < -80:  # Deep oversold = rebond probable
            momentum_score += 4

        # CCI (max 6 pts) ★ NOUVEAU - Détection extrêmes
        cci_20 = self.safe_float(ad.get('cci_20'))
        if 50 < cci_20 < 150:  # Momentum positif sans overbought
            momentum_score += 6
        elif 0 < cci_20 < 50:  # Momentum positif léger
            momentum_score += 3
        elif cci_20 < -100:  # Extreme oversold
            momentum_score += 4

        # ROC (max 5 pts) ★ NOUVEAU - Rate of Change instantané
        roc_10 = self.safe_float(ad.get('roc_10'))
        if roc_10 > 0.15:  # +0.15% momentum fort
            momentum_score += 5
        elif roc_10 > 0.05:  # Momentum positif
            momentum_score += 3

        # Stochastic (max 5 pts) - Légèrement réduit car Williams fait mieux
        stoch_k = self.safe_float(ad.get('stoch_k'))
        stoch_d = self.safe_float(ad.get('stoch_d'))
        stoch_signal = ad.get('stoch_signal')
        if stoch_signal == 'BUY' or (stoch_k > stoch_d and stoch_k > 25):
            momentum_score += 5
        elif stoch_k > 55:
            momentum_score += 3

        # MFI - Money Flow (max 5 pts)
        mfi = self.safe_float(ad.get('mfi_14'))
        if 52 < mfi < 78:
            momentum_score += 5
        elif 42 < mfi < 62:
            momentum_score += 3

        return min(momentum_score, 35)  # Cap à 35 pts

    def _calculate_volume_score(self, ad: dict) -> float:
        """Calcule le score de volume (max 32 pts) - OPTIMISÉ avec OBV, Spikes, Intensity"""
        volume_score = 0

        # Volume Quality Score (max 8 pts)
        vol_quality = self.safe_float(ad.get('volume_quality_score'))
        if vol_quality > 55:
            volume_score += min(8, ((vol_quality - 55) / 45) * 8)

        # OBV Oscillator (max 7 pts) ★ NOUVEAU - Divergence price/volume
        obv_osc = self.safe_float(ad.get('obv_oscillator'))
        if obv_osc > 100:  # Fort momentum OBV
            volume_score += 7
        elif obv_osc > 0:  # OBV positif
            volume_score += 4
        elif obv_osc > -200:  # OBV légèrement négatif
            volume_score += 2

        # Volume Spike Multiplier (max 6 pts) ★ NOUVEAU - Détection explosions volume
        spike = self.safe_float(ad.get('volume_spike_multiplier'), 1.0)
        if spike > 2.5:  # Spike massif
            volume_score += 6
        elif spike > 1.8:  # Spike significatif
            volume_score += 4
        elif spike > 1.3:  # Spike léger
            volume_score += 2

        # Trade Intensity (max 5 pts) ★ NOUVEAU - Nb trades vs moyenne
        intensity = self.safe_float(ad.get('trade_intensity'), 1.0)
        if intensity > 1.5:  # Activité intense
            volume_score += 5
        elif intensity > 1.2:  # Activité élevée
            volume_score += 3
        elif intensity > 0.8:  # Activité normale
            volume_score += 1

        # Volume Context (max 6 pts)
        vol_context = ad.get('volume_context')
        if vol_context == 'ACCUMULATION':
            volume_score += 6
        elif vol_context == 'BREAKOUT':
            volume_score += 5
        elif vol_context in ['PUMP_START', 'SUSTAINED_HIGH']:
            volume_score += 4
        elif vol_context == 'DISTRIBUTION':
            volume_score += 0  # Négatif

        return min(volume_score, 32)  # Cap à 32 pts

    def _calculate_price_action_score(self, ad: dict, current_price: float) -> float:
        """Calcule le score de price action (max 20 pts) - OPTIMISÉ avec VWAP et Volume Profile"""
        price_action_score = 0

        # VWAP Bands (max 7 pts) ★ NOUVEAU - Référence institutionnelle
        vwap = self.safe_float(ad.get('vwap_quote_10'))  # Quote VWAP plus précis
        vwap_lower = self.safe_float(ad.get('vwap_lower_band'))
        vwap_upper = self.safe_float(ad.get('vwap_upper_band'))

        if vwap_lower > 0 and current_price > 0:
            dist_to_vwap_lower = abs(current_price - vwap_lower) / vwap_lower
            if dist_to_vwap_lower < 0.003:  # <0.3% de lower band = rebond probable
                price_action_score += 7
            elif dist_to_vwap_lower < 0.006:  # <0.6%
                price_action_score += 4
            elif current_price > vwap > 0:  # Au-dessus VWAP = bullish
                price_action_score += 2

        # Volume Profile POC (max 6 pts) ★ NOUVEAU - Point of Control
        poc = self.safe_float(ad.get('volume_profile_poc'))
        if poc > 0:
            dist_to_poc = abs(current_price - poc) / poc
            if dist_to_poc > 0.015:  # >1.5% du POC = probable retour vers POC
                price_action_score += 6
            elif dist_to_poc > 0.01:  # >1%
                price_action_score += 4

        # Distance au support/résistance (max 4 pts) - Réduit car VWAP fait mieux
        nearest_support = self.safe_float(ad.get('nearest_support'))
        support_strength = ad.get('support_strength')
        if nearest_support > 0:
            dist_to_support = ((current_price - nearest_support) / nearest_support) * 100
            if 0.5 < dist_to_support < 2 and support_strength in ['STRONG', 'MAJOR']:
                price_action_score += 4
            elif 0.5 < dist_to_support < 3:
                price_action_score += 2

        # Bollinger position & squeeze (max 3 pts) - Réduit
        bb_position = self.safe_float(ad.get('bb_position'), 0.5)
        bb_expansion = ad.get('bb_expansion', False)
        bb_squeeze = ad.get('bb_squeeze', False)

        if bb_expansion and 0.3 < bb_position < 0.7:
            price_action_score += 3
        elif bb_squeeze:
            price_action_score += 2

        return min(price_action_score, 20)  # Cap à 20 pts

    def _calculate_consensus_score(self, ad: dict) -> float:
        """Calcule le score de consensus (max 10 pts)"""
        consensus_score = 0

        # Confluence score de l'analyzer (max 5 pts)
        confluence = self.safe_float(ad.get('confluence_score'))
        consensus_score += min(5, (confluence / 100) * 5)

        # Signal strength (max 3 pts)
        signal_strength = ad.get('signal_strength')
        if signal_strength == 'STRONG':
            consensus_score += 3
        elif signal_strength == 'MODERATE':
            consensus_score += 2

        # Pattern confidence (max 2 pts)
        pattern_conf = self.safe_float(ad.get('pattern_confidence'))
        if pattern_conf > 70:
            consensus_score += 2
        elif pattern_conf > 50:
            consensus_score += 1

        return consensus_score

    def _calculate_institutional_flow(self, ad: dict, current_price: float) -> float:
        """Calcule le score de flux institutionnel (max 20 pts) ★ NOUVEAU PILIER
        Détecte l'entrée du smart money via OBV, trade size, intensity
        """
        score = 0

        # OBV vs Price Divergence (max 8 pts)
        obv_osc = self.safe_float(ad.get('obv_oscillator'))
        if obv_osc > 200:  # OBV très fort
            score += 8
        elif obv_osc > 100:  # OBV fort
            score += 6
        elif obv_osc > 0:  # OBV positif
            score += 3
        elif obv_osc < -300:  # OBV très négatif = sell-off
            score += 0

        # Trade Size moyen (max 6 pts) - Détecte institutionnels
        avg_trade = self.safe_float(ad.get('avg_trade_size'))
        if avg_trade > 0.25:  # Très gros trades
            score += 6
        elif avg_trade > 0.15:  # Gros trades
            score += 4
        elif avg_trade > 0.08:  # Trades moyens
            score += 2

        # Trade Intensity (max 6 pts) - Confirmation par nb trades
        intensity = self.safe_float(ad.get('trade_intensity'), 1.0)
        if intensity > 2.0:  # Activité extrême
            score += 6
        elif intensity > 1.5:  # Forte activité
            score += 4
        elif intensity > 1.2:  # Activité élevée
            score += 2

        return min(score, 20)  # Cap à 20 pts

    def _estimate_hold_time(self, ad: dict, atr_percent: float, momentum_score: float) -> str:
        """Estime la durée de hold basée sur volatilité et momentum - Ajusté pour scalping 1m rapide"""
        regime = ad.get('market_regime', 'UNKNOWN')

        # Durées réduites pour scalping 1m (target 1-2% en 15-30min max)
        hold_time_min = 5
        hold_time_max = 30  # Réduit de 45 à 30

        # Ajuster selon volatilité (ATR) - Durées plus courtes
        if atr_percent > 0.025:  # Haute volatilité
            hold_time_min = 3  # Très rapide
            hold_time_max = 15  # Sortir vite
        elif atr_percent > 0.015:  # Volatilité moyenne
            hold_time_min = 7
            hold_time_max = 20
        else:  # Faible volatilité
            hold_time_min = 12
            hold_time_max = 30

        # Ajuster selon momentum
        if momentum_score > 20:  # Fort momentum
            hold_time_max = min(hold_time_max, 20)  # Sortir avant retournement
        elif momentum_score < 15:  # Momentum faible
            hold_time_min = max(hold_time_min, 10)  # Attendre confirmation

        # Ajuster selon régime
        if regime == 'BREAKOUT_BULL':
            hold_time_min = 3  # Sortir TRÈS vite sur breakout
            hold_time_max = 12
        elif regime == 'TRENDING_BULL':
            hold_time_min = 10
            hold_time_max = 25

        return f"{hold_time_min}-{hold_time_max} min"

    def _build_score_explanation(self, ad: dict, score_details: dict) -> dict:
        """Construit l'explication détaillée de chaque pilier"""
        adx = self.safe_float(ad.get('adx_14'))
        plus_di = self.safe_float(ad.get('plus_di'))
        minus_di = self.safe_float(ad.get('minus_di'))
        regime_conf = self.safe_float(ad.get('regime_confidence'))
        regime = ad.get('market_regime', 'UNKNOWN')

        rsi_14 = self.safe_float(ad.get('rsi_14'), 50)
        rsi_21 = self.safe_float(ad.get('rsi_21'), 50)
        stoch_k = self.safe_float(ad.get('stoch_k'))
        stoch_d = self.safe_float(ad.get('stoch_d'))
        stoch_signal = ad.get('stoch_signal')
        mfi = self.safe_float(ad.get('mfi_14'))

        vol_quality = self.safe_float(ad.get('volume_quality_score'))
        vol_context = ad.get('volume_context')
        rel_volume = self.safe_float(ad.get('relative_volume'), 1.0)

        nearest_support = self.safe_float(ad.get('nearest_support'))
        nearest_resistance = self.safe_float(ad.get('nearest_resistance'))
        support_strength = ad.get('support_strength')
        resistance_strength = ad.get('resistance_strength')
        break_prob = self.safe_float(ad.get('break_probability'))
        bb_position = self.safe_float(ad.get('bb_position'), 0.5)
        bb_squeeze = ad.get('bb_squeeze', False)
        bb_expansion = ad.get('bb_expansion', False)

        confluence = self.safe_float(ad.get('confluence_score'))
        signal_strength = ad.get('signal_strength')
        pattern_conf = self.safe_float(ad.get('pattern_confidence'))

        # Calculer les points réels pour les explications
        williams_r = self.safe_float(ad.get('williams_r'))
        cci_20 = self.safe_float(ad.get('cci_20'))
        roc_10 = self.safe_float(ad.get('roc_10'))
        obv_osc = self.safe_float(ad.get('obv_oscillator'))

        # Points Trend (seuils réels: 45/35/28/22)
        adx_pts = 10 if adx>45 else 8 if adx>35 else 5 if adx>28 else 2 if adx>22 else 0
        di_pts = 8 if (plus_di>minus_di and plus_di>28) else 5 if (plus_di>minus_di and plus_di>23) else 3 if plus_di>minus_di else 0
        regime_pts = 7 if (regime in ['TRENDING_BULL','BREAKOUT_BULL'] and regime_conf>85) else 4 if (regime in ['TRENDING_BULL','BREAKOUT_BULL'] and regime_conf>70) else 0

        # Points Momentum (seuils réels: Williams 6pts, CCI 6pts, ROC 5pts, Stoch 5pts)
        rsi_pts = 8 if (52<rsi_14<68 and rsi_14>rsi_21) else 4 if (45<rsi_14<58) else 3 if rsi_14<28 else 0
        stoch_pts = 5 if (stoch_signal=='BUY' or (stoch_k>stoch_d and stoch_k>25)) else 3 if stoch_k>55 else 0
        williams_pts = 6 if (-30<williams_r<-10) else 3 if (-50<williams_r<-30) else 4 if williams_r<-80 else 0
        cci_pts = 6 if (50<cci_20<150) else 3 if (0<cci_20<50) else 4 if cci_20<-100 else 0
        roc_pts = 5 if roc_10>0.15 else 3 if roc_10>0.05 else 0
        mfi_pts = 5 if (52<mfi<78) else 3 if (42<mfi<62) else 0

        # Points Volume (seuils réels: quality≥55→8pts, OBV>100→7pts)
        vol_quality_pts = min(8, ((vol_quality - 55) / 45) * 8) if vol_quality>55 else 0
        obv_pts = 7 if obv_osc>100 else 4 if obv_osc>0 else 2 if obv_osc>-200 else 0

        return {
            "trend": f"ADX:{adx:.1f} (+{adx_pts}pts), DI+:{plus_di:.1f} vs DI-:{minus_di:.1f} (+{di_pts}pts), Régime:{regime} conf:{regime_conf:.0f}% (+{regime_pts}pts)",
            "momentum": f"RSI14:{rsi_14:.1f} vs RSI21:{rsi_21:.1f} (+{rsi_pts}pts), Stoch:{stoch_k:.1f}/{stoch_d:.1f} (+{stoch_pts}pts), Williams:{williams_r:.1f} (+{williams_pts}pts), CCI:{cci_20:.1f} (+{cci_pts}pts), ROC:{roc_10:.3f} (+{roc_pts}pts), MFI:{mfi:.1f} (+{mfi_pts}pts)",
            "volume": f"Quality:{vol_quality:.0f}/100 (+{vol_quality_pts:.1f}pts), OBV osc:{obv_osc:.0f} (+{obv_pts}pts), Context:{vol_context or 'N/A'}, Rel.volume:{rel_volume:.2f}x",
            "price_action": f"Support:{nearest_support:.4f} ({support_strength or 'N/A'}), Resistance:{nearest_resistance:.4f} ({resistance_strength or 'N/A'}), Break prob:{break_prob:.0f}%, BB pos:{bb_position:.2f} squeeze:{bb_squeeze} expansion:{bb_expansion}",
            "consensus": f"Confluence:{confluence:.0f}/100 (+{min(5, (confluence/100)*5):.1f}pts), Signal strength:{signal_strength or 'N/A'}, Pattern conf:{pattern_conf:.0f}%"
        }

    def _determine_action(
        self,
        ad: dict,
        total_score: float,
        trend_score: float,
        momentum_score: float,
        volume_score: float,
        price_action_score: float,
        current_price: float,
        institutional_score: float
    ) -> tuple[str, str]:
        """Détermine l'action recommandée et la raison"""

        # Extraction des données nécessaires
        rsi_14 = self.safe_float(ad.get('rsi_14'), 50)
        mfi = self.safe_float(ad.get('mfi_14'))
        stoch_k = self.safe_float(ad.get('stoch_k'))
        stoch_d = self.safe_float(ad.get('stoch_d'))
        bb_position = self.safe_float(ad.get('bb_position'), 0.5)

        adx = self.safe_float(ad.get('adx_14'))
        plus_di = self.safe_float(ad.get('plus_di'))
        minus_di = self.safe_float(ad.get('minus_di'))

        regime = ad.get('market_regime', 'UNKNOWN')
        regime_conf = self.safe_float(ad.get('regime_confidence'))

        vol_quality = self.safe_float(ad.get('volume_quality_score'))
        vol_context = ad.get('volume_context')
        rel_volume = self.safe_float(ad.get('relative_volume'), 1.0)

        bb_width = self.safe_float(ad.get('bb_width'), 0.0)
        bb_squeeze = ad.get('bb_squeeze', False)

        nearest_resistance = self.safe_float(ad.get('nearest_resistance'))

        # ============================================================
        # 1. VÉRIFIER OVERBOUGHT (priorité absolue)
        # ============================================================
        is_overbought = False
        overbought_reasons = []

        if rsi_14 > 75:
            is_overbought = True
            overbought_reasons.append(f"RSI {rsi_14:.0f}")

        if mfi > 80:
            is_overbought = True
            overbought_reasons.append(f"MFI {mfi:.0f}")

        if stoch_k > 90 and stoch_d > 90:
            is_overbought = True
            overbought_reasons.append(f"Stoch {stoch_k:.0f}")

        if bb_position > 1.0:
            is_overbought = True
            overbought_reasons.append(f"BB overshoot")

        if is_overbought:
            score_breakdown = f"Trend:{trend_score:.0f}/25 | Momentum:{momentum_score:.0f}/25 | Volume:{volume_score:.0f}/20"
            indicators_detail = f"ADX:{adx:.1f}, RSI:{rsi_14:.1f}, MFI:{mfi:.1f}, Stoch:{stoch_k:.1f}/{stoch_d:.1f}, BB pos:{bb_position:.2f}"
            reason = f"🔴 SURACHETÉ ({score_breakdown}) | {indicators_detail} | " + " + ".join(overbought_reasons) + " → Correction imminente probable, VENDRE ou éviter l'achat"
            return "SELL_OVERBOUGHT", reason

        # ============================================================
        # 2. VÉRIFIER OVERSOLD
        # ============================================================
        if rsi_14 < 30 and stoch_k < 20:
            score_breakdown = f"Score:{total_score:.0f}/100 (Trend:{trend_score:.0f} | Momentum:{momentum_score:.0f} | Volume:{volume_score:.0f})"
            indicators_detail = f"RSI:{rsi_14:.1f} (seuil:<30), Stoch:{stoch_k:.1f} (seuil:<20), MFI:{mfi:.1f}"
            reason = f"🔵 SURVENDU ({score_breakdown}) | {indicators_detail} | Zone de rebond potentiel → Attendre signal d'inversion (RSI>35, volume en hausse, bougie verte)"
            return "WAIT_OVERSOLD", reason

        # ============================================================
        # 3. CRITÈRES BUY NOW - OPTIMISÉS POUR SCALPING (score sur 142 pts)
        # ============================================================
        obv_osc = self.safe_float(ad.get('obv_oscillator'))

        buy_criteria = {
            "score_high": total_score >= 95,  # 95/142 = ~67% (équivalent 70/100)
            "trend_strong": trend_score >= 15,
            "volume_confirmed": volume_score >= 18,  # 18/32 = 56%
            "momentum_aligned": momentum_score >= 20,  # 20/35 = 57%
            "institutional_flow": institutional_score >= 12,  # ★ NOUVEAU - Smart money présent
            "regime_bull": regime in ['TRENDING_BULL', 'BREAKOUT_BULL'],
            "adx_trending": adx > 25,
            "not_overbought": rsi_14 < 68,
            "vol_quality": vol_quality > 55,
            "obv_positive": obv_osc > 0,  # ★ NOUVEAU - OBV positif
            "confluence": self.safe_float(ad.get('confluence_score')) > 65
        }

        buy_score = sum(buy_criteria.values())

        if buy_score >= 9:  # 9/11 critères
            score_breakdown = f"Trend:{trend_score:.0f}/25 | Momentum:{momentum_score:.0f}/35 | Volume:{volume_score:.0f}/32 | Instit:{institutional_score:.0f}/20"
            detail_parts = [
                f"ADX:{adx:.1f} (+DI:{plus_di:.1f} vs -DI:{minus_di:.1f})",
                f"RSI:{rsi_14:.1f}",
                f"OBV osc:{obv_osc:.0f}",  # ★ NOUVEAU
                f"Stoch:{stoch_k:.1f}/{stoch_d:.1f}"
            ]
            if vol_context:
                detail_parts.append(f"Vol:{vol_context} (qual:{vol_quality:.0f})")
            if regime:
                detail_parts.append(f"Régime:{regime} ({regime_conf:.0f}%)")

            reason = f"💎 EXCELLENT ({buy_score}/11 critères, {total_score:.0f}/142 pts) | {score_breakdown} | " + " | ".join(detail_parts)
            return "BUY_NOW", reason

        if buy_score >= 7 and total_score >= 80:  # 7/11 critères + score 80/142 (~56%)
            score_breakdown = f"Trend:{trend_score:.0f}/25 | Momentum:{momentum_score:.0f}/35 | Volume:{volume_score:.0f}/32 | Instit:{institutional_score:.0f}/20"

            good_parts = []
            missing_parts = []

            if trend_score >= 15:
                good_parts.append(f"✓Trend (ADX:{adx:.1f}, +DI:{plus_di:.1f})")
            else:
                missing_parts.append(f"✗Trend faible ({trend_score:.0f}/25)")

            if volume_score >= 18:
                good_parts.append(f"✓Volume (qual:{vol_quality:.0f}, {vol_context or 'N/A'})")
            else:
                missing_parts.append(f"✗Volume ({volume_score:.0f}/32)")

            if momentum_score >= 20:
                good_parts.append(f"✓Momentum (RSI:{rsi_14:.1f}, Williams:{self.safe_float(ad.get('williams_r')):.0f})")
            else:
                missing_parts.append(f"✗Momentum ({momentum_score:.0f}/35)")

            if institutional_score >= 12:
                good_parts.append(f"✓Institutionnel (OBV:{obv_osc:.0f})")
            else:
                missing_parts.append(f"✗Flux institutionnel faible ({institutional_score:.0f}/20)")

            all_details = good_parts + missing_parts
            reason = f"✅ BON ({buy_score}/11 critères, {total_score:.0f}/142 pts) | {score_breakdown} | " + " | ".join(all_details)
            return "BUY_NOW", reason

        # ============================================================
        # 4. CRITÈRES WAIT - AJUSTÉ (score sur 142 pts)
        # ============================================================
        if total_score >= 60:  # 60/142 = ~42%
            score_breakdown = f"Trend:{trend_score:.0f}/25 | Momentum:{momentum_score:.0f}/35 | Volume:{volume_score:.0f}/32 | Instit:{institutional_score:.0f}/20"
            indicators_detail = f"ADX:{adx:.1f}, RSI:{rsi_14:.1f}, OBV:{obv_osc:.0f}, Williams:{self.safe_float(ad.get('williams_r')):.0f}"

            missing_for_buy = []
            if total_score < 95:
                missing_for_buy.append(f"Score {total_score:.0f}/95 requis")
            if trend_score < 15:
                missing_for_buy.append(f"Trend faible ({trend_score:.0f}/15)")
            if volume_score < 18:
                missing_for_buy.append(f"Volume faible ({volume_score:.0f}/18)")
            if momentum_score < 20:
                missing_for_buy.append(f"Momentum faible ({momentum_score:.0f}/20)")
            if institutional_score < 12:
                missing_for_buy.append(f"Flux institutionnel faible ({institutional_score:.0f}/12)")

            if bb_squeeze:
                reason = f"🟡 {score_breakdown} | {indicators_detail} | BB squeeze (BB width:{bb_width:.4f}) → Volatilité imminente | Manque: " + ", ".join(missing_for_buy if missing_for_buy else ["Attendre direction"])
                return "WAIT_BREAKOUT", reason
            elif vol_context == 'DISTRIBUTION':
                reason = f"🟡 {score_breakdown} | {indicators_detail} | Distribution (vente institutionnelle, vol qual:{vol_quality:.0f}) → Attendre accumulation | Manque: " + ", ".join(missing_for_buy)
                return "WAIT", reason
            elif nearest_resistance > 0:
                dist_to_resistance = ((nearest_resistance - current_price) / current_price) * 100
                if dist_to_resistance < 1:
                    reason = f"🟡 {score_breakdown} | {indicators_detail} | Résistance {nearest_resistance:.4f} à {dist_to_resistance:.1f}% → Attendre cassure | Manque: " + ", ".join(missing_for_buy)
                    return "WAIT", reason

            reason = f"🟡 {score_breakdown} | {indicators_detail} | Manque: " + ", ".join(missing_for_buy if missing_for_buy else ["Confluence insuffisante"])
            return "WAIT", reason

        # ============================================================
        # 5. AVOID
        # ============================================================
        score_breakdown = f"Trend:{trend_score:.0f}/25 | Momentum:{momentum_score:.0f}/35 | Volume:{volume_score:.0f}/32 | Instit:{institutional_score:.0f}/20"

        weak_points = []
        if trend_score < 8:
            weak_points.append(f"Trend faible ({trend_score:.0f}/25, ADX:{adx:.0f})")
        if momentum_score < 12:
            weak_points.append(f"Momentum faible ({momentum_score:.0f}/35, RSI:{rsi_14:.0f}, Williams:{self.safe_float(ad.get('williams_r')):.0f})")
        if volume_score < 10:
            weak_points.append(f"Volume insuffisant ({volume_score:.0f}/32, qual:{vol_quality:.0f}, OBV osc:{obv_osc:.0f})")
        if institutional_score < 6:
            weak_points.append(f"Pas de flux institutionnel ({institutional_score:.0f}/20)")
        if price_action_score < 6:
            weak_points.append(f"Price action faible ({price_action_score:.0f}/20)")
        if regime in ['TRENDING_BEAR', 'BREAKOUT_BEAR']:
            weak_points.append(f"Régime baissier ({regime})")

        reason = f"⚫ AVOID ({total_score:.0f}/142 pts) | {score_breakdown} | " + " | ".join(weak_points if weak_points else ["Setup défavorable"])
        return "AVOID", reason
