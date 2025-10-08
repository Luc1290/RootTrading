"""
Calculateur d'opportunités HYBRIDE - Version Scalping SPOT v3.0
Mix optimal de opportunity_calculator_simple + scalp_entry_simple
4 conditions critiques + 2 modes (conservative/momentum)
Optimisé pour capturer setups 1m/5m avec gestion intelligente des résistances
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OpportunityCalculator:
    """
    Calculateur optimisé pour scalping SPOT 1m/5m en BULL MARKET

    Architecture:
    - Base: 4 conditions critiques TOUTES requises (pas de scoring arbitraire)
    - Innovation: Résistance ADAPTATIVE (momentum fort → casse les résistances)

    ✅ 1. VOLUME BREAKOUT - Volume explosif + OBV positif
    ✅ 2. MOMENTUM ALIGNMENT - RSI/Williams/MACD/CCI alignés (3/4 minimum)
    ✅ 3. TREND QUALITY - ADX >20, +DI > -DI, régime bull
    ✅ 4. SMART RESISTANCE - Ignore résistance si momentum/volume assez fort

    Philosophie: En bull market ATH, les résistances sont faites pour être cassées.
    Si ADX >25 ou volume spike >2x → on achète la force, pas la faiblesse.

    Actions: BUY_NOW, WAIT (avec raison précise)
    """

    def __init__(self):
        """
        Calculateur d'opportunités avec logique momentum adaptative
        Les résistances sont ignorées si le momentum est assez fort pour les casser
        """
        pass

    @staticmethod
    def safe_float(value, default=0.0):
        """Convertir en float avec fallback"""
        return float(value) if value is not None else default

    def calculate_opportunity(
        self,
        symbol: str,
        current_price: float,
        analyzer_data: Optional[dict],
        signals_data: Optional[dict] = None,
        higher_tf: Optional[dict] = None
    ) -> dict:
        """
        Évalue si c'est LE moment d'acheter (scalping 1m/5m)

        Args:
            symbol: Symbole à analyser
            current_price: Prix actuel
            analyzer_data: Données techniques (1m)
            signals_data: (optionnel) Données signaux
            higher_tf: (optionnel) Données 5m pour validation contexte

        Returns:
            dict avec action BUY_NOW ou WAIT et raison détaillée
        """
        if not analyzer_data:
            return {
                "symbol": symbol,
                "action": "WAIT",
                "reason": "❌ Pas de données techniques",
                "conditions": {
                    "volume_breakout": False,
                    "momentum_alignment": False,
                    "trend_quality": False,
                    "no_resistance": False
                },
                "current_price": current_price,
                "targets": {
                    "tp1": current_price * 1.01,
                    "tp1_percent": 1.0,
                    "tp2": current_price * 1.015,
                    "tp2_percent": 1.5
                },
                "stop_loss": current_price * 0.988,
                "stop_loss_percent": 1.2,
                "estimated_hold_time": "N/A",
                "market_regime": "UNKNOWN",
                "volume_context": None,
                "raw_data": {
                    "rsi": None,
                    "adx": None,
                    "mfi": None,
                    "obv_osc": None,
                    "rel_volume": 0.0,
                    "volume_quality_score": None,
                    "nearest_resistance": None
                }
            }

        ad = analyzer_data

        # Calculer ATR en pourcentage avec fallback
        atr_value = self.safe_float(ad.get('atr_14'))
        natr = self.safe_float(ad.get('natr'))

        if atr_value > 0 and current_price > 0:
            atr_percent = atr_value / current_price
        elif natr > 0:
            atr_percent = natr / 100.0
        else:
            return {
                "symbol": symbol,
                "action": "WAIT",
                "reason": "❌ ATR/NATR indisponibles → volatilité non mesurable",
                "conditions": {
                    "volume_breakout": False,
                    "momentum_alignment": False,
                    "trend_quality": False,
                    "no_resistance": False
                },
                "current_price": current_price,
                "targets": {
                    "tp1": current_price * 1.01,
                    "tp1_percent": 1.0,
                    "tp2": current_price * 1.015,
                    "tp2_percent": 1.5
                },
                "stop_loss": current_price * 0.988,
                "stop_loss_percent": 1.2,
                "estimated_hold_time": "N/A",
                "market_regime": ad.get('market_regime', 'UNKNOWN'),
                "volume_context": ad.get('volume_context'),
                "raw_data": {
                    "rsi": self.safe_float(ad.get('rsi_14')),
                    "adx": self.safe_float(ad.get('adx_14')),
                    "mfi": self.safe_float(ad.get('mfi_14')),
                    "obv_osc": self.safe_float(ad.get('obv_oscillator')),
                    "rel_volume": self.safe_float(ad.get('relative_volume'), 0.0),
                    "volume_quality_score": self.safe_float(ad.get('volume_quality_score')),
                    "nearest_resistance": self.safe_float(ad.get('nearest_resistance'))
                }
            }

        # ===========================================================
        # QUALITY GATES PRÉ-VALIDATION (blocage immédiat)
        # ===========================================================
        gate_passed, gate_reason = self._check_quality_gates(ad, current_price, atr_percent)
        if not gate_passed:
            return {
                "symbol": symbol,
                "action": "WAIT_QUALITY_GATE",
                "reason": gate_reason,
                "conditions": {
                    "volume_breakout": False,
                    "momentum_alignment": False,
                    "trend_quality": False,
                    "no_resistance": False
                },
                "current_price": current_price,
                "targets": {
                    "tp1": round(current_price * 1.01, 8),
                    "tp1_percent": 1.0,
                    "tp2": round(current_price * 1.015, 8),
                    "tp2_percent": 1.5
                },
                "stop_loss": round(current_price * 0.988, 8),
                "stop_loss_percent": 1.2,
                "estimated_hold_time": "N/A",
                "market_regime": ad.get('market_regime', 'UNKNOWN'),
                "volume_context": ad.get('volume_context'),
                "raw_data": {
                    "rsi": self.safe_float(ad.get('rsi_14')),
                    "adx": self.safe_float(ad.get('adx_14')),
                    "mfi": self.safe_float(ad.get('mfi_14')),
                    "obv_osc": self.safe_float(ad.get('obv_oscillator')),
                    "rel_volume": self.safe_float(ad.get('relative_volume'), 0.0),
                    "volume_quality_score": self.safe_float(ad.get('volume_quality_score')),
                    "nearest_resistance": self.safe_float(ad.get('nearest_resistance'))
                }
            }

        # ===========================================================
        # HIGHER TIMEFRAME VALIDATION (5m)
        # ===========================================================
        htf_ok, htf_reason = self._check_higher_timeframe(higher_tf)
        if not htf_ok:
            return {
                "symbol": symbol,
                "action": "WAIT_HIGHER_TF",
                "reason": htf_reason,
                "conditions": {
                    "volume_breakout": False,
                    "momentum_alignment": False,
                    "trend_quality": False,
                    "no_resistance": False
                },
                "current_price": current_price,
                "targets": {
                    "tp1": round(current_price * 1.01, 8),
                    "tp1_percent": 1.0,
                    "tp2": round(current_price * 1.015, 8),
                    "tp2_percent": 1.5
                },
                "stop_loss": round(current_price * 0.988, 8),
                "stop_loss_percent": 1.2,
                "estimated_hold_time": "N/A",
                "market_regime": ad.get('market_regime', 'UNKNOWN'),
                "volume_context": ad.get('volume_context'),
                "raw_data": {
                    "rsi": self.safe_float(ad.get('rsi_14')),
                    "adx": self.safe_float(ad.get('adx_14')),
                    "mfi": self.safe_float(ad.get('mfi_14')),
                    "obv_osc": self.safe_float(ad.get('obv_oscillator')),
                    "rel_volume": self.safe_float(ad.get('relative_volume'), 0.0),
                    "volume_quality_score": self.safe_float(ad.get('volume_quality_score')),
                    "nearest_resistance": self.safe_float(ad.get('nearest_resistance'))
                }
            }

        # ===========================================================
        # CONDITION 1: VOLUME BREAKOUT ⚡
        # ===========================================================
        volume_ok, volume_reason = self._check_volume_breakout(ad)

        # ===========================================================
        # CONDITION 2: MOMENTUM ALIGNMENT 📈
        # ===========================================================
        momentum_ok, momentum_reason = self._check_momentum_alignment(ad)

        # ===========================================================
        # CONDITION 3: TREND QUALITY 📊
        # ===========================================================
        trend_ok, trend_reason = self._check_trend_quality(ad)

        # ===========================================================
        # CONDITION 4: NO RESISTANCE BLOCK 🚀
        # ===========================================================
        resistance_ok, resistance_reason = self._check_no_resistance_block(
            ad, current_price, atr_percent
        )

        # ===========================================================
        # PATH ALTERNATIF: PRE-BREAKOUT SNIPER 🎯
        # ===========================================================
        # Détecte compression + accumulation AVANT le spike
        prebreakout_ok, prebreakout_reason = self._check_prebreakout_setup(ad, atr_percent)

        # ===========================================================
        # DÉCISION FINALE
        # ===========================================================
        conditions = {
            "volume_breakout": volume_ok,
            "momentum_alignment": momentum_ok,
            "trend_quality": trend_ok,
            "no_resistance": resistance_ok
        }

        all_conditions_met = all(conditions.values())

        # Path A: Pump Catcher (4/4 conditions) OU Path B: Pre-Breakout Setup
        if all_conditions_met or prebreakout_ok:
            # 🎯 SETUP PARFAIT - ACHETER MAINTENANT
            action = "BUY_NOW"

            # Zone d'entrée optimale (pullback hint)
            entry_hint = self._pullback_entry_hint(ad, current_price)

            if all_conditions_met:
                # Path A: Pump Catcher confirmé
                reason = (
                    f"🎯 PUMP CATCHER - Les 4 conditions réunies:\n"
                    f"✅ Volume: {volume_reason}\n"
                    f"✅ Momentum: {momentum_reason}\n"
                    f"✅ Trend: {trend_reason}\n"
                    f"✅ Résistance: {resistance_reason}\n"
                    f"📍 Entrée: {entry_hint}"
                )
            else:
                # Path B: Pre-Breakout Setup
                reason = (
                    f"🎯 PRE-BREAKOUT SNIPER - Setup anticipation:\n"
                    f"{prebreakout_reason}\n"
                    f"📍 Entrée: {entry_hint}"
                )
                # Marquer prebreakout comme condition spéciale
                conditions["prebreakout_sniper"] = True
        else:
            # ⏸️ ATTENDRE - Indiquer ce qui manque
            action = "WAIT"
            missing = []
            if not volume_ok:
                missing.append(f"❌ Volume: {volume_reason}")
            if not momentum_ok:
                missing.append(f"❌ Momentum: {momentum_reason}")
            if not trend_ok:
                missing.append(f"❌ Trend: {trend_reason}")
            if not resistance_ok:
                missing.append(f"❌ Résistance: {resistance_reason}")

            reason = f"⏸️ ATTENDRE - Conditions manquantes:\n" + "\n".join(missing)

        # Calculer targets basiques
        tp1 = current_price * (1 + max(0.01, atr_percent * 0.8))
        tp2 = current_price * (1 + max(0.015, atr_percent * 1.2))

        # SL intelligent: support si dispo, sinon ATR
        nearest_support = self.safe_float(ad.get('nearest_support'))
        if nearest_support > 0 and current_price > nearest_support:
            sl_dist = max(0.007, (current_price - nearest_support) / current_price)
        else:
            sl_dist = max(0.007, atr_percent * 0.7)

        stop_loss = current_price * (1 - sl_dist)

        # Estimation durée hold
        hold_time = self._estimate_hold_time(ad, atr_percent)

        return {
            "symbol": symbol,
            "action": action,
            "reason": reason,
            "conditions": conditions,
            "current_price": current_price,
            "targets": {
                "tp1": round(tp1, 8),
                "tp1_percent": round(((tp1 - current_price) / current_price) * 100, 2),
                "tp2": round(tp2, 8),
                "tp2_percent": round(((tp2 - current_price) / current_price) * 100, 2)
            },
            "stop_loss": round(stop_loss, 8),
            "stop_loss_percent": round(sl_dist * 100, 2),
            "estimated_hold_time": hold_time,
            "market_regime": ad.get('market_regime', 'UNKNOWN'),
            "volume_context": ad.get('volume_context'),
            # Données clés pour debugging
            "raw_data": {
                "rsi": self.safe_float(ad.get('rsi_14'), 50),
                "adx": self.safe_float(ad.get('adx_14')),
                "mfi": self.safe_float(ad.get('mfi_14')),
                "obv_osc": self.safe_float(ad.get('obv_oscillator')),
                "rel_volume": self.safe_float(ad.get('relative_volume'), 1.0),
                "volume_quality_score": self.safe_float(ad.get('volume_quality_score')),
                "nearest_resistance": self.safe_float(ad.get('nearest_resistance'))
            }
        }

    def _check_quality_gates(self, ad: dict, current_price: float, atr_percent: float) -> tuple[bool, str]:
        """
        QUALITY GATES pré-validation (blocage immédiat sans calcul)

        Gates critiques pour SPOT:
        - Gate Volume: refuse DISTRIBUTION forte ou volume mort
        - Gate Surachat: refuse si overbought extrême (RSI >75 ou MFI >80)
        - Gate R/R: refuse si R/R <1.4 ou résistance < target (impossible TP1)

        Returns:
            (gate_passed, reason)
        """
        # ============================================================
        # GATE 1: VOLUME ABSOLU
        # ============================================================
        vol_context = ad.get('volume_context')
        rel_volume = self.safe_float(ad.get('relative_volume'), 1.0)
        obv_osc = self.safe_float(ad.get('obv_oscillator'))
        vol_pattern = ad.get('volume_pattern')

        # Refuse contextes baissiers (valeurs réelles de VolumeContextType en UPPERCASE)
        bearish_contexts = ['REVERSAL_PATTERN', 'DEEP_OVERSOLD']
        if vol_context in bearish_contexts and rel_volume > 1.2:
            return False, f"❌ Gate Volume: {vol_context} {rel_volume:.1f}x → Smart money sort"

        # Refuse volume mort
        if rel_volume < 0.5:
            return False, f"❌ Gate Volume: {rel_volume:.2f}x < 0.5x → Marché mort"

        # Refuse OBV négatif + volume déclinant (valeurs réelles de VolumePatternType en UPPERCASE)
        if obv_osc < -200 and vol_pattern == 'DECLINING':
            return False, f"❌ Gate Volume: OBV {obv_osc:.0f} + DECLINING → Momentum mort"

        # ============================================================
        # GATE 2: SURACHAT EXTRÊME
        # ============================================================
        rsi = self.safe_float(ad.get('rsi_14'), 50)
        mfi = self.safe_float(ad.get('mfi_14'), 50)
        k = self.safe_float(ad.get('stoch_k'))
        d = self.safe_float(ad.get('stoch_d'))
        bb_pos = self.safe_float(ad.get('bb_position'), 0.5)

        # Seuils tolérants pour bull market (on achète la force)
        # Si ADX >30 + volume fort, on tolère plus de surachat
        adx = self.safe_float(ad.get('adx_14'))
        rel_volume = self.safe_float(ad.get('relative_volume'), 1.0)
        plus_di = self.safe_float(ad.get('plus_di'))
        minus_di = self.safe_float(ad.get('minus_di'))
        regime = ad.get('market_regime', 'UNKNOWN')

        strong_market = adx > 30 and rel_volume > 1.5

        # Exception bull fort: TRENDING_BULL avec dominance directionnelle forte (plus_di > 2x minus_di)
        # → On tolère MFI élevé car momentum buying peut persister
        bull_dominance = (
            regime == 'TRENDING_BULL' and
            plus_di > 30 and
            minus_di > 0 and
            (plus_di / minus_di) > 2.0
        )

        # PUMP CATCHING MODE: Si volume spike + momentum directionnel fort
        # → Ignorer Stochastic et BB Position (ils restent saturés pendant les pumps)
        # → Prioriser le momentum directionnel (ADX + Plus DI dominance)
        pump_catching = (
            rel_volume > 1.5 and
            adx > 20 and
            plus_di > 25 and
            minus_di > 0 and
            (plus_di / minus_di) > 1.8 and
            regime in ['TRENDING_BULL', 'BREAKOUT_BULL']
        )

        if pump_catching:
            # Mode PUMP CATCHING: ignore Stochastic/BB, garde RSI/MFI pour protection extrême
            rsi_lim, mfi_lim, stoch_lim, bb_lim = 82, 88, 999, 1.30  # Stoch/BB pratiquement désactivés
        elif strong_market:
            # Marché fort: tolère plus de surachat
            rsi_lim, mfi_lim, stoch_lim, bb_lim = 82, 85, 95, 1.08
        elif bull_dominance:
            # Bull dominant: tolère MFI élevé mais garde RSI strict
            rsi_lim, mfi_lim, stoch_lim, bb_lim = 75, 88, 92, 1.05
        else:
            # Marché normal: seuils standards
            rsi_lim, mfi_lim, stoch_lim, bb_lim = 75, 80, 90, 1.02

        if rsi > rsi_lim or mfi > mfi_lim or (k > stoch_lim and d > stoch_lim) or bb_pos > bb_lim:
            return False, f"❌ Gate Surachat: RSI {rsi:.0f}/{rsi_lim}, MFI {mfi:.0f}/{mfi_lim}, Stoch {k:.0f}/{d:.0f}, BB {bb_pos:.2f}"

        # ============================================================
        # GATE 3: R/R MINIMAL
        # ============================================================
        nearest_support = self.safe_float(ad.get('nearest_support'))
        nearest_resistance = self.safe_float(ad.get('nearest_resistance'))

        # Target minimum
        tp_dist = max(0.01, atr_percent * 0.8)

        # SL intelligent
        if nearest_support > 0 and current_price > nearest_support:
            sl_dist = max(0.007, (current_price - nearest_support) / current_price)
            sl_basis = "support"
        else:
            sl_dist = max(0.007, atr_percent * 0.7)
            sl_basis = "ATR"

        rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0

        # R/R minimum strict (mais pas de gate résistance - géré dans condition 4 avec bypass intelligents)
        if rr_ratio < 1.4:
            return False, f"❌ Gate R/R: {rr_ratio:.2f} < 1.40 (TP {tp_dist*100:.2f}% vs SL {sl_dist*100:.2f}%, base {sl_basis})"

        # Gate résistance SUPPRIMÉ - check fait dans _check_smart_resistance avec bypass momentum/volume
        # Ceci évite de bloquer les setups avec strong momentum qui vont casser la résistance

        return True, f"✅ Quality Gates OK | R/R {rr_ratio:.2f} | Vol {rel_volume:.1f}x"

    def _check_higher_timeframe(self, higher_tf: Optional[dict]) -> tuple[bool, str]:
        """
        Vérifie alignement timeframe supérieur (5m)
        Pour scalping 1m, le 5m doit confirmer la direction

        Returns:
            (is_aligned, reason)
        """
        # Pas de données 5m = mode dégradé OK
        if not higher_tf:
            return True, "✅ 5m: données indisponibles (mode dégradé)"

        rsi_5m = self.safe_float(higher_tf.get('rsi_14'), 50)
        macd_trend_5m = higher_tf.get('macd_trend')
        regime_5m = higher_tf.get('market_regime', 'UNKNOWN')
        plus_di_5m = self.safe_float(higher_tf.get('plus_di'))
        minus_di_5m = self.safe_float(higher_tf.get('minus_di'))

        # Au moins 1 critère bullish + pas de régime baissier fort
        is_bullish = (
            macd_trend_5m == 'BULLISH' or
            rsi_5m > 50 or
            plus_di_5m > minus_di_5m
        )

        is_bear_regime = regime_5m in ['TRENDING_BEAR', 'BREAKOUT_BEAR']

        # Rejet si contre-tendance forte
        if is_bear_regime and not is_bullish:
            return False, f"🟡 5m contre-tendance: {regime_5m}, RSI {rsi_5m:.0f}, MACD {macd_trend_5m or 'N/A'}"

        # Rejet si TOUS baissiers
        if not is_bullish:
            return False, f"🟡 5m baissier: RSI {rsi_5m:.0f}<50, MACD {macd_trend_5m or 'BEARISH'}"

        return True, f"✅ 5m aligné: {regime_5m}, RSI {rsi_5m:.0f}"

    def _check_volume_breakout(self, ad: dict) -> tuple[bool, str]:
        """
        CONDITION 1: Volume Breakout

        Critères:
        - Volume relatif >1.5x OU spike >2x
        - Context: ACCUMULATION, BREAKOUT, PUMP_START, ou SUSTAINED_HIGH
        - OBV oscillator >0 (buying pressure)

        Returns:
            (is_valid, reason)
        """
        rel_volume = self.safe_float(ad.get('relative_volume'), 1.0)
        vol_spike = self.safe_float(ad.get('volume_spike_multiplier'), 1.0)
        vol_context = ad.get('volume_context')
        obv_osc = self.safe_float(ad.get('obv_oscillator'))
        vol_quality = self.safe_float(ad.get('volume_quality_score'), 50.0)

        # Contexts bullish (valeurs réelles de VolumeContextType enum en UPPERCASE depuis DB)
        good_contexts = [
            'BREAKOUT',              # Breakout confirmé
            'PUMP_START',            # Début pump
            'OVERSOLD_BOUNCE',       # Rebond oversold
            'CONSOLIDATION_BREAK',   # Sortie consolidation
            'TREND_CONTINUATION'     # Continuation tendance
        ]

        # Vérifier volume élevé
        volume_high = rel_volume > 1.5 or vol_spike > 2.0

        # Vérifier context favorable
        context_ok = vol_context in good_contexts

        # Vérifier OBV positif
        obv_positive = obv_osc > 0

        # Vérifier qualité minimum
        quality_ok = vol_quality > 45

        if volume_high and context_ok and obv_positive and quality_ok:
            return True, f"Vol {rel_volume:.1f}x, Spike {vol_spike:.1f}x, {vol_context}, OBV +{obv_osc:.0f}, Qual {vol_quality:.0f}"

        # Raison du rejet
        issues = []
        if not volume_high:
            issues.append(f"Vol faible ({rel_volume:.1f}x, spike {vol_spike:.1f}x)")
        if not context_ok:
            issues.append(f"Context {vol_context or 'UNKNOWN'}")
        if not obv_positive:
            issues.append(f"OBV {obv_osc:.0f}")
        if not quality_ok:
            issues.append(f"Qualité {vol_quality:.0f}/100")

        return False, " | ".join(issues)

    def _check_momentum_alignment(self, ad: dict) -> tuple[bool, str]:
        """
        CONDITION 2: Momentum Alignment

        Critères (AU MOINS 3/4):
        - RSI entre 45-75 (zone bullish sans overbought)
        - Williams %R >-50 (momentum positif)
        - MACD trend = BULLISH
        - CCI >0 (momentum positif)

        Returns:
            (is_valid, reason)
        """
        rsi = self.safe_float(ad.get('rsi_14'), 50)
        williams = self.safe_float(ad.get('williams_r'))
        macd_trend = ad.get('macd_trend')
        cci = self.safe_float(ad.get('cci_20'))
        macd_hist = self.safe_float(ad.get('macd_histogram'))

        # Vérifier chaque indicateur
        checks = {
            'rsi': 45 < rsi < 75,
            'williams': williams > -50,
            'macd': macd_trend == 'BULLISH',
            'cci': cci > 0
        }

        valid_count = sum(checks.values())

        if valid_count >= 3:
            # Au moins 3/4 conditions remplies
            indicators = []
            if checks['rsi']:
                indicators.append(f"RSI {rsi:.0f}")
            if checks['williams']:
                indicators.append(f"Williams {williams:.0f}")
            if checks['macd']:
                indicators.append("MACD bullish")
            if checks['cci']:
                indicators.append(f"CCI +{cci:.0f}")

            return True, " | ".join(indicators) + f" ({valid_count}/4 ✓)"

        # Pas assez d'indicateurs alignés
        issues = []
        if not checks['rsi']:
            if rsi > 75:
                issues.append(f"RSI overbought ({rsi:.0f})")
            else:
                issues.append(f"RSI faible ({rsi:.0f})")
        if not checks['williams']:
            issues.append(f"Williams {williams:.0f}")
        if not checks['macd']:
            issues.append(f"MACD {macd_trend or 'UNKNOWN'}")
        if not checks['cci']:
            issues.append(f"CCI {cci:.0f}")

        return False, " | ".join(issues) + f" ({valid_count}/4 seulement)"

    def _check_trend_quality(self, ad: dict) -> tuple[bool, str]:
        """
        CONDITION 3: Trend Quality

        Critères:
        - ADX >20 (tendance confirmée)
        - +DI > -DI (direction haussière)
        - Régime: TRENDING_BULL ou BREAKOUT_BULL

        Returns:
            (is_valid, reason)
        """
        adx = self.safe_float(ad.get('adx_14'))
        plus_di = self.safe_float(ad.get('plus_di'))
        minus_di = self.safe_float(ad.get('minus_di'))
        regime = ad.get('market_regime', 'UNKNOWN')

        if adx < 20:
            return False, f"ADX {adx:.1f} < 20"

        if plus_di <= minus_di:
            return False, f"-DI {minus_di:.1f} >= +DI {plus_di:.1f}"

        if regime in ('TRENDING_BEAR', 'BREAKOUT_BEAR'):
            return False, f"Régime {regime}"

        return True, f"ADX {adx:.1f}, +DI {plus_di:.1f} > -DI {minus_di:.1f}, {regime}"

    def _check_no_resistance_block(
        self, ad: dict, current_price: float, atr_percent: float
    ) -> tuple[bool, str]:
        """
        CONDITION 4: Smart Resistance (LOGIQUE ADAPTATIVE ★)

        PHILOSOPHIE BULL MARKET:
        Les résistances sont faites pour être cassées en bull market fort.
        On refuse SEULEMENT si résistance proche + momentum FAIBLE.

        LOGIQUE:
        1. Si ADX >25 + régime BULL → IGNORE résistance (on achète la force)
        2. Si volume spike >2x → IGNORE résistance (breakout confirmé)
        3. Si résistance loin (>1.5%) → OK
        4. Sinon → Vérifie critères breakout stricts

        Returns:
            (is_valid, reason)
        """
        nearest_resistance = self.safe_float(ad.get('nearest_resistance'))
        break_prob = self.safe_float(ad.get('break_probability'))

        # Pas de résistance = OK
        if nearest_resistance <= 0 or current_price <= 0:
            return True, "Aucune résistance détectée"

        dist_pct = ((nearest_resistance - current_price) / current_price) * 100

        # Données momentum
        adx = self.safe_float(ad.get('adx_14'))
        plus_di = self.safe_float(ad.get('plus_di'))
        minus_di = self.safe_float(ad.get('minus_di'))
        regime = ad.get('market_regime', 'UNKNOWN')
        vol_spike = self.safe_float(ad.get('volume_spike_multiplier'), 1.0)
        bb_expansion = ad.get('bb_expansion')
        rel_volume = self.safe_float(ad.get('relative_volume'), 1.0)
        trend_alignment = self.safe_float(ad.get('trend_alignment'), 0)
        macd_hist = self.safe_float(ad.get('macd_histogram'))

        # ===========================================================
        # PRIORITÉ 1: Momentum FORT → Résistance ignorée
        # ===========================================================
        strong_momentum = (
            adx > 25 and
            plus_di > minus_di and
            regime in ['TRENDING_BULL', 'BREAKOUT_BULL']
        )

        if strong_momentum:
            return True, f"Résistance {dist_pct:.1f}% mais MOMENTUM FORT (ADX {adx:.0f}, +DI {plus_di:.0f}, {regime}) → cassure probable"

        # ===========================================================
        # PRIORITÉ 2: Volume BREAKOUT → Résistance ignorée
        # ===========================================================
        breakout_volume = vol_spike > 2.0

        if breakout_volume:
            return True, f"Résistance {dist_pct:.1f}% mais BREAKOUT VOLUME (spike {vol_spike:.1f}x) → cassure probable"

        # ===========================================================
        # PRIORITÉ 3: Résistance loin → OK
        # ===========================================================
        if dist_pct > 1.5:
            return True, f"Résistance à {dist_pct:.1f}% (assez loin)"

        # ===========================================================
        # PRIORITÉ 4: Résistance proche + momentum FAIBLE → Critères stricts
        # ===========================================================
        # Si on arrive ici: résistance <1.5% + ADX <25 + volume spike <2x
        # On vérifie si les conditions de breakout sont quand même remplies

        issues = []

        if regime not in ('TRENDING_BULL', 'BREAKOUT_BULL'):
            issues.append(f"Régime {regime}")

        if not bb_expansion:
            issues.append("Pas expansion BB")

        if rel_volume < 1.3:
            issues.append(f"Volume {rel_volume:.1f}x < 1.3x")

        if adx < 20:
            issues.append(f"ADX {adx:.0f} < 20")

        if trend_alignment > 0 and trend_alignment < 60:
            issues.append(f"EMA {trend_alignment:.0f}% < 60%")

        if macd_hist <= 0:
            issues.append("MACD hist ≤0")

        # Si trop d'éléments manquent, refuse
        if len(issues) >= 3:
            return False, f"Résistance {dist_pct:.1f}% + momentum faible: " + " | ".join(issues)

        # Sinon, vérifie acceptance au-dessus résistance
        if current_price <= nearest_resistance * 1.001:
            return False, f"Résistance {dist_pct:.1f}%, besoin acceptance >0.1% au-dessus ({nearest_resistance * 1.001:.8f})"

        # Si on passe tous les critères, OK
        return True, f"Résistance {dist_pct:.1f}% mais setup breakout validé (vol {rel_volume:.1f}x, ADX {adx:.0f}, BB expansion)"

    def _pullback_entry_hint(self, ad: dict, price: float) -> str:
        """
        Donne une micro-zone d'achat sur repli (9-EMA/VWAP)
        Utilisé en mode momentum pour optimiser entrée
        """
        ema_7 = self.safe_float(ad.get('ema_7'))
        vwap = self.safe_float(ad.get('vwap_quote_10')) or self.safe_float(ad.get('vwap_10'))

        hint = []
        if ema_7 > 0 and price > ema_7:
            hint.append(f"retour 7-EMA≈{ema_7:.6f}")
        if vwap > 0 and price > vwap:
            hint.append(f"ou VWAP≈{vwap:.6f}")

        if not hint:
            return "entrer par tranches (DCA court)"

        return "pullback léger sur " + " / ".join(hint)

    def _estimate_hold_time(self, ad: dict, atr_percent: float) -> str:
        """
        Estime durée de hold basée sur volatilité et régime
        Ajusté pour scalping 1m rapide
        """
        regime = ad.get('market_regime', 'UNKNOWN')
        adx = self.safe_float(ad.get('adx_14'))

        # Durées par défaut scalping 1m
        hold_min = 5
        hold_max = 30

        # Ajuster selon volatilité
        if atr_percent > 0.025:  # Haute volatilité
            hold_min = 3
            hold_max = 15
        elif atr_percent > 0.015:  # Moyenne
            hold_min = 7
            hold_max = 20
        else:  # Faible
            hold_min = 12
            hold_max = 30

        # Ajuster selon régime
        if regime == 'BREAKOUT_BULL':
            hold_min = 3
            hold_max = 12
        elif regime == 'TRENDING_BULL':
            hold_min = 10
            hold_max = 25

        # Ajuster selon ADX (momentum fort = sortir plus vite)
        if adx > 40:
            hold_max = min(hold_max, 15)

        return f"{hold_min}-{hold_max} min"

    def _check_prebreakout_setup(self, ad: dict, atr_percent: float) -> tuple[bool, str]:
        """
        PRE-BREAKOUT SNIPER: Détecte compression + accumulation AVANT le spike

        Critères (TOUS requis):
        1. Bollinger Squeeze (BB width <0.8% = compression)
        2. Volume declining OU normal (0.4-1.2x) MAIS OBV monte (accumulation silencieuse)
        3. RSI neutre 40-60 (pas encore parti)
        4. ATR faible ou déclinant (volatilité contractée)
        5. Plus DI commence à croiser Minus DI (changement imminent)
        6. Prix proche support ou consolidation (pas extended)

        Returns:
            (is_valid, reason)
        """
        # 1. BOLLINGER SQUEEZE
        bb_width = self.safe_float(ad.get('bb_width'), 999)
        bb_squeeze = ad.get('bb_squeeze', False)
        bb_pos = self.safe_float(ad.get('bb_position'), 0.5)

        # Squeeze = BB width <0.8% OU flag squeeze = True
        squeeze_active = bb_squeeze or bb_width < 0.008

        if not squeeze_active:
            return False, f"Pas de squeeze (BB width {bb_width*100:.2f}%)"

        # 2. VOLUME DECLINING + OBV MONTANT (Accumulation silencieuse)
        rel_volume = self.safe_float(ad.get('relative_volume'), 1.0)
        vol_pattern = ad.get('volume_pattern')
        obv_osc = self.safe_float(ad.get('obv_oscillator'))

        # Volume faible/normal (pas de spike encore)
        volume_quiet = 0.4 <= rel_volume <= 1.2

        # OBV monte (smart money accumule)
        obv_positive = obv_osc > 100

        if not (volume_quiet and obv_positive):
            return False, f"Volume {rel_volume:.1f}x, OBV {obv_osc:.0f} → Pas d'accumulation silencieuse"

        # 3. RSI NEUTRE (40-60)
        rsi = self.safe_float(ad.get('rsi_14'), 50)

        rsi_neutral = 40 <= rsi <= 60

        if not rsi_neutral:
            return False, f"RSI {rsi:.0f} pas neutre (besoin 40-60)"

        # 4. ATR FAIBLE (Volatilité contractée)
        natr = self.safe_float(ad.get('natr'))
        atr_percentile = self.safe_float(ad.get('atr_percentile'))

        # ATR bas = volatilité contractée (avant explosion)
        atr_low = atr_percentile < 40 or natr < 1.5

        if not atr_low:
            return False, f"ATR pas contracté (percentile {atr_percentile:.0f}%, NATR {natr:.1f})"

        # 5. PLUS DI COMMENCE À CROISER MINUS DI
        plus_di = self.safe_float(ad.get('plus_di'))
        minus_di = self.safe_float(ad.get('minus_di'))
        adx = self.safe_float(ad.get('adx_14'))

        # Plus DI proche ou commence à dépasser Minus DI
        # Ratio >0.9 = proche du croisement
        di_crossing = (plus_di / minus_di) > 0.9 if minus_di > 0 else False

        # ADX faible (pas encore en tendance, mais va démarrer)
        adx_low = 15 < adx < 30

        if not (di_crossing and adx_low):
            return False, f"DI pas en croisement (ratio {plus_di/minus_di if minus_di > 0 else 0:.2f}, ADX {adx:.0f})"

        # 6. PRIX PAS EXTENDED (BB position <0.7)
        if bb_pos > 0.7:
            return False, f"Prix extended (BB pos {bb_pos:.2f} >0.7)"

        # 7. RÉGIME ACCEPTABLE (pas bear fort)
        regime = ad.get('market_regime', 'UNKNOWN')

        good_regimes = ['RANGING', 'TRANSITION', 'TRENDING_BULL', 'VOLATILE']

        if regime not in good_regimes:
            return False, f"Régime {regime} pas adapté"

        # TOUS LES CRITÈRES PASSÉS
        return True, (
            f"✅ BB Squeeze {bb_width*100:.2f}% | "
            f"✅ Vol {rel_volume:.1f}x + OBV +{obv_osc:.0f} (accumulation) | "
            f"✅ RSI {rsi:.0f} neutre | "
            f"✅ ATR contracté (p{atr_percentile:.0f}) | "
            f"✅ DI croisement imminent ({plus_di:.0f}/{minus_di:.0f}) | "
            f"✅ ADX {adx:.0f} prêt à exploser"
        )


# ===========================================================
# EXEMPLE D'UTILISATION
# ===========================================================
if __name__ == "__main__":
    # CAS 1: Résistance proche + momentum FAIBLE → BLOQUÉ
    print("\n" + "="*70)
    print("CAS 1: Résistance proche + MOMENTUM FAIBLE → WAIT")
    print("="*70)

    calc = OpportunityCalculator()

    weak_data = {
        # Volume OK mais pas exceptionnel
        'relative_volume': 1.6,
        'volume_spike_multiplier': 1.4,
        'volume_context': 'oversold_bounce',  # Contexte réel
        'obv_oscillator': 80,
        'volume_quality_score': 58,

        # Momentum OK
        'rsi_14': 58,
        'williams_r': -35,
        'macd_trend': 'BULLISH',
        'cci_20': 45,
        'macd_histogram': 0.0002,

        # Trend OK mais ADX faible
        'adx_14': 22,  # ADX <25
        'plus_di': 24,
        'minus_di': 20,
        'market_regime': 'TRENDING_BULL',

        # RÉSISTANCE PROCHE + momentum pas assez fort
        'nearest_resistance': 1.0860,  # 0.93% plus haut
        'break_probability': 45,
        'nearest_support': 1.0720,
        'bb_position': 0.65,

        'atr_14': 0.0012,
        'natr': None
    }

    result1 = calc.calculate_opportunity(
        symbol="BTCUSDC",
        current_price=1.0760,
        analyzer_data=weak_data
    )

    print(f"ACTION: {result1['action']}")
    print(f"\n{result1['reason']}")
    print(f"\nConditions: {result1['conditions']}")

    # CAS 2: Résistance proche + MOMENTUM FORT → PASSE
    print("\n" + "="*70)
    print("CAS 2: Résistance proche + MOMENTUM FORT → BUY_NOW")
    print("="*70)

    strong_data = {
        # Volume BREAKOUT
        'relative_volume': 2.4,
        'volume_spike_multiplier': 3.1,  # Spike massif >2x
        'volume_context': 'breakout',  # Contexte réel (lowercase)
        'obv_oscillator': 220,
        'volume_quality_score': 78,

        # Momentum FORT
        'rsi_14': 66,
        'williams_r': -22,
        'macd_trend': 'BULLISH',
        'cci_20': 95,
        'macd_histogram': 0.0008,

        # Trend FORT
        'adx_14': 38,  # ADX >25 ✅
        'plus_di': 35,
        'minus_di': 15,
        'market_regime': 'TRENDING_BULL',
        'trend_alignment': 75,

        # RÉSISTANCE PROCHE mais momentum ÉCRASE
        'nearest_resistance': 1.0860,  # Même résistance 0.93%
        'break_probability': 45,
        'nearest_support': 1.0720,
        'bb_position': 0.68,
        'bb_expansion': True,

        # Données pour pullback hint
        'ema_7': 1.0745,
        'vwap_quote_10': 1.0735,

        'atr_14': 0.0012,
        'natr': None
    }

    result2 = calc.calculate_opportunity(
        symbol="BTCUSDC",
        current_price=1.0760,
        analyzer_data=strong_data
    )

    print(f"ACTION: {result2['action']}")
    print(f"\n{result2['reason']}")
    print(f"\nConditions: {result2['conditions']}")
    print(f"Hold time: {result2['estimated_hold_time']}")

    # CAS 3: Résistance loin → OK même avec momentum faible
    print("\n" + "="*70)
    print("CAS 3: Résistance LOIN (>1.5%) → BUY_NOW même si momentum normal")
    print("="*70)

    far_resistance_data = {
        # Volume normal
        'relative_volume': 1.7,
        'volume_spike_multiplier': 1.5,
        'volume_context': 'neutral',  # Contexte réel (lowercase)
        'obv_oscillator': 100,
        'volume_quality_score': 60,

        # Momentum OK
        'rsi_14': 60,
        'williams_r': -30,
        'macd_trend': 'BULLISH',
        'cci_20': 50,
        'macd_histogram': 0.0003,

        # Trend normal
        'adx_14': 23,
        'plus_di': 25,
        'minus_di': 19,
        'market_regime': 'TRENDING_BULL',

        # RÉSISTANCE LOIN
        'nearest_resistance': 1.0920,  # 1.49% plus haut (juste <1.5%)
        'break_probability': 50,
        'nearest_support': 1.0720,
        'bb_position': 0.60,

        'atr_14': 0.0012,
        'natr': None
    }

    result3 = calc.calculate_opportunity(
        symbol="BTCUSDC",
        current_price=1.0760,
        analyzer_data=far_resistance_data
    )

    print(f"ACTION: {result3['action']}")
    print(f"\n{result3['reason']}")
    print(f"\nConditions: {result3['conditions']}")
