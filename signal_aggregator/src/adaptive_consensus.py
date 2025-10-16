"""
Module de consensus adaptatif basé sur les familles de stratégies et le régime de marché.

Au lieu d'exiger un nombre fixe de stratégies, le consensus s'adapte selon :
- Le régime de marché actuel
- Les familles de stratégies qui ont émis des signaux
- La cohérence entre familles adaptées au régime
"""

import logging
from typing import Any

from .strategy_classification import (
    STRATEGY_FAMILIES,
    get_strategy_family,
    is_strategy_acceptable_for_regime,
    is_strategy_optimal_for_regime,
)

logger = logging.getLogger(__name__)


class AdaptiveConsensusAnalyzer:
    """
    Analyse le consensus de manière adaptative selon le régime de marché.

    Au lieu d'un seuil fixe (6 stratégies), utilise une approche intelligente :
    - En trending : privilégier les stratégies trend-following et breakout
    - En ranging : privilégier les mean-reversion
    - En volatile : privilégier les breakout et volume-based
    """

    def __init__(self):
        # Consensus minimum par famille selon le régime ET le side
        # RECALIBRÉ: 28 stratégies opérationnelles après déblocage des 8 "muettes"
        # SPOT ONLY: BUY = business principal, SELL = sortie/profits seulement
        self.regime_family_requirements = {
            "TRENDING_BULL": {
                "BUY": {
                    # Réduit 2->1 (pas assez de signaux en réalité)
                    "trend_following": 1,
                    # Réduit 4->3 (signaux rares mais de qualité)
                    "total_min": 3,
                },
                "SELL": {
                    "total_min": 3,  # Durci 2->3 pour éviter sorties prématurées en bull
                    "trend_following": 1,  # Au moins 1 trend pour confirmer retournement
                },
            },
            "TRENDING_BEAR": {
                "BUY": {
                    "trend_following": 1,  # Assoupli pour permettre rebonds
                    "mean_reversion": 2,  # Exiger rebond légitime
                    "total_min": 4,  # 4 stratégies au lieu de 6 - plus de rebonds
                },
                "SELL": {
                    "trend_following": 1,  # Facile de sortir en bear
                    "total_min": 2,  # 2 stratégies suffisent
                },
            },
            "RANGING": {
                "BUY": {
                    "mean_reversion": 2,  # 2 mean-reversion minimum
                    "structure_based": 1,  # Structure pour support
                    "total_min": 4,  # 4 stratégies pour BUY en range
                },
                "SELL": {
                    "mean_reversion": 1,  # 1 reversion suffit pour sortir
                    "total_min": 2,  # 2 stratégies pour SELL
                },
            },
            "VOLATILE": {
                "BUY": {
                    "breakout": 1,  # 1 breakout suffit en volatilité
                    "volume_based": 1,  # Volume pour confirmer
                    "total_min": 3,  # 3 stratégies au lieu de 5 - plus d'opportunités
                },
                "SELL": {
                    "volume_based": 1,  # Volume pour confirmer sortie
                    "total_min": 2,  # 2 stratégies suffisent
                },
            },
            "BREAKOUT_BULL": {
                "BUY": {
                    "breakout": 2,  # 2 breakout minimum
                    "volume_based": 1,  # Volume crucial
                    "total_min": 4,  # 4 stratégies pour BUY breakout
                },
                "SELL": {
                    "total_min": 3,  # Durci 2->3 pour rally explosifs
                    "volume_based": 1,  # Volume pour confirmer essoufflement
                },
            },
            "BREAKOUT_BEAR": {
                "BUY": {
                    "breakout": 1,  # 1 breakout peut suffire
                    "mean_reversion": 2,  # Rebond très confirmé
                    "total_min": 5,  # 5 stratégies au lieu de 7 - moins restrictif
                },
                "SELL": {
                    "breakout": 1,  # 1 breakout suffit pour sortir
                    "total_min": 2,  # 2 stratégies suffisent
                },
            },
            "TRANSITION": {
                "BUY": {
                    "trend_following": 1,  # Direction incertaine
                    "mean_reversion": 1,  # Équilibre
                    # Réduit 4->3 (plus flexible en transition)
                    "total_min": 3,
                },
                "SELL": {"total_min": 2},  # 2 stratégies pour sortir
            },
            "UNKNOWN": {
                "BUY": {
                    "trend_following": 1,  # Au moins 1 trend
                    "mean_reversion": 1,  # Au moins 1 reversion
                    "total_min": 5,  # 5 stratégies (inconnu = prudent)
                },
                "SELL": {"total_min": 2},  # 2 stratégies pour sortir
            },
        }

        # Poids des familles OPTIMISÉS SCALPING
        self.family_weights = {
            "trend_following": 1.0,  # Standard pour suivre les tendances intraday
            # Légèrement pénalisé (moins fiable en crypto directionnelle)
            "mean_reversion": 0.9,
            "breakout": 1.3,  # Augmenté pour scalping (cassures importantes)
            "volume_based": 1.4,  # Crucial en scalping (flux/liquidité)
            "structure_based": 1.1,  # Support/résistance utiles mais secondaires
            "flow": 1.3,  # Analyse de flux d'ordres (si utilisé)
            "contrarian": 0.8,  # Pénalisé car risqué en scalping directionnel
            "unknown": 0.5,  # Stratégies non classifiées = moins fiables
        }

    def analyze_adaptive_consensus(
        self,
        signals: list[dict[str, Any]],
        market_regime: str,
        timeframe: str | None = None,
        volatility_regime: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Analyse si un groupe de signaux forme un consensus adapté au régime.

        Args:
            signals: Liste des signaux du même symbole/direction
            market_regime: Régime de marché actuel
            timeframe: Timeframe des signaux (3m, 5m, 15m, etc.)

        Returns:
            Tuple (has_consensus, analysis_details)
        """
        logger.info(
            f"🔍 Analyse consensus: {len(signals)} signaux, régime: {market_regime}, timeframe: {timeframe}"
        )

        if not signals:
            logger.info("🔍 Consensus: Aucun signal")
            return False, {"reason": "Aucun signal"}

        # Normaliser et unifier volatility_regime
        if volatility_regime is None:
            # Tenter de lire depuis les signaux
            for s in signals:
                vr = (s.get("metadata") or {}).get(
                    "volatility_regime") or s.get("volatility_regime")
                if vr:
                    volatility_regime = vr
                    break

        # S'assurer que volatility_regime n'est jamais None après cette section
        if volatility_regime is None:
            volatility_regime = "normal"

        # UNIFIÉ: Une seule logique de normalisation
        vol_level = str(volatility_regime or "normal").lower()
        if vol_level not in ["low", "normal", "high", "extreme"]:
            vol_level = "normal"

        # PATCH 2: Filtrer dès le départ les signaux inutilisables
        clean_signals = []
        for sig in signals:
            if sig.get("side") is None:  # veto / filtre => hors consensus
                continue
            # Normaliser strategy
            strat = sig.get("strategy") or (
                (sig.get("metadata") or {}).get("strategy"))
            if strat:
                sig["strategy"] = strat
            clean_signals.append(sig)

        if not clean_signals:
            return False, {
                "reason": "Uniquement des veto/None, aucun signal de side exploitable"}

        signals = clean_signals

        # Déterminer le side des signaux (tous doivent être du même side pour
        # le consensus)
        signal_sides = {signal.get("side", "BUY") for signal in signals}
        if len(signal_sides) > 1:
            return False, {
                "reason": f"Signaux de sides différents: {signal_sides}"}

        signal_side = signal_sides.pop()
        logger.info(
            f"🔍 Side des signaux: {signal_side}, volatilité: {vol_level}")

        # Classifier les signaux par famille
        families_count: dict[str, int] = {}
        families_signals: dict[str, list[dict[str, Any]]] = {}
        adaptability_scores = []

        for signal in signals:
            strategy = signal.get("strategy", "Unknown")
            family = get_strategy_family(strategy)

            if family not in families_count:
                families_count[family] = 0
                families_signals[family] = []

            families_count[family] += 1
            families_signals[family].append(signal)

            # Calculer le score d'adaptabilité au régime
            is_optimal = is_strategy_optimal_for_regime(
                strategy, market_regime)
            is_acceptable = is_strategy_acceptable_for_regime(
                strategy, market_regime)

            if is_optimal:
                adaptability_scores.append(1.0)
            elif is_acceptable:
                adaptability_scores.append(0.7)
            else:
                adaptability_scores.append(0.3)

        # Calculer les métriques
        total_strategies = len(signals)
        avg_adaptability = (
            sum(adaptability_scores) / len(adaptability_scores)
            if adaptability_scores
            else 0
        )

        # PATCH 3: Ajouter une qualité minimale (confiance)
        conf = [
            float(s.get("confidence", 0))
            for s in signals
            if s.get("confidence") is not None
        ]
        avg_conf = sum(conf) / len(conf) if conf else 0.0
        hi_conf = sum(1 for c in conf if c >= 0.65)

        min_avg = 0.58 if signal_side == "BUY" else 0.50
        min_hi = 2 if signal_side == "BUY" else 1
        if avg_conf < min_avg or hi_conf < min_hi:
            # PATCH 5: Log précis des causes de rejet qualité
            reason_parts = []
            if avg_conf < min_avg:
                reason_parts.append(f"avg_conf {avg_conf:.2f} < {min_avg:.2f}")
            if hi_conf < min_hi:
                reason_parts.append(f"hi_conf {hi_conf} < {min_hi}")
            return False, {
                "reason": f'Qualité insuffisante ({" & ".join(reason_parts)})',
                "avg_confidence": avg_conf,
                "hi_conf_count": hi_conf,
                "families_count": families_count,
                "total_strategies": total_strategies,
            }

        # AMÉLIORATION: Choisir intelligemment entre régime timeframe et unifié
        # Priorité: 1) Régime timeframe si confidence OK, 2) Régime unifié si
        # disponible, 3) UNKNOWN
        timeframe_regime = (
            signal_sides.pop() if "timeframe_regime" in locals() else None
        )

        # Si on a les deux régimes dans le contexte (depuis les métadonnées)
        has_unified = any(s.get("metadata", {}).get(
            "unified_regime") for s in signals)
        has_timeframe = any(
            s.get("metadata", {}).get("timeframe_regime") for s in signals
        )

        if has_timeframe and has_unified:
            # Chercher la confidence du régime timeframe
            timeframe_conf = next(
                (
                    s.get("metadata", {}).get("timeframe_regime_confidence", 0)
                    for s in signals
                    if s.get("metadata", {}).get("timeframe_regime_confidence")
                ),
                0,
            )

            # Si confidence timeframe > 40%, l'utiliser, sinon utiliser unifié
            if timeframe_conf > 40:
                regime = next(
                    (s.get(
                        "metadata",
                        {}).get(
                        "timeframe_regime",
                        market_regime) for s in signals if s.get(
                        "metadata",
                        {}).get("timeframe_regime")),
                    market_regime,
                )
                logger.debug(
                    f"Utilisation régime TIMEFRAME: {regime} (conf: {timeframe_conf})"
                )
            else:
                regime = next(
                    (s.get(
                        "metadata",
                        {}).get(
                        "unified_regime",
                        market_regime) for s in signals if s.get(
                        "metadata",
                        {}).get("unified_regime")),
                    market_regime,
                )
                logger.debug(
                    f"Utilisation régime UNIFIÉ: {regime} (timeframe conf trop faible: {timeframe_conf})"
                )
        else:
            # Fallback sur le régime standard
            regime = market_regime.upper() if market_regime else "UNKNOWN"

        regime = regime.upper() if regime else "UNKNOWN"
        if regime not in self.regime_family_requirements:
            regime = "UNKNOWN"

        # PATCH 3: Ajouter proxy breakout pour TRENDING_BULL
        if (
            regime in ["TRENDING_BULL", "BREAKOUT_BULL"]
            and families_count.get("breakout", 0) == 0
        ) and (
            "trend_following" in families_count
            and "volume_based" in families_count
            and avg_conf >= 0.90
        ):
            families_count["breakout_proxy"] = 1
            logger.debug(
                f"🚀 Proxy breakout ajouté pour {regime} (trend+volume+conf≥0.9)"
            )

        logger.debug(f"🔍 Familles détectées: {families_count}")
        logger.debug(f"🔍 Scores adaptabilité: {adaptability_scores}")
        logger.info(f"🔍 Qualité: avg_conf={avg_conf:.2f}, hi_conf={hi_conf}")

        # Obtenir les requirements pour ce régime ET ce side

        regime_requirements = self.regime_family_requirements[regime]
        if signal_side in regime_requirements:
            requirements: dict[str, int] = regime_requirements[signal_side]
        else:
            # Fallback si le side n'existe pas (ancien format)
            requirements: dict[str, int] = regime_requirements

        logger.debug(
            f"🔍 Requirements pour {regime}/{signal_side}: {requirements}")

        # PATCH 2: Assouplir TRENDING_BULL/BUY si excellente qualité
        total_min = requirements.get("total_min", 6)
        if regime == "TRENDING_BULL" and signal_side == "BUY":
            family_diversity = len(
                [
                    f
                    for f in families_count
                    if f != "unknown" and families_count[f] > 0
                ]
            )
            if family_diversity >= 3 and avg_conf >= 0.90:
                total_min = 3  # Assouplir 4→3 si excellente qualité
                logger.info(
                    f"🎯 Total_min assoupli 4→3 pour BULL/BUY: diversité={family_diversity}, avg_conf={avg_conf:.2f}"
                )

        # AJUSTEMENT SPÉCIAL TRANSITION: Override si momentum fort
        if regime == "TRANSITION" and signal_side == "BUY":
            momentum_score = None
            # Chercher le momentum dans les métadonnées des signaux
            for sig in signals:
                if sig.get("metadata", {}).get("momentum_score"):
                    momentum_score = float(sig["metadata"]["momentum_score"])
                    break

            # Override si momentum > 55 avec au moins 3 stratégies (au lieu de
            # 4)
            if momentum_score and momentum_score > 55 and total_strategies >= 3:
                logger.info(
                    f"✅ TRANSITION override: momentum {momentum_score:.1f} > 55 avec {total_strategies} stratégies"
                )
                # Continue sans bloquer sur total_min
            elif total_strategies < total_min:
                return False, {
                    "reason": f"Pas assez de stratégies: {total_strategies} < {total_min}",
                    "families_count": families_count,
                    "total_strategies": total_strategies,
                    "required_min": total_min,
                    "avg_adaptability": avg_adaptability,
                    "momentum_score": momentum_score if momentum_score else "N/A",
                }
        elif total_strategies < total_min:
            return False, {
                "reason": f"Pas assez de stratégies: {total_strategies} < {total_min}",
                "families_count": families_count,
                "total_strategies": total_strategies,
                "required_min": total_min,
                "avg_adaptability": avg_adaptability,
            }

        # Vérifier les requirements par famille (si spécifiés)
        missing_families = []
        for family, required_count in requirements.items():
            if family == "total_min":
                continue

            # PATCH 3b: breakout_proxy compte comme breakout
            if family == "breakout":
                actual_count = families_count.get(
                    "breakout", 0) + families_count.get("breakout_proxy", 0)
            else:
                actual_count = families_count.get(family, 0)

            if actual_count < required_count:
                missing_families.append(
                    f"{family}: {actual_count}/{required_count}")

        # Si des familles critiques manquent, TOLÉRER si autres critères OK
        if missing_families:
            # TOLÉRANCE: Pas grave si une famille manque, on continue quand
            # même
            consensus_strength_preview = self._calculate_preview_consensus_strength(
                families_count, regime)
            family_diversity = len(
                [
                    f
                    for f in families_count
                    if f != "unknown" and families_count[f] > 0
                ]
            )

            # PATCH 4: Bypass durci - exige plus de critères si confidence
            # faible
            criteria = 0
            if avg_adaptability >= 0.6:
                criteria += 1
            if consensus_strength_preview >= 1.9:
                criteria += 1
            if family_diversity >= 2:
                criteria += 1
            if total_strategies >= total_min + 1:
                criteria += 1
            if len(missing_families) == 1:
                criteria += 1

            # Durcir si confidence médiocre
            min_criteria = 3 if avg_conf < 0.7 else 2
            can_bypass = criteria >= min_criteria

            if can_bypass:
                logger.info(
                    f"✅ Familles manquantes TOLÉRÉES: {', '.join(missing_families)} - Diversité: {family_diversity}, adaptabilité: {avg_adaptability:.2f}"
                )
            else:
                return False, {
                    "reason": f'Familles manquantes ET critères tous insuffisants: {", ".join(missing_families)} (diversité: {family_diversity}, adaptabilité: {avg_adaptability:.2f})',
                    "families_count": families_count,
                    "missing_families": missing_families,
                    "avg_adaptability": avg_adaptability,
                    "family_diversity": family_diversity,
                    "consensus_preview": consensus_strength_preview,
                }

        # PATCH 5: SELL renforcé - toujours exiger une famille acceptable
        # minimum
        if signal_side == "SELL":
            best_family_hit = any(
                (STRATEGY_FAMILIES.get(f, {}).get("best_regimes") or []).__contains__(
                    regime
                )
                for f, c in families_count.items()
                if c > 0
            )
            acceptable_family_hit = any(
                (STRATEGY_FAMILIES.get(
                    f,
                    {}).get("acceptable_regimes") or []).__contains__(regime) for f,
                c in families_count.items() if c > 0)
            if not best_family_hit and not acceptable_family_hit:
                return False, {
                    "reason": "SELL sans famille optimale/acceptable au régime",
                    "families_count": families_count,
                    "regime": regime,
                    "avg_confidence": avg_conf,
                }

            # En bull fort, durcir encore plus les SELL
            if regime in [
                "TRENDING_BULL",
                    "BREAKOUT_BULL"] and not best_family_hit:
                return False, {
                    "reason": f"SELL en {regime} sans famille OPTIMALE (seulement acceptable)",
                    "families_count": families_count,
                    "regime": regime,
                    "avg_confidence": avg_conf,
                }

        # Calculer le score de consensus pondéré
        weighted_score = 0.0
        total_weight = 0.0

        for family, count in families_count.items():
            if family == "unknown":
                continue

            weight = self.family_weights.get(family, 1.0)
            # Bonus si la famille est optimale pour ce régime
            family_config = STRATEGY_FAMILIES.get(family, {})
            if regime in family_config.get("best_regimes", []):
                weight *= 1.5
            elif regime in family_config.get("poor_regimes", []):
                weight *= 0.5

            weighted_score += count * weight
            total_weight += weight

        consensus_strength: float = weighted_score / max(1, total_weight)

        # PÉNALITÉ DIVERSITÉ PROGRESSIVE : Réduction lissée selon nombre de
        # familles
        unique_families = len(
            [
                f
                for f in families_count
                if f != "unknown" and families_count[f] > 0
            ]
        )
        if unique_families < 3:
            # Pénalité progressive : -25% pour 1 famille, -15% pour 2 familles
            if unique_families == 1:
                diversity_penalty = 0.75  # -25% pour mono-famille
            elif unique_families == 2:
                diversity_penalty = 0.85  # -15% pour bi-famille
            else:
                diversity_penalty = 0.95  # -5% pour edge case

            consensus_strength *= diversity_penalty
            penalty_pct = int((1 - diversity_penalty) * 100)
            logger.info(
                f"🎯 Pénalité diversité progressive: {unique_families} familles < 3 (-{penalty_pct}%)"
            )

        # Décision finale basée sur la force du consensus RAISONNABLE
        # RÉALISTE: Basé sur les vraies données observées (3-10 stratégies
        # simultanées)

        # Utiliser vol_level déjà normalisé (supprime la duplication)
        volatility_level = vol_level  # Réutiliser la valeur déjà calculée

        # Utiliser la méthode dynamique pour calculer le seuil
        min_consensus_strength = self.get_dynamic_consensus_threshold(
            regime, timeframe or "3m", vol_level
        )

        # Ajustement supplémentaire selon adaptabilité OPTIMISÉ SCALPING
        if avg_adaptability > 0.75:
            min_consensus_strength *= (
                0.85  # Réduire de 15% (au lieu de 10%) pour scalping réactif
            )
        elif avg_adaptability > 0.6:
            min_consensus_strength *= 0.92  # Réduire de 8% pour bonne adaptabilité
        elif avg_adaptability < 0.4:
            min_consensus_strength *= (
                # Augmenter de 5% seulement (au lieu de 10%) pour rester
                # réactif
                1.05
            )

        # PATCH 6: Bonus confiance - récompenser les packs très confiants
        if avg_conf >= 0.66 and hi_conf >= 3:
            min_consensus_strength *= 0.92  # -8%
            logger.info(
                f"🎯 Bonus confiance appliqué: avg_conf={avg_conf:.2f}, hi_conf={hi_conf}"
            )

        # PATCH 2b: Rabais spécial breakout manquant si excellente qualité
        if missing_families == [
                "breakout: 0/1"] and avg_conf >= 0.90 and hi_conf >= 3:
            min_consensus_strength *= 0.90  # -10% supplémentaire
            logger.info(
                f"🚀 Rabais breakout manquant: avg_conf={avg_conf:.2f}, hi_conf={hi_conf}"
            )

        # Si familles manquantes ont été TOLÉRÉES, être plus permissif sur
        # consensus strength
        families_were_tolerated = missing_families and (
            avg_adaptability >= 0.6
            or consensus_strength >= 1.8
            or len(
                [
                    f
                    for f in families_count
                    if f != "unknown" and families_count[f] > 0
                ]
            )
            >= 2
            or total_strategies >= total_min + 1
            or len(missing_families) == 1
        )

        if families_were_tolerated:
            min_consensus_strength *= 0.9  # Réduire de 10% si familles tolérées
            logger.info(
                f"📊 Seuil consensus ajusté (familles tolérées): {min_consensus_strength:.2f}"
            )

        has_consensus = consensus_strength >= min_consensus_strength

        return has_consensus, {
            "has_consensus": has_consensus,
            "families_count": families_count,
            "total_strategies": total_strategies,
            "avg_adaptability": avg_adaptability,
            "avg_confidence": avg_conf,
            "hi_conf_count": hi_conf,
            "consensus_strength": consensus_strength,
            "min_required_strength": min_consensus_strength,
            "regime": regime,
            "signal_side": signal_side,
            "volatility_level": vol_level,
            "missing_families": missing_families if missing_families else None,
        }

    def _calculate_preview_consensus_strength(
        self, families_count: dict[str, int], regime: str
    ) -> float:
        """Calcule rapidement le consensus_strength pour décision d'assouplissement."""
        weighted_score = 0.0
        total_weight = 0.0

        for family, count in families_count.items():
            if family == "unknown":
                continue

            weight = self.family_weights.get(family, 1.0)
            family_config = STRATEGY_FAMILIES.get(family, {})

            if regime in family_config.get("best_regimes", []):
                weight *= 1.5
            elif regime in family_config.get("poor_regimes", []):
                weight *= 0.5

            weighted_score += count * weight
            total_weight += weight

        return float(weighted_score) / max(1, total_weight)

    def get_dynamic_consensus_threshold(
        self, regime: str, timeframe: str, volatility_level: str = "normal"
    ) -> float:
        """
        Calcule dynamiquement le seuil de consensus selon régime, timeframe et volatilité.

        JUSTIFICATION DES SEUILS:
        - Les seuils sont basés sur l'analyse empirique de 10,000+ signaux historiques
        - Consensus strength = (weighted_score / total_weight) où weighted_score est la somme
          des signaux pondérés par famille (1.0 à 1.5 selon adaptation au régime)

        CALCUL DU SEUIL DE BASE:
        - 1m: 3.0 = Exige ~6-7 stratégies bien adaptées (fort filtrage du bruit court terme)
        - 3m: 2.5 = Exige ~5-6 stratégies bien adaptées (équilibre signal/bruit)
        - 5m: 2.2 = Exige ~4-5 stratégies bien adaptées (signaux plus fiables)
        - 15m: 2.0 = Exige ~4 stratégies bien adaptées (tendances établies)

        AJUSTEMENTS VOLATILITÉ:
        - Low (×1.1): Marchés calmes = faux signaux rares, donc plus strict
        - Normal (×1.0): Conditions standard
        - High (×0.9): Plus de signaux légitimes, être plus permissif
        - Extreme (×0.8): Chaos de marché, accepter plus de signaux pour ne pas rater les moves

        Returns:
            Seuil de consensus ajusté (typiquement entre 1.6 et 3.3)
        """

        # Seuils de base empiriques AJUSTÉS MODÉRÉMENT pour équilibrer
        # protection/opportunités
        base_thresholds = {
            "1m": 2.0,  # Réduit 2.2->2.0 (légère baisse)
            # Réduit 1.8->1.6 (timeframe principal, ajustement modéré)
            "3m": 1.6,
            "5m": 1.5,  # Réduit 1.6->1.5 (légère baisse)
            "15m": 1.3,  # Réduit 1.4->1.3 (légère baisse)
        }

        # Multiplicateurs selon volatilité (basés sur backtests)
        volatility_multipliers = {
            # +10% strict (peu de mouvements = peu de vrais signaux)
            "low": 1.1,
            "normal": 1.0,  # Standard
            "high": 0.9,  # -10% permissif (beaucoup de mouvements légitimes)
            # -20% très permissif (ne pas rater les gros moves)
            "extreme": 0.8,
        }

        base = base_thresholds.get(timeframe, 2.5)
        multiplier = volatility_multipliers.get(volatility_level, 1.0)

        return base * multiplier

    def analyze_adaptive_consensus_mtf(
        self,
        signals: list[dict[str, Any]],
        market_regime: str,
        original_signal_count: int,
        timeframe: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Analyse le consensus pour les signaux MTF post-conflit avec des critères assouplis.

        Quand le buffer MTF a résolu un conflit et filtré des signaux, on doit adapter
        notre logique car on avait initialement plus de stratégies qui étaient d'accord.

        Args:
            signals: Liste des signaux restants après résolution de conflit
            market_regime: Régime de marché actuel
            original_signal_count: Nombre de signaux avant la résolution de conflit

        Returns:
            Tuple (has_consensus, analysis_details)
        """
        if not signals:
            return False, {"reason": "Aucun signal"}

        # Déterminer le side des signaux (MTF post-conflit)
        signal_sides = {signal.get("side", "BUY") for signal in signals}
        if len(signal_sides) > 1:
            return False, {
                "reason": f"Signaux MTF de sides différents: {signal_sides}"}

        signal_side = signal_sides.pop()

        # Utiliser le nombre original pour la validation du consensus
        # car ces stratégies étaient toutes d'accord avant la résolution de
        # conflit
        effective_strategy_count = max(len(signals), original_signal_count)

        # Classifier les signaux par famille (sur les signaux restants)
        families_count: dict[str, int] = {}
        families_signals: dict[str, list[dict[str, Any]]] = {}
        adaptability_scores = []

        for signal in signals:
            strategy = signal.get("strategy", "Unknown")
            family = get_strategy_family(strategy)

            if family not in families_count:
                families_count[family] = 0
                families_signals[family] = []

            families_count[family] += 1
            families_signals[family].append(signal)

            # Score d'adaptabilité
            is_optimal = is_strategy_optimal_for_regime(
                strategy, market_regime)
            is_acceptable = is_strategy_acceptable_for_regime(
                strategy, market_regime)

            if is_optimal:
                adaptability_scores.append(1.0)
            elif is_acceptable:
                adaptability_scores.append(0.7)
            else:
                adaptability_scores.append(0.3)

        avg_adaptability = (
            sum(adaptability_scores) / len(adaptability_scores)
            if adaptability_scores
            else 0
        )

        # Pour MTF post-conflit, on assouplit les critères
        regime = market_regime.upper() if market_regime else "UNKNOWN"
        if regime not in self.regime_family_requirements:
            regime = "UNKNOWN"

        # Obtenir les requirements selon le side (MTF)
        regime_requirements = self.regime_family_requirements[regime]
        if signal_side in regime_requirements:
            requirements: dict[str, int] = regime_requirements[signal_side]
        else:
            # Fallback si le side n'existe pas (ancien format)
            requirements: dict[str, int] = regime_requirements

        # Ajuster le minimum pour MTF post-conflit
        # Avec 28 stratégies actives, même après conflit on devrait avoir assez
        # de signaux
        total_min = requirements.get(
            "total_min", 5
        )  # Garder le seuil standard car plus de stratégies

        # Vérifier avec le nombre effectif (original)
        if effective_strategy_count < total_min:
            return False, {
                "reason": f"Pas assez de stratégies effectives: {effective_strategy_count} < {total_min}",
                "families_count": families_count,
                "total_strategies": len(signals),
                "original_strategies": original_signal_count,
                "effective_strategies": effective_strategy_count,
                "required_min": total_min,
                "avg_adaptability": avg_adaptability,
                "is_mtf_post_conflict": True,
            }

        # Pour MTF post-conflit, on est plus permissif sur les familles manquantes
        # car le filtrage a pu éliminer certaines familles

        # Calculer le score de consensus pondéré
        weighted_score = 0.0
        total_weight = 0.0

        for family, count in families_count.items():
            if family == "unknown":
                continue

            weight = self.family_weights.get(family, 1.0)
            family_config = STRATEGY_FAMILIES.get(family, {})

            if regime in family_config.get("best_regimes", []):
                weight *= 1.5
            elif regime in family_config.get("poor_regimes", []):
                weight *= 0.5

            # Bonus pour MTF post-conflit car on sait que d'autres stratégies
            # étaient d'accord
            weight *= 1.2

            weighted_score += count * weight
            total_weight += weight

        consensus_strength: float = weighted_score / max(1, total_weight)

        # Plus permissif pour MTF post-conflit (seuils baissés)
        min_consensus_strength = 1.5 if avg_adaptability > 0.6 else 2.0

        has_consensus = (
            consensus_strength >= min_consensus_strength
            or effective_strategy_count >= total_min + 2
        )

        return has_consensus, {
            "has_consensus": has_consensus,
            "families_count": families_count,
            "total_strategies": len(signals),
            "original_strategies": original_signal_count,
            "effective_strategies": effective_strategy_count,
            "avg_adaptability": avg_adaptability,
            "consensus_strength": consensus_strength,
            "min_required_strength": min_consensus_strength,
            "regime": regime,
            "is_mtf_post_conflict": True,
            "mtf_consensus_bonus": "Applied - reduced thresholds for post-conflict MTF signals",
        }

    def get_adjusted_min_strategies(
        self,
        market_regime: str,
        available_families: list[str],
        signal_side: str = "BUY",
        avg_confidence: float | None = None,
    ) -> int:
        """
        Retourne le nombre minimum de stratégies ajusté selon le régime et les familles disponibles.

        Args:
            market_regime: Régime de marché actuel
            available_families: Familles de stratégies disponibles

        Returns:
            Nombre minimum de stratégies requis
        """
        regime = market_regime.upper() if market_regime else "UNKNOWN"
        if regime not in self.regime_family_requirements:
            regime = "UNKNOWN"

        # Obtenir les requirements selon le side
        regime_requirements = self.regime_family_requirements[regime]
        if signal_side in regime_requirements:
            requirements: dict[str, int] = regime_requirements[signal_side]
        else:
            # Fallback si le side n'existe pas (ancien format)
            requirements: dict[str, int] = regime_requirements

        base_min = requirements.get("total_min", 5)

        # Ajuster selon les familles disponibles (avec 28 stratégies actives =
        # plus d'exigence)
        optimal_families = 0
        for family in available_families:
            family_config = STRATEGY_FAMILIES.get(family, {})
            if regime in family_config.get("best_regimes", []):
                optimal_families += 1

        # Prendre en compte la confidence pour ajuster dynamiquement
        confidence_bonus = 0
        if avg_confidence is not None:
            if avg_confidence >= 0.9:
                confidence_bonus = -1  # Assouplir si excellente confidence
            elif avg_confidence < 0.6:
                confidence_bonus = +1  # Durcir si confidence médiocre

        # Avec 28 stratégies, on peut être plus exigeant sur le consensus
        if optimal_families >= 4:
            adjustment = 0  # Standard avec 4+ familles optimales
        elif optimal_families >= 3:
            adjustment = 1  # Légère augmentation avec 3 familles
        elif optimal_families >= 2:
            adjustment = 2  # Plus strict avec 2 familles seulement
        else:
            adjustment = 3  # Peu de familles optimales, très strict

        return max(base_min, base_min + adjustment + confidence_bonus)
