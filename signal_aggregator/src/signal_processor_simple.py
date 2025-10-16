"""
Module de traitement des signaux - VERSION ULTRA-SIMPLIFIÉE.
Remplace l'ancien système complexe par juste consensus adaptatif + filtres critiques.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

from .adaptive_consensus import AdaptiveConsensusAnalyzer
from .critical_filters import CriticalFilters

logger = logging.getLogger(__name__)


class SimpleSignalProcessor:
    """
    Processeur ultra-simplifié pour la validation des signaux.
    Logic: Consensus adaptatif + quelques filtres critiques seulement.
    """

    def __init__(
            self,
            context_manager,
            database_manager=None,
            db_connection=None):
        """
        Initialise le processeur simplifié.

        Args:
            context_manager: Gestionnaire de contexte de marché
            database_manager: Gestionnaire de base de données (optionnel)
            db_connection: Connexion directe à la DB (pour filtres critiques)
        """
        self.context_manager = context_manager
        self.database_manager = database_manager

        # Systèmes simplifiés
        self.consensus_analyzer = AdaptiveConsensusAnalyzer()
        self.critical_filters = CriticalFilters(db_connection=db_connection)

        # SEUIL MINIMUM DE CONFIDENCE POUR SCALPING
        self.min_confidence_threshold = 0.6  # Rejeter signaux < 60% confidence

        # Cache de contexte pour optimisation (TTL dynamique)
        self.context_cache = {}
        self.base_cache_ttl = 5  # Base 5 secondes

        # Statistiques détaillées pour debug et optimisation
        self.stats: dict[str, Any] = {
            "signals_processed": 0,
            "signals_validated": 0,
            "consensus_rejected": 0,
            "critical_filter_rejected": 0,
            "low_confidence_rejected": 0,  # NOUVEAU: Track rejets par faible confidence
            "errors": 0,
            "rejections_by_regime": {},  # Par régime de marché
            "rejections_by_family": {},  # Par famille de stratégies
            "avg_strategies_per_consensus": 0.0,
            "avg_confidence_validated": 0.0,
            "consensus_strength_distribution": [],  # Pour analyser les seuils
            "wave_winner_signals": 0,  # Signaux post-vague
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def process_signal(self, signal_data: str) -> dict[str, Any] | None:
        """
        Traite un signal individuel reçu depuis Redis.
        VERSION SIMPLIFIÉE: Juste structure + passage au buffer.

        Args:
            signal_data: Données du signal au format JSON

        Returns:
            Signal parsé et validé structurellement ou None
        """
        try:
            # Parsing du message
            signal = json.loads(signal_data)
            self.stats["signals_processed"] += 1

            # Validation structure de base uniquement
            if not self._validate_signal_structure(signal):
                logger.debug("Signal rejeté: structure invalide")
                self.stats["errors"] += 1
                return None

            # NORMALISATION CONFIDENCE: Clamp entre 0 et 1
            conf = signal.get("confidence")
            if conf is not None:
                try:
                    conf_val = float(conf)
                except BaseException:
                    conf_val = 0.0
                # Clamp entre 0 et 1
                conf_val = max(conf_val, 0.0)
                conf_val = min(conf_val, 1.0)
                signal["confidence"] = conf_val
            else:
                # Confidence manquante = signal faible par défaut
                signal["confidence"] = 0.0

            # FILTRE CONFIDENCE MINIMUM POUR SCALPING
            if signal["confidence"] < self.min_confidence_threshold:
                logger.debug(
                    f"Signal rejeté: confidence {signal['confidence']:.2f} < {self.min_confidence_threshold}"
                )
                self.stats["low_confidence_rejected"] += 1
                return None

            # Ajouter timestamp de réception
            signal["received_at"] = datetime.utcnow().isoformat()

            return signal

        except json.JSONDecodeError as e:
            # Logger un extrait du message brut pour debug
            signal_excerpt = (
                signal_data[:200] if len(signal_data) > 200 else signal_data
            )
            logger.exception(
                f"Erreur parsing JSON signal: {e}. Message brut (200 chars): {signal_excerpt}"
            )
            self.stats["errors"] += 1
            return None
        except Exception:
            logger.exception("Erreur traitement signal")
            self.stats["errors"] += 1
            return None

    def _get_dynamic_cache_ttl(self, timeframe: str) -> int:
        """Calcule TTL dynamique basé sur le timeframe."""
        if timeframe in ("1m", "3m"):
            return 5
        if timeframe == "5m":
            return 15
        if timeframe == "15m":
            return 30
        return 60  # défaut raisonnable au-delà

    async def _get_cached_context(
        self, symbol: str, timeframe: str
    ) -> dict[str, Any] | None:
        """Récupère le contexte avec cache TTL dynamique."""
        cache_key = f"{symbol}_{timeframe}"
        now = datetime.utcnow()
        dynamic_ttl = self._get_dynamic_cache_ttl(timeframe)

        if cache_key in self.context_cache:
            cached_context, timestamp = self.context_cache[cache_key]
            if (now - timestamp).total_seconds() < dynamic_ttl:
                self.stats["cache_hits"] += 1
                return cached_context

        # Cache miss - récupérer le contexte
        context = self.context_manager.get_market_context(symbol, timeframe)
        if context:
            self.context_cache[cache_key] = (context, now)
        self.stats["cache_misses"] += 1
        return context

    def _normalize_consensus_strength(
        self, strength: float, market_regime: str
    ) -> float:
        """Normalise le consensus_strength en confidence 0-1 basé sur les seuils réels."""
        # Seuils typiques selon adaptive_consensus.py
        if market_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]:
            max_expected = 3.5  # Plus strict en bear
        else:
            max_expected = 2.5  # Normal

        # Normalisation avec saturation à 1.0
        normalized = min(1.0, strength / max_expected)
        return max(0.0, normalized)  # S'assurer que c'est positif

    async def validate_signal_group(
        self, signals: list, symbol: str, timeframe: str, side: str
    ) -> dict[str, Any] | None:
        """
        Valide un groupe de signaux avec le système simplifié.
        REMPLACE TOUTE LA LOGIQUE COMPLEXE DE VALIDATION.

        Args:
            signals: Liste des signaux du même symbole/direction
            symbol: Symbole tradé
            timeframe: Timeframe des signaux
            side: Direction (BUY/SELL)

        Returns:
            Signal de consensus validé ou None si rejeté
        """
        try:
            if not signals:
                return None

            # Détecter si c'est un groupe post-vague
            is_wave_winner = any(
                s.get("metadata", {})
                .get("wave_resolution", {})
                .get("is_wave_winner", False)
                for s in signals
            )

            if is_wave_winner:
                self.stats["wave_winner_signals"] += 1
                logger.info(f"🏆 Validation post-vague pour {symbol} {side}")

            # Récupération du contexte de marché avec cache
            context = await self._get_cached_context(symbol, timeframe)
            if not context:
                logger.warning(
                    f"Pas de contexte marché pour {symbol} {timeframe}")
                return None

            # ÉTAPE 1: Consensus adaptatif (principal système)
            market_regime = context.get("market_regime", "UNKNOWN")
            logger.info(f"🔍 Market regime pour {symbol}: {market_regime}")
            logger.info(
                f"📊 Stratégies: {[s.get('strategy') for s in signals]}")

            # Adapter les critères si c'est un signal post-vague
            if is_wave_winner:
                # Assouplir légèrement les critères car déjà passé une
                # sélection
                logger.info("🏆 Critères assouplis pour gagnant de vague")

            has_consensus, consensus_analysis = (
                self.consensus_analyzer.analyze_adaptive_consensus(
                    signals, market_regime, timeframe
                )
            )

            logger.info(
                f"🔍 Consensus result: has_consensus={has_consensus}, analysis={consensus_analysis}"
            )

            if not has_consensus:
                # Enregistrer stats détaillées du rejet
                if market_regime not in self.stats["rejections_by_regime"]:
                    self.stats["rejections_by_regime"][market_regime] = 0
                self.stats["rejections_by_regime"][market_regime] += 1

                # Construire un message de rejet informatif
                if consensus_analysis:
                    reason = consensus_analysis.get("reason")
                    if not reason:
                        # Déduire la raison du rejet depuis les données
                        strength = consensus_analysis.get(
                            "consensus_strength", 0)
                        min_required = consensus_analysis.get(
                            "min_required_strength", 0
                        )
                        if strength < min_required:
                            reason = (
                                f"Consensus faible: {strength:.2f} < {min_required:.2f}"
                            )
                        else:
                            reason = "Consensus rejeté"

                    details = consensus_analysis.get("details", "")
                    rejection_msg = f"{reason}" + \
                        (f" - {details}" if details else "")
                else:
                    rejection_msg = "Analyse de consensus indisponible"

                logger.info(
                    f"❌ Consensus rejeté {symbol} {side}: {rejection_msg}")
                self.stats["consensus_rejected"] += 1
                return None

            # Enregistrer strength brute pour statistiques
            consensus_strength = consensus_analysis.get(
                "consensus_strength", 0)
            self.stats["consensus_strength_distribution"].append(
                consensus_strength)

            # Calculer confidence normalisée avec seuil final
            market_regime = consensus_analysis.get(
                "regime", context.get("market_regime", "UNKNOWN")
            )
            normalized_confidence = self._normalize_consensus_strength(
                consensus_strength, market_regime
            )

            # SEUIL FINAL: Rejeter si normalized_confidence < 0.5
            if normalized_confidence < 0.5:
                logger.info(
                    f"❌ Consensus rejeté: normalized confidence trop faible {normalized_confidence:.2f} (brute: {consensus_strength:.2f})"
                )
                self.stats["low_confidence_rejected"] += 1
                return None

            # Logger les deux valeurs pour analyse
            logger.info(
                f"📊 Consensus strength: brute={consensus_strength:.2f}, normalisée={normalized_confidence:.2f}"
            )

            # ÉTAPE 2: Filtres critiques seulement (éviter les vrais dangers)
            # Enrichir le contexte avec les métriques de consensus pour
            # fail-safe
            # Version normalisée
            context["consensus_strength"] = normalized_confidence
            context["wave_winner"] = is_wave_winner
            context["total_strategies"] = len(signals)
            context["consensus_regime"] = (
                market_regime  # Éviter d'écraser market_regime du context_manager
            )

            filters_pass, filter_reason = self.critical_filters.apply_critical_filters(
                signals, context)

            if not filters_pass:
                logger.info(
                    f"Filtres critiques rejetent {symbol} {side}: {filter_reason}"
                )
                self.stats["critical_filter_rejected"] += 1
                return None

            # ÉTAPE 3: Générer consensus_id pour traçabilité
            consensus_id = self._generate_consensus_id(
                signals, symbol, side, timeframe)

            # Sauvegarder les signaux individuels en base de données (avec
            # consensus_id)
            if self.database_manager:
                for signal in signals:
                    try:
                        # Vérifier si ce signal a déjà été stocké dans cette
                        # vague
                        signal_hash = self._generate_signal_hash(signal)
                        if self._is_signal_already_stored(
                                signal_hash, consensus_id):
                            logger.debug(
                                f"Signal {signal.get('strategy')} déjà stocké pour consensus {consensus_id[:8]}"
                            )
                            continue

                        # Préparer le signal individuel pour la base de données
                        individual_signal = {
                            "strategy": signal.get("strategy"),
                            "symbol": signal.get("symbol"),
                            "side": signal.get("side"),
                            "timestamp": signal.get(
                                "timestamp", datetime.utcnow().isoformat()
                            ),
                            "confidence": signal.get("confidence"),
                            "price": context.get("current_price", 0.0),
                            "metadata": {
                                "timeframe": signal.get("timeframe"),
                                "original_metadata": signal.get("metadata", {}),
                                "part_of_consensus": True,
                                "consensus_id": consensus_id,  # Traçabilité
                                "signal_hash": signal_hash,  # Anti-doublon
                                "market_regime": consensus_analysis.get(
                                    "regime", "UNKNOWN"
                                ),
                            },
                        }
                        # Stocker le signal individuel
                        self.database_manager.store_validated_signal(
                            individual_signal)
                    except Exception as e:
                        logger.warning(
                            f"Erreur sauvegarde signal individuel {signal.get('strategy')}: {e}"
                        )

            # ÉTAPE 4: Construire signal de consensus validé
            consensus_signal = self._build_consensus_signal(
                signals,
                symbol,
                timeframe,
                side,
                context,
                consensus_analysis,
                filter_reason,
                is_wave_winner,
                consensus_id,
                normalized_confidence,
            )

            # Mettre à jour les statistiques moyennes (avec
            # normalized_confidence)
            self.stats["signals_validated"] += 1
            self.stats["avg_strategies_per_consensus"] = (
                self.stats["avg_strategies_per_consensus"]
                * (self.stats["signals_validated"] - 1)
                + len(signals)
            ) / self.stats["signals_validated"]
            # Utiliser la confidence normalisée pour les stats
            self.stats["avg_confidence_validated"] = (
                self.stats["avg_confidence_validated"]
                * (self.stats["signals_validated"] - 1)
                + normalized_confidence
            ) / self.stats["signals_validated"]

            # ÉTAPE 5: Sauvegarde du consensus (avec consensus_id)
            if self.database_manager:
                try:
                    signal_id = self.database_manager.store_validated_signal(
                        consensus_signal
                    )
                    if signal_id:
                        # Ajouter le db_id dans les métadonnées pour que le
                        # coordinator puisse le trouver
                        consensus_signal["metadata"]["db_id"] = signal_id
                        logger.debug(
                            f"DB ID {signal_id} ajouté au consensus {consensus_id[:8]} pour {symbol}"
                        )
                except Exception:
                    logger.exception(
                        "Erreur sauvegarde consensus {consensus_id[:8]}")

            logger.info(
                f"✅ Signal consensus validé: {symbol} {side} "
                f"({len(signals)} stratégies, ID: {consensus_id[:8]}, "
                f"score brut: {consensus_analysis.get('consensus_strength', 0):.2f}, "
                f"normalisé: {normalized_confidence:.2f})")

            return consensus_signal

        except Exception:
            logger.exception("Erreur validation groupe signaux")
            self.stats["errors"] += 1
            return None

    def _validate_signal_structure(self, signal: dict[str, Any]) -> bool:
        """Valide la structure de base d'un signal."""
        required_fields = [
            "strategy",
            "symbol",
            "side",
            "confidence",
            "timeframe"]

        for field in required_fields:
            if field not in signal:
                return False

        # Validation des valeurs
        if signal["side"] not in ["BUY", "SELL"]:
            return False

        try:
            confidence = float(signal["confidence"])
            if confidence < 0 or confidence > 1:
                return False
        except (ValueError, TypeError):
            return False

        return True

    def _generate_consensus_id(
        self, signals: list, symbol: str, side: str, timeframe: str
    ) -> str:
        """
        Génère un ID unique pour ce consensus basé sur le contenu.
        Permet de traçer les groupes et éviter les doublons.
        """
        # Créer une signature unique du consensus
        strategy_names = sorted([s.get("strategy", "") for s in signals])
        strategies_str = "|".join(strategy_names)

        # Hash du contenu pour unicité
        content_hash = hashlib.md5(
            f"{symbol}_{side}_{timeframe}_{strategies_str}_{datetime.utcnow().strftime('%Y%m%d%H%M')}".encode()
        ).hexdigest()[:8]

        return f"consensus_{content_hash}"

    def _generate_signal_hash(self, signal: dict[str, Any]) -> str:
        """Génère un hash unique pour un signal individuel."""
        key_fields = f"{signal.get('strategy')}_{signal.get('symbol')}_{signal.get('side')}_{signal.get('timestamp', '')}"
        return hashlib.md5(key_fields.encode()).hexdigest()[:8]

    def _is_signal_already_stored(
            self,
            signal_hash: str,
            consensus_id: str) -> bool:
        """Vérifie si un signal a déjà été stocké (simple cache en mémoire)."""
        # Pour l'instant, cache simple en mémoire (peut être amélioré avec
        # Redis)
        if not hasattr(self, "_stored_signals_cache"):
            self._stored_signals_cache: set[str] = set()

        cache_key = f"{signal_hash}_{consensus_id}"
        if cache_key in self._stored_signals_cache:
            return True

        self._stored_signals_cache.add(cache_key)
        return False

    def _build_consensus_signal(
        self,
        signals: list,
        symbol: str,
        timeframe: str,
        side: str,
        context: dict[str, Any],
        consensus_analysis: dict[str, Any],
        filter_status: str,
        is_wave_winner: bool = False,
        consensus_id: str | None = None,
        normalized_confidence: float = 0.0,
    ) -> dict[str, Any]:
        """Construit le signal de consensus final."""

        # Calculs de base
        strategies_count = len(signals)
        avg_confidence = sum(float(s["confidence"])
                             for s in signals) / strategies_count

        # Analyser la distribution des timeframes
        timeframe_distribution: dict[str, int] = {}
        for signal in signals:
            tf = signal.get(
                "timeframe", timeframe
            )  # Utiliser le timeframe par défaut si manquant
            timeframe_distribution[tf] = timeframe_distribution.get(tf, 0) + 1

        dominant_timeframe = (
            max(timeframe_distribution, key=lambda x: timeframe_distribution.get(x, 0))
            if timeframe_distribution
            else timeframe
        )

        # Métadonnées des stratégies
        strategy_names = [s["strategy"] for s in signals]
        family_distribution = consensus_analysis.get("families_count", {})

        # Utiliser la confidence déjà calculée et validée
        market_regime = consensus_analysis.get(
            "regime", context.get("market_regime", "UNKNOWN")
        )
        # normalized_confidence déjà passée en paramètre

        # Signal de consensus compatible avec StrategySignal
        return {
            "strategy": "CONSENSUS",  # Champ requis pour StrategySignal
            "symbol": symbol,
            "side": side,
            "timestamp": datetime.utcnow().isoformat(),
            # Prix actuel du marché
            "price": context.get("current_price", 0.0),
            "confidence": normalized_confidence,  # Déjà validée avec seuil 0.5
            # Toutes les métadonnées dans le champ metadata
            "metadata": {
                "type": "CONSENSUS",
                "timeframe": timeframe,
                "strategies_count": strategies_count,
                "strategy_names": strategy_names,
                "avg_confidence": avg_confidence,
                "consensus_strength": consensus_analysis.get("consensus_strength", 0),
                "market_regime": market_regime,
                "family_distribution": family_distribution,
                "filter_status": filter_status,
                # Distribution des timeframes
                "timeframe_distribution": timeframe_distribution,
                "dominant_timeframe": dominant_timeframe,
                "is_multi_timeframe": len(timeframe_distribution) > 1,
                # Traçabilité et signaux post-vague
                "consensus_id": consensus_id,
                "is_wave_winner": is_wave_winner,
                "is_failsafe_trade": context.get(
                    "is_failsafe_trade", False
                ),  # Tag spécial pour fail-safe
                # Contexte de marché
                "volume_ratio": context.get("volume_ratio"),
                "confluence_score": context.get("confluence_score"),
                "volatility_regime": context.get("volatility_regime"),
                # Debug
                "processor": "SimpleSignalProcessor",
                "consensus_analysis": consensus_analysis,
                "original_signals": signals,
            },
        }

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques détaillées du processeur."""
        total_processed: int = int(self.stats["signals_processed"])
        if total_processed > 0:
            success_rate = (
                int(self.stats["signals_validated"]) / total_processed) * 100
        else:
            success_rate = 0

        # Statistiques de cache
        cache_hits: int = int(self.stats["cache_hits"])
        cache_misses: int = int(self.stats["cache_misses"])
        total_cache_requests = cache_hits + cache_misses
        cache_hit_rate = (
            (cache_hits / total_cache_requests * 100)
            if total_cache_requests > 0
            else 0
        )

        # Statistiques de consensus strength (valeurs brutes seulement)
        consensus_strengths: list = list(
            self.stats["consensus_strength_distribution"])
        consensus_stats: dict[str, Any] = {}
        if consensus_strengths:
            consensus_stats = {
                "raw_min": min(consensus_strengths),
                "raw_max": max(consensus_strengths),
                "raw_avg": sum(consensus_strengths) / len(consensus_strengths),
                "count": len(consensus_strengths),
                "note": "Raw consensus strength values (before normalization)",
            }

        return {
            **self.stats,
            "success_rate_percent": success_rate,
            "cache_hit_rate_percent": cache_hit_rate,
            "consensus_strength_stats": consensus_stats,
            "filter_config": (
                self.critical_filters.get_filter_stats()
                if hasattr(self.critical_filters, "get_filter_stats")
                else {}
            ),
            "performance": {
                "avg_strategies_per_consensus": round(
                    float(self.stats["avg_strategies_per_consensus"]), 2
                ),
                "avg_confidence_validated": round(
                    float(self.stats["avg_confidence_validated"]), 3
                ),
                "wave_winner_percentage": (
                    int(self.stats["wave_winner_signals"])
                    / max(1, int(self.stats["signals_validated"]))
                )
                * 100,
            },
        }

    def reset_stats(self):
        """Remet à zéro les statistiques."""
        for key in self.stats:
            if isinstance(self.stats[key], int | float):
                self.stats[key] = 0
            elif isinstance(self.stats[key], dict):
                self.stats[key] = {}
            elif isinstance(self.stats[key], list):
                self.stats[key] = []

        # Vider le cache de contexte aussi
        self.context_cache = {}
