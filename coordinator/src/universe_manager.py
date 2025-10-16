"""Universe Manager - Sélection dynamique des cryptos à trader"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PairScore:
    """Score de tradeabilité d'une paire"""

    symbol: str
    score: float
    atr_pct: float
    roc: float
    volume_ratio: float
    is_ranging: bool
    timestamp: datetime
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class PairState:
    """État d'une paire dans le système de sélection"""

    symbol: str
    is_selected: bool
    score_history: list[float] = field(default_factory=list)
    last_selected: datetime | None = None
    last_deselected: datetime | None = None
    cooldown_until: datetime | None = None
    consecutive_above_threshold: int = 0
    consecutive_below_threshold: int = 0


class UniverseManager:
    """Gestionnaire de l'univers tradable avec sélection dynamique"""

    def __init__(self, redis_client, db_pool=None, config: dict | None = None):
        self.redis = redis_client
        self.db_pool = db_pool
        self.config = config or self._default_config()

        # État des paires
        self.pair_states: dict[str, PairState] = {}
        self.selected_universe: set[str] = set()
        self.core_pairs: set[str] = set(self.config["core_pairs"])

        # Cache des données de marché
        self.market_data_cache: dict[str, dict] = {}
        self.last_update = datetime.now(tz=timezone.utc)

        # Vérifier le pool DB
        if not self.db_pool:
            logger.warning(
                "Pas de pool DB fourni, utilisation de Redis uniquement")

        logger.info(
            f"UniverseManager initialisé avec {len(self.core_pairs)} paires core"
        )

    def _default_config(self) -> dict:
        """Configuration par défaut"""
        return {
            # Paires toujours actives (format USDC pour Binance)
            # BTC/ETH = bases incontournables, top liquidité
            # SOL = Solana, très liquide, mouvements rapides
            # XRP = Ripple, énorme volume, bonne volatilité
            # ADA = Cardano, stable et liquide
            # LINK = Chainlink, oracle leader, patterns fiables
            "core_pairs": [
                "BTCUSDC",
                "ETHUSDC",
                "SOLUSDC",
                "XRPUSDC",
                "ADAUSDC",
                "LINKUSDC",
                "NEARUSDC",
                "AAVEUSDC",
                "DOGEUSDC",
            ],
            # Nombre max de satellites (en plus des core)
            "max_satellites": 6,  # 6 core + 6 satellites = 12 cryptos max
            # Seuils de score
            "score_threshold_enter": 0.2,  # Abaissé pour permettre plus de satellites
            "score_threshold_exit": -0.3,  # Abaissé aussi pour cohérence
            # Hystérésis (nombre de périodes)
            "periods_above_to_enter": 3,
            "periods_below_to_exit": 6,
            # Cool-down après désélection (minutes)
            "cooldown_minutes": 45,  # Augmenté pour éviter réactivation trop rapide
            # Poids des composants du score
            "weights": {
                "atr": 0.3,
                "roc": 0.3,
                "volume": 0.15,
                "trend": 0.25,  # Nouveau: bonus pour tendance directionnelle
            },
            # Paramètres de calcul
            "atr_period": 14,
            "roc_period_fast": 15,
            "roc_period_slow": 30,
            "volume_period": 20,
            # Seuils de range
            "bb_squeeze_threshold": 0.02,  # Bollinger Band squeeze %
            "adx_range_threshold": 20,  # ADX < 20 = ranging
            # Hard risk (forçage de sortie)
            "hard_risk": {
                "enabled": True,
                "atr_spike_threshold": 0.5,  # % ATR
                "spread_multiplier": 3,  # x median spread
                "slippage_multiplier": 2,  # x baseline slippage
            },
        }

    def update_market_data(self, symbol: str, data: dict) -> None:
        """Met à jour les données de marché pour une paire"""
        self.market_data_cache[symbol] = {**data, "timestamp": datetime.now(tz=timezone.utc)}

    def calculate_score(self, symbol: str) -> PairScore:
        """Calcule le score de tradeabilité d'une paire en utilisant les données de la DB"""
        try:
            # Utiliser la DB si disponible (priorité car données déjà
            # calculées)
            if self.db_pool:
                return self._calculate_score_from_db(symbol)
            # Fallback sur Redis/calculs manuels si pas de DB
            return self._calculate_score_from_redis(symbol)

        except Exception:
            logger.exception("Erreur calcul score {symbol}")
            return PairScore(symbol, -1.0, 0, 0, 0, True, datetime.now(tz=timezone.utc))

    def _calculate_score_from_db(self, symbol: str) -> PairScore:
        """Calcule le score depuis analyzer_data (tout est déjà calculé)"""
        try:
            from shared.src.db_pool import real_dict_cursor

            with real_dict_cursor() as cursor:
                # Récupérer toutes les données calculées depuis analyzer_data
                query = """
                    SELECT * FROM analyzer_data
                    WHERE symbol = %s AND timeframe = '3m'
                    ORDER BY time DESC LIMIT 1
                """
                cursor.execute(query, (symbol,))
                data = cursor.fetchone()

                if not data:
                    logger.warning(f"Pas de données analyzer pour {symbol}")
                    return PairScore(
                        symbol, -1.0, 0, 0, 0, True, datetime.now(tz=timezone.utc))

                # Extraire métriques (tout est déjà calculé !)
                atr_pct = float(data["natr"]) if data["natr"] else 0
                roc = (
                    (float(data["roc_10"]) + float(data["roc_20"])) / 2
                    if data["roc_10"] and data["roc_20"]
                    else 0
                )
                volume_ratio = (
                    float(
                        data["volume_ratio"]) if data["volume_ratio"] else 1.0)
                is_ranging = (
                    data["bb_squeeze"] if data["bb_squeeze"] is not None else False)

                # Calculer trend_score depuis données DB
                trend_score = self._calculate_trend_score_from_db(data)

                # Population pour z-scores
                all_scores = self._get_all_scores_from_db()

                # Z-scores (normalisation relative)
                if len(all_scores) < 3:
                    # Normalisation simple si pas assez de population
                    z_atr = min(1.0, atr_pct / 2.0) if atr_pct > 0 else 0
                    z_roc = min(1.0, abs(roc) / 5.0) * (1 if roc > 0 else -1)
                    z_volume = min(2.0, volume_ratio) - 1.0
                else:
                    z_atr = self._calculate_zscore(
                        atr_pct, [s.atr_pct for s in all_scores]
                    )
                    z_roc = self._calculate_zscore(
                        roc, [s.roc for s in all_scores])
                    z_volume = self._calculate_zscore(
                        volume_ratio, [s.volume_ratio for s in all_scores]
                    )

                # Score final pondéré
                score = (
                    z_atr * self.config["weights"]["atr"]
                    + z_roc * self.config["weights"]["roc"]
                    + z_volume * self.config["weights"]["volume"]
                    + trend_score * self.config["weights"]["trend"]
                )

                # Pénalité ranging (utiliser ADX de la DB)
                if is_ranging:
                    adx = float(data["adx_14"]) if data["adx_14"] else 0
                    range_penalty = self._calculate_range_penalty(adx, atr_pct)
                    score -= range_penalty

                # Bonus DB
                if (
                    data.get("confluence_score")
                    and float(data["confluence_score"]) > 70
                ):
                    score += 0.3
                if data.get("volume_context") in ["BREAKOUT", "PUMP_START"]:
                    score += 0.2

                return PairScore(
                    symbol=symbol,
                    score=score,
                    atr_pct=atr_pct,
                    roc=roc,
                    volume_ratio=volume_ratio,
                    is_ranging=is_ranging,
                    timestamp=datetime.now(tz=timezone.utc),
                    components={
                        "z_atr": z_atr,
                        "z_roc": z_roc,
                        "z_volume": z_volume,
                        "trend_bias": trend_score,
                        "market_regime": data.get("market_regime", "UNKNOWN"),
                        "confluence_score": (
                            float(data["confluence_score"])
                            if data.get("confluence_score")
                            else 0
                        ),
                    },
                )

        except Exception:
            logger.exception("Erreur récupération données DB pour {symbol}")
            return self._calculate_score_from_redis(symbol)

    def _get_all_scores_from_db(self) -> list[PairScore]:
        """Récupère les scores récents de toutes les paires depuis la DB"""
        scores = []

        try:
            from shared.src.db_pool import real_dict_cursor

            with real_dict_cursor() as cursor:
                # Récupérer les données de toutes les paires des 15 dernières
                # minutes
                query = """
                    SELECT DISTINCT ON (a.symbol)
                        a.symbol,
                        a.natr,
                        a.roc_10,
                        a.roc_20,
                        a.volume_ratio,
                        a.bb_squeeze,
                        a.adx_14,
                        a.trend_strength,
                        a.market_regime
                    FROM analyzer_data a
                    WHERE a.timeframe = '3m'
                    AND a.time > NOW() - INTERVAL '15 minutes'
                    ORDER BY a.symbol, a.time DESC
                """
                cursor.execute(query)
                results = cursor.fetchall()

                for data in results:
                    atr_pct = float(data["natr"]) if data["natr"] else 0
                    roc = (
                        (float(data["roc_10"]) + float(data["roc_20"])) / 2
                        if data["roc_10"] and data["roc_20"]
                        else 0
                    )
                    volume_ratio = (
                        float(
                            data["volume_ratio"]) if data["volume_ratio"] else 1.0)
                    is_ranging = (
                        data["bb_squeeze"] if data["bb_squeeze"] is not None else False)

                    # Créer un PairScore basique pour la normalisation
                    score = PairScore(
                        symbol=data["symbol"],
                        score=0.0,  # Sera calculé plus tard
                        atr_pct=atr_pct,
                        roc=roc,
                        volume_ratio=volume_ratio,
                        is_ranging=is_ranging,
                        timestamp=datetime.now(tz=timezone.utc),
                    )
                    scores.append(score)

        except Exception as e:
            logger.debug(f"Erreur récupération population DB: {e}")
            # Fallback sur cache Redis si erreur DB
            scores = self._get_all_recent_scores()

        return scores

    def _calculate_score_from_redis(self, symbol: str) -> PairScore:
        """Fallback minimal si pas de DB (ne devrait jamais arriver)"""
        logger.error(
            f"Pas de données DB disponibles pour {symbol} - retour score par défaut"
        )
        return PairScore(symbol, -1.0, 0, 0, 0, True, datetime.now(tz=timezone.utc))

    def _calculate_zscore(
            self,
            value: float,
            population: list[float]) -> float:
        """Calcule le z-score d'une valeur par rapport à la population"""
        if len(population) < 2:
            return 0.0

        mean = np.mean(population)
        std = np.std(population, ddof=1)  # Sample standard deviation

        if std == 0:
            return 0.0

        return float((value - mean) / std)

    def _calculate_range_penalty(self, adx: float, atr_pct: float) -> float:
        """Calcule la pénalité ranging basée sur ADX et ATR"""
        base_penalty = 0.15

        # Multiplicateur ADX
        if adx < 10:
            adx_multiplier = 2.0
        elif adx < 15:
            adx_multiplier = 1.5
        elif adx < 20:
            adx_multiplier = 1.0
        else:
            adx_multiplier = 0.5

        # Ajustement ATR
        if atr_pct < 0.5:
            atr_adjustment = 1.5
        elif atr_pct > 2.0:
            atr_adjustment = 0.7
        else:
            atr_adjustment = 1.0

        return min(base_penalty * adx_multiplier * atr_adjustment, 0.4)

    def _get_forced_pairs(self) -> list[str]:
        """Récupère les paires forcées (consensus fort) non expirées"""
        forced_pairs = []
        current_time = time.time()

        try:
            # Chercher toutes les clés forced_pair:*
            forced_keys = []
            # Redis scan pattern pour trouver les clés - utiliser redis
            # directement
            for key in self.redis.redis.scan_iter(match="forced_pair:*"):
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                forced_keys.append(key)

            for key in forced_keys:
                try:
                    forced_until_str = self.redis.get(key)
                    if forced_until_str:
                        forced_until = float(forced_until_str)
                        if current_time < forced_until:
                            # Extraire le symbole de la clé
                            symbol = key.replace("forced_pair:", "")
                            forced_pairs.append(symbol)
                        else:
                            # Expirer la clé manuellement
                            self.redis.delete(key)
                            logger.debug(f"Forçage expiré supprimé: {key}")
                except Exception:
                    logger.exception("Erreur traitement forçage {key}")

        except Exception:
            logger.exception("Erreur récupération paires forcées")

        return forced_pairs

    def _batch_calculate_scores_from_db(
        self, symbols: list[str]
    ) -> dict[str, PairScore]:
        """Calcule tous les scores depuis analyzer_data en batch"""
        scores = {}

        try:
            from shared.src.db_pool import real_dict_cursor

            with real_dict_cursor() as cursor:
                # Récupérer toutes les données calculées depuis analyzer_data
                query = """
                    SELECT DISTINCT ON (symbol)
                        symbol, natr, roc_10, roc_20, volume_ratio, bb_squeeze,
                        adx_14, trend_strength, trend_angle, directional_bias,
                        market_regime, confluence_score, volume_context
                    FROM analyzer_data
                    WHERE symbol = ANY(%s) AND timeframe = '3m'
                    AND time > NOW() - INTERVAL '15 minutes'
                    ORDER BY symbol, time DESC
                """
                cursor.execute(query, (symbols,))
                results = cursor.fetchall()

                # Extraire toutes les métriques pour z-scores
                all_atr = [float(r["natr"]) if r["natr"]
                           else 0 for r in results]
                all_roc = [
                    (
                        (float(r["roc_10"]) + float(r["roc_20"])) / 2
                        if r["roc_10"] and r["roc_20"]
                        else 0
                    )
                    for r in results
                ]
                all_volume = [
                    float(r["volume_ratio"]) if r["volume_ratio"] else 1.0
                    for r in results
                ]

                # Traiter chaque résultat
                for i, data in enumerate(results):
                    symbol = data["symbol"]
                    atr_pct = all_atr[i]
                    roc = all_roc[i]
                    volume_ratio = all_volume[i]
                    is_ranging = (
                        data["bb_squeeze"] if data["bb_squeeze"] is not None else False)

                    # Z-scores
                    z_atr = self._calculate_zscore(atr_pct, all_atr)
                    z_roc = self._calculate_zscore(roc, all_roc)
                    z_volume = self._calculate_zscore(volume_ratio, all_volume)

                    # Trend score et pénalités
                    trend_score = self._calculate_trend_score_from_db(data)

                    # Score final
                    score = (
                        z_atr * self.config["weights"]["atr"]
                        + z_roc * self.config["weights"]["roc"]
                        + z_volume * self.config["weights"]["volume"]
                        + trend_score * self.config["weights"]["trend"]
                    )

                    # Pénalité ranging
                    if is_ranging:
                        adx = float(data["adx_14"]) if data["adx_14"] else 0
                        score -= self._calculate_range_penalty(adx, atr_pct)

                    # Bonus
                    if (
                        data.get("confluence_score")
                        and float(data["confluence_score"]) > 70
                    ):
                        score += 0.3
                    if data.get("volume_context") in [
                            "BREAKOUT", "PUMP_START"]:
                        score += 0.2

                    scores[symbol] = PairScore(
                        symbol=symbol,
                        score=score,
                        atr_pct=atr_pct,
                        roc=roc,
                        volume_ratio=volume_ratio,
                        is_ranging=is_ranging,
                        timestamp=datetime.now(tz=timezone.utc),
                        components={
                            "z_atr": z_atr,
                            "z_roc": z_roc,
                            "z_volume": z_volume,
                            "trend_bias": trend_score,
                            "market_regime": data.get(
                                "market_regime",
                                "UNKNOWN"),
                        },
                    )

                    # Cache
                    self.market_data_cache[symbol] = {
                        "score": scores[symbol],
                        "timestamp": datetime.now(tz=timezone.utc),
                    }

                # Symboles manquants = score par défaut
                for symbol in symbols:
                    if symbol not in scores:
                        scores[symbol] = PairScore(
                            symbol, -1.0, 0, 0, 0, True, datetime.now(tz=timezone.utc)
                        )

        except Exception:
            logger.exception("Erreur batch calculate scores")
            # Fallback : score par défaut pour tous
            for symbol in symbols:
                scores[symbol] = PairScore(
                    symbol, -1.0, 0, 0, 0, True, datetime.now(tz=timezone.utc))

        return scores

    def _calculate_trend_score_from_db(self, data: dict) -> float:
        """Calcule le score de tendance depuis les données DB existantes"""
        trend_score = 0.0

        # ADX (déjà calculé dans analyzer_data)
        adx = float(data["adx_14"]) if data["adx_14"] else 0
        if adx > 25:
            trend_score += 0.8
        elif adx > 20:
            trend_score += 0.4

        # Trend strength
        if data["trend_strength"] == "VERY_STRONG":
            trend_score += 0.5
        elif data["trend_strength"] == "STRONG":
            trend_score += 0.3
        elif data["trend_strength"] == "MODERATE":
            trend_score += 0.15

        # Trend angle (pente de la tendance)
        if data.get("trend_angle"):
            angle = float(data["trend_angle"])
            if abs(angle) > 0.5:
                trend_score += 0.3 * np.sign(angle)
            elif abs(angle) > 0.2:
                trend_score += 0.15 * np.sign(angle)

        # Directional bias
        if data.get("directional_bias") == "BULLISH":
            trend_score += 0.2
        elif data.get("directional_bias") == "BEARISH":
            trend_score -= 0.2

        # Market regime (avec gestion NULL)
        market_regime = data.get("market_regime")
        if market_regime in ["TRENDING_BULL", "BREAKOUT_BULL"]:
            trend_score += 0.3
        elif market_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]:
            trend_score -= 0.3
        elif market_regime == "TRANSITION":
            trend_score += 0.1  # Léger bonus pour transition
        elif market_regime == "RANGING":
            trend_score *= 0.5  # Atténuer plutôt qu'annuler

        return max(-1, min(1, trend_score))

    def _get_all_recent_scores(self) -> list[PairScore]:
        """Récupère les scores récents de toutes les paires"""
        scores: list[PairScore] = []

        # Récupérer la liste des symboles depuis Redis
        symbols_data = self.redis.get("trading:symbols")
        if not symbols_data:
            return scores

        symbols = json.loads(symbols_data)

        for symbol in symbols:
            if symbol in self.market_data_cache:
                cache_entry = self.market_data_cache[symbol]
                if (datetime.now(tz=timezone.utc) - cache_entry["timestamp"]).seconds < 180:
                    score = cache_entry.get("score")
                    if isinstance(score, PairScore):
                        scores.append(score)

        return scores

    def apply_hysteresis(self, symbol: str, current_score: float) -> bool:
        """Applique l'hystérésis pour éviter le ping-pong"""
        if symbol not in self.pair_states:
            self.pair_states[symbol] = PairState(
                symbol=symbol, is_selected=False)

        state = self.pair_states[symbol]

        # Ajouter le score à l'historique
        state.score_history.append(current_score)
        if len(state.score_history) > 10:
            state.score_history.pop(0)

        # Vérifier cool-down
        if state.cooldown_until and datetime.now(tz=timezone.utc) < state.cooldown_until:
            return False

        # Logique d'hystérésis
        if not state.is_selected:
            # Pour entrer : score > seuil pendant N périodes
            if current_score > self.config["score_threshold_enter"]:
                state.consecutive_above_threshold += 1
                state.consecutive_below_threshold = 0

                if (
                    state.consecutive_above_threshold
                    >= self.config["periods_above_to_enter"]
                ):
                    state.is_selected = True
                    state.last_selected = datetime.now(tz=timezone.utc)
                    state.consecutive_above_threshold = 0
                    return True
            else:
                state.consecutive_above_threshold = 0
        # Pour sortir : score < seuil pendant M périodes
        elif current_score < self.config["score_threshold_exit"]:
            state.consecutive_below_threshold += 1
            state.consecutive_above_threshold = 0

            if (
                state.consecutive_below_threshold
                >= self.config["periods_below_to_exit"]
            ):
                state.is_selected = False
                state.last_deselected = datetime.now(tz=timezone.utc)
                state.cooldown_until = datetime.now(tz=timezone.utc) + timedelta(
                    minutes=self.config["cooldown_minutes"]
                )
                state.consecutive_below_threshold = 0
                return False
        else:
            state.consecutive_below_threshold = 0

        return state.is_selected

    def update_universe(self) -> tuple[set[str], dict[str, PairScore]]:
        """Met à jour l'univers tradable - TOUS LES SYMBOLES EN PERMANENCE"""
        try:
            # Récupérer tous les symboles
            symbols_data = self.redis.get("trading:symbols")
            if not symbols_data:
                logger.warning("Pas de symboles configurés")
                return set(), {}

            # symbols_data peut être une string JSON ou déjà une liste
            if isinstance(symbols_data, str):
                all_symbols = json.loads(symbols_data)
            else:
                all_symbols = symbols_data

            # Calculer les scores depuis analyzer_data (pour le monitoring
            # seulement)
            scores = {}
            if self.db_pool:
                scores = self._batch_calculate_scores_from_db(all_symbols)

            # TOUS LES SYMBOLES SONT SÉLECTIONNÉS EN PERMANENCE
            selected = set(all_symbols)

            # Enregistrer l'état actuel avant modification
            previous_universe = self.selected_universe.copy()

            # Mettre à jour l'univers
            self.selected_universe = selected
            self.last_update = datetime.now(tz=timezone.utc)

            # Publier dans Redis
            self.redis.set(
                "universe:selected",
                json.dumps(list(selected)),
                expiration=180,  # TTL 3 minutes
            )

            # Log les changements
            added = selected - previous_universe
            removed = previous_universe - selected

            if added:
                logger.info(f"✅ Symboles ajoutés à l'univers: {added}")
            if removed:
                logger.info(f"❌ Symboles retirés de l'univers: {removed}")

            logger.info(
                f"📊 Univers complet: {len(selected)} symboles actifs en permanence"
            )

            return selected, scores

        except Exception:
            logger.exception("Erreur update_universe")
            return set(), {}

    def is_pair_tradable(self, symbol: str) -> bool:
        """Vérifie si une paire peut ouvrir de nouvelles positions"""
        return symbol in self.selected_universe

    def check_hard_risk(self, symbol: str) -> bool:
        """Vérifie les conditions de risque extrême"""
        if not self.config["hard_risk"]["enabled"]:
            return False

        try:
            # Récupérer les métriques de risque depuis Redis
            risk_key = f"risk:{symbol}"
            risk_data = self.redis.get(risk_key)

            if not risk_data:
                return False

            risk = json.loads(risk_data)

            # Vérifier les conditions
            atr_spike = (
                risk.get("atr_spike", 0)
                > self.config["hard_risk"]["atr_spike_threshold"]
            )
            spread_high = (
                risk.get("spread_ratio", 0)
                > self.config["hard_risk"]["spread_multiplier"]
            )
            slippage_high = (
                risk.get("slippage_ratio", 0)
                > self.config["hard_risk"]["slippage_multiplier"]
            )

            # Toutes les conditions doivent être vraies
            return atr_spike and spread_high and slippage_high

        except Exception:
            logger.exception("Erreur check_hard_risk {symbol}")
            return False

    def get_universe_stats(self) -> dict:
        """Retourne les statistiques de l'univers"""
        stats = {
            "selected_count": len(self.selected_universe),
            "selected_symbols": list(self.selected_universe),
            "core_pairs": list(self.core_pairs),
            "satellites": list(self.selected_universe - self.core_pairs),
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "pair_states": {},
        }

        pair_states_dict: dict[str, dict] = {}
        for symbol, state in self.pair_states.items():
            pair_states_dict[symbol] = {
                "is_selected": state.is_selected,
                "recent_scores": (
                    state.score_history[-5:] if state.score_history else []
                ),
                "cooldown_until": (
                    state.cooldown_until.isoformat() if state.cooldown_until else None
                ),
            }

        stats["pair_states"] = pair_states_dict

        return stats

    def force_pair_selection(
            self,
            symbol: str,
            duration_minutes: int = 60) -> None:
        """Force la sélection d'une paire pour consensus fort"""
        # Stocker le forçage avec expiration
        forced_until = time.time() + (duration_minutes * 60)
        forced_key = f"forced_pair:{symbol}"
        self.redis.set(
            forced_key,
            str(forced_until),
            expiration=duration_minutes *
            60 +
            60)  # +60s marge

        # Ajouter à l'univers actuel
        self.selected_universe.add(symbol)
        logger.info(
            f"Paire {symbol} forcée dans l'univers pour {duration_minutes} minutes (jusqu'à {time.strftime('%H:%M:%S', time.localtime(forced_until))})"
        )

        # Publier dans Redis avec TTL cohérent
        self.redis.set(
            "universe:selected",
            json.dumps(list(self.selected_universe)),
            expiration=max(
                300, duration_minutes * 60
            ),  # Au moins 5min, ou la durée demandée
        )
