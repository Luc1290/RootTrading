"""
Module de gestion du contexte de marché pour la validation des signaux.
Récupère et structure les données nécessaires depuis la base de données.
"""

import logging
from typing import Dict, Any, Optional, List
import psycopg2
from psycopg2.extras import RealDictCursor
from .field_converters import FieldConverter  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class ContextManager:
    """Gestionnaire du contexte de marché pour la validation."""

    def __init__(self, db_connection):
        """
        Initialise le gestionnaire de contexte.

        Args:
            db_connection: Connexion à la base de données ou config DB
        """
        # Si c'est une connexion, on la garde pour compatibilité
        # Si c'est un dict de config, on créera des connexions à la volée
        if isinstance(db_connection, dict):
            self.db_config = db_connection
            self.db_connection = None
        else:
            self.db_connection = db_connection
            self.db_config: Optional[Dict[str, Any]] = None

        # Cache pour optimiser les requêtes répétées avec TTL dynamique
        self.context_cache = {}
        self.base_cache_ttl = 5  # TTL de base en secondes

    def get_unified_market_regime(
        self, symbol: str, signal_timeframe: str | None = None
    ) -> Dict[str, Any]:
        """
        Récupère le régime de marché unifié adapté au timeframe du signal.

        Args:
            symbol: Symbole à analyser
            signal_timeframe: Timeframe du signal pour adapter le régime

        Returns:
            Dict contenant le régime unifié et ses métadonnées
        """
        # Adapter le régime de référence au timeframe du signal
        if signal_timeframe:
            if signal_timeframe in ["1m", "3m"]:
                reference_timeframe = "15m"  # Court terme
            elif signal_timeframe in ["5m", "15m"]:
                reference_timeframe = "15m"  # Moyen terme
            else:
                reference_timeframe = "15m"  # Long terme
        else:
            reference_timeframe = "15m"  # Par défaut
        # cache_key = f"regime_unified_{symbol}"  # Non utilisé actuellement

        try:
            # Créer un curseur sans utiliser le context manager de transaction
            if self.db_connection is None:
                return {
                    "market_regime": "UNKNOWN",
                    "regime_strength": 0.5,
                    "regime_confidence": 50.0,
                    "directional_bias": "NEUTRAL",
                    "volatility_regime": "normal",
                    "regime_source": "error",
                    "regime_timestamp": None,
                }
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            try:
                # Récupérer le régime du timeframe de référence (plus récent)
                cursor.execute(
                    """
                    SELECT market_regime, regime_strength, regime_confidence,
                           directional_bias, volatility_regime, time
                    FROM analyzer_data
                    WHERE symbol = %s AND timeframe = %s
                    ORDER BY time DESC
                    LIMIT 1
                """,
                    (symbol, reference_timeframe),
                )

                regime_data = cursor.fetchone()
            finally:
                cursor.close()

            if regime_data:
                return {
                    "market_regime": regime_data["market_regime"],
                    "regime_strength": regime_data["regime_strength"],
                    "regime_confidence": regime_data["regime_confidence"],
                    "directional_bias": regime_data["directional_bias"],
                    "volatility_regime": regime_data["volatility_regime"],
                    "regime_source": f"{reference_timeframe}_unified",
                    "regime_timestamp": regime_data["time"],
                }
            else:
                # Fallback si pas de données 3m
                logger.warning(f"Pas de régime 3m pour {symbol}, fallback sur 1m")
                cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
                try:
                    cursor.execute(
                        """
                            SELECT market_regime, regime_strength, regime_confidence,
                                   directional_bias, volatility_regime, time
                            FROM analyzer_data
                            WHERE symbol = %s AND timeframe = '1m'
                            ORDER BY time DESC
                            LIMIT 1
                        """,
                        (symbol,),
                    )

                    fallback_data = cursor.fetchone()
                finally:
                    cursor.close()
                if fallback_data:
                    return {
                        "market_regime": fallback_data["market_regime"],
                        "regime_strength": fallback_data["regime_strength"],
                        "regime_confidence": fallback_data["regime_confidence"],
                        "directional_bias": fallback_data["directional_bias"],
                        "volatility_regime": fallback_data["volatility_regime"],
                        "regime_source": "1m_fallback",
                        "regime_timestamp": fallback_data["time"],
                    }

            return {
                "market_regime": "UNKNOWN",
                "regime_strength": 0.5,
                "regime_confidence": 50.0,
                "directional_bias": "NEUTRAL",
                "volatility_regime": "normal",
                "regime_source": "default",
                "regime_timestamp": None,
            }

        except Exception as e:
            logger.error(f"Erreur récupération régime unifié {symbol}: {e}")
            return {
                "market_regime": "UNKNOWN",
                "regime_strength": 0.5,
                "regime_confidence": 50.0,
                "directional_bias": "NEUTRAL",
                "volatility_regime": "normal",
                "regime_source": "error",
                "regime_timestamp": None,
            }

    def _get_dynamic_cache_ttl(self, timeframe: str) -> int:
        """Calcule TTL dynamique selon timeframe pour éviter contexte périmé."""
        if timeframe in ("1m", "3m"):
            return 5
        if timeframe == "5m":
            return 15
        if timeframe == "15m":
            return 30
        return 60  # défaut raisonnable au-delà

    def get_market_context(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Récupère le contexte de marché complet pour un symbole et timeframe.

        Args:
            symbol: Symbole à analyser (ex: BTCUSDC)
            timeframe: Timeframe à analyser (ex: 1m, 5m, 15m, 1h)

        Returns:
            Dict contenant tout le contexte nécessaire à la validation
        """
        cache_key = f"{symbol}_{timeframe}"

        # Vérifier le cache avec TTL dynamique
        import time

        current_time = time.time()
        dynamic_ttl = self._get_dynamic_cache_ttl(timeframe)

        if cache_key in self.context_cache:
            cached_context, cache_timestamp = self.context_cache[cache_key]
            if current_time - cache_timestamp < dynamic_ttl:
                return cached_context
            else:
                # Cache expiré, le supprimer
                del self.context_cache[cache_key]

        try:
            # Récupérer les composants du contexte
            ohlcv_data = self._get_ohlcv_data(symbol, timeframe)
            indicators = self._get_indicators(symbol, timeframe)
            market_structure = self._get_market_structure(symbol, timeframe)
            volume_profile = self._get_volume_profile(symbol, timeframe)
            multi_timeframe = self._get_multi_timeframe_context(symbol)
            correlation_data = self._get_correlation_context(symbol)

            # Récupérer les données HTF pour validation MTF stricte
            htf_data = self._get_htf_validation_data(symbol)

            # Récupérer le régime unifié (15m de référence)
            unified_regime = self.get_unified_market_regime(
                symbol, signal_timeframe=timeframe
            )

            # Construire le contexte avec champs racine pour compatibility
            context = {
                "symbol": symbol,
                "timeframe": timeframe,
                "ohlcv_data": ohlcv_data,
                "indicators": indicators,
                "market_structure": market_structure,
                "volume_profile": volume_profile,
                "multi_timeframe": multi_timeframe,
                "correlation_data": correlation_data,
                "unified_regime": unified_regime,  # Régime unifié pour tous les signaux
            }

            # Exposer les champs critiques au niveau racine pour compatibilité validators
            if indicators:
                # IMPORTANT: Préserver TOUS les régimes pour éviter la confusion
                original_regime = indicators.get("market_regime")
                original_regime_strength = indicators.get("regime_strength")
                original_regime_confidence = indicators.get("regime_confidence")
                original_directional_bias = indicators.get("directional_bias")
                original_volatility_regime = indicators.get("volatility_regime")

                context.update(indicators)  # Tous les indicateurs au niveau racine

                # Sauvegarder le régime original du timeframe
                if original_regime:
                    context["timeframe_regime"] = original_regime
                    context["timeframe_regime_strength"] = original_regime_strength if original_regime_strength is not None else 0.5
                    context["timeframe_regime_confidence"] = original_regime_confidence if original_regime_confidence is not None else 50.0
                    context["timeframe_directional_bias"] = original_directional_bias if original_directional_bias is not None else "NEUTRAL"
                    context["timeframe_volatility_regime"] = original_volatility_regime if original_volatility_regime is not None else "normal"

            # Ajouter les données HTF au contexte pour validation MTF
            if htf_data:
                context.update(htf_data)

            # NOUVELLE APPROCHE: Ajouter le régime unifié SANS ÉCRASER l'original
            # Les validateurs peuvent choisir quel régime utiliser
            if unified_regime:
                # Stocker le régime unifié dans des champs séparés
                context["unified_regime"] = unified_regime["market_regime"]
                context["unified_regime_strength"] = unified_regime["regime_strength"]
                context["unified_regime_confidence"] = unified_regime[
                    "regime_confidence"
                ]
                context["unified_directional_bias"] = unified_regime["directional_bias"]
                context["unified_volatility_regime"] = unified_regime[
                    "volatility_regime"
                ]
                context["regime_source"] = unified_regime["regime_source"]

                # PAR DÉFAUT: Utiliser le régime du timeframe SAUF si très faible confidence
                if (
                    original_regime_confidence
                    and float(original_regime_confidence) < 30
                ):
                    # Si le régime du timeframe a une confidence très faible, préférer l'unifié
                    context["market_regime"] = unified_regime["market_regime"]
                    context["regime_strength"] = unified_regime["regime_strength"]
                    context["regime_confidence"] = unified_regime["regime_confidence"]
                    context["directional_bias"] = unified_regime["directional_bias"]
                    context["volatility_regime"] = unified_regime["volatility_regime"]
                    context["regime_decision"] = "unified_used_low_confidence"
                else:
                    # Garder le régime original du timeframe par défaut
                    context["regime_decision"] = "timeframe_regime_kept"

            if market_structure and "current_price" in market_structure:
                context["current_price"] = market_structure[
                    "current_price"
                ]  # Prix actuel au niveau racine

            # Fallback pour current_price si absent - avec protection ohlcv vide
            if "current_price" not in context:
                if ohlcv_data and len(ohlcv_data) > 0:
                    context["current_price"] = float(ohlcv_data[-1]["close"])
                else:
                    logger.warning(
                        f"Pas de données OHLCV pour {symbol} {timeframe}, current_price = 0"
                    )
                    context["current_price"] = 0.0

            # Mise en cache avec timestamp
            self.context_cache[cache_key] = (context, current_time)

            return context

        except Exception as e:
            logger.error(f"Erreur récupération contexte {symbol} {timeframe}: {e}")
            return {}

    def _get_ohlcv_data(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> List[Dict[str, float]]:
        """
        Récupère les données OHLCV historiques.

        Args:
            symbol: Symbole
            timeframe: Timeframe
            limit: Nombre de bougies à récupérer

        Returns:
            Liste des données OHLCV
        """
        try:
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            try:
                cursor.execute(
                    """
                    SELECT time, open, high, low, close, volume, quote_asset_volume
                    FROM market_data
                    WHERE symbol = %s AND timeframe = %s
                    ORDER BY time DESC
                    LIMIT %s
                """,
                    (symbol, timeframe, limit),
                )

                rows = cursor.fetchall()
            finally:
                cursor.close()

            return [
                {
                    "timestamp": row["time"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "quote_volume": float(row["quote_asset_volume"]),
                }
                for row in reversed(rows)  # Ordre chronologique
            ]

        except Exception as e:
            logger.error(f"Erreur récupération OHLCV {symbol} {timeframe}: {e}")
            return []

    def _get_indicators(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Récupère tous les indicateurs techniques pré-calculés.

        Args:
            symbol: Symbole
            timeframe: Timeframe

        Returns:
            Dict des indicateurs avec conversion de type robuste
        """
        try:
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            try:
                cursor.execute(
                    """
                    SELECT * FROM analyzer_data
                    WHERE symbol = %s AND timeframe = %s
                    ORDER BY time DESC
                    LIMIT 1
                """,
                    (symbol, timeframe),
                )

                row = cursor.fetchone()
            finally:
                cursor.close()

            if not row:
                return {}

            # Conversion des indicateurs via FieldConverter
            raw_indicators = {}
            for key, value in row.items():
                if key not in [
                    "time",
                    "symbol",
                    "timeframe",
                    "analysis_timestamp",
                    "analyzer_version",
                ]:
                    raw_indicators[key] = value

            # Utiliser le convertisseur pour harmoniser les types
            indicators = FieldConverter.convert_indicators(raw_indicators)

            # FALLBACK pour volume_quality_score si manquant (critique pour filtres)
            if (
                not indicators.get("volume_quality_score")
                or indicators.get("volume_quality_score") == 0
            ):
                fallback_volume_quality = self._get_volume_quality_fallback(
                    symbol, timeframe
                )
                if fallback_volume_quality is not None:
                    indicators["volume_quality_score"] = fallback_volume_quality
                    logger.info(
                        f"✅ Volume quality fallback {symbol} {timeframe}: {fallback_volume_quality}"
                    )

            # ENRICHISSEMENT: Ajouter indicateurs calculés manquants
            self._enrich_indicators(indicators, symbol, timeframe, cursor)

            # Log temporaire pour debug
            logger.debug(
                f"Indicateurs récupérés pour {symbol} {timeframe}: {len(indicators)} champs"
            )
            if "atr_14" in indicators:
                logger.debug(f"ATR_14 trouvé: {indicators['atr_14']}")
            if "atr_percentile" in indicators:
                logger.debug(f"ATR_percentile trouvé: {indicators['atr_percentile']}")

            return indicators

        except Exception as e:
            logger.error(f"Erreur récupération indicateurs {symbol} {timeframe}: {e}")
            return {}

    def _get_market_structure(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyse la structure de marché (support/résistance, pivots, etc.).

        Args:
            symbol: Symbole
            timeframe: Timeframe

        Returns:
            Dict de la structure de marché
        """
        try:
            # Récupérer les données de prix récentes pour analyser la structure
            ohlcv_data = self._get_ohlcv_data(symbol, timeframe, 50)

            # PROTECTION: Vérifier que ohlcv_data n'est pas vide
            if not ohlcv_data or len(ohlcv_data) == 0:
                logger.warning(
                    f"Pas de données OHLCV pour structure {symbol} {timeframe}"
                )
                return {}

            # Extraction des prix pour analyse
            highs = [candle["high"] for candle in ohlcv_data]
            lows = [candle["low"] for candle in ohlcv_data]
            closes = [candle["close"] for candle in ohlcv_data]

            current_price = closes[-1] if closes else 0

            # Calculs de structure basiques
            recent_high = max(highs[-20:]) if len(highs) >= 20 else max(highs)
            recent_low = min(lows[-20:]) if len(lows) >= 20 else min(lows)

            structure: Dict[str, Any] = {
                "current_price": current_price,
                "recent_high": recent_high,
                "recent_low": recent_low,
                "price_range": recent_high - recent_low,
                "distance_to_high": recent_high - current_price,
                "distance_to_low": current_price - recent_low,
                "range_position": (
                    ((current_price - recent_low) / (recent_high - recent_low))
                    if recent_high != recent_low
                    else 0.5
                ),
            }

            # Détection de niveaux psychologiques simples
            psychological_levels = self._find_psychological_levels(current_price)
            structure["psychological_levels"] = psychological_levels

            return structure

        except Exception as e:
            logger.error(f"Erreur analyse structure marché {symbol} {timeframe}: {e}")
            return {}

    def _get_volume_profile(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyse le profil de volume.

        Args:
            symbol: Symbole
            timeframe: Timeframe

        Returns:
            Dict du profil de volume
        """
        try:
            ohlcv_data = self._get_ohlcv_data(symbol, timeframe, 50)

            # PROTECTION: Vérifier que ohlcv_data n'est pas vide
            if not ohlcv_data or len(ohlcv_data) == 0:
                logger.warning(
                    f"Pas de données OHLCV pour volume profile {symbol} {timeframe}"
                )
                return {}

            volumes = [candle["volume"] for candle in ohlcv_data]
            quote_volumes = [candle["quote_volume"] for candle in ohlcv_data]

            # Calculs de volume
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
            current_volume = volumes[-1] if volumes else 0

            volume_profile = {
                "current_volume": current_volume,
                "average_volume": avg_volume,
                "volume_ratio": current_volume / avg_volume if avg_volume > 0 else 1,
                "total_quote_volume": sum(quote_volumes),
                "volume_trend": self._calculate_volume_trend(volumes),
            }

            return volume_profile

        except Exception as e:
            logger.error(f"Erreur profil volume {symbol} {timeframe}: {e}")
            return {}

    def _get_multi_timeframe_context(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère le contexte multi-timeframe.

        Args:
            symbol: Symbole

        Returns:
            Dict du contexte multi-timeframe
        """
        try:
            timeframes = ["1m", "5m", "15m", "1h"]
            mtf_context = {}

            for tf in timeframes:
                indicators = self._get_indicators(symbol, tf)
                if indicators:
                    mtf_context[tf] = {
                        "trend_direction": indicators.get("directional_bias"),
                        "trend_strength": indicators.get("trend_strength"),
                        "momentum_score": indicators.get("momentum_score"),
                        "market_regime": indicators.get("market_regime"),
                    }

            return mtf_context

        except Exception as e:
            logger.error(f"Erreur contexte multi-timeframe {symbol}: {e}")
            return {}

    def _get_correlation_context(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère le contexte de corrélation avec d'autres actifs.
        DÉSACTIVÉ pour le moment car non implémenté.

        Args:
            symbol: Symbole

        Returns:
            Dict vide (fonctionnalité désactivée)
        """
        try:
            # DÉSACTIVÉ: Corrélations non implémentées, retourner dict vide
            # pour éviter de donner de fausses informations
            return {}

            # Ancien code commenté:
            # correlation_context = {
            #     'major_pairs_sentiment': 'neutral',
            #     'sector_correlation': 'neutral',
            #     'market_wide_sentiment': 'neutral'
            # }
            # return correlation_context

        except Exception as e:
            logger.error(f"Erreur contexte corrélation {symbol}: {e}")
            return {}

    def _find_psychological_levels(self, price: float) -> List[float]:
        """
        Trouve les niveaux psychologiques proches du prix actuel.

        Args:
            price: Prix actuel

        Returns:
            Liste des niveaux psychologiques
        """
        try:
            levels: List[float] = []

            # Niveaux ronds basés sur la magnitude du prix
            if price >= 1000:
                # Pour les prix élevés, utiliser des niveaux de 100
                base = float(int(price / 100) * 100)
                levels.extend([base, base + 100.0, base - 100.0])
            elif price >= 100:
                # Pour les prix moyens, utiliser des niveaux de 10
                base = float(int(price / 10) * 10)
                levels.extend([base, base + 10.0, base - 10.0])
            elif price >= 10:
                # Pour les prix bas, utiliser des niveaux de 1
                base = float(int(price))
                levels.extend([base, base + 1.0, base - 1.0])
            elif price >= 1:
                # Pour les prix autour de 1, utiliser des décimales
                base = round(price, 1)
                levels.extend([base, base + 0.1, base - 0.1])
            else:
                # MICRO-CAPS: Adaptation dynamique pour prix < 1 (PEPE, etc.)
                # Déterminer le nombre de décimales significatives
                import math

                if price > 0:
                    significant_digits = max(2, -int(math.floor(math.log10(price))) + 1)
                    base = round(price, significant_digits)
                    step = 10 ** (
                        -significant_digits + 1
                    )  # Un ordre de grandeur plus grand
                    levels.extend([base, base + step, base - step])
                else:
                    levels = [0.1]  # Fallback pour prix = 0

            # Filtrer les niveaux négatifs et trier
            levels = [level for level in levels if level > 0]
            levels.sort()

            return levels

        except Exception as e:
            logger.error(f"Erreur niveaux psychologiques: {e}")
            return []

    def _get_htf_validation_data(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère les données Higher TimeFrame pour validation MTF stricte.
        Inclut: EMAs 15m, ATR 15m et sa moyenne.

        Args:
            symbol: Symbole à analyser

        Returns:
            Dict avec données HTF pour validation
        """
        try:
            htf_data = {}

            # Récupérer données 15m (direction + ATR) - JOIN avec market_data pour le close
            cursor = self.db_connection.cursor()
            try:
                cursor.execute(
                    """
                    SELECT md.close, ad.ema_26, ad.ema_99, ad.atr_14
                    FROM analyzer_data ad
                    JOIN market_data md ON ad.time = md.time
                        AND ad.symbol = md.symbol
                        AND ad.timeframe = md.timeframe
                    WHERE ad.symbol = %s AND ad.timeframe = '15m'
                    ORDER BY ad.time DESC
                    LIMIT 1
                """,
                    (symbol,),
                )

                result_15m = cursor.fetchone()
                if result_15m:
                    htf_data["htf_close_15m"] = (
                        float(result_15m[0]) if result_15m[0] else None
                    )
                    htf_data["htf_ema20_15m"] = (
                        float(result_15m[1]) if result_15m[1] else None
                    )  # Utilise ema_26 comme proxy
                    htf_data["htf_ema100_15m"] = (
                        float(result_15m[2]) if result_15m[2] else None
                    )  # Utilise ema_99 comme proxy
                    htf_data["htf_atr_15m"] = (
                        float(result_15m[3]) if result_15m[3] else None
                    )  # ATR 15m pour reversal window

                # Récupérer ATR 15m actuel et historique (volatilité)
                cursor.execute(
                    """
                    SELECT atr_14
                    FROM analyzer_data
                    WHERE symbol = %s AND timeframe = '15m'
                      AND atr_14 IS NOT NULL
                    ORDER BY time DESC
                    LIMIT 50
                """,
                    (symbol,),
                )

                results_15m = cursor.fetchall()
                if results_15m and len(results_15m) >= 10:
                    # ATR actuel
                    htf_data["mtf_atr15m"] = (
                        float(results_15m[0][0]) if results_15m[0][0] else None
                    )

                    # Moyenne ATR historique
                    historical_atrs = [float(r[0]) for r in results_15m if r[0]]
                    if historical_atrs:
                        htf_data["mtf_atr15m_ma"] = sum(historical_atrs) / len(
                            historical_atrs
                        )

                        # Calculer ratio ATR 15m pour les filtres
                        atr15m_val = htf_data.get("mtf_atr15m")
                        atr15m_ma_val = htf_data.get("mtf_atr15m_ma")
                        if atr15m_val is not None and atr15m_ma_val is not None and atr15m_ma_val > 0:
                            htf_data["mtf_atr15m_ratio"] = float(atr15m_val) / float(atr15m_ma_val)

            finally:
                cursor.close()

            return htf_data

        except Exception as e:
            logger.error(f"Erreur récupération données HTF {symbol}: {e}")
            return {}

    def _calculate_volume_trend(self, volumes: List[float]) -> str:
        """
        Calcule la tendance du volume.

        Args:
            volumes: Liste des volumes

        Returns:
            Tendance du volume ('increasing', 'decreasing', 'stable')
        """
        try:
            if len(volumes) < 5:
                return "stable"

            # Comparer les 5 derniers volumes avec les 5 précédents
            recent_avg = sum(volumes[-5:]) / 5
            previous_avg = sum(volumes[-10:-5]) / 5

            if recent_avg > previous_avg * 1.2:
                return "increasing"
            elif recent_avg < previous_avg * 0.8:
                return "decreasing"
            else:
                return "stable"

        except Exception as e:
            logger.error(f"Erreur calcul tendance volume: {e}")
            return "stable"

    def _get_volume_quality_fallback(
        self, symbol: str, original_timeframe: str
    ) -> Optional[float]:
        """
        Fallback pour récupérer volume_quality_score depuis timeframes inférieurs.

        Ordre de priorité : 5m → 3m → 1m

        Args:
            symbol: Symbole
            original_timeframe: Timeframe original qui manque la donnée

        Returns:
            volume_quality_score du fallback ou None
        """
        # Définir l'ordre de fallback selon timeframe original
        fallback_sequence = {
            "15m": ["5m", "3m", "1m"],
            "1h": ["15m", "5m", "3m", "1m"],
            "4h": ["1h", "15m", "5m", "3m", "1m"],
            "5m": ["3m", "1m"],
            "3m": ["1m"],
            "1m": [],  # Pas de fallback pour 1m
        }

        timeframes_to_try = fallback_sequence.get(original_timeframe, ["3m", "1m"])

        try:
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            try:
                for fallback_tf in timeframes_to_try:
                    cursor.execute(
                        """
                        SELECT volume_quality_score, volume_ratio
                        FROM analyzer_data
                        WHERE symbol = %s AND timeframe = %s
                          AND volume_quality_score IS NOT NULL
                        ORDER BY time DESC
                        LIMIT 1
                    """,
                        (symbol, fallback_tf),
                    )

                    result = cursor.fetchone()
                    if result and result["volume_quality_score"] is not None:
                        vol_quality = float(result["volume_quality_score"])
                        logger.debug(
                            f"Volume quality fallback {symbol}: {original_timeframe} -> {fallback_tf} = {vol_quality}"
                        )
                        return vol_quality
            finally:
                cursor.close()

            # Si aucun fallback trouvé, calculer une estimation basique
            logger.warning(
                f"Aucun volume_quality_score trouvé pour {symbol}, estimation par défaut"
            )
            return self._estimate_volume_quality(symbol)

        except Exception as e:
            logger.error(f"Erreur fallback volume quality {symbol}: {e}")
            return None

    def _estimate_volume_quality(self, symbol: str) -> float:
        """
        Estime volume_quality_score basé sur volume_ratio récent.

        Args:
            symbol: Symbole

        Returns:
            Estimation de volume_quality_score (20-80)
        """
        try:
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            try:
                # Prendre volume_ratio récent de n'importe quel timeframe
                cursor.execute(
                    """
                    SELECT volume_ratio
                    FROM analyzer_data
                    WHERE symbol = %s AND volume_ratio IS NOT NULL
                    ORDER BY time DESC
                    LIMIT 5
                """,
                    (symbol,),
                )

                results = cursor.fetchall()
            finally:
                cursor.close()

            if results:
                vol_ratios = [
                    float(row["volume_ratio"]) for row in results if row["volume_ratio"]
                ]
                avg_vol_ratio = sum(vol_ratios) / len(vol_ratios)

                # Convertir volume_ratio en estimation volume_quality_score
                if avg_vol_ratio >= 2.0:
                    return 70.0  # Volume fort = qualité élevée
                elif avg_vol_ratio >= 1.5:
                    return 50.0  # Volume modéré = qualité moyenne
                elif avg_vol_ratio >= 1.0:
                    return 35.0  # Volume normal = qualité acceptable
                else:
                    return 22.0  # Volume faible = P10 réel (bas mais valide)

            # Fallback ultime basé sur P10 réel
            return 22.0  # P10 = 10% plus bas de la distribution réelle

        except Exception as e:
            logger.error(f"Erreur estimation volume quality {symbol}: {e}")
            return 22.0  # P10 réel, reste dans les bornes DB

    def _enrich_indicators(
        self, indicators: Dict[str, Any], symbol: str, timeframe: str, cursor
    ) -> None:
        """
        Enrichit les indicateurs avec des champs calculés manquants.

        Args:
            indicators: Dict des indicateurs à enrichir
            symbol: Symbole
            timeframe: Timeframe
            cursor: Curseur DB réutilisable
        """
        try:
            # 1. VOLATILITY_LEVEL basé sur volatility_regime
            volatility_regime = indicators.get("volatility_regime", "normal")
            if volatility_regime in ["extreme_chaos", "chaotic"]:
                indicators["volatility_level"] = "extreme"
            elif volatility_regime in ["high", "elevated"]:
                indicators["volatility_level"] = "high"
            elif volatility_regime in ["normal", "stable"]:
                indicators["volatility_level"] = "normal"
            else:
                indicators["volatility_level"] = "low"

            # 2. BARS_SINCE_EMA20_TOUCH_3M : Calcul optimisé si timeframe = 3m
            if timeframe == "3m":
                indicators["bars_since_ema20_touch_3m"] = (
                    self._calculate_bars_since_ema20_touch(symbol, cursor)
                )

            # 3. VOLUME_RATIO mapping depuis relative_volume
            if "relative_volume" in indicators and not indicators.get("volume_ratio"):
                indicators["volume_ratio"] = indicators["relative_volume"]

        except Exception as e:
            logger.error(f"Erreur enrichissement indicateurs {symbol} {timeframe}: {e}")

    def _calculate_bars_since_ema20_touch(self, symbol: str, cursor) -> Optional[int]:
        """
        Calcule le nombre de bougies depuis le dernier touch de l'EMA20 (3m).

        Args:
            symbol: Symbole
            cursor: Curseur DB

        Returns:
            Nombre de bougies ou None si pas trouvé
        """
        try:
            # Récupérer les 15 dernières bougies 3m avec prix et EMA26 (proxy EMA20)
            cursor.execute(
                """
                SELECT md.close, ad.ema_26, md.time
                FROM market_data md
                JOIN analyzer_data ad ON md.time = ad.time
                    AND md.symbol = ad.symbol
                    AND md.timeframe = ad.timeframe
                WHERE md.symbol = %s AND md.timeframe = '3m'
                    AND ad.ema_26 IS NOT NULL
                ORDER BY md.time DESC
                LIMIT 15
            """,
                (symbol,),
            )

            rows = cursor.fetchall()
            if not rows or len(rows) < 3:
                return None

            # Chercher la dernière fois où le prix était proche de l'EMA20 (±0.5%)
            for i, row in enumerate(rows):
                close_price = float(row[0]) if row[0] else 0
                ema20 = float(row[1]) if row[1] else 0

                if close_price > 0 and ema20 > 0:
                    distance_pct = abs(close_price - ema20) / close_price
                    if distance_pct <= 0.005:  # Touch = distance ≤ 0.5%
                        return i

            return 15  # Plus de 15 bougies depuis dernier touch

        except Exception as e:
            logger.error(f"Erreur calcul bars_since_ema20_touch {symbol}: {e}")
            return None

    def clear_cache(self):
        """Vide le cache du contexte."""
        self.context_cache.clear()
        logger.info("Cache contexte vidé")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques du cache.

        Returns:
            Dict des statistiques du cache
        """
        return {
            "cache_size": len(self.context_cache),
            "cached_symbols": list(
                set(key.split("_")[0] for key in self.context_cache.keys())
            ),
            "base_cache_ttl": self.base_cache_ttl,
            "cache_strategy": "dynamic_ttl_per_timeframe",
        }
