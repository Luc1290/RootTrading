"""
Filtres critiques essentiels pour le signal aggregator.
Remplace le système complexe de 23+ validators par 3-4 filtres vraiment utiles.
"""

import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class CriticalFilters:
    """
    Filtres critiques minimalistes pour éviter les signaux vraiment dangereux.
    Focus sur les risques majeurs seulement.
    """

    def __init__(self, db_connection=None):
        self.db_connection = db_connection

        # Seuils ATR dynamiques - calculés depuis l'univers actuel
        self.atr_universe_multiplier = 3.0  # 3x médiane univers = volatilité extrême
        self.fallback_max_atr_percent = 20.0  # Fallback 20% si calcul échoue (PEPE-friendly)
        self.extreme_bb_width_threshold = 0.12  # BB width > 12% = chaos

        # Filtres volume mort OPTIMISÉS SCALPING (exige plus de liquidité)
        self.min_volume_ratio = 0.30  # Volume < 30% moyenne = marché mort (durci pour scalping)
        self.min_volume_quality = 20   # Quality < 20% = données douteuses (durci pour scalping)

        # Filtres conflits multi-timeframe - ajustés selon analyse
        self.max_conflicting_directions = 0.7  # > 70% contradictoire = problème
        self.min_mtf_consistency = 0.4        # 40% minimum (évite les cas 60% sell vs 40% buy)

        # Filtre anomalies système (non utilisé actuellement)
        # self.max_data_staleness_minutes = 10  # Données > 10min = problème technique

        # ========== NOUVELLE VALIDATION MTF STRICTE ==========
        # Mode Shaolin : Peu de trades mais très propres (1-3/jour)
        self.strict_mtf_enabled = True  # Activer validation stricte

        # Paramètres filtre directionnel 15m (changé de 1h)
        self.htf_ema_fast = 20    # EMA rapide pour trend 15m
        self.htf_ema_slow = 100   # EMA lente pour trend 15m

        # Paramètres volatilité 15m
        self.mtf_atr_lookback = 50  # Période pour moyenne ATR
        self.mtf_min_atr_ratio = 1.0  # ATR doit être >= moyenne

        # Risk/Reward minimum
        self.min_risk_reward = 2.0  # R/R minimum exigé

        # Tolérance pullback configurable (en basis points)
        import os
        self.pullback_tolerance_bp = int(os.environ.get('STRICT_PULLBACK_BP', '25'))  # 25 bp par défaut (0.25%)
        self.pullback_tolerance = self.pullback_tolerance_bp / 10000  # Conversion en ratio
        
    def apply_critical_filters(self, signals: List[Dict[str, Any]],
                              context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Applique uniquement les filtres critiques vraiment nécessaires.

        Args:
            signals: Liste des signaux à valider
            context: Contexte de marché avec indicateurs

        Returns:
            Tuple (is_valid, reason_if_rejected)
        """
        if not signals:
            return False, "Aucun signal à valider"

        # ========== VALIDATION MTF ADAPTATIVE (INTELLIGENTE) ==========
        if self.strict_mtf_enabled:
            volatility_level = context.get('volatility_level', 'normal')
            market_regime = context.get('market_regime', 'UNKNOWN')
            consensus_strength = context.get('consensus_strength', 0.5)

            # Mode adaptatif selon conditions de marché
            # PROTECTION MAXIMALE en volatilité extrême ou régime très incertain
            apply_full_mtf = (
                volatility_level in ['extreme'] or
                (volatility_level == 'high' and market_regime in ['BREAKOUT_BEAR', 'VOLATILE']) or
                (market_regime == 'UNKNOWN' and consensus_strength < 0.6)
            )

            if apply_full_mtf:
                # MODE STRICT : Tous les filtres MTF (protection maximale)
                logger.info(f"🛡️ MODE MTF STRICT activé: vol={volatility_level}, regime={market_regime}")

                # FILTRE MTF 1: Direction 15m (filtre principal)
                htf_direction_check = self._check_htf_direction(signals, context)
                if not htf_direction_check[0]:
                    return False, f"DIRECTION 15M INVALIDE: {htf_direction_check[1]}"

                # FILTRE MTF 2: Volatilité 15m suffisante
                mtf_volatility_check = self._check_mtf_volatility(context)
                if not mtf_volatility_check[0]:
                    return False, f"VOLATILITÉ 15M INSUFFISANTE: {mtf_volatility_check[1]}"

                # FILTRE MTF 3: Timing pullback 3m
                ltf_timing_check = self._check_ltf_pullback_timing(signals, context)
                if not ltf_timing_check[0]:
                    return False, f"TIMING 3M INVALIDE: {ltf_timing_check[1]}"

                # FILTRE MTF 4: Risk/Reward minimum
                rr_check = self._check_min_risk_reward(context)
                if not rr_check[0]:
                    return False, f"RISK/REWARD INSUFFISANT: {rr_check[1]}"

            else:
                # MODE ALLÉGÉ : Filtres essentiels seulement (plus d'opportunités)
                logger.info(f"⚡ MODE MTF ALLÉGÉ activé: vol={volatility_level}, regime={market_regime}")

                # FILTRE ESSENTIEL 1: Direction 15m reste obligatoire
                htf_direction_check = self._check_htf_direction(signals, context)
                if not htf_direction_check[0]:
                    return False, f"DIRECTION 15M INVALIDE: {htf_direction_check[1]}"

                # FILTRE ESSENTIEL 2: Risk/Reward minimum
                rr_check = self._check_min_risk_reward(context)
                if not rr_check[0]:
                    return False, f"RISK/REWARD INSUFFISANT: {rr_check[1]}"

                # Skip volatilité 15m et timing pullback en marché calme/normal
                logger.info("✅ Filtres volatilité 15m et timing 3m SKIPPÉS (mode allégé)")

        # ========== FILTRES CLASSIQUES (SECONDAIRES) ==========
        # FILTRE 0: Régime de marché - PAS D'ACHAT EN TENDANCE BAISSIÈRE
        regime_check = self._check_market_regime_compatibility(signals, context)
        if not regime_check[0]:
            return False, f"RÉGIME INCOMPATIBLE: {regime_check[1]}"

        # FILTRE 1: Volatilité extrême dangereuse (avec seuil dynamique)
        volatility_check = self._check_extreme_volatility(context)
        if not volatility_check[0]:
            return False, f"VOLATILITÉ EXTRÊME: {volatility_check[1]}"

        # FILTRE 2: Volume insuffisant (marché mort)
        volume_check = self._check_volume_sufficiency(context)
        if not volume_check[0]:
            return False, f"VOLUME MORT: {volume_check[1]}"

        # FILTRE 3: Conflits multi-timeframe majeurs
        mtf_check = self._check_mtf_conflicts(signals, context)
        if not mtf_check[0]:
            return False, f"CONFLIT MTF MAJEUR: {mtf_check[1]}"

        # FILTRE 4: Anomalies techniques système
        technical_check = self._check_technical_anomalies(context)
        if not technical_check[0]:
            return False, f"ANOMALIE TECHNIQUE: {technical_check[1]}"

        # Tous les filtres critiques sont passés
        return True, "Tous les filtres critiques passés"
        
    def _check_market_regime_compatibility(self, signals: List[Dict[str, Any]], 
                                          context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Vérifie la compatibilité du signal avec le régime de marché.
        CRUCIAL: Pas d'achats en tendance baissière forte!
        """
        if not signals:
            return True, "Pas de signaux"
            
        # Déterminer la direction du signal
        signal_side = signals[0].get('side') if signals else None
        if not signal_side:
            return True, "Direction non définie"
            
        # AMÉLIORATION: Utiliser le bon régime selon la situation
        # Préférer le régime du timeframe sauf si confidence < 30%
        timeframe_confidence = context.get('timeframe_regime_confidence', 100)

        if context.get('timeframe_regime') and float(timeframe_confidence) >= 30:
            # Utiliser le régime du timeframe (plus précis pour le signal)
            market_regime = context.get('timeframe_regime', 'UNKNOWN')
            regime_strength_str = context.get('timeframe_regime_strength', 'weak')
            regime_confidence = context.get('timeframe_regime_confidence', 0.0)
            logger.debug(f"Utilisation régime TIMEFRAME: {market_regime} (conf: {regime_confidence})")
        else:
            # Fallback sur régime unifié ou standard
            market_regime = context.get('unified_regime', context.get('market_regime', 'UNKNOWN'))
            regime_strength_str = context.get('unified_regime_strength', context.get('regime_strength', 'weak'))
            regime_confidence = context.get('unified_regime_confidence', context.get('regime_confidence', 0.0))
            logger.debug(f"Utilisation régime UNIFIÉ/STANDARD: {market_regime} (conf: {regime_confidence})")
        
        # RÈGLES STRICTES MAIS PERMETTANT LES REBONDS
        if signal_side == 'BUY':
            # En régime baissier, permettre les rebonds avec conditions strictes
            if market_regime in ['TRENDING_BEAR', 'BREAKOUT_BEAR']:
                # Vérifier les signaux de rebond potentiel
                rsi_14 = context.get('rsi_14', 50)
                stoch_rsi = context.get('stoch_rsi', 50)
                williams_r = context.get('williams_r', -50)
                volume_ratio = context.get('volume_ratio', 1.0)
                
                # Conditions pour permettre un rebond (AJUSTÉ MODÉRÉMENT)
                oversold_conditions = 0
                strong_oversold = 0  # Conditions très oversold

                if rsi_14 and float(rsi_14) < 30:  # RSI oversold
                    oversold_conditions += 1
                    if float(rsi_14) < 25:  # RSI très oversold
                        strong_oversold += 1

                if stoch_rsi and float(stoch_rsi) < 20:  # StochRSI oversold
                    oversold_conditions += 1
                    if float(stoch_rsi) < 15:  # StochRSI très oversold
                        strong_oversold += 1

                if williams_r and float(williams_r) < -80:  # Williams %R oversold
                    oversold_conditions += 1
                    if float(williams_r) < -85:  # Williams %R très oversold
                        strong_oversold += 1

                if volume_ratio and float(volume_ratio) > 1.5:  # Volume spike (capitulation)
                    oversold_conditions += 1
                    if float(volume_ratio) > 2.0:  # Volume spike majeur
                        strong_oversold += 1

                # AJUSTEMENT PRUDENT:
                # - 2 conditions normales + 1 forte OU
                # - 3 conditions normales (comme avant)
                min_conditions = 2 if strong_oversold >= 1 else 3

                if oversold_conditions < min_conditions:
                    return False, f"Achat en {market_regime} rejeté: {oversold_conditions}/{min_conditions} conditions oversold (fort: {strong_oversold})"
                else:
                    # Signal de rebond potentiel accepté mais avec prudence
                    logger.info(f"⚠️ REBOND POTENTIEL détecté en {market_regime}: {oversold_conditions} conditions oversold (dont {strong_oversold} fortes)")
                
            # Prudence en régime inconnu avec indicateurs baissiers
            if market_regime == 'UNKNOWN':
                # Vérifier les indicateurs de tendance (strings en DB)
                trend_strength_str = context.get('trend_strength', 'neutral')
                directional_bias_str = context.get('directional_bias', 'neutral')
                momentum_score = context.get('momentum_score', 50)
                
                # Si indicateurs majoritairement baissiers, rejeter l'achat
                bearish_indicators = 0
                total_indicators = 0
                
                if trend_strength_str is not None:
                    total_indicators += 1
                    # trend_strength valeurs possibles: 'unknown', 'absent', 'weak', 'strong', 'very_strong', 'extreme'
                    trend_str = str(trend_strength_str).lower()
                    if trend_str in ['unknown', 'absent', 'weak', 'very_weak']:
                        bearish_indicators += 1
                        
                if directional_bias_str is not None:
                    total_indicators += 1
                    # directional_bias valeurs possibles: 'BULLISH', 'BEARISH', 'NEUTRAL' (majuscules)
                    bias_str = str(directional_bias_str).upper()
                    if bias_str == 'BEARISH':
                        bearish_indicators += 1
                        
                if momentum_score is not None:
                    total_indicators += 1
                    if float(momentum_score) < 40:
                        bearish_indicators += 1
                        
                # Si plus de 50% des indicateurs sont baissiers, rejeter
                if total_indicators > 0 and bearish_indicators / total_indicators > 0.5:
                    return False, f"Achat rejeté: {bearish_indicators}/{total_indicators} indicateurs baissiers en régime {market_regime}"
                    
            # Prudence en transition MAIS permettre si momentum positif
            if market_regime == 'TRANSITION':
                momentum_score = context.get('momentum_score', 50)
                if regime_confidence < 30 and float(momentum_score) < 55:
                    return False, f"Achat rejeté: Transition faible (conf {regime_confidence:.0f}%, momentum {momentum_score:.0f})"
                elif float(momentum_score) >= 55:
                    logger.info(f"✅ TRANSITION acceptée: momentum {momentum_score:.0f} > 55")
                
        elif signal_side == 'SELL':
            # Les ventes sont OK dans tous les régimes (protection du capital)
            # Mais on peut être plus sélectif en régime haussier fort
            # OPTIMISÉ SCALPING: Inclut BREAKOUT_BULL pour éviter de shorter les rallyes explosifs
            if market_regime in ['TRENDING_BULL', 'BREAKOUT_BULL'] and regime_confidence > 70:  # confidence en %
                # En bull fort ou breakout haussier, être très prudent avec les shorts
                momentum_score = context.get('momentum_score', 50)
                if momentum_score and float(momentum_score) > 70:
                    return False, f"Vente risquée: {market_regime} fort (confidence {regime_confidence:.0f}%) avec momentum {momentum_score:.0f}"
                    
        return True, "Régime compatible"
        
    def _check_extreme_volatility(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Vérifie si la volatilité n'est pas dangereusement élevée avec seuil dynamique."""
        try:
            # ATR dynamique basé sur l'univers
            natr = context.get('natr')  # Normalized ATR déjà en %
            if natr is not None:
                natr_value = float(natr)

                # Calculer seuil dynamique depuis l'univers
                dynamic_threshold = self._get_dynamic_atr_threshold()

                if natr_value > dynamic_threshold:
                    return False, f"NATR {natr_value:.1f}% > seuil dynamique {dynamic_threshold:.1f}%"
                    
            # BB width extrême - avec protection division par zéro
            bb_width = context.get('bb_width')
            bb_middle = context.get('bb_middle')

            if bb_width is not None and bb_middle is not None and bb_middle > 0:
                bb_width_percent = float(bb_width) / float(bb_middle)
                if bb_width_percent > self.extreme_bb_width_threshold:
                    return False, f"BB width {bb_width_percent:.2%} > {self.extreme_bb_width_threshold:.1%}"
            elif bb_middle is not None and bb_middle <= 0:
                return False, f"BB middle invalide: {bb_middle} (données corrompues)"
                    
            # Volatilité régime
            volatility_regime = context.get('volatility_regime')
            if volatility_regime in ['chaotic', 'extreme_chaos']:
                return False, f"Régime volatilité: {volatility_regime}"
                
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Erreur check volatilité: {e}")
            
        return True, "Volatilité acceptable"
        
    def _check_volume_sufficiency(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Vérifie que le volume n'est pas mort avec logs détaillés."""
        try:
            # Volume ratio vs moyenne
            volume_ratio = context.get('volume_ratio')
            relative_volume = context.get('relative_volume', volume_ratio)  # Fallback

            if volume_ratio is not None:
                vol_ratio = float(volume_ratio)
                if vol_ratio < self.min_volume_ratio:
                    return False, f"Volume mort: ratio {vol_ratio:.2f} < {self.min_volume_ratio} (relatif: {relative_volume:.2f})"

            # Volume quality score avec valeurs brutes
            volume_quality = context.get('volume_quality_score')
            avg_volume_20 = context.get('avg_volume_20')

            if volume_quality is not None:
                vol_quality = float(volume_quality)
                if vol_quality < self.min_volume_quality:
                    avg_vol_info = f", avg_20j: {avg_volume_20:.0f}" if avg_volume_20 else ""
                    return False, f"Volume quality {vol_quality:.0f}% < {self.min_volume_quality}%{avg_vol_info}"
                    
            # Pattern volume spécifique
            volume_pattern = context.get('volume_pattern')
            if volume_pattern in ['DEAD', 'STAGNANT', 'ERROR']:
                return False, f"Pattern volume problématique: {volume_pattern}"
                
        except (ValueError, TypeError) as e:
            logger.debug(f"Erreur check volume: {e}")
            
        return True, "Volume suffisant"
        
    def _check_mtf_conflicts(self, signals: List[Dict[str, Any]], 
                           context: Dict[str, Any]) -> Tuple[bool, str]:
        """Vérifie les conflits multi-timeframe majeurs."""
        try:
            # Si tous les signaux vont dans la même direction, pas de conflit majeur
            if not signals:
                return True, "Pas de signaux à vérifier"
                
            sides = [s.get('side') for s in signals if s.get('side')]
            if not sides:
                return True, "Pas de directions à vérifier"
                
            # Calculer le pourcentage de cohérence directionnelle
            buy_count = sides.count('BUY')
            sell_count = sides.count('SELL')
            total_count = len(sides)
            
            if total_count > 0:
                max_direction_ratio = max(buy_count, sell_count) / total_count
                if max_direction_ratio < self.min_mtf_consistency:
                    return False, f"Cohérence directionnelle {max_direction_ratio:.1%} < {self.min_mtf_consistency:.1%} ({buy_count}B/{sell_count}S sur {total_count})"
                    
            # Confluence score générale
            confluence_score = context.get('confluence_score')
            if confluence_score is not None:
                conf_score = float(confluence_score)
                # Seuil très bas pour filtre critique seulement
                if conf_score < 15:  # Seulement les cas vraiment chaotiques
                    return False, f"Confluence extrêmement faible {conf_score:.0f}"
                    
        except (ValueError, TypeError) as e:
            logger.debug(f"Erreur check MTF: {e}")
            
        return True, "Pas de conflit MTF majeur"
        
    def _check_technical_anomalies(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Vérifie les anomalies techniques système."""
        try:
            # Anomalie détectée par le système
            anomaly_detected = context.get('anomaly_detected')
            if anomaly_detected and str(anomaly_detected).lower() == 'true':
                # Anomalie n'est pas toujours mauvaise, mais vérifier gravité
                data_quality = context.get('data_quality')
                if data_quality in ['POOR', 'CORRUPTED', 'MISSING']:
                    return False, f"Anomalie + qualité données: {data_quality}"
                    
            # Market regime unknown prolongé
            market_regime = context.get('market_regime')
            regime_confidence = context.get('regime_confidence')
            
            if market_regime == 'UNKNOWN' and regime_confidence is not None:
                conf = float(regime_confidence)
                if conf < 10:  # Vraiment perdu
                    return False, f"Régime inconnu + confidence {conf:.0f}%"
                    
        except (ValueError, TypeError) as e:
            logger.debug(f"Erreur check techniques: {e}")
            
        return True, "Pas d'anomalie technique critique"

    def _get_dynamic_atr_threshold(self) -> float:
        """
        Calcule un seuil ATR dynamique basé sur la médiane de l'univers actuel.
        Évite de bloquer PEPE et autres cryptos naturellement volatiles.
        """
        try:
            if not self.db_connection:
                logger.warning("Pas de connexion DB disponible, utilisation fallback ATR")
                return self.fallback_max_atr_percent

            import psycopg2.extras

            with self.db_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Récupérer la médiane NATR des 20 dernières heures pour l'univers actuel
                query = """
                    SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY natr) as median_natr,
                           percentile_cont(0.9) WITHIN GROUP (ORDER BY natr) as p90_natr
                    FROM analyzer_data
                    WHERE natr IS NOT NULL
                        AND time >= NOW() - INTERVAL '20 hours'
                        AND timeframe = '3m'
                """
                cursor.execute(query)
                result = cursor.fetchone()

                if result and result['median_natr']:
                    median_natr = float(result['median_natr'])
                    p90_natr = float(result['p90_natr']) if result['p90_natr'] else median_natr * 5

                    # Seuil = médiane × multiplicateur, avec plancher raisonnable
                    dynamic_threshold = max(
                        median_natr * self.atr_universe_multiplier,
                        p90_natr * 1.2,  # Au moins 20% au-dessus du P90
                        5.0  # Plancher absolu 5% pour éviter de tout bloquer
                    )

                    logger.debug(f"ATR dynamique: médiane {median_natr:.2f}%, P90 {p90_natr:.2f}%, seuil {dynamic_threshold:.1f}%")
                    return dynamic_threshold

        except Exception as e:
            logger.warning(f"Erreur calcul seuil ATR dynamique: {e}")

        # Fallback sur valeur fixe (déjà en %)
        return self.fallback_max_atr_percent

    def _check_htf_direction(self, signals: List[Dict[str, Any]],
                            context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Vérifie que le signal est dans le sens de la tendance 15m.
        Utilise EMA20 et EMA100 sur le 15m pour déterminer la direction.
        """
        try:
            if not signals:
                return True, "Pas de signaux"

            signal_side = signals[0].get('side')
            if not signal_side:
                return True, "Direction non définie"

            # Utiliser les données du context (15m au lieu de 1h)
            close_15m = context.get('htf_close_15m')
            ema20_15m = context.get('htf_ema20_15m')  # Utilise ema_26 comme proxy pour ema_20
            ema100_15m = context.get('htf_ema100_15m')  # Utilise ema_99 comme proxy pour ema_100

            if not all([close_15m, ema20_15m, ema100_15m]):
                logger.warning("EMAs 15m manquantes dans le contexte")
                return True, "EMAs 15m manquantes"  # On laisse passer si pas de données

            close_15m = float(close_15m)
            ema20_15m = float(ema20_15m)
            ema100_15m = float(ema100_15m)

            # Déterminer la direction HTF (15m)
            htf_direction = None
            if close_15m > ema100_15m and ema20_15m > ema100_15m:
                htf_direction = 'BUY'  # Tendance haussière
            elif close_15m < ema100_15m and ema20_15m < ema100_15m:
                htf_direction = 'SELL'  # Tendance baissière
            else:
                # Zone neutre/transition - vérifier règle fail-safe LTF
                failsafe_valid, failsafe_reason = self._check_failsafe_ltf_alignment(signal_side, context)
                if failsafe_valid:
                    logger.info(f"✅ FAIL-SAFE LTF activée: HTF neutre mais LTF alignés - {failsafe_reason}")
                    return True, f"Fail-safe LTF: {failsafe_reason}"
                else:
                    return False, f"Zone neutre 15m rejetée: {failsafe_reason} (Close:{close_15m:.4f}, EMA20:{ema20_15m:.4f}, EMA100:{ema100_15m:.4f})"

            # Vérifier que le signal est dans le bon sens
            if signal_side != htf_direction:
                # 🚀 PRIORITÉ #1: Signal exceptionnel (fail-safe)
                failsafe_valid, failsafe_reason = self._check_failsafe_ltf_alignment(signal_side, context)
                if failsafe_valid:
                    logger.info(f"✅ FAIL-SAFE OVERRIDE: HTF {htf_direction} vs signal {signal_side} - {failsafe_reason}")
                    return True, f"Fail-safe override: {failsafe_reason}"

                # 🔄 PRIORITÉ #2: HTF REVERSAL WINDOW (breadth flip)
                if self._check_htf_reversal_window(signal_side, context, close_15m, ema20_15m, ema100_15m):
                    return True, "HTF reversal window: breadth flip détecté"

                # ❌ REJET: Aucune exception trouvée
                return False, f"Signal {signal_side} contre tendance 15m {htf_direction} (fail-safe: {failsafe_reason})"

            logger.info(f"✅ Direction HTF validée: {signal_side} aligné avec 15m {htf_direction}")
            return True, "Direction HTF valide"

        except Exception as e:
            logger.error(f"Erreur check HTF direction: {e}")
            return True, "Erreur validation HTF"  # On laisse passer en cas d'erreur

    def _check_mtf_volatility(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Vérifie que la volatilité 15m est suffisante pour trader.
        ATR(15m) doit être >= moyenne des 50 dernières bougies.
        """
        try:
            # Utiliser les données du context (injectées par context_manager)
            current_atr = context.get('mtf_atr15m')
            avg_atr = context.get('mtf_atr15m_ma')

            if not current_atr or not avg_atr:
                logger.warning("ATR 15m manquant dans le contexte")
                return True, "ATR 15m manquant"

            current_atr = float(current_atr)
            avg_atr = float(avg_atr)
            atr_ratio = current_atr / avg_atr if avg_atr > 0 else 0

            # Vérifier que volatilité est suffisante
            if atr_ratio < self.mtf_min_atr_ratio:
                return False, f"ATR 15m faible: {atr_ratio:.2f}x < {self.mtf_min_atr_ratio}x moyenne"

            logger.info(f"✅ Volatilité 15m OK: ATR ratio {atr_ratio:.2f}x")
            return True, "Volatilité 15m suffisante"

        except Exception as e:
            logger.error(f"Erreur check volatilité MTF: {e}")
            return True, "Erreur validation volatilité"

    def _check_ltf_pullback_timing(self, signals: List[Dict[str, Any]],
                                  context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Vérifie que l'entrée se fait sur un pullback 3m dans le bon sens.
        BUY: Prix touche EMA20(3m) tout en restant > EMA100(3m)
        SELL: Prix touche EMA20(3m) tout en restant < EMA100(3m)

        V3: Momentum Fast-Lane - Bypass timing si consensus ultra-fort + TRENDING_BULL
        """
        try:
            if not signals:
                return True, "Pas de signaux"

            signal_side = signals[0].get('side')
            if not signal_side:
                return True, "Direction non définie"

            # Récupération des métriques de consensus pour fast-lane
            consensus_strength = context.get('consensus_strength', 0.5)
            consensus_regime = context.get('consensus_regime', 'UNKNOWN')
            is_failsafe = context.get('is_failsafe_trade', False)
            wave_winner = context.get('wave_winner', False)

            # 🚀 MOMENTUM FAST-LANE: Bypass timing si conditions exceptionnelles
            if (consensus_strength >= 0.80 and
                consensus_regime in ['TRENDING_BULL', 'BREAKOUT_BULL'] and
                signal_side == 'BUY'):

                logger.info(f"🚀 MOMENTUM FAST-LANE activée: consensus {consensus_strength:.2f}, régime {consensus_regime}")
                return True, f"Momentum fast-lane: {consensus_strength:.2f} @ {consensus_regime}"

            # Utiliser les EMAs du contexte actuel (3m)
            current_price = context.get('current_price', 0)
            # Utiliser directement les EMAs disponibles en DB
            ema20_3m = context.get('ema_26')  # ema_26 comme proxy pour ema_20
            ema100_3m = context.get('ema_99')  # ema_99 comme proxy pour ema_100

            if not all([current_price, ema20_3m, ema100_3m]):
                logger.warning("EMAs 3m manquantes pour timing pullback")
                return True, "EMAs 3m manquantes"

            current_price = float(current_price)
            ema20_3m = float(ema20_3m)
            ema100_3m = float(ema100_3m)

            distance_to_ema20 = abs(current_price - ema20_3m)

            # A. ATR FLOOR DYNAMIQUE : Respiration basée sur volatilité 15m
            atr15m_ratio = context.get('mtf_atr15m_ratio', 1.0)  # ratio vs moyenne
            volatility_level = context.get('volatility_level', 'normal')  # low/normal/high/extreme

            # Boost volatilité selon niveau
            vol_boost_map = {'low': 1.0, 'normal': 1.1, 'high': 1.3, 'extreme': 1.5}
            vol_boost = vol_boost_map.get(volatility_level, 1.0)

            # ATR floor : tolérance minimale basée sur volatilité (0.30% - 0.75%)
            atr_floor_pct = min(0.0075, max(0.0030, 0.0060 * atr15m_ratio)) * vol_boost

            # Tolérance de base
            base_tolerance = self.pullback_tolerance

            # Assouplissement progressif pour signaux forts
            if consensus_strength >= 0.75:
                # Consensus fort: +100% de tolérance (25bp → 50bp)
                adaptive_tolerance = base_tolerance * 2.0
            elif consensus_strength >= 0.70:
                # Consensus correct: +50% de tolérance (25bp → 37.5bp)
                adaptive_tolerance = base_tolerance * 1.5
            elif consensus_strength >= 0.65:
                # Consensus acceptable: +25% de tolérance (25bp → 31.25bp)
                adaptive_tolerance = base_tolerance * 1.25
            else:
                adaptive_tolerance = base_tolerance

            # Bonus supplémentaire si trade fail-safe validé
            if is_failsafe:
                adaptive_tolerance *= 1.3  # +30% bonus fail-safe (augmenté)

            # Bonus wave winner (signal dominant de la vague)
            if wave_winner:
                adaptive_tolerance *= 1.2  # +20% bonus wave winner

            # Appliquer l'ATR floor (garantir respiration minimale)
            final_tolerance = max(adaptive_tolerance, atr_floor_pct)

            price_pct = current_price * final_tolerance
            tolerance_bp = int(final_tolerance * 10000)

            logger.info(f"📊 Tolérance pullback calculée: {tolerance_bp}bp (base:{base_tolerance*10000:.0f}bp → adaptive:{adaptive_tolerance*10000:.0f}bp → final:{tolerance_bp}bp, ATR floor:{atr_floor_pct*10000:.0f}bp)")

            # B. RECENT TOUCH : Vérifier si prix a touché EMA20 récemment
            bars_since_touch = context.get('bars_since_ema20_touch_3m')
            if bars_since_touch is not None and bars_since_touch <= 8:
                logger.info(f"📈 RECENT TOUCH: Prix a touché EMA20 il y a {bars_since_touch} bougies")
                return True, f"Momentum après touch EMA20 ({bars_since_touch} bars)"

            if signal_side == 'BUY':
                # Pour BUY: Prix proche EMA20 ET au-dessus EMA100
                if current_price < ema100_3m:
                    return False, f"Prix {current_price:.4f} < EMA100(3m) {ema100_3m:.4f}"

                # Vérifier pullback vers EMA20 (tolérance adaptative)
                if distance_to_ema20 > price_pct:
                    # C. MOMENTUM PROPRE : Bypass pullback si conditions exceptionnelles
                    momentum_bypass = self._check_momentum_bypass(
                        signal_side, context, consensus_strength, wave_winner,
                        distance_to_ema20, final_tolerance, current_price, ema100_3m
                    )
                    if momentum_bypass[0]:
                        return True, momentum_bypass[1]
                    else:
                        return False, f"TIMING 3M INVALIDE: Prix trop loin EMA20: {distance_to_ema20:.4f} > {price_pct:.4f} ({tolerance_bp}bp, consensus:{consensus_strength:.2f})"

            elif signal_side == 'SELL':
                # Pour SELL: Prix proche EMA20 ET en-dessous EMA100
                if current_price > ema100_3m:
                    return False, f"Prix {current_price:.4f} > EMA100(3m) {ema100_3m:.4f}"

                # Vérifier pullback vers EMA20 (tolérance adaptative)
                if distance_to_ema20 > price_pct:
                    return False, f"TIMING 3M INVALIDE: Prix trop loin EMA20: {distance_to_ema20:.4f} > {price_pct:.4f} ({tolerance_bp}bp, consensus:{consensus_strength:.2f})"

            # Mini-log détaillé pour audit
            entry_style = context.get('entry_style', 'pullback')
            bars_since_touch = context.get('bars_since_ema20_touch_3m', 'N/A')
            vol_level = context.get('volatility_level', 'normal')

            logger.info(f"✅ Timing pullback 3m validé: {signal_side} à {current_price:.4f} "
                       f"(tolerance={tolerance_bp}bp, atr15m_ratio={atr15m_ratio:.2f}, "
                       f"entry_style={entry_style}, bars_since_touch={bars_since_touch}, vol={vol_level})")

            return True, f"Timing pullback valide ({tolerance_bp}bp, {entry_style})"

        except Exception as e:
            logger.error(f"Erreur check pullback timing: {e}")
            return True, "Erreur validation timing"

    def _check_min_risk_reward(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Vérifie que le Risk/Reward estimé est suffisant.
        SL estimé = max(swing 3m, 0.7*ATR)
        TP estimé = 1.5*ATR
        """
        try:
            # Récupérer ATR (chercher dans plusieurs champs possibles)
            atr = context.get('atr_14') or context.get('mtf_atr15m') or context.get('atr')
            if not atr:
                logger.warning("ATR indisponible pour calcul R/R")
                return True, "ATR indisponible"  # On laisse passer si pas d'ATR

            atr = float(atr)

            # Utiliser la distance au support/résistance comme proxy pour swing distance
            current_price = context.get('current_price', 0)
            nearest_support = context.get('nearest_support')
            nearest_resistance = context.get('nearest_resistance')

            swing_distance = 0
            if current_price and nearest_support:
                # Pour un BUY, distance au support = potentiel SL
                swing_distance = abs(float(current_price) - float(nearest_support))
            elif current_price and nearest_resistance:
                # Pour un SELL, distance à la résistance = potentiel SL
                swing_distance = abs(float(current_price) - float(nearest_resistance))

            # Calculer SL et TP estimés
            sl_estimated = max(swing_distance, 0.7 * atr) if swing_distance > 0 else 0.7 * atr
            tp_estimated = 1.5 * atr

            # Vérifier que SL est valide et protéger division par zéro
            if sl_estimated <= 0:
                return False, f"SL estimé invalide: {sl_estimated:.4f}"

            # Calculer Risk/Reward avec protection division par zéro
            risk_reward = tp_estimated / sl_estimated if sl_estimated > 0 else 0

            # R/R dynamique selon régime et volatilité
            market_regime = context.get('market_regime', 'UNKNOWN')
            volatility_level = context.get('volatility_level', 'normal')

            if market_regime == 'TRENDING_BULL':
                if volatility_level in ['low', 'normal']:
                    min_rr = 1.8  # Plus souple en bull calme
                else:  # high/extreme
                    min_rr = 2.0  # Standard en bull volatil
            else:  # RANGING, BEAR, etc.
                if volatility_level in ['extreme']:
                    min_rr = 2.2  # Plus strict en volatilité extrême
                else:
                    min_rr = 2.0  # Standard

            logger.info(f"📊 Seuil R/R dynamique calculé: {min_rr:.1f} (régime: {market_regime}, volatilité: {volatility_level})")

            # Vérifier minimum R/R dynamique
            if risk_reward < min_rr:
                return False, f"R/R {risk_reward:.2f} < {min_rr:.1f} (regime:{market_regime}, vol:{volatility_level})"

            # Format adaptatif selon la taille des prix
            current_price = context.get('current_price', 0)
            if current_price > 10:
                # Prix > 10 : 2 décimales
                sl_fmt = f"{sl_estimated:.2f}"
                tp_fmt = f"{tp_estimated:.2f}"
            elif current_price > 1:
                # Prix 1-10 : 4 décimales
                sl_fmt = f"{sl_estimated:.4f}"
                tp_fmt = f"{tp_estimated:.4f}"
            else:
                # Prix < 1 : 6 décimales
                sl_fmt = f"{sl_estimated:.6f}"
                tp_fmt = f"{tp_estimated:.6f}"

            logger.info(f"✅ Risk/Reward OK: {risk_reward:.2f} (SL:{sl_fmt}, TP:{tp_fmt})")
            return True, f"R/R valide: {risk_reward:.2f}"

        except Exception as e:
            logger.error(f"Erreur check R/R: {e}")
            return True, "Erreur validation R/R"

    def get_filter_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de configuration des filtres."""
        quota_stats = self.get_quota_stats()

        return {
            'mode': 'STRICT_MTF' if self.strict_mtf_enabled else 'STANDARD',
            'atr_universe_multiplier': self.atr_universe_multiplier,
            'fallback_max_atr_percent': self.fallback_max_atr_percent,
            'min_volume_ratio': self.min_volume_ratio,
            'min_mtf_consistency': self.min_mtf_consistency,
            'strict_mtf_params': {
                'htf_timeframe': '15m',  # Mis à jour de 1h vers 15m
                'htf_ema_fast': self.htf_ema_fast,
                'htf_ema_slow': self.htf_ema_slow,
                'min_atr_ratio': self.mtf_min_atr_ratio,
                'min_risk_reward': self.min_risk_reward,
                'pullback_tolerance_bp': self.pullback_tolerance_bp
            },
            'trade_quota': quota_stats,
            'filters_count': 8 if self.strict_mtf_enabled else 4,
            'description': 'Mode Scalping: Validation MTF stricte 15m/3m + Quota 15min/symbole - Optimisé crypto'
        }

    def _check_failsafe_ltf_alignment(self, signal_side: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Règle fail-safe améliorée: autorise le trade quand HTF neutre SI signaux exceptionnels.
        V2: Plus intelligent et adaptatif selon la force du consensus.
        """
        try:
            # Récupérer les métriques consensus (injectées par signal_processor)
            consensus_strength = context.get('consensus_strength', 0.5)
            wave_winner = context.get('wave_winner', False)
            market_regime = context.get('market_regime', 'NEUTRAL')
            strategies_count = context.get('total_strategies', 0)

            # NOUVELLE LOGIQUE: Fail-safe progressive selon force du signal
            # Plus le consensus est fort, moins on est restrictif

            # 1. Vérifier signal exceptionnel (wave_winner + consensus fort)
            if not wave_winner:
                return False, "Fail-safe: nécessite wave_winner"

            # Seuil adaptatif selon volatilité
            volatility_level = context.get('volatility_level', 'normal')
            if volatility_level in ['low', 'normal']:
                min_consensus = 0.70  # Plus souple en vol faible/normale
            else:  # high/extreme
                min_consensus = 0.75  # Plus strict en vol élevée

            logger.info(f"📊 Seuil consensus fail-safe calculé: {min_consensus:.2f} (volatilité: {volatility_level})")

            if consensus_strength < min_consensus:
                return False, f"Fail-safe: consensus {consensus_strength:.2f} < {min_consensus:.2f} requis (vol:{volatility_level})"

            if strategies_count < 6:
                return False, f"Fail-safe: {strategies_count} stratégies < 6 requises"

            # 2. Vérifier cohérence directionnelle de base (assouplie pour rebonds)
            directional_bias = context.get('directional_bias', 'NEUTRAL')
            momentum_score = context.get('momentum_score', 50)
            consensus_regime = context.get('consensus_regime', market_regime)  # Utiliser consensus_regime si dispo

            # Seuils adaptatifs selon consensus strength
            momentum_threshold_buy = 55 if consensus_strength >= 0.80 else 52  # Assoupli
            momentum_threshold_sell = 45 if consensus_strength >= 0.80 else 48

            logger.info(f"📊 Seuils momentum calculés: BUY>{momentum_threshold_buy}, SELL<{momentum_threshold_sell} (consensus: {consensus_strength:.2f})")

            ltf_aligned = False
            if signal_side == 'BUY':
                # Logique assouplie pour rebonds: si momentum > seuil, accepter même avec bias BEARISH
                momentum_ok = momentum_score and float(momentum_score) > momentum_threshold_buy
                bias_ok = directional_bias in ['BULLISH', 'NEUTRAL']

                # CAS SPÉCIAL: Rebond avec consensus très fort (≥0.80) + momentum positif
                strong_rebound = (consensus_strength >= 0.80 and momentum_ok and
                                 float(momentum_score) > 50)  # Au-dessus de la neutralité

                ltf_aligned = bias_ok or strong_rebound

                # Bonus si régime favorable (utiliser consensus_regime)
                if consensus_regime in ['TRENDING_BULL', 'BREAKOUT_BULL'] and consensus_strength >= 0.80:
                    ltf_aligned = True  # Override si conditions exceptionnelles
            elif signal_side == 'SELL':
                momentum_ok = momentum_score and float(momentum_score) < momentum_threshold_sell
                bias_ok = directional_bias in ['BEARISH', 'NEUTRAL']

                strong_breakdown = (consensus_strength >= 0.80 and momentum_ok and
                                   float(momentum_score) < 50)  # En-dessous de la neutralité

                ltf_aligned = bias_ok or strong_breakdown

                if consensus_regime in ['TRENDING_BEAR', 'BREAKOUT_BEAR'] and consensus_strength >= 0.80:
                    ltf_aligned = True

            if not ltf_aligned:
                debug_info = f"bias:{directional_bias}, mom:{momentum_score}, consensus:{consensus_strength:.2f}"
                if signal_side == 'BUY' and 'strong_rebound' in locals() and strong_rebound:
                    debug_info += ", strong_rebound:YES"
                return False, f"LTF non alignés ({debug_info})"

            # 3. Vérifier volatilité suffisante (adaptatif selon force consensus)
            current_atr = context.get('mtf_atr15m') or context.get('atr_14')
            avg_atr = context.get('mtf_atr15m_ma') or context.get('atr_14_ma')
            market_regime = context.get('market_regime', 'NEUTRAL')
            total_strats = context.get('total_strategies', 0)

            if current_atr and avg_atr:
                atr_ratio = float(current_atr) / float(avg_atr)

                # Seuil adaptatif intelligent
                required_atr = 1.05  # Base fail-safe

                # Assouplissement selon contexte
                if market_regime == 'RANGING':
                    required_atr -= 0.03  # 1.02 (ATR plus bas en range)

                if consensus_strength >= 0.95 and total_strats >= 10:
                    required_atr -= 0.05  # Super-consensus exceptionnel (-5 bp de plus)
                elif consensus_strength >= 0.90 and total_strats >= 8:
                    required_atr -= 0.04  # Consensus exceptionnel
                elif consensus_strength >= 0.85:
                    required_atr -= 0.03  # Consensus très fort
                elif consensus_strength >= 0.80:
                    required_atr -= 0.02  # Consensus fort

                # Garde-fous: jamais < 1.00
                required_atr = max(1.00, round(required_atr, 2))

                if atr_ratio < required_atr:
                    return False, f"ATR ratio {atr_ratio:.2f} < {required_atr:.2f}x requis (regime:{market_regime}, consensus:{consensus_strength:.2f})"
            # On ne bloque plus si ATR manquant (trop restrictif)

            # 4. Vérifier R:R minimal (assoupli pour scalping)
            atr = current_atr or context.get('atr_14')
            if atr:
                atr = float(atr)
                # R:R adaptatif: 1.8 si consensus >= 0.80, sinon 2.0
                sl_estimated = 0.8 * atr
                tp_estimated = 1.5 * atr if consensus_strength >= 0.80 else 1.6 * atr
                risk_reward = tp_estimated / sl_estimated if sl_estimated > 0 else 0

                min_rr = 1.8 if consensus_strength >= 0.80 else 2.0
                if risk_reward < min_rr:
                    return False, f"R:R {risk_reward:.2f} < {min_rr} requis"

            # 5. Traçabilité du trade fail-safe
            context['is_failsafe_trade'] = True  # Tag pour analyse post-trade

            # Message de succès avec détails
            rebound_flag = " [REBOUND]" if signal_side == 'BUY' and directional_bias == 'BEARISH' and consensus_strength >= 0.80 else ""
            success_msg = f"🚀 FAIL-SAFE V2{rebound_flag}: consensus={consensus_strength:.2f}, strats={strategies_count}, regime={consensus_regime or market_regime}"
            logger.info(f"✅ {success_msg} pour {signal_side}")
            return True, success_msg

        except Exception as e:
            logger.error(f"Erreur fail-safe LTF: {e}")
            return False, f"Erreur fail-safe: {e}"

    def _check_htf_reversal_window(self, signal_side: str, context: Dict[str, Any],
                                  close_15m: float, ema20_15m: float, ema100_15m: float) -> bool:
        """
        Détecte les fenêtres de retournement HTF (breadth flip).

        Conditions pour autoriser le passage quand 15m pas encore aligné :
        - Consensus ultra-fort (≥0.95) + wave_winner
        - Beaucoup d'autres signaux déclenchés (≥10 stratégies)
        - 15m "soft SELL" : proche EMA20 ou pente EMA20 positive
        Note: Quota global 15min/symbole déjà vérifié en amont
        """
        try:
            # Vérifier conditions de base pour reversal window
            consensus_strength = context.get('consensus_strength', 0.5)
            wave_winner = context.get('wave_winner', False)
            total_strategies = context.get('total_strategies', 0)
            consensus_regime = context.get('consensus_regime', 'UNKNOWN')

            # Conditions strictes pour breadth flip
            if not (consensus_strength >= 0.95 and
                   wave_winner and
                   total_strategies >= 10 and
                   signal_side == 'BUY' and
                   consensus_regime in ['TRENDING_BULL', 'TRANSITION', 'RANGING']):
                return False

            # Vérifier si le 15m est "soft SELL" (pas encore vraiment baissier)
            atr_15m = context.get('htf_atr_15m', 0)
            if atr_15m:
                atr_15m = float(atr_15m)
                # Condition 1: Close proche EMA20 (dans 0.25*ATR)
                distance_to_ema20 = abs(close_15m - ema20_15m)
                close_near_ema20 = distance_to_ema20 <= (0.25 * atr_15m)

                # Condition 2: Pente EMA20 positive (simulée par proximité avec EMA100)
                ema20_trending_up = ema20_15m >= (ema100_15m * 0.999)  # EMA20 pas en chute libre

                if close_near_ema20 or ema20_trending_up:
                    # Le quota est déjà vérifié globalement au début de apply_critical_filters
                    logger.info(f"🔄 HTF REVERSAL WINDOW activée: close_near_ema20={close_near_ema20}, "
                              f"ema20_trending_up={ema20_trending_up}, consensus={consensus_strength:.2f}")

                    # Marquer le trade comme reversal pour tracking
                    context['htf_reversal_window'] = True
                    return True

            return False

        except Exception as e:
            logger.error(f"Erreur HTF reversal window: {e}")
            return False

    def _check_momentum_bypass(self, signal_side: str, context: Dict[str, Any],
                              consensus_strength: float, wave_winner: bool,
                              distance_to_ema20: float, tolerance: float,
                              current_price: float, ema100_3m: float) -> Tuple[bool, str]:
        """
        Bypass pullback timing pour momentum propre avec conditions ultra-bornées.

        Conditions :
        - Régime TRENDING_BULL/BREAKOUT_BULL + consensus ≥0.95 + wave_winner
        - Volume ratio ≥1.3 + price > EMA100 + pente EMA20 positive
        - Dépassement limité : distance ≤ tolerance + 0.20%
        - Cooldown 30-60min par symbole (TODO)
        """
        try:
            consensus_regime = context.get('consensus_regime', 'UNKNOWN')
            volume_ratio = context.get('volume_ratio', 1.0)

            # Conditions strictes pour momentum bypass
            if not (consensus_strength >= 0.95 and
                   wave_winner and
                   signal_side == 'BUY' and
                   consensus_regime in ['TRENDING_BULL', 'BREAKOUT_BULL']):
                return False, "Momentum bypass: conditions de base non remplies"

            # Vérifier volume et structure
            if volume_ratio < 1.3:
                return False, f"Momentum bypass: volume {volume_ratio:.2f} < 1.3x requis"

            if current_price <= ema100_3m:
                return False, f"Momentum bypass: prix {current_price:.4f} ≤ EMA100 {ema100_3m:.4f}"

            # Simuler pente EMA20 positive (proxy : EMA26 stable/croissante)
            ema20_3m = context.get('ema_26', 0)
            if ema20_3m:
                ema20_vs_ema100_ratio = float(ema20_3m) / ema100_3m
                if ema20_vs_ema100_ratio < 1.001:  # EMA20 pas vraiment au-dessus EMA100
                    return False, f"Momentum bypass: EMA20 pas en pente positive ({ema20_vs_ema100_ratio:.4f})"

            # Vérifier dépassement limité (tolerance + 20bp max)
            max_overshoot = tolerance + 0.0020  # +0.20%
            distance_pct = distance_to_ema20 / current_price
            if distance_pct > max_overshoot:
                return False, f"Momentum bypass: dépassement {distance_pct:.4f} > {max_overshoot:.4f} max"

            logger.info(f"🚀 MOMENTUM BYPASS activé: distance={distance_pct:.4f}, "
                      f"vol={volume_ratio:.2f}, regime={consensus_regime}")

            # Marquer pour audit
            context['entry_style'] = 'momentum'
            context['momentum_bypass'] = True

            return True, f"EntryStyle=momentum (skip pullback, distance={distance_pct:.4f})"

        except Exception as e:
            logger.error(f"Erreur momentum bypass: {e}")
            return False, f"Momentum bypass error: {e}"

