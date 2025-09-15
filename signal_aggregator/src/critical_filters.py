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

        # Paramètres filtre directionnel 1h
        self.htf_ema_fast = 20    # EMA rapide pour trend 1h
        self.htf_ema_slow = 100   # EMA lente pour trend 1h

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

        # ========== VALIDATION MTF STRICTE (PRIORITAIRE) ==========
        if self.strict_mtf_enabled:
            symbol = context.get('symbol', 'UNKNOWN')

            # FILTRE MTF 1: Direction 1h (filtre principal)
            htf_direction_check = self._check_htf_direction(signals, context)
            if not htf_direction_check[0]:
                return False, f"DIRECTION 1H INVALIDE: {htf_direction_check[1]}"

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
            
        # Récupérer le régime de marché
        market_regime = context.get('market_regime', 'UNKNOWN')
        # regime_strength est un string en DB, pas un float
        regime_strength_str = context.get('regime_strength', 'weak')
        regime_confidence = context.get('regime_confidence', 0.0)
        
        # RÈGLES STRICTES MAIS PERMETTANT LES REBONDS
        if signal_side == 'BUY':
            # En régime baissier, permettre les rebonds avec conditions strictes
            if market_regime in ['TRENDING_BEAR', 'BREAKOUT_BEAR']:
                # Vérifier les signaux de rebond potentiel
                rsi_14 = context.get('rsi_14', 50)
                stoch_rsi = context.get('stoch_rsi', 50)
                williams_r = context.get('williams_r', -50)
                volume_ratio = context.get('volume_ratio', 1.0)
                
                # Conditions pour permettre un rebond
                oversold_conditions = 0
                if rsi_14 and float(rsi_14) < 30:  # RSI oversold
                    oversold_conditions += 1
                if stoch_rsi and float(stoch_rsi) < 20:  # StochRSI oversold
                    oversold_conditions += 1
                if williams_r and float(williams_r) < -80:  # Williams %R oversold
                    oversold_conditions += 1
                if volume_ratio and float(volume_ratio) > 1.5:  # Volume spike (capitulation)
                    oversold_conditions += 1
                    
                # Permettre le rebond si au moins 3 conditions oversold
                if oversold_conditions < 3:
                    return False, f"Achat en {market_regime} rejeté: seulement {oversold_conditions}/3 conditions oversold"
                else:
                    # Signal de rebond potentiel accepté mais avec prudence
                    logger.info(f"⚠️ REBOND POTENTIEL détecté en {market_regime}: {oversold_conditions} conditions oversold")
                
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
                    
            # Prudence en transition (regime_confidence est en % 0-100)
            if market_regime == 'TRANSITION' and regime_confidence < 30:
                return False, f"Achat rejeté: Transition faible (confidence {regime_confidence:.0f}%)"
                
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
        Vérifie que le signal est dans le sens de la tendance 1h.
        Utilise EMA20 et EMA100 sur le 1h pour déterminer la direction.
        """
        try:
            if not signals:
                return True, "Pas de signaux"

            signal_side = signals[0].get('side')
            if not signal_side:
                return True, "Direction non définie"

            # Utiliser les données du context (injectées par context_manager)
            close_1h = context.get('htf_close_1h')
            ema20_1h = context.get('htf_ema20_1h')  # Utilise ema_26 comme proxy pour ema_20
            ema100_1h = context.get('htf_ema100_1h')  # Utilise ema_99 comme proxy pour ema_100

            if not all([close_1h, ema20_1h, ema100_1h]):
                logger.warning("EMAs 1h manquantes dans le contexte")
                return True, "EMAs 1h manquantes"  # On laisse passer si pas de données

            close_1h = float(close_1h)
            ema20_1h = float(ema20_1h)
            ema100_1h = float(ema100_1h)

            # Déterminer la direction HTF
            htf_direction = None
            if close_1h > ema100_1h and ema20_1h > ema100_1h:
                htf_direction = 'BUY'  # Tendance haussière
            elif close_1h < ema100_1h and ema20_1h < ema100_1h:
                htf_direction = 'SELL'  # Tendance baissière
            else:
                # Zone neutre/transition - pas de trade
                return False, f"Zone neutre 1h (Close:{close_1h:.4f}, EMA20:{ema20_1h:.4f}, EMA100:{ema100_1h:.4f})"

            # Vérifier que le signal est dans le bon sens
            if signal_side != htf_direction:
                return False, f"Signal {signal_side} contre tendance 1h {htf_direction}"

            logger.info(f"✅ Direction HTF validée: {signal_side} aligné avec 1h {htf_direction}")
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
        """
        try:
            if not signals:
                return True, "Pas de signaux"

            signal_side = signals[0].get('side')
            if not signal_side:
                return True, "Direction non définie"

            # Utiliser les EMAs du contexte actuel (3m)
            current_price = context.get('current_price', 0)
            # Utiliser ema_26 comme proxy pour ema_20, ema_99 pour ema_100
            ema20_3m = context.get('ema_20') or context.get('ema_26')  # Fallback sur ema_26
            ema100_3m = context.get('ema_100') or context.get('ema_99')  # Fallback sur ema_99

            if not all([current_price, ema20_3m, ema100_3m]):
                logger.warning("EMAs 3m manquantes pour timing pullback (même avec fallback)")
                return True, "EMAs 3m manquantes"

            current_price = float(current_price)
            ema20_3m = float(ema20_3m)
            ema100_3m = float(ema100_3m)

            distance_to_ema20 = abs(current_price - ema20_3m)
            price_pct = current_price * self.pullback_tolerance  # Tolérance configurable

            if signal_side == 'BUY':
                # Pour BUY: Prix proche EMA20 ET au-dessus EMA100
                if current_price < ema100_3m:
                    return False, f"Prix {current_price:.4f} < EMA100(3m) {ema100_3m:.4f}"

                # Vérifier pullback vers EMA20 (tolérance configurable)
                if distance_to_ema20 > price_pct:
                    return False, f"Prix trop loin EMA20: {distance_to_ema20:.4f} > {price_pct:.4f} ({self.pullback_tolerance_bp}bp)"

            elif signal_side == 'SELL':
                # Pour SELL: Prix proche EMA20 ET en-dessous EMA100
                if current_price > ema100_3m:
                    return False, f"Prix {current_price:.4f} > EMA100(3m) {ema100_3m:.4f}"

                # Vérifier pullback vers EMA20
                if distance_to_ema20 > price_pct:
                    return False, f"Prix trop loin EMA20: {distance_to_ema20:.4f} > {price_pct:.4f}"

            logger.info(f"✅ Timing pullback 3m validé: {signal_side} à {current_price:.4f}")
            return True, "Timing pullback valide"

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

            # Vérifier que SL est valide
            if sl_estimated <= 0:
                return False, f"SL estimé invalide: {sl_estimated:.4f}"

            # Calculer Risk/Reward
            risk_reward = tp_estimated / sl_estimated

            # Vérifier minimum R/R
            if risk_reward < self.min_risk_reward:
                return False, f"R/R {risk_reward:.2f} < {self.min_risk_reward}"

            logger.info(f"✅ Risk/Reward OK: {risk_reward:.2f} (SL:{sl_estimated:.4f}, TP:{tp_estimated:.4f})")
            return True, f"R/R valide: {risk_reward:.2f}"

        except Exception as e:
            logger.error(f"Erreur check R/R: {e}")
            return True, "Erreur validation R/R"

    def get_filter_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de configuration des filtres."""
        return {
            'mode': 'STRICT_MTF' if self.strict_mtf_enabled else 'STANDARD',
            'atr_universe_multiplier': self.atr_universe_multiplier,
            'fallback_max_atr_percent': self.fallback_max_atr_percent,
            'min_volume_ratio': self.min_volume_ratio,
            'min_mtf_consistency': self.min_mtf_consistency,
            'strict_mtf_params': {
                'htf_ema_fast': self.htf_ema_fast,
                'htf_ema_slow': self.htf_ema_slow,
                'min_atr_ratio': self.mtf_min_atr_ratio,
                'min_risk_reward': self.min_risk_reward,
                'pullback_tolerance_bp': self.pullback_tolerance_bp
            },
            'filters_count': 8 if self.strict_mtf_enabled else 4,
            'description': 'Mode Shaolin: Validation MTF stricte 1h/15m/3m - Max 3 trades/jour'
        }