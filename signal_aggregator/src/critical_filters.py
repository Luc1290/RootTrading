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
        
        # Filtre anomalies système
        self.max_data_staleness_minutes = 10  # Données > 10min = problème technique
        
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

    def get_filter_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de configuration des filtres."""
        return {
            'atr_universe_multiplier': self.atr_universe_multiplier,
            'fallback_max_atr_percent': self.fallback_max_atr_percent,
            'min_volume_ratio': self.min_volume_ratio,
            'min_mtf_consistency': self.min_mtf_consistency,
            'filters_count': 4,
            'description': 'Filtres critiques optimisés - ATR dynamique, BB safe, volume détaillé'
        }