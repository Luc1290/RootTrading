"""
Gestionnaire de trailing sell intelligent avec tracking du prix maximum historique.
Extrait du coordinator pour alléger le code et améliorer la maintenance.
"""
import logging
import time
import json
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class TrailingSellManager:
    """
    Gère la logique de trailing sell avec tracking intelligent du prix maximum.
    """
    
    def __init__(self, redis_client, service_client, db_connection=None):
        """
        Initialise le gestionnaire de trailing sell.
        
        Args:
            redis_client: Client Redis pour stocker les références
            service_client: Client des services pour récupérer les positions
            db_connection: Connexion DB pour les données d'analyse (optionnel)
        """
        self.redis_client = redis_client
        self.service_client = service_client
        self.db_connection = db_connection
        
        # Configuration trailing sell - OPTIMISÉE avec ATR
        self.base_min_gain_for_trailing = 0.005  # 0.5% base (augmenté de 0.3%)
        self.base_sell_margin = 0.012  # 1.2% marge base (augmenté de 0.8%)
        self.max_drop_threshold = 0.015  # 1.5% de chute max depuis le pic (augmenté de 1.0%)
        self.immediate_sell_drop = 0.020  # 2.0% de chute = vente immédiate (augmenté de 1.5%)
        
        # Configuration stop-loss adaptatif - PLUS STRICT pour couper les pertes rapidement
        self.stop_loss_percent_base = 0.010  # 1.0% de base - très strict (réduit de 1.5%)
        self.stop_loss_percent_bullish = 0.012  # 1.2% en tendance haussière (réduit de 1.8%)
        self.stop_loss_percent_strong_bullish = 0.015  # 1.5% en tendance très haussière (réduit de 2.2%)
        
        logger.info("✅ TrailingSellManager initialisé")
    
    def check_trailing_sell(self, symbol: str, current_price: float, 
                           entry_price: float, entry_time: Any) -> Tuple[bool, str]:
        """
        Vérifie si on doit exécuter le SELL selon la logique de trailing sell améliorée.
        Utilise le prix maximum historique du cycle pour une meilleure décision.
        
        Args:
            symbol: Symbole de trading
            current_price: Prix actuel
            entry_price: Prix d'entrée de la position
            entry_time: Timestamp d'entrée (ISO string ou epoch)
            
        Returns:
            (should_sell, reason)
        """
        logger.info(f"🔍 DEBUT check_trailing_sell pour {symbol} @ {current_price}")
        
        try:
            # Convertir timestamp ISO en epoch si nécessaire
            if isinstance(entry_time, str):
                entry_time_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                entry_time_epoch = entry_time_dt.timestamp()
            else:
                entry_time_epoch = float(entry_time) if entry_time else time.time()
            
            precision = self._get_price_precision(current_price)
            logger.info(f"🔍 Prix entrée: {entry_price:.{precision}f}, Prix actuel: {current_price:.{precision}f}")
            
            # Calculer la performance actuelle
            loss_percent = (entry_price - current_price) / entry_price
            
            # Récupérer et afficher le prix max historique dès le début
            historical_max = self._get_and_update_max_price(symbol, current_price, entry_price)
            drop_from_max = (historical_max - current_price) / historical_max
            logger.info(f"📊 Prix max historique: {historical_max:.{precision}f}, Chute depuis max: {drop_from_max*100:.2f}%")
            
            # === STOP-LOSS ADAPTATIF INTELLIGENT ===
            adaptive_threshold = self._calculate_adaptive_threshold(symbol, entry_price, entry_time_epoch)
            
            # Afficher gain ou perte selon le signe
            if loss_percent < 0:
                logger.info(f"🧠 Stop-loss adaptatif pour {symbol}: {adaptive_threshold*100:.2f}% (gain actuel: {abs(loss_percent)*100:.2f}%)")
            else:
                logger.info(f"🧠 Stop-loss adaptatif pour {symbol}: {adaptive_threshold*100:.2f}% (perte actuelle: {loss_percent*100:.2f}%)")
            
            # Si perte dépasse le seuil adaptatif : SELL immédiat
            if loss_percent >= adaptive_threshold:
                logger.info(f"📉 Stop-loss adaptatif déclenché pour {symbol}: perte {loss_percent*100:.2f}% ≥ seuil {adaptive_threshold*100:.2f}%")
                return True, f"Stop-loss adaptatif déclenché (perte {loss_percent*100:.2f}% ≥ {adaptive_threshold*100:.2f}%)"
            
            # Si position perdante mais dans la tolérance : garder
            if current_price <= entry_price:
                logger.info(f"🟡 Position perdante mais dans tolérance pour {symbol}: perte {loss_percent*100:.2f}% < seuil {adaptive_threshold*100:.2f}%")
                return False, f"Position perdante mais dans tolérance (perte {loss_percent*100:.2f}% < {adaptive_threshold*100:.2f}%)"
            
            # === POSITION GAGNANTE : TAKE PROFIT BINAIRE + TRAILING SELL ===
            gain_percent = (current_price - entry_price) / entry_price
            logger.info(f"🔍 Position gagnante détectée: +{gain_percent*100:.2f}%, vérification take profit et trailing")
            
            # === TAKE PROFIT PROGRESSIF - RIDE LES PUMPS MAIS FERME EFFICACEMENT ===
            # Désactiver temporairement le TP progressif si gain < 1.5% pour permettre sortie rapide
            if gain_percent >= 0.015:  # Activer TP progressif seulement après +1.5%
                should_take_profit, tp_reason = self._check_progressive_take_profit(symbol, gain_percent)
                if should_take_profit:
                    logger.info(f"💰 TAKE PROFIT PROGRESSIF DÉCLENCHÉ: {tp_reason}")
                    self._cleanup_references(symbol)
                    return True, tp_reason
            else:
                logger.debug(f"TP progressif désactivé pour {symbol} (gain {gain_percent*100:.2f}% < 1.5%)")
            
            
            # Seuils adaptatifs basés sur ATR
            atr_based_thresholds = self._get_atr_based_thresholds(symbol)
            min_gain_for_trailing = atr_based_thresholds['activate_trailing_gain']
            sell_margin = atr_based_thresholds['trailing_margin']
            
            logger.info(f"📊 Seuils ATR pour {symbol}: activation={min_gain_for_trailing*100:.2f}%, marge={sell_margin*100:.2f}%")
            
            # Vérifier si le gain minimum est atteint pour activer le trailing
            if gain_percent < min_gain_for_trailing:
                logger.info(f"📊 Gain insuffisant pour trailing ({gain_percent*100:.2f}% < {min_gain_for_trailing*100:.2f}%), position continue")
                return False, f"Gain insuffisant pour activer le trailing ({gain_percent*100:.2f}% < {min_gain_for_trailing*100:.2f}%)"
            
            # Prix max déjà récupéré plus haut
            logger.info(f"🎯 TRAILING LOGIC: Utilisation du prix MAX ({historical_max:.{precision}f}) pour décision, PAS le prix d'entrée")
            
            # Récupérer le prix SELL précédent
            previous_sell_price = self._get_previous_sell_price(symbol)
            logger.info(f"🔍 Prix SELL précédent: {previous_sell_price}")
            
            # === DÉCISION DE VENTE BASÉE SUR LE PRIX MAX ===
            
            # Si chute importante depuis le max (>2.0%), vendre immédiatement
            if drop_from_max >= self.immediate_sell_drop:
                logger.warning(f"📉 CHUTE IMPORTANTE depuis max ({drop_from_max*100:.2f}%), SELL IMMÉDIAT!")
                self._cleanup_references(symbol)
                return True, f"Chute de {drop_from_max*100:.2f}% depuis max {historical_max:.{precision}f}, SELL immédiat"
            
            if previous_sell_price is None:
                # Premier SELL gagnant : on est déjà à +1% minimum, donc on tolère la chute configurée
                if drop_from_max > sell_margin:  # Si déjà chuté selon marge ATR depuis le max
                    logger.info(f"⚠️ Premier SELL mais déjà {drop_from_max*100:.2f}% sous le max historique")
                    
                    # Si chute atteint le seuil normal (>1.5%), vendre
                    if drop_from_max >= self.max_drop_threshold:
                        logger.info(f"📉 Chute significative depuis max, SELL!")
                        self._cleanup_references(symbol)
                        return True, f"Chute de {drop_from_max*100:.2f}% depuis max {historical_max:.{precision}f}, SELL exécuté"
                
                # Sinon stocker comme référence
                self._update_sell_reference(symbol, current_price)
                logger.info(f"🎯 Premier SELL @ {current_price:.{precision}f} stocké (max: {historical_max:.{precision}f})")
                return False, f"Premier SELL stocké, max historique: {historical_max:.{precision}f}"
            
            # === SELL SUIVANTS : LOGIQUE CLASSIQUE + VÉRIFICATION MAX ===
            
            # D'abord vérifier la chute depuis le max
            if drop_from_max >= self.max_drop_threshold:
                logger.warning(f"📉 Chute de {drop_from_max*100:.2f}% depuis max, SELL!")
                self._cleanup_references(symbol)
                return True, f"Chute de {drop_from_max*100:.2f}% depuis max {historical_max:.{precision}f}"
            
            # Ensuite logique classique de trailing avec marge ATR
            sell_threshold = previous_sell_price * (1 - sell_margin)
            logger.info(f"🔍 Seuil de vente calculé: {sell_threshold:.{precision}f} (marge {sell_margin*100:.2f}%)")
            
            if current_price > previous_sell_price:
                # Prix monte : mettre à jour référence
                self._update_sell_reference(symbol, current_price)
                logger.info(f"📈 Prix monte: {current_price:.{precision}f} > {previous_sell_price:.{precision}f}, référence mise à jour")
                return False, f"Prix monte, référence mise à jour (max: {historical_max:.{precision}f})"
                
            elif current_price > sell_threshold:
                # Prix dans la marge de tolérance
                logger.info(f"🟡 Prix stable: {current_price:.{precision}f} > seuil {sell_threshold:.{precision}f}")
                return False, f"Prix dans marge de tolérance (max: {historical_max:.{precision}f})"
                
            else:
                # Prix baisse significativement : VENDRE
                logger.warning(f"📉 Baisse significative: {current_price:.{precision}f} ≤ {sell_threshold:.{precision}f}, SELL!")
                self._cleanup_references(symbol)
                return True, f"Baisse sous seuil trailing ({current_price:.{precision}f} ≤ {sell_threshold:.{precision}f})"
                
        except Exception as e:
            logger.error(f"❌ Erreur dans check_trailing_sell pour {symbol}: {str(e)}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            # En cas d'erreur, autoriser le SELL par sécurité
            return True, f"Erreur technique, SELL autorisé par défaut"
    
    def update_max_price_if_needed(self, symbol: str, current_price: float) -> bool:
        """
        Met à jour le prix max si le prix actuel est plus élevé.
        Appelé par le monitoring automatique.
        
        Args:
            symbol: Symbole
            current_price: Prix actuel
            
        Returns:
            True si le prix max a été mis à jour
        """
        try:
            max_price_key = f"cycle_max_price:{symbol}"
            max_price_data = self.redis_client.get(max_price_key)
            
            historical_max = None
            if max_price_data:
                try:
                    if isinstance(max_price_data, dict):
                        historical_max = float(max_price_data.get("price", 0))
                    elif isinstance(max_price_data, (str, bytes)):
                        if isinstance(max_price_data, bytes):
                            max_price_data = max_price_data.decode('utf-8')
                        max_price_dict = json.loads(max_price_data)
                        historical_max = float(max_price_dict.get("price", 0))
                except Exception as e:
                    logger.error(f"Erreur parsing prix max: {e}")
            
            if historical_max is None or current_price > historical_max:
                self._update_cycle_max_price(symbol, current_price)
                precision = self._get_price_precision(current_price)
                logger.info(f"📈 Nouveau max pour {symbol}: {current_price:.{precision}f}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Erreur mise à jour prix max {symbol}: {e}")
            return False
    
    def _get_and_update_max_price(self, symbol: str, current_price: float, 
                                  entry_price: float) -> float:
        """
        Récupère et met à jour le prix max historique du cycle.
        
        Args:
            symbol: Symbole
            current_price: Prix actuel
            entry_price: Prix d'entrée (utilisé si pas de max stocké)
            
        Returns:
            Prix maximum historique
        """
        # Récupérer le prix max historique
        max_price_key = f"cycle_max_price:{symbol}"
        max_price_data = self.redis_client.get(max_price_key)
        historical_max = None
        
        if max_price_data:
            try:
                if isinstance(max_price_data, dict):
                    historical_max = float(max_price_data.get("price", 0))
                elif isinstance(max_price_data, (str, bytes)):
                    if isinstance(max_price_data, bytes):
                        max_price_data = max_price_data.decode('utf-8')
                    max_price_dict = json.loads(max_price_data)
                    historical_max = float(max_price_dict.get("price", 0))
            except Exception as e:
                logger.error(f"Erreur récupération prix max: {e}")
        
        # Si pas de prix max, initialiser avec le prix d'entrée
        if historical_max is None:
            historical_max = entry_price
            self._update_cycle_max_price(symbol, entry_price)
        
        # Mettre à jour si le prix actuel est plus élevé
        if current_price > historical_max:
            historical_max = current_price
            self._update_cycle_max_price(symbol, current_price)
            precision = self._get_price_precision(current_price)
            logger.info(f"📊 Nouveau prix max pour {symbol}: {current_price:.{precision}f}")
        
        return historical_max
    
    def _get_previous_sell_price(self, symbol: str) -> Optional[float]:
        """
        Récupère le prix du SELL précédent stocké en référence.
        
        Args:
            symbol: Symbole à vérifier
            
        Returns:
            Prix du SELL précédent ou None
        """
        try:
            ref_key = f"sell_reference:{symbol}"
            price_data = self.redis_client.get(ref_key)
            
            if not price_data:
                return None
            
            logger.debug(f"🔍 Récupération sell reference {symbol}: type={type(price_data)}, data={price_data}")
            
            # Gérer tous les cas possibles de retour Redis
            if isinstance(price_data, dict):
                if "price" in price_data:
                    return float(price_data["price"])
                else:
                    logger.warning(f"Clé 'price' manquante dans dict Redis pour {symbol}")
                    return None
            
            elif isinstance(price_data, (str, bytes)):
                try:
                    if isinstance(price_data, bytes):
                        price_data = price_data.decode('utf-8')
                    
                    parsed_data = json.loads(price_data)
                    if isinstance(parsed_data, dict) and "price" in parsed_data:
                        return float(parsed_data["price"])
                    else:
                        logger.warning(f"Format JSON invalide pour {symbol}: {parsed_data}")
                        return None
                except json.JSONDecodeError as e:
                    logger.error(f"Erreur JSON decode pour {symbol}: {e}")
                    return None
            
            else:
                logger.warning(f"Type Redis inattendu pour {symbol}: {type(price_data)}")
                return None
                
        except Exception as e:
            logger.error(f"Erreur récupération sell reference pour {symbol}: {e}")
            return None
    
    def _update_sell_reference(self, symbol: str, price: float) -> None:
        """
        Met à jour la référence de prix SELL pour un symbole.
        
        Args:
            symbol: Symbole
            price: Nouveau prix de référence
        """
        try:
            ref_key = f"sell_reference:{symbol}"
            ref_data = {
                "price": price,
                "timestamp": int(time.time() * 1000)
            }
            # TTL de 2 heures pour éviter les références obsolètes
            self.redis_client.set(ref_key, json.dumps(ref_data), expiration=7200)
        except Exception as e:
            logger.error(f"Erreur mise à jour sell reference pour {symbol}: {e}")
    
    def _clear_sell_reference(self, symbol: str) -> None:
        """
        Supprime la référence de prix SELL pour un symbole.
        
        Args:
            symbol: Symbole
        """
        try:
            ref_key = f"sell_reference:{symbol}"
            self.redis_client.delete(ref_key)
            logger.info(f"🧹 Référence SELL supprimée pour {symbol}")
        except Exception as e:
            logger.error(f"Erreur suppression sell reference pour {symbol}: {e}")
    
    def _update_cycle_max_price(self, symbol: str, price: float) -> None:
        """
        Met à jour le prix maximum historique d'un cycle.
        
        Args:
            symbol: Symbole
            price: Nouveau prix maximum
        """
        try:
            max_key = f"cycle_max_price:{symbol}"
            max_data = {
                "price": price,
                "timestamp": int(time.time() * 1000)
            }
            # TTL de 24 heures pour le prix max
            self.redis_client.set(max_key, json.dumps(max_data), expiration=86400)
            logger.debug(f"📈 Prix max mis à jour pour {symbol}: {price}")
        except Exception as e:
            logger.error(f"Erreur mise à jour prix max pour {symbol}: {e}")
    
    def _clear_cycle_max_price(self, symbol: str) -> None:
        """
        Supprime le prix maximum historique d'un cycle.
        
        Args:
            symbol: Symbole
        """
        try:
            max_key = f"cycle_max_price:{symbol}"
            self.redis_client.delete(max_key)
            logger.info(f"🧹 Prix max historique supprimé pour {symbol}")
        except Exception as e:
            logger.error(f"Erreur suppression prix max pour {symbol}: {e}")
    
    # SUPPRIMÉ - fonction dupliquée, gardée seulement la version mise à jour plus bas
    
    def _get_atr_based_thresholds(self, symbol: str) -> Dict[str, float]:
        """
        Calcule les seuils adaptatifs basés sur l'ATR pour trailing et activation.
        
        Args:
            symbol: Symbole à analyser
            
        Returns:
            Dict avec trailing_margin, activate_trailing_gain, adaptive_sl
        """
        try:
            # Récupérer ATR depuis l'analyse
            atr_percent = self._get_atr_percentage(symbol)
            
            if atr_percent is None:
                logger.debug(f"Pas d'ATR pour {symbol}, seuils par défaut")
                return {
                    'trailing_margin': self.base_sell_margin,
                    'activate_trailing_gain': self.base_min_gain_for_trailing,
                    'adaptive_sl': self.stop_loss_percent_base
                }
            
            # Calculer les seuils adaptatifs selon tes formules - PLUS PERMISSIFS
            # trailing_margin = clamp(0.8%, 2.0%, 1.0 * ATR%)
            trailing_margin = max(0.008, min(0.020, 1.0 * atr_percent))
            
            # activate_trailing_gain = max(0.5%, 0.3 * ATR%)  
            activate_trailing_gain = max(0.005, 0.3 * atr_percent)
            
            # adaptive_sl = max(1.2*ATR%, 1.0%) capped at 2.0% - PLUS STRICT
            adaptive_sl = min(0.020, max(1.2 * atr_percent, 0.010))
            
            logger.debug(f"🧠 Seuils ATR pour {symbol}: trailing={trailing_margin*100:.2f}%, activation={activate_trailing_gain*100:.2f}%, SL={adaptive_sl*100:.2f}% (ATR={atr_percent*100:.2f}%)")
            
            return {
                'trailing_margin': trailing_margin,
                'activate_trailing_gain': activate_trailing_gain,
                'adaptive_sl': adaptive_sl
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul seuils ATR {symbol}: {e}")
            return {
                'trailing_margin': self.base_sell_margin,
                'activate_trailing_gain': self.base_min_gain_for_trailing, 
                'adaptive_sl': self.stop_loss_percent_base
            }
    
    def _get_atr_percentage(self, symbol: str) -> Optional[float]:
        """
        Récupère l'ATR en pourcentage depuis les données d'analyse.
        
        Args:
            symbol: Symbole
            
        Returns:
            ATR en pourcentage (ex: 0.025 = 2.5%) ou None
        """
        if not self.db_connection:
            return None
            
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    SELECT ad.atr_14, md.close
                    FROM analyzer_data ad
                    JOIN market_data md ON (ad.symbol = md.symbol AND ad.timeframe = md.timeframe AND ad.time = md.time)
                    WHERE ad.symbol = %s AND ad.timeframe = '1m'
                    AND ad.atr_14 IS NOT NULL
                    ORDER BY ad.time DESC 
                    LIMIT 1
                """, (symbol,))
                
                result = cursor.fetchone()
                if result and result[0] and result[1]:
                    atr_value = float(result[0])
                    close_price = float(result[1])
                    atr_percent = atr_value / close_price if close_price > 0 else 0
                    return atr_percent
                return None
        except Exception as e:
            logger.error(f"Erreur récupération ATR {symbol}: {e}")
            return None
    
    def _calculate_adaptive_threshold(self, symbol: str, entry_price: float, 
                                     entry_time: float) -> float:
        """
        Calcule le seuil de stop-loss adaptatif avec ATR et données d'analyse.
        
        Args:
            symbol: Symbole à analyser
            entry_price: Prix d'entrée de la position
            entry_time: Timestamp d'entrée (epoch)
            
        Returns:
            Seuil de perte acceptable avant stop-loss (ex: 0.015 = 1.5%)
        """
        try:
            # Récupérer les seuils ATR
            atr_thresholds = self._get_atr_based_thresholds(symbol)
            atr_based_sl = atr_thresholds['adaptive_sl']
            
            # Récupérer les données d'analyse si disponibles
            analysis = self._get_latest_analysis_data(symbol) if self.db_connection else None
            
            if not analysis:
                logger.debug(f"Pas de données d'analyse pour {symbol}, utilisation ATR seul: {atr_based_sl*100:.2f}%")
                return atr_based_sl
            
            # Récupérer le régime de marché
            regime = analysis.get('market_regime', 'UNKNOWN')
            
            # Calculer les facteurs d'ajustement (version allégée)
            regime_factor = self._calculate_regime_factor(analysis)
            support_factor = self._calculate_support_factor(analysis, entry_price)
            time_factor = self._calculate_time_factor(entry_time)
            
            # Combiner ATR avec les autres facteurs
            adaptive_threshold = float(atr_based_sl) * float(regime_factor) * float(support_factor) * float(time_factor)
            
            # Contraintes finales - bornes plus strictes pour couper les pertes rapidement
            adaptive_threshold = max(0.008, min(0.020, adaptive_threshold))  # 0.8%-2.0% (réduit de 1.5%-3.5%)
            
            logger.debug(f"🧠 Stop-loss adaptatif ATR+analyse {symbol}: {adaptive_threshold*100:.2f}%")
            
            return adaptive_threshold
            
        except Exception as e:
            logger.error(f"Erreur calcul stop-loss adaptatif {symbol}: {e}")
            return self.stop_loss_percent_base
    
    def _get_latest_analysis_data(self, symbol: str) -> Optional[Dict]:
        """
        Récupère les données d'analyse les plus récentes.
        """
        if not self.db_connection:
            return None
            
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        market_regime, regime_strength, regime_confidence,
                        volatility_regime, atr_percentile,
                        nearest_support, support_strength,
                        trend_alignment, directional_bias
                    FROM analyzer_data 
                    WHERE symbol = %s AND timeframe = '1m'
                    ORDER BY time DESC 
                    LIMIT 1
                """, (symbol,))
                
                result = cursor.fetchone()
                if result:
                    return {
                        'market_regime': result[0],
                        'regime_strength': result[1], 
                        'regime_confidence': float(result[2]) if result[2] else 50.0,
                        'volatility_regime': result[3],
                        'atr_percentile': float(result[4]) if result[4] else 50.0,
                        'nearest_support': float(result[5]) if result[5] else None,
                        'support_strength': result[6],
                        'trend_alignment': result[7],
                        'directional_bias': result[8]
                    }
                return None
        except Exception as e:
            logger.error(f"Erreur récupération données analyse {symbol}: {e}")
            return None
    
    def _calculate_regime_factor(self, analysis: Dict) -> float:
        """Calcule le facteur basé sur le régime de marché."""
        regime = analysis.get('market_regime', 'UNKNOWN')
        strength = analysis.get('regime_strength', 'WEAK')
        confidence = float(analysis.get('regime_confidence', 50))
        
        # CORRECTION : logique inversée - plus strict en bear = seuil plus PETIT
        regime_multipliers = {
            'TRENDING_BULL': 1.2,      # Bull = plus tolérant (seuil plus large)
            'BREAKOUT_BULL': 1.1,      # Breakout bull = modérément tolérant
            'RANGING': 1.0,            # Range = neutre
            'TRANSITION': 0.9,         # Transition = légèrement strict
            'TRENDING_BEAR': 0.6,      # Bear = plus strict (seuil plus petit)
            'VOLATILE': 0.7,           # Volatile = strict
            'BREAKOUT_BEAR': 0.5       # Breakout bear = très strict
        }
        
        base_factor = regime_multipliers.get(regime, 1.0)
        
        strength_multipliers = {
            'EXTREME': 1.2,     # Réduit de 1.3 à 1.2
            'STRONG': 1.1, 
            'MODERATE': 1.0,
            'WEAK': 0.9         # Augmenté de 0.8 à 0.9 (moins punitif)
        }
        
        strength_factor = strength_multipliers.get(strength, 1.0)
        confidence_factor = 0.7 + (float(confidence) / 100.0) * 0.6
        
        return float(base_factor * strength_factor * confidence_factor)
    
    def _calculate_volatility_factor(self, analysis: Dict) -> float:
        """Calcule le facteur basé sur la volatilité."""
        volatility_regime = analysis.get('volatility_regime', 'normal')
        atr_percentile = float(analysis.get('atr_percentile', 50))
        
        volatility_multipliers = {
            'low': 0.7,
            'normal': 1.0,
            'high': 1.4,
            'extreme': 1.8
        }
        
        base_factor = volatility_multipliers.get(volatility_regime, 1.0)
        percentile_factor = 0.6 + (float(atr_percentile) / 100.0) * 0.8
        
        return float(base_factor * percentile_factor)
    
    def _calculate_support_factor(self, analysis: Dict, entry_price: float) -> float:
        """Calcule le facteur basé sur la proximité des supports."""
        nearest_support = analysis.get('nearest_support')
        support_strength = analysis.get('support_strength', 'WEAK')
        
        if not nearest_support:
            return 1.0
            
        support_price = float(nearest_support)
        entry_price_float = float(entry_price)
        support_distance = abs(entry_price_float - support_price) / entry_price_float
        
        strength_multipliers = {
            'MAJOR': 1.6,
            'STRONG': 1.3,
            'MODERATE': 1.1,
            'WEAK': 0.9
        }
        
        strength_factor = strength_multipliers.get(support_strength, 1.0)
        
        if support_distance < 0.01:
            distance_factor = 1.4
        elif support_distance < 0.02:
            distance_factor = 1.2
        elif support_distance < 0.05:
            distance_factor = 1.0
        else:
            distance_factor = 0.8
            
        return float(strength_factor * distance_factor)
    
    def _calculate_time_factor(self, entry_time: float) -> float:
        """Calcule le facteur basé sur le temps écoulé - AJUSTÉ POUR CRYPTO RAPIDE."""
        time_elapsed = float(time.time() - float(entry_time))
        minutes_elapsed = time_elapsed / 60.0
        
        # Facteur temps plus strict pour protection rapide
        if minutes_elapsed < 2:
            return 1.2  # Très récent = modérément permissif (réduit de 1.5)
        elif minutes_elapsed < 10:
            return 1.1  # Récent = légèrement permissif (réduit de 1.3)
        elif minutes_elapsed < 30:
            return 1.0  # Modéré = neutre (réduit de 1.1)
        elif minutes_elapsed < 120:
            return 0.9  # Normal = légèrement strict (réduit de 1.0)
        elif minutes_elapsed < 360:
            return 0.8  # Plus ancien = plus strict (réduit de 0.9)
        else:
            return 0.7  # Très ancien = strict (réduit de 0.85)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Récupère le prix actuel d'un symbole depuis Redis ou DB en fallback.
        
        Args:
            symbol: Symbole à récupérer
            
        Returns:
            Prix actuel ou None
        """
        try:
            # Essayer de récupérer depuis Redis (ticker)
            ticker_key = f"ticker:{symbol}"
            ticker_data = self.redis_client.get(ticker_key)
            
            if ticker_data:
                if isinstance(ticker_data, dict):
                    price = ticker_data.get('price')
                    if price:
                        return float(price)
                elif isinstance(ticker_data, str):
                    ticker_dict = json.loads(ticker_data)
                    price = ticker_dict.get('price')
                    if price:
                        return float(price)
            
            # Fallback: essayer market_data Redis
            market_key = f"market_data:{symbol}:1m"
            market_data = self.redis_client.get(market_key)
            
            if market_data:
                if isinstance(market_data, dict):
                    price = market_data.get('close')
                    if price:
                        return float(price)
                elif isinstance(market_data, str):
                    market_dict = json.loads(market_data)
                    price = market_dict.get('close')
                    if price:
                        return float(price)
            
            # Fallback final: récupérer depuis la base de données
            if self.db_connection:
                try:
                    with self.db_connection.cursor() as cursor:
                        cursor.execute("""
                            SELECT close 
                            FROM market_data 
                            WHERE symbol = %s 
                            ORDER BY time DESC 
                            LIMIT 1
                        """, (symbol,))
                        
                        result = cursor.fetchone()
                        if result and result[0]:
                            logger.info(f"💾 Prix récupéré depuis DB pour {symbol}: {result[0]}")
                            return float(result[0])
                        else:
                            logger.warning(f"💾 Aucun résultat DB pour {symbol}")
                except Exception as db_error:
                    logger.error(f"❌ Erreur DB pour {symbol}: {db_error}")
            else:
                logger.warning(f"💾 Pas de connexion DB pour {symbol}")
            
            logger.warning(f"⚠️ Prix non trouvé pour {symbol} (Redis + DB)")
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération prix pour {symbol}: {e}")
            return None

    def _check_progressive_take_profit(self, symbol: str, gain_percent: float) -> Tuple[bool, str]:
        """
        Take profit progressif AMÉLIORÉ : vend si rechute significative depuis le palier atteint.
        Permet de rider les pumps tout en fermant les cycles efficacement.
        
        Args:
            symbol: Symbole pour tracking du palier
            gain_percent: Pourcentage de gain actuel (ex: 0.025 = 2.5%)
            
        Returns:
            (should_sell, reason)
        """
        # Paliers de take profit ÉQUILIBRÉS - garde les seuils rentables après frais
        tp_levels = [0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.012, 0.010, 0.008]  # Garde 0.8% et 1.0%, supprimé 0.6% et 0.4%
        
        # Trouver le palier le plus élevé atteint actuellement
        current_tp_level = None
        for level in tp_levels:
            if gain_percent >= level:
                current_tp_level = level
                break
        
        if current_tp_level is None:
            # Aucun palier atteint, pas de TP
            return False, f"Aucun palier TP atteint (+{gain_percent*100:.2f}%)"
        
        # Récupérer le palier max historique pour ce symbole
        historical_tp_key = f"max_tp_level:{symbol}"
        historical_tp_data = self.redis_client.get(historical_tp_key)
        historical_max_tp = None
        
        if historical_tp_data:
            try:
                if isinstance(historical_tp_data, (str, bytes)):
                    if isinstance(historical_tp_data, bytes):
                        historical_tp_data = historical_tp_data.decode('utf-8')
                    historical_tp_dict = json.loads(historical_tp_data)
                    historical_max_tp = float(historical_tp_dict.get("level", 0))
                elif isinstance(historical_tp_data, dict):
                    historical_max_tp = float(historical_tp_data.get("level", 0))
            except Exception as e:
                logger.error(f"Erreur récupération palier TP historique {symbol}: {e}")
        
        # Initialiser si pas de palier historique
        if historical_max_tp is None:
            historical_max_tp = 0
        
        # Mettre à jour le palier max si on a atteint un nouveau sommet
        if current_tp_level > historical_max_tp:
            self._update_max_tp_level(symbol, current_tp_level)
            logger.info(f"🎯 Nouveau palier TP pour {symbol}: +{current_tp_level*100:.1f}% (était +{historical_max_tp*100:.1f}%)")
            historical_max_tp = current_tp_level
        
        # VENDRE si rechute significative depuis le palier max (tolérance plus large pour éviter sur-trading)
        # Ex: palier 2% -> vendre si on descend sous 1.6% (2% - 20% de 2%)
        tolerance_factor = 0.80  # Garde 80% du palier atteint (plus permissif, était 70%)
        adjusted_threshold = historical_max_tp * tolerance_factor
        
        if gain_percent < adjusted_threshold:
            logger.warning(f"📉 Rechute significative pour {symbol}: +{gain_percent*100:.2f}% < seuil ajusté +{adjusted_threshold*100:.2f}% (palier max: +{historical_max_tp*100:.1f}%)")
            self._clear_max_tp_level(symbol)  # Nettoyer après vente
            return True, f"Rechute sous seuil TP ajusté +{adjusted_threshold*100:.2f}% (palier: +{historical_max_tp*100:.1f}%, gain: +{gain_percent*100:.2f}%)"
        
        # Sinon, continuer à surveiller
        return False, f"Au-dessus palier TP +{historical_max_tp*100:.1f}% (+{gain_percent*100:.2f}%), surveillance active"
    
    def _update_max_tp_level(self, symbol: str, tp_level: float) -> None:
        """
        Met à jour le palier TP maximum atteint pour un symbole.
        
        Args:
            symbol: Symbole
            tp_level: Nouveau palier TP maximum (ex: 0.025 = 2.5%)
        """
        try:
            tp_key = f"max_tp_level:{symbol}"
            tp_data = {
                "level": tp_level,
                "timestamp": int(time.time() * 1000)
            }
            # TTL de 24 heures pour le palier TP
            self.redis_client.set(tp_key, json.dumps(tp_data), expiration=86400)
            logger.debug(f"🎯 Palier TP mis à jour pour {symbol}: +{tp_level*100:.1f}%")
        except Exception as e:
            logger.error(f"Erreur mise à jour palier TP pour {symbol}: {e}")
    
    def _clear_max_tp_level(self, symbol: str) -> None:
        """
        Supprime le palier TP maximum pour un symbole.
        
        Args:
            symbol: Symbole
        """
        try:
            tp_key = f"max_tp_level:{symbol}"
            self.redis_client.delete(tp_key)
            logger.info(f"🧹 Palier TP max supprimé pour {symbol}")
        except Exception as e:
            logger.error(f"Erreur suppression palier TP pour {symbol}: {e}")
    
    def _cleanup_references(self, symbol: str) -> None:
        """
        Nettoie toutes les références pour un symbole après une vente.
        
        Args:
            symbol: Symbole
        """
        self._clear_sell_reference(symbol)
        self._clear_cycle_max_price(symbol)
        self._clear_max_tp_level(symbol)  # Ajouter nettoyage palier TP
        logger.info(f"🧹 Toutes les références nettoyées pour {symbol}")
    
    def _get_price_precision(self, price: float) -> int:
        """
        Détermine la précision d'affichage selon le niveau de prix.
        """
        if price >= 1000:
            return 2
        elif price >= 100:
            return 3
        elif price >= 1:
            return 6
        elif price >= 0.01:
            return 6
        elif price >= 0.0001:
            return 10
        else:
            return 12