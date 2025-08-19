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
        
        # Configuration trailing sell - ÉQUILIBRE GAINS/PROTECTION
        self.min_gain_for_trailing = 0.010  # 1.0% de gain minimum avant activation du trailing
        self.sell_margin = 0.008  # 0.8% de marge pour les micro-variations
        self.max_drop_threshold = 0.015  # 1.5% de chute max depuis le pic (cohérent avec activation à 1%)
        self.immediate_sell_drop = 0.020  # 2.0% de chute = vente immédiate
        
        # Configuration stop-loss adaptatif - PROTECTION CAPITAL
        self.stop_loss_percent_base = 0.015  # 1.5% de base - protection stricte
        self.stop_loss_percent_bullish = 0.020  # 2.0% en tendance haussière - légèrement plus souple
        self.stop_loss_percent_strong_bullish = 0.025  # 2.5% en tendance très haussière - reste prudent
        
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
            
            # === POSITION GAGNANTE : LOGIQUE TRAILING SELL AMÉLIORÉE ===
            gain_percent = (current_price - entry_price) / entry_price
            logger.info(f"🔍 Position gagnante détectée: +{gain_percent*100:.2f}%, vérification trailing sell")
            
            # Vérifier si le gain minimum est atteint pour activer le trailing
            if gain_percent < self.min_gain_for_trailing:
                logger.info(f"📊 Gain insuffisant pour trailing ({gain_percent*100:.2f}% < {self.min_gain_for_trailing*100:.1f}%), position continue")
                return False, f"Gain insuffisant pour activer le trailing ({gain_percent*100:.2f}% < {self.min_gain_for_trailing*100:.1f}%)"
            
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
                if drop_from_max > self.sell_margin:  # Si déjà chuté de >0.8% depuis le max
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
            
            # Ensuite logique classique de trailing
            sell_threshold = previous_sell_price * (1 - self.sell_margin)
            logger.info(f"🔍 Seuil de vente calculé: {sell_threshold:.{precision}f} (marge {self.sell_margin*100:.1f}%)")
            
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
    
    def _cleanup_references(self, symbol: str) -> None:
        """
        Nettoie toutes les références pour un symbole après une vente.
        
        Args:
            symbol: Symbole
        """
        self._clear_sell_reference(symbol)
        self._clear_cycle_max_price(symbol)
        logger.info(f"🧹 Toutes les références nettoyées pour {symbol}")
    
    def _calculate_adaptive_threshold(self, symbol: str, entry_price: float, 
                                     entry_time: float) -> float:
        """
        Calcule le seuil de stop-loss adaptatif basé sur les données d'analyse.
        
        Args:
            symbol: Symbole à analyser
            entry_price: Prix d'entrée de la position
            entry_time: Timestamp d'entrée (epoch)
            
        Returns:
            Seuil de perte acceptable avant stop-loss (ex: 0.015 = 1.5%)
        """
        try:
            # Récupérer les données d'analyse si disponibles
            analysis = self._get_latest_analysis_data(symbol) if self.db_connection else None
            
            if not analysis:
                logger.debug(f"Pas de données d'analyse pour {symbol}, seuil par défaut")
                return self.stop_loss_percent_base
            
            # Récupérer le régime de marché
            regime = analysis.get('market_regime', 'UNKNOWN')
            
            # Calculer les facteurs d'ajustement
            regime_factor = self._calculate_regime_factor(analysis)
            volatility_factor = self._calculate_volatility_factor(analysis)
            support_factor = self._calculate_support_factor(analysis, entry_price)
            time_factor = self._calculate_time_factor(entry_time)
            
            # Calcul du seuil final - RESSERRÉ POUR PROTECTION
            # En bear market, utiliser le seuil le PLUS STRICT directement
            if regime == 'TRENDING_BEAR' or regime == 'BREAKOUT_BEAR':
                base_threshold = self.stop_loss_percent_base  # 1.5% - le plus strict
            elif regime == 'TRENDING_BULL' or regime == 'BREAKOUT_BULL':
                base_threshold = self.stop_loss_percent_strong_bullish  # 3.5% - le plus tolérant
            else:  # RANGING, TRANSITION, VOLATILE
                base_threshold = self.stop_loss_percent_bullish  # 2.5% - intermédiaire
            
            adaptive_threshold = float(base_threshold) * float(regime_factor) * float(volatility_factor) * float(support_factor) * float(time_factor)
            
            # Contraintes min/max - ULTRA STRICTES POUR PROTECTION CAPITAL
            adaptive_threshold = max(0.008, min(0.018, adaptive_threshold))  # 0.8%-1.8% (ultra strict)
            
            logger.debug(f"🧠 Stop-loss adaptatif {symbol}: {adaptive_threshold*100:.2f}%")
            
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
        
        # PROTECTION CAPITAL : seuils plus stricts globalement
        regime_multipliers = {
            'TRENDING_BULL': 1.5,      # Bull = modérément tolérant (réduit de 2.0)
            'BREAKOUT_BULL': 1.3,      # Breakout bull = moins tolérant (réduit de 1.8)
            'RANGING': 1.0,            # Range = neutre (réduit de 1.2)
            'TRANSITION': 0.9,         # Transition = légèrement strict (réduit de 1.0)
            'TRENDING_BEAR': 0.5,      # Bear = ULTRA strict (réduit de 0.6)
            'VOLATILE': 0.6,           # Volatile = plus strict (réduit de 0.8)
            'BREAKOUT_BEAR': 0.4       # Breakout bear = MAXIMUM strict (réduit de 0.5)
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