#!/usr/bin/env python3
"""
Module pour la validation des signaux de trading.
Contient toutes les méthodes de validation extraites du signal_aggregator principal.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import json
import numpy as np

logger = logging.getLogger(__name__)


class SignalValidator:
    """Classe pour valider les signaux de trading selon différents critères"""
    
    def __init__(self, redis_client, ema_incremental_cache: Dict = None):
        self.redis = redis_client
        self.ema_incremental_cache = ema_incremental_cache or {}
        
    async def validate_signal_with_higher_timeframe(self, signal: Dict[str, Any]) -> bool:
        """
        Valide un signal 1m avec le contexte 15m enrichi pour éviter les faux signaux.
        
        Logique de validation enrichie :
        - Signal BUY : validé si tendance 15m haussière + BB position favorable + Stochastic non surachat
        - Signal SELL : validé si tendance 15m baissière + BB position favorable + Stochastic non survente
        - Utilise ATR pour adapter dynamiquement les seuils
        
        Args:
            signal: Signal 1m à valider
            
        Returns:
            True si le signal est validé, False sinon
        """
        try:
            symbol = signal['symbol']
            side = signal['side']
            
            logger.debug(f"🔍 Validation 5m pour {symbol} {side}")

            # Récupérer les données 5m récentes depuis Redis (MODE SWING CRYPTO)
            market_data_key = f"market_data:{symbol}:5m"
            data_5m = self.redis.get(market_data_key)
            
            if not data_5m:
                # Si pas de données 5m, on accepte le signal (mode dégradé)
                logger.warning(f"Pas de données 5m pour {symbol}, validation swing en mode dégradé")
                return True
            
            # Le RedisClient parse automatiquement les données JSON
            if not isinstance(data_5m, dict):
                logger.warning(f"Données 5m invalides pour {symbol}: type {type(data_5m)}")
                return True
            
            # CORRECTION: Vérifier la fraîcheur des données 5m
            last_update = data_5m.get('last_update')
            if last_update:
                try:
                    if isinstance(last_update, str):
                        update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                    else:
                        update_time = datetime.fromtimestamp(last_update, tz=timezone.utc)
                    
                    age_seconds = (datetime.now(timezone.utc) - update_time).total_seconds()
                    if age_seconds > 310:  # Plus de 5 minutes = données stales
                        logger.warning(f"Données 5m trop anciennes pour {symbol} ({age_seconds:.0f}s), bypass validation")
                        return True
                except Exception as e:
                    logger.warning(f"Erreur parsing timestamp 5m pour {symbol}: {e}")
                    return True
            
            # Calculer la tendance 5m avec une EMA simple (MODE SCALPING)
            prices = data_5m.get('prices', [])
            if len(prices) < 5:
                # Pas assez de données pour une tendance fiable (seuil réduit pour scalping)
                return True
            
            # HARMONISATION: EMA 21 vs EMA 50 pour cohérence avec les stratégies
            if len(prices) < 50:
                return True  # Pas assez de données pour EMA 50
            
            # AMÉLIORATION : Utiliser EMAs incrémentales pour courbes lisses
            current_price = prices[-1] if prices else 0
            ema_21 = self._get_or_calculate_ema_incremental(symbol, current_price, 21)
            ema_50 = self._get_or_calculate_ema_incremental(symbol, current_price, 50)
            
            # LOGIQUE SOPHISTIQUÉE : Analyser la force et le momentum de la tendance
            
            # Calculer la vélocité des EMAs (momentum) - méthode améliorée
            # Récupérer les EMAs précédentes depuis le cache pour vélocité
            timeframe = "1m"
            cache = self.ema_incremental_cache.get(symbol, {}).get(timeframe, {})
            ema_21_prev = cache.get('ema_21_prev', ema_21)
            ema_50_prev = cache.get('ema_50_prev', ema_50)
            
            # Calculer vélocité avec EMAs lisses
            ema_21_velocity = (ema_21 - ema_21_prev) / ema_21_prev if ema_21_prev > 0 else 0
            ema_50_velocity = (ema_50 - ema_50_prev) / ema_50_prev if ema_50_prev > 0 else 0
            
            # Stocker les valeurs actuelles comme "précédentes" pour le prochain calcul
            cache['ema_21_prev'] = ema_21
            cache['ema_50_prev'] = ema_50
            
            # Calculer la force de la tendance
            trend_strength = abs(ema_21 - ema_50) / ema_50 if ema_50 > 0 else 0
            
            # Classification sophistiquée de la tendance
            if ema_21 > ema_50 * 1.015:  # +1.5% = forte haussière
                trend_5m = "STRONG_BULLISH"
            elif ema_21 > ema_50 * 1.005:  # +0.5% = faible haussière
                trend_5m = "WEAK_BULLISH"
            elif ema_21 < ema_50 * 0.985:  # -1.5% = forte baissière  
                trend_5m = "STRONG_BEARISH"
            elif ema_21 < ema_50 * 0.995:  # -0.5% = faible baissière
                trend_5m = "WEAK_BEARISH"
            else:
                trend_5m = "NEUTRAL"
            
            # Détecter si la tendance s'affaiblit (divergence)
            trend_weakening = False
            if trend_5m in ["STRONG_BULLISH", "WEAK_BULLISH"] and ema_21_velocity < 0:
                trend_weakening = True  # Tendance haussière qui ralentit
            elif trend_5m in ["STRONG_BEARISH", "WEAK_BEARISH"] and ema_21_velocity > 0:
                trend_weakening = True  # Tendance baissière qui ralentit
            
            # DEBUG: Log détaillé pour comprendre les rejets
            logger.info(f"🔍 {symbol} | Prix={current_price:.4f} | EMA21={ema_21:.4f} | EMA50={ema_50:.4f} | Tendance={trend_5m} | Signal={side} | Velocity21={ema_21_velocity*100:.2f}% | Weakening={trend_weakening}")
            
            # LOGIQUE SOPHISTIQUÉE DE VALIDATION
            rejection_reason = None
            
            # NOUVEAU: Validation stricte de la position relative aux EMAs
            price_above_ema21 = current_price > ema_21
            price_above_ema50 = current_price > ema_50
            
            # Position relative en pourcentage
            distance_to_ema21 = ((current_price - ema_21) / ema_21) * 100
            distance_to_ema50 = ((current_price - ema_50) / ema_50) * 100
            
            if side == "BUY":
                # NOUVEAU: Validation stricte position EMA pour BUY
                if not price_above_ema21:
                    rejection_reason = f"prix sous EMA21 ({distance_to_ema21:.2f}%), momentum baissier"
                elif trend_5m in ["STRONG_BEARISH", "WEAK_BEARISH"] and not price_above_ema50:
                    rejection_reason = f"contre-tendance baissière 5m et prix sous EMA50"
                # Éviter d'acheter dans une forte montée (risque de sommet)
                elif trend_5m == "STRONG_BULLISH" and not trend_weakening:
                    rejection_reason = "forte tendance haussière en cours, risque de sommet"
                # Éviter d'acheter un crash violent (couteau qui tombe)
                elif trend_5m == "STRONG_BEARISH" and ema_21_velocity < -0.01:  # Accélération baissière > 1%
                    rejection_reason = "crash violent en cours, éviter le couteau qui tombe"
                # Prix trop éloigné au-dessus des EMAs (surachat)
                elif distance_to_ema21 > 2.0:  # Plus de 2% au-dessus
                    rejection_reason = f"prix trop éloigné de EMA21 (+{distance_to_ema21:.2f}%), risque de retour"
                    
            elif side == "SELL":
                # NOUVEAU: Validation stricte position EMA pour SELL
                if price_above_ema50 and trend_5m not in ["STRONG_BEARISH", "WEAK_BEARISH"]:
                    rejection_reason = f"prix au-dessus EMA50 ({distance_to_ema50:.2f}%), momentum haussier"
                elif trend_5m in ["STRONG_BULLISH", "WEAK_BULLISH"] and price_above_ema21:
                    rejection_reason = f"contre-tendance haussière 5m et prix au-dessus EMA21"
                # Éviter de vendre dans une forte baisse (risque de creux)  
                elif trend_5m == "STRONG_BEARISH" and not trend_weakening:
                    rejection_reason = "forte tendance baissière en cours, risque de creux"
                # Éviter de vendre une pump violente (FOMO manqué)
                elif trend_5m == "STRONG_BULLISH" and ema_21_velocity > 0.01:  # Accélération haussière > 1%
                    rejection_reason = "pump violent en cours, éviter de rater la montée"
                # Prix trop éloigné en-dessous des EMAs (survente)
                elif distance_to_ema21 < -2.0:  # Plus de 2% en-dessous
                    rejection_reason = f"prix trop éloigné sous EMA21 ({distance_to_ema21:.2f}%), risque de rebond"
            
            # Appliquer le rejet si raison trouvée
            if rejection_reason:
                logger.info(f"Signal {side} {symbol} rejeté : {rejection_reason}")
                return False
            
            # NOUVELLE VALIDATION ENRICHIE : Bollinger Bands position pour timing optimal
            bb_position = data_5m.get('bb_position')
            if bb_position is not None:
                if side == "BUY" and bb_position > 0.8:  # Prix proche de la bande haute
                    logger.info(f"Signal BUY {symbol} rejeté : BB position trop haute ({bb_position:.2f})")
                    return False
                elif side == "SELL" and bb_position < 0.2:  # Prix proche de la bande basse
                    logger.info(f"Signal SELL {symbol} rejeté : BB position trop basse ({bb_position:.2f})")
                    return False
            
            # NOUVELLE VALIDATION : Stochastic pour confirmer oversold/overbought
            stoch_k = data_5m.get('stoch_k')
            stoch_d = data_5m.get('stoch_d')
            if stoch_k is not None and stoch_d is not None:
                if side == "BUY" and stoch_k > 85 and stoch_d > 85:  # Surachat confirmé
                    logger.info(f"Signal BUY {symbol} rejeté : Stochastic surachat K={stoch_k:.1f} D={stoch_d:.1f}")
                    return False
                elif side == "SELL" and stoch_k < 15 and stoch_d < 15:  # Survente confirmé
                    logger.info(f"Signal SELL {symbol} rejeté : Stochastic survente K={stoch_k:.1f} D={stoch_d:.1f}")
                    return False
            
            # VALIDATION ADAPTATIVE : Ajuster seuils RSI selon ATR (volatilité)
            atr_15m = data_5m.get('atr_14')
            current_price = data_5m.get('close', prices[-1] if prices else 0)
            atr_percent = (atr_15m / current_price * 100) if atr_15m and current_price > 0 else 2.0
            
            # Seuils RSI adaptatifs selon volatilité
            if atr_percent > 5.0:  # Haute volatilité
                rsi_buy_threshold = 75  # Plus tolérant
                rsi_sell_threshold = 25
            elif atr_percent > 3.0:  # Volatilité moyenne
                rsi_buy_threshold = 80
                rsi_sell_threshold = 20
            else:  # Faible volatilité
                rsi_buy_threshold = 85  # Plus strict
                rsi_sell_threshold = 15
            
            # Validation RSI avec seuils adaptatifs
            rsi_5m = data_5m.get('rsi_14')
            if rsi_5m:
                if side == "BUY" and rsi_5m > rsi_buy_threshold:
                    logger.info(f"Signal BUY {symbol} rejeté : RSI 5m surachat ({rsi_5m:.1f} > {rsi_buy_threshold}) - ATR={atr_percent:.1f}%")
                    return False
                elif side == "SELL" and rsi_5m < rsi_sell_threshold:
                    logger.info(f"Signal SELL {symbol} rejeté : RSI 5m survente ({rsi_5m:.1f} < {rsi_sell_threshold}) - ATR={atr_percent:.1f}%")
                    return False

            logger.debug(f"Signal {side} {symbol} validé par analyse multi-indicateurs 5m - tendance={trend_5m} BB={bb_position} ATR={atr_percent:.1f}%")
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation multi-timeframe: {e}")
            return True  # Mode dégradé : accepter le signal
    
    def check_price_momentum_divergence(self, symbol: str, prices: List[float], 
                                       rsi: Optional[float], side: str) -> bool:
        """
        Détecte les divergences prix/RSI qui indiquent des retournements potentiels.
        
        Args:
            symbol: Symbole de trading
            prices: Liste des prix récents
            rsi: Valeur RSI actuelle
            side: Direction du signal (BUY/SELL)
            
        Returns:
            True si divergence détectée (signal à rejeter), False sinon
        """
        if not rsi or len(prices) < 10:
            return False
        
        try:
            # Tendance des prix sur les 5 dernières périodes
            price_trend_up = prices[-1] > prices[-5]
            price_trend_strong_up = prices[-1] > prices[-5] * 1.005  # +0.5%
            price_trend_down = prices[-1] < prices[-5]
            price_trend_strong_down = prices[-1] < prices[-5] * 0.995  # -0.5%
            
            # RSI extrêmes
            rsi_overbought = rsi > 70
            rsi_oversold = rsi < 30
            rsi_neutral_high = 60 < rsi < 70
            rsi_neutral_low = 30 < rsi < 40
            
            # Divergences baissières (prix monte, RSI faiblit)
            if side == "BUY":
                # Prix en forte hausse mais RSI surachat = divergence baissière
                if price_trend_strong_up and rsi_overbought:
                    logger.info(f"🔴 Divergence baissière {symbol}: prix ↑↑ mais RSI={rsi:.1f} (surachat)")
                    return True
                # Prix monte mais RSI stagne ou baisse
                elif price_trend_up and rsi < 50:
                    logger.info(f"⚠️ Divergence potentielle {symbol}: prix ↑ mais RSI={rsi:.1f} < 50")
                    return True
                    
            # Divergences haussières (prix baisse, RSI se renforce)
            elif side == "SELL":
                # Prix en forte baisse mais RSI survente = divergence haussière
                if price_trend_strong_down and rsi_oversold:
                    logger.info(f"🔴 Divergence haussière {symbol}: prix ↓↓ mais RSI={rsi:.1f} (survente)")
                    return True
                # Prix baisse mais RSI stagne ou monte
                elif price_trend_down and rsi > 50:
                    logger.info(f"⚠️ Divergence potentielle {symbol}: prix ↓ mais RSI={rsi:.1f} > 50")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur détection divergence {symbol}: {e}")
            return False
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calcule une EMA simple (fallback)"""
        if not prices or period <= 0:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _get_or_calculate_ema_incremental(self, symbol: str, current_price: float, period: int) -> float:
        """
        Récupère ou calcule EMA de manière incrémentale pour éviter les dents de scie.
        Utilise le même principe que le Gateway WebSocket.
        """
        timeframe = "1m"  # Timeframe principal du Signal Aggregator
        cache_key = f'ema_{period}'
        
        # Initialiser le cache si nécessaire
        if symbol not in self.ema_incremental_cache:
            self.ema_incremental_cache[symbol] = {}
        if timeframe not in self.ema_incremental_cache[symbol]:
            self.ema_incremental_cache[symbol][timeframe] = {}
            
        cache = self.ema_incremental_cache[symbol][timeframe]
        
        # Récupérer EMA précédente du cache
        prev_ema = cache.get(cache_key)
        
        if prev_ema is None:
            # Première fois : utiliser le prix actuel comme base
            new_ema = current_price
        else:
            # Calcul incrémental standard
            from shared.src.technical_indicators import calculate_ema_incremental
            new_ema = calculate_ema_incremental(current_price, prev_ema, period)
        
        # Mettre à jour le cache
        cache[cache_key] = new_ema
        
        return new_ema
    
    async def validate_signal_correlation(self, signals: List[Dict]) -> float:
        """
        Valide la corrélation entre les signaux multiples
        
        Returns:
            Score de corrélation (0-1)
        """
        if len(signals) < 2:
            return 1.0
        
        # Analyser les métadonnées des signaux
        correlation_score = 1.0
        
        # Vérifier que les signaux pointent dans la même direction générale
        price_targets = []
        stop_losses = []
        
        for signal in signals:
            metadata = signal.get('metadata', {})
            
            # Extraire les niveaux clés
            if 'stop_price' in metadata:
                stop_losses.append(metadata['stop_price'])
            
            # Analyser les indicateurs sous-jacents
            if 'rsi' in metadata:
                rsi = metadata['rsi']
                side = signal.get('side', signal.get('side'))
                
                # Pénaliser si RSI contradictoire
                if side == 'BUY' and rsi > 70:
                    correlation_score *= 0.7
                elif side == 'SELL' and rsi < 30:
                    correlation_score *= 0.7
        
        # Vérifier la cohérence des stops
        if len(stop_losses) >= 2:
            stop_std = np.std(stop_losses) / np.mean(stop_losses)
            if stop_std > 0.02:  # Plus de 2% d'écart
                correlation_score *= 0.8
        
        return correlation_score
    
    async def check_btc_correlation(self, symbol: str) -> float:
        """
        Vérifie la corrélation avec BTC sur les dernières heures
        """
        try:
            # Récupérer les prix récents
            try:
                symbol_prices = self.redis.lrange(f"prices_1h:{symbol}", 0, 24)
                btc_prices = self.redis.lrange(f"prices_1h:BTCUSDC", 0, 24)
            except AttributeError:
                # Fallback pour RedisClientPool customisé
                symbol_prices_str = self.redis.get(f"prices_1h:{symbol}")
                btc_prices_str = self.redis.get(f"prices_1h:BTCUSDC")
                
                symbol_prices = []
                btc_prices = []
                
                if symbol_prices_str:
                    parsed = json.loads(symbol_prices_str) if isinstance(symbol_prices_str, str) else symbol_prices_str
                    symbol_prices = parsed[-24:] if isinstance(parsed, list) else []
                
                if btc_prices_str:
                    parsed = json.loads(btc_prices_str) if isinstance(btc_prices_str, str) else btc_prices_str
                    btc_prices = parsed[-24:] if isinstance(parsed, list) else []
            
            if len(symbol_prices) < 10 or len(btc_prices) < 10:
                return 0.0
            
            # Convertir en float
            symbol_prices = [float(p) for p in symbol_prices]
            btc_prices = [float(p) for p in btc_prices]
            
            # Calculer la corrélation
            symbol_returns = np.diff(np.array(symbol_prices))
            btc_returns = np.diff(np.array(btc_prices))
            
            if len(symbol_returns) > 0 and len(btc_returns) > 0:
                min_length = min(len(symbol_returns), len(btc_returns))
                correlation = np.corrcoef(
                    symbol_returns[:min_length],
                    btc_returns[:min_length]
                )[0, 1]
                
                return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Erreur calcul corrélation BTC: {e}")
            
        return 0.0
    
    async def apply_market_context_filters(self, signal: Dict, min_confidence_threshold: float) -> bool:
        """
        Applique des filtres basés sur le contexte de marché global
        
        Returns:
            True si le signal passe les filtres, False sinon
        """
        symbol = signal['symbol']
        
        # 1. Vérifier les heures de trading (éviter les heures creuses)
        current_hour = datetime.now().hour
        if current_hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Heures creuses crypto
            # Augmenter le seuil de confiance pendant ces heures
            min_confidence = 0.8
        else:
            min_confidence = min_confidence_threshold
        
        if signal.get('confidence', 0) < min_confidence:
            logger.debug(f"Signal filtré: confiance insuffisante pendant heures creuses")
            return False
        
        # 2. Vérifier le spread bid/ask
        spread_key = f"spread:{symbol}"
        spread = self.redis.get(spread_key)
        if spread and float(spread) > 0.003:  # Spread > 0.3%
            logger.info(f"Signal filtré: spread trop large ({float(spread):.3%})")
            return False
        
        # 3. Vérifier la liquidité récente
        volume_key = f"volume_1h:{symbol}"
        recent_volume = self.redis.get(volume_key)
        if recent_volume and float(recent_volume) < 100000:  # Volume < 100k
            logger.info(f"Signal filtré: volume insuffisant ({float(recent_volume):.0f})")
            return False
        
        # 4. Vérifier les corrélations avec BTC (pour les altcoins)
        if symbol != "BTCUSDC":
            btc_correlation = await self.check_btc_correlation(symbol)
            if btc_correlation < -0.7:  # Forte corrélation négative
                logger.info(f"Signal filtré: corrélation BTC négative ({btc_correlation:.2f})")
                return False
        
        return True
    
    async def check_regime_transition(self, symbol: str) -> bool:
        """
        Vérifie si on est en transition de régime (moment délicat)
        
        Returns:
            True si en transition, False sinon
        """
        try:
            # Récupérer l'historique des régimes
            regime_history_key = f"regime_history:{symbol}"
            try:
                # Essayer d'utiliser la méthode Redis standard
                history = self.redis.lrange(regime_history_key, 0, 10)
            except AttributeError:
                # Fallback pour RedisClientPool customisé
                history_str = self.redis.get(regime_history_key)
                if history_str:
                    history = json.loads(history_str) if isinstance(history_str, str) else history_str
                    if isinstance(history, list):
                        history = history[:10]  # Prendre les 10 premiers
                    else:
                        history = []
                else:
                    history = []
            
            if len(history) < 3:
                return False
            
            # Analyser les changements récents
            recent_regimes = []
            for h in history[:3]:
                if isinstance(h, str):
                    regime_data = json.loads(h)
                else:
                    regime_data = h
                recent_regimes.append(regime_data.get('regime'))
            
            unique_regimes = len(set(recent_regimes))
            
            # Si plus de 2 régimes différents dans les 3 derniers = transition
            if unique_regimes > 2:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur vérification transition régime: {e}")
            return False