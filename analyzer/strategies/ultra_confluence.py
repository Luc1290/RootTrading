"""
Stratégie ULTRA-CONFLUENTE qui exploite toutes les données enrichies
Utilise 20+ indicateurs sur 5 timeframes pour des signaux d'une précision extrême
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import deque, defaultdict
import time

# Imports partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from shared.src.enums import OrderSide, SignalStrength
from shared.src.schemas import StrategySignal
from analyzer.strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class UltraConfluenceStrategy(BaseStrategy):
    """
    Stratégie ultra-avancée qui combine TOUS les indicateurs disponibles
    pour générer des signaux d'une précision institutionnelle
    """
    
    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.strategy_name = "UltraConfluence"
        
        # Cache multi-timeframe
        self.mtf_data = defaultdict(lambda: deque(maxlen=100))
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']
        
        # Scores de confluence
        self.confluence_thresholds = {
            SignalStrength.VERY_STRONG: 18,  # 18+ confirmations sur 20 possibles
            SignalStrength.STRONG: 14,       # 14+ confirmations
            SignalStrength.MODERATE: 10,     # 10+ confirmations  
            SignalStrength.WEAK: 6           # 6+ confirmations
        }
        
        # Poids par timeframe (plus le timeframe est élevé, plus le poids est important)
        self.timeframe_weights = {
            '1m': 1.0,
            '5m': 1.5,
            '15m': 2.0,
            '1h': 3.0,
            '4h': 4.0
        }
        
        logger.info(f"🔥 UltraConfluence initialisée pour {symbol} - 20+ indicateurs sur 5 timeframes")
    
    @property
    def name(self) -> str:
        """Nom de la stratégie"""
        return self.strategy_name
        
    def generate_signal(self) -> Optional[StrategySignal]:
        """Méthode principale pour générer un signal"""
        # Utiliser les dernières données disponibles dans le cache multi-timeframe
        if not self.mtf_data or not any(self.mtf_data.values()):
            return None
        
        # Analyser avec les données les plus récentes du timeframe 5m
        if '5m' in self.mtf_data and len(self.mtf_data['5m']) > 0:
            latest_data = list(self.mtf_data['5m'])[-1]
            return self.process_market_data(latest_data)
        
        return None
    
    def add_market_data(self, data: Dict[str, Any]) -> None:
        """Override pour alimenter le cache multi-timeframe"""
        # Appeler la méthode parent d'abord
        super().add_market_data(data)
        
        # Ajouter au cache multi-timeframe si les données sont enrichies
        if data.get('enhanced') and data.get('timeframe'):
            timeframe = data['timeframe']
            if timeframe in self.timeframes:
                self.mtf_data[timeframe].append(data)
        
    def should_generate_signal(self, data: Dict[str, Any]) -> bool:
        """Détermine si on doit analyser ce timeframe"""
        # Analyser tous les timeframes avec données enrichies
        return (data.get('enhanced') and 
                data.get('is_closed') and 
                data.get('timeframe') in self.timeframes)
                
    def process_market_data(self, data: Dict[str, Any]) -> Optional[StrategySignal]:
        """Traite les données multi-timeframes pour confluence maximale"""
        try:
            if not self.should_generate_signal(data):
                return None
                
            timeframe = data['timeframe']
            
            # Ajouter au cache multi-timeframe
            self.mtf_data[timeframe].append(data)
            
            # Attendre d'avoir au moins 20 bougies sur le timeframe principal (5m)
            if timeframe == '5m' and len(self.mtf_data['5m']) >= 20:
                return self._analyze_ultra_confluence()
                
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur UltraConfluence {self.symbol}: {e}")
            return None
            
    def _analyze_ultra_confluence(self) -> Optional[StrategySignal]:
        """Analyse de confluence ultra-avancée sur tous les timeframes"""
        try:
            confluence_scores = {'BUY': 0, 'SELL': 0}
            all_confirmations = []
            signal_details = {}
            
            # Analyser chaque timeframe disponible
            for tf in self.timeframes:
                if tf in self.mtf_data and len(self.mtf_data[tf]) > 0:
                    tf_analysis = self._analyze_timeframe(tf)
                    if tf_analysis:
                        weight = self.timeframe_weights[tf]
                        
                        # Appliquer le poids du timeframe
                        confluence_scores['BUY'] += tf_analysis['buy_score'] * weight
                        confluence_scores['SELL'] += tf_analysis['sell_score'] * weight
                        
                        # Collecter les confirmations avec timeframe
                        for conf in tf_analysis['confirmations']:
                            all_confirmations.append(f"{tf}_{conf}")
                            
                        signal_details[tf] = tf_analysis
                        
            # Déterminer la direction dominante
            total_buy = confluence_scores['BUY']
            total_sell = confluence_scores['SELL']
            
            if total_buy == 0 and total_sell == 0:
                return None
                
            # Calculer la force relative
            total_score = total_buy + total_sell
            buy_ratio = total_buy / total_score
            sell_ratio = total_sell / total_score
            
            # Exiger une dominance claire (60%+)
            if max(buy_ratio, sell_ratio) < 0.6:
                logger.debug(f"Pas de dominance claire: BUY {buy_ratio:.2f} vs SELL {sell_ratio:.2f}")
                return None
                
            # Déterminer côté et force
            side = OrderSide.BUY if buy_ratio > sell_ratio else OrderSide.SELL
            dominant_score = max(total_buy, total_sell)
            
            # Déterminer la force basée sur le nombre de confirmations
            confirmation_count = len(all_confirmations)
            strength = self._determine_strength_from_confirmations(confirmation_count)
            
            if strength == SignalStrength.WEAK:
                logger.debug(f"Signal trop faible: {confirmation_count} confirmations")
                return None
                
            # Calculer la confiance finale
            confidence = min(dominant_score / (len(self.timeframes) * 4 * max(self.timeframe_weights.values())), 1.0)
            
            # Obtenir les données du timeframe d'exécution (5m)
            execution_data = list(self.mtf_data['5m'])[-1]
            
            # Calculer les niveaux de prix avec confluence multi-timeframe
            price_levels = self._calculate_confluence_price_levels(side, signal_details)
            
            # Métadonnées ultra-riches
            metadata = {
                'ultra_confluence': True,
                'timeframes_analyzed': list(signal_details.keys()),
                'confirmation_count': confirmation_count,
                'total_confirmations': all_confirmations,
                'confluence_scores': confluence_scores,
                'timeframe_analysis': signal_details,
                'price_levels': price_levels,
                'execution_timeframe': '5m',
                
                # Indicateurs clés du timeframe d'exécution
                'rsi_14': execution_data.get('rsi_14'),
                'macd_histogram': execution_data.get('macd_histogram'),
                'bb_position': execution_data.get('bb_position'),
                'adx_14': execution_data.get('adx_14'),
                'volume_spike': execution_data.get('volume_spike', False),
                'spread_pct': execution_data.get('spread_pct'),
                
                # Confluence multi-timeframe
                'mtf_trend_alignment': self._check_trend_alignment(signal_details),
                'mtf_momentum_strength': self._calculate_momentum_strength(signal_details),
                'volume_confirmation': self._check_volume_confirmation(signal_details)
            }
            
            # Créer le signal ultra-enrichi
            signal = StrategySignal(
                strategy=self.strategy_name,
                symbol=self.symbol,
                side=side,
                price=execution_data['close'],
                confidence=confidence,
                strength=strength,
                timestamp=time.time(),
                metadata=metadata
            )
            
            logger.info(f"🎯 UltraConfluence {self.symbol}: {side.value} @ {execution_data['close']:.4f} "
                       f"| Conf: {confidence:.2f} | Confirmations: {confirmation_count} "
                       f"| TF: {len(signal_details)} | Force: {strength.value}")
                       
            return signal
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse confluence: {e}")
            return None
            
    def _analyze_timeframe(self, timeframe: str) -> Optional[Dict]:
        """Analyse approfondie d'un timeframe spécifique"""
        try:
            data_list = list(self.mtf_data[timeframe])
            if not data_list:
                return None
                
            latest = data_list[-1]
            buy_score = 0
            sell_score = 0
            confirmations = []
            
            # 1. RSI Analysis (2 confirmations possibles)
            rsi = latest.get('rsi_14')
            if rsi:
                if rsi < 30:
                    buy_score += 1
                    confirmations.append('RSI_OVERSOLD')
                elif rsi > 70:
                    sell_score += 1
                    confirmations.append('RSI_OVERBOUGHT')
                elif 30 < rsi < 45:
                    buy_score += 0.5
                    confirmations.append('RSI_BULLISH_ZONE')
                elif 55 < rsi < 70:
                    sell_score += 0.5
                    confirmations.append('RSI_BEARISH_ZONE')
                    
            # 2. Stochastic RSI (1 confirmation)
            stoch_rsi = latest.get('stoch_rsi')
            if stoch_rsi:
                if stoch_rsi < 20:
                    buy_score += 1
                    confirmations.append('STOCH_RSI_OVERSOLD')
                elif stoch_rsi > 80:
                    sell_score += 1
                    confirmations.append('STOCH_RSI_OVERBOUGHT')
                    
            # 3. MACD Analysis (3 confirmations possibles)
            macd_line = latest.get('macd_line')
            macd_signal = latest.get('macd_signal')
            macd_histogram = latest.get('macd_histogram')
            
            if macd_line and macd_signal:
                if macd_line > macd_signal:
                    buy_score += 1
                    confirmations.append('MACD_BULLISH')
                else:
                    sell_score += 1
                    confirmations.append('MACD_BEARISH')
                    
            if macd_histogram:
                if macd_histogram > 0:
                    buy_score += 0.5
                    confirmations.append('MACD_HIST_POSITIVE')
                else:
                    sell_score += 0.5
                    confirmations.append('MACD_HIST_NEGATIVE')
                    
            # 4. Bollinger Bands (2 confirmations)
            bb_position = latest.get('bb_position')
            bb_width = latest.get('bb_width')
            
            if bb_position is not None:
                if bb_position < 0.2:
                    buy_score += 1
                    confirmations.append('BB_OVERSOLD')
                elif bb_position > 0.8:
                    sell_score += 1
                    confirmations.append('BB_OVERBOUGHT')
                    
            if bb_width and bb_width > 3.0:  # Expansion des bandes
                buy_score += 0.5
                sell_score += 0.5
                confirmations.append('BB_EXPANSION')
                
            # 5. EMA Alignment (2 confirmations)
            ema_12 = latest.get('ema_12')
            ema_26 = latest.get('ema_26')
            price = latest['close']
            
            if ema_12 and ema_26:
                if ema_12 > ema_26 and price > ema_12:
                    buy_score += 1
                    confirmations.append('EMA_BULLISH_ALIGNMENT')
                elif ema_12 < ema_26 and price < ema_12:
                    sell_score += 1
                    confirmations.append('EMA_BEARISH_ALIGNMENT')
                    
            # 6. SMA Confluence (1 confirmation)
            sma_20 = latest.get('sma_20')
            sma_50 = latest.get('sma_50')
            
            if sma_20 and sma_50:
                if sma_20 > sma_50 and price > sma_20:
                    buy_score += 1
                    confirmations.append('SMA_BULLISH_CONFLUENCE')
                elif sma_20 < sma_50 and price < sma_20:
                    sell_score += 1
                    confirmations.append('SMA_BEARISH_CONFLUENCE')
                    
            # 7. ADX Trend Strength (1 confirmation)
            adx = latest.get('adx_14')
            if adx and adx > 25:
                confirmations.append('ADX_TRENDING')
                # Boost selon la direction dominante
                if buy_score > sell_score:
                    buy_score += 0.5
                else:
                    sell_score += 0.5
                    
            # 8. Volume Analysis (2 confirmations)
            volume_spike = latest.get('volume_spike', False)
            volume_trend = latest.get('volume_trend', 'stable')
            
            if volume_spike:
                buy_score += 0.5
                sell_score += 0.5
                confirmations.append('VOLUME_SPIKE')
                
            if volume_trend == 'increasing':
                buy_score += 0.5
                sell_score += 0.5
                confirmations.append('VOLUME_INCREASING')
                
            # 9. Momentum (1 confirmation)
            momentum = latest.get('momentum_10')
            if momentum:
                if momentum > 1:
                    buy_score += 1
                    confirmations.append('MOMENTUM_BULLISH')
                elif momentum < -1:
                    sell_score += 1
                    confirmations.append('MOMENTUM_BEARISH')
                    
            # 10. Williams %R (1 confirmation)
            williams_r = latest.get('williams_r')
            if williams_r:
                if williams_r < -80:
                    buy_score += 1
                    confirmations.append('WILLIAMS_R_OVERSOLD')
                elif williams_r > -20:
                    sell_score += 1
                    confirmations.append('WILLIAMS_R_OVERBOUGHT')
                    
            # 11. CCI (1 confirmation)
            cci = latest.get('cci_20')
            if cci:
                if cci < -100:
                    buy_score += 1
                    confirmations.append('CCI_OVERSOLD')
                elif cci > 100:
                    sell_score += 1
                    confirmations.append('CCI_OVERBOUGHT')
                    
            # 12. VWAP (1 confirmation)
            vwap = latest.get('vwap_10')
            if vwap:
                if price > vwap * 1.005:  # 0.5% au-dessus VWAP
                    buy_score += 0.5
                    confirmations.append('ABOVE_VWAP')
                elif price < vwap * 0.995:  # 0.5% en-dessous VWAP
                    sell_score += 0.5
                    confirmations.append('BELOW_VWAP')
                    
            return {
                'buy_score': buy_score,
                'sell_score': sell_score,
                'confirmations': confirmations,
                'total_possible': 20,  # Total des confirmations possibles
                'strength': len(confirmations) / 20
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse {timeframe}: {e}")
            return None
            
    def _determine_strength_from_confirmations(self, count: int) -> SignalStrength:
        """Détermine la force basée sur le nombre de confirmations"""
        for strength, threshold in self.confluence_thresholds.items():
            if count >= threshold:
                return strength
        return SignalStrength.WEAK
        
    def _calculate_confluence_price_levels(self, side: OrderSide, tf_analysis: Dict) -> Dict:
        """Calcule les niveaux de prix basés sur la confluence multi-timeframe"""
        try:
            # Utiliser les données 5m pour l'exécution
            execution_data = list(self.mtf_data['5m'])[-1]
            current_price = execution_data['close']
            
            # Support/Resistance basés sur les Bollinger Bands multi-timeframe
            support_levels = []
            resistance_levels = []
            
            for tf, analysis in tf_analysis.items():
                tf_data = list(self.mtf_data[tf])[-1]
                
                bb_lower = tf_data.get('bb_lower')
                bb_upper = tf_data.get('bb_upper')
                sma_20 = tf_data.get('sma_20')
                
                if bb_lower:
                    support_levels.append(bb_lower)
                if bb_upper:
                    resistance_levels.append(bb_upper)
                if sma_20:
                    if side == OrderSide.BUY:
                        support_levels.append(sma_20)
                    else:
                        resistance_levels.append(sma_20)
                        
            # Calculer les niveaux moyens
            avg_support = sum(support_levels) / len(support_levels) if support_levels else current_price * 0.98
            avg_resistance = sum(resistance_levels) / len(resistance_levels) if resistance_levels else current_price * 1.02
            
            # ATR pour volatilité
            atr = execution_data.get('atr_14', current_price * 0.01)
            
            if side == OrderSide.BUY:
                stop_loss = max(avg_support, current_price - (atr * 2))
                take_profit_1 = min(avg_resistance, current_price + (atr * 1.5))
                take_profit_2 = current_price + (atr * 3)
            else:
                stop_loss = min(avg_resistance, current_price + (atr * 2))
                take_profit_1 = max(avg_support, current_price - (atr * 1.5))
                take_profit_2 = current_price - (atr * 3)
                
            return {
                'entry': current_price,
                'stop_loss': stop_loss,
                'take_profit_1': take_profit_1,
                'take_profit_2': take_profit_2,
                'atr_used': atr,
                'support_confluence': avg_support,
                'resistance_confluence': avg_resistance
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul niveaux: {e}")
            return {}
            
    def _check_trend_alignment(self, tf_analysis: Dict) -> float:
        """Vérifie l'alignement des tendances multi-timeframe"""
        try:
            alignments = []
            
            for tf in ['1h', '4h']:  # Timeframes plus élevés
                if tf in tf_analysis:
                    analysis = tf_analysis[tf]
                    buy_score = analysis['buy_score']
                    sell_score = analysis['sell_score']
                    
                    if buy_score > sell_score:
                        alignments.append(1)  # Bullish
                    elif sell_score > buy_score:
                        alignments.append(-1)  # Bearish
                    else:
                        alignments.append(0)  # Neutral
                        
            if not alignments:
                return 0.5
                
            # Calculer l'alignement (1.0 = parfait, 0.0 = opposé)
            avg_alignment = sum(alignments) / len(alignments)
            return (avg_alignment + 1) / 2  # Normaliser entre 0 et 1
            
        except Exception as e:
            logger.error(f"❌ Erreur alignement tendance: {e}")
            return 0.5
            
    def _calculate_momentum_strength(self, tf_analysis: Dict) -> float:
        """Calcule la force du momentum multi-timeframe"""
        try:
            momentum_scores = []
            
            for tf, analysis in tf_analysis.items():
                confirmations = analysis['confirmations']
                
                # Compter les confirmations de momentum
                momentum_confirmations = [c for c in confirmations if 'MOMENTUM' in c or 'MACD' in c]
                momentum_score = len(momentum_confirmations) / 5  # Max 5 confirmations momentum
                momentum_scores.append(momentum_score)
                
            return sum(momentum_scores) / len(momentum_scores) if momentum_scores else 0
            
        except Exception as e:
            logger.error(f"❌ Erreur momentum: {e}")
            return 0
            
    def _check_volume_confirmation(self, tf_analysis: Dict) -> bool:
        """Vérifie la confirmation par le volume multi-timeframe"""
        try:
            volume_confirmations = 0
            total_tf = 0
            
            for tf, analysis in tf_analysis.items():
                total_tf += 1
                confirmations = analysis['confirmations']
                
                volume_signals = [c for c in confirmations if 'VOLUME' in c]
                if volume_signals:
                    volume_confirmations += 1
                    
            # Exiger confirmation sur au moins 50% des timeframes
            return volume_confirmations >= (total_tf * 0.5)
            
        except Exception as e:
            logger.error(f"❌ Erreur volume confirmation: {e}")
            return False