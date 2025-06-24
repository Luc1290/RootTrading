"""
Système de scoring ultra-avancé pour évaluer la qualité des signaux
Utilise machine learning et analyse quantitative pour scorer chaque signal
"""
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Imports partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.enums import OrderSide, SignalStrength
from shared.src.schemas import StrategySignal

logger = logging.getLogger(__name__)

class SignalQuality(Enum):
    """Niveaux de qualité des signaux"""
    INSTITUTIONAL = "institutional"    # 95-100 points
    EXCELLENT = "excellent"           # 85-94 points
    VERY_GOOD = "very_good"          # 75-84 points
    GOOD = "good"                    # 65-74 points
    FAIR = "fair"                    # 50-64 points
    POOR = "poor"                    # 0-49 points

@dataclass
class SignalScore:
    """Score détaillé d'un signal"""
    total_score: float
    quality: SignalQuality
    confidence_score: float
    technical_score: float
    confluence_score: float
    sentiment_score: float
    risk_reward_score: float
    timing_score: float
    volume_score: float
    breakdown: Dict[str, float]
    recommendation: str

class UltraSignalScorer:
    """
    Système de scoring ultra-avancé pour signaux de trading
    Évalue chaque signal sur 100 points selon 7 critères principaux
    """
    
    def __init__(self):
        self.scorer_name = "UltraSignalScorer"
        
        # Historique des performances par stratégie
        self.strategy_performance = defaultdict(lambda: {
            'total_signals': 0,
            'successful_signals': 0,
            'avg_profit': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        })
        
        # Cache des scores récents
        self.recent_scores = deque(maxlen=1000)
        
        # Poids des critères de scoring
        self.scoring_weights = {
            'technical_indicators': 0.25,     # 25 points max
            'confluence_strength': 0.20,     # 20 points max
            'sentiment_analysis': 0.15,      # 15 points max
            'risk_reward_ratio': 0.15,       # 15 points max
            'timing_quality': 0.10,          # 10 points max
            'volume_confirmation': 0.10,     # 10 points max
            'market_context': 0.05           # 5 points max
        }
        
        logger.info("🎯 UltraSignalScorer initialisé avec scoring sur 100 points")
        
    def score_signal(self, signal: StrategySignal, market_data: Dict = None) -> SignalScore:
        """
        Score un signal sur 100 points selon tous les critères
        
        Args:
            signal: Signal à scorer
            market_data: Données de marché actuelles
            
        Returns:
            SignalScore avec détails complets
        """
        try:
            scores = {}
            
            # 1. Score des indicateurs techniques (25 points)
            scores['technical'] = self._score_technical_indicators(signal)
            
            # 2. Score de confluence (20 points)
            scores['confluence'] = self._score_confluence_strength(signal)
            
            # 3. Score d'analyse de sentiment (15 points)
            scores['sentiment'] = self._score_sentiment_analysis(signal, market_data)
            
            # 4. Score risk/reward (15 points)
            scores['risk_reward'] = self._score_risk_reward_ratio(signal)
            
            # 5. Score de timing (10 points)
            scores['timing'] = self._score_timing_quality(signal, market_data)
            
            # 6. Score de confirmation volume (10 points)
            scores['volume'] = self._score_volume_confirmation(signal)
            
            # 7. Score de contexte marché (5 points)
            scores['market_context'] = self._score_market_context(signal, market_data)
            
            # Calculer le score total pondéré
            total_score = sum(
                scores[criterion] * self.scoring_weights[f"{criterion}_{'indicators' if criterion == 'technical' else 'strength' if criterion == 'confluence' else 'analysis' if criterion == 'sentiment' else 'ratio' if criterion == 'risk_reward' else 'quality' if criterion == 'timing' else 'confirmation' if criterion == 'volume' else 'context'}"]
                for criterion in scores
            )
            
            # Déterminer la qualité
            quality = self._determine_quality(total_score)
            
            # Calculer les scores individuels normalisés
            confidence_score = min(signal.confidence * 100, 100)
            technical_score = scores['technical']
            confluence_score = scores['confluence'] 
            sentiment_score = scores['sentiment']
            risk_reward_score = scores['risk_reward']
            timing_score = scores['timing']
            volume_score = scores['volume']
            
            # Générer une recommandation
            recommendation = self._generate_recommendation(total_score, quality, scores)
            
            # Créer le score final
            signal_score = SignalScore(
                total_score=round(total_score, 2),
                quality=quality,
                confidence_score=round(confidence_score, 2),
                technical_score=round(technical_score, 2),
                confluence_score=round(confluence_score, 2),
                sentiment_score=round(sentiment_score, 2),
                risk_reward_score=round(risk_reward_score, 2),
                timing_score=round(timing_score, 2),
                volume_score=round(volume_score, 2),
                breakdown=scores,
                recommendation=recommendation
            )
            
            # Enregistrer dans l'historique
            self.recent_scores.append({
                'timestamp': time.time(),
                'symbol': signal.symbol,
                'strategy': signal.strategy,
                'score': total_score,
                'quality': quality.value
            })
            
            logger.info(f"🎯 Signal {signal.symbol} scoré: {total_score:.1f}/100 ({quality.value}) - {recommendation}")
            
            return signal_score
            
        except Exception as e:
            logger.error(f"❌ Erreur scoring signal: {e}")
            return self._create_default_score()
            
    def _score_technical_indicators(self, signal: StrategySignal) -> float:
        """Score les indicateurs techniques (0-25 points)"""
        try:
            metadata = signal.metadata or {}
            score = 0
            
            # Analyser les confirmations techniques
            confirmations = metadata.get('total_confirmations', [])
            confirmation_count = len(confirmations)
            
            # Base score sur le nombre de confirmations
            if confirmation_count >= 15:
                score += 15  # Excellente confluence
            elif confirmation_count >= 10:
                score += 12  # Bonne confluence
            elif confirmation_count >= 6:
                score += 8   # Confluence modérée
            elif confirmation_count >= 3:
                score += 5   # Confluence minimale
            else:
                score += 1   # Très peu de confirmations
                
            # Bonus pour indicateurs clés
            key_indicators = ['RSI', 'MACD', 'EMA', 'BB', 'ADX', 'VOLUME']
            present_indicators = sum(1 for ind in key_indicators 
                                   if any(ind in conf for conf in confirmations))
            
            score += min(present_indicators * 1.5, 10)  # Max 10 points bonus
            
            return min(score, 25)
            
        except Exception as e:
            logger.error(f"❌ Erreur score technique: {e}")
            return 5  # Score par défaut
            
    def _score_confluence_strength(self, signal: StrategySignal) -> float:
        """Score la force de confluence multi-timeframe (0-20 points)"""
        try:
            metadata = signal.metadata or {}
            score = 0
            
            # Vérifier si c'est une analyse multi-timeframe
            if metadata.get('ultra_confluence'):
                timeframes = metadata.get('timeframes_analyzed', [])
                score += min(len(timeframes) * 3, 12)  # 3 points par timeframe, max 12
                
                # Bonus pour l'alignement des tendances
                trend_alignment = metadata.get('mtf_trend_alignment', 0)
                score += trend_alignment * 5  # Max 5 points
                
                # Bonus pour la force du momentum
                momentum_strength = metadata.get('mtf_momentum_strength', 0)
                score += momentum_strength * 3  # Max 3 points
                
            else:
                # Signal single timeframe, score réduit
                score += 5
                
            return min(score, 20)
            
        except Exception as e:
            logger.error(f"❌ Erreur score confluence: {e}")
            return 5
            
    def _score_sentiment_analysis(self, signal: StrategySignal, market_data: Dict = None) -> float:
        """Score l'analyse de sentiment (0-15 points)"""
        try:
            score = 0
            metadata = signal.metadata or {}
            
            if market_data and 'orderbook_sentiment' in market_data:
                sentiment = market_data['orderbook_sentiment']
                
                # Score basé sur le sentiment de l'orderbook
                sentiment_signal = sentiment.get('sentiment_signal', 'NEUTRAL')
                sentiment_score_val = sentiment.get('sentiment_score', 0)
                
                if signal.side == OrderSide.BUY:
                    if sentiment_signal in ['VERY_BULLISH', 'BULLISH']:
                        score += min(abs(sentiment_score_val) * 10, 8)
                    elif sentiment_signal == 'NEUTRAL':
                        score += 3
                    else:  # Bearish
                        score += 1  # Sentiment contre le signal
                else:  # SELL
                    if sentiment_signal in ['VERY_BEARISH', 'BEARISH']:
                        score += min(abs(sentiment_score_val) * 10, 8)
                    elif sentiment_signal == 'NEUTRAL':
                        score += 3
                    else:  # Bullish
                        score += 1
                        
                # Bonus pour imbalance orderbook
                imbalance = sentiment.get('imbalance', 0)
                if (signal.side == OrderSide.BUY and imbalance > 0.2) or \
                   (signal.side == OrderSide.SELL and imbalance < -0.2):
                    score += 3
                    
                # Bonus pour walls favorables
                bid_walls = len(sentiment.get('bid_walls', []))
                ask_walls = len(sentiment.get('ask_walls', []))
                
                if signal.side == OrderSide.BUY and bid_walls > ask_walls:
                    score += 2
                elif signal.side == OrderSide.SELL and ask_walls > bid_walls:
                    score += 2
                    
                # Bonus pour faible spread
                spread_pct = sentiment.get('spread_pct', 1)
                if spread_pct < 0.05:  # Spread < 0.05%
                    score += 2
                    
            else:
                # Pas de données sentiment, score par défaut
                score += 5
                
            return min(score, 15)
            
        except Exception as e:
            logger.error(f"❌ Erreur score sentiment: {e}")
            return 5
            
    def _score_risk_reward_ratio(self, signal: StrategySignal) -> float:
        """Score le ratio risk/reward (0-15 points)"""
        try:
            metadata = signal.metadata or {}
            score = 0
            
            # Chercher les niveaux de prix
            price_levels = metadata.get('price_levels', {})
            
            if price_levels:
                entry = price_levels.get('entry', signal.price)
                stop_loss = price_levels.get('stop_loss')
                take_profit_1 = price_levels.get('take_profit_1')
                
                if stop_loss and take_profit_1:
                    # Calculer R/R
                    risk = abs(entry - stop_loss)
                    reward = abs(take_profit_1 - entry)
                    
                    if risk > 0:
                        rr_ratio = reward / risk
                        
                        if rr_ratio >= 3.0:
                            score += 15  # Excellent R/R
                        elif rr_ratio >= 2.0:
                            score += 12  # Très bon R/R
                        elif rr_ratio >= 1.5:
                            score += 9   # Bon R/R
                        elif rr_ratio >= 1.0:
                            score += 6   # R/R acceptable
                        else:
                            score += 2   # R/R faible
                            
            # Si pas de niveaux, score par défaut basé sur la force
            if score == 0:
                if signal.strength == SignalStrength.VERY_STRONG:
                    score += 8
                elif signal.strength == SignalStrength.STRONG:
                    score += 6
                else:
                    score += 4
                    
            return min(score, 15)
            
        except Exception as e:
            logger.error(f"❌ Erreur score R/R: {e}")
            return 5
            
    def _score_timing_quality(self, signal: StrategySignal, market_data: Dict = None) -> float:
        """Score la qualité du timing (0-10 points)"""
        try:
            score = 0
            metadata = signal.metadata or {}
            
            # Score basé sur l'heure de trading
            current_hour = time.gmtime().tm_hour
            
            # Heures de forte liquidité (sessions London/NY)
            if 8 <= current_hour <= 16:  # London + NY overlap
                score += 4
            elif 13 <= current_hour <= 21:  # NY session
                score += 3
            elif 7 <= current_hour <= 15:  # London session
                score += 3
            else:
                score += 1  # Sessions asiatiques ou faible liquidité
                
            # Bonus pour momentum récent
            if market_data:
                momentum = market_data.get('momentum_10', 0)
                if abs(momentum) > 2:  # Momentum fort
                    score += 2
                elif abs(momentum) > 1:  # Momentum modéré
                    score += 1
                    
            # Bonus pour volatilité appropriée
            if market_data:
                atr = market_data.get('atr_14', 0)
                if atr and 0.5 < atr < 2.0:  # Volatilité optimale
                    score += 2
                elif atr:
                    score += 1
                    
            # Malus pour signaux trop fréquents
            recent_signals = [s for s in self.recent_scores 
                            if s['symbol'] == signal.symbol and 
                            time.time() - s['timestamp'] < 3600]  # Dernière heure
            
            if len(recent_signals) > 5:  # Plus de 5 signaux par heure
                score -= 2
                
            return max(0, min(score, 10))
            
        except Exception as e:
            logger.error(f"❌ Erreur score timing: {e}")
            return 5
            
    def _score_volume_confirmation(self, signal: StrategySignal) -> float:
        """Score la confirmation par le volume (0-10 points)"""
        try:
            metadata = signal.metadata or {}
            score = 0
            
            # Vérifier la confirmation volume multi-timeframe
            volume_confirmation = metadata.get('volume_confirmation', False)
            if volume_confirmation:
                score += 4
                
            # Score basé sur le spike de volume
            if metadata.get('volume_spike', False):
                score += 3
                
            # Score basé sur la tendance du volume
            volume_trend = metadata.get('volume_trend', 'stable')
            if volume_trend == 'increasing':
                score += 2
            elif volume_trend == 'stable':
                score += 1
                
            # Bonus pour ratio de volume élevé
            volume_ratio = metadata.get('volume_ratio', 1)
            if volume_ratio > 2.0:
                score += 1
                
            return min(score, 10)
            
        except Exception as e:
            logger.error(f"❌ Erreur score volume: {e}")
            return 3
            
    def _score_market_context(self, signal: StrategySignal, market_data: Dict = None) -> float:
        """Score le contexte de marché (0-5 points)"""
        try:
            score = 0
            metadata = signal.metadata or {}
            
            # Score basé sur ADX (force de tendance)
            adx = metadata.get('adx_14') or (market_data.get('adx_14') if market_data else None)
            if adx:
                if adx > 30:
                    score += 2  # Tendance forte
                elif adx > 20:
                    score += 1  # Tendance modérée
                    
            # Score basé sur la volatilité 24h
            if market_data:
                price_change_24h = market_data.get('price_change_pct_24h', 0)
                if abs(price_change_24h) > 5:  # Mouvement significatif
                    score += 1
                    
            # Bonus pour spread faible
            if market_data:
                spread_pct = market_data.get('spread_pct', 1)
                if spread_pct < 0.05:
                    score += 1
                    
            # Score basé sur la force du signal
            if signal.strength == SignalStrength.VERY_STRONG:
                score += 1
                
            return min(score, 5)
            
        except Exception as e:
            logger.error(f"❌ Erreur score contexte: {e}")
            return 2
            
    def _determine_quality(self, score: float) -> SignalQuality:
        """Détermine la qualité basée sur le score"""
        if score >= 95:
            return SignalQuality.INSTITUTIONAL
        elif score >= 85:
            return SignalQuality.EXCELLENT
        elif score >= 75:
            return SignalQuality.VERY_GOOD
        elif score >= 65:
            return SignalQuality.GOOD
        elif score >= 50:
            return SignalQuality.FAIR
        else:
            return SignalQuality.POOR
            
    def _generate_recommendation(self, total_score: float, quality: SignalQuality, scores: Dict) -> str:
        """Génère une recommandation basée sur le score"""
        if quality in [SignalQuality.INSTITUTIONAL, SignalQuality.EXCELLENT]:
            return "SIGNAL PREMIUM - Exécution recommandée avec position normale"
        elif quality == SignalQuality.VERY_GOOD:
            return "SIGNAL DE QUALITÉ - Exécution recommandée avec position réduite"
        elif quality == SignalQuality.GOOD:
            return "SIGNAL CORRECT - Exécution avec prudence et position minimale"
        elif quality == SignalQuality.FAIR:
            return "SIGNAL MOYEN - Considérer seulement si confluence parfaite"
        else:
            return "SIGNAL FAIBLE - Éviter l'exécution"
            
    def _create_default_score(self) -> SignalScore:
        """Crée un score par défaut en cas d'erreur"""
        return SignalScore(
            total_score=50.0,
            quality=SignalQuality.FAIR,
            confidence_score=50.0,
            technical_score=12.5,
            confluence_score=10.0,
            sentiment_score=7.5,
            risk_reward_score=7.5,
            timing_score=5.0,
            volume_score=5.0,
            breakdown={'default': 50.0},
            recommendation="SIGNAL NON ÉVALUÉ - Analyse manuelle requise"
        )
        
    def get_scoring_statistics(self) -> Dict:
        """Retourne les statistiques de scoring"""
        if not self.recent_scores:
            return {}
            
        recent_data = list(self.recent_scores)
        
        return {
            'total_signals_scored': len(recent_data),
            'average_score': sum(s['score'] for s in recent_data) / len(recent_data),
            'quality_distribution': {
                quality.value: sum(1 for s in recent_data if s['quality'] == quality.value)
                for quality in SignalQuality
            },
            'score_range': {
                'min': min(s['score'] for s in recent_data),
                'max': max(s['score'] for s in recent_data)
            }
        }