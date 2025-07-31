"""
Volume Context Analyzer

This module provides intelligent volume analysis with contextual adaptation:
- Market context detection (oversold, breakout, consolidation, etc.)
- Adaptive volume thresholds based on technical indicators
- Progressive volume buildup detection
- Volume spike analysis with market context
- Intelligent volume tolerance for extreme market conditions

Enhanced with cached indicators for optimal performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class VolumeContextType(Enum):
    """Types de contextes volume détectés."""
    DEEP_OVERSOLD = "deep_oversold"
    MODERATE_OVERSOLD = "moderate_oversold" 
    OVERSOLD_BOUNCE = "oversold_bounce"
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    CONSOLIDATION_BREAK = "consolidation_break"
    BREAKOUT = "breakout"
    TREND_CONTINUATION = "trend_continuation"
    PUMP_START = "pump_start"
    REVERSAL_PATTERN = "reversal_pattern"
    NEUTRAL = "neutral"


class VolumePatternType(Enum):
    """Types de patterns de volume."""
    BUILDUP = "buildup"  # Accumulation progressive
    SPIKE = "spike"  # Pic soudain
    SUSTAINED_HIGH = "sustained_high"  # Volume élevé soutenu
    DECLINING = "declining"  # Volume décroissant
    NORMAL = "normal"  # Volume normal


@dataclass
class VolumeContext:
    """Contexte volume avec seuils adaptés."""
    context_type: VolumeContextType
    min_ratio: float  # Seuil minimum adapté
    ideal_ratio: float  # Seuil idéal adapté
    confidence: float  # Confiance dans la détection (0-1)
    pattern_detected: VolumePatternType
    tolerance_factor: float  # Facteur de tolérance appliqué
    description: str
    market_conditions: Dict[str, float]  # RSI, CCI, ADX, etc.
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour export."""
        return {
            'context_type': self.context_type.value,
            'min_ratio': self.min_ratio,
            'ideal_ratio': self.ideal_ratio,
            'confidence': self.confidence,
            'pattern_detected': self.pattern_detected.value,
            'tolerance_factor': self.tolerance_factor,
            'description': self.description,
            'market_conditions': self.market_conditions
        }


@dataclass 
class VolumeAnalysis:
    """Analyse complète du volume avec contexte."""
    current_volume_ratio: float
    context: VolumeContext
    pattern_strength: float  # Force du pattern détecté (0-100)
    quality_score: float  # Score de qualité global (0-100)
    volume_trend: str  # 'increasing', 'decreasing', 'stable'
    buildup_detected: bool
    spike_detected: bool
    recommendations: List[str]  # Recommandations d'action
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour export."""
        return {
            'current_volume_ratio': self.current_volume_ratio,
            'context': self.context.to_dict(),
            'pattern_strength': self.pattern_strength,
            'quality_score': self.quality_score,
            'volume_trend': self.volume_trend,
            'buildup_detected': self.buildup_detected,
            'spike_detected': self.spike_detected,
            'recommendations': self.recommendations
        }


class VolumeContextAnalyzer:
    """
    Analyseur de contexte volume intelligent avec adaptation contextuelle.
    
    Utilise les indicateurs techniques (RSI, CCI, ADX) pour adapter
    automatiquement les seuils de validation volume selon le contexte market.
    """
    
    def __init__(self, 
                 default_volume_threshold: float = 1.5,
                 buildup_lookback: int = 5,
                 spike_multiplier: float = 2.5):
        """
        Args:
            default_volume_threshold: Seuil volume par défaut
            buildup_lookback: Périodes pour détecter volume buildup
            spike_multiplier: Multiplicateur pour détecter volume spikes
        """
        self.default_threshold = default_volume_threshold
        self.buildup_lookback = buildup_lookback
        self.spike_multiplier = spike_multiplier
        
        # Configurations des contextes avec seuils adaptés
        self.context_configs = {
            VolumeContextType.DEEP_OVERSOLD: {
                'min_ratio': 0.8, 'ideal_ratio': 1.8, 'description': 'Oversold extrême - tolérance élevée'
            },
            VolumeContextType.MODERATE_OVERSOLD: {
                'min_ratio': 1.0, 'ideal_ratio': 2.0, 'description': 'Oversold modéré - tolérance moyenne'
            },
            VolumeContextType.OVERSOLD_BOUNCE: {
                'min_ratio': 1.2, 'ideal_ratio': 2.2, 'description': 'Bounce oversold - volume important'
            },
            VolumeContextType.LOW_VOLATILITY: {
                'min_ratio': 0.9, 'ideal_ratio': 1.6, 'description': 'Faible volatilité - seuils réduits'
            },
            VolumeContextType.HIGH_VOLATILITY: {
                'min_ratio': 1.8, 'ideal_ratio': 3.0, 'description': 'Forte volatilité - seuils élevés'
            },
            VolumeContextType.CONSOLIDATION_BREAK: {
                'min_ratio': 1.4, 'ideal_ratio': 2.5, 'description': 'Sortie consolidation - confirmation volume'
            },
            VolumeContextType.BREAKOUT: {
                'min_ratio': 2.0, 'ideal_ratio': 4.0, 'description': 'Breakout - volume critique'
            },
            VolumeContextType.TREND_CONTINUATION: {
                'min_ratio': 1.3, 'ideal_ratio': 2.3, 'description': 'Continuation tendance - volume normal'
            },
            VolumeContextType.PUMP_START: {
                'min_ratio': 2.5, 'ideal_ratio': 5.0, 'description': 'Début pump - volume massif requis'
            },
            VolumeContextType.REVERSAL_PATTERN: {
                'min_ratio': 1.6, 'ideal_ratio': 2.8, 'description': 'Pattern reversal - volume confirmateur'
            },
            VolumeContextType.NEUTRAL: {
                'min_ratio': 1.5, 'ideal_ratio': 2.5, 'description': 'Contexte neutre - seuils standards'
            }
        }
    
    def analyze_volume_context(self,
                              volumes: Union[List[float], np.ndarray],
                              closes: Union[List[float], np.ndarray],
                              highs: Optional[Union[List[float], np.ndarray]] = None,
                              lows: Optional[Union[List[float], np.ndarray]] = None,
                              symbol: Optional[str] = None,
                              signal_type: str = "BUY",
                              enable_cache: bool = True) -> VolumeAnalysis:
        """
        Analyse complète du contexte volume avec adaptation intelligente.
        
        Args:
            volumes: Série des volumes
            closes: Série des prix de clôture
            highs: Série des prix hauts (optionnel)
            lows: Série des prix bas (optionnel) 
            symbol: Symbole pour cache (optionnel)
            signal_type: Type de signal ("BUY", "SELL")
            enable_cache: Activer le cache d'indicateurs
            
        Returns:
            VolumeAnalysis complète avec contexte et recommandations
        """
        try:
            volumes = np.array(volumes, dtype=float)
            closes = np.array(closes, dtype=float)
            
            if len(volumes) < 20 or len(closes) < 20:
                return self._create_default_analysis(volumes, closes)
            
            # 1. Calcul volume ratio actuel
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[-20:])
            current_volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 2. Calcul des indicateurs techniques (avec cache si possible)
            market_conditions = self._calculate_market_indicators(
                np.asarray(closes), 
                np.asarray(highs) if highs is not None else None, 
                np.asarray(lows) if lows is not None else None, 
                symbol, enable_cache
            )
            
            # 3. Détection du contexte market
            context = self._detect_market_context(
                market_conditions, signal_type, volumes, closes
            )
            
            # 4. Détection des patterns de volume
            buildup_detected = self._detect_volume_buildup(volumes)
            spike_detected = self._detect_volume_spike(volumes)
            volume_trend = self._analyze_volume_trend(volumes)
            
            # 5. Calcul pattern strength et qualité
            pattern_strength = self._calculate_pattern_strength(
                volumes, context, buildup_detected, spike_detected
            )
            
            quality_score = self._calculate_volume_quality_score(
                current_volume_ratio, context, pattern_strength
            )
            
            # 6. Génération des recommandations
            recommendations = self._generate_recommendations(
                context, current_volume_ratio, buildup_detected, spike_detected
            )
            
            return VolumeAnalysis(
                current_volume_ratio=current_volume_ratio,
                context=context,
                pattern_strength=pattern_strength,
                quality_score=quality_score,
                volume_trend=volume_trend,
                buildup_detected=buildup_detected,
                spike_detected=spike_detected,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Erreur analyse contexte volume: {e}")
            return self._create_default_analysis(np.asarray(volumes), np.asarray(closes))
    
    def _calculate_market_indicators(self,
                                   closes: np.ndarray,
                                   highs: Optional[np.ndarray] = None,
                                   lows: Optional[np.ndarray] = None,
                                   symbol: Optional[str] = None,
                                   enable_cache: bool = True) -> Dict[str, float]:
        """Calcule les indicateurs techniques nécessaires."""
        indicators = {}
        
        try:
            # RSI (avec cache si symbol fourni)
            from ..indicators.momentum.rsi import calculate_rsi  # type: ignore
            rsi = calculate_rsi(closes, 14, symbol, enable_cache)
            if rsi is not None:
                indicators['rsi'] = rsi
            
            # CCI (avec cache si symbol fourni)  
            from ..indicators.momentum.cci import calculate_cci  # type: ignore
            if highs is not None and lows is not None:
                cci = calculate_cci(highs, lows, closes, 20)
                if cci is not None:
                    indicators['cci'] = cci
            
            # ADX (sans cache - non supporté)
            from ..indicators.trend.adx import calculate_adx  # type: ignore
            if highs is not None and lows is not None:
                adx = calculate_adx(highs, lows, closes, 14)
                if adx is not None:
                    indicators['adx'] = adx
                    
        except Exception as e:
            logger.warning(f"Erreur calcul indicateurs: {e}")
        
        return indicators
    
    def _detect_market_context(self,
                             market_conditions: Dict[str, float],
                             signal_type: str,
                             volumes: np.ndarray,
                             closes: np.ndarray) -> VolumeContext:
        """Détecte le contexte market approprié."""
        detected_contexts = []
        
        rsi = market_conditions.get('rsi')
        cci = market_conditions.get('cci') 
        adx = market_conditions.get('adx')
        
        # 1. Conditions oversold (pour signaux BUY)
        if signal_type == "BUY" and rsi is not None:
            if rsi < 30 and cci is not None and cci < -200:
                detected_contexts.append((VolumeContextType.DEEP_OVERSOLD, 0.9))
            elif rsi < 40 and cci is not None and cci < -150:
                detected_contexts.append((VolumeContextType.MODERATE_OVERSOLD, 0.8))
            elif rsi < 40:
                detected_contexts.append((VolumeContextType.OVERSOLD_BOUNCE, 0.7))
        
        # 2. Conditions de volatilité
        if adx is not None:
            if adx < 20:
                detected_contexts.append((VolumeContextType.LOW_VOLATILITY, 0.6))
            elif adx > 35:
                detected_contexts.append((VolumeContextType.HIGH_VOLATILITY, 0.7))
        
        # 3. Patterns de volume
        if self._detect_volume_buildup(volumes):
            detected_contexts.append((VolumeContextType.CONSOLIDATION_BREAK, 0.7))
        elif self._detect_volume_spike(volumes):
            detected_contexts.append((VolumeContextType.BREAKOUT, 0.8))
        
        # 4. Détection pump/reversal patterns
        price_change = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
        volume_increase = volumes[-1] / np.mean(volumes[-5:]) if len(volumes) >= 5 else 1.0
        
        if price_change > 10 and volume_increase > 3:
            detected_contexts.append((VolumeContextType.PUMP_START, 0.85))
        elif abs(price_change) > 5 and volume_increase > 2:
            detected_contexts.append((VolumeContextType.REVERSAL_PATTERN, 0.7))
        
        # 5. Tendance continuation
        if adx is not None and adx > 25 and rsi is not None and 40 <= rsi <= 70:
            detected_contexts.append((VolumeContextType.TREND_CONTINUATION, 0.6))
        
        # Sélection du meilleur contexte
        if detected_contexts:
            detected_contexts.sort(key=lambda x: x[1], reverse=True)
            best_context, confidence = detected_contexts[0]
        else:
            best_context, confidence = VolumeContextType.NEUTRAL, 0.5
        
        # Création du contexte avec configuration
        config = self.context_configs[best_context]
        
        # Calcul du facteur de tolérance
        tolerance_factor = self._calculate_tolerance_factor(market_conditions)
        
        # Pattern détecté
        pattern_detected = self._identify_volume_pattern(volumes)
        
        # Extraction sécurisée des valeurs de configuration
        min_ratio = config.get('min_ratio', 1.5)
        ideal_ratio = config.get('ideal_ratio', 2.5)
        description = config.get('description', 'Contexte neutre')
        
        return VolumeContext(
            context_type=best_context,
            min_ratio=float(min_ratio if isinstance(min_ratio, (int, float, str)) else 0.0) * tolerance_factor,
            ideal_ratio=float(ideal_ratio if isinstance(ideal_ratio, (int, float, str)) else 0.0) * tolerance_factor,
            confidence=confidence,
            pattern_detected=pattern_detected,
            tolerance_factor=tolerance_factor,
            description=str(description),
            market_conditions=market_conditions
        )
    
    def _detect_volume_buildup(self, volumes: np.ndarray) -> bool:
        """Détecte une accumulation progressive de volume."""
        if len(volumes) < self.buildup_lookback + 1:
            return False
        
        recent_volumes = volumes[-self.buildup_lookback:]
        increases = 0
        min_increase = 1.1  # 10% d'augmentation minimum
        
        for i in range(1, len(recent_volumes)):
            if recent_volumes[i] > recent_volumes[i-1] * min_increase:
                increases += 1
        
        # Au moins 60% des périodes doivent montrer une augmentation
        return increases / (len(recent_volumes) - 1) >= 0.6
    
    def _detect_volume_spike(self, volumes: np.ndarray) -> bool:
        """Détecte un pic de volume soudain."""
        if len(volumes) < 5:
            return False
        
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-5:-1])  # Moyenne des 4 périodes précédentes
        
        return current_volume > avg_volume * self.spike_multiplier
    
    def _analyze_volume_trend(self, volumes: np.ndarray) -> str:
        """Analyse la tendance du volume."""
        if len(volumes) < 10:
            return 'stable'
        
        recent_volumes = volumes[-10:]
        slope = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
        avg_volume = np.mean(recent_volumes)
        
        relative_slope = slope / avg_volume if avg_volume > 0 else 0
        
        if relative_slope > 0.05:
            return 'increasing'
        elif relative_slope < -0.05:
            return 'decreasing'
        else:
            return 'stable'
    
    def _identify_volume_pattern(self, volumes: np.ndarray) -> VolumePatternType:
        """Identifie le pattern de volume dominant."""
        if self._detect_volume_spike(volumes):
            return VolumePatternType.SPIKE
        elif self._detect_volume_buildup(volumes):
            return VolumePatternType.BUILDUP
        else:
            # Analyse volume soutenu vs déclinant
            if len(volumes) >= 10:
                recent_avg = np.mean(volumes[-5:])
                older_avg = np.mean(volumes[-10:-5])
                
                if recent_avg > older_avg * 1.5:
                    return VolumePatternType.SUSTAINED_HIGH
                elif recent_avg < older_avg * 0.7:
                    return VolumePatternType.DECLINING
            
            return VolumePatternType.NORMAL
    
    def _calculate_tolerance_factor(self, market_conditions: Dict[str, float]) -> float:
        """Calcule le facteur de tolérance basé sur les conditions market."""
        tolerance_factors = []
        
        rsi = market_conditions.get('rsi')
        cci = market_conditions.get('cci')
        adx = market_conditions.get('adx')
        
        # RSI oversold extrême
        if rsi is not None and rsi < 30:
            tolerance_factors.append(0.7)  # Réduction de 30%
        
        # CCI oversold sévère  
        if cci is not None and cci < -200:
            tolerance_factors.append(0.6)  # Réduction de 40%
        
        # ADX faible (marché calme)
        if adx is not None and adx < 20:
            tolerance_factors.append(0.8)  # Réduction de 20%
        
        # Retourner la tolérance la plus forte (facteur le plus bas)
        return min(tolerance_factors) if tolerance_factors else 1.0
    
    def _calculate_pattern_strength(self,
                                  volumes: np.ndarray,
                                  context: VolumeContext,
                                  buildup_detected: bool,
                                  spike_detected: bool) -> float:
        """Calcule la force du pattern volume détecté."""
        strength = 50.0  # Base
        
        if len(volumes) < 5:
            return strength
        
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else np.mean(volumes)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Bonus basé sur le ratio volume
        if volume_ratio > 3.0:
            strength += 30
        elif volume_ratio > 2.0:
            strength += 20
        elif volume_ratio > 1.5:
            strength += 10
        
        # Bonus pour patterns détectés
        if spike_detected:
            strength += 20
        if buildup_detected:
            strength += 15
        
        # Bonus pour cohérence avec le contexte
        if context.confidence > 0.7:
            strength += 10
        
        return min(strength, 100.0)
    
    def _calculate_volume_quality_score(self,
                                      current_volume_ratio: float,
                                      context: VolumeContext,
                                      pattern_strength: float) -> float:
        """Calcule le score de qualité global du volume."""
        # Score basé sur la conformité aux seuils contextuels
        if current_volume_ratio >= context.ideal_ratio:
            threshold_score = 100.0
        elif current_volume_ratio >= context.min_ratio:
            # Score graduel entre min et ideal
            ratio_range = context.ideal_ratio - context.min_ratio
            actual_range = current_volume_ratio - context.min_ratio
            threshold_score = 50.0 + (actual_range / ratio_range) * 50.0
        else:
            # En dessous du minimum
            threshold_score = (current_volume_ratio / context.min_ratio) * 50.0
        
        # Score pondéré : seuils (60%) + pattern (30%) + confiance contexte (10%)
        quality_score = (
            threshold_score * 0.6 +
            pattern_strength * 0.3 +
            context.confidence * 100 * 0.1
        )
        
        return min(quality_score, 100.0)
    
    def _generate_recommendations(self,
                                context: VolumeContext,
                                current_volume_ratio: float,
                                buildup_detected: bool,
                                spike_detected: bool) -> List[str]:
        """Génère des recommandations basées sur l'analyse."""
        recommendations = []
        
        if current_volume_ratio >= context.ideal_ratio:
            recommendations.append("✅ Volume excellent - Signal très fiable")
        elif current_volume_ratio >= context.min_ratio:
            recommendations.append("⚠️ Volume acceptable - Signal modérément fiable") 
        else:
            recommendations.append("❌ Volume insuffisant - Prudence requise")
        
        if buildup_detected:
            recommendations.append("📈 Volume buildup détecté - Momentum building")
        
        if spike_detected:
            recommendations.append("🚀 Volume spike détecté - Action immédiate possible")
        
        if context.context_type in [VolumeContextType.DEEP_OVERSOLD, VolumeContextType.MODERATE_OVERSOLD]:
            recommendations.append("💡 Conditions oversold - Tolérance volume appliquée")
        
        if context.context_type == VolumeContextType.BREAKOUT:
            recommendations.append("⚡ Contexte breakout - Volume critique pour confirmation")
        
        if context.context_type == VolumeContextType.PUMP_START:
            recommendations.append("🔥 Début pump détecté - Volume massif requis")
        
        return recommendations
    
    def _create_default_analysis(self, volumes: np.ndarray, closes: np.ndarray) -> VolumeAnalysis:
        """Crée une analyse par défaut en cas de données insuffisantes."""
        default_context = VolumeContext(
            context_type=VolumeContextType.NEUTRAL,
            min_ratio=1.5,
            ideal_ratio=2.5,
            confidence=0.3,
            pattern_detected=VolumePatternType.NORMAL,
            tolerance_factor=1.0,
            description="Données insuffisantes - contexte neutre",
            market_conditions={}
        )
        
        current_volume_ratio = 1.0
        if len(volumes) >= 2:
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[:-1])
            current_volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        return VolumeAnalysis(
            current_volume_ratio=current_volume_ratio,
            context=default_context,
            pattern_strength=30.0,
            quality_score=30.0,
            volume_trend='stable',
            buildup_detected=False,
            spike_detected=False,
            recommendations=["⚠️ Données insuffisantes pour analyse contextuelle"]
        )
    
    # ============ METHODS PUBLIQUES UTILITAIRES ============
    
    def get_adaptive_threshold(self,
                             volumes: Union[List[float], np.ndarray],
                             closes: Union[List[float], np.ndarray],
                             signal_type: str = "BUY",
                             symbol: Optional[str] = None) -> Tuple[float, str]:
        """
        Retourne le seuil volume adaptatif et le contexte détecté.
        
        Returns:
            Tuple[float, str]: (seuil_adaptatif, contexte_détecté)
        """
        analysis = self.analyze_volume_context(volumes, closes, symbol=symbol, signal_type=signal_type)
        return analysis.context.min_ratio, analysis.context.context_type.value
    
    def is_volume_acceptable(self,
                           current_volume_ratio: float,
                           volumes: Union[List[float], np.ndarray],
                           closes: Union[List[float], np.ndarray],
                           symbol: Optional[str] = None) -> Tuple[bool, float]:
        """
        Vérifie si le volume est acceptable selon le contexte.
        
        Returns:
            Tuple[bool, float]: (acceptable, score_qualité)
        """
        analysis = self.analyze_volume_context(volumes, closes, symbol=symbol)
        acceptable = current_volume_ratio >= analysis.context.min_ratio
        return acceptable, analysis.quality_score
    
    def get_volume_quality_description(self, quality_score: float) -> str:
        """Retourne une description qualitative du score volume."""
        if quality_score >= 80:
            return "Excellent"
        elif quality_score >= 60:
            return "Très bon"
        elif quality_score >= 40:
            return "Acceptable"
        elif quality_score >= 20:
            return "Faible"
        else:
            return "Insuffisant"