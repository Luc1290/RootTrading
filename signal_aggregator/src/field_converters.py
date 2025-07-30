"""
Module de conversion des champs de la base de données.
Gère la conversion des valeurs string en valeurs numériques pour les validators.
"""

from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FieldConverter:
    """Convertisseur de champs pour harmoniser les types de données."""
    
    # Mappings pour convertir les strings en scores numériques
    STRENGTH_MAPPING = {
        # Force générale
        'ABSENT': 0.0,
        'VERY_WEAK': 0.2,
        'WEAK': 0.3,
        'MODERATE': 0.5,
        'STRONG': 0.7,
        'VERY_STRONG': 0.85,
        'EXTREME': 0.95,
        
        # Variantes lowercase
        'absent': 0.0,
        'very_weak': 0.2,
        'weak': 0.3,
        'moderate': 0.5,
        'strong': 0.7,
        'very_strong': 0.85,
        'extreme': 0.95,
        
        # Support/Resistance
        'MINOR': 0.3,
        'MAJOR': 0.7,
        'CRITICAL': 0.9
    }
    
    REGIME_MAPPING = {
        # Régimes de marché
        'TRENDING_BULL': 'trending',
        'TRENDING_BEAR': 'trending',
        'RANGING': 'ranging',
        'CONSOLIDATION': 'ranging',
        'TRANSITION': 'transition',
        'CHAOTIC': 'chaotic',
        'NORMAL': 'normal',
        
        # Régimes de volatilité
        'low': 'low',
        'normal': 'normal',
        'high': 'high',
        'extreme': 'extreme'
    }
    
    PATTERN_MAPPING = {
        'NORMAL': 'normal',
        'PRICE_SPIKE_UP': 'spike_up',
        'PRICE_SPIKE_DOWN': 'spike_down',
        'VOLUME_SPIKE': 'volume_spike',
        'BREAKOUT': 'breakout',
        'BREAKDOWN': 'breakdown',
        'REVERSAL': 'reversal'
    }
    
    VOLUME_CONTEXT_MAPPING = {
        'LOW_VOLATILITY': 'low_vol',
        'NEUTRAL': 'neutral',
        'HIGH_VOLATILITY': 'high_vol',
        'EXTREME': 'extreme'
    }
    
    VOLUME_PATTERN_MAPPING = {
        'NORMAL': 'normal',
        'SUSTAINED_HIGH': 'sustained_high',
        'SUSTAINED_LOW': 'sustained_low',
        'INCREASING': 'increasing',
        'DECREASING': 'decreasing',
        'SPIKE': 'spike'
    }
    
    @classmethod
    def convert_indicators(cls, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convertit tous les indicateurs en types appropriés pour les validators.
        
        Args:
            indicators: Dict des indicateurs bruts de la DB
            
        Returns:
            Dict des indicateurs convertis
        """
        converted = {}
        
        # Log pour debug
        logger.debug(f"FieldConverter: Conversion de {len(indicators)} indicateurs")
        
        for key, value in indicators.items():
            try:
                # Conversion spécifique par type de champ
                if 'strength' in key or 'STRENGTH' in key:
                    converted[key] = cls._convert_strength(value)
                elif key in ['market_regime', 'volatility_regime', 'trend_regime', 'momentum_regime']:
                    converted[key] = cls._convert_regime(value)
                elif key == 'pattern_detected':
                    converted[key] = cls._convert_pattern(value)
                elif key == 'volume_context':
                    converted[key] = cls._convert_volume_context(value)
                elif key == 'volume_pattern':
                    converted[key] = cls._convert_volume_pattern(value)
                elif key == 'regime_conf' or key == 'pattern_conf':
                    # Ces champs sont déjà numériques mais peuvent nécessiter /100
                    converted[key] = cls._ensure_float(value) / 100.0 if cls._ensure_float(value) > 1 else cls._ensure_float(value)
                elif key == 'break_prob':
                    # Déjà entre 0 et 1
                    converted[key] = cls._ensure_float(value)
                elif key in ['confluence', 'vol_quality', 'momentum_score', 'volume_quality_score']:
                    # Score sur 100 à convertir en 0-1
                    converted[key] = cls._ensure_float(value) / 100.0 if cls._ensure_float(value) > 1 else cls._ensure_float(value)
                elif key == 'atr_pct':
                    # Percentile ATR
                    converted[key + 'entile'] = cls._ensure_float(value)
                    converted[key] = cls._ensure_float(value)
                elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                    # Conversion string numérique en float
                    converted[key] = float(value)
                else:
                    # Garder la valeur originale
                    converted[key] = value
                    
            except Exception as e:
                logger.warning(f"Erreur conversion champ {key}={value}: {e}")
                converted[key] = value
                
        # Ajout de champs calculés si nécessaire
        cls._add_calculated_fields(converted, indicators)
        
        return converted
    
    @classmethod
    def _convert_strength(cls, value: Any) -> float:
        """Convertit une force (string) en score numérique."""
        if value is None:
            return 0.5
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return cls.STRENGTH_MAPPING.get(value, 0.5)
        return 0.5
    
    @classmethod
    def _convert_regime(cls, value: Any) -> str:
        """Convertit un régime en format standardisé."""
        if value is None:
            return 'unknown'
        if isinstance(value, str):
            return cls.REGIME_MAPPING.get(value, value.lower())
        return str(value)
    
    @classmethod
    def _convert_pattern(cls, value: Any) -> str:
        """Convertit un pattern en format standardisé."""
        if value is None:
            return 'none'
        if isinstance(value, str):
            return cls.PATTERN_MAPPING.get(value, value.lower())
        return str(value)
    
    @classmethod
    def _convert_volume_context(cls, value: Any) -> str:
        """Convertit un contexte de volume."""
        if value is None:
            return 'neutral'
        if isinstance(value, str):
            return cls.VOLUME_CONTEXT_MAPPING.get(value, value.lower())
        return str(value)
    
    @classmethod
    def _convert_volume_pattern(cls, value: Any) -> str:
        """Convertit un pattern de volume."""
        if value is None:
            return 'normal'
        if isinstance(value, str):
            return cls.VOLUME_PATTERN_MAPPING.get(value, value.lower())
        return str(value)
    
    @classmethod
    def _ensure_float(cls, value: Any, default: float = 0.0) -> float:
        """Assure la conversion en float."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    @classmethod
    def _add_calculated_fields(cls, converted: Dict[str, Any], original: Dict[str, Any]):
        """Ajoute des champs calculés nécessaires aux validators."""
        
        # Calcul du score de force de régime numérique
        if 'regime_strength' in converted and isinstance(converted['regime_strength'], float):
            # Déjà converti, pas besoin de recalcul
            pass
        
        # Calcul du score de tendance numérique
        if 'trend_strength' in converted and isinstance(converted['trend_strength'], float):
            # Déjà converti
            pass
        
        # Score de qualité volume si manquant
        if 'volume_quality_score' not in converted and 'vol_quality' in converted:
            converted['volume_quality_score'] = converted['vol_quality']
        
        # Ajout du percentile ATR si manquant
        if 'atr_percentile' not in converted and 'atr_pctentile' in converted:
            converted['atr_percentile'] = converted['atr_pctentile']
        
        # Conversion volume_trend si c'est un nombre
        if 'volume_trend' in original:
            try:
                # Si c'est un nombre, le convertir en string descriptif
                vol_trend_num = cls._ensure_float(original['volume_trend'])
                if vol_trend_num > 0.1:
                    converted['volume_trend'] = 'increasing'
                    converted['volume_trend_numeric'] = vol_trend_num
                elif vol_trend_num < -0.1:
                    converted['volume_trend'] = 'decreasing'
                    converted['volume_trend_numeric'] = vol_trend_num
                else:
                    converted['volume_trend'] = 'stable'
                    converted['volume_trend_numeric'] = vol_trend_num
            except:
                # Si ce n'est pas un nombre, garder tel quel
                pass
        
        # Ajout de champs manquants avec valeurs par défaut
        defaults = {
            'directional_bias': 'neutral',
            'tf_alignment': 0.5,
            'consensus_score': 0.5,
            'regime_confidence': 0.5,
            'momentum_score': 0.5,
            'pattern_confidence': 0.0,
            'liquidity_score': 0.5,
            'accumulation_distribution_score': 0.5,
            'buy_sell_pressure': 0.5,
            'money_flow_index': 0.5,
            'obv_trend': 0.0,
            'trend_angle': 0.0,
            'ema_alignment_score': 0.5,
            'timeframe_consensus_score': 0.5,
            'aligned_timeframes_count': 0,
            'regime_stability': 0.5,
            'regime_persistence': 0.5,
            'regime_momentum': 0.0,
            'volume_buildup_bars': 0,
            'volume_buildup_slope': 0.0,
            'volume_buildup_consistency': 0.0
        }
        
        for key, default_value in defaults.items():
            if key not in converted:
                converted[key] = default_value