"""
Module de conversion des champs de la base de données.
Gère la conversion des valeurs string en valeurs numériques pour les validators.
"""

from typing import Any, Optional, Dict
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class FieldConverter:
    """Convertisseur de champs pour harmoniser les types de données."""
    
    # Mappings pour convertir les strings en scores numériques
    STRENGTH_MAPPING = {
        # Force générale (case insensitive sera géré dans la méthode)
        'ABSENT': 0.0,
        'VERY_WEAK': 0.2,
        'WEAK': 0.3,
        'MODERATE': 0.5,
        'STRONG': 0.7,
        'VERY_STRONG': 0.85,
        'EXTREME': 0.95,
        
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
    
    # Mapping des noms de champs de la DB vers les noms attendus par les validators
    FIELD_NAME_MAPPING = {
        # Pivot points (corrections basées sur l'analyse DB avec noms réels)
        'nearest_support': 'pivot_support',
        'nearest_resistance': 'pivot_resistance', 
        'support_strength': 'pivot_support_strength',
        'resistance_strength': 'pivot_resistance_strength',
        'pivot_count': 'pivot_strength',
        
        # Volume spikes (corrections basées sur l'analyse DB)
        # Note: volume_spike_multiplier existe déjà dans la DB, pas besoin de mapping
        'rel_volume': 'relative_volume',          # rel_volume -> relative_volume (DB field name)
        'vol_ratio': 'volume_ratio',              # vol_ratio -> volume_ratio
        
        # Z-Scores (création de proxies à partir des données existantes)
        # On utilisera les oscillateurs normalisés comme proxies de Z-Scores
        'rsi_14': 'zscore_rsi',           # RSI normalisé comme proxy Z-Score RSI
        'cci_20': 'zscore_cci',           # CCI normalisé comme proxy Z-Score CCI (corrected field name)
        'stoch_k': 'zscore_stoch',        # Stochastic normalisé
        'williams_r': 'zscore_williams',  # Williams %R normalisé
        
        # Ajout d'autres mappings utiles
        'bb_position': 'bollinger_position',
        'vol_quality': 'volume_quality_score',
        'confluence': 'confluence_score',
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
        converted: Dict[str, Any] = {}
        
        # Log pour debug
        logger.debug(f"FieldConverter: Conversion de {len(indicators)} indicateurs")
        
        # Étape 1: Appliquer le mapping de noms de champs
        mapped_indicators = {}
        for key, value in indicators.items():
            # Ajouter le champ original
            mapped_indicators[key] = value
            # Ajouter le champ mappé s'il existe
            if key in cls.FIELD_NAME_MAPPING:
                mapped_key = cls.FIELD_NAME_MAPPING[key]
                mapped_indicators[mapped_key] = value
                logger.debug(f"FieldConverter: Mapped {key} -> {mapped_key}")
        
        # Étape 2: Création de Z-Scores synthétiques
        cls._add_synthetic_zscores(mapped_indicators)
        
        # Étape 3: Conversion des types  
        for key, value in mapped_indicators.items():
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
                    # Score sur 100 - GARDER le format 0-100 pour les validators
                    converted[key] = cls._ensure_float(value)
                elif key == 'atr_percentile':
                    # Percentile ATR - garder tel quel
                    converted[key] = cls._ensure_float(value)
                elif isinstance(value, (int, float, Decimal)):
                    # Valeur déjà numérique (incluant Decimal de PostgreSQL)
                    converted[key] = float(value)
                elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                    # Conversion string numérique en float
                    converted[key] = float(value)
                else:
                    # Garder la valeur originale
                    converted[key] = value
                    
            except Exception as e:
                logger.warning(f"Erreur conversion champ {key}={value}: {e}")
                converted[key] = value
                
        # Étape 4: Ajout de champs calculés si nécessaire
        cls._add_calculated_fields(converted, indicators)
        
        # Étape 5: Ajout de champs Z-Score spécialisés pour validators spécifiques
        cls._add_specialized_zscore_fields(converted)
        
        # Étape 6: Ajout de champs volume spike spécialisés
        cls._add_specialized_volume_spike_fields(converted)
        
        return converted
    
    @classmethod
    def _convert_strength(cls, value: Any) -> float:
        """Convertit une force (string) en score numérique."""
        if value is None:
            return 0.5
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Convertir en majuscules pour la recherche case-insensitive
            return cls.STRENGTH_MAPPING.get(value.upper(), 0.5)
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
        
        # Le percentile ATR devrait déjà être présent, pas besoin de mapping
        
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
            'consensus_score': 50.0,    # Format 0-100 pour validators
            'regime_confidence': 50.0,  # Format 0-100 pour validators
            'momentum_score': 50.0,     # Format 0-100 pour validators
            'pattern_confidence': 0.0,
            'liquidity_score': 50.0,   # Format 0-100 pour validators
            'accumulation_distribution_score': 50.0,  # Format 0-100 pour validators
            'buy_sell_pressure': 0.5,
            'money_flow_index': 0.5,
            'obv_trend': 0.0,
            'trend_angle': 0.0,
            'ema_alignment_score': 50.0,  # Format 0-100 pour validators
            'timeframe_consensus_score': 50.0,  # Format 0-100 pour validators
            'aligned_timeframes_count': 0,
            'regime_stability': 50.0,   # Format 0-100 pour validators
            'regime_persistence': 50.0, # Format 0-100 pour validators
            'regime_momentum': 0.0,
            'volume_buildup_bars': 0,
            'volume_buildup_slope': 0.0,
            'volume_buildup_consistency': 0.0
        }
        
        for key, default_value in defaults.items():
            if key not in converted:
                converted[key] = default_value
    
    @classmethod
    def _add_synthetic_zscores(cls, indicators: Dict[str, Any]) -> None:
        """
        Ajoute des Z-Scores synthétiques basés sur les oscillateurs disponibles.
        Utilise la même logique que ZScore_Extreme_Reversal_Strategy.
        """
        try:
            # 1. Z-Score basé sur Bollinger Bands position (méthode principale)
            bb_position = indicators.get('bb_position')
            if bb_position is not None:
                try:
                    bb_pos = float(bb_position)
                    # Convertir position BB (0-1) vers Z-Score (-3 à +3)
                    zscore_bb = (bb_pos - 0.5) * 6
                    indicators['zscore_20'] = zscore_bb
                    indicators['zscore_50'] = zscore_bb * 0.8  # Version atténuée
                    logger.debug(f"FieldConverter: Created zscore_20={zscore_bb:.2f} from bb_position={bb_pos:.3f}")
                except (ValueError, TypeError):
                    pass
            
            # 2. Z-Score basé sur RSI (normalisation -3 à +3)
            rsi_14 = indicators.get('rsi_14')
            if rsi_14 is not None:
                try:
                    rsi_val = float(rsi_14)
                    # RSI 0-100 vers Z-Score -3 à +3
                    # RSI 50 = Z-Score 0, RSI 100 = Z-Score +3, RSI 0 = Z-Score -3
                    zscore_rsi = (rsi_val - 50) / 50 * 3
                    indicators['zscore_rsi'] = zscore_rsi
                    logger.debug(f"FieldConverter: Created zscore_rsi={zscore_rsi:.2f} from rsi={rsi_val:.1f}")
                except (ValueError, TypeError):
                    pass
            
            # 3. Z-Score basé sur Williams %R (normalisation)
            williams_r = indicators.get('williams_r')
            if williams_r is not None:
                try:
                    williams_val = float(williams_r)
                    # Williams %R -100 à 0 vers Z-Score -3 à +3
                    zscore_williams = (williams_val + 50) / 50 * 3
                    indicators['zscore_williams'] = zscore_williams
                    logger.debug(f"FieldConverter: Created zscore_williams={zscore_williams:.2f} from williams_r={williams_val:.1f}")
                except (ValueError, TypeError):
                    pass
            
            # 4. Z-Score basé sur CCI (normalisation)
            cci_20 = indicators.get('cci_20')
            if cci_20 is not None:
                try:
                    cci_val = float(cci_20)
                    # CCI normalement entre -200 et +200, normaliser vers Z-Score -3 à +3
                    zscore_cci = max(-3, min(3, cci_val / 66.67))  # 200/3 = 66.67
                    indicators['zscore_cci'] = zscore_cci
                    logger.debug(f"FieldConverter: Created zscore_cci={zscore_cci:.2f} from cci_20={cci_val:.1f}")
                except (ValueError, TypeError):
                    pass
            
            # 5. Z-Score basé sur MACD histogram (approximation)
            macd_histogram = indicators.get('macd_histogram')
            if macd_histogram is not None:
                try:
                    macd_val = float(macd_histogram)
                    # Normalisation approximative du MACD histogram
                    # Assumons une plage typique de -100 à +100 pour le MACD
                    zscore_macd = max(-3, min(3, macd_val / 50))
                    indicators['zscore_macd'] = zscore_macd
                    logger.debug(f"FieldConverter: Created zscore_macd={zscore_macd:.2f} from macd_histogram={macd_val:.2f}")
                except (ValueError, TypeError):
                    pass
                    
            # 6. Ajout de Z-Scores manquants avec des valeurs par défaut
            zscore_defaults = {
                'zscore_20': 0.0,
                'zscore_50': 0.0, 
                'zscore_rsi': 0.0,
                'zscore_cci': 0.0,
                'zscore_macd': 0.0,
                'zscore_volume': 0.0,
                'zscore_price': 0.0,
                # Z-Scores spécialisés pour validators
                'price_zscore': 0.0,
                'volume_zscore': 0.0,
                'returns_zscore': 0.0
            }
            
            for zscore_key, default_val in zscore_defaults.items():
                if zscore_key not in indicators:
                    indicators[zscore_key] = default_val
                    
        except Exception as e:
            logger.error(f"FieldConverter: Erreur création Z-Scores synthétiques: {e}")
            # En cas d'erreur, ajouter des valeurs par défaut
            for zscore_key in ['zscore_20', 'zscore_50', 'zscore_rsi', 'zscore_cci', 'zscore_macd', 
                             'price_zscore', 'volume_zscore', 'returns_zscore']:
                if zscore_key not in indicators:
                    indicators[zscore_key] = 0.0
    
    @classmethod
    def _add_specialized_zscore_fields(cls, indicators: Dict[str, Any]) -> None:
        """
        Ajoute des champs Z-Score spécialisés requis par certains validators.
        Utilise les indicateurs existants comme proxies.
        """
        try:
            # 1. price_zscore - utiliser zscore_20 comme proxy ou créer depuis BB position
            if 'price_zscore' not in indicators or indicators['price_zscore'] is None:
                if 'zscore_20' in indicators and indicators['zscore_20'] is not None:
                    indicators['price_zscore'] = indicators['zscore_20']
                elif 'bb_position' in indicators and indicators['bb_position'] is not None:
                    try:
                        bb_pos = float(indicators['bb_position'])
                        indicators['price_zscore'] = (bb_pos - 0.5) * 6  # Conversion BB vers Z-Score
                    except (ValueError, TypeError):
                        indicators['price_zscore'] = 0.0
                else:
                    indicators['price_zscore'] = 0.0
                    
            # 2. volume_zscore - utiliser relative_volume comme proxy
            if 'volume_zscore' not in indicators or indicators['volume_zscore'] is None:
                if 'relative_volume' in indicators and indicators['relative_volume'] is not None:
                    try:
                        rel_vol = float(indicators['relative_volume'])
                        # Normaliser relative_volume (1.0 = normal) vers Z-Score
                        # rel_vol > 1 = Z-Score positif, rel_vol < 1 = Z-Score négatif
                        indicators['volume_zscore'] = max(-3, min(3, (rel_vol - 1.0) * 3))
                    except (ValueError, TypeError):
                        indicators['volume_zscore'] = 0.0
                elif 'volume_ratio' in indicators and indicators['volume_ratio'] is not None:
                    try:
                        vol_ratio = float(indicators['volume_ratio'])
                        indicators['volume_zscore'] = max(-3, min(3, (vol_ratio - 1.0) * 2.5))
                    except (ValueError, TypeError):
                        indicators['volume_zscore'] = 0.0
                else:
                    indicators['volume_zscore'] = 0.0
                    
            # 3. returns_zscore - utiliser momentum comme proxy
            if 'returns_zscore' not in indicators or indicators['returns_zscore'] is None:
                if 'momentum_score' in indicators and indicators['momentum_score'] is not None:
                    try:
                        momentum = float(indicators['momentum_score'])
                        # momentum_score est entre 0-100 (50 = neutre), convertir vers Z-Score
                        # Normaliser : (momentum - 50) / 50 * 3 pour obtenir Z-Score -3 à +3
                        indicators['returns_zscore'] = (momentum - 50.0) / 50.0 * 3.0
                    except (ValueError, TypeError):
                        indicators['returns_zscore'] = 0.0
                elif 'roc_10' in indicators and indicators['roc_10'] is not None:
                    try:
                        roc = float(indicators['roc_10'])
                        # ROC en pourcentage, normaliser vers Z-Score
                        indicators['returns_zscore'] = max(-3, min(3, roc / 5))  # ROC/5 pour scaling
                    except (ValueError, TypeError):
                        indicators['returns_zscore'] = 0.0
                else:
                    indicators['returns_zscore'] = 0.0
                    
            # 4. Champs de contexte statistique manquants
            if 'distribution_normality' not in indicators:
                # Approximer normalité avec régime de marché
                regime = indicators.get('market_regime', 'unknown')
                if regime == 'ranging':
                    indicators['distribution_normality'] = 0.8  # Marché ranging = plus normal
                elif regime == 'trending':
                    indicators['distribution_normality'] = 0.6  # Trending = moins normal
                else:
                    indicators['distribution_normality'] = 0.5
                    
            if 'statistical_confluence' not in indicators:
                # Utiliser confluence_score si disponible
                if 'confluence_score' in indicators and indicators['confluence_score'] is not None:
                    try:
                        conf_score = float(indicators['confluence_score'])
                        # Convertir score 0-100 vers 0-1 si nécessaire
                        if conf_score > 1:
                            indicators['statistical_confluence'] = conf_score / 100.0
                        else:
                            indicators['statistical_confluence'] = conf_score
                    except (ValueError, TypeError):
                        indicators['statistical_confluence'] = 0.5
                else:
                    indicators['statistical_confluence'] = 0.5
                    
            if 'zscore_stability' not in indicators:
                # Approximer stabilité avec régime confidence
                regime_conf = indicators.get('regime_confidence', 50.0)  # Format 0-100
                try:
                    regime_conf_val = float(regime_conf)
                    # Convertir 0-100 vers 0-1 pour zscore_stability
                    indicators['zscore_stability'] = regime_conf_val / 100.0
                except (ValueError, TypeError):
                    indicators['zscore_stability'] = 0.5
                    
            logger.debug(f"FieldConverter: Added specialized Z-Score fields - "
                        f"price_zscore: {indicators.get('price_zscore', 'N/A'):.2f}, "
                        f"volume_zscore: {indicators.get('volume_zscore', 'N/A'):.2f}, "
                        f"returns_zscore: {indicators.get('returns_zscore', 'N/A'):.2f}")
                        
        except Exception as e:
            logger.error(f"FieldConverter: Erreur ajout champs Z-Score spécialisés: {e}")
            # Valeurs par défaut en cas d'erreur
            default_zscore_fields = {
                'price_zscore': 0.0,
                'volume_zscore': 0.0,
                'returns_zscore': 0.0,
                'distribution_normality': 0.5,
                'statistical_confluence': 0.5,
                'zscore_stability': 0.5
            }
            for field, default_val in default_zscore_fields.items():
                if field not in indicators:
                    indicators[field] = default_val
    
    @classmethod
    def _add_specialized_volume_spike_fields(cls, indicators: Dict[str, Any]) -> None:
        """
        Ajoute des champs volume spike spécialisés requis par Volume_Spike_Validator.
        Utilise les indicateurs volume existants comme proxies.
        """
        try:
            # 1. current_volume_spike - utiliser volume_spike_multiplier si disponible
            if 'current_volume_spike' not in indicators:
                if 'volume_spike_multiplier' in indicators and indicators['volume_spike_multiplier'] is not None:
                    indicators['current_volume_spike'] = indicators['volume_spike_multiplier']
                elif 'volume_ratio' in indicators and indicators['volume_ratio'] is not None:
                    indicators['current_volume_spike'] = indicators['volume_ratio']
                else:
                    indicators['current_volume_spike'] = 1.0
                    
            # 2. spike_quality_score - créer depuis volume_quality_score ou relative_volume
            if 'spike_quality_score' not in indicators:
                if 'volume_quality_score' in indicators and indicators['volume_quality_score'] is not None:
                    try:
                        vol_quality = float(indicators['volume_quality_score'])
                        # GARDER le format 0-100 pour les validators
                        indicators['spike_quality_score'] = vol_quality
                    except (ValueError, TypeError):
                        indicators['spike_quality_score'] = 0.5
                elif 'relative_volume' in indicators and indicators['relative_volume'] is not None:
                    try:
                        rel_vol = float(indicators['relative_volume'])
                        # Si relative_volume > 1.5, considérer comme bonne qualité
                        if rel_vol >= 2.0:
                            indicators['spike_quality_score'] = 0.8
                        elif rel_vol >= 1.5:
                            indicators['spike_quality_score'] = 0.6
                        else:
                            indicators['spike_quality_score'] = 0.4
                    except (ValueError, TypeError):
                        indicators['spike_quality_score'] = 0.5
                else:
                    indicators['spike_quality_score'] = 0.5
                    
            # 3. spike_duration_bars - approximer avec une valeur raisonnable
            if 'spike_duration_bars' not in indicators:
                # Estimer durée spike selon intensité
                volume_spike = indicators.get('volume_spike_multiplier', 1.0)
                try:
                    spike_val = float(volume_spike)
                    if spike_val >= 3.0:
                        indicators['spike_duration_bars'] = 2  # Spike fort, durée courte
                    elif spike_val >= 2.0:
                        indicators['spike_duration_bars'] = 3  # Spike modéré
                    else:
                        indicators['spike_duration_bars'] = 1  # Spike faible
                except (ValueError, TypeError):
                    indicators['spike_duration_bars'] = 1
                    
            # 4. time_since_spike - approximer avec 0 (spike actuel)
            if 'time_since_spike' not in indicators:
                indicators['time_since_spike'] = 0  # Considérer spike comme actuel
                
            # 5. relative_spike_strength - utiliser volume_spike_multiplier normalisé
            if 'relative_spike_strength' not in indicators:
                volume_spike = indicators.get('volume_spike_multiplier', 1.0)
                try:
                    spike_val = float(volume_spike)
                    # Normaliser vers 0-1 (1.0=pas de spike, 2.0+=spike fort)
                    if spike_val >= 4.0:
                        indicators['relative_spike_strength'] = 1.0  # Spike maximal
                    elif spike_val >= 2.0:
                        indicators['relative_spike_strength'] = (spike_val - 1.0) / 3.0  # Normalisation
                    else:
                        indicators['relative_spike_strength'] = 0.0  # Pas de spike
                except (ValueError, TypeError):
                    indicators['relative_spike_strength'] = 0.0
                    
            logger.debug(f"FieldConverter: Added volume spike fields - "
                        f"current_volume_spike: {indicators.get('current_volume_spike', 'N/A'):.2f}, "
                        f"spike_quality_score: {indicators.get('spike_quality_score', 'N/A'):.2f}, "
                        f"relative_spike_strength: {indicators.get('relative_spike_strength', 'N/A'):.2f}")
                        
        except Exception as e:
            logger.error(f"FieldConverter: Erreur ajout champs volume spike: {e}")
            # Valeurs par défaut en cas d'erreur
            default_spike_fields = {
                'current_volume_spike': 1.0,
                'spike_quality_score': 50.0,  # Format 0-100 pour validators
                'spike_duration_bars': 1,
                'time_since_spike': 0,
                'relative_spike_strength': 0.0
            }
            for field, default_val in default_spike_fields.items():
                if field not in indicators:
                    indicators[field] = default_val