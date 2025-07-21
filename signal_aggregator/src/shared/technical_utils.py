"""
Utilitaires techniques partagés pour éviter la duplication de code
NOTES: 
- Les calculs EMA/ADX robustes sont dans /shared/src/technical_indicators.py
- Ce module contient des helpers spécifiques au signal_aggregator (Redis, fallbacks)
- Pour les calculs techniques complets, préférer l'import du module global
"""

import json
import logging
import sys
import os
from typing import List, Optional, Dict, Any

# Import du module technique global (pour référence future)
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
    from shared.src.technical_indicators import TechnicalIndicators as GlobalTechnicalIndicators
    GLOBAL_INDICATORS_AVAILABLE = True
except ImportError:
    GLOBAL_INDICATORS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Module technique global non disponible, utilisation des implémentations locales")

logger = logging.getLogger(__name__)


class TechnicalCalculators:
    """
    Calculateurs techniques centralisés pour le signal_aggregator
    
    NOTE: Pour des calculs techniques robustes (avec talib/pandas), 
    préférer GlobalTechnicalIndicators qui a des implémentations complètes.
    Ce module contient des helpers simples + spécificités Redis du service.
    """
    
    @staticmethod
    def calculate_ema(data: List[float], period: int) -> List[float]:
        """
        Calcule une EMA complète en utilisant le module technique global robuste
        
        Args:
            data: Liste des prix
            period: Période pour le calcul EMA
            
        Returns:
            Liste des valeurs EMA
        """
        if not data or period <= 0:
            return data.copy() if data else []
        
        try:
            if GLOBAL_INDICATORS_AVAILABLE:
                # Utiliser l'implémentation globale robuste avec talib/pandas
                indicators = GlobalTechnicalIndicators()
                ema_series = indicators.calculate_ema_series(data, period)
                # Convertir en liste Python standard
                if hasattr(ema_series, 'tolist'):
                    return ema_series.tolist()
                elif hasattr(ema_series, '__iter__'):
                    return list(ema_series)
                else:
                    return [float(ema_series)]
            else:
                # Fallback simple si module global indisponible
                return TechnicalCalculators._calculate_ema_fallback(data, period)
        except Exception as e:
            logger.warning(f"Erreur calcul EMA global: {e}, utilisation fallback")
            return TechnicalCalculators._calculate_ema_fallback(data, period)
    
    @staticmethod
    def _calculate_ema_fallback(data: List[float], period: int) -> List[float]:
        """Fallback EMA simple si module global indisponible"""
        if len(data) < period:
            return data.copy()
        
        alpha = 2.0 / (period + 1)
        ema = [data[0]]
        
        for i in range(1, len(data)):
            ema_value = alpha * data[i] + (1 - alpha) * ema[-1]
            ema.append(ema_value)
        
        return ema
    
    @staticmethod
    def calculate_single_ema(prices: List[float], period: int) -> float:
        """
        Calcule une seule valeur EMA en utilisant le module technique global
        
        Args:
            prices: Liste des prix
            period: Période pour le calcul EMA
            
        Returns:
            Dernière valeur EMA
        """
        if not prices or period <= 0:
            return prices[-1] if prices else 0.0
        
        try:
            if GLOBAL_INDICATORS_AVAILABLE:
                # Utiliser l'implémentation globale robuste
                indicators = GlobalTechnicalIndicators()
                ema_value = indicators.calculate_ema(prices, period)
                return float(ema_value) if ema_value is not None else 0.0
            else:
                # Fallback simple si module global indisponible
                return TechnicalCalculators._calculate_single_ema_fallback(prices, period)
        except Exception as e:
            logger.warning(f"Erreur calcul EMA single global: {e}, utilisation fallback")
            return TechnicalCalculators._calculate_single_ema_fallback(prices, period)
    
    @staticmethod
    def _calculate_single_ema_fallback(prices: List[float], period: int) -> float:
        """Fallback EMA single si module global indisponible"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
            
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    @staticmethod
    async def get_current_adx(redis_client, symbol: str) -> Optional[float]:
        """
        Récupère la valeur ADX actuelle depuis Redis avec fallback multi-timeframes
        
        Args:
            redis_client: Client Redis
            symbol: Symbole concerné
            
        Returns:
            Valeur ADX ou None si non disponible
        """
        try:
            # Essayer d'abord les données 1m (plus récentes)
            for timeframe in ['1m', '5m', '15m']:
                market_data_key = f"market_data:{symbol}:{timeframe}"
                data = redis_client.get(market_data_key)
                
                if data:
                    try:
                        if isinstance(data, str):
                            market_data = json.loads(data)
                        else:
                            market_data = data
                        
                        # Essayer différents formats d'ADX
                        adx = market_data.get('adx_14', market_data.get('adx', None))
                        if adx is not None:
                            return float(adx)
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue
            
            logger.debug(f"ADX non trouvé pour {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Erreur récupération ADX pour {symbol}: {e}")
            return None
    
    @staticmethod
    def extract_adx_from_data(market_data: Dict) -> float:
        """
        Extrait ADX depuis des données de marché
        
        Args:
            market_data: Dictionnaire des données de marché
            
        Returns:
            Valeur ADX ou 0.0 par défaut
        """
        try:
            adx = market_data.get('adx_14', market_data.get('adx', 0))
            return float(adx) if adx is not None else 0.0
        except (ValueError, TypeError):
            return 0.0


class VolumeAnalyzer:
    """Analyseur de volumes centralisé"""
    
    @staticmethod
    def calculate_volume_boost(volume_ratio: float) -> float:
        """
        Calcule le boost basé sur le ratio de volume
        
        Args:
            volume_ratio: Ratio du volume par rapport à la moyenne
            
        Returns:
            Facteur de boost
        """
        if volume_ratio >= 2.0:  # Excellent
            return 1.15
        elif volume_ratio >= 1.5:  # Très bon
            return 1.10
        elif volume_ratio >= 1.2:  # Bon
            return 1.05
        else:  # Normal
            return 1.0
    
    @staticmethod
    def analyze_volume_strength(metadata: Dict[str, Any]) -> tuple[float, str]:
        """
        Analyse la force du volume depuis les métadonnées
        
        Args:
            metadata: Métadonnées du signal
            
        Returns:
            (boost_factor, description)
        """
        volume_ratio = metadata.get('volume_ratio', 1.0)
        boost = VolumeAnalyzer.calculate_volume_boost(volume_ratio)
        
        if volume_ratio >= 2.0:
            description = f"Volume exceptionnel ({volume_ratio:.1f}x)"
        elif volume_ratio >= 1.5:
            description = f"Volume fort ({volume_ratio:.1f}x)"
        elif volume_ratio >= 1.2:
            description = f"Volume élevé ({volume_ratio:.1f}x)"
        else:
            description = f"Volume normal ({volume_ratio:.1f}x)"
            
        return boost, description


class SignalValidators:
    """Validateurs de signaux centralisés"""
    
    @staticmethod
    def validate_confidence_threshold(signal_confidence: float, min_confidence: float, 
                                    symbol: str = None, strategy: str = None) -> bool:
        """
        Valide si la confiance du signal dépasse le seuil minimum
        
        Args:
            signal_confidence: Confiance du signal (0-1)
            min_confidence: Seuil minimum requis (0-1)
            symbol: Symbole pour le logging (optionnel)
            strategy: Stratégie pour le logging (optionnel)
            
        Returns:
            True si validation réussie
        """
        if signal_confidence < min_confidence:
            context = ""
            if strategy and symbol:
                context = f" ({strategy} pour {symbol})"
            elif strategy:
                context = f" ({strategy})"
            elif symbol:
                context = f" pour {symbol}"
                
            logger.info(f"Signal rejeté{context}: confiance {signal_confidence:.2f} < {min_confidence:.2f}")
            return False
        return True
    
    @staticmethod
    def validate_signal_side(side: str, symbol: str = None) -> bool:
        """
        Valide que le side du signal est correct
        
        Args:
            side: Side du signal ('BUY' ou 'SELL')
            symbol: Symbole pour le logging (optionnel)
            
        Returns:
            True si valid
        """
        if side not in ['BUY', 'SELL']:
            context = f" pour {symbol}" if symbol else ""
            logger.warning(f"❌ Side invalide{context}: {side}")
            return False
        return True
    
    @staticmethod
    def validate_price_range(price: float, symbol: str = None, min_price: float = 0.0001, 
                           max_price: float = 1000000.0) -> bool:
        """
        Valide qu'un prix est dans une plage acceptable
        
        Args:
            price: Prix à valider
            symbol: Symbole pour le logging (optionnel)
            min_price: Prix minimum acceptable
            max_price: Prix maximum acceptable
            
        Returns:
            True si valid
        """
        if not isinstance(price, (int, float)) or price <= 0:
            context = f" pour {symbol}" if symbol else ""
            logger.warning(f"❌ Prix invalide{context}: {price}")
            return False
            
        if price < min_price or price > max_price:
            context = f" pour {symbol}" if symbol else ""
            logger.warning(f"❌ Prix hors limites{context}: {price} (limites: {min_price}-{max_price})")
            return False
            
        return True
    
    @staticmethod
    def validate_volume_threshold(volume: float, min_volume: float = 1000.0, 
                                symbol: str = None) -> bool:
        """
        Valide qu'un volume dépasse le seuil minimum
        
        Args:
            volume: Volume à valider
            min_volume: Volume minimum requis
            symbol: Symbole pour le logging (optionnel)
            
        Returns:
            True si valid
        """
        if not isinstance(volume, (int, float)) or volume < 0:
            context = f" pour {symbol}" if symbol else ""
            logger.warning(f"❌ Volume invalide{context}: {volume}")
            return False
            
        if volume < min_volume:
            context = f" pour {symbol}" if symbol else ""
            logger.debug(f"Volume insuffisant{context}: {volume} < {min_volume}")
            return False
            
        return True