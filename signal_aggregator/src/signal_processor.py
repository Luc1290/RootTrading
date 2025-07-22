#!/usr/bin/env python3
"""
Module pour le traitement des signaux de trading.
Contient les méthodes de traitement spécialisé extraites du signal_aggregator principal.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import sys, os
sys.path.append(os.path.dirname(__file__))
try:
    from .shared.technical_utils import TechnicalCalculators
except ImportError:
    # Fallback pour l'exécution dans le conteneur
    sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))
    from technical_utils import TechnicalCalculators

logger = logging.getLogger(__name__)


class SignalProcessor:
    """Classe pour le traitement spécialisé des signaux"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        
    async def process_institutional_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Traitement express pour signaux de qualité institutionnelle (95+ points)"""
        try:
            symbol = signal['symbol']
            metadata = signal.get('metadata', {})
            
            # Traitement express - validation minimale
            current_price = signal['price']
            confidence = min(signal.get('confidence', 0.9), 1.0)  # Cap à 1.0
            
            # Force basée sur le score
            score = metadata.get('total_score', 95)
            if score >= 98:
                strength = 'very_strong'
            else:
                strength = 'strong'
                
            # Utiliser les niveaux de prix calculés par ultra-confluence
            price_levels = metadata.get('price_levels', {})
            # Stop-loss correct selon le side: SELL stop au dessus, BUY stop en dessous
            side = signal.get('side', 'BUY')
            default_stop = current_price * (1.025 if side == 'SELL' else 0.975)
            stop_loss = price_levels.get('stop_loss', default_stop)  # Stop plus serré pour signaux premium
            
            # Métadonnées enrichies
            enhanced_metadata = {
                'aggregated': True,
                'institutional_grade': True,
                'ultra_confluence': True,
                'total_score': score,
                'quality': metadata.get('quality', 'institutional'),
                'confirmation_count': metadata.get('confirmation_count', 0),
                'express_processing': True,
                'timeframes_analyzed': metadata.get('timeframes_analyzed', []),
                'stop_price': stop_loss,
                'trailing_delta': 2.0,  # Trailing plus serré pour signaux premium
                'recommended_size_multiplier': 1.2  # Taille légèrement augmentée
            }
            
            # Log pour debug stop-loss
            logger.info(f"🎯 Signal institutionnel {side} {symbol}: entry={current_price:.4f}, stop={stop_loss:.4f}")
            
            result = {
                'symbol': symbol,
                'side': signal['side'],
                'price': current_price,
                'strategy': 'UltraConfluence_Institutional',
                'confidence': confidence,
                'strength': strength,
                'stop_loss': stop_loss,
                'trailing_delta': 2.0,
                'contributing_strategies': ['UltraConfluence'],
                'metadata': enhanced_metadata
            }
            
            logger.info(f"⭐ Signal INSTITUTIONNEL traité: {symbol} {signal['side']} @ {current_price:.4f} (score={score:.1f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement signal institutionnel: {e}")
            return None
            
    async def process_excellent_signal(self, signal: Dict[str, Any], cooldown_setter) -> Optional[Dict[str, Any]]:
        """Traitement prioritaire pour signaux excellents (85+ points)"""
        try:
            symbol = signal['symbol']
            metadata = signal.get('metadata', {})
            
            # Validation légère mais présente
            if await self._is_in_cooldown(symbol):
                logger.debug(f"Signal excellent {symbol} en cooldown, ignoré")
                return None
                
            # Vérification ADX allégée pour signaux excellents
            adx_value = await self._get_current_adx(symbol)
            score = metadata.get('total_score', 85)
            
            if adx_value and adx_value < 20 and score < 90:  # Seuil ADX plus strict seulement pour scores < 90
                logger.info(f"Signal excellent rejeté: ADX trop faible ({adx_value:.1f}) pour score {score:.1f}")
                return None
                
            current_price = signal['price']
            confidence = signal.get('confidence', 0.85)
            
            # Ajuster la confiance basée sur le score
            confidence_boost = min((score - 85) / 15 * 0.1, 0.1)  # Max 10% boost
            confidence = min(confidence + confidence_boost, 1.0)
            
            # Force basée sur le score et la confluence
            confirmation_count = metadata.get('confirmation_count', 0)
            if score >= 90 and confirmation_count >= 15:
                strength = 'very_strong'
            elif score >= 85:
                strength = 'strong'
            else:
                strength = 'moderate'
                
            # Prix et stop loss optimisés
            price_levels = metadata.get('price_levels', {})
            # Stop-loss correct selon le side: SELL stop au dessus, BUY stop en dessous
            side = signal.get('side', 'BUY')
            default_stop = current_price * (1.02 if side == 'SELL' else 0.98)
            stop_loss = price_levels.get('stop_loss', default_stop)  # Stop modéré
            
            enhanced_metadata = {
                'aggregated': True,
                'excellent_grade': True,
                'ultra_confluence': True,
                'total_score': score,
                'quality': metadata.get('quality', 'excellent'),
                'confirmation_count': confirmation_count,
                'priority_processing': True,
                'timeframes_analyzed': metadata.get('timeframes_analyzed', []),
                'stop_price': stop_loss,
                'trailing_delta': 2.5,
                'recommended_size_multiplier': 1.1
            }
            
            # Log pour debug stop-loss
            logger.info(f"🎯 Signal excellent {side} {symbol}: entry={current_price:.4f}, stop={stop_loss:.4f}")
            
            result = {
                'symbol': symbol,
                'side': signal['side'],
                'price': current_price,
                'strategy': 'UltraConfluence_Excellent',
                'confidence': confidence,
                'strength': strength,
                'stop_loss': stop_loss,
                'trailing_delta': 2.5,
                'contributing_strategies': ['UltraConfluence'],
                'metadata': enhanced_metadata
            }
            
            # Définir cooldown court pour signaux excellents
            await cooldown_setter(symbol, 60)  # 1 minute seulement
            
            logger.info(f"✨ Signal EXCELLENT traité: {symbol} {signal['side']} @ {current_price:.4f} (score={score:.1f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement signal excellent: {e}")
            return None
    
    async def _is_in_cooldown(self, symbol: str) -> bool:
        """Vérifie si le symbole est en cooldown"""
        cooldown_key = f"signal_cooldown:{symbol}"
        cooldown = self.redis.get(cooldown_key)
        return cooldown is not None
    
    async def _get_current_adx(self, symbol: str) -> Optional[float]:
        """
        Récupère la valeur ADX actuelle depuis Redis - utilise l'implémentation partagée
        
        Args:
            symbol: Symbole concerné
            
        Returns:
            Valeur ADX ou None si non disponible
        """
        return await TechnicalCalculators.get_current_adx(self.redis, symbol)
    
    def get_signal_timestamp(self, signal: Dict[str, Any]) -> datetime:
        """Extract timestamp from signal with multiple format support"""
        timestamp_str = signal.get('timestamp', signal.get('created_at'))
        if timestamp_str:
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                return timestamp
            else:
                return datetime.fromtimestamp(timestamp_str / 1000 if timestamp_str > 1e10 else timestamp_str, tz=timezone.utc)
        return datetime.now(timezone.utc)
    
    async def track_signal_accuracy(self, signal: Dict):
        """
        Suit la précision des signaux pour ajuster les poids dynamiquement
        """
        # Stocker le signal pour vérification future
        signal_key = f"pending_signal:{signal['symbol']}:{signal['strategy']}"
        signal_data = {
            'entry_price': signal['price'],
            'side': signal['side'],
            'timestamp': datetime.now().isoformat(),
            'stop_loss': signal.get('stop_loss'),
            'confidence': signal.get('confidence')
        }
        
        # Gérer les différents types de clients Redis
        try:
            import json
            self.redis.set(signal_key, json.dumps(signal_data), ex=3600)
        except TypeError:
            # Fallback pour RedisClientPool customisé
            import json
            self.redis.set(signal_key, json.dumps(signal_data), expiration=3600)