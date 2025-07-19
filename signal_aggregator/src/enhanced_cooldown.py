"""
Module de gestion de cooldown amélioré
Évite les signaux BUY/SELL trop rapprochés et gère les cooldowns adaptatifs
"""
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class EnhancedCooldownManager:
    """
    Gestionnaire de cooldown avancé qui empêche les signaux opposés trop rapprochés
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        
        # Paramètres de cooldown
        self.base_cooldown_minutes = 15  # Cooldown de base entre signaux identiques
        self.opposite_signal_cooldown_minutes = 30  # Cooldown plus long entre signaux opposés
        self.spike_cooldown_minutes = 60  # Cooldown après un spike important
        
        # Cache local pour performance
        self.cooldown_cache = {}
        self.last_signals = {}  # {symbol: {'side': 'BUY/SELL', 'time': datetime, 'price': float}}
        
    def check_cooldown(self, symbol: str, new_side: str, 
                      current_price: Optional[float] = None) -> Tuple[bool, str]:
        """
        Vérifie si un nouveau signal respecte les périodes de cooldown
        
        Args:
            symbol: Symbole à vérifier
            new_side: 'BUY' ou 'SELL'
            current_price: Prix actuel (optionnel)
            
        Returns:
            (is_allowed, reason): True si le signal est autorisé
        """
        try:
            # Vérifier le cooldown Redis (distribué)
            cooldown_key = f"signal_cooldown:{symbol}"
            redis_cooldown = self.redis.get(cooldown_key)
            
            if redis_cooldown:
                remaining = self._get_remaining_cooldown(cooldown_key)
                return False, f"Cooldown actif encore {remaining:.0f} secondes"
            
            # Vérifier l'historique des derniers signaux
            if symbol in self.last_signals:
                last_signal = self.last_signals[symbol]
                last_side = last_signal['side']
                last_time = last_signal['time']
                last_price = last_signal.get('price', 0)
                
                time_since_last = (datetime.now(timezone.utc) - last_time).total_seconds() / 60
                
                # Si même direction, cooldown standard
                if new_side == last_side:
                    if time_since_last < self.base_cooldown_minutes:
                        return False, f"Cooldown même direction: attendez {self.base_cooldown_minutes - time_since_last:.0f} min"
                
                # Si direction opposée, cooldown plus long
                else:
                    if time_since_last < self.opposite_signal_cooldown_minutes:
                        # Exception: si le prix a beaucoup bougé (>5%), autoriser plus tôt
                        if current_price and last_price:
                            price_change_pct = abs((current_price - last_price) / last_price * 100)
                            if price_change_pct > 5.0 and time_since_last > 10:
                                logger.info(f"⚡ Cooldown opposé bypass: changement prix {price_change_pct:.1f}%")
                                return True, ""
                        
                        remaining = self.opposite_signal_cooldown_minutes - time_since_last
                        return False, f"Cooldown signal opposé: attendez {remaining:.0f} min"
            
            # Vérifier cooldown de spike
            spike_key = f"spike_cooldown:{symbol}"
            spike_cooldown = self.redis.get(spike_key)
            if spike_cooldown:
                remaining = self._get_remaining_cooldown(spike_key)
                return False, f"Cooldown post-spike: attendez {remaining:.0f} secondes"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Erreur check_cooldown: {e}")
            return True, ""  # En cas d'erreur, on laisse passer
    
    def set_cooldown(self, symbol: str, side: str, duration_minutes: Optional[int] = None,
                    price: Optional[float] = None, is_spike: bool = False):
        """
        Définit un cooldown pour un symbole
        
        Args:
            symbol: Symbole concerné
            side: 'BUY' ou 'SELL'
            duration_minutes: Durée en minutes (optionnel)
            price: Prix du signal
            is_spike: Si True, applique un cooldown de spike
        """
        try:
            # Déterminer la durée
            if is_spike:
                duration = self.spike_cooldown_minutes
                key = f"spike_cooldown:{symbol}"
            else:
                duration = duration_minutes or self.base_cooldown_minutes
                key = f"signal_cooldown:{symbol}"
            
            # Définir cooldown Redis
            self.redis.set(key, "1", expiration=int(duration * 60))
            
            # Mettre à jour l'historique local
            self.last_signals[symbol] = {
                'side': side,
                'time': datetime.now(timezone.utc),
                'price': price or 0
            }
            
            logger.info(f"⏱️ Cooldown défini pour {symbol}: {duration} minutes")
            
        except Exception as e:
            logger.error(f"Erreur set_cooldown: {e}")
    
    def get_adaptive_cooldown(self, symbol: str, volatility: Optional[float] = None, 
                            volume_ratio: Optional[float] = None) -> int:
        """
        Calcule un cooldown adaptatif basé sur les conditions du marché
        
        Args:
            symbol: Symbole
            volatility: Volatilité actuelle (ATR/prix)
            volume_ratio: Ratio volume actuel vs moyenne
            
        Returns:
            Durée du cooldown en minutes
        """
        base = self.base_cooldown_minutes
        
        # Ajuster selon la volatilité
        if volatility:
            if volatility < 0.01:  # Très faible volatilité
                base *= 2  # Doubler le cooldown
            elif volatility > 0.03:  # Haute volatilité
                base *= 0.7  # Réduire le cooldown
        
        # Ajuster selon le volume - SEUILS STANDARDISÉS
        if volume_ratio:
            if volume_ratio < 0.5:  # Volume faible
                base *= 1.5
            elif volume_ratio > 2.0:  # STANDARDISÉ: Excellent (début pump confirmé)
                base *= 0.8
            elif volume_ratio > 1.5:  # STANDARDISÉ: Très bon
                base *= 0.9
        
        # Limiter entre 5 et 60 minutes
        return max(5, min(60, int(base)))
    
    def _get_remaining_cooldown(self, key: str) -> float:
        """
        Retourne le temps restant de cooldown en secondes
        """
        try:
            ttl = self.redis.ttl(key)
            return max(0, ttl if ttl else 0)
        except:
            return 0
    
    def clear_cooldown(self, symbol: str):
        """
        Efface le cooldown pour un symbole (utilisé en cas d'urgence)
        """
        try:
            self.redis.delete(f"signal_cooldown:{symbol}")
            self.redis.delete(f"spike_cooldown:{symbol}")
            if symbol in self.last_signals:
                del self.last_signals[symbol]
            logger.info(f"🔓 Cooldown effacé pour {symbol}")
        except Exception as e:
            logger.error(f"Erreur clear_cooldown: {e}")
    
    def get_cooldown_status(self, symbol: str) -> Dict:
        """
        Retourne le statut détaillé du cooldown pour un symbole
        """
        status = {
            'symbol': symbol,
            'in_cooldown': False,
            'remaining_seconds': 0,
            'last_signal': None,
            'cooldown_type': None
        }
        
        # Vérifier les différents cooldowns
        for cooldown_type, key in [
            ('signal', f"signal_cooldown:{symbol}"),
            ('spike', f"spike_cooldown:{symbol}")
        ]:
            remaining = self._get_remaining_cooldown(key)
            if remaining > 0:
                status['in_cooldown'] = True
                status['remaining_seconds'] = remaining
                status['cooldown_type'] = cooldown_type
                break
        
        # Ajouter info du dernier signal
        if symbol in self.last_signals:
            last = self.last_signals[symbol]
            status['last_signal'] = {
                'side': last['side'],
                'time': last['time'].isoformat(),
                'price': last.get('price', 0)
            }
        
        return status