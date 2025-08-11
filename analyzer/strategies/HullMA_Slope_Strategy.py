"""
HullMA_Slope_Strategy - Stratégie basée sur la pente de Hull Moving Average.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class HullMA_Slope_Strategy(BaseStrategy):
    """
    Stratégie utilisant la pente de Hull Moving Average pour détecter les changements de tendance.
    
    Hull MA est une moyenne mobile rapide et lisse qui réduit le lag tout en éliminant le bruit.
    
    Signaux générés:
    - BUY: Hull MA en pente haussière + prix au-dessus + confirmations
    - SELL: Hull MA en pente baissière + prix en-dessous + confirmations
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres HullMA - OPTIMISÉS
        self.hullma_period = 20        # Période par défaut
        self.min_slope_threshold = 0.0008  # Pente minimum AUGMENTÉE (0.08% au lieu de 0.1%)
        self.strong_slope_threshold = 0.0015  # Pente forte AUGMENTÉE (0.15% au lieu de 0.5%)
        self.price_distance_max = 0.015   # Distance max RÉDUITE (1.5% au lieu de 2%)
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs."""
        return {
            # Hull MA
            'hull_20': self.indicators.get('hull_20'),
            # Autres moyennes mobiles pour comparaison
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            'sma_20': self.indicators.get('sma_20'),
            'sma_50': self.indicators.get('sma_50'),
            # Indicateurs de pente/angle (si disponibles)
            'trend_angle': self.indicators.get('trend_angle'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            # Momentum pour confirmation
            'momentum_score': self.indicators.get('momentum_score'),
            'rsi_14': self.indicators.get('rsi_14'),
            'macd_line': self.indicators.get('macd_line'),
            'macd_histogram': self.indicators.get('macd_histogram'),
            # Volume
            'volume_ratio': self.indicators.get('volume_ratio'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            # Contexte marché
            'market_regime': self.indicators.get('market_regime'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            # Confluence
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score')
        }
        
    def _get_price_data(self) -> Dict[str, Optional[float]]:
        """Récupère les données de prix pour calcul de pente."""
        try:
            if self.data and 'close' in self.data and self.data['close'] and len(self.data['close']) >= 3:
                prices = self.data['close']
                return {
                    'current_price': float(prices[-1]),
                    'prev_price_1': float(prices[-2]),
                    'prev_price_2': float(prices[-3]) if len(prices) >= 3 else None
                }
        except (IndexError, ValueError, TypeError):
            pass
        return {'current_price': None, 'prev_price_1': None, 'prev_price_2': None}
        
    def _calculate_slope_approximation(self, current_hull: float, price_data: Dict[str, Optional[float]]) -> Optional[float]:
        """
        Calcule une approximation de la pente Hull MA en utilisant le prix et trend_angle.
        
        Note: Idéalement on aurait besoin de plusieurs valeurs Hull MA historiques,
        mais on peut approximer avec les données disponibles.
        """
        try:
            current_price = price_data['current_price']
            prev_price_1 = price_data['prev_price_1']
            
            if current_price is None or prev_price_1 is None:
                return None
                
            # Approximation simple : différence relative entre Hull MA et prix
            price_change = (current_price - prev_price_1) / prev_price_1
            hull_price_ratio = current_hull / current_price
            
            # Si Hull MA est proche du prix et prix monte, Hull MA probablement en pente haussière
            # Cette approximation n'est pas parfaite mais utilise les données disponibles
            slope_approx = price_change * hull_price_ratio
            
            return slope_approx
            
        except (ZeroDivisionError, TypeError):
            return None
            
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur la pente de Hull MA.
        """
        if not self.validate_data():
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données insuffisantes",
                "metadata": {"strategy": self.name}
            }
            
        values = self._get_current_values()
        price_data = self._get_price_data()
        
        # Vérification Hull MA
        try:
            hull_20 = float(values['hull_20']) if values['hull_20'] is not None else None
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion Hull MA: {e}",
                "metadata": {"strategy": self.name}
            }
            
        current_price = price_data['current_price']
        
        if hull_20 is None or current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Hull MA ou prix non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # Calcul de la position prix vs Hull MA
        price_hull_distance = (current_price - hull_20) / hull_20
        price_above_hull = current_price > hull_20
        
        # Vérification que le prix n'est pas trop loin de Hull MA
        if abs(price_hull_distance) > self.price_distance_max:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Prix trop éloigné de Hull MA ({price_hull_distance*100:.1f}%)",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "current_price": current_price,
                    "hull_20": hull_20,
                    "distance_pct": price_hull_distance * 100
                }
            }
            
        # Approximation de la pente Hull MA
        slope_approx = self._calculate_slope_approximation(hull_20, price_data)
        
        # Utilisation de trend_angle si disponible (plus précis)
        trend_angle = values.get('trend_angle')
        if trend_angle is not None:
            try:
                angle_val = float(trend_angle)
                # Normaliser l'angle en pente approximative
                slope_approx = angle_val / 45.0  # Convertir degrés en ratio approximatif
            except (ValueError, TypeError):
                pass
                
        # Si pas de pente calculable, pas de signal
        if slope_approx is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Impossible de calculer la pente Hull MA",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "hull_20": hull_20,
                    "current_price": current_price
                }
            }
            
        signal_side = None
        reason = ""
        base_confidence = 0.50  # Standardisé à 0.50 pour équité avec autres stratégies
        confidence_boost = 0.0
        slope_strength = "flat"
        
        # Analyse de la pente
        slope_abs = abs(slope_approx)
        
        if slope_abs < self.min_slope_threshold:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Hull MA en pente plate ({slope_approx*100:.2f}%) - pas de tendance claire",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "hull_20": hull_20,
                    "current_price": current_price,
                    "slope_approx": slope_approx
                }
            }
            
        # Détermination du signal selon la pente
        if slope_approx > self.min_slope_threshold:
            # Pente haussière
            if price_above_hull:
                signal_side = "BUY"
                reason = f"Hull MA pente haussière ({slope_approx*100:.2f}%) + prix au-dessus"
                confidence_boost += 0.18  # AUGMENTÉ car c'est un signal de qualité
            else:
                # Prix sous Hull MA haussière = attendre confirmation
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Hull MA haussière mais prix en-dessous ({price_hull_distance*100:.1f}%)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "hull_20": hull_20,
                        "current_price": current_price,
                        "slope_approx": slope_approx
                    }
                }
                
        elif slope_approx < -self.min_slope_threshold:
            # Pente baissière
            if not price_above_hull:
                signal_side = "SELL"
                reason = f"Hull MA pente baissière ({slope_approx*100:.2f}%) + prix en-dessous"
                confidence_boost += 0.18  # AUGMENTÉ car c'est un signal de qualité
            else:
                # Prix au-dessus Hull MA baissière = attendre confirmation
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Hull MA baissière mais prix au-dessus (+{price_hull_distance*100:.1f}%)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "hull_20": hull_20,
                        "current_price": current_price,
                        "slope_approx": slope_approx
                    }
                }
                
        # Classification de la force de la pente
        if slope_abs >= self.strong_slope_threshold:
            slope_strength = "strong"
            confidence_boost += 0.20  # AUGMENTÉ - pente forte = excellent signal
            reason += " - pente FORTE"
        elif slope_abs >= self.min_slope_threshold * 2.5:  # Seuil augmenté
            slope_strength = "moderate"
            confidence_boost += 0.12  # Légèrement augmenté
            reason += " - pente modérée"
        else:
            slope_strength = "weak"
            confidence_boost += 0.03  # RÉDUIT - pente faible moins valorisée
            reason += " - pente faible"
            
        # Confirmation avec autres moyennes mobiles
        ema_50 = values.get('ema_50')
        if ema_50 is not None:
            try:
                ema50_val = float(ema_50)
                if signal_side == "BUY" and current_price > ema50_val:
                    confidence_boost += 0.08
                    reason += " + prix > EMA50"
                elif signal_side == "SELL" and current_price < ema50_val:
                    confidence_boost += 0.08
                    reason += " + prix < EMA50"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec trend_strength (VARCHAR: weak/absent/strong/very_strong/extreme)
        trend_strength = values.get('trend_strength')
        if trend_strength is not None:
            trend_str = str(trend_strength).lower()
            if trend_str in ['extreme', 'very_strong']:
                confidence_boost += 0.18  # AUGMENTÉ
                reason += f" + tendance {trend_str}"
            elif trend_str == 'strong':
                confidence_boost += 0.12  # AUGMENTÉ
                reason += f" + tendance {trend_str}"
            elif trend_str in ['moderate', 'present']:
                confidence_boost += 0.06  # Légèrement augmenté
                reason += f" + tendance {trend_str}"
            elif trend_str in ['weak', 'absent']:  # NOUVEAU: pénalité
                confidence_boost -= 0.08
                reason += f" MAIS tendance {trend_str}"
                
        # Confirmation avec directional_bias (VARCHAR: BULLISH/BEARISH/NEUTRAL)
        directional_bias = values.get('directional_bias')
        if directional_bias:
            bias_str = str(directional_bias).upper()
            if (signal_side == "BUY" and bias_str == "BULLISH") or \
               (signal_side == "SELL" and bias_str == "BEARISH"):
                confidence_boost += 0.10
                reason += f" + bias {bias_str.lower()}"
                
        # Confirmation avec trend_alignment (format décimal)
        trend_alignment = values.get('trend_alignment')
        if trend_alignment is not None:
            try:
                alignment = float(trend_alignment)
                if abs(alignment) > 0.3:  # Format décimal : 0.3 = strong alignment
                    confidence_boost += 0.10
                    reason += " + MA alignées"
                elif abs(alignment) > 0.1:
                    confidence_boost += 0.05
                    reason += " + MA partiellement alignées"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec momentum
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                # MOMENTUM PLUS STRICT (format 0-100, 50=neutre)
                if (signal_side == "BUY" and momentum > 60) or \
                   (signal_side == "SELL" and momentum < 40):
                    confidence_boost += 0.12  # Augmenté
                    reason += " + momentum FORT"
                elif (signal_side == "BUY" and momentum > 52) or \
                     (signal_side == "SELL" and momentum < 48):
                    confidence_boost += 0.06
                    reason += " + momentum favorable"
                elif (signal_side == "BUY" and momentum < 45) or \
                     (signal_side == "SELL" and momentum > 55):  # NOUVEAU: pénalité
                    confidence_boost -= 0.12
                    reason += " MAIS momentum CONTRAIRE"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec RSI (éviter zones extrêmes)
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                # RSI PLUS STRICT
                if signal_side == "BUY" and rsi < 65:  # Plus strict
                    confidence_boost += 0.06
                elif signal_side == "SELL" and rsi > 35:  # Plus strict
                    confidence_boost += 0.06
                elif signal_side == "BUY" and rsi >= 75:  # Seuil abaissé
                    confidence_boost -= 0.15  # Pénalité augmentée
                    reason += " ATTENTION: RSI surachat ({rsi:.1f})"
                elif signal_side == "SELL" and rsi <= 25:  # Seuil relevé
                    confidence_boost -= 0.15  # Pénalité augmentée
                    reason += " ATTENTION: RSI survente ({rsi:.1f})"
            except (ValueError, TypeError):
                pass
                
        # Volume pour confirmation
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                # VOLUME PLUS STRICT
                if vol_ratio >= 2.0:  # Seuil augmenté
                    confidence_boost += 0.15
                    reason += f" + volume TRÈS élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.5:
                    confidence_boost += 0.10
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.1:
                    confidence_boost += 0.05  # Réduit
                    reason += f" + volume modéré ({vol_ratio:.1f}x)"
                elif vol_ratio < 0.8:  # NOUVEAU: pénalité volume faible
                    confidence_boost -= 0.08
                    reason += f" mais volume FAIBLE ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # Volume quality (format 0-100)
        volume_quality_score = values.get('volume_quality_score')
        if volume_quality_score is not None:
            try:
                vol_quality = float(volume_quality_score)
                if vol_quality > 70:
                    confidence_boost += 0.08
                    reason += f" + volume qualité ({vol_quality:.0f})"
                elif vol_quality > 50:
                    confidence_boost += 0.05
                    reason += f" + volume correct ({vol_quality:.0f})"
            except (ValueError, TypeError):
                pass
                
        # Contexte marché
        market_regime = values.get('market_regime')
        if market_regime in ["TRENDING_BULL", "TRENDING_BEAR"]:
            confidence_boost += 0.08
            reason += " (marché trending)"
        elif market_regime == "RANGING":
            confidence_boost -= 0.05
            reason += " (marché ranging)"
            
        # Signal strength (VARCHAR: WEAK/MODERATE/STRONG/VERY_WEAK)
        signal_strength_calc = values.get('signal_strength')
        if signal_strength_calc is not None:
            sig_str = str(signal_strength_calc).upper()
            if sig_str == 'STRONG':
                confidence_boost += 0.10
                reason += " + signal fort"
            elif sig_str == 'MODERATE':
                confidence_boost += 0.05
                reason += " + signal modéré"
                
        # Confluence score (format 0-100)
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                # CONFLUENCE PLUS STRICTE
                if confluence > 75:  # Seuil augmenté
                    confidence_boost += 0.18
                    reason += f" + confluence EXCELLENTE ({confluence:.0f})"
                elif confluence > 65:  # Seuil augmenté
                    confidence_boost += 0.12
                    reason += f" + confluence élevée ({confluence:.0f})"
                elif confluence > 55:  # Seuil augmenté
                    confidence_boost += 0.06
                    reason += f" + confluence correcte ({confluence:.0f})"
                elif confluence < 45:  # NOUVEAU: pénalité
                    confidence_boost -= 0.10
                    reason += f" mais confluence FAIBLE ({confluence:.0f})"
            except (ValueError, TypeError):
                pass
                
        # NOUVEAU: Filtre final - rejeter si confidence trop faible
        raw_confidence = base_confidence * (1 + confidence_boost)
        if raw_confidence < 0.42:  # Seuil minimum 42%
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal HullMA rejeté - confiance insuffisante ({raw_confidence:.2f} < 0.42)",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "rejected_signal": signal_side,
                    "raw_confidence": raw_confidence,
                    "hull_slope": slope_approx,
                    "slope_strength": slope_strength
                }
            }
        
        confidence = self.calculate_confidence(base_confidence, 1 + confidence_boost)
        strength = self.get_strength_from_confidence(confidence)
        
        return {
            "side": signal_side,
            "confidence": confidence,
            "strength": strength,
            "reason": reason,
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "current_price": current_price,
                "hull_20": hull_20,
                "slope_approx": slope_approx,
                "slope_strength": slope_strength,
                "price_hull_distance_pct": price_hull_distance * 100,
                "price_above_hull": price_above_hull,
                "trend_angle": trend_angle,
                "trend_strength": trend_strength,
                "directional_bias": directional_bias,
                "trend_alignment": trend_alignment,
                "momentum_score": momentum_score,
                "rsi_14": rsi_14,
                "volume_ratio": volume_ratio,
                "market_regime": market_regime,
                "confluence_score": confluence_score
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs Hull MA requis sont présents."""
        if not super().validate_data():
            return False
            
        required = ['hull_20']
        
        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        # Vérifier aussi qu'on a des données de prix
        if not self.data or 'close' not in self.data or not self.data['close']:
            logger.warning(f"{self.name}: Données de prix manquantes")
            return False
            
        return True
