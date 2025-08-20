"""
OBV_Crossover_Strategy - Stratégie basée sur les croisements OBV avec sa moyenne mobile.
OPTIMISÉE POUR ÉQUILIBRE RÉALISTE - VERSION SÉLECTIVE MAIS PRATICABLE
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class OBV_Crossover_Strategy(BaseStrategy):
    """
    Stratégie utilisant les croisements de l'On-Balance Volume (OBV) avec sa moyenne mobile.
    VERSION ÉQUILIBRÉE - Sélective mais pas excessive.
    
    L'OBV accumule le volume selon la direction du prix :
    - Volume ajouté si clôture > clôture précédente
    - Volume soustrait si clôture < clôture précédente
    - Volume neutre si clôture = clôture précédente
    
    Signaux générés:
    - BUY: OBV croise au-dessus de sa MA + confirmations haussières
    - SELL: OBV croise en-dessous de sa MA + confirmations baissières
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        
        # Paramètres OBV ÉQUILIBRÉS - Ni trop lâche ni trop strict
        self.min_obv_ma_distance = 0.018       # 1.8% distance minimum (raisonnable)
        self.strong_separation_threshold = 0.04 # 4% séparation forte 
        self.extreme_separation_threshold = 0.08 # 8% séparation extrême
        
        # Volume raisonnable
        self.min_volume_ratio = 1.3            # Volume 1.3x minimum (accessible)
        self.strong_volume_ratio = 2.0         # Volume 2x pour signal fort
        self.extreme_volume_ratio = 3.5        # Volume 3.5x pour signal exceptionnel
        
        # Confluence équilibrée
        self.min_confluence_score = 45         # 45 minimum (raisonnable)
        self.strong_confluence_score = 65      # 65 pour signal fort
        
        # Filtres équilibrés
        self.min_confidence_threshold = 0.48   # 48% confidence minimum (accessible)
        self.min_confirmations_required = 2    # 2 confirmations minimum (réaliste)
        
        # Filtres de marché équilibrés - autoriser plus de régimes
        self.blocked_market_regimes = ['VOLATILE', 'TRANSITION']  # Bloquer seulement les pires
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs OBV et volume."""
        return {
            # OBV et sa moyenne mobile
            'obv': self.indicators.get('obv'),
            'obv_ma_10': self.indicators.get('obv_ma_10'),
            'obv_oscillator': self.indicators.get('obv_oscillator'),
            'ad_line': self.indicators.get('ad_line'),
            # Volume pour confirmation
            'volume_ratio': self.indicators.get('volume_ratio'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            'relative_volume': self.indicators.get('relative_volume'),
            # Contexte prix pour confirmation tendance
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            # VWAP pour contexte
            'vwap_10': self.indicators.get('vwap_10'),
            # Momentum pour confluence
            'rsi_14': self.indicators.get('rsi_14'),
            'momentum_score': self.indicators.get('momentum_score'),
            # Structure de marché
            'market_regime': self.indicators.get('market_regime'),
            'support_levels': self.indicators.get('support_levels'),
            'resistance_levels': self.indicators.get('resistance_levels'),
            # Confluence
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score')
        }
        
    def _get_current_price(self) -> Optional[float]:
        """Récupère le prix actuel depuis les données OHLCV."""
        try:
            if self.data and 'close' in self.data and self.data['close']:
                return float(self.data['close'][-1])
        except (IndexError, ValueError, TypeError):
            pass
        return None
        
    def _count_confirmations(self, values: Dict[str, Any], signal_side: str) -> tuple:
        """Compte le nombre de confirmations pour le signal et retourne (count, details)."""
        confirmations = 0
        confirmation_details = []
        
        # 1. Trend Strength - moins strict
        trend_strength = values.get('trend_strength')
        if trend_strength and str(trend_strength).lower() in ['moderate', 'strong', 'very_strong', 'extreme']:
            confirmations += 1
            confirmation_details.append(f"tendance {trend_strength}")
            
        # 2. Directional Bias - bonus mais pas obligatoire
        directional_bias = values.get('directional_bias')
        if directional_bias:
            if (signal_side == "BUY" and directional_bias == 'BULLISH') or \
               (signal_side == "SELL" and directional_bias == 'BEARISH'):
                confirmations += 1
                confirmation_details.append(f"bias {directional_bias}")
                
        # 3. Volume Quality - seuil plus accessible
        volume_quality_score = values.get('volume_quality_score')
        if volume_quality_score is not None:
            try:
                vol_quality = float(volume_quality_score)
                if vol_quality >= 50:  # Seuil abaissé
                    confirmations += 1
                    confirmation_details.append(f"volume qualité {vol_quality:.0f}")
            except (ValueError, TypeError):
                pass
                
        # 4. OBV Oscillator - seuil moins strict
        obv_oscillator = values.get('obv_oscillator')
        if obv_oscillator is not None:
            try:
                obv_osc = float(obv_oscillator)
                if (signal_side == "BUY" and obv_osc > 0.01) or \
                   (signal_side == "SELL" and obv_osc < -0.01):  # Seuil abaissé
                    confirmations += 1
                    confirmation_details.append(f"OBV oscillator {obv_osc:.3f}")
            except (ValueError, TypeError):
                pass
                
        # 5. A/D Line alignment
        ad_line = values.get('ad_line')
        obv = values.get('obv')
        if ad_line is not None and obv is not None:
            try:
                ad_val = float(ad_line)
                obv_val = float(obv)
                ad_direction = 1 if ad_val > 0 else -1
                obv_direction = 1 if obv_val > 0 else -1
                
                if (signal_side == "BUY" and ad_direction == obv_direction == 1) or \
                   (signal_side == "SELL" and ad_direction == obv_direction == -1):
                    confirmations += 1
                    confirmation_details.append("A/D Line alignée")
            except (ValueError, TypeError):
                pass
                
        # 6. Price vs EMA alignment - plus tolérant
        ema_12 = values.get('ema_12')
        ema_26 = values.get('ema_26')
        current_price = self._get_current_price()
        if ema_12 is not None and ema_26 is not None and current_price is not None:
            try:
                ema12_val = float(ema_12)
                ema26_val = float(ema_26)
                # Plus tolérant - juste prix vs EMA12
                if (signal_side == "BUY" and current_price > ema12_val) or \
                   (signal_side == "SELL" and current_price < ema12_val):
                    confirmations += 1
                    confirmation_details.append("prix/EMA alignés")
            except (ValueError, TypeError):
                pass
                
        # 7. VWAP confirmation - moins strict
        vwap_10 = values.get('vwap_10')
        if vwap_10 is not None and current_price is not None:
            try:
                vwap_val = float(vwap_10)
                if (signal_side == "BUY" and current_price > vwap_val) or \
                   (signal_side == "SELL" and current_price < vwap_val):
                    confirmations += 1
                    confirmation_details.append("VWAP aligné")
            except (ValueError, TypeError):
                pass
                
        # 8. RSI in reasonable zone - plus large
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if signal_side == "BUY" and 25 <= rsi <= 70:  # Plus large
                    confirmations += 1
                    confirmation_details.append(f"RSI favorable ({rsi:.1f})")
                elif signal_side == "SELL" and 30 <= rsi <= 75:  # Plus large
                    confirmations += 1
                    confirmation_details.append(f"RSI favorable ({rsi:.1f})")
            except (ValueError, TypeError):
                pass
                
        return confirmations, confirmation_details
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les croisements OBV/MA avec filtres équilibrés.
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
        current_price = self._get_current_price()
        
        # === FILTRES PRÉLIMINAIRES ÉQUILIBRÉS ===
        
        # 1. Vérification des indicateurs OBV essentiels
        try:
            obv = float(values['obv']) if values['obv'] is not None else None
            obv_ma = float(values['obv_ma_10']) if values['obv_ma_10'] is not None else None
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion OBV: {e}",
                "metadata": {"strategy": self.name}
            }
            
        if obv is None or obv_ma is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "OBV ou OBV MA non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # 2. FILTRE: Market regime - bloquer seulement les pires
        market_regime = values.get('market_regime')
        if market_regime in self.blocked_market_regimes:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Marché {market_regime} défavorable pour OBV",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "market_regime": market_regime,
                    "filter": "market_regime"
                }
            }
            
        # 3. FILTRE: Volume minimum raisonnable
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is None or float(volume_ratio) < self.min_volume_ratio:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Volume insuffisant pour OBV ({volume_ratio:.1f}x < {self.min_volume_ratio}x)",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "volume_ratio": volume_ratio,
                    "filter": "volume"
                }
            }
            
        # 4. FILTRE: Confluence minimum raisonnable
        confluence_score = values.get('confluence_score')
        if confluence_score is None or float(confluence_score) < self.min_confluence_score:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Confluence insuffisante pour OBV ({confluence_score:.1f} < {self.min_confluence_score})",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "confluence_score": confluence_score,
                    "filter": "confluence"
                }
            }
            
        # 5. FILTRE: Distance OBV/MA raisonnable
        obv_distance = abs(obv - obv_ma) / abs(obv_ma) if obv_ma != 0 else 0
        if obv_distance < self.min_obv_ma_distance:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Séparation OBV/MA insuffisante ({obv_distance:.3f} < {self.min_obv_ma_distance})",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "obv": obv,
                    "obv_ma": obv_ma,
                    "distance": obv_distance,
                    "filter": "separation"
                }
            }
            
        # === DÉTERMINATION DU SIGNAL ===
        
        obv_above_ma = obv > obv_ma
        signal_side = None
        reason = ""
        base_confidence = 0.42  # Base équilibrée
        confidence_boost = 0.0
        
        # Déterminer direction du signal
        directional_bias = values.get('directional_bias')
        
        if obv_above_ma:
            # BUY - mais pénaliser si bias très défavorable
            if directional_bias == 'BEARISH':
                confidence_boost -= 0.15  # Pénalité mais pas rejet
                reason = f"OBV croise MA haussier ({obv:.0f} > {obv_ma:.0f}) MAIS bias baissier"
            else:
                reason = f"OBV croise MA haussier ({obv:.0f} > {obv_ma:.0f})"
            signal_side = "BUY"
            
        else:
            # SELL - mais pénaliser si bias très défavorable
            if directional_bias == 'BULLISH':
                confidence_boost -= 0.15  # Pénalité mais pas rejet
                reason = f"OBV croise MA baissier ({obv:.0f} < {obv_ma:.0f}) MAIS bias haussier"
            else:
                reason = f"OBV croise MA baissier ({obv:.0f} < {obv_ma:.0f})"
            signal_side = "SELL"
            
        # === COMPTAGE DES CONFIRMATIONS RÉALISTES ===
        
        confirmations_count, confirmation_details = self._count_confirmations(values, signal_side)
        
        if confirmations_count < self.min_confirmations_required:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal OBV {signal_side} - confirmations insuffisantes ({confirmations_count}/{self.min_confirmations_required})",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "confirmations_count": confirmations_count,
                    "confirmations_required": self.min_confirmations_required,
                    "confirmations_found": confirmation_details,
                    "filter": "confirmations"
                }
            }
            
        # === CALCUL DE LA CONFIANCE AVEC BONUS ÉQUILIBRÉS ===
        
        # Bonus selon force de la séparation
        if obv_distance >= self.extreme_separation_threshold:
            confidence_boost += 0.25
            reason += f" - séparation extrême ({obv_distance:.3f})"
        elif obv_distance >= self.strong_separation_threshold:
            confidence_boost += 0.18
            reason += f" - séparation forte ({obv_distance:.3f})"
        elif obv_distance >= 0.025:
            confidence_boost += 0.12
            reason += f" - séparation bonne ({obv_distance:.3f})"
        else:
            confidence_boost += 0.06
            reason += f" - séparation modérée ({obv_distance:.3f})"
            
        # Bonus volume
        vol_ratio = float(volume_ratio)
        if vol_ratio >= self.extreme_volume_ratio:
            confidence_boost += 0.22
            reason += f" + volume extrême ({vol_ratio:.1f}x)"
        elif vol_ratio >= self.strong_volume_ratio:
            confidence_boost += 0.15
            reason += f" + volume élevé ({vol_ratio:.1f}x)"
        elif vol_ratio >= self.min_volume_ratio:
            confidence_boost += 0.08
            reason += f" + volume correct ({vol_ratio:.1f}x)"
            
        # Bonus confluence
        conf_val = float(confluence_score)
        if conf_val >= 80:
            confidence_boost += 0.20
            reason += f" + confluence excellente ({conf_val:.0f})"
        elif conf_val >= self.strong_confluence_score:
            confidence_boost += 0.15
            reason += f" + confluence forte ({conf_val:.0f})"
        elif conf_val >= self.min_confluence_score:
            confidence_boost += 0.08
            reason += f" + confluence correcte ({conf_val:.0f})"
            
        # Bonus confirmations
        confirmation_bonus = min(confirmations_count * 0.06, 0.24)  # Max 24% pour 4+ confirmations
        confidence_boost += confirmation_bonus
        reason += f" + {confirmations_count} confirmations ({', '.join(confirmation_details[:2])})"
        
        # Bonus trend strength
        trend_strength = values.get('trend_strength')
        if trend_strength:
            trend_str = str(trend_strength).lower()
            if trend_str in ['extreme', 'very_strong']:
                confidence_boost += 0.15
                reason += f" + tendance {trend_str}"
            elif trend_str == 'strong':
                confidence_boost += 0.12
                reason += f" + tendance forte"
            elif trend_str == 'moderate':
                confidence_boost += 0.06
                reason += f" + tendance modérée"
                
        # Bonus signal strength
        signal_strength_calc = values.get('signal_strength')
        if signal_strength_calc:
            sig_str = str(signal_strength_calc).upper()
            if sig_str == 'STRONG':
                confidence_boost += 0.12
                reason += " + signal fort"
            elif sig_str == 'MODERATE':
                confidence_boost += 0.06
                reason += " + signal modéré"
                
        # Bonus market regime favorable
        if market_regime:
            regime_str = str(market_regime).upper()
            if (signal_side == "BUY" and regime_str in ["TRENDING_BULL", "BREAKOUT_BULL"]) or \
               (signal_side == "SELL" and regime_str in ["TRENDING_BEAR", "BREAKOUT_BEAR"]):
                confidence_boost += 0.10
                reason += f" + marché {regime_str.lower()}"
            elif regime_str == "RANGING":
                confidence_boost -= 0.05  # Légère pénalité en range
                reason += " (marché ranging)"
                
        # === FILTRE FINAL ÉQUILIBRÉ ===
        
        raw_confidence = base_confidence * (1 + confidence_boost)
        if raw_confidence < self.min_confidence_threshold:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal OBV {signal_side} - confidence finale insuffisante ({raw_confidence:.2f} < {self.min_confidence_threshold})",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "rejected_signal": signal_side,
                    "raw_confidence": raw_confidence,
                    "min_required": self.min_confidence_threshold,
                    "filter": "final_confidence"
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
                "obv": obv,
                "obv_ma_10": obv_ma,
                "obv_distance": obv_distance,
                "volume_ratio": volume_ratio,
                "confluence_score": confluence_score,
                "confirmations_count": confirmations_count,
                "confirmations_details": confirmation_details,
                "trend_strength": trend_strength,
                "directional_bias": directional_bias,
                "market_regime": market_regime
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs OBV requis sont présents."""
        if not super().validate_data():
            return False
            
        required = ['obv', 'obv_ma_10', 'volume_ratio', 'confluence_score']
        
        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        return True