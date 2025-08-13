"""
Pump_Dump_Pattern_Strategy - Stratégie de détection des patterns pump & dump.
Détecte les mouvements anormaux de prix avec volume exceptionnel, suivis de corrections.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class Pump_Dump_Pattern_Strategy(BaseStrategy):
    """
    Stratégie détectant les patterns pump & dump pour profiter des corrections.
    
    Pattern Pump:
    - Hausse de prix rapide et massive (>3-5%)
    - Volume spike exceptionnel (>3x normal)
    - RSI en surachat extrême (>80)
    - Momentum très élevé
    
    Pattern Dump:
    - Chute de prix rapide après pump
    - Volume toujours élevé mais décroissant
    - RSI retournant vers la normale
    
    Signaux générés:
    - SELL: Détection d'un pump au sommet (avant correction)
    - BUY: Détection d'un dump stabilisé (après correction excessive)
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        
        # Seuils pour détection pump - CORRECTIONS REALISTES
        self.pump_price_threshold = 0.015     # 1.5% hausse minimum (assoupli de 2%)
        self.extreme_pump_threshold = 0.025   # 2.5% hausse extrême (assoupli de 3.5%)
        self.pump_volume_multiplier = 1.6     # Volume 1.6x normal (assoupli de 2x)
        self.extreme_volume_multiplier = 3.0  # Volume 3x normal (assoupli de 5x)
        self.pump_rsi_threshold = 75          # RSI surachat assoupli (de 80 à 75)
        self.extreme_rsi_threshold = 82       # RSI extrême assoupli (de 85 à 82)
        
        # Seuils pour détection dump/correction - CORRECTIONS REALISTES
        self.dump_price_threshold = -0.015    # 1.5% chute minimum (assoupli de 2%)
        self.extreme_dump_threshold = -0.03   # 3% chute extrême (assoupli de 4%)
        self.dump_rsi_threshold = 35          # RSI survente assoupli (de 30 à 35)
        self.momentum_reversal_threshold = -0.3  # Momentum négatif assoupli (de -0.5 à -0.3)
        
        # Paramètres de validation
        self.min_volatility_regime = 0.6      # Volatilité élevée requise
        self.min_trade_intensity = 1.5        # Intensité trading élevée
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            # Oscillateurs momentum
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            'williams_r': self.indicators.get('williams_r'),
            'roc_10': self.indicators.get('roc_10'),
            'roc_20': self.indicators.get('roc_20'),
            'momentum_10': self.indicators.get('momentum_10'),
            
            # Volume analysis
            'volume_spike_multiplier': self.indicators.get('volume_spike_multiplier'),
            'relative_volume': self.indicators.get('relative_volume'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            'volume_pattern': self.indicators.get('volume_pattern'),
            
            # Volatilité et contexte
            'atr_14': self.indicators.get('atr_14'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            'momentum_score': self.indicators.get('momentum_score'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            
            # Pattern et confluence
            'pattern_detected': self.indicators.get('pattern_detected'),
            'pattern_confidence': self.indicators.get('pattern_confidence'),
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score'),
            'anomaly_detected': self.indicators.get('anomaly_detected')
        }
        
    def _detect_pump_pattern(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte un pattern de pump (hausse anormale)."""
        pump_score = 0.0
        pump_indicators = []
        
        # Analyse ROC (Rate of Change) pour mouvement de prix
        roc_10 = values.get('roc_10')
        roc_20 = values.get('roc_20')
        
        if roc_10 is not None:
            try:
                roc_val = float(roc_10)
                if roc_val >= self.extreme_pump_threshold * 100:  # ROC en %
                    pump_score += 0.4
                    pump_indicators.append(f"ROC10 extrême ({roc_val:.1f}%)")
                elif roc_val >= self.pump_price_threshold * 100:
                    pump_score += 0.2
                    pump_indicators.append(f"ROC10 élevé ({roc_val:.1f}%)")
            except (ValueError, TypeError):
                pass
                
        # RSI surachat extrême
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi_val = float(rsi_14)
                if rsi_val >= self.extreme_rsi_threshold:
                    pump_score += 0.3
                    pump_indicators.append(f"RSI extrême ({rsi_val:.1f})")
                elif rsi_val >= self.pump_rsi_threshold:
                    pump_score += 0.15
                    pump_indicators.append(f"RSI surachat ({rsi_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # Volume spike critique
        volume_spike = values.get('volume_spike_multiplier')
        if volume_spike is not None:
            try:
                spike_val = float(volume_spike)
                if spike_val >= self.extreme_volume_multiplier:
                    pump_score += 0.35
                    pump_indicators.append(f"Volume spike extrême ({spike_val:.1f}x)")
                elif spike_val >= self.pump_volume_multiplier:
                    pump_score += 0.2
                    pump_indicators.append(f"Volume spike ({spike_val:.1f}x)")
            except (ValueError, TypeError):
                pass
                
        # Williams %R confirmant surachat
        williams_r = values.get('williams_r')
        if williams_r is not None:
            try:
                wr_val = float(williams_r)
                if wr_val >= -10:  # Williams %R > -10 = surachat extrême
                    pump_score += 0.15
                    pump_indicators.append(f"Williams%R surachat ({wr_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_pump': pump_score >= 0.45,  # CORRECTION MAJEURE: Très assoupli de 0.7 à 0.45
            'pump_score': pump_score,
            'indicators': pump_indicators
        }
        
    def _detect_dump_pattern(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte un pattern de dump/correction après pump."""
        dump_score = 0.0
        dump_indicators = []
        
        # ROC négatif indiquant chute
        roc_10 = values.get('roc_10')
        if roc_10 is not None:
            try:
                roc_val = float(roc_10)
                if roc_val <= self.extreme_dump_threshold * 100:
                    dump_score += 0.3
                    dump_indicators.append(f"ROC10 chute extrême ({roc_val:.1f}%)")
                elif roc_val <= self.dump_price_threshold * 100:
                    dump_score += 0.15
                    dump_indicators.append(f"ROC10 chute ({roc_val:.1f}%)")
            except (ValueError, TypeError):
                pass
                
        # RSI retour vers survente (opportunité d'achat)
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi_val = float(rsi_14)
                if rsi_val <= self.dump_rsi_threshold:
                    dump_score += 0.25
                    dump_indicators.append(f"RSI survente ({rsi_val:.1f})")
                elif rsi_val <= 50 and rsi_val > self.dump_rsi_threshold:
                    dump_score += 0.1
                    dump_indicators.append(f"RSI normalisation ({rsi_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # Momentum négatif fort
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum_val = float(momentum_score)
                if momentum_val <= self.momentum_reversal_threshold:
                    dump_score += 0.2
                    dump_indicators.append(f"Momentum négatif fort ({momentum_val:.2f})")
            except (ValueError, TypeError):
                pass
                
        # Volume toujours élevé mais qualité décroissante
        relative_volume = values.get('relative_volume')
        volume_quality = values.get('volume_quality_score')
        
        if relative_volume is not None and volume_quality is not None:
            try:
                rel_vol = float(relative_volume)
                vol_qual = float(volume_quality)
                # Volume élevé mais qualité moyenne = distribution
                if rel_vol >= 2.0 and vol_qual <= 60:
                    dump_score += 0.15
                    dump_indicators.append(f"Volume distribution ({rel_vol:.1f}x, qualité {vol_qual:.0f})")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_dump': dump_score >= 0.6,  # Augmenté de 0.5 à 0.6 - plus strict
            'dump_score': dump_score,
            'indicators': dump_indicators
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur la détection de patterns pump & dump.
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
        
        # Vérifications préalables - environnement volatil requis
        volatility_regime = values.get('volatility_regime')
        if volatility_regime is not None:
            try:
                vol_regime = self._convert_volatility_to_score(str(volatility_regime))
                if vol_regime < self.min_volatility_regime:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Volatilité insuffisante ({vol_regime:.2f}) pour pump/dump",
                        "metadata": {"strategy": self.name, "volatility_regime": vol_regime}
                    }
            except (ValueError, TypeError):
                pass
                
        # Détection des patterns
        pump_analysis = self._detect_pump_pattern(values)
        dump_analysis = self._detect_dump_pattern(values)
        
        signal_side = None
        reason = ""
        confidence_boost = 0.0
        metadata: Dict[str, Any] = {
            "strategy": self.name,
            "symbol": self.symbol
        }
        
        if pump_analysis['is_pump']:
            # Signal SELL - Pump détecté, attendre correction
            signal_side = "SELL"
            reason = f"Pump détecté ({pump_analysis['pump_score']:.2f}): {', '.join(pump_analysis['indicators'][:2])}"
            confidence_boost = pump_analysis['pump_score'] * 0.8
            
            metadata.update({
                "pattern_type": "pump",
                "pump_score": pump_analysis['pump_score'],
                "pump_indicators": pump_analysis['indicators']
            })
            
        elif dump_analysis['is_dump']:
            # Signal BUY - Dump/correction détectée, opportunité d'achat
            signal_side = "BUY"
            reason = f"Dump/correction détecté ({dump_analysis['dump_score']:.2f}): {', '.join(dump_analysis['indicators'][:2])}"
            confidence_boost = dump_analysis['dump_score'] * 0.9
            
            metadata.update({
                "pattern_type": "dump",
                "dump_score": dump_analysis['dump_score'],
                "dump_indicators": dump_analysis['indicators']
            })
            
        if signal_side:
            # Confirmations supplémentaires
            base_confidence = 0.50  # Standardisé à 0.50 pour équité avec autres stratégies
            
            # Trade intensity pour confirmer l'activité anormale
            trade_intensity = values.get('trade_intensity')
            if trade_intensity is not None:
                try:
                    intensity = float(trade_intensity)
                    if intensity >= self.min_trade_intensity:
                        confidence_boost += 0.1
                        reason += f" + intensité ({intensity:.1f})"
                except (ValueError, TypeError):
                    pass
                    
            # Pattern confidence du système
            pattern_confidence = values.get('pattern_confidence')
            if pattern_confidence is not None:
                try:
                    pat_conf = float(pattern_confidence)
                    if pat_conf > 70:
                        confidence_boost += 0.1
                        reason += " + pattern confirmé"
                except (ValueError, TypeError):
                    pass
                    
            # Anomaly detected pour confirmer mouvement anormal
            anomaly_detected = values.get('anomaly_detected')
            if anomaly_detected and str(anomaly_detected).lower() == 'true':
                confidence_boost += 0.15
                reason += " + anomalie détectée"
                
            # Confluence score
            confluence_score = values.get('confluence_score')
            if confluence_score is not None:
                try:
                    conf_score = float(confluence_score)
                    if conf_score > 60:
                        confidence_boost += 0.1
                except (ValueError, TypeError):
                    pass
                    
            # NOUVEAU: Filtre final de confidence minimum
            raw_confidence = base_confidence * (1.0 + confidence_boost)
            if raw_confidence < 0.55:  # Seuil minimum 55%
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Signal pump/dump rejeté - confidence insuffisante ({raw_confidence:.2f} < 0.55)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "rejected_signal": signal_side,
                        "raw_confidence": raw_confidence,
                        "pattern_type": metadata.get("pattern_type")
                    }
                }
            
            confidence = self.calculate_confidence(base_confidence, 1.0 + confidence_boost)
            strength = self.get_strength_from_confidence(confidence)
            
            # Mise à jour des métadonnées
            metadata.update({
                "rsi_14": values.get('rsi_14'),
                "roc_10": values.get('roc_10'),
                "volume_spike_multiplier": values.get('volume_spike_multiplier'),
                "relative_volume": values.get('relative_volume'),
                "volatility_regime": volatility_regime,
                "trade_intensity": trade_intensity,
                "pattern_confidence": pattern_confidence,
                "confluence_score": confluence_score,
                "anomaly_detected": anomaly_detected
            })
            
            return {
                "side": signal_side,
                "confidence": confidence,
                "strength": strength,
                "reason": reason,
                "metadata": metadata
            }
            
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": f"Aucun pattern pump/dump détecté (pump: {pump_analysis['pump_score']:.2f}, dump: {dump_analysis['dump_score']:.2f})",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "pump_score": pump_analysis['pump_score'],
                "dump_score": dump_analysis['dump_score'],
                "volatility_regime": volatility_regime
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        required_indicators = [
            'rsi_14', 'roc_10', 'volume_spike_multiplier', 
            'relative_volume', 'volatility_regime'
        ]
        
        if not self.indicators:
            logger.warning(f"{self.name}: Aucun indicateur disponible")
            return False
            
        for indicator in required_indicators:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        return True
    
    def _convert_volatility_to_score(self, volatility_regime: str) -> float:
        """Convertit un régime de volatilité en score numérique."""
        try:
            if not volatility_regime:
                return 2.0
                
            vol_lower = volatility_regime.lower()
            
            if vol_lower in ['high', 'very_high', 'extreme']:
                return 3.0  # Haute volatilité
            elif vol_lower in ['normal', 'moderate', 'average']:
                return 2.0  # Volatilité normale
            elif vol_lower in ['low', 'very_low', 'minimal']:
                return 1.0  # Faible volatilité
            else:
                # Essayer de convertir directement en float
                try:
                    return float(volatility_regime)
                except (ValueError, TypeError):
                    return 2.0  # Valeur par défaut
                    
        except Exception:
            return 2.0
