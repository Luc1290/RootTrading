"""
HullMA_Slope_Strategy - Stratégie CONTRARIAN utilisant Hull MA comme filtre de qualité.
TRANSFORMATION D'UNE STRATÉGIE PERDANTE EN STRATÉGIE PROFITABLE
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class HullMA_Slope_Strategy(BaseStrategy):
    """
    Stratégie CONTRARIAN avec Hull MA comme filtre de qualité et timing.
    
    NOUVELLE APPROCHE - Stratégie anti-lagging:
    - Hull MA sert de FILTRE de tendance (pas de signal direct)
    - Achète sur les PULLBACKS dans une tendance haussière Hull MA
    - Vend sur les BOUNCES dans une tendance baissière Hull MA
    - Focus sur les RETOURNEMENTS et CORRECTIONS plutôt que suivre
    
    Signaux générés:
    - BUY: Prix SOUS Hull MA haussière + oscillateurs survente + momentum retournement
    - SELL: Prix AU-DESSUS Hull MA baissière + oscillateurs surachat + momentum retournement
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        
        # PARAMÈTRES CONTRARIAN OPTIMISÉS WINRATE
        self.hull_trend_threshold = 0.002      # 0.20% pente minimum (plus sélectif)
        self.price_pullback_min = 0.008        # 0.8% pullback minimum (zone optimale)
        self.price_pullback_max = 0.030        # 3.0% pullback maximum (réduit overextension)
        self.price_bounce_min = 0.008          # 0.8% bounce minimum (zone optimale)
        self.price_bounce_max = 0.030          # 3.0% bounce maximum (réduit overextension)
        
        # Seuils oscillateurs STRICTS pour WINRATE
        self.rsi_oversold_entry = 32           # RSI survente vraie zone (plus strict)
        self.rsi_overbought_entry = 68         # RSI surachat vraie zone (plus strict)
        self.stoch_oversold_entry = 25         # Stoch survente extrême
        self.stoch_overbought_entry = 75       # Stoch surachat extrême
        
        # Filtres qualité RÉALISTES pour CRYPTO
        self.min_volume_ratio = 0.8            # Volume minimum ACCESSIBLE (réduction -33%)
        self.min_confluence_score = 35         # Confluence minimum RÉALISTE (réduction -30%) 
        self.min_confidence_threshold = 0.45   # Confidence minimum ACCESSIBLE (réduction -10%)
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs."""
        return {
            # Hull MA principal
            'hull_20': self.indicators.get('hull_20'),
            # Moyennes mobiles pour contexte
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            'sma_20': self.indicators.get('sma_20'),
            # Trend analysis CRITIQUE pour nouvelle approche
            'trend_angle': self.indicators.get('trend_angle'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            # Oscillateurs pour contrarian entries
            'momentum_score': self.indicators.get('momentum_score'),
            'rsi_14': self.indicators.get('rsi_14'),
            'macd_line': self.indicators.get('macd_line'),
            'macd_histogram': self.indicators.get('macd_histogram'),
            # Volume critique
            'volume_ratio': self.indicators.get('volume_ratio'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            # Contexte marché
            'market_regime': self.indicators.get('market_regime'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            # Confluence finale
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score')
        }
        
    def _get_price_data(self) -> Dict[str, Optional[float]]:
        """Récupère les données de prix pour analyse."""
        try:
            if self.data and 'close' in self.data and self.data['close'] and len(self.data['close']) >= 5:
                prices = self.data['close']
                return {
                    'current_price': float(prices[-1]),
                    'prev_price_1': float(prices[-2]),
                    'prev_price_2': float(prices[-3]),
                    'prev_price_3': float(prices[-4]),
                    'prev_price_4': float(prices[-5])
                }
        except (IndexError, ValueError, TypeError):
            pass
        return {
            'current_price': None, 'prev_price_1': None, 'prev_price_2': None,
            'prev_price_3': None, 'prev_price_4': None
        }
        
    def _analyze_hull_trend_direction(self, values: Dict[str, Any], price_data: Dict[str, Optional[float]]) -> Dict[str, Any]:
        """Analyse la direction de tendance de Hull MA avec trend_angle comme source principale."""
        hull_20 = values.get('hull_20')
        trend_angle = values.get('trend_angle')
        current_price = price_data['current_price']
        
        if hull_20 is None or current_price is None:
            return {'direction': None, 'strength': 'unknown', 'reliable': False}
            
        try:
            hull_val = float(hull_20)
            
            # Méthode 1: Utiliser trend_angle si disponible (le plus fiable)
            if trend_angle is not None:
                try:
                    angle = float(trend_angle)
                    # Convertir angle en pente normalisée
                    angle_threshold_deg = 1.0  # 1 degré minimum (était 2.0)
                    
                    if angle >= angle_threshold_deg:
                        return {
                            'direction': 'bullish',
                            'strength': 'strong' if angle >= 5.0 else 'moderate',
                            'reliable': True,
                            'slope_proxy': angle / 45.0  # Normaliser
                        }
                    elif angle <= -angle_threshold_deg:
                        return {
                            'direction': 'bearish', 
                            'strength': 'strong' if angle <= -5.0 else 'moderate',
                            'reliable': True,
                            'slope_proxy': angle / 45.0
                        }
                    else:
                        return {
                            'direction': 'sideways',
                            'strength': 'flat',
                            'reliable': True,
                            'slope_proxy': angle / 45.0
                        }
                except (ValueError, TypeError):
                    pass
                    
            # Méthode 2: Fallback avec prix relatif et directional_bias
            directional_bias = values.get('directional_bias')
            trend_strength = values.get('trend_strength')
            
            # Distance prix/Hull MA comme indicateur secondaire
            price_hull_ratio = current_price / hull_val
            
            if directional_bias == 'BULLISH' and trend_strength in ['weak', 'moderate', 'strong', 'very_strong', 'extreme']:
                return {
                    'direction': 'bullish',
                    'strength': str(trend_strength).lower() if trend_strength else 'moderate',
                    'reliable': trend_strength in ['moderate', 'strong', 'very_strong', 'extreme'],
                    'slope_proxy': min((price_hull_ratio - 1.0) * 10, 0.1)  # Approximation
                }
            elif directional_bias == 'BEARISH' and trend_strength in ['weak', 'moderate', 'strong', 'very_strong', 'extreme']:
                return {
                    'direction': 'bearish',
                    'strength': str(trend_strength).lower() if trend_strength else 'moderate',
                    'reliable': trend_strength in ['moderate', 'strong', 'very_strong', 'extreme'],
                    'slope_proxy': max((price_hull_ratio - 1.0) * 10, -0.1)  # Approximation
                }
            else:
                return {
                    'direction': 'sideways',
                    'strength': 'weak',
                    'reliable': False,
                    'slope_proxy': 0.0
                }
                
        except (ValueError, TypeError):
            return {'direction': None, 'strength': 'unknown', 'reliable': False}
            
    def _detect_pullback_opportunity(self, hull_trend: Dict[str, Any], hull_20: float, 
                                   current_price: float, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte une opportunité de pullback (prix temporairement contre la tendance Hull)."""
        
        if hull_trend['direction'] != 'bullish' or not hull_trend['reliable']:
            return {'is_pullback': False, 'reason': 'Pas de tendance haussière Hull fiable'}
            
        # Prix doit être SOUS Hull MA (pullback dans tendance haussière)
        if current_price >= hull_20:
            return {'is_pullback': False, 'reason': f'Prix au-dessus Hull MA ({current_price:.2f} >= {hull_20:.2f})'}
            
        # Calculer l'amplitude du pullback
        pullback_pct = (hull_20 - current_price) / hull_20
        
        if pullback_pct < self.price_pullback_min:
            return {'is_pullback': False, 'reason': f'Pullback trop faible ({pullback_pct*100:.1f}% < {self.price_pullback_min*100:.1f}%)'}
            
        if pullback_pct > self.price_pullback_max:
            return {'is_pullback': False, 'reason': f'Pullback trop important ({pullback_pct*100:.1f}% > {self.price_pullback_max*100:.1f}%) - falling knife'}
            
        # Vérifier oscillateurs survente pour confirmation
        rsi_14 = values.get('rsi_14')
        momentum_score = values.get('momentum_score')
        
        oversold_signals = 0
        oversold_details = []
        
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if rsi <= self.rsi_oversold_entry:
                    oversold_signals += 1
                    oversold_details.append(f"RSI survente ({rsi:.1f})")
            except (ValueError, TypeError):
                pass
                
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                if momentum <= 40:  # Momentum VRAIMENT faible pour BUY
                    oversold_signals += 1
                    oversold_details.append(f"momentum très faible ({momentum:.0f})")
            except (ValueError, TypeError):
                pass
                
        # MACD histogram pour retournement momentum
        macd_histogram = values.get('macd_histogram')
        if macd_histogram is not None:
            try:
                hist = float(macd_histogram)
                # Chercher un retournement (MACD qui redevient positif)
                if hist > 0.00005:  # Légèrement positif (seuil assoupli)
                    oversold_signals += 1
                    oversold_details.append(f"MACD retournement (+{hist:.4f})")
            except (ValueError, TypeError):
                pass
                
        # CORRECTION: Assouplir de 2 à 1 confirmation minimum
        if oversold_signals < 1:  # Au moins 1 confirmation (était 2)
            return {
                'is_pullback': False, 
                'reason': f'Pullback détecté mais aucune confirmation ({oversold_signals}/1)',
                'pullback_pct': pullback_pct,
                'oversold_signals': oversold_signals
            }
            
        return {
            'is_pullback': True,
            'pullback_pct': pullback_pct,
            'oversold_signals': oversold_signals,
            'oversold_details': oversold_details,
            'reason': f'Pullback {pullback_pct*100:.1f}% avec {oversold_signals} confirmations'
        }
        
    def _detect_bounce_opportunity(self, hull_trend: Dict[str, Any], hull_20: float,
                                 current_price: float, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte une opportunité de bounce (prix temporairement contre la tendance Hull)."""
        
        if hull_trend['direction'] != 'bearish' or not hull_trend['reliable']:
            return {'is_bounce': False, 'reason': 'Pas de tendance baissière Hull fiable'}
            
        # Prix doit être AU-DESSUS Hull MA (bounce dans tendance baissière)
        if current_price <= hull_20:
            return {'is_bounce': False, 'reason': f'Prix sous Hull MA ({current_price:.2f} <= {hull_20:.2f})'}
            
        # Calculer l'amplitude du bounce
        bounce_pct = (current_price - hull_20) / hull_20
        
        if bounce_pct < self.price_bounce_min:
            return {'is_bounce': False, 'reason': f'Bounce trop faible ({bounce_pct*100:.1f}% < {self.price_bounce_min*100:.1f}%)'}
            
        if bounce_pct > self.price_bounce_max:
            return {'is_bounce': False, 'reason': f'Bounce trop important ({bounce_pct*100:.1f}% > {self.price_bounce_max*100:.1f}%) - dead cat bounce'}
            
        # Vérifier oscillateurs surachat pour confirmation
        rsi_14 = values.get('rsi_14')
        momentum_score = values.get('momentum_score')
        
        overbought_signals = 0
        overbought_details = []
        
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if rsi >= self.rsi_overbought_entry:
                    overbought_signals += 1
                    overbought_details.append(f"RSI surachat ({rsi:.1f})")
            except (ValueError, TypeError):
                pass
                
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                if momentum >= 60:  # Momentum VRAIMENT élevé pour SELL
                    overbought_signals += 1
                    overbought_details.append(f"momentum très élevé ({momentum:.0f})")
            except (ValueError, TypeError):
                pass
                
        # MACD histogram pour retournement momentum
        macd_histogram = values.get('macd_histogram')
        if macd_histogram is not None:
            try:
                hist = float(macd_histogram)
                # Chercher un retournement (MACD qui redevient négatif)
                if hist < -0.00005:  # Légèrement négatif (seuil assoupli)
                    overbought_signals += 1
                    overbought_details.append(f"MACD retournement ({hist:.4f})")
            except (ValueError, TypeError):
                pass
                
        # CORRECTION: Assouplir de 2 à 1 confirmation minimum
        if overbought_signals < 1:  # Au moins 1 confirmation (était 2)
            return {
                'is_bounce': False,
                'reason': f'Bounce détecté mais aucune confirmation ({overbought_signals}/1)',
                'bounce_pct': bounce_pct,
                'overbought_signals': overbought_signals
            }
            
        return {
            'is_bounce': True,
            'bounce_pct': bounce_pct,
            'overbought_signals': overbought_signals,
            'overbought_details': overbought_details,
            'reason': f'Bounce {bounce_pct*100:.1f}% avec {overbought_signals} confirmations'
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal CONTRARIAN basé sur Hull MA comme filtre de tendance.
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
        
        # === FILTRES PRÉLIMINAIRES ===
        
        # Vérification Hull MA et prix
        hull_20 = values.get('hull_20')
        current_price = price_data['current_price']
        
        if hull_20 is None or current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Hull MA ou prix non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        try:
            hull_val = float(hull_20)
            price_val = float(current_price)
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion Hull MA/prix: {e}",
                "metadata": {"strategy": self.name}
            }
            
        # CORRECTION: Volume comme bonus au lieu de filtre obligatoire
        volume_ratio = values.get('volume_ratio')
        volume_penalty = 0.0
        if volume_ratio is None:
            volume_penalty = -0.05  # Légère pénalité si pas de données volume
        else:
            try:
                vol_val = float(volume_ratio)
                if vol_val < 0.5:  # Volume très faible (était rejet)
                    volume_penalty = -0.10  # Pénalité au lieu de rejet
            except (ValueError, TypeError):
                volume_penalty = -0.05
            
        # CORRECTION: Confluence comme bonus au lieu de filtre strict
        confluence_score = values.get('confluence_score')
        confluence_penalty = 0.0
        if confluence_score is None:
            confluence_penalty = -0.08  # Pénalité si pas de données confluence
        else:
            try:
                conf_val = float(confluence_score)
                if conf_val < 25:  # Confluence très faible (était rejet à 50)
                    confluence_penalty = -0.15  # Forte pénalité au lieu de rejet
                elif conf_val < self.min_confluence_score:  # 25-35
                    confluence_penalty = -0.08  # Pénalité modérée
            except (ValueError, TypeError):
                confluence_penalty = -0.08
            
        # Pénaliser (mais ne pas bannir) marchés très volatiles
        volatility_regime = values.get('volatility_regime')
        volatility_penalty = 0.0
        if volatility_regime == 'extreme':
            volatility_penalty = -0.10  # Réduction confidence au lieu de rejet total
            
        # === ANALYSE TENDANCE HULL MA ===
        
        hull_trend = self._analyze_hull_trend_direction(values, price_data)
        
        if hull_trend['direction'] is None or not hull_trend.get('reliable', False):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Direction tendance Hull MA non fiable",
                "metadata": {
                    "strategy": self.name,
                    "hull_trend": hull_trend
                }
            }
            
        # === DÉTECTION OPPORTUNITÉS CONTRARIAN ===
        
        signal_side = None
        reason = ""
        opportunity_data = {}
        base_confidence = 0.45  # Base RELEVÉE pour meilleur winrate (+50%)
        confidence_boost = 0.0
        
        if hull_trend['direction'] == 'bullish':
            # Chercher PULLBACK dans tendance haussière
            pullback_analysis = self._detect_pullback_opportunity(hull_trend, hull_val, price_val, values)
            
            if pullback_analysis['is_pullback']:
                signal_side = "BUY"
                reason = f"CONTRARIAN BUY: {pullback_analysis['reason']} en tendance Hull haussière"
                opportunity_data = pullback_analysis
                confidence_boost += 0.35  # Bonus AUGMENTÉ pour vrais setups (+40%)
            else:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak", 
                    "reason": f"Hull haussière mais pas de pullback valide: {pullback_analysis['reason']}",
                    "metadata": {
                        "strategy": self.name,
                        "hull_trend": hull_trend,
                        "pullback_analysis": pullback_analysis
                    }
                }
                
        elif hull_trend['direction'] == 'bearish':
            # Chercher BOUNCE dans tendance baissière
            bounce_analysis = self._detect_bounce_opportunity(hull_trend, hull_val, price_val, values)
            
            if bounce_analysis['is_bounce']:
                signal_side = "SELL"
                reason = f"CONTRARIAN SELL: {bounce_analysis['reason']} en tendance Hull baissière"
                opportunity_data = bounce_analysis
                confidence_boost += 0.35  # Bonus AUGMENTÉ pour vrais setups (+40%)
            else:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Hull baissière mais pas de bounce valide: {bounce_analysis['reason']}",
                    "metadata": {
                        "strategy": self.name,
                        "hull_trend": hull_trend,
                        "bounce_analysis": bounce_analysis
                    }
                }
                
        else:  # sideways
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Hull MA en tendance latérale - pas de setup contrarian",
                "metadata": {
                    "strategy": self.name,
                    "hull_trend": hull_trend
                }
            }
            
        # === BONUS DE CONFIANCE ===
        
        # Bonus force tendance Hull (AJUSTÉ)
        if hull_trend['strength'] == 'strong':
            confidence_boost += 0.15
            reason += " + tendance Hull FORTE"
        elif hull_trend['strength'] == 'moderate':
            confidence_boost += 0.10  # Était 0.08
            reason += " + tendance Hull modérée"
        elif hull_trend['strength'] == 'weak':
            confidence_boost += 0.05  # Nouveau bonus pour weak
            reason += " + tendance Hull faible"
            
        # Bonus nombre de confirmations oscillateurs (AJUSTÉ)
        if signal_side == "BUY":
            oversold_count = opportunity_data.get('oversold_signals', 0)
            if oversold_count >= 3:
                confidence_boost += 0.20  # Augmenté
                reason += f" + {oversold_count} confirmations survente"
            elif oversold_count >= 2:
                confidence_boost += 0.15  # Augmenté
                reason += f" + {oversold_count} confirmations"
            elif oversold_count >= 1:
                confidence_boost += 0.08  # Nouveau bonus 1 confirmation
                reason += f" + {oversold_count} confirmation"
                
        elif signal_side == "SELL":
            overbought_count = opportunity_data.get('overbought_signals', 0)
            if overbought_count >= 3:
                confidence_boost += 0.20  # Augmenté
                reason += f" + {overbought_count} confirmations surachat"
            elif overbought_count >= 2:
                confidence_boost += 0.15  # Augmenté
                reason += f" + {overbought_count} confirmations"
            elif overbought_count >= 1:
                confidence_boost += 0.08  # Nouveau bonus 1 confirmation
                reason += f" + {overbought_count} confirmation"
                
        # Bonus alignement EMA pour contexte
        ema_12 = values.get('ema_12')
        ema_26 = values.get('ema_26')
        if ema_12 is not None and ema_26 is not None:
            try:
                ema12_val = float(ema_12)
                ema26_val = float(ema_26)
                
                if signal_side == "BUY" and ema12_val > ema26_val:
                    confidence_boost += 0.08
                    reason += " + EMA12>26"
                elif signal_side == "SELL" and ema12_val < ema26_val:
                    confidence_boost += 0.08
                    reason += " + EMA12<26"
            except (ValueError, TypeError):
                pass
                
        # Bonus volume élevé (volume_ratio peut être None maintenant)
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= 2.0:
                    confidence_boost += 0.15
                    reason += f" + volume très élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.5:
                    confidence_boost += 0.10
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= self.min_volume_ratio:
                    confidence_boost += 0.05
                    reason += f" + volume correct ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
            
        # Bonus confluence (confluence_score peut être None maintenant)
        if confluence_score is not None:
            try:
                conf_val = float(confluence_score)
                if conf_val >= 70:
                    confidence_boost += 0.18
                    reason += f" + confluence excellente ({conf_val:.0f})"
                elif conf_val >= 60:
                    confidence_boost += 0.12
                    reason += f" + confluence forte ({conf_val:.0f})"
                elif conf_val >= self.min_confluence_score:
                    confidence_boost += 0.06
                    reason += f" + confluence ({conf_val:.0f})"
            except (ValueError, TypeError):
                pass
            
        # Bonus signal strength
        signal_strength = values.get('signal_strength')
        if signal_strength == 'STRONG':
            confidence_boost += 0.12
            reason += " + signal fort"
        elif signal_strength == 'MODERATE':
            confidence_boost += 0.06
            reason += " + signal modéré"
            
        # Bonus trend alignment
        trend_alignment = values.get('trend_alignment')
        if trend_alignment is not None:
            try:
                alignment = float(trend_alignment)
                if abs(alignment) >= 0.3:
                    confidence_boost += 0.10
                    reason += " + MA alignées"
            except (ValueError, TypeError):
                pass
                
        # Bonus market regime favorable
        market_regime = values.get('market_regime')
        if market_regime in ['TRENDING_BULL', 'TRENDING_BEAR']:
            confidence_boost += 0.08
            reason += f" + marché {market_regime.lower()}"
            
        # === FILTRE FINAL ===
        
        # CORRECTION: Appliquer toutes les pénalités
        total_penalty = volatility_penalty + volume_penalty + confluence_penalty
        confidence_boost += total_penalty
        
        # CORRECTION: Calcul direct au lieu de multiplicateur
        final_confidence = base_confidence + confidence_boost
        raw_confidence = min(max(final_confidence, 0.0), 1.0)  # Clamp 0-1
        
        if raw_confidence < self.min_confidence_threshold:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal CONTRARIAN {signal_side} rejeté - confidence insuffisante ({raw_confidence:.2f} < {self.min_confidence_threshold})",
                "metadata": {
                    "strategy": self.name,
                    "rejected_signal": signal_side,
                    "raw_confidence": raw_confidence,
                    "hull_trend": hull_trend,
                    "opportunity_data": opportunity_data,
                    "penalties_applied": {
                        "volume": volume_penalty,
                        "confluence": confluence_penalty,
                        "volatility": volatility_penalty
                    }
                }
            }
            
        confidence = raw_confidence  # Utiliser directement la confidence calculée
        strength = self.get_strength_from_confidence(confidence)
        
        return {
            "side": signal_side,
            "confidence": confidence,
            "strength": strength,
            "reason": reason,
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "approach": "CONTRARIAN",
                "current_price": price_val,
                "hull_20": hull_val,
                "hull_trend": hull_trend,
                "opportunity_data": opportunity_data,
                "volume_ratio": volume_ratio,
                "confluence_score": confluence_score,
                "market_regime": market_regime,
                "volatility_regime": volatility_regime
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs Hull MA requis sont présents."""
        if not super().validate_data():
            return False
            
        required = ['hull_20', 'volume_ratio', 'confluence_score']
        
        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        # Vérifier données de prix suffisantes
        if not self.data or 'close' not in self.data or not self.data['close'] or len(self.data['close']) < 3:
            logger.warning(f"{self.name}: Données de prix insuffisantes")
            return False
            
        return True