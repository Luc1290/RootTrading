#!/usr/bin/env python3
"""
Module de protection contre les crashes et chutes brutales du marché.
Fournit des mécanismes défensifs pour toutes les stratégies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CrashProtection:
    """
    Système de protection défensive contre les chutes brutales du marché.
    """
    
    def __init__(self):
        # Seuils de protection
        self.CRASH_THRESHOLD_5M = -0.05  # -5% en 5 minutes = crash
        self.RAPID_DROP_THRESHOLD = -0.02  # -2% en 1 minute = chute rapide
        self.VOLUME_PANIC_MULTIPLIER = 3.0  # Volume 3x supérieur = panique
        self.RSI_COLLAPSE_THRESHOLD = 25   # RSI < 25 = effondrement momentum
        self.ATR_EXTREME_MULTIPLIER = 2.5  # ATR 2.5x normal = volatilité extrême
        
        # Stop-loss d'urgence
        self.EMERGENCY_STOP_LOSS = -0.08  # -8% = stop-loss d'urgence
        self.TRAILING_STOP_LOSS = -0.05   # -5% = trailing stop serré
        
        logger.info("🛡️ CrashProtection initialisé")
    
    def analyze_crash_conditions(self, df: pd.DataFrame, current_price: float, 
                                entry_price: Optional[float] = None) -> Dict:
        """
        Analyse complète des conditions de crash/chute.
        
        Returns:
            Dict avec conditions détectées et niveaux de priorité
        """
        try:
            if df is None or len(df) < 10:
                return {"crash_detected": False, "conditions": []}
            
            conditions = []
            priority_level = 0
            
            # 1. DÉTECTION CHUTE RAPIDE
            crash_data = self._detect_rapid_price_drop(df)
            if crash_data["is_crash"]:
                conditions.append(crash_data)
                priority_level = max(priority_level, crash_data["priority"])
            
            # 2. DÉTECTION VOLUME DE PANIQUE  
            panic_data = self._detect_panic_volume(df)
            if panic_data["is_panic"]:
                conditions.append(panic_data)
                priority_level = max(priority_level, panic_data["priority"])
            
            # 3. DÉTECTION MOMENTUM COLLAPSE
            momentum_data = self._detect_momentum_collapse(df)
            if momentum_data["is_collapse"]:
                conditions.append(momentum_data)
                priority_level = max(priority_level, momentum_data["priority"])
            
            # 4. DÉTECTION VOLATILITÉ EXTRÊME
            volatility_data = self._detect_extreme_volatility(df)
            if volatility_data["is_extreme"]:
                conditions.append(volatility_data)
                priority_level = max(priority_level, volatility_data["priority"])
            
            # 5. STOP-LOSS D'URGENCE (si position ouverte)
            if entry_price:
                stop_data = self._check_emergency_stop_loss(current_price, entry_price)
                if stop_data["emergency_stop"]:
                    conditions.append(stop_data)
                    priority_level = max(priority_level, stop_data["priority"])
            
            # RÉSULTAT FINAL
            crash_detected = len(conditions) > 0
            if crash_detected:
                logger.warning(f"🚨 PROTECTION DÉFENSIVE: {len(conditions)} conditions détectées (priorité {priority_level})")
                for condition in conditions:
                    logger.warning(f"   - {condition['type']}: {condition['description']}")
            
            return {
                "crash_detected": crash_detected,
                "conditions": conditions,
                "priority": priority_level,
                "emergency_sell_recommended": priority_level >= 3
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse crash protection: {e}")
            return {"crash_detected": False, "conditions": []}
    
    def _detect_rapid_price_drop(self, df: pd.DataFrame) -> Dict:
        """Détecte les chutes rapides de prix."""
        try:
            if len(df) < 5:
                return {"is_crash": False}
            
            # Chute sur 5 minutes (5 bougies)
            price_5m_ago = df['close'].iloc[-5]
            current_price = df['close'].iloc[-1]
            change_5m = (current_price - price_5m_ago) / price_5m_ago
            
            # Chute sur 1 minute (1 bougie)
            price_1m_ago = df['close'].iloc[-2] if len(df) >= 2 else current_price
            change_1m = (current_price - price_1m_ago) / price_1m_ago
            
            # Analyser la sévérité
            if change_5m <= self.CRASH_THRESHOLD_5M:  # -5% en 5min
                return {
                    "is_crash": True,
                    "type": "CRASH_RAPIDE",
                    "description": f"Chute {change_5m:.1%} en 5min",
                    "severity": "CRITIQUE",
                    "priority": 4,
                    "change_5m": change_5m,
                    "change_1m": change_1m
                }
            elif change_1m <= self.RAPID_DROP_THRESHOLD:  # -2% en 1min
                return {
                    "is_crash": True,
                    "type": "CHUTE_RAPIDE", 
                    "description": f"Chute {change_1m:.1%} en 1min",
                    "severity": "ÉLEVÉ",
                    "priority": 3,
                    "change_5m": change_5m,
                    "change_1m": change_1m
                }
            
            return {"is_crash": False}
            
        except Exception as e:
            logger.error(f"Erreur détection chute rapide: {e}")
            return {"is_crash": False}
    
    def _detect_panic_volume(self, df: pd.DataFrame) -> Dict:
        """Détecte les volumes de panique."""
        try:
            if len(df) < 20 or 'volume' not in df.columns:
                return {"is_panic": False}
            
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[-20:].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Prix en baisse + volume élevé = panique
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            
            if volume_ratio >= self.VOLUME_PANIC_MULTIPLIER and price_change < -0.005:  # Volume 3x + baisse
                return {
                    "is_panic": True,
                    "type": "VOLUME_PANIQUE",
                    "description": f"Volume {volume_ratio:.1f}x + baisse {price_change:.1%}",
                    "severity": "ÉLEVÉ",
                    "priority": 3,
                    "volume_ratio": volume_ratio,
                    "price_change": price_change
                }
            
            return {"is_panic": False}
            
        except Exception as e:
            logger.error(f"Erreur détection volume panique: {e}")
            return {"is_panic": False}
    
    def _detect_momentum_collapse(self, df: pd.DataFrame) -> Dict:
        """Détecte l'effondrement du momentum."""
        try:
            if len(df) < 14:
                return {"is_collapse": False}
            
            # RSI actuel
            rsi_current = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else 50
            
            # RSI précédent pour voir la chute
            rsi_previous = df['rsi_14'].iloc[-5] if len(df) >= 5 else rsi_current
            rsi_drop = rsi_previous - rsi_current
            
            # Momentum MACD (si disponible)
            macd_negative = False
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd = df['macd'].iloc[-1]
                macd_signal = df['macd_signal'].iloc[-1]
                macd_negative = macd < macd_signal and macd < 0
            
            # Conditions d'effondrement
            if rsi_current < self.RSI_COLLAPSE_THRESHOLD:  # RSI < 25
                return {
                    "is_collapse": True,
                    "type": "MOMENTUM_COLLAPSE",
                    "description": f"RSI effondré {rsi_current:.0f} (chute {rsi_drop:.0f})",
                    "severity": "ÉLEVÉ",
                    "priority": 3,
                    "rsi_current": rsi_current,
                    "rsi_drop": rsi_drop
                }
            elif rsi_drop > 20:  # RSI a chuté de 20+ points
                return {
                    "is_collapse": True,
                    "type": "RSI_CHUTE",
                    "description": f"RSI chute brutale -{rsi_drop:.0f} points",
                    "severity": "MODÉRÉ",
                    "priority": 2,
                    "rsi_current": rsi_current,
                    "rsi_drop": rsi_drop
                }
            
            return {"is_collapse": False}
            
        except Exception as e:
            logger.error(f"Erreur détection momentum collapse: {e}")
            return {"is_collapse": False}
    
    def _detect_extreme_volatility(self, df: pd.DataFrame) -> Dict:
        """Détecte la volatilité extrême."""
        try:
            if len(df) < 14 or 'atr_14' not in df.columns:
                return {"is_extreme": False}
            
            current_atr = df['atr_14'].iloc[-1]
            avg_atr = df['atr_14'].iloc[-14:].mean()
            atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1
            
            current_price = df['close'].iloc[-1]
            atr_percent = (current_atr / current_price * 100) if current_price > 0 else 0
            
            if atr_ratio >= self.ATR_EXTREME_MULTIPLIER:  # ATR 2.5x normal
                return {
                    "is_extreme": True,
                    "type": "VOLATILITÉ_EXTRÊME",
                    "description": f"ATR {atr_ratio:.1f}x normal ({atr_percent:.1f}%)",
                    "severity": "MODÉRÉ",
                    "priority": 2,
                    "atr_ratio": atr_ratio,
                    "atr_percent": atr_percent
                }
            
            return {"is_extreme": False}
            
        except Exception as e:
            logger.error(f"Erreur détection volatilité extrême: {e}")
            return {"is_extreme": False}
    
    def _check_emergency_stop_loss(self, current_price: float, entry_price: float) -> Dict:
        """Vérifie les conditions de stop-loss d'urgence."""
        try:
            if not entry_price or entry_price <= 0:
                return {"emergency_stop": False}
            
            loss_percent = (current_price - entry_price) / entry_price
            
            if loss_percent <= self.EMERGENCY_STOP_LOSS:  # -8% = stop d'urgence
                return {
                    "emergency_stop": True,
                    "type": "STOP_URGENCE",
                    "description": f"Perte {loss_percent:.1%} atteint stop d'urgence",
                    "severity": "CRITIQUE",
                    "priority": 4,
                    "loss_percent": loss_percent,
                    "entry_price": entry_price,
                    "current_price": current_price
                }
            elif loss_percent <= self.TRAILING_STOP_LOSS:  # -5% = trailing stop
                return {
                    "emergency_stop": True,
                    "type": "TRAILING_STOP",
                    "description": f"Perte {loss_percent:.1%} atteint trailing stop",
                    "severity": "ÉLEVÉ", 
                    "priority": 3,
                    "loss_percent": loss_percent,
                    "entry_price": entry_price,
                    "current_price": current_price
                }
            
            return {"emergency_stop": False}
            
        except Exception as e:
            logger.error(f"Erreur vérification stop-loss: {e}")
            return {"emergency_stop": False}
    
    def should_override_technical_signals(self, crash_analysis: Dict) -> bool:
        """
        Détermine si les conditions de crash doivent surpasser les signaux techniques.
        """
        if not crash_analysis.get("crash_detected", False):
            return False
        
        # Priorité 4 = court-circuiter TOUT (urgence absolue)
        # Priorité 3 = court-circuiter les signaux techniques normaux
        return crash_analysis.get("priority", 0) >= 3
    
    def get_defensive_sell_signal(self, symbol: str, crash_analysis: Dict) -> Optional[Dict]:
        """
        Génère un signal de vente défensive basé sur l'analyse de crash.
        """
        if not crash_analysis.get("crash_detected", False):
            return None
        
        # Construire le signal défensif
        conditions = crash_analysis.get("conditions", [])
        priority = crash_analysis.get("priority", 1)
        
        signal = {
            "symbol": symbol,
            "side": "SELL",
            "strategy": "CRASH_PROTECTION",
            "signal_type": "DEFENSIVE",
            "timestamp": datetime.now().isoformat(),
            "strength": min(priority / 4.0, 1.0),  # Force basée sur priorité
            "confidence": 0.95,  # Haute confiance pour signaux défensifs
            "reason": "Protection défensive activée",
            "metadata": {
                "crash_protection": True,
                "priority": priority,
                "conditions": [c.get("type", "UNKNOWN") for c in conditions],
                "emergency": priority >= 4,
                "details": conditions
            }
        }
        
        logger.warning(f"🛡️ Signal défensif généré pour {symbol}: priorité {priority}")
        return signal