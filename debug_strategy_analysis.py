#!/usr/bin/env python3
"""
Script de débogage pour analyser pourquoi les stratégies RSI et EMA Cross retournent False.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import logging
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Imports pour accéder aux données
from analyzer.src.indicators.db_indicators import DatabaseIndicators
from analyzer.strategies.rsi import RSIStrategy
from analyzer.strategies.ema_cross import EMACrossStrategy
from shared.src.config import SYMBOLS

async def analyze_strategy_conditions():
    """
    Analyse les conditions exactes des stratégies pour comprendre pourquoi elles retournent False.
    """
    print("=== ANALYSE DES CONDITIONS DES STRATÉGIES ===")
    
    # Initialiser le gestionnaire d'indicateurs DB
    db_indicators = DatabaseIndicators()
    
    # Analyser SOLUSDC
    symbol = "SOLUSDC"
    print(f"\n🔍 Analyse de {symbol}")
    
    try:
        # Récupérer les données depuis la DB
        df, indicators = await db_indicators.get_indicators_from_db(symbol)
        
        if df is None or len(df) == 0:
            print(f"❌ Pas de données disponibles pour {symbol}")
            return
        
        print(f"✅ Données récupérées: {len(df)} chandelles")
        print(f"✅ Indicateurs disponibles: {len(indicators)}")
        
        # Afficher les valeurs actuelles des indicateurs clés
        print(f"\n📊 VALEURS ACTUELLES DES INDICATEURS:")
        current_price = df['close'].iloc[-1]
        print(f"Prix actuel: {current_price:.4f}")
        
        # RSI
        rsi_14 = indicators.get('rsi_14')
        if rsi_14 is not None:
            if isinstance(rsi_14, (list, np.ndarray)):
                current_rsi = rsi_14[-1] if len(rsi_14) > 0 else None
            else:
                current_rsi = rsi_14
            print(f"RSI (14): {current_rsi:.2f}")
        else:
            print("RSI (14): Non disponible")
        
        # EMAs
        ema_12 = indicators.get('ema_12')
        ema_26 = indicators.get('ema_26')
        ema_50 = indicators.get('ema_50')
        
        if ema_12 is not None:
            if isinstance(ema_12, (list, np.ndarray)):
                current_ema_12 = ema_12[-1] if len(ema_12) > 0 else None
            else:
                current_ema_12 = ema_12
            print(f"EMA (12): {current_ema_12:.4f}")
        else:
            print("EMA (12): Non disponible")
            
        if ema_26 is not None:
            if isinstance(ema_26, (list, np.ndarray)):
                current_ema_26 = ema_26[-1] if len(ema_26) > 0 else None
            else:
                current_ema_26 = ema_26
            print(f"EMA (26): {current_ema_26:.4f}")
        else:
            print("EMA (26): Non disponible")
            
        if ema_50 is not None:
            if isinstance(ema_50, (list, np.ndarray)):
                current_ema_50 = ema_50[-1] if len(ema_50) > 0 else None
            else:
                current_ema_50 = ema_50
            print(f"EMA (50): {current_ema_50:.4f}")
        else:
            print("EMA (50): Non disponible")
        
        # MACD
        macd_line = indicators.get('macd_line')
        macd_signal = indicators.get('macd_signal')
        if macd_line is not None and macd_signal is not None:
            if isinstance(macd_line, (list, np.ndarray)):
                current_macd = macd_line[-1] if len(macd_line) > 0 else None
            else:
                current_macd = macd_line
            if isinstance(macd_signal, (list, np.ndarray)):
                current_macd_signal = macd_signal[-1] if len(macd_signal) > 0 else None
            else:
                current_macd_signal = macd_signal
            print(f"MACD Line: {current_macd:.6f}")
            print(f"MACD Signal: {current_macd_signal:.6f}")
        
        # ADX
        adx = indicators.get('adx')
        if adx is not None:
            if isinstance(adx, (list, np.ndarray)):
                current_adx = adx[-1] if len(adx) > 0 else None
            else:
                current_adx = adx
            print(f"ADX: {current_adx:.2f}")
        
        # Volume
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        print(f"Volume actuel: {current_volume:.0f}")
        print(f"Volume moyen (20): {avg_volume:.0f}")
        print(f"Ratio volume: {volume_ratio:.2f}")
        
        # Volatilité
        returns = df['close'].pct_change().dropna()
        volatility = returns.tail(20).std()
        print(f"Volatilité (20): {volatility:.4f}")
        
        # Maintenant, analysons pourquoi les stratégies retournent False
        print(f"\n🔍 ANALYSE DÉTAILLÉE DES CONDITIONS:")
        
        # === STRATÉGIE RSI ===
        print(f"\n📈 STRATÉGIE RSI:")
        rsi_strategy = RSIStrategy(symbol)
        
        # Conditions RSI
        print(f"Seuils RSI Ultra-Précis:")
        print(f"  - Zone extrême survente: <= {rsi_strategy.extreme_oversold}")
        print(f"  - Zone extrême surachat: >= {rsi_strategy.extreme_overbought}")
        print(f"  - RSI actuel: {current_rsi:.2f}")
        
        if current_rsi <= rsi_strategy.extreme_oversold:
            print(f"  ✅ RSI en zone d'achat (<=20)")
        elif current_rsi >= rsi_strategy.extreme_overbought:
            print(f"  ✅ RSI en zone de vente (>=80)")
        else:
            print(f"  ❌ RSI pas en zone extrême (entre 20 et 80)")
        
        # Filtres de qualité RSI
        print(f"Filtres de qualité RSI:")
        print(f"  - Volume minimum: {rsi_strategy.min_volume_ratio:.1f}x")
        print(f"  - Volume actuel: {volume_ratio:.2f}x")
        print(f"  - Volume OK: {'✅' if volume_ratio >= rsi_strategy.min_volume_ratio else '❌'}")
        
        print(f"  - Volatilité maximum: {rsi_strategy.max_volatility:.3f}")
        print(f"  - Volatilité actuelle: {volatility:.4f}")
        print(f"  - Volatilité OK: {'✅' if volatility <= rsi_strategy.max_volatility else '❌'}")
        
        print(f"  - Confiance minimum: {rsi_strategy.min_confidence:.2f}")
        
        # === STRATÉGIE EMA CROSS ===
        print(f"\n📈 STRATÉGIE EMA CROSS:")
        ema_strategy = EMACrossStrategy(symbol)
        
        # Conditions EMA Cross
        print(f"Conditions EMA Cross:")
        print(f"  - EMA 12: {current_ema_12:.4f}")
        print(f"  - EMA 26: {current_ema_26:.4f}")
        print(f"  - EMA 50: {current_ema_50:.4f}")
        print(f"  - Prix: {current_price:.4f}")
        
        # Alignement haussier
        bullish_alignment = (current_ema_12 > current_ema_26 > current_ema_50 and current_price > current_ema_12)
        bearish_alignment = (current_ema_12 < current_ema_26 < current_ema_50 and current_price < current_ema_12)
        
        print(f"  - Alignement haussier (EMA12 > EMA26 > EMA50 et Prix > EMA12): {'✅' if bullish_alignment else '❌'}")
        print(f"  - Alignement baissier (EMA12 < EMA26 < EMA50 et Prix < EMA12): {'✅' if bearish_alignment else '❌'}")
        
        # Croisement récent
        if len(ema_12) >= 3 and len(ema_26) >= 3:
            prev_ema_12 = ema_12[-2]
            prev_ema_26 = ema_26[-2]
            
            bullish_crossover = (prev_ema_12 <= prev_ema_26 and current_ema_12 > current_ema_26)
            bearish_crossover = (prev_ema_12 >= prev_ema_26 and current_ema_12 < current_ema_26)
            
            print(f"  - Croisement haussier récent: {'✅' if bullish_crossover else '❌'}")
            print(f"  - Croisement baissier récent: {'✅' if bearish_crossover else '❌'}")
        
        # Filtres de qualité EMA
        print(f"Filtres de qualité EMA:")
        print(f"  - Volume minimum: {ema_strategy.min_volume_confirmation:.1f}x")
        print(f"  - Volume actuel: {volume_ratio:.2f}x")
        print(f"  - Volume OK: {'✅' if volume_ratio >= ema_strategy.min_volume_confirmation else '❌'}")
        
        print(f"  - Volatilité maximum: {ema_strategy.max_volatility:.3f}")
        print(f"  - Volatilité actuelle: {volatility:.4f}")
        print(f"  - Volatilité OK: {'✅' if volatility <= ema_strategy.max_volatility else '❌'}")
        
        print(f"  - Confiance minimum: {ema_strategy.min_confidence:.2f}")
        
        # Momentum
        if current_rsi is not None:
            momentum_favorable = 40 <= current_rsi <= 60
            print(f"  - Momentum RSI favorable (40-60): {'✅' if momentum_favorable else '❌'}")
        
        # Tester les stratégies
        print(f"\n🧪 TEST DES STRATÉGIES:")
        
        # Test RSI
        rsi_result = rsi_strategy.analyze(symbol, df, indicators)
        print(f"RSI Strategy résultat: {rsi_result}")
        
        # Test EMA Cross
        ema_result = ema_strategy.analyze(symbol, df, indicators)
        print(f"EMA Cross Strategy résultat: {ema_result}")
        
        # Diagnostic final
        print(f"\n📋 RÉSUMÉ DES CONDITIONS:")
        print(f"Pour un signal RSI d'achat, il faut:")
        print(f"  1. RSI <= 20: {'✅' if current_rsi <= 20 else '❌'}")
        print(f"  2. Volume >= 1.4x: {'✅' if volume_ratio >= 1.4 else '❌'}")
        print(f"  3. Volatilité <= 6%: {'✅' if volatility <= 0.06 else '❌'}")
        print(f"  4. Divergence haussière: (nécessite analyse)")
        print(f"  5. Proche du support: (nécessite analyse)")
        
        print(f"\nPour un signal EMA Cross d'achat, il faut:")
        print(f"  1. Croisement haussier récent: (nécessite analyse)")
        print(f"  2. Alignement haussier: {'✅' if bullish_alignment else '❌'}")
        print(f"  3. Volume >= 1.4x: {'✅' if volume_ratio >= 1.4 else '❌'}")
        print(f"  4. Volatilité <= 8%: {'✅' if volatility <= 0.08 else '❌'}")
        print(f"  5. Momentum favorable: {'✅' if 40 <= current_rsi <= 60 else '❌'}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(analyze_strategy_conditions())