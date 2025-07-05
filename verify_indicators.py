import numpy as np
import pandas as pd
import talib
import sys
import os

# Ajouter le chemin du projet pour que l'import fonctionne
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from shared.src.technical_indicators import calculate_rsi, calculate_ema, calculate_bollinger_bands

def run_verification():
    """
    Exécute la vérification des calculs d'indicateurs.
    """
    # Jeu de données de test (prix de clôture sur 50 périodes)
    close_prices = np.array([
        100.0, 101.5, 102.3, 101.8, 103.2, 103.5, 104.1, 103.9, 105.0, 104.8,
        106.2, 105.7, 107.1, 106.8, 108.2, 107.9, 109.3, 108.7, 110.1, 109.5,
        111.0, 110.3, 111.8, 111.2, 112.7, 112.1, 113.6, 113.0, 114.5, 113.9,
        115.4, 114.8, 116.3, 115.7, 117.2, 116.6, 118.1, 117.5, 119.0, 118.4,
        120.0, 119.2, 118.5, 117.8, 116.9, 115.5, 114.2, 113.1, 112.5, 113.8
    ], dtype=float)

    print("--- Vérification des Indicateurs Techniques ---")

    # --- 1. Vérification du RSI (14 périodes) ---
    print("\n--- 1. RSI (14) ---")
    try:
        # Votre calcul
        your_rsi = calculate_rsi(close_prices, period=14)
        
        # Calcul TA-Lib
        talib_rsi = talib.RSI(close_prices, timeperiod=14)[-1]
        
        print(f"Votre calcul RSI: {your_rsi:.4f}")
        print(f"Calcul TA-Lib RSI: {talib_rsi:.4f}")
        if np.isclose(your_rsi, talib_rsi):
            print("✅ RSI: OK")
        else:
            print(f"❌ RSI: INCOHÉRENCE (Diff: {your_rsi - talib_rsi:.4f})")
    except Exception as e:
        print(f"Erreur lors de la vérification du RSI: {e}")

    # --- 2. Vérification de l'EMA (12 périodes) ---
    print("\n--- 2. EMA (12) ---")
    try:
        # Votre calcul
        your_ema = calculate_ema(close_prices, period=12)
        
        # Calcul TA-Lib
        talib_ema = talib.EMA(close_prices, timeperiod=12)[-1]
        
        print(f"Votre calcul EMA: {your_ema:.4f}")
        print(f"Calcul TA-Lib EMA: {talib_ema:.4f}")
        if np.isclose(your_ema, talib_ema):
            print("✅ EMA: OK")
        else:
            print(f"❌ EMA: INCOHÉRENCE (Diff: {your_ema - talib_ema:.4f})")
    except Exception as e:
        print(f"Erreur lors de la vérification de l'EMA: {e}")

    # --- 3. Vérification des Bandes de Bollinger (20 périodes, 2 std) ---
    print("\n--- 3. Bollinger Bands (20, 2) ---")
    try:
        # Votre calcul
        your_bb = calculate_bollinger_bands(close_prices, period=20, std_dev=2.0)
        
        # Calcul TA-Lib
        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        talib_bb = {
            'bb_upper': upper[-1],
            'bb_middle': middle[-1],
            'bb_lower': lower[-1]
        }
        
        print("Votre calcul BB:")
        print(f"  Upper:  {your_bb['bb_upper']:.4f}")
        print(f"  Middle: {your_bb['bb_middle']:.4f}")
        print(f"  Lower:  {your_bb['bb_lower']:.4f}")
        
        print("Calcul TA-Lib BB:")
        print(f"  Upper:  {talib_bb['bb_upper']:.4f}")
        print(f"  Middle: {talib_bb['bb_middle']:.4f}")
        print(f"  Lower:  {talib_bb['bb_lower']:.4f}")
        
        if (np.isclose(your_bb['bb_upper'], talib_bb['bb_upper']) and
            np.isclose(your_bb['bb_middle'], talib_bb['bb_middle']) and
            np.isclose(your_bb['bb_lower'], talib_bb['bb_lower'])):
            print("✅ Bollinger Bands: OK")
        else:
            print("❌ Bollinger Bands: INCOHÉRENCE")

    except Exception as e:
        print(f"Erreur lors de la vérification des Bandes de Bollinger: {e}")

if __name__ == "__main__":
    run_verification()
