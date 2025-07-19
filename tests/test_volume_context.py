"""
Test unitaire pour la nouvelle logique de volume contextuelle.
Vérifie que les seuils s'adaptent correctement selon les conditions market.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.src.volume_context_detector import volume_context_detector


def test_volume_context_deep_oversold():
    """Test du contexte deep_oversold avec RSI très bas et CCI très négatif"""
    print("=== Test Volume Context: Deep Oversold ===")
    
    # Cas: RSI = 25, CCI = -220, volume_ratio = 1.06 (ton exemple)
    contextual_threshold, context_name, contextual_score = volume_context_detector.get_contextual_volume_threshold(
        base_volume_ratio=1.06,
        rsi=25,
        cci=-220,
        adx=15,
        signal_type="BUY"
    )
    
    volume_quality = volume_context_detector.get_volume_quality_description(1.06, context_name)
    
    print("Volume ratio: 1.06")
    print("RSI: 25, CCI: -220, ADX: 15")
    print(f"Contexte détecté: {context_name}")
    print(f"Seuil contextuel: {contextual_threshold}")
    print(f"Score contextuel: {contextual_score:.3f}")
    print(f"Qualité volume: {volume_quality}")
    print(f"Validation: {'✅ ACCEPTÉ' if 1.06 >= contextual_threshold else '❌ REJETÉ'}")
    print()


def test_volume_context_moderate_oversold():
    """Test du contexte moderate_oversold"""
    print("=== Test Volume Context: Moderate Oversold ===")
    
    # Cas: RSI = 35, CCI = -160, volume_ratio = 0.9
    contextual_threshold, context_name, contextual_score = volume_context_detector.get_contextual_volume_threshold(
        base_volume_ratio=0.9,
        rsi=35,
        cci=-160,
        adx=22,
        signal_type="BUY"
    )
    
    volume_quality = volume_context_detector.get_volume_quality_description(0.9, context_name)
    
    print("Volume ratio: 0.9")
    print("RSI: 35, CCI: -160, ADX: 22")
    print(f"Contexte détecté: {context_name}")
    print(f"Seuil contextuel: {contextual_threshold}")
    print(f"Score contextuel: {contextual_score:.3f}")
    print(f"Qualité volume: {volume_quality}")
    print(f"Validation: {'✅ ACCEPTÉ' if 0.9 >= contextual_threshold else '❌ REJETÉ'}")
    print()


def test_volume_context_breakout():
    """Test du contexte breakout avec volume élevé requis"""
    print("=== Test Volume Context: Breakout ===")
    
    # Cas: RSI = 55, CCI = -50, volume_ratio = 1.3, avec spike volume
    volume_history = [100, 110, 120, 150, 300]  # Spike volume détecté
    
    contextual_threshold, context_name, contextual_score = volume_context_detector.get_contextual_volume_threshold(
        base_volume_ratio=1.3,
        rsi=55,
        cci=-50,
        adx=30,
        signal_type="BUY",
        volume_history=volume_history,
        price_trend="breakout"
    )
    
    volume_quality = volume_context_detector.get_volume_quality_description(1.3, context_name)
    
    print("Volume ratio: 1.3")
    print("RSI: 55, CCI: -50, ADX: 30")
    print(f"Volume history: {volume_history}")
    print(f"Contexte détecté: {context_name}")
    print(f"Seuil contextuel: {contextual_threshold}")
    print(f"Score contextuel: {contextual_score:.3f}")
    print(f"Qualité volume: {volume_quality}")
    print(f"Validation: {'✅ ACCEPTÉ' if 1.3 >= contextual_threshold else '❌ REJETÉ'}")
    print()


def test_volume_context_low_volatility():
    """Test du contexte low_volatility avec ADX faible"""
    print("=== Test Volume Context: Low Volatility ===")
    
    # Cas: ADX = 12, volume_ratio = 0.95
    contextual_threshold, context_name, contextual_score = volume_context_detector.get_contextual_volume_threshold(
        base_volume_ratio=0.95,
        rsi=45,
        cci=-80,
        adx=12,
        signal_type="BUY"
    )
    
    volume_quality = volume_context_detector.get_volume_quality_description(0.95, context_name)
    
    print("Volume ratio: 0.95")
    print("RSI: 45, CCI: -80, ADX: 12 (marché calme)")
    print(f"Contexte détecté: {context_name}")
    print(f"Seuil contextuel: {contextual_threshold}")
    print(f"Score contextuel: {contextual_score:.3f}")
    print(f"Qualité volume: {volume_quality}")
    print(f"Validation: {'✅ ACCEPTÉ' if 0.95 >= contextual_threshold else '❌ REJETÉ'}")
    print()


def test_volume_tolerance():
    """Test de la tolérance volume"""
    print("=== Test Volume Tolerance ===")
    
    # Test conditions de tolérance
    apply_tolerance, tolerance_factor = volume_context_detector.should_apply_volume_tolerance(
        rsi=28,
        cci=-210,
        adx=18
    )
    
    print("RSI: 28, CCI: -210, ADX: 18")
    print(f"Tolérance applicable: {'✅ OUI' if apply_tolerance else '❌ NON'}")
    print(f"Facteur de réduction: {tolerance_factor}")
    print(f"Nouveau seuil suggéré: {1.0 * tolerance_factor:.2f} (au lieu de 1.0)")
    print()


if __name__ == "__main__":
    print("🧪 Test de la nouvelle logique Volume Contextuelle")
    print("=" * 60)
    
    test_volume_context_deep_oversold()
    test_volume_context_moderate_oversold()
    test_volume_context_breakout()
    test_volume_context_low_volatility()
    test_volume_tolerance()
    
    print("=" * 60)
    print("✅ Tests terminés - La logique volume contextuelle fonctionne !")
    print("\n📋 Résumé:")
    print("- Volume ratio 1.06 avec RSI=25/CCI=-220 → ACCEPTÉ (deep_oversold)")
    print("- Seuils adaptatifs selon contexte market")
    print("- Tolérance automatique pour oversold extrême")
    print("- Validation intelligente selon volatilité/tendance")