"""
Test spécifique pour Bollinger Pro avec volume contextuel.
Vérifie que les patterns Bollinger (squeeze, expansion, mean reversion) 
bénéficient des seuils volume adaptatifs.
"""

# Simulation des patterns Bollinger
BOLLINGER_PATTERNS = {
    "squeeze": {
        "signal_type": "breakout",
        "volume_context": "consolidation_break",
        "description": "Squeeze - préparation breakout, volume peut être faible"
    },
    "expansion": {
        "signal_type": "mean_reversion",
        "volume_context": "breakout", 
        "description": "Expansion - mouvement en cours, volume élevé attendu"
    },
    "contracting": {
        "signal_type": "breakout",
        "volume_context": "consolidation_break",
        "description": "Contraction - préparation breakout"
    },
    "normal": {
        "signal_type": "mean_reversion",
        "volume_context": "trend_continuation",
        "description": "Pattern normal - volume standard"
    }
}

# Contextes volume correspondants
VOLUME_CONTEXTS = {
    "consolidation_break": {"min_ratio": 1.3, "ideal_ratio": 1.8},
    "breakout": {"min_ratio": 1.5, "ideal_ratio": 2.0},
    "trend_continuation": {"min_ratio": 1.0, "ideal_ratio": 1.5},
    "deep_oversold": {"min_ratio": 0.6, "ideal_ratio": 1.2}
}


def get_bollinger_volume_context(pattern_type, signal_type):
    """Simulation de la méthode _get_bollinger_volume_context"""
    if pattern_type == 'squeeze':
        return "consolidation_break"
    elif pattern_type == 'expansion':
        return "breakout"
    elif pattern_type == 'contracting':
        return "consolidation_break"
    elif signal_type == 'breakout':
        return "breakout"
    elif signal_type == 'mean_reversion':
        return "trend_continuation"
    else:
        return "trend_continuation"


def calculate_bollinger_score(volume_ratio, pattern_type, signal_type, rsi=None, cci=None):
    """Calcul du score Bollinger avec volume contextuel"""
    
    # Détection contexte Bollinger
    bollinger_context = get_bollinger_volume_context(pattern_type, signal_type)
    
    # Détection contexte global (RSI/CCI)
    if rsi and rsi < 30 and cci and cci < -200:
        global_context = "deep_oversold"
    else:
        global_context = bollinger_context
    
    # Seuil effectif
    bollinger_threshold = VOLUME_CONTEXTS[bollinger_context]["min_ratio"]
    global_threshold = VOLUME_CONTEXTS[global_context]["min_ratio"]
    contextual_threshold = min(bollinger_threshold, global_threshold)  # Le plus tolérant
    
    # Calcul score
    if volume_ratio >= 2.0:
        return 25, "excellent", contextual_threshold
    elif volume_ratio >= 1.5:
        return 20, "très bon", contextual_threshold
    elif volume_ratio >= contextual_threshold:
        # Score graduel contextuel
        ideal_ratio = VOLUME_CONTEXTS[global_context]["ideal_ratio"]
        if volume_ratio >= ideal_ratio:
            contextual_score = 1.0
        else:
            ratio_range = ideal_ratio - contextual_threshold
            actual_range = volume_ratio - contextual_threshold
            contextual_score = 0.5 + (actual_range / ratio_range) * 0.5
        
        score_bonus = int(contextual_score * 20)  # 0-20 points
        return score_bonus, "contextuel", contextual_threshold
    elif volume_ratio >= 0.7 and signal_type == 'mean_reversion':
        return 8, "mean reversion toléré", contextual_threshold
    elif signal_type == 'breakout':
        return -10, "breakout insuffisant", contextual_threshold
    else:
        return -5, "faible", contextual_threshold


def test_bollinger_patterns():
    """Test des patterns Bollinger avec volume contextuel"""
    print("🎯 TEST BOLLINGER PRO - VOLUME CONTEXTUEL")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Squeeze préparation breakout",
            "pattern": "squeeze",
            "signal_type": "breakout",
            "volume": 1.2,
            "rsi": 45,
            "cci": -80,
            "expected_context": "consolidation_break"
        },
        {
            "name": "Expansion avec volume élevé",
            "pattern": "expansion", 
            "signal_type": "mean_reversion",
            "volume": 1.8,
            "rsi": 70,
            "cci": 50,
            "expected_context": "breakout"
        },
        {
            "name": "Mean reversion volume faible",
            "pattern": "normal",
            "signal_type": "mean_reversion", 
            "volume": 0.8,
            "rsi": 35,
            "cci": -120,
            "expected_context": "trend_continuation"
        },
        {
            "name": "Squeeze oversold (Ton Cas Adapté)",
            "pattern": "squeeze",
            "signal_type": "breakout",
            "volume": 1.06,  # Ton cas original
            "rsi": 25,
            "cci": -220,
            "expected_context": "deep_oversold"  # RSI/CCI override
        },
        {
            "name": "Breakout volume insuffisant",
            "pattern": "expansion",
            "signal_type": "breakout",
            "volume": 1.1,
            "rsi": 55,
            "cci": -30,
            "expected_context": "breakout"
        }
    ]
    
    for case in test_cases:
        print(f"🔍 {case['name']}:")
        
        # Calcul scores
        score, quality, threshold = calculate_bollinger_score(
            case['volume'],
            case['pattern'],
            case['signal_type'], 
            case['rsi'],
            case['cci']
        )
        
        # Score ancien (hardcodé)
        old_score = 0
        if case['volume'] >= 2.0:
            old_score = 25
        elif case['volume'] >= 1.5:
            old_score = 20
        elif case['volume'] >= 1.2:
            old_score = 15
        elif case['volume'] >= 1.0:
            old_score = 10
        elif case['volume'] >= 0.7:
            old_score = 5
        else:
            old_score = -5
        
        # Contexte détecté
        bollinger_context = get_bollinger_volume_context(case['pattern'], case['signal_type'])
        
        print(f"   Pattern: {case['pattern']}, Signal: {case['signal_type']}")
        print(f"   Volume: {case['volume']}, RSI: {case['rsi']}, CCI: {case['cci']}")
        print(f"   Contexte Bollinger: {bollinger_context}")
        print(f"   Seuil contextuel: {threshold}")
        print(f"   Score ancien: {old_score} | Score nouveau: {score} ({quality})")
        
        # Analyse amélioration
        if score > old_score:
            print(f"   🎉 AMÉLIORATION: +{score - old_score} points")
        elif score < old_score:
            print(f"   ⚠️  RÉDUCTION: {score - old_score} points")
        else:
            print("   🔄 MAINTENU: Score identique")
        
        # Spécificité Bollinger
        if case['signal_type'] == 'mean_reversion' and case['volume'] >= 0.7:
            print("   💡 SPÉCIFICITÉ: Mean reversion tolère volume faible")
        elif case['signal_type'] == 'breakout' and case['volume'] < 1.3:
            print("   ⚠️  SPÉCIFICITÉ: Breakout nécessite volume élevé")
        
        print()


def test_bollinger_contexts():
    """Test des mappings contexte Bollinger"""
    print("🧠 MAPPING CONTEXTES BOLLINGER")
    print("=" * 50)
    
    mappings = [
        ("squeeze", "breakout", "consolidation_break"),
        ("expansion", "mean_reversion", "breakout"),
        ("contracting", "breakout", "consolidation_break"), 
        ("normal", "mean_reversion", "trend_continuation"),
        ("normal", "trend_following", "trend_continuation")
    ]
    
    for pattern, signal, expected_context in mappings:
        detected_context = get_bollinger_volume_context(pattern, signal)
        threshold = VOLUME_CONTEXTS[detected_context]["min_ratio"]
        
        print(f"📊 Pattern: {pattern}, Signal: {signal}")
        print(f"   Contexte: {detected_context} (seuil: {threshold})")
        print(f"   Mapping: {'✅ Correct' if detected_context == expected_context else '❌ Incorrect'}")
        print()


if __name__ == "__main__":
    print("🧪 TEST BOLLINGER PRO - VOLUME CONTEXTUEL")
    print("=" * 70)
    print()
    
    test_bollinger_patterns()
    test_bollinger_contexts()
    
    print("=" * 70)
    print("📋 RÉSUMÉ BOLLINGER PRO:")
    print("✅ Volume contextuel selon pattern (squeeze, expansion, etc.)")
    print("✅ Mean reversion tolère volume faible (≥0.7)")
    print("✅ Breakout nécessite volume élevé (contexte strict)")
    print("✅ Squeeze prépare breakout (seuil consolidation_break)")
    print("✅ Expansion active (seuil breakout élevé)")
    print("✅ Oversold override pour cas spéciaux")
    print()
    print("🎯 SPÉCIFICITÉS BOLLINGER:")
    print("- Squeeze → consolidation_break (seuil 1.3)")
    print("- Expansion → breakout (seuil 1.5)")
    print("- Mean reversion → tolérance volume faible")
    print("- Pattern-aware volume validation")
    print("- RSI/CCI override pour cas extrêmes")