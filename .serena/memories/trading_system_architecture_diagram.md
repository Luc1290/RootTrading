# 🏗️ Architecture Détaillée du Système de Trading

## Diagramme de Dépendances

```
┌─────────────────────────────────────────────────────────────┐
│         OpportunityCalculatorPro (Orchestrateur)            │
│                                                             │
│  __init__(enable_early_detection=True)                     │
│    ├─→ self.scorer = OpportunityScoring()                  │
│    ├─→ self.validator = OpportunityValidator()             │
│    └─→ self.early_detector = OpportunityEarlyDetector()    │
└─────────────────────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌───────────────────┐
│  Scoring     │ │  Validator   │ │ Early Detector    │
│              │ │              │ │                   │
│ Score 0-100  │ │ Data Quality │ │ Signal précoce    │
│ 7 catégories │ │ 3 niveaux    │ │ 4 composants      │
└──────────────┘ └──────────────┘ └───────────────────┘
```

## Flux de Données Complet

```
analyzer_data (dict avec 75 indicateurs)
    │
    ├─→ OpportunityScoring.calculate_opportunity_score()
    │       │
    │       ├─→ _score_vwap_position() → 25%
    │       ├─→ _score_ema_trend() → 18%
    │       ├─→ _score_volume_scalping() → 20% (🆕 OBV direction)
    │       ├─→ _score_rsi_scalping() → 12%
    │       ├─→ _score_bollinger() → 10%
    │       ├─→ _score_macd() → 5%
    │       └─→ _score_sr_simple() → 10% (🆕 10% vs 5%)
    │       │
    │       └─→ OpportunityScore (total_score, grade, category_scores)
    │
    ├─→ OpportunityValidator.validate_opportunity()
    │       │
    │       ├─→ _validate_data_quality() [BLOQUANT]
    │       │   - data_quality: EXCELLENT/GOOD
    │       │   - Indicateurs critiques présents
    │       │
    │       ├─→ _validate_indicator_coherence() [INFORMATIF] 🆕
    │       │   - Pullbacks VWAP/EMA tolérés
    │       │   - RSI/MACD cohérence
    │       │   - Volume/Bollinger cohérence
    │       │
    │       └─→ _validate_risk_parameters() [INFORMATIF]
    │           - ATR disponible
    │           - R/R calculable
    │           - Résistance JAMAIS bloquante
    │       │
    │       └─→ ValidationSummary (all_passed, warnings)
    │
    └─→ OpportunityEarlyDetector.detect_early_opportunity() [OPTIONNEL]
            │
            ├─→ _score_velocity_acceleration() → 35 pts (INVERSÉ)
            ├─→ _score_volume_buildup() → 30 pts (🆕 warnings contextualisés)
            ├─→ _score_micro_patterns() → 20 pts (🆕 RSI >70 contextualisé)
            └─→ _score_order_flow() → 15 pts
            │
            └─→ EarlySignal (level, score, timing)

            ▼
OpportunityCalculatorPro.calculate_opportunity()
            │
            ├─→ _make_decision(score, validation, early_signal)
            │   - Score 70+ → BUY_NOW
            │   - Score 60-70 → BUY_DCA
            │   - Score 50-60 → WAIT
            │   - Score <50 → AVOID
            │
            ├─→ _calculate_entry_prices()
            │   - entry_optimal (limit order)
            │   - entry_aggressive (market order)
            │
            ├─→ _calculate_targets() 🆕 ADAPTATIF
            │   - Score 75+ → (0.8, 1.3, 1.8) ATR
            │   - Score 60-75 → (0.7, 1.1, 1.5) ATR
            │   - Score <60 → (0.6, 0.9, None) ATR
            │
            ├─→ _calculate_stop_loss()
            │   - support - 0.3 ATR ou current - 0.8 ATR
            │
            ├─→ _calculate_risk_metrics()
            │   - R/R ratio, risk level, max position
            │
            └─→ _calculate_timing()
                - hold time, urgency
            │
            ▼
    TradingOpportunity (dataclass complet)
            │
            └─→ to_dict() → JSON pour API
```

## Interfaces de Données

### analyzer_data (Entrée du système)
```python
{
    # VWAP (25%)
    "vwap_10": float,
    "vwap_quote_10": float,
    
    # EMA (18%)
    "ema_7": float,
    "ema_12": float,
    "ema_26": float,
    "adx_14": float,
    
    # Volume (20%)
    "relative_volume": float,
    "obv_oscillator": float,  # 🆕 v4.1
    "volume_spike_multiplier": float,
    
    # RSI (12%)
    "rsi_14": float,
    
    # Bollinger (10%)
    "bb_upper": float,
    "bb_lower": float,
    "bb_squeeze": bool,
    "bb_expansion": bool,
    "bb_position": float,
    
    # MACD (5%)
    "macd_trend": str,  # BULLISH/NEUTRAL/BEARISH
    "macd_histogram": float,
    
    # S/R (10%)
    "nearest_support": float,
    "nearest_resistance": float,
    "resistance_strength": float,  # 🆕 v4.1
    "support_strength": float,     # 🆕 v4.1
    "break_probability": float,    # 🆕 v4.1
    
    # Autres essentiels
    "atr_14": float,
    "current_price": float,
    "data_quality": str,  # EXCELLENT/GOOD
    "market_regime": str,
    "volatility_regime": str,
    "volume_context": str,
    
    # Early Detection (optionnel)
    "roc_10": float,
    "volume_buildup_periods": int,
    "trade_intensity": float,
    "quote_volume_ratio": float,
}
```

### OpportunityScore (Sortie scoring)
```python
@dataclass
class OpportunityScore:
    total_score: float  # 0-100
    grade: str  # S, A, B, C, D, F
    category_scores: dict[ScoreCategory, CategoryScore]
    confidence: float  # 0-100
    risk_level: str  # LOW, MEDIUM, HIGH, EXTREME
    recommendation: str  # BUY_NOW, BUY_DCA, WAIT, AVOID
    reasons: list[str]
    warnings: list[str]
```

### ValidationSummary (Sortie validation)
```python
@dataclass
class ValidationSummary:
    all_passed: bool  # True si DATA_QUALITY OK
    level_results: dict[ValidationLevel, ValidationResult]
    overall_score: float  # 0-100
    blocking_issues: list[str]  # Vide si all_passed
    warnings: list[str]  # Informatifs
    recommendations: list[str]
```

### EarlySignal (Sortie early detector)
```python
@dataclass
class EarlySignal:
    level: EarlySignalLevel  # ENTRY_NOW, PREPARE, WATCH, NONE, TOO_LATE
    score: float  # 0-100
    confidence: float  # 0-100
    velocity_score: float
    volume_buildup_score: float
    micro_pattern_score: float
    order_flow_score: float
    estimated_entry_window_seconds: int
    estimated_move_completion_pct: float
    reasons: list[str]
    warnings: list[str]
    recommendations: list[str]
```

### TradingOpportunity (Sortie finale)
```python
@dataclass
class TradingOpportunity:
    # Décision
    symbol: str
    action: str  # BUY_NOW, BUY_DCA, WAIT, AVOID
    confidence: float  # 0-100
    
    # Scores & Validation
    score: OpportunityScore
    validation: ValidationSummary
    early_signal: EarlySignal | None
    is_early_entry: bool
    
    # Pricing
    current_price: float
    entry_price_optimal: float
    entry_price_aggressive: float
    
    # Targets (adaptatifs selon score)
    tp1: float
    tp1_percent: float
    tp2: float
    tp2_percent: float
    tp3: float | None
    tp3_percent: float | None
    
    # Stop Loss
    stop_loss: float
    stop_loss_percent: float
    stop_loss_basis: str  # "support" ou "ATR"
    
    # Risk Management
    rr_ratio: float
    risk_level: str
    max_position_size_pct: float
    
    # Timing
    estimated_hold_time: str
    entry_urgency: str
    
    # Context
    market_regime: str
    volume_context: str
    volatility_regime: str
    
    # Messages
    reasons: list[str]
    warnings: list[str]
    recommendations: list[str]
    
    # Raw data
    raw_score_details: dict
    raw_validation_details: dict
    raw_analyzer_data: dict
```

## Points de Décision Critiques

### 1. Validation Bloquante
```python
if not validation.all_passed:
    # SEUL cas de rejet: DATA_QUALITY insuffisante
    return AVOID
```

### 2. Décision basée Score
```python
if score.total_score >= 70:
    action = "BUY_NOW"
    confidence = min(95.0, score.total_score)
elif score.total_score >= 60:
    action = "BUY_DCA"
    confidence = min(85.0, score.total_score * 1.1)
elif score.total_score >= 50:
    action = "WAIT"
    confidence = score.total_score * 0.8
else:
    action = "AVOID"
    confidence = score.total_score * 0.5
```

### 3. Boost Early Signal
```python
if is_early_entry and early_signal:
    if early_signal.level == ENTRY_NOW:
        confidence_boost = min(10.0, early_signal.confidence * 0.15)
    elif early_signal.level == PREPARE:
        confidence_boost = min(5.0, early_signal.confidence * 0.1)
```

### 4. Targets Adaptatifs 🆕 v4.1
```python
if score.total_score >= 75:
    # Setup FORT
    tp1_mult, tp2_mult, tp3_mult = 0.8, 1.3, 1.8
elif score.total_score >= 60:
    # Setup BON
    tp1_mult, tp2_mult, tp3_mult = 0.7, 1.1, 1.5
else:
    # Setup MOYEN/FAIBLE
    tp1_mult, tp2_mult, tp3_mult = 0.6, 0.9, None
```

## Interdépendances des Fichiers

```
opportunity_calculator_pro.py (Orchestrateur)
    imports:
        - OpportunityScoring, OpportunityScore from opportunity_scoring
        - OpportunityValidator, ValidationSummary from opportunity_validator
        - OpportunityEarlyDetector, EarlySignal, EarlySignalLevel from opportunity_early_detector
    
    NO dependencies: opportunity_scoring.py, opportunity_validator.py, opportunity_early_detector.py
    (Modules indépendants, réutilisables)
```

## Complexité et Maintenance

### Métriques
- **opportunity_scoring.py**: 750 lignes, 15 méthodes, complexité moyenne
- **opportunity_validator.py**: 546 lignes, 7 méthodes, complexité faible
- **opportunity_early_detector.py**: 759 lignes, 11 méthodes, complexité moyenne
- **opportunity_calculator_pro.py**: 715 lignes, 11 méthodes, complexité moyenne-élevée

### Points d'Extension
1. **Nouveaux indicateurs**: Ajouter méthode `_score_new_indicator()` dans OpportunityScoring
2. **Nouveaux niveaux validation**: Ajouter dans `ValidationLevel` enum
3. **Nouveaux signaux early**: Ajouter composant dans `_score_*()` methods
4. **Nouveaux calculs risque**: Modifier `_calculate_risk_metrics()`

### Tests Recommandés
1. **Unit tests**: Chaque méthode `_score_*()`, `_validate_*()`, `_calculate_*()`
2. **Integration tests**: Workflow complet avec analyzer_data réels
3. **Edge cases**: Data manquante, valeurs extrêmes, anomalies
4. **Performance**: Temps calcul < 50ms par opportunité
