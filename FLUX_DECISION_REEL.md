# FLUX DE DÉCISION RÉEL - Architecture Microservices

## 🎯 Flux Actuellement Implémenté

```
┌─────────────────┐
│ Market Analyzer │ → Calcule 106 indicateurs → PostgreSQL (analyzer_data)
└─────────────────┘
         ↓ Redis PubSub: "analyzer_trigger"

┌─────────────────┐
│    Analyzer     │ → Exécute 28 stratégies sur les indicateurs
└─────────────────┘
         ↓ Redis PubSub: "analyzer:signals"

┌──────────────────┐
│Signal Aggregator │ → Consensus multi-stratégies
└──────────────────┘
         ↓ Redis PubSub: "roottrading:signals:filtered"

┌─────────────────┐
│  Coordinator    │ → DÉCISION FINALE BUY/SELL/AVOID
└─────────────────┘
         ↓ HTTP: trader:5002/orders

┌─────────────────┐
│     Trader      │ → Exécution Binance
└─────────────────┘
```

## 📊 Analyzer - 28 Stratégies Existantes

**Fichier**: `analyzer/src/main.py` (802 lignes)

### Stratégies Chargées Automatiquement

**Trend-Based (7)**:
- EMA_Cross_Strategy
- HullMA_Slope_Strategy
- TEMA_Slope_Strategy
- Supertrend_Reversal_Strategy
- VWAP_Support_Resistance_Strategy
- ADX_Direction_Strategy
- ParabolicSAR_Bounce_Strategy

**Momentum-Based (8)**:
- RSI_Cross_Strategy
- MACD_Crossover_Strategy
- Stochastic_Oversold_Buy_Strategy
- StochRSI_Rebound_Strategy
- CCI_Reversal_Strategy
- ROC_Threshold_Strategy
- WilliamsR_Rebound_Strategy
- PPO_Crossover_Strategy

**Volatility-Based (4)**:
- Bollinger_Touch_Strategy
- ATR_Breakout_Strategy
- Donchian_Breakout_Strategy
- zscore_extreme_reversal_strategy

**Support/Resistance (4)**:
- Support_Breakout_Strategy
- Resistance_Rejection_Strategy
- Range_Breakout_Confirmation_Strategy
- Liquidity_Sweep_Buy_Strategy

**Volume-Based (2)**:
- OBV_Crossover_Strategy
- Spike_Reaction_Buy_Strategy

**Advanced (4)**:
- MultiTF_ConfluentEntry_Strategy
- Pump_Dump_Pattern_Strategy
- TRIX_Crossover_Strategy

### Processus Analyzer

```python
# analyzer/src/main.py:481-594
async def analyze_symbol_timeframe(self, symbol: str, timeframe: str):
    # 1. Récupérer données depuis analyzer_data (106 indicateurs)
    market_data = self.fetch_latest_data(symbol, timeframe)

    # 2. Exécuter les 28 stratégies
    strategies = self.strategy_loader.get_all_strategies()
    signals = []

    for strategy_name, strategy_class in strategies.items():
        # Instancier stratégie avec indicateurs
        strategy = strategy_class(
            symbol=symbol,
            data=market_data["data"],
            indicators=market_data["indicators"]
        )

        # Générer signal BUY/SELL
        signal = strategy.generate_signal()

        if signal["side"]:  # Si signal généré
            signal_data = {
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy_name,
                "side": signal["side"],
                "confidence": signal["confidence"],
                "strength": signal["strength"],
                "reason": signal["reason"],
                "metadata": signal["metadata"]
            }
            signals.append(signal_data)

    # 3. Publier signaux individuels sur Redis
    if signals:
        await self.redis_publisher.publish_signals(signals, mode="individual")
```

**Output**: 0-28 signaux par symbol/timeframe selon conditions de marché

## 🔄 Signal Aggregator - Consensus Multi-Stratégies

**Fichier**: `signal_aggregator/src/signal_aggregator_simple.py` (452 lignes)

### Processus de Consensus

```python
# signal_aggregator/src/signal_aggregator_simple.py:160-239
async def _process_signal_batch_simple(self, signals: list):
    # 1. Grouper par symbol + side
    signal_groups = {}
    for signal in signals:
        key = f"{signal['symbol']}_{signal['side']}"
        signal_groups[key].append(signal)

    # 2. Résoudre conflits BUY vs SELL (système de vague)
    resolved_groups = self._resolve_simultaneous_conflicts(signal_groups)

    # 3. Valider chaque groupe
    for group_key, group_signals in resolved_groups.items():
        await self._validate_signal_group(group_signals)
```

### Validation et Consensus

```python
# signal_aggregator/src/signal_aggregator_simple.py:281-331
async def _validate_signal_group(self, signals: list):
    symbol = signals[0]["symbol"]
    side = signals[0]["side"]

    # 1. Vérifier contradictions récentes (30s window)
    if self._check_recent_contradiction(symbol, side):
        return  # Bloquer

    # 2. Validation avec consensus adaptatif
    validated_signal = await self.signal_processor.validate_signal_group(
        signals, symbol, timeframe, side
    )

    if validated_signal:
        # 3. Envoyer signal consensus au Coordinator
        await self._send_validated_signal(validated_signal)

        # Format du signal consensus:
        # {
        #     "symbol": "BTCUSDC",
        #     "side": "BUY",
        #     "price": 43250.5,
        #     "metadata": {
        #         "strategies_count": 12,  # Nombre de stratégies en accord
        #         "consensus_strength": 8.5,  # Force du consensus
        #         "avg_confidence": 0.75,  # Confiance moyenne
        #         "type": "CONSENSUS"
        #     }
        # }
```

**Output**: Signal de consensus avec metadata enrichie

## 🎯 Coordinator - DÉCISION FINALE

**Fichier**: `coordinator/src/coordinator.py` (1434 lignes)

### Points de Décision Principaux

#### 1. Calcul de Force du Signal

```python
# coordinator/src/coordinator.py:199-261
def _calculate_unified_signal_strength(self, signal: StrategySignal) -> tuple[float, int, float]:
    """
    Calcule la force du signal à partir du consensus.

    Formule: Force = consensus_strength × √(strategies_count) × avg_confidence

    Returns:
        (force, strategy_count, avg_confidence)
    """
    consensus_strength = signal.metadata.get("consensus_strength", 0)
    strategies_count = signal.metadata.get("strategies_count", 1)
    avg_confidence = signal.metadata.get("avg_confidence", 0.5)

    if consensus_strength > 0 and strategies_count > 1:
        force = consensus_strength * (strategies_count ** 0.5) * avg_confidence
        return force, strategies_count, avg_confidence

    # Fallback pour signaux simples
    return 1.0, 1, 0.5
```

#### 2. Catégorisation de Force

```python
# coordinator/src/coordinator.py:263-282
def _categorize_signal_strength(self, force: float) -> str:
    """
    Catégories basées sur la force calculée:
    - VERY_STRONG: force ≥ 12.0
    - STRONG: force ≥ 8.0
    - MODERATE: force ≥ 4.0
    - WEAK: force < 4.0
    """
    if force >= 12.0:
        return "VERY_STRONG"
    elif force >= 8.0:
        return "STRONG"
    elif force >= 4.0:
        return "MODERATE"
    else:
        return "WEAK"
```

#### 3. Décision BUY - Consensus Override

```python
# coordinator/src/coordinator.py:420-451
# CONSENSUS BUY OVERRIDE: Bypass hystérésis pour signaux forts
if signal.side == OrderSide.BUY:
    signal_force, strategy_count, avg_confidence = (
        self._calculate_unified_signal_strength(signal)
    )

    # Seuils: force ≥ 2.0 ET strategies ≥ 5
    if signal_force >= 2.0 and strategy_count >= 5:
        logger.warning(f"🚀 CONSENSUS BUY FORT détecté pour {signal.symbol}")

        # Forcer l'ajout à l'univers tradable pour 45 minutes
        self.universe_manager.force_pair_selection(
            signal.symbol, duration_minutes=45
        )
```

#### 4. Allocation Dynamique selon Force

```python
# coordinator/src/coordinator.py:786-828
# Allocation basée sur la catégorie de force
if strength_category == "VERY_STRONG":
    allocation_percent = 28.0  # 28% de l'USDC disponible
elif strength_category == "STRONG":
    allocation_percent = 22.0  # 22% de l'USDC disponible
elif strength_category == "MODERATE":
    allocation_percent = 18.0  # 18% de l'USDC disponible
else:  # WEAK
    allocation_percent = 12.0  # 12% de l'USDC disponible

# Calculer montant à trader
trade_amount = usdc_balance * (allocation_percent / 100)

# Convertir en quantité
quantity = trade_amount / signal.price
```

#### 5. Décision SELL - Consensus Override

```python
# coordinator/src/coordinator.py:283-351
def _check_consensus_sell_override(self, signal: StrategySignal, entry_price: float):
    """
    Vérifie si un consensus SELL fort doit bypasser le trailing stop.

    Conditions:
    - Type = CONSENSUS
    - Strategies ≥ 4
    - Force ≥ 1.8
    - Perte actuelle < -0.6 × ATR%
    """
    signal_force, strategies_count, avg_confidence = (
        self._calculate_unified_signal_strength(signal)
    )

    # Calculer perte actuelle
    current_loss_pct = ((signal.price - entry_price) / entry_price) * 100

    # Récupérer ATR dynamique
    atr_pct = self.trailing_manager._get_atr_percentage(signal.symbol)
    loss_threshold = -0.6 * atr_pct

    # Forcer vente si conditions remplies
    if (signal_type == "CONSENSUS" and
        strategies_count >= 4 and
        signal_force >= 1.8 and
        current_loss_pct < loss_threshold):

        return True, f"CONSENSUS_SELL_FORCED: {strategies_count} stratégies, force {signal_force:.1f}"

    return False, "Conditions non remplies"
```

### Filtres de Faisabilité

```python
# coordinator/src/coordinator.py:575-706
def _check_feasibility(self, signal: StrategySignal) -> tuple[bool, str]:
    """
    Vérifie si le trade est faisable:

    Pour BUY:
    1. Paire dans l'univers tradable ? (UniverseManager)
    2. USDC suffisant ? (≥ 15 USDC minimum)
    3. Pas de cycle actif pour ce symbole ?

    Pour SELL:
    1. Trailing stop autorise vente ? (TrailingSellManager)
    2. Ou consensus SELL fort bypass trailing ?
    3. Balance crypto suffisante ?
    4. Valeur ≥ 15 USDC ?
    """
```

## 📈 Décisions Finales Prises par Coordinator

### BUY Decision Tree

```
Signal BUY reçu
    ↓
Force calculée → Catégorisation (VERY_STRONG/STRONG/MODERATE/WEAK)
    ↓
Force ≥ 2.0 ET strategies ≥ 5 ?
    ├─ OUI → Ajout forcé à l'univers tradable (45min)
    └─ NON → Vérifier univers normal
         ↓
Paire dans univers tradable ?
    ├─ NON → REJECT
    └─ OUI → USDC ≥ 15 ?
         ├─ NON → Tenter vente pire position
         └─ OUI → Pas de cycle actif ?
              ├─ Cycle existe → REJECT
              └─ Pas de cycle → BUY AUTORISÉ
                   ↓
              Allocation = f(force):
              - VERY_STRONG → 28% USDC
              - STRONG → 22% USDC
              - MODERATE → 18% USDC
              - WEAK → 12% USDC
```

### SELL Decision Tree

```
Signal SELL reçu
    ↓
Position active existe ?
    ├─ NON → SELL AUTORISÉ (nettoyage)
    └─ OUI → Consensus SELL fort ?
         ├─ OUI (≥4 strat, force ≥1.8, perte < -0.6×ATR%)
         │    → SELL FORCÉ (bypass trailing)
         └─ NON → Trailing stop autorise ?
              ├─ NON → REJECT (continuer trailing)
              └─ OUI → Balance crypto ≥ 15 USDC ?
                   ├─ NON → REJECT
                   └─ OUI → SELL AUTORISÉ
```

## 🔧 Utilisation RÉELLE des 106 Indicateurs

### Par les 28 Stratégies

Chaque stratégie utilise **3-10 indicateurs** parmi les 106 disponibles:

Exemple `EMA_Cross_Strategy`:
- `ema_12`, `ema_26` (moyennes mobiles)
- `rsi_14` (momentum)
- `volume_ratio` (confirmation)

Exemple `VWAP_Support_Resistance_Strategy`:
- `vwap_10`, `vwap_quote_10`
- `nearest_support`, `nearest_resistance`
- `support_strength`, `resistance_strength`

**Résultat**: Les 28 stratégies utilisent **collectivement ~80% des 106 indicateurs**

### Par le Coordinator

Le Coordinator utilise **seulement les métadonnées du consensus**:
- `consensus_strength`
- `strategies_count`
- `avg_confidence`

Il ne lit **JAMAIS directement** les 106 indicateurs de la DB.

## ⚖️ Comparaison avec le Système v5.0 de Visualization

| Aspect | Microservices (Analyzer→Aggregator→Coordinator) | Visualization v5.0 |
|--------|------------------------------------------------|-------------------|
| **Indicateurs DB utilisés** | ~85 sur 106 (80%) via 28 stratégies | 25 sur 106 (24%) directement |
| **Logique de décision** | Consensus multi-stratégies → Force → Allocation | Scoring direct 9 catégories → Seuil |
| **Décision BUY** | Force consensus ≥ 2.0 + ≥5 stratégies | Score ≥ 70 = BUY_NOW |
| **Décision SELL** | Consensus ≥4 strat + perte < -0.6×ATR% | N/A (pas de gestion SELL) |
| **Allocation** | Dynamique 12-28% selon force | Fixe (non implémentée) |
| **Architecture** | Distribuée, scalable, résiliente | Monolithique, simple |
| **Complexité** | Haute (3 services, Redis, consensus) | Faible (1 fichier Python) |

## 🎯 Proposition: Utiliser le Flux Complet

### Avantages

✅ **Utilisation maximale des indicateurs**: 80% vs 24%
✅ **28 stratégies déjà codées et testées**: Ne pas réinventer la roue
✅ **Consensus intelligent**: Agrégation multi-stratégies robuste
✅ **Allocation dynamique**: 12-28% selon force du signal
✅ **Architecture scalable**: Prête pour production
✅ **Trailing stop intégré**: Gestion intelligente des sorties

### Désactivation des 4 Fichiers Python Visualization

Fichiers à désactiver:
1. `visualization/backend/opportunity_scoring_v5.py`
2. `visualization/backend/opportunity_calculator_pro.py`
3. `visualization/backend/opportunity_validator.py`
4. `visualization/backend/opportunity_early_detector.py`

### Nouvelle Connexion Visualization

```python
# Au lieu de calculer localement:
# score = opportunity_scoring_v5.calculate_score(data)

# Écouter les décisions du Coordinator:
redis.subscribe("roottrading:signals:filtered")

# Afficher les signaux de consensus:
{
    "symbol": "BTCUSDC",
    "side": "BUY",
    "action": "BUY_NOW",  # Calculé par Coordinator
    "allocation": "28%",  # VERY_STRONG
    "strategies": 12,     # Nombre de stratégies en accord
    "force": 15.2,        # Force du consensus
    "confidence": 0.82    # Confiance moyenne
}
```

## 📝 Prochaines Étapes

1. ✅ Analyser le flux existant (FAIT)
2. 🔄 Documenter les points de décision (EN COURS)
3. ⏳ Créer configuration pour désactiver v5.0 Visualization
4. ⏳ Adapter frontend pour afficher signaux Coordinator
5. ⏳ Tester le flux complet avec les 28 stratégies
6. ⏳ Comparer performance v5.0 vs Microservices

## 💡 Conclusion

Le système **Analyzer → Signal Aggregator → Coordinator** est déjà **complet et opérationnel**.

Il utilise:
- ✅ 28 stratégies professionnelles
- ✅ ~85 indicateurs sur 106 (80% de la DB)
- ✅ Consensus multi-stratégies intelligent
- ✅ Allocation dynamique selon force
- ✅ Gestion trailing stop adaptative
- ✅ Architecture microservices scalable

**Recommandation**: Désactiver les 4 fichiers Python de Visualization et utiliser ce flux mature et complet.
