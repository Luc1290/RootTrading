# 🔄 Freqtrade Integration pour ROOT Trading Bot

Adaptateurs bidirectionnels pour backtester stratégies ROOT avec Freqtrade et importer stratégies communauté Freqtrade.

---

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Installation](#installation)
- [Usage rapide](#usage-rapide)
- [Adaptateurs](#adaptateurs)
- [Exemples](#exemples)
- [Workflows](#workflows)
- [FAQ](#faq)

---

## 🎯 Vue d'ensemble

Cette intégration permet de:

✅ **ROOT → Freqtrade**: Backtester vos stratégies ROOT avec l'engine Freqtrade
✅ **Freqtrade → ROOT**: Importer stratégies communauté (400+ disponibles)
✅ **Conversion batch**: Convertir toutes les stratégies en une commande
✅ **Validation**: Tester performance avant déploiement production

### Architecture

```
analyzer/
├── strategies/              # Stratégies ROOT existantes
│   ├── EMA_Cross_Strategy.py
│   ├── MACD_Crossover_Strategy.py
│   └── ...
│
└── freqtrade_integration/   # Module d'intégration
    ├── __init__.py
    ├── data_converter.py           # Conversion données dict ↔ DataFrame
    ├── root_to_freqtrade.py        # Adaptateur ROOT → Freqtrade
    ├── freqtrade_to_root.py        # Adaptateur Freqtrade → ROOT
    ├── batch_converter.py          # Conversion en masse
    │
    ├── examples/                   # Exemples d'utilisation
    │   └── backtest_example.py
    │
    └── converted_strategies/       # Stratégies converties (auto-généré)
        ├── freqtrade/              # ROOT → Freqtrade
        └── root/                   # Freqtrade → ROOT
```

---

## 🚀 Installation

### 1. Installer Freqtrade

```bash
# Créer environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Installer Freqtrade
pip install freqtrade

# Setup Freqtrade (config interactive)
freqtrade new-config --config user_data/config.json
```

### 2. Télécharger données historiques

```bash
# Télécharger 30 jours de données 5min pour paires principales
freqtrade download-data \
    --exchange binance \
    --pairs BTC/USDT ETH/USDT BNB/USDT SOL/USDT \
    --timeframe 5m \
    --days 30
```

### 3. Vérifier installation

```bash
freqtrade --version
# Devrait afficher: freqtrade 2024.x
```

---

## ⚡ Usage rapide

### 1️⃣ Backtester une stratégie ROOT

```python
from analyzer.freqtrade_integration import RootToFreqtradeAdapter
from analyzer.strategies.EMA_Cross_Strategy import EMA_Cross_Strategy

# Convertir stratégie ROOT → Freqtrade
adapter = RootToFreqtradeAdapter(EMA_Cross_Strategy)
freqtrade_strategy = adapter.convert()

# Exporter vers fichier
adapter.export_to_file("./strategies/EMA_Cross_Freqtrade.py")
```

```bash
# Backtester avec CLI Freqtrade
freqtrade backtesting \
    --strategy EMA_Cross_Strategy_Freqtrade \
    --strategy-path ./strategies \
    --timeframe 5m \
    --timerange 20240101-20240131
```

### 2️⃣ Convertir toutes les stratégies (batch)

```python
from analyzer.freqtrade_integration import BatchConverter

converter = BatchConverter()

# Convertir toutes ROOT → Freqtrade
stats = converter.convert_all_root_to_freqtrade()
print(f"Succès: {stats['success']}/{stats['total']}")

# Générer rapport
converter.generate_report()
```

```bash
# Ou via CLI
python -m analyzer.freqtrade_integration.batch_converter \
    root2ft \
    --output ./converted_strategies/freqtrade \
    --report
```

### 3️⃣ Importer stratégie Freqtrade

```python
from analyzer.freqtrade_integration import FreqtradeToRootAdapter
from some_freqtrade_strategy import MyFreqtradeStrategy

# Convertir Freqtrade → ROOT
adapter = FreqtradeToRootAdapter(MyFreqtradeStrategy)
root_strategy = adapter.convert()

# Analyser métadonnées
info = adapter.analyze_strategy()
print(info)
```

---

## 🔧 Adaptateurs

### DataConverter

Convertit données entre formats ROOT (dict) et Freqtrade (DataFrame).

```python
from analyzer.freqtrade_integration import DataConverter

# ROOT → DataFrame
data = {'close_price': 50000, 'volume': 1000, ...}
indicators = {'rsi': 45, 'ema_12': 49500, ...}
df = DataConverter.root_to_dataframe(data, indicators)

# DataFrame → ROOT
data, indicators = DataConverter.dataframe_to_root(df, symbol='BTCUSDT')

# Valider DataFrame
valid = DataConverter.validate_dataframe(df)

# Resample timeframe
df_1h = DataConverter.resample_dataframe(df, timeframe='1h')
```

### RootToFreqtradeAdapter

Convertit stratégie ROOT → Freqtrade pour backtesting.

```python
from analyzer.freqtrade_integration import RootToFreqtradeAdapter
from analyzer.strategies.MACD_Crossover_Strategy import MACD_Crossover_Strategy

adapter = RootToFreqtradeAdapter(MACD_Crossover_Strategy)

# Convertir en classe Freqtrade IStrategy
freqtrade_strategy = adapter.convert()

# Métadonnées disponibles
print(freqtrade_strategy.minimal_roi)    # {'0': 0.10, '30': 0.05, ...}
print(freqtrade_strategy.stoploss)       # -0.05
print(freqtrade_strategy.timeframe)      # '5m'

# Exporter vers fichier Python
adapter.export_to_file("./MACD_Freqtrade.py")
```

### FreqtradeToRootAdapter

Convertit stratégie Freqtrade → ROOT pour production.

```python
from analyzer.freqtrade_integration import FreqtradeToRootAdapter

# Import stratégie Freqtrade (depuis communauté par ex)
from freqtrade_strategies import SampleStrategy

adapter = FreqtradeToRootAdapter(SampleStrategy)

# Analyser la stratégie avant conversion
info = adapter.analyze_strategy()
print(f"Timeframe: {info['timeframe']}")
print(f"Stoploss: {info['stoploss']}")
print(f"Supports short: {info['supports_short']}")  # False pour ROOT (SPOT)

# Convertir avec mapping confidence custom
confidence_map = {
    'very_strong': 0.9,
    'strong': 0.75,
    'moderate': 0.6,
    'weak': 0.45
}
root_strategy = adapter.convert(confidence_mapping=confidence_map)

# Utiliser en production
from analyzer.src.strategy_loader import StrategyLoader

loader = StrategyLoader()
loader.strategies['SampleStrategy_ROOT'] = root_strategy

# Instancier et générer signal
strategy = root_strategy(
    symbol='BTCUSDT',
    data=data,
    indicators=indicators
)
signal = strategy.generate_signal()
```

### BatchConverter

Conversion en masse de stratégies.

```python
from analyzer.freqtrade_integration import BatchConverter

converter = BatchConverter()

# ROOT → Freqtrade (toutes les stratégies du dossier)
stats = converter.convert_all_root_to_freqtrade(
    output_dir='./freqtrade_strategies',
    exclude_patterns=['test_', 'debug_']
)

print(f"Converties: {stats['success']}")
print(f"Erreurs: {stats['failed']}")

# Freqtrade → ROOT
stats = converter.convert_all_freqtrade_to_root(
    freqtrade_dir='./external_strategies',
    output_dir='./root_strategies'
)

# Générer rapport détaillé
converter.generate_report('conversion_report.txt')
```

---

## 📚 Exemples

### Exemple 1: Backtester stratégie ROOT avec résultats

```python
from analyzer.freqtrade_integration.examples import backtest_example

# Convertir et exporter stratégie
output_file = backtest_example.backtest_root_strategy_simple()
# → Génère: strategies_freqtrade/EMA_Cross_Strategy_freqtrade.py
```

```bash
# Backtester
freqtrade backtesting \
    --strategy EMA_Cross_Strategy_Freqtrade \
    --strategy-path ./strategies_freqtrade \
    --timeframe 5m \
    --timerange 20240101-20240131 \
    --export trades

# Voir résultats
freqtrade backtesting-show

# Exemple output:
# ========================================
# Total trades: 142
# Win rate: 58.45%
# Total return: +23.47%
# Max drawdown: -8.32%
# Sharpe ratio: 1.84
# Avg trade duration: 2h 15min
# ========================================
```

### Exemple 2: Comparer plusieurs stratégies

```bash
# Script bash pour comparer toutes les stratégies
#!/bin/bash

STRATEGIES=(
    "EMA_Cross_Strategy"
    "MACD_Crossover_Strategy"
    "Bollinger_Touch_Strategy"
    "RSI_Cross_Strategy"
)

for strat in "${STRATEGIES[@]}"; do
    echo "Testing $strat..."

    freqtrade backtesting \
        --strategy ${strat}_Freqtrade \
        --strategy-path ./converted_strategies/freqtrade \
        --timeframe 5m \
        --timerange 20240101-20240131 \
        --export trades

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
done

# Comparer résultats
freqtrade backtesting-show --backtest-dir user_data/backtest_results/
```

### Exemple 3: Optimiser paramètres avec Hyperopt

```bash
# Hyperopt cherche meilleurs paramètres pour stratégie ROOT
freqtrade hyperopt \
    --strategy EMA_Cross_Strategy_Freqtrade \
    --hyperopt-loss SharpeHyperOptLoss \
    --spaces roi stoploss trailing \
    --epochs 1000 \
    --timeframe 5m \
    --timerange 20240101-20240131

# Résultats affichent meilleurs paramètres trouvés
# Ex:
# Best ROI: {0: 0.158, 30: 0.078, 60: 0.032, 120: 0}
# Best stoploss: -0.034
# Best trailing_stop_positive: 0.018
```

### Exemple 4: Import stratégie communauté Freqtrade

```python
# 1. Cloner repo stratégies Freqtrade
# git clone https://github.com/freqtrade/freqtrade-strategies

# 2. Importer stratégie
import sys
sys.path.append('./freqtrade-strategies')

from user_data.strategies import BinHV45  # Stratégie populaire

from analyzer.freqtrade_integration import FreqtradeToRootAdapter

adapter = FreqtradeToRootAdapter(BinHV45)

# Analyser
info = adapter.analyze_strategy()
print(f"Timeframe: {info['timeframe']}")  # 5m
print(f"ROI: {info['minimal_roi']}")       # {0: 0.4, 10: 0.2, ...}

# Convertir vers ROOT
root_strategy = adapter.convert()

# Tester en simulation ROOT
strategy_instance = root_strategy(
    symbol='BTCUSDT',
    data={'close_price': 50000, 'volume': 1000, ...},
    indicators={'rsi': 45, 'ema_12': 49500, ...}
)

signal = strategy_instance.generate_signal()
print(signal)
# {
#   'side': 'BUY',
#   'confidence': 0.72,
#   'strength': 'strong',
#   'reason': 'Freqtrade entry signal: BinHV45_strong',
#   'metadata': {'source': 'freqtrade', ...}
# }
```

---

## 🔄 Workflows

### Workflow 1: Développement nouvelle stratégie ROOT

```
1. Créer stratégie ROOT
   ├─ analyzer/strategies/My_New_Strategy.py
   └─ Hérite de BaseStrategy

2. Convertir → Freqtrade
   └─ python -c "from analyzer.freqtrade_integration import RootToFreqtradeAdapter; ..."

3. Backtester avec données historiques
   └─ freqtrade backtesting --strategy My_New_Strategy_Freqtrade

4. Analyser résultats
   ├─ Win rate OK? (>55%)
   ├─ Drawdown acceptable? (<15%)
   └─ Sharpe ratio? (>1.5)

5. SI OK → Déployer en production ROOT
   └─ SI KO → Ajuster paramètres → Retour à 2
```

### Workflow 2: Import stratégie communauté

```
1. Trouver stratégie Freqtrade intéressante
   └─ https://github.com/freqtrade/freqtrade-strategies

2. Backtester avec vos données
   └─ Vérifier performance sur vos paires crypto

3. Convertir → ROOT
   └─ FreqtradeToRootAdapter

4. Tester en simulation ROOT (paper trading)
   └─ Vérifier signaux cohérents

5. Intégrer dans signal_aggregator
   └─ Ajouter poids dans agrégation multi-stratégies
```

### Workflow 3: Optimisation stratégie existante

```
1. Sélectionner stratégie ROOT à optimiser

2. Convertir → Freqtrade

3. Hyperopt (recherche paramètres optimaux)
   └─ freqtrade hyperopt --epochs 1000

4. Récupérer meilleurs paramètres
   ├─ ROI optimal
   ├─ Stoploss optimal
   └─ Trailing stop optimal

5. Appliquer dans stratégie ROOT
   └─ Modifier paramètres dans classe Python

6. Backtester à nouveau pour confirmer

7. Déployer en production
```

---

## 🛠️ Configuration avancée

### Personnaliser ROI et Stoploss

```python
from analyzer.freqtrade_integration import RootToFreqtradeAdapter

adapter = RootToFreqtradeAdapter(MyStrategy)
freqtrade_strategy = adapter.convert()

# Personnaliser paramètres Freqtrade
freqtrade_strategy.minimal_roi = {
    "0": 0.15,    # 15% ROI immédiat
    "30": 0.08,   # 8% après 30min
    "60": 0.04,   # 4% après 1h
    "120": 0.01   # 1% après 2h
}

freqtrade_strategy.stoploss = -0.03  # Stop loss -3%

freqtrade_strategy.trailing_stop = True
freqtrade_strategy.trailing_stop_positive = 0.015
freqtrade_strategy.trailing_stop_positive_offset = 0.025
```

### Filtrer paires à backtester

```json
// config.json
{
  "exchange": {
    "name": "binance",
    "pair_whitelist": [
      "BTC/USDT",
      "ETH/USDT",
      "BNB/USDT"
    ],
    "pair_blacklist": [
      ".*BUSD.*",
      ".*TUSD.*"
    ]
  }
}
```

### Exporter résultats pour analyse

```bash
# Exporter tous les trades en JSON
freqtrade backtesting \
    --strategy MyStrategy \
    --export trades \
    --export-filename backtest_results.json

# Générer graphiques HTML
freqtrade plot-dataframe \
    --strategy MyStrategy \
    --indicators1 ema_12,ema_26,ema_50 \
    --indicators2 rsi,macd_line \
    --export-filename plot.html

# Ouvrir dans navigateur
# → user_data/plot/freqtrade-plot-BTCUSDT-5m.html
```

---

## 📊 Métriques de backtesting

### Métriques clés à surveiller

| Métrique | Cible | Description |
|----------|-------|-------------|
| **Total Return** | >20% | Profit total sur période |
| **Win Rate** | >55% | % de trades gagnants |
| **Profit Factor** | >1.5 | Gain moyen / Perte moyenne |
| **Max Drawdown** | <15% | Perte maximale depuis pic |
| **Sharpe Ratio** | >1.5 | Rendement ajusté au risque |
| **Avg Trade Duration** | Varie | Durée moyenne trade (2-4h idéal) |
| **Total Trades** | >50 | Nombre trades (validité statistique) |

### Interpréter résultats

✅ **Stratégie prometteuse:**
```
Total Return: +28.5%
Win Rate: 62.3%
Max Drawdown: -9.2%
Sharpe Ratio: 2.14
Total Trades: 187
→ Déployer en production ROOT
```

⚠️ **Stratégie à améliorer:**
```
Total Return: +12.1%
Win Rate: 48.5%
Max Drawdown: -22.7%
Sharpe Ratio: 0.73
Total Trades: 42
→ Ajuster paramètres, retester
```

❌ **Stratégie à rejeter:**
```
Total Return: -8.4%
Win Rate: 39.2%
Max Drawdown: -35.1%
Sharpe Ratio: -0.42
Total Trades: 231
→ Stratégie non viable
```

---

## 🐛 Troubleshooting

### Erreur: "Freqtrade not installed"

```bash
pip install freqtrade
# Ou avec extras
pip install freqtrade[plot,hyperopt]
```

### Erreur: "No data found"

```bash
# Télécharger données historiques
freqtrade download-data \
    --exchange binance \
    --pairs BTC/USDT ETH/USDT \
    --timeframe 5m \
    --days 30
```

### Erreur: "Strategy has no entry signal"

→ Vérifier que stratégie ROOT génère bien des signaux BUY
→ Logger les appels à `generate_signal()` dans adaptateur
→ Vérifier indicateurs disponibles dans DataFrame

### Performance backtesting lente

```python
# Dans stratégie Freqtrade, activer optimisations
class MyStrategy(IStrategy):
    process_only_new_candles = True  # Process 1 fois par chandelier
    startup_candle_count = 50        # Minimum chandeliers requis
```

---

## 🔗 Ressources

- [Freqtrade Documentation](https://www.freqtrade.io/en/stable/)
- [Freqtrade Strategies Community](https://github.com/freqtrade/freqtrade-strategies)
- [Backtesting Guide](https://www.freqtrade.io/en/stable/backtesting/)
- [Hyperopt Guide](https://www.freqtrade.io/en/stable/hyperopt/)
- [Strategy Development](https://www.freqtrade.io/en/stable/strategy-customization/)

---

## 📝 FAQ

**Q: Puis-je backtester plusieurs stratégies ROOT en parallèle?**
R: Oui, utilisez `BatchConverter` pour convertir toutes vos stratégies, puis script bash pour backtester en séquence.

**Q: Les résultats de backtest garantissent-ils performance future?**
R: Non. Backtest = validation historique. Toujours tester en paper trading avant production.

**Q: Quelle période de backtest recommandée?**
R: Minimum 30 jours, idéal 90+ jours incluant différentes conditions marché (bull, bear, range).

**Q: Puis-je utiliser stratégies Freqtrade SHORT?**
R: Non, ROOT = SPOT uniquement (BUY/SELL). Signaux SHORT sont ignorés.

**Q: Comment ajouter indicateurs custom?**
R: Modifiez `populate_indicators()` dans stratégie Freqtrade générée.

**Q: Différence entre ROI et trailing stop?**
R: ROI = take profit fixe selon durée. Trailing stop = stop loss qui suit le prix à la hausse.

---

## 📄 Licence

Intégration compatible avec licence ROOT Trading Bot.

---

## 🤝 Contribution

Pour améliorer cette intégration:

1. Tester avec vos stratégies ROOT
2. Reporter bugs/suggestions
3. Proposer optimisations adaptateurs
4. Partager résultats backtests

---

**Version:** 1.0.0
**Dernière mise à jour:** 2024
**Mainteneur:** ROOT Trading Team
