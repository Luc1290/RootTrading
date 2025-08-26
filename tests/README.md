# Tests ROOT Trading System

## 🎯 Vue d'ensemble

Suite de tests complète pour valider le système de trading automatisé ROOT. Tests unitaires, d'intégration et validation des formats de données.

## 📁 Structure

```
tests/
├── unit/                           # Tests unitaires
│   ├── strategies/                 # Tests des 27 stratégies
│   │   ├── test_base_strategy.py  # BaseStrategy
│   │   ├── test_macd_crossover_strategy.py
│   │   ├── test_rsi_cross_strategy.py
│   │   └── test_ema_cross_strategy.py
│   ├── market_analyzer/           # Tests market_analyzer
│   │   └── test_moving_averages.py
│   └── db_formats/                # Validation formats DB
│       ├── test_market_data_format.py
│       └── test_indicators_format.py
├── integration/                   # Tests d'intégration
├── fixtures/                     # Données de test
└── utils/                        # Utilitaires de test
```

## 🚀 Lancement des tests

### Installation
```bash
cd tests/
pip install -r requirements.txt
```

### Exécution
```bash
# Tous les tests
pytest

# Tests spécifiques
pytest unit/strategies/ -v                    # Toutes les stratégies
pytest unit/strategies/test_macd* -v          # MACD uniquement
pytest unit/db_formats/ -v                   # Formats DB
pytest unit/market_analyzer/ -v              # Market analyzer

# Avec couverture
pytest --cov=analyzer --cov=market_analyzer --cov-report=html

# Tests par marqueurs
pytest -m strategies                         # Stratégies uniquement
pytest -m db_format                         # Formats DB uniquement
pytest -m unit                              # Tests unitaires uniquement
```

## 🧪 Types de tests

### 1. Tests des Stratégies
- **Validation des entrées** : Données manquantes, formats invalides
- **Logique métier** : Croisements, seuils, filtres
- **Gestion des erreurs** : Cas limites, valeurs extrêmes
- **Cohérence des signaux** : Format, confidence, strength

### 2. Tests des Formats DB
- **MarketData** : OHLCV, timestamps, validation Pydantic
- **Indicateurs** : Plages valides, cohérence (MACD, Bollinger, etc.)
- **Sérialisation** : JSON, types SQL compatibles

### 3. Tests Market Analyzer
- **Moyennes mobiles** : SMA, EMA, calculs précis
- **Indicateurs techniques** : RSI, MACD, Stochastic
- **Performance** : Vitesse de calcul, cache Redis

## 📊 Couverture

Objectifs de couverture :
- **Stratégies** : 90%+ (logique critique)
- **Market Analyzer** : 85%+ (calculs techniques)
- **Formats DB** : 95%+ (validation données)

## 🔧 Configuration

### Fixtures disponibles
- `sample_ohlcv_data` : Données OHLCV réalistes
- `sample_indicators` : Indicateurs techniques complets
- `market_data_db_format` : Format DB market_data
- `indicators_db_format` : Format DB indicateurs
- `mock_strategy_data` : Données complètes pour stratégies

### Marqueurs pytest
```python
@pytest.mark.unit          # Test unitaire
@pytest.mark.strategies    # Test de stratégie
@pytest.mark.db_format     # Test format DB
@pytest.mark.slow          # Test lent
@pytest.mark.requires_redis # Nécessite Redis
```

## 📋 Checklist Tests Stratégies

Pour chaque stratégie, vérifier :

- [ ] **Initialisation** : Paramètres, nom, symbole
- [ ] **Validation données** : `validate_data()` fonctionne
- [ ] **Signaux BUY** : Conditions haussières
- [ ] **Signaux SELL** : Conditions baissières  
- [ ] **Pas de signal** : Zone neutre, données insuffisantes
- [ ] **Filtres** : Confluence, régime marché, volume
- [ ] **Gestion erreurs** : Données manquantes, types invalides
- [ ] **Format signal** : Structure complète et cohérente
- [ ] **Confidence** : Calcul et plages [0-1]
- [ ] **Strength** : Mapping confidence → strength

## 🚨 Tests Critiques

Tests qui **DOIVENT** passer avant déploiement :

1. **BaseStrategy** : Classe abstraite fonctionnelle
2. **MACD_Crossover** : Logique croisement + filtres
3. **RSI_Cross** : Zones survente/surachat + confirmations
4. **EMA_Cross** : Croisements + filtres tendance
5. **MarketData** : Validation Pydantic complète
6. **Indicateurs** : Plages et cohérence (BB, MACD, etc.)

## 📈 Exécution continue

```bash
# Monitoring continu pendant développement
pytest --cov --cov-report=term-missing -f

# Tests rapides (sans lents)
pytest -m "not slow"

# Tests critique uniquement
pytest unit/strategies/test_base_strategy.py unit/db_formats/ -x
```

## 🔍 Debug Tests

```bash
# Mode verbeux avec détails
pytest -vvs unit/strategies/test_macd_crossover_strategy.py

# Arrêt au premier échec
pytest -x

# Pdb au premier échec
pytest --pdb

# Logs détaillés
pytest -o log_cli=true --log-cli-level=DEBUG
```