# Règles de développement RootTrading

## Règle #1 : Module partagé pour indicateurs techniques

**RÈGLE ABSOLUE** : Toujours utiliser le module partagé `shared/src/technical_indicators.py` pour TOUS les calculs d'indicateurs techniques.

### ❌ Ne JAMAIS faire :
- Dupliquer les calculs RSI, EMA, MACD, Bollinger, ADX, etc. dans les différents services
- Créer des méthodes `_calculate_xxx()` dans gateway, analyzer, signal_aggregator
- Implémenter des calculs manuels quand le module partagé existe

### ✅ Toujours faire :
```python
from shared.src.technical_indicators import TechnicalIndicators
indicators = TechnicalIndicators()
result = indicators.calculate_xxx(data, period)
```

### Services concernés :
- Gateway (`binance_ws.py`, `ultra_data_fetcher.py`)
- Analyzer (toutes les stratégies)
- Signal_aggregator 
- Visualization
- Tous les autres services

### Avantages :
- **Cohérence garantie** entre tous les services
- **Un seul endroit** pour maintenir et débugger les calculs
- **Moins d'erreurs** et plus de fiabilité
- **Performance optimisée** avec TA-Lib quand disponible

---

*Cette règle est née suite aux problèmes de cohérence ADX entre gateway et signal_aggregator - 2025-07-02*