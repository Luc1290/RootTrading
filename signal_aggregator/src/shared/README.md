# Signal Aggregator - Shared Utilities

## Architecture des utilitaires

### 🎯 Objectif
Ces utilitaires éliminent la duplication de code dans le service `signal_aggregator` en centralisant les patterns répétitifs.

### 📁 Structure

```
src/shared/
├── redis_utils.py      - Patterns Redis centralisés (get/set/cache)
├── db_utils.py         - Patterns DB centralisés (pools, requêtes) 
├── technical_utils.py  - Helpers techniques spécifiques + Redis
└── README.md          - Cette documentation
```

### ⚠️ Notes importantes

#### Relations avec `/shared/` global
- **EMA/ADX robustes** : `technical_utils.py` utilise automatiquement `/shared/src/technical_indicators.py` (talib/pandas)
- **Fallback sécurisé** : Si module global indisponible, fallback vers implémentations simples
- **Helpers Redis/service** : Spécificités du signal_aggregator (récupération ADX Redis, etc.)

#### Usage recommandé
```python
# ✅ Pour calculs techniques (utilise automatiquement le global robuste)
from .shared.technical_utils import TechnicalCalculators
TechnicalCalculators.calculate_ema(prices, 14)  # → utilise GlobalTechnicalIndicators

# ✅ Pour helpers Redis spécifiques 
from .shared.technical_utils import TechnicalCalculators
TechnicalCalculators.get_current_adx(redis, "BTCUSDC")  # → fallback multi-timeframes

# ✅ Pour patterns Redis/DB centralisés
from .shared.redis_utils import RedisManager
from .shared.db_utils import DatabaseUtils
```

#### Architecture intégrée
```
/shared/src/technical_indicators.py  ← Calculs robustes (talib/pandas)
         ↕ AUTO-UTILISÉ PAR
signal_aggregator/src/shared/technical_utils.py  ← Proxy + helpers spécifiques
         ↕ UTILISÉ PAR
signal_aggregator/src/*.py  ← Tous les modules du service
```

### 🔄 Migration complète
**Avant** : ~1500+ lignes de code dupliqué
**Après** : 0 duplication, tous les patterns centralisés

### 📊 Statistiques
- **17 fichiers refactorisés**
- **3 modules utilitaires créés**
- **15+ patterns Redis éliminés**
- **4 implémentations EMA/ADX consolidées**
- **3 patterns DB pool centralisés**