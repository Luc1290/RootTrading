# Analyzer Service

## Vue d'ensemble
Le service **Analyzer** est le moteur d'exécution des stratégies de trading dans l'écosystème RootTrading. Il charge les données enrichies avec indicateurs et exécute toutes les stratégies pour générer des signaux de trading.

## Responsabilités
- **Exécution de stratégies** : Application de toutes les stratégies de trading configurées
- **Chargement dynamique** : Découverte et chargement automatique des stratégies
- **Multi-processing** : Exécution parallèle pour des performances optimales
- **Génération de signaux** : Production de signaux BUY/SELL qualifiés
- **Monitoring** : Suivi des performances et statistiques d'exécution

## Architecture

### Composants principaux
- `main.py` : Service principal et orchestration
- `strategy_loader.py` : Chargement dynamique des stratégies
- `multiproc_manager.py` : Gestion du traitement parallèle
- `redis_subscriber.py` : Publication des signaux vers Redis

### Flux de données
```
analyzer_data (DB) → Analyzer → Stratégies → Signaux → Redis/Kafka
```

## Configuration

### Variables d'environnement requises
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` : Configuration PostgreSQL
- `REDIS_URL` : URL de connexion Redis
- `ANALYSIS_INTERVAL` : Intervalle d'analyse en secondes (défaut: 60s)
- `DEBUG_LOGS` : Active les logs détaillés

### Ports
- **5012** : Port HTTP pour l'API de monitoring

## Stratégies disponibles

### Stratégies techniques
- `ADX_Direction_Strategy` : Direction basée sur l'ADX
- `ATR_Breakout_Strategy` : Cassures basées sur l'ATR
- `Bollinger_Touch_Strategy` : Touches des bandes de Bollinger
- `CCI_Reversal_Strategy` : Retournements CCI
- `Donchian_Breakout_Strategy` : Cassures des canaux de Donchian
- `EMA_Cross_Strategy` : Croisements de moyennes exponentielles
- `HullMA_Slope_Strategy` : Pente de la Hull Moving Average
- `MACD_Crossover_Strategy` : Croisements MACD
- `OBV_Crossover_Strategy` : Croisements OBV
- `ParabolicSAR_Bounce_Strategy` : Rebonds Parabolic SAR
- `PPO_Crossover_Strategy` : Croisements Price Percentage Oscillator

### Stratégies avancées
- `Liquidity_Sweep_Buy_Strategy` : Détection de sweeps de liquidité
- `MultiTF_ConfluentEntry_Strategy` : Entrées multi-timeframes confluentes
- `Pump_Dump_Pattern_Strategy` : Détection de patterns pump & dump

## Endpoints API

### Health Check
```http
GET /health
```
Retourne le statut du service et les statistiques globales.

### Statistiques détaillées
```http
GET /stats
```
Métriques de performance par stratégie et timeframe.

### Signaux récents
```http
GET /recent-signals?limit=100
```
Liste des derniers signaux générés.

### Contrôle du cycle
```http
POST /trigger-analysis
```
Force l'exécution d'un cycle d'analyse.

## Performance

### Multi-processing
- Exécution parallèle des stratégies par processus
- Gestion intelligente de la charge CPU
- Isolation des erreurs par stratégie
- Pool de processus configurables

### Optimisations
- **Cache des données** : Réutilisation des données chargées
- **Analyse par batch** : Traitement groupé des symboles
- **Filtrage intelligent** : Skip des analyses redondantes
- **Gestion mémoire** : Nettoyage automatique des données

## Métriques de monitoring

### Statistiques globales
- Nombre total d'analyses effectuées
- Signaux générés par cycle
- Stratégies exécutées par timeframe
- Temps d'exécution moyen

### Performance par stratégie
- Taux de succès des signaux
- Temps d'exécution moyen
- Fréquence de génération
- Couverture des symboles

## Démarrage

### Avec Docker
```bash
docker-compose up analyzer
```

### En développement
```bash
cd analyzer
python src/main.py
```

## Structure des stratégies

### Classe de base
Toutes les stratégies héritent de `BaseStrategy` :

```python
class BaseStrategy:
    def analyze(self, symbol: str, timeframe: str, data: pd.DataFrame) -> List[Dict]:
        # Logique de la stratégie
        pass
    
    def get_required_indicators(self) -> List[str]:
        # Liste des indicateurs requis
        pass
```

### Format des signaux
```json
{
    "signal_type": "BUY|SELL",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "price": 45000.0,
    "strength": 0.85,
    "strategy": "EMA_Cross_Strategy",
    "timestamp": "2024-01-01T12:00:00Z",
    "metadata": {...}
}
```

## Gestion d'erreurs
- Isolation des erreurs par stratégie
- Retry automatique avec backoff
- Logging détaillé des échecs
- Continuité du service malgré les erreurs

## Dépendances
- **TimescaleDB** : Source des données enrichies
- **Redis** : Publication des signaux
- **Kafka** : Distribution des signaux (optionnel)
- **multiprocessing** : Parallélisation des calculs

## Architecture technique
```
AnalyzerService
├── StrategyLoader (découverte automatique)
├── MultiProcessManager (parallélisation)
└── RedisPublisher (diffusion signaux)
    └── Strategy_1, Strategy_2, ..., Strategy_N
```