# Signal Aggregator Service

## Vue d'ensemble
Le service **Signal Aggregator** est le centre de fusion et validation des signaux dans l'écosystème RootTrading. Il traite les signaux provenant de toutes les stratégies et produit des signaux de trading consolidés et qualifiés.

## Responsabilités
- **Agrégation de signaux** : Consolidation des signaux multi-sources et multi-timeframes
- **Consensus adaptatif** : Logique de consensus basée sur les régimes de marché
- **Filtrage critique** : Application de filtres essentiels pour la qualité des signaux
- **Protection contradictions** : Évite les signaux opposés simultanés
- **Buffering intelligent** : Gestion optimisée des signaux entrants

## Architecture (Version Simplifiée v2.0)

### Composants principaux
- `main_simple.py` : Service principal ultra-simplifié
- `signal_aggregator_simple.py` : Moteur d'agrégation allégé
- `adaptive_consensus.py` : Consensus adaptatif par régime
- `context_manager.py` : Gestion du contexte de marché
- `critical_filters.py` : Filtres essentiels minimalistes
- `signal_buffer.py` : Buffer intelligent des signaux

### Flux de données
```
Signaux multiples → Signal Aggregator → Consensus → Filtres → Signal consolidé
```

## Configuration

### Variables d'environnement requises
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` : Configuration PostgreSQL
- `DEBUG_LOGS` : Active les logs détaillés
- `REDIS_URL` : URL Redis pour la communication

### Ports
- **5013** : Port Docker (externe)
- **8080** : Port interne pour health checks

## Endpoints API

### Health Check
```http
GET /health
```
Retourne le statut du service et résumé des performances.

### Statistiques détaillées
```http
GET /stats
```
Métriques complètes d'agrégation et de performance.

## Fonctionnalités Simplifiées v2.0

### ✅ Features Actives
- **Consensus adaptatif** : Adaptation aux régimes de marché
- **Filtres critiques** : Maximum 4 filtres essentiels
- **Buffer intelligent** : Gestion optimisée des flux
- **Protection contradictions** : Anti-collision des signaux

### ❌ Features Supprimées (Optimisation)
- 23+ validators complexes
- Système hiérarchique
- Pouvoir de veto
- Scoring pondéré complexe

## Consensus Adaptatif

### Régimes de marché supportés
- **Trending** : Consensus basé sur la direction
- **Ranging** : Consensus sur les niveaux de support/résistance
- **Volatile** : Consensus renforcé pour la stabilité
- **Low Volume** : Consensus plus strict

### Logique de consensus
```python
# Exemple simplifié
if regime == "trending":
    consensus_threshold = 0.6  # 60% des stratégies
elif regime == "volatile":
    consensus_threshold = 0.75 # 75% pour plus de sûreté
```

## Filtres Critiques (Max 4)

### Filtres essentiels
1. **Contradiction Filter** : Évite BUY+SELL simultanés
2. **Volume Filter** : Vérifie le volume suffisant
3. **Spread Filter** : Contrôle l'écart bid-ask
4. **Timing Filter** : Évite les signaux trop fréquents

## Démarrage

### Avec Docker
```bash
docker-compose up signal_aggregator
```

### En développement
```bash
cd signal_aggregator/src
python main_simple.py
```

## Performance & Optimisation

### Améliorations v2.0
- **Réduction complexité** : 90% moins de code que v1.0
- **Performance** : 5x plus rapide sans validators lourds
- **Mémoire** : 70% moins d'utilisation RAM
- **Maintenance** : Code simplifié et compréhensible

### Métriques
- Signaux reçus par seconde
- Taux de validation (success rate)
- Temps de traitement moyen
- Couverture par symbole/timeframe

## Structure des signaux

### Format d'entrée
```json
{
    "signal_type": "BUY|SELL",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "strategy": "EMA_Cross_Strategy",
    "strength": 0.85,
    "price": 45000.0,
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### Format de sortie (consolidé)
```json
{
    "signal_type": "BUY|SELL",
    "symbol": "BTCUSDT",
    "consensus_strength": 0.78,
    "contributing_strategies": 5,
    "market_regime": "trending",
    "filters_passed": ["volume", "spread", "timing"],
    "price": 45000.0,
    "timestamp": "2024-01-01T12:00:00Z"
}
```

## Monitoring

### Dashboard de santé
- Status : healthy/unhealthy
- Version et features actives
- Uptime et statistiques temps réel
- Taux de succès par filtre

### Logging
- Signaux traités et validés
- Consensus par régime de marché
- Performance des filtres
- Erreurs et alertes

## Gestion d'erreurs
- Continuité malgré les pannes de stratégies individuelles
- Fallback sur consensus minimal si données insuffisantes
- Logs détaillés pour le debugging
- Health checks constants

## Dépendances
- **PostgreSQL** : Stockage des signaux et contexte
- **Redis** : Communication temps réel
- **Analyzer** : Source des signaux
- **asyncio** : Architecture asynchrone

## Architecture technique
```
SimpleSignalAggregatorApp
├── ContextManager (régimes marché)
├── AdaptiveConsensus (fusion intelligente)
├── CriticalFilters (validation minimale)
└── SignalBuffer (gestion flux)
```

## Migration v1.0 → v2.0

### Changements majeurs
- Suppression des validators complexes
- Simplification du consensus
- Réduction à 4 filtres essentiels
- Architecture asynchrone native
- API REST simplifiée