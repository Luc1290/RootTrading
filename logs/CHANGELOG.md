# Changelog - RootTrading

Ce fichier documente tous les changements importants apportés au système RootTrading.

## [26 Mai 2025] - Session de debug et corrections majeures

### 🐛 Bugs corrigés

#### 1. Cache obsolète du coordinator
**Problème** : Le coordinator gardait en cache de vieilles positions même quand la requête vers `/orders` échouait, causant des faux positifs sur les positions contradictoires.

**Solution** : 
- Fichier : `/coordinator/src/signal_handler.py`
- Modification : Vider le cache et mettre à jour `cache_update_time` en cas d'échec de requête
```python
if not response:
    logger.warning("Impossible de récupérer les cycles actifs")
    # En cas d'échec, vider le cache pour éviter d'utiliser des données obsolètes
    self.active_cycles_cache = {}
    self.cache_update_time = time.time()
    return
```

#### 2. Double déclenchement stop-loss et prix cible
**Problème** : Un même cycle pouvait déclencher à la fois son stop-loss ET son prix cible, causant des tentatives de fermeture en double.

**Solution** :
- Fichier : `/trader/src/trading/stop_manager.py`
- Modification : Utiliser un set pour éviter les doublons et prioriser le stop-loss
```python
# Déclencher les stops et targets (en évitant les doublons)
cycles_to_close = set()
close_reasons = {}

for cycle_id in stops_to_trigger:
    cycles_to_close.add(cycle_id)
    close_reasons[cycle_id] = "stop-loss"
    
for cycle_id in targets_to_trigger:
    if cycle_id not in cycles_to_close:
        cycles_to_close.add(cycle_id)
        close_reasons[cycle_id] = "target"
```

#### 3. Messages de log incorrects lors de la fermeture des cycles
**Problème** : Le message "Stop loss déclenché" s'affichait même quand c'était le prix cible qui était atteint.

**Solution** :
- Fichier : `/trader/src/trading/cycle_manager.py`
- Modification : Afficher le bon message selon la raison de fermeture
```python
if is_stop_loss:
    logger.info(f"🛑 Stop loss déclenché - Annulation de l'ordre limite {cycle.exit_order_id}")
else:
    logger.info(f"🎯 Prix cible atteint - Annulation de l'ordre limite {cycle.exit_order_id}")
```

#### 4. Bug d'annulation d'ordres avec IDs élevés
**Problème** : Les ordres Binance avec ID >= 10000000 étaient considérés comme des ordres de démo et n'étaient jamais annulés.

**Solutions** :
- Fichier : `/trader/src/exchange/binance_executor.py`
  - Suppression de la vérification `if order_id >= 10000000`
  - Utilisation de `self.demo_trades` pour vérifier si un ordre est démo

- Fichier : `/trader/src/exchange/binance_utils.py`
  - Suppression de 2 vérifications d'ID numérique dans `fetch_order_status()` et `cancel_order()`

#### 5. Cycles non chargés avec statut waiting_buy
**Problème** : La requête SQL dans `get_active_cycles()` excluait les cycles avec statut `waiting_buy`.

**Solution** :
- Fichier : `/trader/src/trading/cycle_repository.py`
- Modification : Utiliser LOWER() pour une comparaison insensible à la casse
```sql
where_clauses = [f"LOWER(status) NOT IN ({', '.join(['%s'] * len(terminal_statuses))})"]
```

#### 6. Réconciliation ignorant les vrais cycles
**Problème** : La réconciliation considérait tous les cycles avec order ID >= 10000000 comme démo.

**Solutions** :
- Fichier : `/trader/src/trading/reconciliation.py`
  - Suppression de la vérification dans `reconcile_cycle()`
  - Suppression de la vérification dans `_check_orphan_orders()`
  - Ajout de `AND demo = false` dans la requête SQL des ordres orphelins

### 🗄️ Nettoyage de la base de données

#### Marquage des cycles démo
- 90 cycles avec des order IDs de démo (10000000+) ont été marqués comme `demo = true`
- Requête exécutée :
```sql
UPDATE trade_cycles 
SET demo = true 
WHERE (entry_order_id::text LIKE '100000%' OR exit_order_id::text LIKE '100000%') 
AND demo = false;
```

### 📊 Résultats

Après toutes ces corrections :
- ✅ Plus d'erreurs 400 lors de la réconciliation
- ✅ Cache du coordinator toujours à jour
- ✅ Plus de double déclenchement de stops
- ✅ Messages de log corrects
- ✅ Tous les ordres peuvent être annulés correctement
- ✅ La réconciliation ne vérifie plus les ordres démo sur Binance

### 🔧 Services rebuild

Les services suivants ont été rebuild pendant cette session :
- `trader` : 4 fois
- `coordinator` : 1 fois

---

## [22 Mai 2025] - Corrections des contraintes Binance

### Implémentation des contraintes dynamiques
- Les contraintes Binance (min_qty, min_notional) sont maintenant récupérées en temps réel
- Plus de valeurs hardcodées

### Vraies contraintes découvertes
- BTCUSDC : min_notional = 5.0 USDC (pas 10.0)
- ETHUSDC : min_notional = 5.0 USDC (pas 10.0)
- ETHBTC : min_notional = 0.0001 BTC

---

## [21 Mai 2025] - Système de réconciliation automatique

### Nouveau système implémenté
- Fichier : `/trader/src/trading/reconciliation.py`
- Réconciliation automatique toutes les 10 minutes
- Détection et nettoyage des cycles fantômes
- Synchronisation avec l'état réel sur Binance