# 📈 RootTrading - Guide Complet & Session Recap

## 🎯 Vue d'ensemble du système

**RootTrading** est un bot de trading crypto automatisé (SPOT uniquement) avec cycles BUY optimisés, trailing stop intelligent, allocation dynamique et consensus override pour les signaux forts.

### Architecture principale
```
📊 Analyzer → 🤖 Signal Aggregator → 🎯 Coordinator → 💰 Trader → 📋 Portfolio
                     ↓
                 🗄️ Database (PostgreSQL) + 🔄 Redis
```

---

## 🔧 Corrections importantes de cette session

### 1. **Fix Consensus Override (CRITIQUE)**
**Problème** : Les signaux forts (COMP, SHIB) étaient rejetés car vérification univers AVANT consensus override.

**Solution** : Déplacé la vérification consensus override AVANT `_check_feasibility()` dans `coordinator.py`
```python
# AVANT la faisabilité, vérifier consensus fort
if signal.side == OrderSide.BUY:
    signal_force, strategy_count, avg_confidence = self._calculate_unified_signal_strength(signal)
    if signal_force >= min_force and strategy_count >= min_strategies:
        # Forcer l'ajout à l'univers AVANT rejet
        self.universe_manager.force_pair_selection(signal.symbol, duration_minutes=45)
```

### 2. **Système de libération USDC intelligent**
**Problème** : Positions ridicules quand USDC faible (6 USDC au lieu de 48 USDC au début).

**Solution** : Auto-liquidation de la pire position EN PERTE seulement
```python
# Si USDC insuffisant ET positions en perte → vendre la pire
if all_positions_positive:
    logger.info("💚 Toutes gagnantes - Pas de vente auto")
    return 0.0
else:
    # Vendre position la plus négative uniquement
```

### 3. **Script signaux manuels optimisé**
**Fichier** : `debug/send_manual_buy_signals.py`
**Structure exacte** pour bypass univers :
```python
{
    "strategy": "CONSENSUS",  # Obligatoire
    "metadata": {
        "strategies_count": 6,     # ≥ 5 pour override
        "consensus_strength": 5.0,  # ≥ 2.0 pour override
        "type": "CONSENSUS"
    }
}
```

### 4. **Trailing Sell v2.0 - Optimisation Scalp**

**Problème** : Activation trailing à 2.0% trop tardive, pas de breakeven protection, marges non adaptées au scalp.

**Solutions appliquées** :
```python
# Activation trailing abaissée à 1.5% (scalp)
activate_trailing_gain = max(0.015, 0.8 * atr_percent)

# Breakeven intelligent multi-niveaux
if gain_percent >= 0.020:  # +2.0%
    breakeven_price = entry_price * 1.002  # Entry + 0.2%
elif gain_percent >= 0.012:  # +1.2%
    breakeven_price = entry_price * (1 + 2 * fee_percent)  # Entry + fees

# Marges adaptatives selon palier
# ≥8%: 0.4%, 5-8%: 0.6%, 3-5%: 0.8%, 2-3%: 1.0%

# Protection avancée
- Clés Redis par position_id (évite collisions)
- TTL 7 jours avec refresh auto
- Pump rider: >5% + vitesse <10min
- TP progressif inversé (strict gros gains)
```

**Impact** :
- Protection plus précoce (+1.5% vs +2.0%)
- Breakeven dès +1.2% évite retours en rouge
- Marges serrées sur gros gains (0.4% sur +8%)
- SL base réduit à 1.2% (scalp rapide)

---

## 🗄️ Base de données - Commandes essentielles

### **Connexion rapide**
```bash
docker exec roottrading-db-1 psql -U postgres -d trading
```

### **Tables principales**
```sql
-- Signaux de trading
SELECT COUNT(*), side FROM trading_signals
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY side;

-- Cycles de trading actifs
SELECT symbol, status, entry_price,
       (SELECT current_price FROM market_data WHERE symbol = tc.symbol ORDER BY timestamp DESC LIMIT 1) as current_price,
       created_at
FROM trade_cycles tc
WHERE status = 'active_buy'
ORDER BY created_at DESC;

-- Performance des positions
SELECT symbol, entry_price,
       ROUND(((current_price - entry_price) / entry_price * 100)::numeric, 2) as performance_pct,
       ROUND((quantity * current_price)::numeric, 2) as value_usdc
FROM positions
WHERE status = 'open'
ORDER BY performance_pct DESC;

-- Top signaux par crypto (dernières 6h)
SELECT symbol, side, COUNT(*) as signals, AVG(confidence) as avg_conf
FROM trading_signals
WHERE created_at > NOW() - INTERVAL '6 hours'
GROUP BY symbol, side
HAVING COUNT(*) > 5
ORDER BY signals DESC, avg_conf DESC;
```

### **Nettoyage historique**
```sql
-- Supprimer signaux avant date
DELETE FROM trading_signals WHERE created_at < '2025-09-16';
DELETE FROM trade_cycles WHERE created_at < '2025-09-16';
```

---

## 📊 Analyse des cryptos & indicateurs

### **Portfolio actuel**
```bash
curl -s http://localhost:8000/summary | python3 -m json.tool
```

### **Symboles tradés**
```bash
curl -s http://localhost:8000/symbols/traded
```

### **Positions récentes**
```bash
curl -s http://localhost:8000/positions/recent?hours=24
```

### **Analyse technique via DB**
```sql
-- Régimes de marché actuels
SELECT symbol, market_regime, regime_strength, regime_confidence,
       momentum_score, rsi_14, adx_14
FROM analyzer_data
WHERE timeframe = '15m'
  AND time = (SELECT MAX(time) FROM analyzer_data WHERE timeframe = '15m')
ORDER BY momentum_score DESC;

-- Cryptos en TRENDING_BULL
SELECT symbol, momentum_score, confluence_score, volume_quality_score
FROM analyzer_data
WHERE timeframe = '15m'
  AND market_regime = 'TRENDING_BULL'
  AND time > NOW() - INTERVAL '1 hour'
ORDER BY momentum_score DESC;

-- Détection de breakouts
SELECT symbol, market_regime, regime_confidence,
       volume_ratio, break_probability
FROM analyzer_data
WHERE market_regime LIKE '%BREAKOUT%'
  AND timeframe = '15m'
  AND time > NOW() - INTERVAL '2 hours'
ORDER BY break_probability DESC;
```

### **Monitoring Trailing Sell**

```sql
-- Positions avec trailing actif (gain ≥1.5%)
SELECT tc.symbol,
       tc.entry_price,
       md.close as current_price,
       ROUND(((md.close - tc.entry_price) / tc.entry_price * 100)::numeric, 2) as gain_pct,
       tc.created_at
FROM trade_cycles tc
JOIN LATERAL (
    SELECT close FROM market_data
    WHERE symbol = tc.symbol
    ORDER BY time DESC LIMIT 1
) md ON true
WHERE tc.status = 'active_buy'
  AND (md.close - tc.entry_price) / tc.entry_price >= 0.015
ORDER BY gain_pct DESC;

-- Positions en breakeven zone (+1.2% à +2.0%)
SELECT tc.symbol,
       tc.entry_price,
       md.close as current_price,
       ROUND(((md.close - tc.entry_price) / tc.entry_price * 100)::numeric, 2) as gain_pct
FROM trade_cycles tc
JOIN LATERAL (
    SELECT close FROM market_data
    WHERE symbol = tc.symbol
    ORDER BY time DESC LIMIT 1
) md ON true
WHERE tc.status = 'active_buy'
  AND (md.close - tc.entry_price) / tc.entry_price BETWEEN 0.012 AND 0.020
ORDER BY gain_pct DESC;
```

---

## 🎯 Recherche d'opportunités

### **1. Signaux consensus forts (recommandé)**
```sql
-- Signaux avec consensus récent
SELECT symbol, side, COUNT(*) as signal_count,
       AVG(confidence) as avg_confidence,
       MAX(created_at) as last_signal
FROM trading_signals
WHERE created_at > NOW() - INTERVAL '12 hours'
  AND side = 'BUY'
GROUP BY symbol, side
HAVING COUNT(*) > 10 AND AVG(confidence) > 0.85
ORDER BY signal_count DESC, avg_confidence DESC;
```

### **2. Analyse momentum + volume**
```sql
-- Top momentum avec volume confirmé
SELECT ad.symbol, ad.momentum_score, ad.volume_ratio,
       ad.confluence_score, ad.market_regime,
       ts.signal_count
FROM analyzer_data ad
LEFT JOIN (
    SELECT symbol, COUNT(*) as signal_count
    FROM trading_signals
    WHERE created_at > NOW() - INTERVAL '6 hours' AND side = 'BUY'
    GROUP BY symbol
) ts ON ad.symbol = ts.symbol
WHERE ad.timeframe = '15m'
  AND ad.time > NOW() - INTERVAL '2 hours'
  AND ad.momentum_score > 50
  AND ad.volume_ratio > 2.0
ORDER BY ad.momentum_score DESC, ts.signal_count DESC NULLS LAST;
```

### **3. Cryptos sous-pondérées dans le portfolio**
```python
# Script Python pour identifier les opportunités
import requests

# Portfolio actuel
portfolio = requests.get('http://localhost:8000/summary').json()
current_holdings = {b['asset']: b['value_usdc'] for b in portfolio['balances'] if b['value_usdc'] > 1}

# Symboles tradables
tradable = requests.get('http://localhost:8000/symbols/traded').json()
missing_assets = [s['asset'] for s in tradable if s['asset'] not in current_holdings]

print("Cryptos absentes du portfolio mais tradables:", missing_assets)
```

---

## 🚀 Actions fréquentes

### **Envoyer signal manuel fort**
```bash
# Modifier les cryptos dans le script puis :
python3 debug/send_manual_buy_signals.py
```

### **Forcer achat d'une crypto spécifique**
1. Modifier `debug/send_manual_buy_signals.py` :
```python
buy_opportunities = [
    {
        "symbol": "SUIUSDC",  # Votre crypto
        "current_price": 1.85,  # Prix actuel
        "confidence": 0.95,
        "reason": "Opportunité identifiée manuellement"
    }
]
```
2. Exécuter : `python3 debug/send_manual_buy_signals.py`

### **Monitoring en temps réel**
```bash
# Logs coordinator (signaux reçus)
docker logs roottrading-coordinator-1 --tail=50 -f

# Logs trader (ordres exécutés)
docker logs roottrading-trader-1 --tail=30 -f

# Portfolio live
watch -n 10 'curl -s http://localhost:8000/summary | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(f\"Total: {d[\"total_value\"]:.0f} USDC\"); [print(f\"{b[\"asset\"]:6} {b[\"value_usdc\"]:8.2f} USDC ({((b[\"value_usdc\"]/d[\"total_value\"])*100):5.1f}%)\") for b in sorted(d[\"balances\"], key=lambda x: x[\"value_usdc\"], reverse=True) if b[\"value_usdc\"] > 5]"'
```

---

## 🔄 Restart & Rebuild

### **Restart services**
```bash
docker restart roottrading-coordinator-1
docker restart roottrading-trader-1
```

### **Rebuild après modifications**
```bash
docker-compose down
docker-compose up --build -d
```

---

## 📈 Évaluation de cryptos - Checklist

Quand tu veux évaluer une crypto, voici l'ordre optimal :

### **1. Signaux récents (urgent)**
```sql
SELECT COUNT(*) as signals_6h, AVG(confidence) as avg_conf
FROM trading_signals
WHERE symbol = 'XXXUSDC'
  AND created_at > NOW() - INTERVAL '6 hours'
  AND side = 'BUY';
```

### **2. Analyse technique (tendance)**
```sql
SELECT market_regime, regime_strength, momentum_score,
       rsi_14, volume_ratio, confluence_score
FROM analyzer_data
WHERE symbol = 'XXXUSDC'
  AND timeframe = '15m'
ORDER BY time DESC LIMIT 1;
```

### **3. Position dans portfolio**
```bash
curl -s http://localhost:8000/summary | grep -A5 -B5 "XXX"
```

### **4. Cycles de trading histoire**
```sql
SELECT status, entry_price,
       CASE WHEN status = 'completed'
            THEN ROUND(((exit_price - entry_price) / entry_price * 100)::numeric, 2)
            ELSE NULL
       END as performance_pct,
       created_at
FROM trade_cycles
WHERE symbol = 'XXXUSDC'
ORDER BY created_at DESC LIMIT 10;
```

---

## 🎪 Résultats des sessions

### **Session 17/09/2025 - Consensus Override & Auto-liquidation**

**Trades exécutés avec succès** :
- ✅ **COMPUSDC** : 0.134 COMP = 6.08 USDC (STRONG signal, force 11.51)
- ✅ **SHIBUSDC** : 460,813 SHIB = 6.11 USDC (STRONG signal, force 11.02)
- ✅ **BTCUSDC** : 0.00005612 BTC = 6.50 USDC (MODERATE signal, force 10.78)

**Améliorations système** :
1. Consensus override opérationnel
2. Auto-liquidation intelligente (positions négatives seulement)
3. Scripts manuels optimisés
4. Documentation complète

### **Session 02/10/2025 - Optimisation Trailing Sell v2.0**

**Corrections appliquées** :

- ✅ **Activation trailing** : 2.0% → 1.5% (scalp optimisé)
- ✅ **Breakeven intelligent** : 2 niveaux (+1.2% et +2.0%)
- ✅ **Stop-loss base** : 1.6% → 1.2% (protection rapide)
- ✅ **Time factor corrigé** : strict récent (0.8), tolérant ancien (1.2)
- ✅ **Régime bear corrigé** : tolérant (1.1) pour reversals
- ✅ **TP tolerance inversé** : strict gros gains (0.85), permissif petits (0.70)
- ✅ **Pump rider amélioré** : 5% + vitesse <10min
- ✅ **TTL Redis** : 7 jours avec refresh
- ✅ **Clés Redis** : namespace par position_id

**Positions actives testées** :

- 🟢 **BTCUSDC** : Entry 116,876 → Max 119,734 (+2.45%) - Trailing armé, marge 1.0%
- 🟢 **XRPUSDC** : Entry 2.9145 → Max 2.9806 (+2.27%) - Trailing armé, marge 1.0%
- 🟡 **ETHUSDC** : Entry 4,403 → +0.24% - Pas de trailing (<1.5%)
- 🟡 **SOLUSDC** : Entry 225.88 → +0.04% - Pas de trailing (<1.5%)

**Documentation mise à jour** :

- ✅ README : 28 stratégies documentées + section Trailing Sell
- ✅ Gestion risques actualisée (SL 1.2-1.8%, trailing 1.5%)
- ✅ Version v1.0.9.908

---

## 🆘 Dépannage fréquent

### **Signal rejeté "pas dans univers"**
→ Vérifier que `consensus_strength >= 2.0` et `strategies_count >= 5`

### **Positions trop petites**
→ Vérifier USDC disponible, sinon auto-liquidation se déclenchera

### **Pas de signaux générés**
→ `docker logs roottrading-signal-aggregator-1` pour voir les rejets

### **Prix/données manquantes**

→ Redémarrer `roottrading-analyzer-1` qui alimente les prix

### **Trailing sell ne se déclenche pas**

→ Vérifier gain ≥ 1.5% : `SELECT ((current_price - entry_price) / entry_price) FROM ...`
→ Logs : `docker logs roottrading-coordinator-1 | grep "TRAILING ADAPTATIF"`

### **Position revenue en rouge après être montée**

→ Normal si gain <1.2% (pas de breakeven armé)
→ À partir de +1.2%, breakeven protège entry+fees
→ À partir de +2.0%, profit minimum garanti (entry+0.2%)

---

**Dernière mise à jour** : Session du 02/10/2025 - Trailing Sell v2.0 optimisé
