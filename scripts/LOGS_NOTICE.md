# 📊 ROOT Trading - Guide d'Analyse des Logs

## 🚀 Script Principal : `root_logs.py`

Script unifié pour analyser tous les logs Docker de ROOT Trading avec affichage coloré et statistiques détaillées.

### Installation
```bash
cd /mnt/e/RootTrading/RootTrading
chmod +x scripts/root_logs.py
```

### Commandes Principales

#### 1. 📈 Analyser une Crypto
```bash
# Analyse complète de SOLUSDC
python scripts/root_logs.py crypto SOL

# Avec plus de contexte (10000 lignes)
python scripts/root_logs.py crypto BTC -t 10000

# Autres cryptos disponibles
python scripts/root_logs.py crypto ETH
python scripts/root_logs.py crypto LINK
python scripts/root_logs.py crypto AVAX
```

**Informations affichées :**
- ✅ Nombre de signaux BUY
- ❌ Nombre de signaux SELL  
- 📊 Ratio BUY/SELL
- 📈 Statistiques de confidence (min/moy/max)
- ⏱️ Timeframes analysés
- 🎯 Top 5 stratégies actives
- 📝 Derniers signaux avec détails

#### 2. ⚠️ Voir les Erreurs
```bash
# Toutes les erreurs récentes
python scripts/root_logs.py errors

# Plus d'erreurs (500 lignes analysées)
python scripts/root_logs.py errors -t 5000

# Erreurs d'un service spécifique
python scripts/root_logs.py errors -s analyzer -t 1000
python scripts/root_logs.py errors -s trader
python scripts/root_logs.py errors -s coordinator
```

#### 3. 🔍 Rechercher un Pattern
```bash
# Rechercher toutes les high confidence (>0.8)
python scripts/root_logs.py search "confidence.*0\.[89]"

# Rechercher une stratégie spécifique
python scripts/root_logs.py search "MACD.*Strategy"

# Rechercher les signaux acceptés
python scripts/root_logs.py search "accepté"

# Rechercher avec plus de contexte
python scripts/root_logs.py search "breakout" -t 5000
```

#### 4. 📡 Suivre en Temps Réel
```bash
# Suivre tous les logs
python scripts/root_logs.py follow

# Suivre une crypto spécifique
python scripts/root_logs.py follow -p SOLUSDC

# Suivre les erreurs en temps réel
python scripts/root_logs.py follow -p "error|warning"

# Suivre un service spécifique
python scripts/root_logs.py follow -s trader
python scripts/root_logs.py follow -s analyzer -p "Signal BUY"
```

### 🎨 Code Couleur
- 🟢 **VERT** : Signaux BUY, messages positifs
- 🔴 **ROUGE** : Signaux SELL, erreurs
- 🟡 **JAUNE** : Warnings, patterns trouvés
- 🔵 **BLEU** : Informations, statistiques
- 🟣 **VIOLET** : Métriques de confidence
- 🟦 **CYAN** : Titres, timeframes

### 📝 Exemples d'Utilisation Avancée

#### Analyse Comparative
```bash
# Comparer plusieurs cryptos
for crypto in BTC ETH SOL LINK; do
    echo "=== $crypto ==="
    python scripts/root_logs.py crypto $crypto -t 1000 | grep -E "BUY:|SELL:|Ratio"
done
```

#### Export vers Fichier
```bash
# Sauvegarder l'analyse
python scripts/root_logs.py crypto SOL -t 10000 > analyse_sol_$(date +%Y%m%d).txt

# Sauvegarder les erreurs
python scripts/root_logs.py errors -t 5000 > errors_$(date +%Y%m%d).txt
```

#### Monitoring Continu
```bash
# Dans un terminal, suivre les signaux
python scripts/root_logs.py follow -p "Signal.*généré"

# Dans un autre, suivre les erreurs
python scripts/root_logs.py follow -p "error|exception"
```

### 🔧 Scripts Complémentaires

#### `analyze_crypto_logs.py`
Script spécialisé pour l'analyse approfondie d'une crypto avec contexte étendu.
```bash
python scripts/analyze_crypto_logs.py SOL 5000
```

#### `quick_log_search.py` 
Script léger pour recherches rapides.
```bash
python scripts/quick_log_search.py crypto SOL
python scripts/quick_log_search.py errors
python scripts/quick_log_search.py list
```

#### `search_logs.sh`
Script bash interactif avec menu.
```bash
./scripts/search_logs.sh
```

### 💡 Tips & Astuces

1. **Performance** : Pour les analyses lourdes, limitez le nombre de lignes avec `-t`
2. **Filtrage** : Combinez avec `grep` pour affiner les résultats
3. **Monitoring** : Ouvrez plusieurs terminaux pour suivre différents aspects
4. **Historique** : Les logs Docker sont limités, analysez régulièrement

### 🐛 Troubleshooting

**Problème : "Permission denied"**
```bash
chmod +x scripts/*.py scripts/*.sh
```

**Problème : "Container not found"**
```bash
docker ps  # Vérifier que les containers sont actifs
docker-compose up -d  # Redémarrer si nécessaire
```

**Problème : Encodage sur Windows**
- Les scripts gèrent automatiquement l'UTF-8
- Si problème persiste, utilisez PowerShell ou WSL

### 📊 Comprendre les Logs

#### Structure typique d'un log
```
2025-08-04 13:35:05,826 - analyzer.Strategy_Name - LEVEL - Message
```

#### Signaux importants
- `Signal BUY/SELL généré` : Un signal de trading
- `confidence=X.XX` : Niveau de confiance (0.0 à 1.0)
- `Cycle déjà actif` : Position déjà ouverte
- `Signal accepté` : Signal validé et exécuté

#### Services principaux
- **analyzer** : Génère les signaux de trading
- **coordinator** : Coordonne et valide les signaux
- **trader** : Exécute les trades
- **signal_aggregator** : Agrège les signaux multiples
- **market_analyzer** : Analyse le marché global

---

📅 *Dernière mise à jour : 04/08/2025*
🤖 *ROOT Trading v1.0.9.83*