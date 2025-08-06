# 📁 Guide d'Export des Logs ROOT Trading

## 🔧 Configuration Windows

### Problème d'encodage
Sur Windows, l'export de logs peut générer des erreurs d'encodage à cause des emojis et caractères spéciaux.

### Solutions

#### 1. Utiliser la variable d'environnement (Recommandé)
```powershell
# Pour la session courante
$env:PYTHONIOENCODING="utf-8"

# Pour toutes les sessions (permanent)
[System.Environment]::SetEnvironmentVariable('PYTHONIOENCODING','utf-8','User')
```

#### 2. Utiliser le flag Python
```powershell
python -X utf8 scripts/root_logs_export.py [commande]
```

## 📊 Scripts d'Export

### `root_logs_export.py`
Version simplifiée sans emojis ni couleurs, optimisée pour l'export vers fichiers.

### Commandes Principales

#### 1. Export des Erreurs
```powershell
# Export simple
python scripts/root_logs_export.py errors -t 5000 > errors.txt

# Export avec date
python scripts/root_logs_export.py errors -t 5000 > "errors_$(Get-Date -Format 'yyyyMMdd').txt"

# Export d'un service spécifique
python scripts/root_logs_export.py errors -s analyzer -t 5000 > errors_analyzer.txt
```

#### 2. Export d'Analyse Crypto
```powershell
# Analyse SOL
python scripts/root_logs_export.py crypto SOL -t 10000 > "analyse_sol_$(Get-Date -Format 'yyyyMMdd').txt"

# Analyse BTC avec plus de contexte
python scripts/root_logs_export.py crypto BTC -t 50000 > analyse_btc.txt
```

#### 3. Export de Recherche
```powershell
# Recherche de patterns
python scripts/root_logs_export.py search "Signal accepté" -t 10000 > signaux_acceptes.txt

# Recherche de positions
python scripts/root_logs_export.py search "position_" -t 5000 > positions.txt

# Recherche d'erreurs critiques
python scripts/root_logs_export.py search "CRITICAL|FATAL" -t 10000 > erreurs_critiques.txt
```

## 🗓️ Exports Automatisés

### Script Batch Windows
Le fichier `export_logs.bat` permet un export rapide :
```batch
.\scripts\export_logs.bat
```

### PowerShell - Export Quotidien
```powershell
# Créer un dossier pour les logs
New-Item -ItemType Directory -Force -Path ".\logs_exports\$(Get-Date -Format 'yyyy-MM-dd')"

# Export complet
$date = Get-Date -Format 'yyyyMMdd'
$folder = ".\logs_exports\$(Get-Date -Format 'yyyy-MM-dd')"

# Erreurs
python scripts/root_logs_export.py errors -t 10000 > "$folder\errors_$date.txt"

# Analyse des principales cryptos
@("BTC", "ETH", "SOL", "LINK", "AVAX") | ForEach-Object {
    python scripts/root_logs_export.py crypto $_ -t 5000 > "$folder\analyse_$($_)_$date.txt"
}

# Signaux et positions
python scripts/root_logs_export.py search "Signal" -t 5000 > "$folder\signaux_$date.txt"
python scripts/root_logs_export.py search "position_" -t 2000 > "$folder\positions_$date.txt"
```

## 📈 Analyse des Exports

### Avec PowerShell
```powershell
# Compter les erreurs par type
Select-String -Pattern "ERROR|WARNING|CRITICAL" errors_20250804.txt | Group-Object Pattern | Select-Object Count, Name

# Chercher les cryptos avec le plus de signaux
Select-String -Pattern "[A-Z]+USDC" analyse_sol_20250804.txt | Group-Object Matches | Sort-Object Count -Descending
```

### Avec Python
```python
# Analyser un fichier exporté
with open('errors_20250804.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
errors = [l for l in lines if 'ERROR' in l]
warnings = [l for l in lines if 'WARNING' in l]

print(f"Erreurs: {len(errors)}")
print(f"Warnings: {len(warnings)}")
```

## 🚀 Exports Utiles

### 1. Rapport Quotidien Complet
```powershell
$env:PYTHONIOENCODING="utf-8"
$date = Get-Date -Format 'yyyyMMdd_HHmm'

# Créer un rapport complet
@"
=== RAPPORT ROOT TRADING - $date ===

--- ERREURS RECENTES ---
"@ > "rapport_$date.txt"

python scripts/root_logs_export.py errors -t 1000 >> "rapport_$date.txt"

@"

--- ANALYSE SOL ---
"@ >> "rapport_$date.txt"

python scripts/root_logs_export.py crypto SOL -t 2000 >> "rapport_$date.txt"

@"

--- POSITIONS ACTIVES ---
"@ >> "rapport_$date.txt"

python scripts/root_logs_export.py search "Cycle déjà actif" -t 500 >> "rapport_$date.txt"
```

### 2. Export pour Excel
```powershell
# Export CSV-like pour analyse Excel
python scripts/root_logs_export.py search "Signal.*généré" -t 10000 | 
    Select-String -Pattern "(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Signal (\w+).*([A-Z]+USDC).*confidence[=:](\d+\.\d+)" |
    ForEach-Object { "$($_.Matches[0].Groups[1].Value);$($_.Matches[0].Groups[2].Value);$($_.Matches[0].Groups[3].Value);$($_.Matches[0].Groups[4].Value)" } > signaux.csv
```

## 🛠️ Troubleshooting

### Erreur "UnicodeEncodeError"
```powershell
# Solution 1
$env:PYTHONIOENCODING="utf-8"

# Solution 2
chcp 65001  # Change le code page vers UTF-8

# Solution 3
python -X utf8 scripts/root_logs_export.py [commande]
```

### Fichier vide
- Vérifier que les containers sont actifs : `docker ps`
- Augmenter le paramètre `-t` pour chercher plus loin
- Vérifier le pattern de recherche (sensible à la casse)

### Performance
- Pour de gros exports (>50000 lignes), prévoir du temps
- Diviser les exports par service si nécessaire
- Utiliser des patterns précis pour réduire les résultats

---

📅 *Guide créé le 04/08/2025*
🤖 *ROOT Trading v1.0.9.83*