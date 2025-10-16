# Configuration WSL - Stockage sur E: au lieu de C:

## Le fichier `nul` n'est PAS la solution !

Le fichier `nul` est un device Windows (équivalent de `/dev/null`), il ne configure rien pour WSL.

## ✅ Solution correcte : Déplacer WSL vers E:

### 1. Créer .wslconfig dans votre profil Windows
Fichier : `C:\Users\<VotreNom>\.wslconfig`

```ini
[wsl2]
# Déplacer le swap file vers E:
swapFile=E:\\temp\\wsl-swap.vhdx

# Limiter la mémoire (optionnel)
memory=8GB

# Limiter les processeurs (optionnel)
processors=4

# Désactiver la localisation par défaut sur C:
localhostForwarding=true
```

### 2. Exporter et réimporter votre distribution WSL

Dans PowerShell (Admin):

```powershell
# 1. Voir vos distributions
wsl --list --verbose

# 2. Exporter (exemple avec Ubuntu)
wsl --export Ubuntu E:\WSL\ubuntu-backup.tar

# 3. Désenregistrer l'ancienne
wsl --unregister Ubuntu

# 4. Réimporter vers E:
wsl --import Ubuntu E:\WSL\Ubuntu E:\WSL\ubuntu-backup.tar --version 2

# 5. Définir comme distro par défaut
wsl --set-default Ubuntu
```

### 3. Vérifier que c'est sur E:

```powershell
# Voir l'emplacement du VHDX
wsl --list --verbose
# Le VHDX sera maintenant dans E:\WSL\Ubuntu\ext4.vhdx
```

### 4. Libérer l'espace sur C:

```powershell
# Supprimer l'ancien VHDX sur C:
# Il était dans: C:\Users\<VotreNom>\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu*\LocalState\ext4.vhdx
```

## 🎯 Résultat

- ✅ VHDX WSL sur E: au lieu de C:
- ✅ Swap file sur E:
- ✅ C: libéré
- ✅ Performances identiques

## 📝 Notes

- Le fichier `nul` dans ce repo n'a RIEN à voir avec cette config
- `nul` est probablement utilisé pour rediriger des sorties dans des scripts
- La vraie config WSL se fait via `.wslconfig` + `wsl --import`
