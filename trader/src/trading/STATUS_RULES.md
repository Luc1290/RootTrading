# Règles de gestion des statuts de cycles

## CycleStatus.CANCELED
Utilisé pour les annulations **volontaires** :
- Annulation manuelle via `cancel_cycle()`
- Annulation par l'utilisateur via l'API
- Annulation pour raison stratégique (ex: conditions de marché changées)

## CycleStatus.FAILED  
Utilisé pour les échecs **techniques** :
- Ordre rejeté par Binance (REJECTED)
- Ordre introuvable sur Binance (fantôme)
- Fonds insuffisants
- Erreur de connexion/API
- Ordre annulé automatiquement par Binance (CANCELED)
- Dépassement de délai (TTL expiré)
- Cycle sans prix cible
- Cycle sans ordre d'entrée/sortie

## Transitions possibles
- INITIATING → WAITING_SELL/BUY (ordre d'entrée exécuté)
- WAITING_* → COMPLETED (ordre de sortie exécuté)
- WAITING_* → CANCELED (annulation volontaire)
- WAITING_* → FAILED (problème technique)
- Tout statut → FAILED (erreur critique)

## Gestion des ordres Binance
- OrderStatus.NEW → Ordre en attente
- OrderStatus.FILLED → Ordre exécuté → Cycle continue ou COMPLETED
- OrderStatus.PARTIALLY_FILLED → Ordre partiellement exécuté
- OrderStatus.CANCELED → Ordre annulé → Cycle FAILED (sauf si annulation volontaire)
- OrderStatus.REJECTED → Ordre rejeté → Cycle FAILED
- OrderStatus.EXPIRED → Ordre expiré → Cycle FAILED