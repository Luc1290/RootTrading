#!/bin/bash

# Script pour appliquer les migrations SQL
# Ce script est exécuté automatiquement lors du démarrage du conteneur db

set -e

echo "🔄 Application des migrations..."

# Attendre que PostgreSQL soit prêt
until pg_isready -U ${POSTGRES_USER:-postgres} -h localhost; do
  echo "En attente de PostgreSQL..."
  sleep 2
done

# Créer la table de suivi des migrations si elle n'existe pas
psql -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-trading} <<EOF
CREATE TABLE IF NOT EXISTS migrations (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL UNIQUE,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
EOF

# Appliquer les migrations dans l'ordre
for migration in /docker-entrypoint-initdb.d/migrations/*.sql; do
    filename=$(basename "$migration")
    
    # Vérifier si la migration a déjà été appliquée
    applied=$(psql -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-trading} -tAc "SELECT COUNT(*) FROM migrations WHERE filename='$filename'")
    
    if [ "$applied" -eq "0" ]; then
        echo "🚀 Application de la migration: $filename"
        psql -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-trading} -f "$migration"
        
        # Enregistrer la migration comme appliquée
        psql -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-trading} -c "INSERT INTO migrations (filename) VALUES ('$filename')"
        echo "✅ Migration $filename appliquée avec succès"
    else
        echo "⏭️  Migration $filename déjà appliquée, ignorée"
    fi
done

echo "✅ Toutes les migrations ont été appliquées"