#!/usr/bin/env python3
"""
Script pour appliquer les migrations de base de données
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import glob
import sys

def get_db_connection():
    """Créer une connexion à la base de données"""
    connection_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'trading'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    try:
        conn = psycopg2.connect(**connection_params)
        return conn
    except psycopg2.Error as e:
        print(f"❌ Erreur de connexion à la base de données: {e}")
        sys.exit(1)

def create_migrations_table(cursor):
    """Créer la table de suivi des migrations si elle n'existe pas"""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(255) PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL DEFAULT NOW(),
            checksum VARCHAR(64)
        );
    """)

def get_applied_migrations(cursor):
    """Récupérer la liste des migrations déjà appliquées"""
    cursor.execute("SELECT version FROM schema_migrations ORDER BY version;")
    return {row['version'] for row in cursor.fetchall()}

def calculate_checksum(content):
    """Calculer un checksum du contenu du fichier"""
    import hashlib
    return hashlib.sha256(content.encode()).hexdigest()

def apply_migration(cursor, migration_file, version):
    """Appliquer une migration spécifique"""
    try:
        with open(migration_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Calculer le checksum
        checksum = calculate_checksum(content)
        
        # Exécuter la migration
        cursor.execute(content)
        
        # Enregistrer dans la table des migrations
        cursor.execute(
            "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s);",
            (version, checksum)
        )
        
        print(f"✅ Migration appliquée: {version}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'application de la migration {version}: {e}")
        return False

def main():
    """Appliquer toutes les migrations en attente"""
    print("🔄 Début de l'application des migrations...")
    
    # Connexion à la base de données
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Créer la table de migrations
        create_migrations_table(cursor)
        conn.commit()
        
        # Récupérer les migrations appliquées
        applied_migrations = get_applied_migrations(cursor)
        
        # Trouver tous les fichiers de migration
        migrations_dir = os.path.dirname(os.path.abspath(__file__)) + "/migrations"
        migration_files = glob.glob(os.path.join(migrations_dir, "*.sql"))
        migration_files.sort()
        
        migrations_applied = 0
        
        for migration_file in migration_files:
            # Extraire le nom de version du fichier
            version = os.path.basename(migration_file)
            
            if version not in applied_migrations:
                print(f"📋 Application de la migration: {version}")
                
                if apply_migration(cursor, migration_file, version):
                    conn.commit()
                    migrations_applied += 1
                else:
                    conn.rollback()
                    print(f"❌ Échec de la migration {version}, arrêt du processus")
                    break
            else:
                print(f"⏭️  Migration déjà appliquée: {version}")
        
        if migrations_applied > 0:
            print(f"✅ {migrations_applied} migration(s) appliquée(s) avec succès")
        else:
            print("✅ Aucune migration en attente")
            
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        conn.rollback()
        sys.exit(1)
        
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()