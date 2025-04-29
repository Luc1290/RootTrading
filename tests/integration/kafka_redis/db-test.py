import sys
import os
from dotenv import load_dotenv
import psycopg2

# Ajouter le répertoire racine au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Charger les variables d'environnement
load_dotenv()

from shared.src.config import get_db_url

def test_db_connection():
    """Teste la connexion à la base de données PostgreSQL."""
    db_url = get_db_url()
    print(f"Tentative de connexion à la base de données avec URL: {db_url}")
    
    try:
        # Essayer de se connecter à la base de données
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # Vérifier si les tables existent
        print("Liste des tables dans la base de données:")
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        
        tables = cur.fetchall()
        if not tables:
            print("Aucune table trouvée dans la base de données!")
        else:
            for table in tables:
                print(f" - {table[0]}")
        
        # Fermer la connexion
        cur.close()
        conn.close()
        print("✅ Connexion à la base de données réussie!")
        return True
    
    except Exception as e:
        print(f"❌ Erreur de connexion à la base de données: {str(e)}")
        return False

if __name__ == "__main__":
    test_db_connection()