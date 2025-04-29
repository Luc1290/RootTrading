#!/usr/bin/env python3
"""
Script de validation pour vérifier l'intégrité du système RootTrading
avant le démarrage des conteneurs Docker.

Usage:
    python pre-start-validation.py [--skip-db] [--skip-redis] [--skip-kafka]

Options:
    --skip-db     : Ignorer les tests de connexion à la base de données
    --skip-redis  : Ignorer les tests de connexion à Redis
    --skip-kafka  : Ignorer les tests de connexion à Kafka
"""

import sys
import os
import argparse
import importlib.util
import subprocess
import time
from dotenv import load_dotenv

# Ajouter le répertoire racine au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Charger les variables d'environnement
load_dotenv()

class ValidationResult:
    """Classe pour stocker les résultats de validation."""
    def __init__(self):
        self.tests = {}
        self.warnings = []
        self.errors = []
    
    def add_test(self, name, passed, message=""):
        """Ajoute un résultat de test."""
        self.tests[name] = {"passed": passed, "message": message}
        
    def add_warning(self, message):
        """Ajoute un avertissement."""
        self.warnings.append(message)
        
    def add_error(self, message):
        """Ajoute une erreur."""
        self.errors.append(message)
        
    def all_passed(self):
        """Vérifie si tous les tests ont réussi."""
        return all(test["passed"] for test in self.tests.values())
    
    def summary(self):
        """Génère un résumé des tests."""
        total = len(self.tests)
        passed = sum(1 for test in self.tests.values() if test["passed"])
        
        # Catégorisation des résultats
        categories = {}
        for name, result in self.tests.items():
            category = name.split(':')[0]
            if category not in categories:
                categories[category] = {"total": 0, "passed": 0}
            
            categories[category]["total"] += 1
            if result["passed"]:
                categories[category]["passed"] += 1
        
        # Calculer le taux de réussite global
        success_rate = (passed / total * 100) if total > 0 else 0
        
        # Générer le résumé
        summary_text = f"\n=== RÉSUMÉ DES TESTS ({passed}/{total}, {success_rate:.1f}%) ===\n\n"
        
        # Résumé par catégorie
        for category, stats in categories.items():
            cat_success = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            status = "✅" if stats["passed"] == stats["total"] else "❌"
            summary_text += f"{status} {category}: {stats['passed']}/{stats['total']} ({cat_success:.1f}%)\n"
        
        # Détails des tests en échec
        if passed < total:
            summary_text += "\nTests en échec:\n"
            for name, result in self.tests.items():
                if not result["passed"]:
                    summary_text += f"❌ {name}"
                    if result["message"]:
                        summary_text += f": {result['message']}"
                    summary_text += "\n"
        
        # Avertissements
        if self.warnings:
            summary_text += "\nAvertissements:\n"
            for warning in self.warnings:
                summary_text += f"⚠️ {warning}\n"
        
        # Erreurs
        if self.errors:
            summary_text += "\nErreurs:\n"
            for error in self.errors:
                summary_text += f"🔴 {error}\n"
        
        # Recommandation finale
        if self.all_passed() and not self.errors:
            summary_text += "\n✅ Tous les tests ont réussi! Vous pouvez lancer les conteneurs Docker."
        else:
            summary_text += "\n⚠️ Certains tests ont échoué. Corrigez les problèmes avant de lancer Docker."
        
        return summary_text

def run_command(command):
    """Exécute une commande shell et retourne le résultat."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            text=True, 
            capture_output=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except Exception as e:
        return False, str(e)

def check_python_deps():
    """Vérifie la présence des dépendances Python requises."""
    result = ValidationResult()
    
    # Liste des packages essentiels
    essential_packages = [
        'pydantic', 'psycopg2', 'confluent_kafka', 
        'redis', 'fastapi', 'pandas', 'numpy', 'requests'
    ]
    
    for package in essential_packages:
        spec = importlib.util.find_spec(package)
        if spec is not None:
            result.add_test(f"Deps:Package {package}", True)
        else:
            result.add_test(f"Deps:Package {package}", False, f"Package {package} non installé")
    
    return result

def check_file_structure():
    """Vérifie la structure des fichiers du projet."""
    result = ValidationResult()
    
    # Fichiers essentiels
    essential_files = [
        'docker-compose.yml',
        'shared/src/config.py',
        'shared/src/schemas.py',
        'shared/src/enums.py',
        'shared/src/kafka_client.py',
        'shared/src/redis_client.py',
        'shared/src/db_pool.py',
        'database/schema.sql'
    ]
    
    # Dossiers essentiels
    essential_dirs = [
        'gateway/src',
        'analyzer/src',
        'analyzer/strategies',
        'trader/src',
        'portfolio/src',
        'coordinator/src',
        'dispatcher/src',
        'logger/src'
    ]
    
    # Vérifier les fichiers
    for file_path in essential_files:
        if os.path.isfile(file_path):
            result.add_test(f"Files:{file_path}", True)
        else:
            result.add_test(f"Files:{file_path}", False, f"Fichier manquant")
    
    # Vérifier les dossiers
    for dir_path in essential_dirs:
        if os.path.isdir(dir_path):
            result.add_test(f"Dirs:{dir_path}", True)
        else:
            result.add_test(f"Dirs:{dir_path}", False, f"Dossier manquant")
    
    return result

def check_code_syntax():
    """Vérifie la syntaxe Python des fichiers principaux."""
    result = ValidationResult()
    
    # Liste des fichiers Python importants à vérifier
    important_python_files = [
        'shared/src/config.py',
        'shared/src/schemas.py',
        'shared/src/enums.py',
        'shared/src/kafka_client.py',
        'shared/src/redis_client.py',
        'shared/src/db_pool.py',
        'gateway/src/main.py',
        'analyzer/src/main.py',
        'trader/src/main.py',
        'portfolio/src/main.py'
    ]
    
    for file_path in important_python_files:
        if not os.path.isfile(file_path):
            result.add_test(f"Syntax:{file_path}", False, "Fichier manquant")
            continue
            
        # Utiliser Python pour vérifier la syntaxe
        cmd = f"python -m py_compile {file_path}"
        success, output = run_command(cmd)
        
        if success:
            result.add_test(f"Syntax:{file_path}", True)
        else:
            result.add_test(f"Syntax:{file_path}", False, f"Erreur de syntaxe: {output}")
    
    return result

def check_imports():
    """Vérifie que les imports critiques fonctionnent."""
    result = ValidationResult()
    
    # Liste des modules critiques
    critical_modules = [
        'shared.src.config',
        'shared.src.schemas',
        'shared.src.enums',
        'shared.src.kafka_client',
        'shared.src.redis_client',
        'shared.src.db_pool'
    ]
    
    for module_name in critical_modules:
        try:
            module = __import__(module_name, fromlist=['*'])
            result.add_test(f"Import:{module_name}", True)
        except ImportError as e:
            result.add_test(f"Import:{module_name}", False, str(e))
        except Exception as e:
            result.add_test(f"Import:{module_name}", False, f"Erreur inattendue: {str(e)}")
    
    return result

def check_db_connection(skip=False):
    """Vérifie la connexion à la base de données."""
    result = ValidationResult()
    
    if skip:
        result.add_warning("Test de connexion à la base de données ignoré")
        return result
    
    try:
        from shared.src.db_pool import DBConnectionPool
        
        # Tenter d'établir une connexion
        db_pool = DBConnectionPool.get_instance()
        conn = db_pool.get_connection()
        
        if conn:
            # Vérifier si les tables principales existent
            cursor = conn.cursor()
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                AND table_name IN ('trade_cycles', 'trade_executions', 'trading_signals', 'market_data');
            """)
            
            tables = cursor.fetchall()
            found_tables = [table[0] for table in tables]
            
            # Libérer les ressources
            cursor.close()
            db_pool.release_connection(conn)
            
            # Vérifier que toutes les tables essentielles existent
            essential_tables = ['trade_cycles', 'trade_executions', 'trading_signals', 'market_data']
            missing_tables = [table for table in essential_tables if table not in found_tables]
            
            if missing_tables:
                result.add_test("DB:Connection", True)
                result.add_test("DB:Tables", False, f"Tables manquantes: {', '.join(missing_tables)}")
                result.add_warning("Le schéma de base de données n'est peut-être pas complètement installé")
            else:
                result.add_test("DB:Connection", True)
                result.add_test("DB:Tables", True)
        else:
            result.add_test("DB:Connection", False, "Impossible d'obtenir une connexion")
            
    except ImportError:
        result.add_test("DB:Import", False, "Module db_pool non trouvé")
    except Exception as e:
        result.add_test("DB:Connection", False, str(e))
    
    return result

def check_redis_connection(skip=False):
    """Vérifie la connexion à Redis."""
    result = ValidationResult()
    
    if skip:
        result.add_warning("Test de connexion à Redis ignoré")
        return result
    
    try:
        from shared.src.redis_client import RedisClient
        
        # Tenter de se connecter à Redis
        redis_client = RedisClient()
        ping_result = redis_client.redis.ping()
        
        if ping_result:
            result.add_test("Redis:Connection", True)
            
            # Tester set/get
            test_key = f"test:validation:{time.time()}"
            test_value = "Validation test"
            
            set_result = redis_client.set(test_key, test_value, expiration=60)
            get_result = redis_client.get(test_key)
            
            redis_client.delete(test_key)  # Nettoyage
            
            if set_result and get_result == test_value:
                result.add_test("Redis:Operations", True)
            else:
                result.add_test("Redis:Operations", False, "Échec des opérations set/get")
        else:
            result.add_test("Redis:Connection", False, "Échec du ping Redis")
            
    except ImportError:
        result.add_test("Redis:Import", False, "Module redis_client non trouvé")
    except Exception as e:
        result.add_test("Redis:Connection", False, str(e))
    
    return result

def check_kafka_connection(skip=False):
    """Vérifie la connexion à Kafka."""
    result = ValidationResult()
    
    if skip:
        result.add_warning("Test de connexion à Kafka ignoré")
        return result
    
    try:
        from shared.src.kafka_client import KafkaClient
        
        # Tenter de créer un client Kafka
        kafka_client = KafkaClient()
        producer = kafka_client._create_producer()
        
        if producer:
            result.add_test("Kafka:Producer", True)
            
            # Tenter de lister les topics
            try:
                admin_client = kafka_client.admin_client or kafka_client._ensure_topics_exist(["test"])
                topics = admin_client.list_topics(timeout=10).topics
                
                if topics:
                    result.add_test("Kafka:Topics", True)
                else:
                    result.add_test("Kafka:Topics", False, "Aucun topic trouvé")
                    
            except Exception as e:
                result.add_test("Kafka:Topics", False, str(e))
                
        else:
            result.add_test("Kafka:Producer", False, "Échec de création du producteur")
            
    except ImportError:
        result.add_test("Kafka:Import", False, "Module kafka_client non trouvé")
    except Exception as e:
        result.add_test("Kafka:Connection", False, str(e))
    
    return result

def check_env_config():
    """Vérifie la configuration des variables d'environnement."""
    result = ValidationResult()
    
    try:
        from shared.src.config import (
            BINANCE_API_KEY, BINANCE_SECRET_KEY, 
            KAFKA_BROKER, KAFKA_GROUP_ID,
            REDIS_HOST, REDIS_PORT,
            PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE,
            SYMBOLS, TRADING_MODE, TRADE_QUANTITIES,
            POCKET_CONFIG
        )
        
        # Vérifier les configurations critiques
        if not BINANCE_API_KEY:
            result.add_test("Config:Binance API", False, "Clé API Binance manquante")
        else:
            result.add_test("Config:Binance API", True)
            
        if not KAFKA_BROKER:
            result.add_test("Config:Kafka", False, "Broker Kafka non configuré")
        else:
            result.add_test("Config:Kafka", True)
            
        if not REDIS_HOST:
            result.add_test("Config:Redis", False, "Hôte Redis non configuré")
        else:
            result.add_test("Config:Redis", True)
            
        if not PGHOST:
            result.add_test("Config:Database", False, "Paramètres de base de données manquants")
        else:
            result.add_test("Config:Database", True)
        
        # Vérifier les symboles et le mode
        symbols_list = SYMBOLS if isinstance(SYMBOLS, list) else SYMBOLS.split(',')
        if not symbols_list:
            result.add_test("Config:Symbols", False, "Aucun symbole configuré")
        else:
            result.add_test("Config:Symbols", True)
            
        if TRADING_MODE not in ['demo', 'live']:
            result.add_test("Config:Mode", False, f"Mode de trading invalide: {TRADING_MODE}")
        else:
            result.add_test("Config:Mode", True)
            
        # Vérifier les quantités de trading
        if not TRADE_QUANTITIES:
            result.add_test("Config:Quantities", False, "Quantités de trading non configurées")
        else:
            result.add_test("Config:Quantities", True)
            
        # Vérifier la configuration des poches
        pocket_total = sum(POCKET_CONFIG.values())
        if abs(pocket_total - 1.0) > 0.001:  # Tenir compte des erreurs d'arrondi
            result.add_test("Config:Pockets", False, f"Allocation des poches incorrecte: {pocket_total*100}%")
        else:
            result.add_test("Config:Pockets", True)
            
    except ImportError:
        result.add_test("Config:Import", False, "Module de configuration non trouvé")
    except Exception as e:
        result.add_test("Config:General", False, str(e))
    
    return result

def main():
    """Fonction principale exécutant la validation."""
    
    # Analyser les arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Validation du système RootTrading avant démarrage")
    parser.add_argument("--skip-db", action="store_true", help="Ignorer les tests de base de données")
    parser.add_argument("--skip-redis", action="store_true", help="Ignorer les tests Redis")
    parser.add_argument("--skip-kafka", action="store_true", help="Ignorer les tests Kafka")
    args = parser.parse_args()
    
    print("========================================")
    print("= VALIDATION PRÉ-DÉMARRAGE ROOTTRADING =")
    print("========================================")
    print("")
    
    # Stocker tous les résultats
    all_results = ValidationResult()
    
    print("1. Vérification des dépendances Python...")
    deps_result = check_python_deps()
    all_results.tests.update(deps_result.tests)
    all_results.warnings.extend(deps_result.warnings)
    all_results.errors.extend(deps_result.errors)
    
    print("2. Vérification de la structure des fichiers...")
    files_result = check_file_structure()
    all_results.tests.update(files_result.tests)
    all_results.warnings.extend(files_result.warnings)
    all_results.errors.extend(files_result.errors)
    
    print("3. Vérification de la syntaxe Python...")
    syntax_result = check_code_syntax()
    all_results.tests.update(syntax_result.tests)
    all_results.warnings.extend(syntax_result.warnings)
    all_results.errors.extend(syntax_result.errors)
    
    print("4. Vérification des imports Python...")
    imports_result = check_imports()
    all_results.tests.update(imports_result.tests)
    all_results.warnings.extend(imports_result.warnings)
    all_results.errors.extend(imports_result.errors)
    
    print("5. Vérification de la configuration...")
    config_result = check_env_config()
    all_results.tests.update(config_result.tests)
    all_results.warnings.extend(config_result.warnings)
    all_results.errors.extend(config_result.errors)
    
    print("6. Vérification de la connexion PostgreSQL...")
    db_result = check_db_connection(args.skip_db)
    all_results.tests.update(db_result.tests)
    all_results.warnings.extend(db_result.warnings)
    all_results.errors.extend(db_result.errors)
    
    print("7. Vérification de la connexion Redis...")
    redis_result = check_redis_connection(args.skip_redis)
    all_results.tests.update(redis_result.tests)
    all_results.warnings.extend(redis_result.warnings)
    all_results.errors.extend(redis_result.errors)
    
    print("8. Vérification de la connexion Kafka...")
    kafka_result = check_kafka_connection(args.skip_kafka)
    all_results.tests.update(kafka_result.tests)
    all_results.warnings.extend(kafka_result.warnings)
    all_results.errors.extend(kafka_result.errors)
    
    # Afficher le résumé
    print(all_results.summary())

if __name__ == "__main__":
    main()