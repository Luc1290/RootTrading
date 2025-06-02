#!/usr/bin/env python3
"""
Script pour créer les topics Kafka nécessaires.
"""
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_CONTAINER = "roottrading-kafka-1"
TOPICS = [
    # Topic pour les signaux filtrés (nouveau)
    {
        "name": "signals.filtered",
        "partitions": 3,
        "replication": 1,
        "config": {
            "retention.ms": "86400000",  # 24 heures
            "compression.type": "lz4"
        }
    },
    # Topic pour les signaux bruts de l'analyzer (si pas déjà créé)
    {
        "name": "analyzer.signals", 
        "partitions": 3,
        "replication": 1,
        "config": {
            "retention.ms": "86400000",  # 24 heures
            "compression.type": "lz4"
        }
    }
]

def create_topic(topic_info):
    """Créer un topic Kafka avec la configuration spécifiée."""
    name = topic_info["name"]
    partitions = topic_info["partitions"]
    replication = topic_info["replication"]
    
    # Commande de création du topic
    cmd = [
        "docker", "exec", KAFKA_CONTAINER,
        "kafka-topics", "--create",
        "--bootstrap-server", "localhost:9092",
        "--topic", name,
        "--partitions", str(partitions),
        "--replication-factor", str(replication),
        "--if-not-exists"
    ]
    
    # Ajouter les configurations supplémentaires
    for key, value in topic_info.get("config", {}).items():
        cmd.extend(["--config", f"{key}={value}"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Topic '{name}' créé avec succès")
        else:
            if "already exists" in result.stderr:
                logger.info(f"ℹ️ Topic '{name}' existe déjà")
            else:
                logger.error(f"❌ Erreur lors de la création du topic '{name}': {result.stderr}")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création du topic '{name}': {e}")

def list_topics():
    """Lister tous les topics Kafka existants."""
    cmd = [
        "docker", "exec", KAFKA_CONTAINER,
        "kafka-topics", "--list",
        "--bootstrap-server", "localhost:9092"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            topics = result.stdout.strip().split('\n')
            logger.info("📋 Topics Kafka existants:")
            for topic in topics:
                logger.info(f"  - {topic}")
        else:
            logger.error(f"❌ Erreur lors du listing des topics: {result.stderr}")
    except Exception as e:
        logger.error(f"❌ Erreur lors du listing des topics: {e}")

def main():
    """Fonction principale."""
    logger.info("🚀 Création des topics Kafka...")
    
    # Attendre que Kafka soit prêt
    logger.info("⏳ Attente que Kafka soit prêt...")
    time.sleep(5)
    
    # Créer les topics
    for topic in TOPICS:
        create_topic(topic)
    
    # Lister les topics pour vérification
    logger.info("\n📋 Vérification des topics...")
    list_topics()
    
    logger.info("\n✅ Script terminé")

if __name__ == "__main__":
    main()