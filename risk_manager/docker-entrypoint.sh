#!/bin/bash
# docker-entrypoint.sh pour les services RootTrading
set -e

# Fonction pour attendre les services dépendants
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_retries=${4:-30}
    local retry_interval=${5:-1}
    
    echo "Waiting for $service at $host:$port..."
    
    local retries=0
    until nc -z $host $port; do
        retries=$((retries + 1))
        if [ $retries -ge $max_retries ]; then
            echo "Error: $service is not available after $max_retries attempts. Exiting."
            exit 1
        fi
        echo "Waiting for $service... retry $retries/$max_retries"
        sleep $retry_interval
    done
    
    echo "$service is available at $host:$port"
}

# Vérifier les dépendances selon l'environnement
if [ -n "$WAIT_FOR_REDIS" ]; then
    REDIS_HOST=${REDIS_HOST:-redis}
    REDIS_PORT=${REDIS_PORT:-6379}
    wait_for_service $REDIS_HOST $REDIS_PORT "Redis"
fi

if [ -n "$WAIT_FOR_KAFKA" ]; then
    KAFKA_HOST=$(echo $KAFKA_BROKER | cut -d':' -f1)
    KAFKA_PORT=$(echo $KAFKA_BROKER | cut -d':' -f2)
    wait_for_service $KAFKA_HOST $KAFKA_PORT "Kafka"
fi

if [ -n "$WAIT_FOR_DB" ]; then
    PGHOST=${PGHOST:-db}
    PGPORT=${PGPORT:-5432}
    wait_for_service $PGHOST $PGPORT "PostgreSQL"
fi

# Vérifier si on doit attendre d'autres services
if [ -n "$WAIT_FOR" ]; then
    for service in $(echo $WAIT_FOR | tr ',' ' '); do
        host=$(echo $service | cut -d':' -f1)
        port=$(echo $service | cut -d':' -f2)
        wait_for_service $host $port "Service $host"
    done
fi

# Configurer le niveau de log
export PYTHONUNBUFFERED=1

# Traitement des arguments
if [ "${1:0:1}" = '-' ]; then
    set -- python src/main.py "$@"
fi

# Exécuter la commande
echo "Starting service: $@"
exec "$@"