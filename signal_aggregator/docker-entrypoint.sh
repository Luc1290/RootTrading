#!/bin/bash
set -e

echo "Starting Signal Aggregator service..."

if [ "$WAIT_FOR_SERVICES" = "true" ]; then
    echo "Waiting for required services..."
    sleep 10
fi

exec "$@"