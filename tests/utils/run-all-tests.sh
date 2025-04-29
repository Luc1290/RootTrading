#!/bin/bash

echo "=== TEST CONFIGURATION ==="
python scripts/config-validator.py
echo

echo "=== TEST UNITAIRES ==="
python scripts/test-unitaires.py
echo

echo "=== TEST DOCKER ==="
python scripts/docker-test.py
echo

echo "=== TEST DB ==="
python scripts/db-test.py
echo

echo "=== TEST MIDDLEWARE ==="
python scripts/middleware-test.py
echo

echo "=== SIMULATION GATEWAY ==="
python scripts/gateway-simulator.py
echo

echo "=== VALIDATION GLOBALE ==="
python scripts/pre-start-validation.py
echo

echo "=== TESTS ANALYZER ==="
python scripts/analyzer-test.py
echo

echo "=== TESTS INTEGRATION ==="
python scripts/integration-test.py

