@echo off
echo Demarrage du systeme RootTrading...

REM Arreter tous les conteneurs
docker-compose down

REM Demarrer l'infrastructure
echo Demarrage de Redis, Kafka et DB...
docker-compose up -d redis kafka db
timeout /t 15

REM Demarrer Kafka-init
echo Initialisation des topics Kafka...
docker-compose up -d kafka-init
timeout /t 10

REM Demarrer les services principaux
echo Demarrage des services principaux...
docker-compose up -d gateway
timeout /t 10
docker-compose up -d portfolio trader
timeout /t 15
docker-compose up -d analyzer
timeout /t 10
docker-compose up -d coordinator
timeout /t 10

REM Demarrer les services secondaires
echo Demarrage des services secondaires...
docker-compose up -d dispatcher logger risk_manager scheduler
timeout /t 10
docker-compose up -d pnl_tracker
timeout /t 5
docker-compose up -d frontend

echo Tous les services sont demarres!
docker-compose ps