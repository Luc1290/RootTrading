@echo off
echo ========================================
echo   DEMARRAGE SYSTEME DE TRADING - v1.0
echo ========================================
echo.

REM Définition des couleurs pour les messages
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "CYAN=[96m"
set "RESET=[0m"

echo %CYAN%[INFO]%RESET% Arrêt des conteneurs existants...
docker-compose down

echo.
echo %CYAN%[INFO]%RESET% Démarrage des services d'infrastructure (Redis, Kafka, DB)...
docker-compose up -d redis kafka db
echo %YELLOW%[WAIT]%RESET% Attente du démarrage des services d'infrastructure...

REM Vérification du statut de Redis
:CHECK_REDIS
timeout /t 2 > nul
docker-compose exec -T redis redis-cli ping > nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%[WAIT]%RESET% Attente de Redis...
    goto CHECK_REDIS
) else (
    echo %GREEN%[OK]%RESET% Redis est prêt
)

REM Vérification du statut de Kafka
:CHECK_KAFKA
timeout /t 2 > nul
docker-compose logs --tail=20 kafka | findstr "started" > nul
if %errorlevel% neq 0 (
    echo %YELLOW%[WAIT]%RESET% Attente de Kafka...
    goto CHECK_KAFKA
) else (
    echo %GREEN%[OK]%RESET% Kafka est prêt
)

REM Initialisation de Kafka
echo.
echo %CYAN%[INFO]%RESET% Initialisation des topics Kafka...
docker-compose up -d kafka-init
timeout /t 5 > nul
echo %GREEN%[OK]%RESET% Topics Kafka initialisés

REM Vérification du statut de la base de données
:CHECK_DB
timeout /t 2 > nul
docker-compose exec -T db pg_isready -U postgres > nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%[WAIT]%RESET% Attente de la base de données...
    goto CHECK_DB
) else (
    echo %GREEN%[OK]%RESET% Base de données prête
)

echo.
echo %CYAN%[INFO]%RESET% Démarrage du gateway et du dispatcher...
docker-compose up -d gateway dispatcher
timeout /t 10 > nul
echo %GREEN%[OK]%RESET% Gateway et dispatcher démarrés

echo.
echo %CYAN%[INFO]%RESET% Démarrage de l'analyzer...
docker-compose up -d analyzer
timeout /t 5 > nul
echo %GREEN%[OK]%RESET% Analyzer démarré

echo.
echo %CYAN%[INFO]%RESET% Démarrage du trader et du portfolio...
docker-compose up -d trader portfolio
timeout /t 10 > nul
echo %GREEN%[OK]%RESET% Trader et portfolio démarrés

echo.
echo %CYAN%[INFO]%RESET% Démarrage du risk_manager...
docker-compose up -d risk_manager
timeout /t 5 > nul
echo %GREEN%[OK]%RESET% Risk manager démarré

echo.
echo %CYAN%[INFO]%RESET% Démarrage du coordinator...
docker-compose up -d coordinator
timeout /t 5 > nul
echo %GREEN%[OK]%RESET% Coordinator démarré

echo.
echo %CYAN%[INFO]%RESET% Démarrage des services auxiliaires...
docker-compose up -d pnl_tracker scheduler frontend logger tester
timeout /t 10 > nul
echo %GREEN%[OK]%RESET% Services auxiliaires démarrés

echo.
echo %GREEN%========================================
echo   SYSTEME DE TRADING DÉMARRÉ AVEC SUCCÈS
echo ========================================%RESET%
echo.

echo %CYAN%Vérification de la santé des services principaux...%RESET%
echo.

REM Vérifier la santé des services principaux
for %%s in (gateway dispatcher analyzer trader portfolio risk_manager coordinator) do (
    docker-compose ps %%s | findstr "Up" > nul
    if %errorlevel% neq 0 (
        echo %RED%[ERROR]%RESET% Service %%s: NON DISPONIBLE
    ) else (
        echo %GREEN%[OK]%RESET% Service %%s: EN FONCTIONNEMENT
    )
)

echo.
echo %CYAN%Que souhaitez-vous faire maintenant?%RESET%
echo 1. Voir les logs de tous les services
echo 2. Voir les logs d'un service spécifique
echo 3. Surveiller la santé des services
echo 4. Quitter
echo.

:MENU
set /p choice="Entrez votre choix (1-4): "

if "%choice%"=="1" (
    echo %CYAN%Affichage des logs de tous les services (Ctrl+C pour quitter)...%RESET%
    docker-compose logs -f
    goto MENU
) else if "%choice%"=="2" (
    set /p service="Entrez le nom du service: "
    echo %CYAN%Affichage des logs de %service% (Ctrl+C pour quitter)...%RESET%
    docker-compose logs -f %service%
    goto MENU
) else if "%choice%"=="3" (
    echo %CYAN%Surveillance de la santé des services (Ctrl+C pour quitter)...%RESET%
    :HEALTH_CHECK
    cls
    echo %CYAN%== STATUT DES SERVICES == %RESET%(Actualisation toutes les 10 secondes, Ctrl+C pour quitter)
    echo Date et heure: %date% %time%
    echo.
    echo %CYAN%Services d'infrastructure:%RESET%
    for %%s in (redis kafka db) do (
        docker-compose ps %%s | findstr "Up" > nul
        if %errorlevel% neq 0 (
            echo %RED%[ERROR]%RESET% %%s: NON DISPONIBLE
        ) else (
            echo %GREEN%[OK]%RESET% %%s: EN FONCTIONNEMENT
        )
    )
    echo.
    echo %CYAN%Services principaux:%RESET%
    for %%s in (gateway dispatcher analyzer trader portfolio risk_manager coordinator) do (
        docker-compose ps %%s | findstr "Up" > nul
        if %errorlevel% neq 0 (
            echo %RED%[ERROR]%RESET% %%s: NON DISPONIBLE
        ) else (
            echo %GREEN%[OK]%RESET% %%s: EN FONCTIONNEMENT
        )
    )
    echo.
    echo %CYAN%Services auxiliaires:%RESET%
    for %%s in (pnl_tracker scheduler frontend logger tester) do (
        docker-compose ps %%s | findstr "Up" > nul
        if %errorlevel% neq 0 (
            echo %RED%[ERROR]%RESET% %%s: NON DISPONIBLE
        ) else (
            echo %GREEN%[OK]%RESET% %%s: EN FONCTIONNEMENT
        )
    )
    timeout /t 10 > nul
    goto HEALTH_CHECK
) else if "%choice%"=="4" (
    echo Au revoir!
    exit /b 0
) else (
    echo Choix invalide, veuillez réessayer.
    goto MENU
)