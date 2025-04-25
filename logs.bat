@echo off
title RootTrading - Service Launcher

echo Lancement des fenetres de logs pour les services RootTrading...

:: Répertoire racine (chemin spécifique fourni)
set ROOT_DIR=E:\RootTrading\RootTrading

:: Services principaux
start powershell -NoExit -Command "cd '%ROOT_DIR%'; $host.UI.RawUI.WindowTitle = 'Gateway'; Write-Host 'Logs Gateway' -ForegroundColor Green; docker-compose logs -f gateway"
timeout /t 1 > nul

start powershell -NoExit -Command "cd '%ROOT_DIR%'; $host.UI.RawUI.WindowTitle = 'Analyzer'; Write-Host 'Logs Analyzer' -ForegroundColor Cyan; docker-compose logs -f analyzer"
timeout /t 1 > nul

start powershell -NoExit -Command "cd '%ROOT_DIR%'; $host.UI.RawUI.WindowTitle = 'Trader'; Write-Host 'Logs Trader' -ForegroundColor Yellow; docker-compose logs -f trader"
timeout /t 1 > nul

start powershell -NoExit -Command "cd '%ROOT_DIR%'; $host.UI.RawUI.WindowTitle = 'Portfolio'; Write-Host 'Logs Portfolio' -ForegroundColor Magenta; docker-compose logs -f portfolio"
timeout /t 1 > nul

start powershell -NoExit -Command "cd '%ROOT_DIR%'; $host.UI.RawUI.WindowTitle = 'Frontend'; Write-Host 'Logs Frontend' -ForegroundColor DarkGreen; docker-compose logs -f frontend"
timeout /t 1 > nul

:: Services secondaires
start powershell -NoExit -Command "cd '%ROOT_DIR%'; $host.UI.RawUI.WindowTitle = 'Coordinator'; Write-Host 'Logs Coordinator' -ForegroundColor Blue; docker-compose logs -f coordinator"
timeout /t 1 > nul

start powershell -NoExit -Command "cd '%ROOT_DIR%'; $host.UI.RawUI.WindowTitle = 'Dispatcher'; Write-Host 'Logs Dispatcher' -ForegroundColor Red; docker-compose logs -f dispatcher"
timeout /t 1 > nul

start powershell -NoExit -Command "cd '%ROOT_DIR%'; $host.UI.RawUI.WindowTitle = 'PnL Tracker'; Write-Host 'Logs PnL Tracker' -ForegroundColor DarkYellow; docker-compose logs -f pnl_tracker"
timeout /t 1 > nul

start powershell -NoExit -Command "cd '%ROOT_DIR%'; $host.UI.RawUI.WindowTitle = 'Risk Manager'; Write-Host 'Logs Risk Manager' -ForegroundColor DarkCyan; docker-compose logs -f risk_manager"
timeout /t 1 > nul

start powershell -NoExit -Command "cd '%ROOT_DIR%'; $host.UI.RawUI.WindowTitle = 'Scheduler'; Write-Host 'Logs Scheduler' -ForegroundColor DarkMagenta; docker-compose logs -f scheduler"
timeout /t 1 > nul

:: Infrastructure
start powershell -NoExit -Command "cd '%ROOT_DIR%'; $host.UI.RawUI.WindowTitle = 'Redis'; Write-Host 'Logs Redis' -ForegroundColor White; docker-compose logs -f redis"
timeout /t 1 > nul

start powershell -NoExit -Command "cd '%ROOT_DIR%'; $host.UI.RawUI.WindowTitle = 'Kafka'; Write-Host 'Logs Kafka' -ForegroundColor DarkBlue; docker-compose logs -f kafka"
timeout /t 1 > nul

start powershell -NoExit -Command "cd '%ROOT_DIR%'; $host.UI.RawUI.WindowTitle = 'Kafka Init'; Write-Host 'Logs Kafka Init' -ForegroundColor DarkRed; docker-compose logs -f kafka-init"
timeout /t 1 > nul

start powershell -NoExit -Command "cd '%ROOT_DIR%'; $host.UI.RawUI.WindowTitle = 'Database'; Write-Host 'Logs Database' -ForegroundColor Gray; docker-compose logs -f db"

echo Toutes les fenetres de logs ont ete lancees.
echo Vous devrez organiser manuellement les fenetres pour eviter qu'elles se chevauchent.
echo Vous pouvez fermer cette fenetre ou la garder pour reference.