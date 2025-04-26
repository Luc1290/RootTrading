@echo off
REM launch-logs.bat - Version corrigée pour les titres d'onglets

REM Aller dans le répertoire du projet
cd /d E:\RootTrading\RootTrading

REM Lancer Windows Terminal avec des onglets correctement nommés
start wt new-tab --title "Backend" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f gateway" ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f coordinator" ^
; move-focus left ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f dispatcher" ^
; move-focus right ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f analyzer" ^
; new-tab --title "Analyse" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f analyzer" ^
; split-pane -H -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f trader" ^
; split-pane -V -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f portfolio" ^
; new-tab --title "Infra" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f redis" ^
; split-pane -H -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f kafka" ^
; split-pane -V -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f kafka-init" ^
; split-pane -H -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f db" ^
; new-tab --title "Interfaces" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f frontend" ^
; split-pane -H -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f logger" ^
; split-pane -V -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f scheduler" ^
; split-pane -H -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f risk_manager" ^
; split-pane -V -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f pnl_tracker"

REM Fin du script