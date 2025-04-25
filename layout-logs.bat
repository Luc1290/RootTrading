@echo off
REM launch-logs.bat - Lanceur simple pour la disposition des logs

REM Aller dans le répertoire du projet
cd /d E:\RootTrading\RootTrading

REM Lancer Windows Terminal avec la configuration prédéfinie
start wt new-tab --title "Backend" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f gateway" ^
; split-pane -H -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f coordinator" ^
; split-pane -V -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f dispatcher" ^
; new-tab --title "Analyse" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f analyzer" ^
; split-pane -H -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f trader" ^
; split-pane -V -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f pnl_tracker" ^
; new-tab --title "Infra" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f redis" ^
; split-pane -H -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f kafka" ^
; split-pane -V -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f kafka-init" ^
; split-pane -H -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f db" ^
; new-tab --title "Interfaces" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f frontend" ^
; split-pane -H -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f logger" ^
; split-pane -V -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f scheduler" ^
; split-pane -H -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f risk_manager" ^
; split-pane -V -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f portfolio"

REM Pas de pause ici pour que le script se termine immédiatement une fois la commande lancée