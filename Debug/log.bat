@echo off
cd /d E:\RootTrading\RootTrading

echo [LOG] Ouverture des logs Docker - Relancez les commandes si services demarres apres
echo.

start wt new-tab --title "Backend" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f gateway" ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f market_analyzer" ^
; move-focus left ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f dispatcher" ^
; move-focus right ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f portfolio" ^
; new-tab --title "Analyse" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f analyzer" ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f coordinator" ^
; move-focus left ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f signal_aggregator" ^
; move-focus right ^
; split-pane  -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f trader" ^
; new-tab --title "Infra" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f redis" ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f kafka" ^
; move-focus left ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f visualization" ^
; move-focus right ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f db"