@echo off
cd /d E:\RootTrading\RootTrading

start wt new-tab --title "Backend" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f gateway" ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f market_analyzer" ^
; move-focus left ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f dispatcher" ^
; move-focus right ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f analyzer" ^
; new-tab --title "Analyse" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f signal_aggregator" ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f trader" ^
; move-focus left ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f coordinator" ^
; move-focus right ^
; split-pane  -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f portfolio" ^
; new-tab --title "Infra" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f redis" ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f kafka" ^
; move-focus left ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f visualization" ^
; move-focus right ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f db" ^