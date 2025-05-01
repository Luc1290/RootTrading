@echo off
cd /d E:\RootTrading\RootTrading

start wt new-tab --title "Backend" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f gateway" ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f analyzer" ^
; move-focus left ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f dispatcher" ^
; move-focus right ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f risk_manager" ^
; new-tab --title "Analyse" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f coordinator" ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f trader" ^
; move-focus left ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f portfolio" ^
; move-focus right ^
; split-pane  -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f pnl_tracker" ^
; new-tab --title "Infra" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f redis" ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f kafka" ^
; move-focus left ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f kafka-init" ^
; move-focus right ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f db" ^
; new-tab --title "Interfaces" -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f frontend" ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f logger" ^
; move-focus left ^
; split-pane -d "E:\RootTrading\RootTrading" powershell -NoExit -Command "docker compose logs -f scheduler" 
; move-focus right ^