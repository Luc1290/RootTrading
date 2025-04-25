# layout-logs.ps1

# Liste des services
$services = @(
    "gateway",
    "analyzer",
    "trader",
    "portfolio",
    "coordinator",
    "dispatcher",
    "redis",
    "kafka",
    "kafka-init",
    "db",
    "frontend",
    "logger",
    "pnl_tracker",
    "risk_manager",
    "scheduler"
)

# Dossier où est ton docker-compose.yml
$basePath = "E:\RootTrading\RootTrading"  # <-- ADAPTE CETTE LIGNE

# Début de la commande Windows Terminal
$wtCommand = ""

foreach ($service in $services) {
    $wtCommand += "new-tab --title $service -d `"$basePath`" powershell.exe -NoExit -Command `"docker compose logs -f $service`"; "
}

# Exécuter le tout
Invoke-Expression "wt.exe $wtCommand"
