# Script de surveillance de la santé des services
param (
    [switch]$Continuous = $false,
    [int]$IntervalSeconds = 30
)

$services = @(
    "gateway", "dispatcher", "analyzer", "trader", 
    "portfolio", "risk_manager", "coordinator"
)

function Get-ServiceHealth {
    param (
        [string]$ServiceName
    )
    
    try {
        $containerIp = docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $(docker-compose ps -q $ServiceName)
        
        if ([string]::IsNullOrEmpty($containerIp)) {
            return @{
                Name = $ServiceName
                Status = "Non démarré"
                StatusCode = 0
                Color = "Red"
            }
        }
        
        $healthEndpoint = "http://$($containerIp):8000/health"
        $response = Invoke-WebRequest -Uri $healthEndpoint -TimeoutSec 2 -ErrorAction Stop
        
        return @{
            Name = $ServiceName
            Status = "Disponible"
            StatusCode = $response.StatusCode
            Color = "Green"
        }
    }
    catch {
        return @{
            Name = $ServiceName
            Status = "Non disponible"
            StatusCode = 0
            Color = "Red"
        }
    }
}

function Show-ServicesStatus {
    Clear-Host
    Write-Host "=== STATUT DES SERVICES ===" -ForegroundColor Cyan
    Write-Host "Actualisé à: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" -ForegroundColor Gray
    
    $allHealthy = $true
    
    foreach ($service in $services) {
        $health = Get-ServiceHealth -ServiceName $service
        
        $statusMessage = "[$($health.Status)]".PadRight(18)
        Write-Host "$($service.PadRight(15)) $statusMessage" -ForegroundColor $health.Color
        
        if ($health.Color -ne "Green") {
            $allHealthy = $false
        }
    }
    
    Write-Host "`nStatut général: " -NoNewline
    if ($allHealthy) {
        Write-Host "TOUS LES SERVICES SONT DISPONIBLES" -ForegroundColor Green
    } else {
        Write-Host "CERTAINS SERVICES NE SONT PAS DISPONIBLES" -ForegroundColor Red
    }
    
    # Afficher les logs récents de services problématiques
    if (-not $allHealthy) {
        Write-Host "`nLogs récents des services en erreur:" -ForegroundColor Yellow
        foreach ($service in $services) {
            $health = Get-ServiceHealth -ServiceName $service
            if ($health.Color -ne "Green") {
                Write-Host "`n--- Derniers logs de $service ---" -ForegroundColor Yellow
                docker-compose logs --tail=10 $service
            }
        }
    }
}

if ($Continuous) {
    while ($true) {
        Show-ServicesStatus
        Write-Host "`nSurveillance continue. Ctrl+C pour quitter. Prochaine vérification dans $IntervalSeconds secondes..." -ForegroundColor Gray
        Start-Sleep -Seconds $IntervalSeconds
    }
} else {
    Show-ServicesStatus
}