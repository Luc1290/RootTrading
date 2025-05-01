# Script de surveillance des ressources des conteneurs
param (
    [int]$RefreshInterval = 5,
    [switch]$SortByCPU = $false,
    [switch]$SortByMemory = $false,
    [string]$FilterService = ""
)

function Get-FormattedSize {
    param (
        [double]$Bytes
    )
    
    if ($Bytes -lt 1024) {
        return "$($Bytes.ToString("0.00")) B"
    }
    elseif ($Bytes -lt 1048576) {
        return "$([math]::Round($Bytes / 1024, 2)) KB"
    }
    elseif ($Bytes -lt 1073741824) {
        return "$([math]::Round($Bytes / 1048576, 2)) MB"
    }
    else {
        return "$([math]::Round($Bytes / 1073741824, 2)) GB"
    }
}

function Show-Header {
    Clear-Host
    Write-Host "===== SURVEILLANCE DES RESSOURCES - SYSTÈME DE TRADING =====" -ForegroundColor Cyan
    Write-Host "Actualisé à: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "Intervalle de rafraîchissement: $RefreshInterval secondes" -ForegroundColor Gray
    
    if (-not [string]::IsNullOrEmpty($FilterService)) {
        Write-Host "Filtre actif: $FilterService" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host ("CONTENEUR".PadRight(25) + "CPU %".PadRight(10) + "MEM USAGE".PadRight(15) + "MEM %".PadRight(10) + "NET I/O".PadRight(20) + "STATUT".PadRight(12)) -ForegroundColor Cyan
    Write-Host ("-" * 90) -ForegroundColor Gray
}

function Show-ContainerStats {
    $statCommand = "docker stats --no-stream --format '{{.Name}};{{.CPUPerc}};{{.MemUsage}};{{.MemPerc}};{{.NetIO}};{{.Status}}'"
    $stats = Invoke-Expression $statCommand
    
    $containerStats = @()
    
    foreach ($stat in $stats) {
        $parts = $stat -split ';'
        if ($parts.Count -ge 6) {
            $containerName = $parts[0] -replace '_', '-'  # Convertir les underscores en tirets pour correspondre au format docker-compose
            
            # Si un filtre est spécifié, ne garder que les services correspondants
            if (-not [string]::IsNullOrEmpty($FilterService) -and $containerName -notlike "*$FilterService*") {
                continue
            }
            
            $cpuPerc = $parts[1] -replace '%', ''
            if ([string]::IsNullOrEmpty($cpuPerc)) { $cpuPerc = "0.00" }
            
            $memUsage = $parts[2]
            $memPerc = $parts[3] -replace '%', ''
            if ([string]::IsNullOrEmpty($memPerc)) { $memPerc = "0.00" }
            
            $netIO = $parts[4]
            $status = $parts[5]
            
            $containerStats += [PSCustomObject]@{
                Name = $containerName
                CPUPerc = [double]$cpuPerc
                MemUsage = $memUsage
                MemPerc = [double]$memPerc
                NetIO = $netIO
                Status = $status
            }
        }
    }
    
    # Tri selon les paramètres
    if ($SortByCPU) {
        $containerStats = $containerStats | Sort-Object -Property CPUPerc -Descending
    }
    elseif ($SortByMemory) {
        $containerStats = $containerStats | Sort-Object -Property MemPerc -Descending
    }
    else {
        $containerStats = $containerStats | Sort-Object -Property Name
    }
    
    # Affichage des stats
    foreach ($container in $containerStats) {
        $cpuColor = if ($container.CPUPerc -gt 50) { "Red" } elseif ($container.CPUPerc -gt 20) { "Yellow" } else { "Green" }
        $memColor = if ($container.MemPerc -gt 75) { "Red" } elseif ($container.MemPerc -gt 40) { "Yellow" } else { "Green" }
        $statusColor = if ($container.Status -like "*Up*") { "Green" } else { "Red" }
        
        $name = $container.Name.PadRight(25)
        $cpu = "$($container.CPUPerc)%".PadRight(10)
        $mem = $container.MemUsage.PadRight(15)
        $memPerc = "$($container.MemPerc)%".PadRight(10)
        $netIO = $container.NetIO.PadRight(20)
        $status = $container.Status.PadRight(12)
        
        Write-Host $name -NoNewline
        Write-Host $cpu -NoNewline -ForegroundColor $cpuColor
        Write-Host $mem -NoNewline
        Write-Host $memPerc -NoNewline -ForegroundColor $memColor
        Write-Host $netIO -NoNewline
        Write-Host $status -ForegroundColor $statusColor
    }
    
    # Calcul des totaux
    $totalCPU = ($containerStats | Measure-Object -Property CPUPerc -Sum).Sum
    $totalMemPerc = ($containerStats | Measure-Object -Property MemPerc -Sum).Sum
    
    Write-Host ("-" * 90) -ForegroundColor Gray
    Write-Host "TOTAUX:".PadRight(25) -NoNewline
    Write-Host "$($totalCPU.ToString("0.00"))%".PadRight(10) -NoNewline -ForegroundColor Cyan
    Write-Host "".PadRight(15) -NoNewline
    Write-Host "$($totalMemPerc.ToString("0.00"))%".PadRight(10) -ForegroundColor Cyan
    
    Write-Host "`nCommandes disponibles:" -ForegroundColor Cyan
    Write-Host " - [C] Trier par CPU" -ForegroundColor Gray
    Write-Host " - [M] Trier par mémoire" -ForegroundColor Gray
    Write-Host " - [N] Trier par nom" -ForegroundColor Gray
    Write-Host " - [F] Filtrer par service" -ForegroundColor Gray
    Write-Host " - [R] Réinitialiser les filtres" -ForegroundColor Gray
    Write-Host " - [Q] Quitter" -ForegroundColor Gray
}

function Get-KeyPress {
    if ($host.UI.RawUI.KeyAvailable) {
        $key = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        return $key.Character
    }
    return $null
}

# Boucle principale
try {
    while ($true) {
        Show-Header
        Show-ContainerStats
        
        # Attente avec vérification des touches
        $startTime = Get-Date
        $endTime = $startTime.AddSeconds($RefreshInterval)
        
        while ((Get-Date) -lt $endTime) {
            Start-Sleep -Milliseconds 100
            $keyPress = Get-KeyPress
            
            if ($null -ne $keyPress) {
                switch ($keyPress.ToString().ToUpper()) {
                    "C" {
                        $SortByCPU = $true
                        $SortByMemory = $false
                        Show-Header
                        Show-ContainerStats
                    }
                    "M" {
                        $SortByCPU = $false
                        $SortByMemory = $true
                        Show-Header
                        Show-ContainerStats
                    }
                    "N" {
                        $SortByCPU = $false
                        $SortByMemory = $false
                        Show-Header
                        Show-ContainerStats
                    }
                    "F" {
                        $SortByCPU = $false
                        $SortByMemory = $false
                        Write-Host "`nEntrez le nom du service à filtrer: " -ForegroundColor Yellow -NoNewline
                        $FilterService = Read-Host
                        Show-Header
                        Show-ContainerStats
                    }
                    "R" {
                        $SortByCPU = $false
                        $SortByMemory = $false
                        $FilterService = ""
                        Show-Header
                        Show-ContainerStats
                    }
                    "Q" {
                        Write-Host "`nFin de la surveillance." -ForegroundColor Cyan
                        exit
                    }
                }
            }
        }
    }
}
catch {
    Write-Host "Erreur: $_" -ForegroundColor Red
}
finally {
    Write-Host "Fin de la surveillance." -ForegroundColor Cyan
}