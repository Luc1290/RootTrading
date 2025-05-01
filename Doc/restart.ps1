# Script de redémarrage intelligent d'un service docker-compose
param (
    [Parameter(Mandatory=$true)]
    [string]$ServiceName,
    
    [switch]$Force = $false,
    
    [switch]$SkipHealthCheck = $false
)

# Configuration des couleurs pour les messages
$GREEN = [char]27 + "[92m"
$YELLOW = [char]27 + "[93m"
$RED = [char]27 + "[91m"
$CYAN = [char]27 + "[96m"
$RESET = [char]27 + "[0m"

# Fonction pour vérifier si un service existe dans docker-compose
function Test-ServiceExists {
    param (
        [string]$Service
    )
    
    $services = docker-compose config --services
    return $services -contains $Service
}

# Fonction pour vérifier quels services dépendent du service cible
function Get-DependentServices {
    param (
        [string]$Service
    )
    
    $dependents = @()
    $allServices = docker-compose config --services
    
    foreach ($srv in $allServices) {
        # Rechercher si ce service dépend du service cible
        $dependencies = docker-compose config | Select-String -Pattern "depends_on:[\s\S]*?$Service"
        
        if ($dependencies) {
            $dependents += $srv
        }
    }
    
    return $dependents
}

# Fonction pour vérifier la santé d'un service
function Test-ServiceHealth {
    param (
        [string]$Service,
        [int]$MaxRetries = 10,
        [int]$RetryIntervalSeconds = 3
    )
    
    Write-Host "${CYAN}Vérification de la santé de $Service...${RESET}"
    
    for ($i = 1; $i -le $MaxRetries; $i++) {
        try {
            $containerIp = docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $(docker-compose ps -q $Service)
            
            if ([string]::IsNullOrEmpty($containerIp)) {
                Write-Host "${YELLOW}  Attente du démarrage du conteneur $Service (tentative $i/$MaxRetries)...${RESET}"
                Start-Sleep -Seconds $RetryIntervalSeconds
                continue
            }
            
            # Déterminer le port du service à partir du docker-compose.yml
            $servicePort = docker-compose port $Service 8000 2>$null
            if ([string]::IsNullOrEmpty($servicePort)) {
                $servicePort = docker-compose port $Service 5000 2>$null
            }
            if ([string]::IsNullOrEmpty($servicePort)) {
                $servicePort = docker-compose port $Service 3000 2>$null
            }
            
            if ([string]::IsNullOrEmpty($servicePort)) {
                # Si aucun port n'est mappé, essayons avec le port interne du conteneur
                $healthEndpoint = "http://$($containerIp):8000/health"
            } else {
                $hostPort = ($servicePort -split ':')[1]
                $healthEndpoint = "http://localhost:$hostPort/health"
            }
            
            $response = Invoke-WebRequest -Uri $healthEndpoint -TimeoutSec 2 -ErrorAction Stop
            
            if ($response.StatusCode -eq 200) {
                Write-Host "${GREEN}  ✅ $Service est prêt!${RESET}"
                return $true
            }
        }
        catch {
            Write-Host "${YELLOW}  Attente de $Service (tentative $i/$MaxRetries)...${RESET}"
            Start-Sleep -Seconds $RetryIntervalSeconds
        }
    }
    
    Write-Host "${RED}  ❌ $Service n'est pas disponible après $MaxRetries tentatives${RESET}"
    return $false
}

# Vérifier si le service existe
if (-not (Test-ServiceExists -Service $ServiceName)) {
    Write-Host "${RED}Erreur: Le service '$ServiceName' n'existe pas dans votre docker-compose.yml${RESET}"
    exit 1
}

# Vérifier les services dépendants
$dependentServices = Get-DependentServices -Service $ServiceName
if ($dependentServices.Count -gt 0 -and -not $Force) {
    Write-Host "${YELLOW}Attention:${RESET} Les services suivants dépendent de '$ServiceName':"
    $dependentServices | ForEach-Object { Write-Host "  - $_" }
    
    $confirmation = Read-Host "Voulez-vous continuer? Cela pourrait affecter ces services (o/N)"
    if ($confirmation -ne "o" -and $confirmation -ne "O") {
        Write-Host "${CYAN}Opération annulée.${RESET}"
        exit 0
    }
}

# Redémarrer le service
Write-Host "${CYAN}Redémarrage du service '$ServiceName'...${RESET}"
docker-compose stop $ServiceName
Write-Host "${GREEN}Service arrêté.${RESET}"

# Attendre un court instant pour s'assurer que le service est bien arrêté
Start-Sleep -Seconds 2

# Démarrer le service
Write-Host "${CYAN}Démarrage du service '$ServiceName'...${RESET}"
docker-compose up -d $ServiceName

# Vérifier la santé du service redémarré si nécessaire
if (-not $SkipHealthCheck) {
    $isHealthy = Test-ServiceHealth -Service $ServiceName -MaxRetries 15 -RetryIntervalSeconds 2
    
    if ($isHealthy) {
        Write-Host "${GREEN}✅ Le service '$ServiceName' a été redémarré avec succès et est disponible.${RESET}"
    } else {
        Write-Host "${RED}⚠️ Le service '$ServiceName' a été redémarré mais ne répond pas aux vérifications de santé.${RESET}"
        Write-Host "${YELLOW}Vous pouvez vérifier les logs avec: docker-compose logs -f $ServiceName${RESET}"
    }
} else {
    Write-Host "${YELLOW}La vérification de santé a été ignorée.${RESET}"
    Write-Host "${GREEN}✅ Le service '$ServiceName' a été redémarré.${RESET}"
}

# Si des services dépendent de celui qui a été redémarré, proposer de les redémarrer aussi
if ($dependentServices.Count -gt 0) {
    $restartDependents = Read-Host "Voulez-vous également redémarrer les services dépendants? (o/N)"
    
    if ($restartDependents -eq "o" -or $restartDependents -eq "O") {
        foreach ($depService in $dependentServices) {
            Write-Host "${CYAN}Redémarrage du service dépendant '$depService'...${RESET}"
            docker-compose stop $depService
            docker-compose up -d $depService
            
            if (-not $SkipHealthCheck) {
                Test-ServiceHealth -Service $depService -MaxRetries 10 -RetryIntervalSeconds 2
            }
        }
        
        Write-Host "${GREEN}✅ Tous les services dépendants ont été redémarrés.${RESET}"
    }
}

Write-Host "${CYAN}Opération terminée.${RESET}"