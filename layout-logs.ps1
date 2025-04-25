# Add necessary assemblies
Add-Type -AssemblyName System.Windows.Forms

# List of services
$services = @(
    "gateway",
    "analyzer",
    "trader",
    "portfolio",
    "coordinator",
    "dispatcher"
    "redis",
    "db",
    "risk_manager"
)

# Get screen dimensions
$screenWidth = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea.Width
$screenHeight = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea.Height

# Define grid layout
$columns = 2
$rows = [Math]::Ceiling($services.Count / $columns)

$windowWidth = [Math]::Floor($screenWidth / $columns)
$windowHeight = [Math]::Floor($screenHeight / $rows)

Write-Host "Screen dimensions: $screenWidth x $screenHeight"
Write-Host "Grid layout: $columns x $rows"
Write-Host "Window size: $windowWidth x $windowHeight"

# Function to generate script content
function Get-LogScript {
    param (
        [string]$Service,
        [int]$PosX,
        [int]$PosY,
        [int]$Width,
        [int]$Height
    )
    
    return @"
# Window setup
try {
    # Import Windows API functions
    Add-Type @'
    using System;
    using System.Runtime.InteropServices;
    
    public class WinPos {
        [DllImport("user32.dll", SetLastError = true)]
        public static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);
        
        [DllImport("kernel32.dll")]
        public static extern IntPtr GetConsoleWindow();
    }
'@ -ErrorAction SilentlyContinue

    # Get console window handle
    `$handle = [WinPos]::GetConsoleWindow()
    
    # Position and resize window
    [void][WinPos]::MoveWindow(`$handle, $PosX, $PosY, $Width, $Height, `$true)
    
    # Set window title
    `$host.UI.RawUI.WindowTitle = "Docker Logs: $Service"
} catch {
    Write-Host "Error positioning window: `$_" -ForegroundColor Red
}

# Display logs
Write-Host "Logs for service: $Service" -ForegroundColor Cyan
Write-Host "--------------------" -ForegroundColor Cyan
docker-compose logs -f $Service
"@
}

# Open a window for each service
for ($i = 0; $i -lt $services.Count; $i++) {
    $service = $services[$i]
    
    # Calculate position
    $col = $i % $columns
    $row = [Math]::Floor($i / $columns)
    
    $posX = $col * $windowWidth
    $posY = $row * $windowHeight
    
    Write-Host "Setting up window $($i+1) for $service at position ($posX, $posY)"
    
    # Generate script for this service
    $scriptContent = Get-LogScript -Service $service -PosX $posX -PosY $posY -Width $windowWidth -Height $windowHeight
    $tempScriptPath = [System.IO.Path]::GetTempFileName() + ".ps1"
    $scriptContent | Out-File -FilePath $tempScriptPath -Encoding utf8
    
    # Launch PowerShell window with this script
    Start-Process powershell.exe -ArgumentList "-NoExit", "-File", $tempScriptPath
    
    # Pause to avoid timing issues
    Start-Sleep -Milliseconds 500
}

Write-Host "All log windows have been launched and should be positioned." -ForegroundColor Green