$ErrorActionPreference = "Stop"
$ProgressPreference = 'SilentlyContinue'

$pgDir = "$PSScriptRoot\pgsql"
$zipFile = "$PSScriptRoot\postgres.zip"
$dataDir = "$pgDir\data"
$logFile = "$pgDir\logfile"

$urls = @(
    "https://get.enterprisedb.com/postgresql/postgresql-17.2-2-windows-x64-binaries.zip",
    "https://get.enterprisedb.com/postgresql/postgresql-17.2-1-windows-x64-binaries.zip"
)

if (-not (Test-Path $pgDir)) {
    Write-Host "Downloading PostgreSQL 17 binaries..."
    $downloaded = $false
    foreach ($url in $urls) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $zipFile -UseBasicParsing
            $downloaded = $true
            break
        } catch {
            Write-Host "Failed to download from $url, trying next..."
        }
    }
    
    if (-not $downloaded) {
        Write-Error "Failed to download PostgreSQL binaries."
    }

    Write-Host "Extracting..."
    & tar -xf $zipFile
    Remove-Item $zipFile -Force
}

if (-not (Test-Path $dataDir)) {
    Write-Host "Initializing database cluster..."
    & "$pgDir\bin\initdb.exe" -D $dataDir -U postgres --auth=trust
}

Write-Host "Starting PostgreSQL server..."
& "$pgDir\bin\pg_ctl.exe" -D $dataDir -l $logFile start

Start-Sleep -Seconds 3

Write-Host "Creating database grid_db..."
try {
    & "$pgDir\bin\createdb.exe" -U postgres grid_db
} catch {
    Write-Host "Database creation returned an error, it might already exist."
}

Write-Host "Setup complete!"
