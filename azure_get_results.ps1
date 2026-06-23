# =============================================================================
# CHORUS Protocol - azure_get_results.ps1
# Download all benchmark artifacts from Azure File Share:
#   - SQLite database (chorus_benchmark.db)
#   - All JSON-Lines log files (chorus_*.log)
#
# Usage:
#   .\azure_get_results.ps1 [-ResourcePrefix chorus] [-OutputDir .\results]
# =============================================================================

param(
    [string]$ResourcePrefix = "chorus",
    [string]$OutputDir      = ".\chorus_results"
)

$ErrorActionPreference = "Stop"

# ── Load endpoints from deploy output ─────────────────────────────────────────
if (-not (Test-Path "chorus_azure_endpoints.json")) {
    Write-Error "chorus_azure_endpoints.json not found. Run azure_deploy.ps1 first."
    exit 1
}
$endpoints       = Get-Content "chorus_azure_endpoints.json" | ConvertFrom-Json
$STORAGE_ACCOUNT = $endpoints.storage_account
$RG_DE           = $endpoints.rg_germany
$SHARE_NAME      = "chorus-data"

Write-Host "Downloading CHORUS benchmark artifacts from Azure File Share..." -ForegroundColor Cyan
Write-Host "  Storage account: $STORAGE_ACCOUNT"
Write-Host "  Output dir:      $OutputDir"

# ── Create local output dir ───────────────────────────────────────────────────
New-Item -ItemType Directory -Force $OutputDir | Out-Null

# ── Get storage key ───────────────────────────────────────────────────────────
$storageKey = az storage account keys list `
    --resource-group $RG_DE `
    --account-name $STORAGE_ACCOUNT `
    --query "[0].value" -o tsv

# ── List all files on the share ───────────────────────────────────────────────
Write-Host "`nFiles on Azure File Share '$SHARE_NAME':" -ForegroundColor Yellow
$files = az storage file list `
    --account-name $STORAGE_ACCOUNT `
    --account-key $storageKey `
    --share-name $SHARE_NAME `
    --query "[].{name:name, size:properties.contentLength}" `
    -o json | ConvertFrom-Json

if (-not $files -or $files.Count -eq 0) {
    Write-Host "  No files found yet. Wait for the benchmark container to finish." -ForegroundColor Red
    exit 0
}

foreach ($f in $files) {
    $sizeMB = [math]::Round($f.size / 1MB, 2)
    Write-Host "  $($f.name)  ($sizeMB MB)"
}

# ── Download SQLite DB ────────────────────────────────────────────────────────
$dbFile = $files | Where-Object { $_.name -eq "chorus_benchmark.db" }
if ($dbFile) {
    $localDb = Join-Path $OutputDir "chorus_benchmark.db"
    Write-Host "`nDownloading database..." -ForegroundColor Cyan
    az storage file download `
        --account-name $STORAGE_ACCOUNT `
        --account-key $storageKey `
        --share-name $SHARE_NAME `
        --path "chorus_benchmark.db" `
        --dest $localDb `
        --output none
    $sizeMB = [math]::Round((Get-Item $localDb).Length / 1MB, 1)
    Write-Host "  Saved: $localDb  ($sizeMB MB)" -ForegroundColor Green
} else {
    Write-Host "  No database file found yet." -ForegroundColor Yellow
}

# ── Download all log files ────────────────────────────────────────────────────
$logFiles = $files | Where-Object { $_.name -like "chorus_*.log" }
if ($logFiles -and $logFiles.Count -gt 0) {
    Write-Host "`nDownloading log files..." -ForegroundColor Cyan
    foreach ($lf in $logFiles) {
        $localLog = Join-Path $OutputDir $lf.name
        az storage file download `
            --account-name $STORAGE_ACCOUNT `
            --account-key $storageKey `
            --share-name $SHARE_NAME `
            --path $lf.name `
            --dest $localLog `
            --output none
        $sizeKB = [math]::Round((Get-Item $localLog).Length / 1KB, 0)
        Write-Host "  Saved: $localLog  ($sizeKB KB)" -ForegroundColor Green
    }
} else {
    Write-Host "  No log files found yet." -ForegroundColor Yellow
}

# ── Also capture live container logs to files ─────────────────────────────────
Write-Host "`nCapturing container logs (last 1000 lines each)..." -ForegroundColor Cyan

$containers = @(
    @{ rg = $endpoints.rg_germany; name = "$ResourcePrefix-control-plane"; label = "control-plane" },
    @{ rg = $endpoints.rg_germany; name = "$ResourcePrefix-target";        label = "target-de" },
    @{ rg = $endpoints.rg_us;      name = "$ResourcePrefix-relay";         label = "relay-us" },
    @{ rg = $endpoints.rg_us;      name = "$ResourcePrefix-benchmark";     label = "benchmark-us" }
)

foreach ($c in $containers) {
    $logFile = Join-Path $OutputDir "container_$($c.label).log"
    try {
        az container logs `
            --resource-group $c.rg `
            --name $c.name `
            --tail 1000 `
            2>$null | Out-File $logFile -Encoding utf8
        $lines = (Get-Content $logFile | Measure-Object -Line).Lines
        Write-Host "  $($c.label): $lines lines -> $logFile" -ForegroundColor Green
    } catch {
        Write-Host "  $($c.label): unavailable (container may not exist yet)" -ForegroundColor Yellow
    }
}

# ── Run analysis if DB downloaded ─────────────────────────────────────────────
$localDb = Join-Path $OutputDir "chorus_benchmark.db"
if (Test-Path $localDb) {
    Write-Host "`nRunning analysis..." -ForegroundColor Cyan
    python analyze_results.py --db $localDb
}

Write-Host "`nAll artifacts saved to: $OutputDir" -ForegroundColor Green
Write-Host @"

  Files in $OutputDir:
    chorus_benchmark.db         - SQLite: order book data + all benchmark metrics
    chorus_benchmark_*.log      - JSON-Lines: per-chunk signal telemetry
    container_control-plane.log - stdout log from control plane
    container_target-de.log     - stdout log from target pod (Germany)
    container_relay-us.log      - stdout log from relay node (US)
    container_benchmark-us.log  - stdout log from benchmark client (US)
"@
