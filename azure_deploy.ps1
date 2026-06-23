# =============================================================================
# CHORUS Protocol - azure_deploy.ps1
# Cross-Region Azure Container Instances Deployment
#
# Provisions:
#   - Resource groups in germanywestcentral + eastus
#   - Azure Container Registry (ACR) to host the CHORUS image
#   - Azure File Shares (one per region) for SQLite persistence
#   - ACI containers: control-plane, target, relay, benchmark-client
#     Split across two regions:
#       Germany (germanywestcentral): control-plane + target-pod
#       US      (eastus):             relay-node  + benchmark-client
#
# Prerequisites:
#   az login
#   az account set --subscription "<your-subscription-id>"
#   Docker Desktop running (for image build + push)
#
# Usage:
#   .\azure_deploy.ps1 [-ResourcePrefix chorus] [-Destroy]
# =============================================================================

param(
    [string]$ResourcePrefix = "chorus",
    [switch]$Destroy
)

Set-StrictMode -Version Latest
# Use Continue so az CLI / docker stderr lines don't abort the script.
# We check $LASTEXITCODE manually after commands that must succeed.
$ErrorActionPreference = "Continue"

# ── Configuration ─────────────────────────────────────────────────────────────
$RG_DE     = "$ResourcePrefix-rg-de"
$RG_US     = "$ResourcePrefix-rg-us"
$REGION_DE = "germanywestcentral"
$REGION_US = "eastus"
$SubSuffix = (az account show --query id -o tsv).Replace("-","").Substring(0,6)
$ACR_NAME  = "$($ResourcePrefix)acr$SubSuffix"
$ACR_RG    = "$ResourcePrefix-rg-acr"
$ACR_REGION = "westeurope"
$IMAGE_NAME = "chorus-protocol"
$IMAGE_TAG  = "latest"
$STORAGE_ACCOUNT = "$($ResourcePrefix)sa$SubSuffix"
$SHARE_NAME = "chorus-data"
$CHORUS_DIM = "128"
$CP_PORT    = 50051
$RELAY_PORT = 50052
$TARGET_PORT = 50053

# ── Teardown mode ─────────────────────────────────────────────────────────────
if ($Destroy) {
    Write-Host "Destroying all CHORUS Azure resources..." -ForegroundColor Red
    az group delete --name $RG_DE --yes --no-wait 2>$null
    az group delete --name $RG_US --yes --no-wait 2>$null
    az group delete --name $ACR_RG --yes --no-wait 2>$null
    Write-Host "Deletion initiated (async). Run 'az group list' to confirm." -ForegroundColor Yellow
    exit 0
}

# ── Helper functions ──────────────────────────────────────────────────────────
function Write-Step([string]$msg) {
    Write-Host "`n>> $msg" -ForegroundColor Cyan
}

function Check-Command([string]$cmd) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: Required command '$cmd' not found." -ForegroundColor Red
        exit 1
    }
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────
Write-Step "Pre-flight checks"
Check-Command "az"
Check-Command "docker"

$account = az account show --query "{name:name, id:id}" -o json | ConvertFrom-Json
Write-Host "  Azure subscription: $($account.name) ($($account.id))"

$dockerInfo = docker info 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Docker is not running" -ForegroundColor Red; exit 1 }
Write-Host "  Docker: running"

# ── 1. Create Resource Groups ─────────────────────────────────────────────────
Write-Step "Creating resource groups"
az group create --name $RG_DE    --location $REGION_DE --output none
az group create --name $RG_US    --location $REGION_US --output none
az group create --name $ACR_RG   --location $ACR_REGION --output none
Write-Host "  Created: $RG_DE ($REGION_DE), $RG_US ($REGION_US)"

# ── 2. Create Azure Container Registry ───────────────────────────────────────
Write-Step "Creating Azure Container Registry: $ACR_NAME"
az acr create `
    --resource-group $ACR_RG `
    --name $ACR_NAME `
    --sku Basic `
    --admin-enabled true `
    --output none
Write-Host "  ACR created: $ACR_NAME"

# Get ACR credentials
$acrCreds = az acr credential show --name $ACR_NAME --query "{username:username, password:passwords[0].value}" -o json | ConvertFrom-Json
$ACR_LOGIN_SERVER = az acr show --name $ACR_NAME --query loginServer -o tsv

# ── 3. Build and push Docker image ───────────────────────────────────────────
Write-Step "Building and pushing CHORUS Docker image"
$imageFull = "$ACR_LOGIN_SERVER/${IMAGE_NAME}:${IMAGE_TAG}"

az acr login --name $ACR_NAME

# Docker writes progress to stderr; use $ErrorActionPreference=Continue so
# PowerShell doesn't treat that output as a fatal error.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
docker build -t $imageFull "." --progress=plain
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: docker build failed" -ForegroundColor Red; exit 1 }
docker push $imageFull
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: docker push failed" -ForegroundColor Red; exit 1 }
$ErrorActionPreference = $prevEAP
Write-Host "  Image pushed: $imageFull"

# ── 4. Create Storage Account + File Shares (for SQLite persistence) ──────────
Write-Step "Creating Azure Storage for SQLite persistence"
az storage account create `
    --name $STORAGE_ACCOUNT `
    --resource-group $RG_DE `
    --location $REGION_DE `
    --sku Standard_LRS `
    --kind StorageV2 `
    --output none

$storageKey = az storage account keys list `
    --resource-group $RG_DE `
    --account-name $STORAGE_ACCOUNT `
    --query "[0].value" -o tsv

az storage share create `
    --name $SHARE_NAME `
    --account-name $STORAGE_ACCOUNT `
    --account-key $storageKey `
    --quota 10 `
    --output none

Write-Host "  Storage account: $STORAGE_ACCOUNT  share: $SHARE_NAME"

# ── 5. Deploy Control Plane (Germany) ────────────────────────────────────────
Write-Step "Deploying Control Plane in $REGION_DE"
$cpContainer = az container create `
    --resource-group $RG_DE `
    --name "$ResourcePrefix-control-plane" `
    --image $imageFull `
    --registry-login-server $ACR_LOGIN_SERVER `
    --registry-username $acrCreds.username `
    --registry-password $acrCreds.password `
    --os-type Linux `
    --cpu 1 `
    --memory 1 `
    --ports $CP_PORT `
    --ip-address Public `
    --dns-name-label "$ResourcePrefix-cp-de" `
    --environment-variables `
        CHORUS_ROLE=control_plane `
        CONTROL_PLANE_PORT=$CP_PORT `
        CHORUS_DIM=$CHORUS_DIM `
        CHORUS_SESSION_TTL=7200 `
        CHORUS_TARGET_HOST="$ResourcePrefix-target-de.$REGION_DE.azurecontainer.io" `
        CHORUS_TARGET_PORT=$TARGET_PORT `
        CHORUS_AMPLIFY_FACTOR=1.0 `
    --query ipAddress.fqdn -o tsv

$CP_FQDN = "$ResourcePrefix-cp-de.$REGION_DE.azurecontainer.io"
Write-Host "  Control Plane FQDN: $CP_FQDN"

# ── 6. Deploy Target Pod (Germany) ───────────────────────────────────────────
Write-Step "Deploying Target Pod in $REGION_DE"
az container create `
    --resource-group $RG_DE `
    --name "$ResourcePrefix-target" `
    --image $imageFull `
    --registry-login-server $ACR_LOGIN_SERVER `
    --registry-username $acrCreds.username `
    --registry-password $acrCreds.password `
    --os-type Linux `
    --cpu 2 `
    --memory 2 `
    --ports $TARGET_PORT `
    --ip-address Public `
    --dns-name-label "$ResourcePrefix-target-de" `
    --azure-file-volume-account-name $STORAGE_ACCOUNT `
    --azure-file-volume-account-key $storageKey `
    --azure-file-volume-share-name $SHARE_NAME `
    --azure-file-volume-mount-path "/data" `
    --environment-variables `
        CHORUS_ROLE=target `
        TARGET_PORT=$TARGET_PORT `
        CONTROL_PLANE_HOST=$CP_FQDN `
        CONTROL_PLANE_PORT=$CP_PORT `
        CHORUS_DIM=$CHORUS_DIM `
        BENCHMARK_DB=/data/chorus_benchmark.db `
    --output none

$TARGET_FQDN = "$ResourcePrefix-target-de.$REGION_DE.azurecontainer.io"
Write-Host "  Target Pod FQDN: $TARGET_FQDN"

# ── 7. Deploy Relay Node (US) ─────────────────────────────────────────────────
Write-Step "Deploying Relay Node in $REGION_US"
az container create `
    --resource-group $RG_US `
    --name "$ResourcePrefix-relay" `
    --image $imageFull `
    --registry-login-server $ACR_LOGIN_SERVER `
    --registry-username $acrCreds.username `
    --registry-password $acrCreds.password `
    --os-type Linux `
    --cpu 1 `
    --memory 1 `
    --ports $RELAY_PORT `
    --ip-address Public `
    --dns-name-label "$ResourcePrefix-relay-us" `
    --environment-variables `
        CHORUS_ROLE=relay `
        RELAY_PORT=$RELAY_PORT `
        CONTROL_PLANE_HOST=$CP_FQDN `
        CONTROL_PLANE_PORT=$CP_PORT `
        CHORUS_TARGET_HOST=$TARGET_FQDN `
        CHORUS_TARGET_PORT=$TARGET_PORT `
        CHORUS_DIM=$CHORUS_DIM `
    --output none

$RELAY_FQDN = "$ResourcePrefix-relay-us.$REGION_US.azurecontainer.io"
Write-Host "  Relay Node FQDN: $RELAY_FQDN"

# ── 8. Deploy Benchmark Client (US) ──────────────────────────────────────────
Write-Step "Deploying Benchmark Client in $REGION_US"
az container create `
    --resource-group $RG_US `
    --name "$ResourcePrefix-benchmark" `
    --image $imageFull `
    --registry-login-server $ACR_LOGIN_SERVER `
    --registry-username $acrCreds.username `
    --registry-password $acrCreds.password `
    --os-type Linux `
    --cpu 2 `
    --memory 2 `
    --azure-file-volume-account-name $STORAGE_ACCOUNT `
    --azure-file-volume-account-key $storageKey `
    --azure-file-volume-share-name $SHARE_NAME `
    --azure-file-volume-mount-path "/data" `
    --environment-variables `
        CHORUS_ROLE=benchmark `
        CONTROL_PLANE_HOST=$CP_FQDN `
        CONTROL_PLANE_PORT=$CP_PORT `
        RELAY_HOST=$RELAY_FQDN `
        RELAY_PORT=$RELAY_PORT `
        CHORUS_TARGET_HOST=$TARGET_FQDN `
        CHORUS_TARGET_PORT=$TARGET_PORT `
        CHORUS_USE_RELAY=true `
        CHORUS_DIM=$CHORUS_DIM `
        BENCHMARK_DB=/data/chorus_benchmark.db `
        SOURCE_REGION=$REGION_US `
        TARGET_REGION=$REGION_DE `
        BENCHMARK_CHUNKS=200 `
    --output none

# ── 9. Output connection summary ──────────────────────────────────────────────
Write-Step "Deployment complete"

$summary = @{
    "control_plane"     = "${CP_FQDN}:${CP_PORT}"
    "target_pod_de"     = "${TARGET_FQDN}:${TARGET_PORT}"
    "relay_node_us"     = "${RELAY_FQDN}:${RELAY_PORT}"
    "acr"               = $ACR_LOGIN_SERVER
    "storage_account"   = $STORAGE_ACCOUNT
    "rg_germany"        = $RG_DE
    "rg_us"             = $RG_US
}

$summary | ConvertTo-Json | Out-File "chorus_azure_endpoints.json" -Encoding utf8
Write-Host "`nEndpoints saved to: chorus_azure_endpoints.json"

Write-Host "`n  CHORUS Azure Topology:" -ForegroundColor Green
Write-Host "  +--------------Germany ($REGION_DE)-----------+"
Write-Host "  |  Control Plane:  $($summary.control_plane)"
Write-Host "  |  Target Pod:     $($summary.target_pod_de)"
Write-Host "  +---------------------------------------------+"
Write-Host "  +--------------US ($REGION_US)------------------+"
Write-Host "  |  Relay Node:     $($summary.relay_node_us)"
Write-Host "  |  Benchmark:      (see logs)"
Write-Host "  +---------------------------------------------+"

Write-Host "`n  Useful commands:"
Write-Host "  # Stream logs from target pod (Germany):"
Write-Host "  az container logs --resource-group $RG_DE --name $ResourcePrefix-target --follow"
Write-Host ""
Write-Host "  # Stream benchmark client logs (US):"
Write-Host "  az container logs --resource-group $RG_US --name $ResourcePrefix-benchmark --follow"
Write-Host ""
Write-Host "  # Download SQLite results:"
Write-Host "  .\azure_get_results.ps1"
Write-Host ""
Write-Host "  # Destroy everything:"
Write-Host "  .\azure_deploy.ps1 -Destroy"
