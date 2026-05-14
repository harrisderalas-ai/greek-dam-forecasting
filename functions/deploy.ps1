# functions/deploy.ps1
#
# Deploy the Function App to Azure.
# Copies src/, scripts/, infra/ from the repo root into functions/ before
# publishing, since Azure Functions can only upload files in its own directory.
# The copies are gitignored and recreated each deploy.

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

Write-Host ""
Write-Host "==================================" -ForegroundColor Cyan
Write-Host " Function App deployment" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# 1. Clean up old copies
Write-Host "[1/3] Cleaning up old copies..." -ForegroundColor Yellow
foreach ($d in @("src", "scripts", "infra")) {
    if (Test-Path "$ScriptDir\$d") {
        Remove-Item -Recurse -Force "$ScriptDir\$d"
    }
}
Write-Host "  Done."
Write-Host ""

# 2. Copy from repo root
Write-Host "[2/3] Copying src/, scripts/, infra/ from repo root..." -ForegroundColor Yellow
Copy-Item -Recurse -Force "$RepoRoot\src" "$ScriptDir\src"
Copy-Item -Recurse -Force "$RepoRoot\scripts" "$ScriptDir\scripts"
Copy-Item -Recurse -Force "$RepoRoot\infra" "$ScriptDir\infra"
Write-Host "  Copied src/, scripts/, infra/"
Write-Host ""

# 3. Publish
Write-Host "[3/3] Publishing to Azure..." -ForegroundColor Yellow
Push-Location $ScriptDir
try {
    func azure functionapp publish "func-greekdam-dev-westeu" --python
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "==================================" -ForegroundColor Green
Write-Host " Deployment complete!" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green
Write-Host ""