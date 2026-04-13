$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$bundlePath = Join-Path $repoRoot "ML_source_bundle.zip.base64"
$zipPath = Join-Path $repoRoot "ML_source_bundle.zip"
$extractDir = Join-Path $repoRoot "restored_source_bundle"

if (-not (Test-Path -LiteralPath $bundlePath)) {
    throw "Bundle file not found: $bundlePath"
}

$base64 = Get-Content -LiteralPath $bundlePath -Raw
[IO.File]::WriteAllBytes($zipPath, [Convert]::FromBase64String($base64))

if (Test-Path -LiteralPath $extractDir) {
    Remove-Item -LiteralPath $extractDir -Recurse -Force
}

Expand-Archive -LiteralPath $zipPath -DestinationPath $extractDir -Force
Write-Output "Restored source bundle to: $extractDir"
