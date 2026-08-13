param(
    [string]$ProjectRoot = (Join-Path $HOME "Documents\ARL-FL")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$DatasetUrl = "https://archive.ics.uci.edu/static/public/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring.zip"
$DatasetPage = "https://archive.ics.uci.edu/dataset/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring"
$DatasetDoi = "10.24432/C5NW2H"

$DataRoot = Join-Path $ProjectRoot "data"
$ArchiveDir = Join-Path $DataRoot "archives"
$RawDir = Join-Path $DataRoot "raw"
$ManifestDir = Join-Path $DataRoot "manifests"
$PamapDir = Join-Path $RawDir "pamap2"
$OuterExtract = Join-Path $RawDir "_pamap2_outer"
$InnerExtract = Join-Path $RawDir "_pamap2_inner"
$ZipPath = Join-Path $ArchiveDir "PAMAP2_Dataset.zip"

Write-Host ""
Write-Host "=== ARL-FL | PAMAP2 dataset setup v3 ===" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"

foreach ($dir in @($ProjectRoot, $DataRoot, $ArchiveDir, $RawDir, $ManifestDir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

# -------------------------------------------------------------------------
# 1. Reuse or download the official outer archive
# -------------------------------------------------------------------------
$NeedDownload = $true

if (Test-Path $ZipPath) {
    $ExistingSize = (Get-Item $ZipPath).Length
    if ($ExistingSize -gt 500MB) {
        $NeedDownload = $false
        Write-Host ("[OK] Existing archive found: {0}" -f $ZipPath)
        Write-Host ("     Size: {0:N2} MB" -f ($ExistingSize / 1MB))
    }
    else {
        Write-Warning "Existing archive is unexpectedly small and will be replaced."
        Remove-Item $ZipPath -Force
    }
}

if ($NeedDownload) {
    Write-Host ""
    Write-Host "[1/5] Downloading PAMAP2 from the official UCI repository..."
    try {
        if (Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue) {
            Start-BitsTransfer -Source $DatasetUrl -Destination $ZipPath -DisplayName "PAMAP2 dataset"
        }
        else {
            Invoke-WebRequest -Uri $DatasetUrl -OutFile $ZipPath
        }
    }
    catch {
        Write-Warning "Primary download method failed. Retrying with curl.exe..."
        if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
            throw "Download failed and curl.exe is not available. Original error: $($_.Exception.Message)"
        }
        & curl.exe -L --fail --retry 3 --output $ZipPath $DatasetUrl
        if ($LASTEXITCODE -ne 0) {
            throw "curl.exe failed with exit code $LASTEXITCODE"
        }
    }
}

$ZipSize = (Get-Item $ZipPath).Length
if ($ZipSize -lt 500MB) {
    throw "Outer archive is unexpectedly small ($([math]::Round($ZipSize / 1MB, 2)) MB)."
}
Write-Host ("[OK] Outer archive ready: {0:N2} MB" -f ($ZipSize / 1MB))

# -------------------------------------------------------------------------
# 2. Extract the official outer archive
# -------------------------------------------------------------------------
$InstalledProtocol = Join-Path $PamapDir "Protocol"

if (-not (Test-Path $InstalledProtocol)) {
    Write-Host ""
    Write-Host "[2/5] Extracting the outer UCI archive..."

    if (Test-Path $OuterExtract) {
        Remove-Item $OuterExtract -Recurse -Force
    }
    New-Item -ItemType Directory -Path $OuterExtract -Force | Out-Null

    Expand-Archive -Path $ZipPath -DestinationPath $OuterExtract -Force
    Write-Host "[OK] Outer archive extracted."

    # The UCI package contains another ZIP inside the downloaded ZIP.
    $InnerZip = Get-ChildItem -Path $OuterExtract -Filter "*.zip" -File -Recurse |
        Sort-Object Length -Descending |
        Select-Object -First 1

    if ($null -eq $InnerZip) {
        throw "The outer archive was extracted, but no nested ZIP file was found."
    }

    Write-Host "[OK] Nested archive found:"
    Write-Host "     $($InnerZip.FullName)"
    Write-Host ("     Size: {0:N2} MB" -f ($InnerZip.Length / 1MB))

    # ---------------------------------------------------------------------
    # 3. Extract the nested dataset ZIP
    # ---------------------------------------------------------------------
    Write-Host ""
    Write-Host "[3/5] Extracting the nested PAMAP2 dataset archive..."

    if (Test-Path $InnerExtract) {
        Remove-Item $InnerExtract -Recurse -Force
    }
    New-Item -ItemType Directory -Path $InnerExtract -Force | Out-Null

    Expand-Archive -Path $InnerZip.FullName -DestinationPath $InnerExtract -Force
    Write-Host "[OK] Nested archive extracted."

    $ProtocolDir = Get-ChildItem -Path $InnerExtract -Directory -Recurse -ErrorAction Stop |
        Where-Object { $_.Name -ieq "Protocol" } |
        Select-Object -First 1

    if ($null -eq $ProtocolDir) {
        Write-Host ""
        Write-Host "Top-level nested content:" -ForegroundColor Yellow
        Get-ChildItem -Path $InnerExtract -Force | Format-Table Name, Mode, Length -AutoSize
        throw "Could not find any extracted directory named 'Protocol' inside the nested archive."
    }

    $DatasetRoot = $ProtocolDir.Parent.FullName
    Write-Host "[OK] Located Protocol directory:"
    Write-Host "     $($ProtocolDir.FullName)"
    Write-Host "[OK] Dataset root:"
    Write-Host "     $DatasetRoot"

    if (Test-Path $PamapDir) {
        Remove-Item $PamapDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $PamapDir -Force | Out-Null

    Get-ChildItem -Path $DatasetRoot -Force | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination $PamapDir -Recurse -Force
    }
}
else {
    Write-Host "[OK] Extracted dataset already exists: $PamapDir"
}

# -------------------------------------------------------------------------
# 4. Verify structure and sample row width
# -------------------------------------------------------------------------
Write-Host ""
Write-Host "[4/5] Verifying file structure..."

$ProtocolPath = Join-Path $PamapDir "Protocol"
if (-not (Test-Path $ProtocolPath)) {
    throw "Protocol directory is missing after installation: $ProtocolPath"
}

$ProtocolFiles = @(Get-ChildItem -Path $ProtocolPath -Filter "*.dat" -File | Sort-Object Name)
$OptionalPath = Join-Path $PamapDir "Optional"
$OptionalFiles = @()

if (Test-Path $OptionalPath) {
    $OptionalFiles = @(Get-ChildItem -Path $OptionalPath -Filter "*.dat" -File | Sort-Object Name)
}

if ($ProtocolFiles.Count -lt 9) {
    throw "Expected at least 9 protocol .dat files, but found $($ProtocolFiles.Count)."
}

$FirstLine = Get-Content -Path $ProtocolFiles[0].FullName -TotalCount 1
$ColumnCount = ($FirstLine -split '\s+' | Where-Object { $_ -ne "" }).Count

if ($ColumnCount -ne 54) {
    throw "Expected 54 columns in a PAMAP2 data row, but found $ColumnCount."
}

Write-Host "[OK] Protocol files: $($ProtocolFiles.Count)"
Write-Host "[OK] Optional files: $($OptionalFiles.Count)"
Write-Host "[OK] Sample row columns: $ColumnCount"
Write-Host "[OK] First protocol file: $($ProtocolFiles[0].Name)"

# -------------------------------------------------------------------------
# 5. Record provenance and finish
# -------------------------------------------------------------------------
$SourceFile = Join-Path $PamapDir "SOURCE.txt"
@"
Dataset: PAMAP2 Physical Activity Monitoring
Official source page: $DatasetPage
Official download URL: $DatasetUrl
DOI: $DatasetDoi
License: CC BY 4.0
Installed by: 01_download_pamap2_v3.ps1
Packaging note: official UCI archive contains a nested PAMAP2_Dataset.zip
"@ | Set-Content -Path $SourceFile -Encoding UTF8

$ManifestFile = Join-Path $ManifestDir "pamap2_installation.txt"
@"
dataset=PAMAP2 Physical Activity Monitoring
official_page=$DatasetPage
doi=$DatasetDoi
license=CC BY 4.0
outer_archive_path=$ZipPath
outer_archive_bytes=$ZipSize
raw_path=$PamapDir
protocol_files=$($ProtocolFiles.Count)
optional_files=$($OptionalFiles.Count)
sample_columns=$ColumnCount
installed_utc=$([DateTime]::UtcNow.ToString("o"))
"@ | Set-Content -Path $ManifestFile -Encoding UTF8

Write-Host ""
Write-Host "[5/5] PAMAP2 installation completed successfully." -ForegroundColor Green
Write-Host "Raw dataset: $PamapDir"
Write-Host "Manifest:    $ManifestFile"

# Clean temporary extraction directories only after successful verification.
foreach ($TempDir in @($OuterExtract, $InnerExtract)) {
    if (Test-Path $TempDir) {
        Remove-Item $TempDir -Recurse -Force
    }
}
Write-Host "[OK] Temporary extraction folders removed."
Write-Host ""
