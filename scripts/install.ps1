$ErrorActionPreference = "Stop"
$PythonVersion = "3.12.10"
$AppName = "BallonsTranslator"

# Detect if we are running locally inside an existing clone
# We are inside a clone if $PSScriptRoot contains "scripts" and parent directory has "ballontranslator" folder
$IsLocal = $false
$ProjectRoot = ""

if ($PSScriptRoot) {
    $ParentDir = Split-Path $PSScriptRoot -Parent
    if (Test-Path (Join-Path $ParentDir "ballontranslator")) {
        $IsLocal = $true
        $ProjectRoot = $ParentDir
    }
}

if (-not $IsLocal) {
    # Web Run (one-liner): Install from scratch in the current directory
    $CurrentDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(".")
    $ProjectRoot = Join-Path $CurrentDir $AppName
    Write-Host "=== BallonsTranslator Web Installer ==="
    Write-Host "Installing fresh version of $AppName to $ProjectRoot..."
} else {
    Write-Host "=== BallonsTranslator Local Installer ==="
    Write-Host "Setting up local environment in $ProjectRoot..."
}

$PyLibsDir = Join-Path $ProjectRoot "ballontrans_pylibs_win"
$BuildDir = Join-Path $ProjectRoot "install_temp"

# If we are NOT local, we must download and extract the source code first
if (-not $IsLocal) {
    # Backup existing folder if present
    if (Test-Path $ProjectRoot) {
        $Timestamp = Get-Date -Format "yyyyMMddHHmmss"
        $BackupRoot = "${ProjectRoot}.backup.${Timestamp}"
        Write-Host "Existing folder found. Moving to $BackupRoot ..."
        Rename-Item -Path $ProjectRoot -NewName (Split-Path $BackupRoot -Leaf)
    }

    # Create directories
    New-Item -ItemType Directory -Force -Path $ProjectRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

    # Download source code
    $SourceUrl = "https://github.com/dmMaze/BallonsTranslator/archive/refs/heads/dev.zip"
    $SourceZip = Join-Path $BuildDir "source.zip"
    Write-Host "Downloading BallonsTranslator source code..."
    Invoke-WebRequest -Uri $SourceUrl -OutFile $SourceZip

    Write-Host "Extracting source code..."
    $ExtractTemp = Join-Path $BuildDir "extract_temp"
    Expand-Archive -Path $SourceZip -DestinationPath $ExtractTemp

    # Move files from extracted subfolder to ProjectRoot
    $ExtractedFolder = Get-ChildItem -Path $ExtractTemp -Directory | Select-Object -First 1
    Get-ChildItem -Path $ExtractedFolder.FullName | Move-Item -Destination $ProjectRoot -Force

    Remove-Item -Recurse -Force $ExtractTemp
    Remove-Item $SourceZip
} else {
    # Create build directory if needed
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
}

# Now, set up the Python environment (common for both modes)
if (Test-Path $PyLibsDir) {
    Write-Host "Removing existing ballontrans_pylibs_win..."
    Remove-Item -Recurse -Force $PyLibsDir
}
New-Item -ItemType Directory -Force -Path $PyLibsDir | Out-Null

# Step 1: Download and Extract Python Embeddable
$PythonZipName = "python-$PythonVersion-embed-amd64.zip"
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/$PythonZipName"
$PythonZipPath = Join-Path $BuildDir $PythonZipName

Write-Host "Downloading Python $PythonVersion embeddable..."
Invoke-WebRequest -Uri $PythonUrl -OutFile $PythonZipPath

Write-Host "Extracting Python..."
Expand-Archive -Path $PythonZipPath -DestinationPath $PyLibsDir
Remove-Item $PythonZipPath

# Step 2: Modify python._pth to enable site-packages and project root path
$VersionParts = $PythonVersion.Split('.')
$MajorMinor = $VersionParts[0] + $VersionParts[1]
$PthFile = Join-Path $PyLibsDir "python$MajorMinor._pth"

if (-not (Test-Path $PthFile)) {
    throw "Could not find path file: $PthFile"
}

Write-Host "Modifying $PthFile to enable site-packages and project root path..."
$PthContent = Get-Content -Path $PthFile
$UpdatedContent = @()
foreach ($line in $PthContent) {
    if ($line.Trim() -eq "#import site") {
        $UpdatedContent += "import site"
    } elseif ($line.Trim() -eq ".") {
        $UpdatedContent += "."
        $UpdatedContent += ".."
    } else {
        $UpdatedContent += $line
    }
}
$UpdatedContent | Set-Content -Path $PthFile

# Step 3: Install pip
Write-Host "Downloading get-pip.py..."
$GetPipUrl = "https://bootstrap.pypa.io/get-pip.py"
$GetPipPath = Join-Path $BuildDir "get-pip.py"
Invoke-WebRequest -Uri $GetPipUrl -OutFile $GetPipPath

Write-Host "Installing pip..."
$PythonExe = Join-Path $PyLibsDir "python.exe"
Start-Process -FilePath $PythonExe -ArgumentList "$GetPipPath --no-warn-script-location" -Wait -NoNewWindow
Remove-Item $GetPipPath

# Step 4: Download and extract uv
$UvUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
$UvZipPath = Join-Path $BuildDir "uv.zip"

Write-Host "Downloading uv..."
Invoke-WebRequest -Uri $UvUrl -OutFile $UvZipPath

Write-Host "Extracting uv..."
$UvTempDir = Join-Path $BuildDir "uv_temp"
Expand-Archive -Path $UvZipPath -DestinationPath $UvTempDir
Remove-Item $UvZipPath

# Copy uv.exe to ballontrans_pylibs_win
$UvExe = Get-ChildItem -Path $UvTempDir -Filter "uv.exe" -Recurse | Select-Object -First 1
if (-not $UvExe) {
    throw "Could not find uv.exe in the downloaded archive"
}
Copy-Item -Path $UvExe.FullName -Destination $PyLibsDir -Force
Remove-Item -Recurse -Force $UvTempDir

# Clean up temp folder
Remove-Item -Recurse -Force $BuildDir

Write-Host "=== Installation Completed Successfully! ==="
Write-Host "You can now run launch_win.bat to start BallonsTranslator."

# Step 5: Automatically launch launch_win.bat
$LaunchBat = Join-Path $ProjectRoot "launch_win.bat"
if (Test-Path $LaunchBat) {
    Write-Host "Launching BallonsTranslator..."
    Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$LaunchBat`"" -WorkingDirectory $ProjectRoot
}
