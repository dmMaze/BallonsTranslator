$ErrorActionPreference = "Stop"
$PythonVersion = "3.12.10"
$ProjectRoot = $PSScriptRoot
if ($ProjectRoot) {
    $ProjectRoot = Split-Path $ProjectRoot -Parent
} else {
    $ProjectRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(".")
}
$PyLibsDir = Join-Path $ProjectRoot "ballontrans_pylibs_win"
$BuildDir = Join-Path $ProjectRoot "install_temp"

Write-Host "=== BallonsTranslator Windows Local Installer ==="
Write-Host "Setting up Python $PythonVersion and uv locally..."

# Clean up any existing ballontrans_pylibs_win or install_temp
if (Test-Path $PyLibsDir) {
    Write-Host "Removing existing ballontrans_pylibs_win..."
    Remove-Item -Recurse -Force $PyLibsDir
}
if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}

# Create folders
New-Item -ItemType Directory -Force -Path $PyLibsDir | Out-Null
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

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

Write-Host "=== Local Installation Completed Successfully! ==="
Write-Host "You can now run launch_win.bat to start BallonsTranslator."
