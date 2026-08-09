param (
    [string]$PythonVersion = "3.12.10",
    [string]$Tag = "",
    [switch]$UseLocalSource = $false
)

$ErrorActionPreference = "Stop"

# Define directories and names
$BuildDir = Join-Path $PSScriptRoot "..\build_temp"
$DestName = "Ballonstranslator_win_minium"
$DestDir = Join-Path $BuildDir $DestName
$PyLibsDir = Join-Path $DestDir "ballontrans_pylibs_win"
$ZipFile = Join-Path $PSScriptRoot "..\$DestName.zip"

Write-Host "=== Starting BallonsTranslator Minimal Build ==="
Write-Host "Python version: $PythonVersion"
if ($Tag) { Write-Host "Build Tag: $Tag" }
Write-Host "Destination: $DestDir"

# Clean up previous build directories and zip
if (Test-Path $BuildDir) {
    Write-Host "Cleaning up existing build directory..."
    Remove-Item -Recurse -Force $BuildDir
}
if (Test-Path $ZipFile) {
    Write-Host "Cleaning up existing zip file..."
    Remove-Item -Force $ZipFile
}

# Create build directory structure
New-Item -ItemType Directory -Force -Path $PyLibsDir | Out-Null

# Step 1: Get Source Code
if ($UseLocalSource) {
    Write-Host "Using local workspace source..."
    $RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
    
    # Copy project files excluding git, github, temp, and current build dirs
    $ExcludePatterns = @(".git", ".github", "build_temp", "*.zip", ".idea", ".vscode", "__pycache__")
    
    # Simple copy filter
    Get-ChildItem -Path $RepoRoot | Where-Object {
        $name = $_.Name
        $match = $false
        foreach ($pattern in $ExcludePatterns) {
            if ($name -like $pattern) { $match = $true; break }
        }
        -not $match
    } | Copy-Item -Destination $DestDir -Recurse -Force
} else {
    # Determine download URL
    $SourceUrl = "https://github.com/dmMaze/BallonsTranslator/archive/refs/heads/dev.zip"
    if ($Tag) {
        $SourceUrl = "https://github.com/dmMaze/BallonsTranslator/archive/refs/tags/$Tag.zip"
    }
    
    Write-Host "Downloading source code from $SourceUrl ..."
    $SourceZip = Join-Path $BuildDir "source.zip"
    Invoke-WebRequest -Uri $SourceUrl -OutFile $SourceZip
    
    Write-Host "Extracting source code..."
    $ExtractTemp = Join-Path $BuildDir "extract_temp"
    Expand-Archive -Path $SourceZip -DestinationPath $ExtractTemp
    
    # Find the extracted root folder (usually BallonsTranslator-dev or BallonsTranslator-tag)
    $ExtractedFolder = Get-ChildItem -Path $ExtractTemp -Directory | Select-Object -First 1
    if (-not $ExtractedFolder) {
        throw "Could not find extracted folder in source zip"
    }
    
    Write-Host "Moving source files to target destination..."
    Move-Item -Path $ExtractedFolder.FullName -Destination $DestDir
    Remove-Item -Recurse -Force $ExtractTemp
    Remove-Item $SourceZip
}

# Step 2: Download and Extract Python Embeddable
$PythonZipName = "python-$PythonVersion-embed-amd64.zip"
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/$PythonZipName"
$PythonZipPath = Join-Path $BuildDir $PythonZipName

Write-Host "Downloading Python $PythonVersion embeddable..."
Invoke-WebRequest -Uri $PythonUrl -OutFile $PythonZipPath

Write-Host "Extracting Python..."
Expand-Archive -Path $PythonZipPath -DestinationPath $PyLibsDir
Remove-Item $PythonZipPath

# Step 3: Modify python._pth to uncomment "import site"
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

# Step 4: Install pip inside the embeddable Python
Write-Host "Downloading get-pip.py..."
$GetPipUrl = "https://bootstrap.pypa.io/get-pip.py"
$GetPipPath = Join-Path $BuildDir "get-pip.py"
Invoke-WebRequest -Uri $GetPipUrl -OutFile $GetPipPath

Write-Host "Installing pip..."
$PythonExe = Join-Path $PyLibsDir "python.exe"
# Run python.exe get-pip.py to install pip
Start-Process -FilePath $PythonExe -ArgumentList "$GetPipPath --no-warn-script-location" -Wait -NoNewWindow
Remove-Item $GetPipPath

# Step 5: Download and extract uv
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

# Step 6: Create Final Zip Archive
Write-Host "Creating final zip package: $ZipFile ..."
# Compress-Archive stores Windows separators, so write portable entry names explicitly.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$ArchiveRoot = (Resolve-Path $DestDir).Path
$Archive = [System.IO.Compression.ZipFile]::Open(
    $ZipFile,
    [System.IO.Compression.ZipArchiveMode]::Create
)
try {
    Get-ChildItem -LiteralPath $ArchiveRoot -File -Recurse -Force |
        Where-Object { $_.FullName -notmatch '[\\/]__pycache__[\\/]' } |
        ForEach-Object {
            $RelativePath = $_.FullName.Substring($ArchiveRoot.Length + 1).Replace('\', '/')
            $EntryName = "$DestName/$RelativePath"
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $Archive,
                $_.FullName,
                $EntryName,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
} finally {
    $Archive.Dispose()
}

# Clean up build temp directory
Write-Host "Cleaning up temporary build folder..."
Remove-Item -Recurse -Force $BuildDir

Write-Host "=== Build Completed Successfully! ==="
Write-Host "Output: $ZipFile"
