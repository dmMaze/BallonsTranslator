# install_ps_script.ps1
# Auto-installer for BallonTranslator Photoshop Bridge

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  BallonTranslator Photoshop Bridge Auto-Installer" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Definition }
if (-not $scriptDir) { $scriptDir = (Get-Location).Path }

$sourceJsx = Join-Path $scriptDir "BallonTranslator_PS_Bridge.jsx"

if (-not (Test-Path $sourceJsx)) {
    Write-Host "[ERROR] File not found: $sourceJsx" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit..."
    exit 1
}

# 1. Look up Photoshop paths in Windows Registry
$regKeys = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Photoshop.exe",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths\Photoshop.exe",
    "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\Photoshop.exe"
)

$installedCount = 0
$fallbackDirs = @()
$errors = @()

foreach ($key in $regKeys) {
    if (Test-Path $key) {
        $prop = Get-ItemProperty $key -ErrorAction SilentlyContinue
        $appDir = $prop.Path
        if (-not $appDir -and $prop.'(default)') {
            $appDir = [System.IO.Path]::GetDirectoryName($prop.'(default)')
        }
        if ($appDir -and (Test-Path $appDir)) {
            $scriptsDir = Join-Path $appDir "Presets\Scripts"
            $fallbackDirs += $scriptsDir
            if (-not (Test-Path $scriptsDir)) {
                try {
                    New-Item -ItemType Directory -Path $scriptsDir -Force | Out-Null
                } catch {
                    $errors += "Cannot create folder: $scriptsDir ($($_.Exception.Message))"
                }
            }
            $target = Join-Path $scriptsDir "BallonTranslator_PS_Bridge.jsx"
            try {
                Copy-Item -Path $sourceJsx -Destination $target -Force -ErrorAction Stop
                Write-Host "[+] Installed to: $target" -ForegroundColor Green
                $installedCount++
            } catch {
                $errors += "Failed to copy to $target : $($_.Exception.Message)"
                Write-Host "[-] Failed to copy to $target : $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
}

# 2. Fallback disk search if registry not present
if ($installedCount -eq 0) {
    $globPaths = @(
        "C:\Program Files\Adobe\Adobe Photoshop *",
        "D:\Program Files\Adobe\Adobe Photoshop *",
        "E:\Program Files\Adobe\Adobe Photoshop *",
        "F:\Program Files\Adobe\Adobe Photoshop *"
    )
    foreach ($pattern in $globPaths) {
        Get-Item $pattern -ErrorAction SilentlyContinue | ForEach-Object {
            $sDir = Join-Path $_.FullName "Presets\Scripts"
            $fallbackDirs += $sDir
            if (Test-Path $sDir) {
                $target = Join-Path $sDir "BallonTranslator_PS_Bridge.jsx"
                try {
                    Copy-Item -Path $sourceJsx -Destination $target -Force -ErrorAction Stop
                    Write-Host "[+] Installed to: $target" -ForegroundColor Green
                    $installedCount++
                } catch {
                    $errors += "Failed to copy to $target : $($_.Exception.Message)"
                    Write-Host "[-] Failed to copy to $target : $($_.Exception.Message)" -ForegroundColor Red
                }
            }
        }
    }
}

# Result output
Write-Host ""
if ($installedCount -gt 0) {
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host " [SUCCESS] Installed into $installedCount Photoshop version(s)!" -ForegroundColor Green
    Write-Host " In Photoshop: File -> Scripts -> BallonTranslator_PS_Bridge" -ForegroundColor Green
    Write-Host "========================================================" -ForegroundColor Green
    Start-Sleep -Milliseconds 1200
    exit 0
} else {
    Write-Host "========================================================" -ForegroundColor Red
    Write-Host " [ERROR] Installation failed! Details:" -ForegroundColor Red
    foreach ($err in $errors) {
        Write-Host "   - $err" -ForegroundColor Red
    }
    Write-Host "========================================================" -ForegroundColor Red
    Write-Host "[WARNING] Opening target folder in Explorer for manual copy..." -ForegroundColor Yellow
    $openDir = if ($fallbackDirs.Count -gt 0 -and (Test-Path $fallbackDirs[0])) { $fallbackDirs[0] } else { "C:\Program Files\Adobe" }
    if (Test-Path $openDir) { Start-Process explorer.exe $openDir }
    Start-Process explorer.exe ("/select,`"$sourceJsx`"")
    Write-Host "Please manually copy BallonTranslator_PS_Bridge.jsx to your Presets\Scripts folder." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit..."
    exit 1
}
