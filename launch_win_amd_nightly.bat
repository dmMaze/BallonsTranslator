@echo off
cd /d "%~dp0"

set "VENV_DIR=%~dp0env"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PADDLE_PATH=%VENV_DIR%\Lib\site-packages\torch\lib"
set "PATH=%PADDLE_PATH%;%VENV_DIR%\Scripts;PortableGit\cmd;%PATH%"
set "ERROR_REPORTING=FALSE"

mkdir tmp 2>NUL

if exist "%PYTHON%" goto :check_pip

echo Creating Python virtual environment in "%VENV_DIR%"...
python -m venv "%VENV_DIR%" >tmp\stdout.txt 2>tmp\stderr.txt
if %ERRORLEVEL% == 0 goto :install_requirements
echo Couldn't create Python virtual environment
goto :show_stdout_stderr

:check_pip
"%PYTHON%" -m pip --help >tmp\stdout.txt 2>tmp\stderr.txt
if %ERRORLEVEL% == 0 goto :launch
echo Couldn't launch pip from virtual environment
goto :show_stdout_stderr

:install_requirements
"%PYTHON%" -m pip install --upgrade pip wheel setuptools >tmp\stdout.txt 2>tmp\stderr.txt
if not %ERRORLEVEL% == 0 goto :show_stdout_stderr
"%PYTHON%" -m pip install -r requirements.txt >tmp\stdout.txt 2>tmp\stderr.txt
if not %ERRORLEVEL% == 0 goto :show_stdout_stderr
goto :launch

:launch
"%PYTHON%" launch.py --nightly %*
pause
exit /b

:show_stdout_stderr
echo.
echo exit code: %errorlevel%

for %%i in (tmp\stdout.txt) do set size=%%~zi
if "%size%"=="0" goto :show_stderr
echo.
echo stdout:
type tmp\stdout.txt

:show_stderr
for %%i in (tmp\stderr.txt) do set size=%%~zi
if "%size%"=="0" goto :endofscript
echo.
echo stderr:
type tmp\stderr.txt

:endofscript
echo.
echo Launch unsuccessful. Exiting.
pause
