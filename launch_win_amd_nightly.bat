@REM dependencies\libraries\py310\python.exe F:\repos\BallonsTranslator\ballontranslator
@REM @echo %PATH%

cd %~dp0

@echo off

:: Set the path for PaddleOCR and PyTorch libraries
set "PADDLE_PATH=%~dp0ballontrans_pylibs_win\Lib\site-packages\torch\lib"
set "PATH=%PADDLE_PATH%;%PATH%"

@REM if not defined PYTHON (set PATH=pylibs;pylibs\Scripts;%%PATH%%
set PATH=ballontrans_pylibs_win;ballontrans_pylibs_win\Scripts;PortableGit\cmd;%PATH%
set PYTHON=python.exe

set ERROR_REPORTING=FALSE

mkdir tmp 2>NUL

%PYTHON% -c "" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_pip
echo Couldn't launch python
goto :show_stdout_stderr

:check_pip
%PYTHON% -mpip --help >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :launch
if "%PIP_INSTALLER_LOCATION%" == "" goto :show_stdout_stderr
%PYTHON% "%PIP_INSTALLER_LOCATION%" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :launch
echo Couldn't install pip
goto :show_stdout_stderr


:launch
%PYTHON% launch.py --nightly%*
pause
exit /b


:show_stdout_stderr

echo.
echo exit code: %errorlevel%

for /f %%i in ("tmp\stdout.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stdout:
type tmp\stdout.txt

:show_stderr
for /f %%i in ("tmp\stderr.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stderr:
type tmp\stderr.txt

:endofscript

echo.
echo Launch unsuccessful. Exiting.
pause