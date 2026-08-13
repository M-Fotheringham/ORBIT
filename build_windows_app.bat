@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

pushd "%ROOT%"
if errorlevel 1 exit /b 1

echo Synchronizing ORBIT and build dependencies...
uv sync --locked --group build
if errorlevel 1 goto :failed

echo.
echo Staging the verified cpsam_v2 model for the installer...
uv run python scripts\stage_cellpose_model.py
if errorlevel 1 goto :failed

echo.
echo Building the standalone ORBIT application...
uv run --group build pyinstaller --clean --noconfirm ORBIT.spec
if errorlevel 1 goto :failed

if not exist "%ROOT%\dist\ORBIT\ORBIT.exe" (
    echo ERROR: PyInstaller finished, but dist\ORBIT\ORBIT.exe was not found.
    goto :failed
)
if not exist "%ROOT%\dist\ORBIT\cellpose_models\cpsam_v2" (
    echo ERROR: ORBIT.exe was built without the bundled cpsam_v2 model.
    goto :failed
)

echo.
echo Build completed successfully:
echo   %ROOT%\dist\ORBIT\ORBIT.exe
echo The complete dist\ORBIT directory must be included by the installer.
popd
exit /b 0

:failed
echo.
echo ERROR: ORBIT application build failed.
popd
exit /b 1
