@echo off
setlocal EnableExtensions

rem ============================================================
rem ORBIT Nuitka build script
rem Uses Python from the currently active environment.
rem ============================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "PYTHON=python"
set "ENTRY=%ROOT%\src\orbit\app.py"
set "ICON=%ROOT%\docs\figs\icon_logo.ico"
set "BUILD_DIR=%ROOT%\build"
set "NUITKA_DIST=%BUILD_DIR%\app.dist"
set "FINAL_DIST=%BUILD_DIR%\ORBIT.dist"
set "APP_VERSION=1.1.1"
set "FILE_VERSION=1.1.1.0"

if defined PYTHONPATH (
    set "PYTHONPATH=%ROOT%\src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%ROOT%\src"
)

rem ------------------------------------------------------------
rem Validate Python and required files
rem ------------------------------------------------------------

where "%PYTHON%" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not available in the current environment.
    echo Activate your environment before running this script.
    exit /b 1
)

if not exist "%ENTRY%" (
    echo ERROR: Application entry point not found:
    echo %ENTRY%
    exit /b 1
)

if not exist "%ICON%" (
    echo ERROR: Application icon not found:
    echo %ICON%
    exit /b 1
)

echo.
"%PYTHON%" -c "import sys; print('Using Python:', sys.executable)"
if errorlevel 1 exit /b 1

echo.
echo Checking Nuitka installation...
"%PYTHON%" -m nuitka --version
if errorlevel 1 (
    echo.
    echo ERROR: Nuitka is not installed in the active environment.
    echo Install it with:
    echo python -m pip install nuitka ordered-set zstandard
    exit /b 1
)

echo.
echo Checking ORBIT and runtime dependencies...
"%PYTHON%" -c "import orbit.app, orbit.gui.fov_viewer, PySide6, numpy, pandas, scipy, sklearn, tifffile, skimage, joblib, qptifffile, cellpose, torch; print('Dependency check passed.')"
if errorlevel 1 (
    echo.
    echo ERROR: One or more ORBIT dependencies are unavailable.
    echo Install the project in the active environment with:
    echo python -m pip install -e .
    exit /b 1
)

rem ------------------------------------------------------------
rem Prepare output
rem ------------------------------------------------------------

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

if exist "%FINAL_DIST%" (
    echo Removing previous ORBIT distribution...
    rmdir /s /q "%FINAL_DIST%"
)

if exist "%NUITKA_DIST%" (
    echo Removing previous Nuitka distribution...
    rmdir /s /q "%NUITKA_DIST%"
)

rem ------------------------------------------------------------
rem Build
rem ------------------------------------------------------------

echo.
echo Building ORBIT %APP_VERSION%...
echo This can take several minutes. Nuitka will print progress below.
echo.

pushd "%ROOT%"
if errorlevel 1 (
    echo ERROR: Could not enter repository directory:
    echo %ROOT%
    exit /b 1
)

"%PYTHON%" -m nuitka ^
    --standalone ^
    --enable-plugin=pyside6 ^
    --windows-console-mode=disable ^
    "--windows-icon-from-ico=%ICON%" ^
    "--include-data-files=%ICON%=docs/figs/icon_logo.ico" ^
    --include-package=orbit ^
    --include-package=cellpose ^
    "--output-dir=%BUILD_DIR%" ^
    --output-filename=ORBIT.exe ^
    "--company-name=Michael Fotheringham" ^
    "--product-name=ORBIT" ^
    "--file-description=ORBIT Phenotype Viewer" ^
    "--file-version=%FILE_VERSION%" ^
    "--product-version=%FILE_VERSION%" ^
    --assume-yes-for-downloads ^
    --remove-output ^
    "--report=%BUILD_DIR%\nuitka-report.xml" ^
    "%ENTRY%"

if errorlevel 1 goto :build_failed

rem Nuitka may name the folder after app.py.
rem Rename app.dist to ORBIT.dist for a predictable installer path.

if exist "%NUITKA_DIST%" (
    if exist "%FINAL_DIST%" rmdir /s /q "%FINAL_DIST%"
    move "%NUITKA_DIST%" "%FINAL_DIST%" >nul
)

if not exist "%FINAL_DIST%\ORBIT.exe" (
    echo.
    echo ERROR: Build completed, but ORBIT.exe was not found:
    echo %FINAL_DIST%\ORBIT.exe
    popd
    exit /b 1
)

echo.
echo ============================================================
echo Build completed successfully.
echo ============================================================
echo.
echo Executable:
echo %FINAL_DIST%\ORBIT.exe
echo.
echo Test the executable before creating the installer.
echo Then compile orbit_installer_setup.iss with Inno Setup.
echo.

popd
exit /b 0

:build_failed
echo.
echo ============================================================
echo ERROR: Nuitka build failed.
echo ============================================================
echo.
popd
exit /b 1
