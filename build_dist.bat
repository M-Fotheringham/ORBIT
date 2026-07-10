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
set "FINAL_DIST=%BUILD_DIR%\ORBIT.dist"

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

"%PYTHON%" -m nuitka --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Nuitka is not installed in the active environment.
    echo Install it with:
    echo python -m pip install nuitka ordered-set zstandard
    exit /b 1
)

"%PYTHON%" -c "import PySide6, numpy, pandas, sklearn, tifffile, skimage, joblib" >nul 2>&1
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

set "PYTHONPATH=%ROOT%\src"

rem ------------------------------------------------------------
rem Build
rem ------------------------------------------------------------

echo.
echo Building ORBIT...
echo.

pushd "%ROOT%"

"%PYTHON%" -m nuitka ^
    --mode=standalone ^
    --enable-plugin=pyside6 ^
    --windows-console-mode=disable ^
    "--windows-icon-from-ico=%ICON%" ^
    "--include-data-files=%ICON%=docs/figs/icon_logo.ico" ^
    --include-package=orbit ^
    "--output-dir=%BUILD_DIR%" ^
    --output-filename=ORBIT.exe ^
    --product-name=ORBIT ^
    --file-description="ORBIT Phenotype Viewer" ^
    --file-version=1.0.0.0 ^
    --product-version=1.0.0.0 ^
    --assume-yes-for-downloads ^
    --remove-output ^
    "--report=%BUILD_DIR%\nuitka-report.xml" ^
    "%ENTRY%"

if errorlevel 1 goto :build_failed

rem Nuitka may name the folder after app.py.
rem Rename app.dist to ORBIT.dist for a predictable installer path.

if exist "%BUILD_DIR%\app.dist" (
    if exist "%FINAL_DIST%" rmdir /s /q "%FINAL_DIST%"
    move "%BUILD_DIR%\app.dist" "%FINAL_DIST%" >nul
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