\
@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo   AI Driver Safety - Windows Setup
echo ==========================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py -3.11"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python 3.11 is not installed.
        echo Install Python 3.11 from https://www.python.org/downloads/
        pause
        exit /b 1
    )
    set "PY=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    %PY% -m venv .venv
    if errorlevel 1 goto :error
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :error

echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :error

echo Installing Windows dependencies...
python -m pip install -r requirements-windows.txt
if errorlevel 1 goto :error

echo.
echo Setup complete.
echo Run the application with run_windows.bat
echo.
pause
exit /b 0

:error
echo.
echo SETUP FAILED.
echo Check the message above and try again.
pause
exit /b 1
