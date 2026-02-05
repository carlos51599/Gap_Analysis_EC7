@echo off
REM Zone Coverage Visualization Server Launcher
REM Run from Embankment_Grid directory or Gap_Analysis_EC7 directory

REM Check if Windows Terminal is available and use it for better emoji/color support
where wt >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    REM Windows Terminal found - relaunch in it if not already in wt
    if "%WT_SESSION%"=="" (
        start "" wt new-tab -d . cmd /k call "%~nx0"
        exit /b
    )
)

REM Enable UTF-8 support for emojis and special characters
chcp 65001 >nul 2>&1

echo Starting Zone Coverage Visualization Server...
echo.

REM Open browser after a short delay (start in background)
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:5051"

REM Check if we're in the right directory
if exist "zone_coverage_viz\server.py" (
    echo Found server in current directory
    python -m zone_coverage_viz.server
) else if exist "Main\Gap_Analysis_EC7\zone_coverage_viz\server.py" (
    echo Found server in Main\Gap_Analysis_EC7
    cd Main\Gap_Analysis_EC7
    python -m zone_coverage_viz.server
) else (
    echo ERROR: Cannot find zone_coverage_viz\server.py
    echo Please run this script from Embankment_Grid or Gap_Analysis_EC7 directory
    pause
    exit /b 1
)
