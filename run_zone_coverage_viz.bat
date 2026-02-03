@echo off
REM Zone Coverage Visualization Server Launcher
REM Run from Embankment_Grid directory or Gap_Analysis_EC7 directory

echo Starting Zone Coverage Visualization Server...
echo.

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
