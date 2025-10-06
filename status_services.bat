@echo off
REM Check Status of HayStack Multi-Tenant Services (Windows)

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "LOG_DIR=%SCRIPT_DIR%logs"

echo.
echo ^📊 HayStack Multi-Tenant Services Status
echo =================================================================
echo.

REM Check for running services by window title
echo ^🔍 Checking for active services...

set "SERVICES_FOUND=0"

REM Check Architecture Demo
tasklist /fi "WindowTitle eq HayStack Architecture Demo*" 2>nul | find /i "cmd.exe" >nul
if not errorlevel 1 (
    echo ✅ Architecture Demo Service - RUNNING
    set /a SERVICES_FOUND+=1
) else (
    echo ❌ Architecture Demo Service - STOPPED
)

REM Check MCP Tools
tasklist /fi "WindowTitle eq HayStack MCP Tools*" 2>nul | find /i "cmd.exe" >nul
if not errorlevel 1 (
    echo ✅ MCP Tools Service - RUNNING
    set /a SERVICES_FOUND+=1
) else (
    echo ❌ MCP Tools Service - STOPPED
)

REM Check Test Runner
tasklist /fi "WindowTitle eq HayStack Test Runner*" 2>nul | find /i "cmd.exe" >nul
if not errorlevel 1 (
    echo ✅ Test Runner Service - RUNNING
    set /a SERVICES_FOUND+=1
) else (
    echo ❌ Test Runner Service - STOPPED
)

echo.
echo ^📋 Process Details:
echo --------------------------------

REM Check for Python processes running our scripts
echo Python processes running HayStack components:

for %%f in (demo_architecture.py tenant_aware_mcp_tools.py test_universal_rbac.py test_multitenant_connector.py) do (
    wmic process where "name='python.exe' and commandline like '%%%%f'" get processid,commandline /format:list 2>nul | findstr "ProcessId\|CommandLine" | findstr /v "^$" >nul
    if not errorlevel 1 (
        echo   📍 %%f processes found
    )
)

echo.
echo ^📁 Log File Status:
echo --------------------------------

if exist "%LOG_DIR%" (
    echo Log directory: %LOG_DIR%
    for %%f in ("%LOG_DIR%\*.log") do (
        if exist "%%f" (
            echo   📄 %%~nxf - %~zf bytes
        )
    )
    
    REM Check if any log files exist
    dir "%LOG_DIR%\*.log" >nul 2>&1
    if errorlevel 1 (
        echo   📁 No active log files found
    )
) else (
    echo   📁 Log directory does not exist
)

echo.
echo ^🔧 System Information:
echo --------------------------------
echo   🐍 Python: 
python --version 2>&1
echo   📦 Key Dependencies:
python -c "import sqlalchemy; print('   ✅ SQLAlchemy:', sqlalchemy.__version__)" 2>nul || echo "   ❌ SQLAlchemy not available"
python -c "import pandas; print('   ✅ Pandas:', pandas.__version__)" 2>nul || echo "   ❌ Pandas not available"

echo.
echo ^📊 Summary:
echo =================================================================
echo   🎯 Active Services: !SERVICES_FOUND!
echo   📁 Log Directory: %LOG_DIR%

if !SERVICES_FOUND! gtr 0 (
    echo   🟢 Status: SERVICES RUNNING
    echo.
    echo ^⚡ Quick Actions:
    echo     🛑 Stop services: stop_services.bat
    echo     📋 View logs: type logs\[service_name].log
) else (
    echo   🔴 Status: ALL SERVICES STOPPED
    echo.
    echo ^⚡ Quick Actions:
    echo     🚀 Start services: start_services.bat [dev^|prod^|test]
    echo     🧪 Run tests: python test_universal_rbac.py
)

echo.
pause