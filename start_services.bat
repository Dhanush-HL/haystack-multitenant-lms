@echo off
REM Start All HayStack Multi-Tenant Services (Windows)
REM Usage: start_services.bat [dev|prod|test]

setlocal EnableDelayedExpansion

set "ENVIRONMENT=%1"
if "%ENVIRONMENT%"=="" set "ENVIRONMENT=dev"

set "SCRIPT_DIR=%~dp0"
set "LOG_DIR=%SCRIPT_DIR%logs"
set "PID_DIR=%SCRIPT_DIR%.pids"

REM Create directories
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%PID_DIR%" mkdir "%PID_DIR%"

echo.
echo ^🚀 Starting HayStack Multi-Tenant Services (%ENVIRONMENT%)
echo =================================================================
echo.

REM Function to check if Python virtual environment exists
if exist "%SCRIPT_DIR%venv\Scripts\activate.bat" (
    echo ^🐍 Activating Python virtual environment...
    call "%SCRIPT_DIR%venv\Scripts\activate.bat"
    echo ✅ Python environment activated
) else if exist "%SCRIPT_DIR%.venv\Scripts\activate.bat" (
    echo ^🐍 Activating Python virtual environment...
    call "%SCRIPT_DIR%.venv\Scripts\activate.bat"
    echo ✅ Python environment activated
) else (
    echo ⚠️  No virtual environment found, using system Python
)

echo.
echo ^📦 Checking Python dependencies...
python -c "import sqlalchemy, pandas" >nul 2>&1
if errorlevel 1 (
    echo Installing missing dependencies...
    pip install -r requirements.txt
    echo ✅ Dependencies installed
) else (
    echo ✅ Dependencies satisfied
)

REM Health checks for non-production
if not "%ENVIRONMENT%"=="prod" (
    echo.
    echo ^🔍 Running health checks...
    
    python test_universal_rbac.py >nul 2>&1
    if errorlevel 1 (
        echo ❌ RBAC service tests failed
        pause
        exit /b 1
    ) else (
        echo ✅ RBAC service tests passed
    )
    
    python test_multitenant_connector.py >nul 2>&1
    if errorlevel 1 (
        echo ❌ Multi-tenant connector tests failed
        pause
        exit /b 1
    ) else (
        echo ✅ Multi-tenant connector tests passed
    )
    
    echo ✅ All health checks passed
)

echo.
echo ^📋 Starting Services...
echo --------------------------------

REM Start Architecture Demo Service (for development/testing)
if "%ENVIRONMENT%"=="dev" (
    echo Starting architecture demo service...
    start /b "HayStack Architecture Demo" cmd /c "python demo_architecture.py > \"%LOG_DIR%\architecture_demo.log\" 2>&1"
    echo ✅ Architecture demo service started
)

REM Start MCP Tools Service
echo Starting MCP tools service...
start /b "HayStack MCP Tools" cmd /c "python -c \"
from src.tenant_aware_mcp_tools import create_totara_mcp_tools
import time
import os
print('🔧 Starting MCP Tools Service...')
try:
    # Example configuration - update with your actual DB settings
    tools = create_totara_mcp_tools(
        db_host=os.getenv('DB_HOST', 'localhost'),
        db_name=os.getenv('DB_NAME', 'totara_db'),
        db_user=os.getenv('DB_USER', 'totara_user'),
        db_password=os.getenv('DB_PASSWORD', 'password'),
        tenant_key='main_tenant'
    )
    print('✅ MCP Tools initialized and ready')
    
    # Keep service running
    while True:
        status = tools.get_status()
        print(f'📊 Service status: {status[\\\"mcp_tools\\\"][\\\"current_tenant\\\"]}')
        time.sleep(30)
        
except KeyboardInterrupt:
    print('🛑 MCP Tools service stopped')
except Exception as e:
    print(f'❌ MCP Tools service error: {e}')
    raise
\" > \"%LOG_DIR%\mcp_tools.log\" 2>&1"
echo ✅ MCP tools service started

REM Start Test Runner for continuous testing (dev mode only)
if "%ENVIRONMENT%"=="dev" (
    echo Starting continuous test runner...
    start /b "HayStack Test Runner" cmd /c "python -c \"
import time
import subprocess
import os
print('🧪 Starting Continuous Test Runner...')
while True:
    try:
        result1 = subprocess.run(['python', 'test_universal_rbac.py'], capture_output=True, text=True)
        result2 = subprocess.run(['python', 'test_multitenant_connector.py'], capture_output=True, text=True)
        
        if result1.returncode == 0 and result2.returncode == 0:
            print('✅ All tests passing')
        else:
            print('❌ Some tests failing - check logs')
            
        time.sleep(120)  # Run tests every 2 minutes
    except KeyboardInterrupt:
        print('🛑 Test runner stopped')
        break
    except Exception as e:
        print(f'⚠️ Test runner error: {e}')
        time.sleep(60)
\" > \"%LOG_DIR%\test_runner.log\" 2>&1"
    echo ✅ Test runner service started
)

echo.
echo ^🎉 All services started successfully!
echo =================================================================
echo ^📋 Service Management:
echo    🛑 To stop all services: stop_services.bat
echo    📊 To check status: status_services.bat  
echo    📋 To view logs: type logs\[service_name].log
echo.
echo ^📁 Log Files Location: %LOG_DIR%
echo ^📋 Available services:
if "%ENVIRONMENT%"=="dev" echo    - Architecture Demo
echo    - MCP Tools Service
if "%ENVIRONMENT%"=="dev" echo    - Test Runner
echo.
echo ^⚡ Services are running in background. Check log files for output.

pause