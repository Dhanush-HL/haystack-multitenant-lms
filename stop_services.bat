@echo off
REM Stop All HayStack Multi-Tenant Services (Windows)
REM Usage: stop_services.bat [force]

setlocal EnableDelayedExpansion

set "FORCE_STOP=%1"
set "SCRIPT_DIR=%~dp0"
set "LOG_DIR=%SCRIPT_DIR%logs"
set "PID_DIR=%SCRIPT_DIR%.pids"

echo.
echo ^🛑 Stopping HayStack Multi-Tenant Services
echo =================================================================
echo.

REM Function to stop processes by window title pattern
echo ^📋 Looking for HayStack services...

REM Stop Architecture Demo
echo Stopping Architecture Demo service...
taskkill /f /fi "WindowTitle eq HayStack Architecture Demo*" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Architecture Demo service not found or already stopped
) else (
    echo ✅ Architecture Demo service stopped
)

REM Stop MCP Tools
echo Stopping MCP Tools service...
taskkill /f /fi "WindowTitle eq HayStack MCP Tools*" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  MCP Tools service not found or already stopped
) else (
    echo ✅ MCP Tools service stopped
)

REM Stop Test Runner
echo Stopping Test Runner service...
taskkill /f /fi "WindowTitle eq HayStack Test Runner*" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Test Runner service not found or already stopped
) else (
    echo ✅ Test Runner service stopped
)

REM Force stop option - kill all Python processes running HayStack scripts
if "%FORCE_STOP%"=="force" (
    echo.
    echo ^⚠️  FORCE STOP MODE - Terminating all Python processes...
    
    REM Stop any Python processes running our scripts
    for %%f in (demo_architecture.py tenant_aware_mcp_tools.py test_universal_rbac.py test_multitenant_connector.py) do (
        echo Checking for %%f processes...
        wmic process where "name='python.exe' and commandline like '%%%%f'" get processid /value 2>nul | findstr "ProcessId" >nul
        if not errorlevel 1 (
            echo Terminating %%f processes...
            wmic process where "name='python.exe' and commandline like '%%%%f'" call terminate >nul 2>&1
        )
    )
    
    echo ✅ Force termination completed
)

REM Cleanup operations
echo.
echo ^🧹 Performing cleanup operations...

REM Archive log files with timestamp
if exist "%LOG_DIR%" (
    echo Archiving log files...
    set "TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
    set "TIMESTAMP=!TIMESTAMP: =0!"
    
    if not exist "%LOG_DIR%\archived" mkdir "%LOG_DIR%\archived"
    
    for %%f in ("%LOG_DIR%\*.log") do (
        if exist "%%f" (
            copy "%%f" "%LOG_DIR%\archived\%%~nf_!TIMESTAMP!.log" >nul 2>&1
            del "%%f" >nul 2>&1
        )
    )
    echo ✅ Log files archived to %LOG_DIR%\archived\
)

REM Clean up PID directory
if exist "%PID_DIR%" (
    echo Cleaning up PID files...
    del /q "%PID_DIR%\*" >nul 2>&1
    echo ✅ PID files cleaned
)

REM Clean up temporary Python files
echo Cleaning up Python cache files...
for /d /r "%SCRIPT_DIR%" %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d" >nul 2>&1
)

for /r "%SCRIPT_DIR%" %%f in (*.pyc *.pyo) do (
    if exist "%%f" del "%%f" >nul 2>&1
)
echo ✅ Python cache files cleaned

REM Final status check
echo.
echo ^📊 Final Status Check...
echo --------------------------------

set "ACTIVE_SERVICES=0"
tasklist /fi "WindowTitle eq HayStack*" 2>nul | find /i "HayStack" >nul
if not errorlevel 1 (
    set /a ACTIVE_SERVICES+=1
    echo ⚠️  Some HayStack services may still be running
)

wmic process where "name='python.exe'" get commandline 2>nul | findstr /i "demo_architecture\|tenant_aware\|test_" >nul
if not errorlevel 1 (
    set /a ACTIVE_SERVICES+=1
    echo ⚠️  Some Python HayStack processes may still be running
)

if !ACTIVE_SERVICES! equ 0 (
    echo ✅ All HayStack services successfully stopped
) else (
    echo ⚠️  !ACTIVE_SERVICES! service(s) may still be running
    echo 💡 Try running: stop_services.bat force
)

echo.
echo ^🎉 Service Shutdown Complete!
echo =================================================================
echo ^📋 Cleanup Summary:
echo    📁 Logs archived to: %LOG_DIR%\archived\
echo    🧹 PID files cleaned
echo    🐍 Python cache files removed
echo    📊 !ACTIVE_SERVICES! services potentially still active
echo.

if !ACTIVE_SERVICES! gtr 0 (
    echo ^💡 If services are still running, you can:
    echo    1. Run: stop_services.bat force
    echo    2. Manually check Task Manager for Python processes
    echo    3. Restart your terminal/command prompt
    echo.
)

pause