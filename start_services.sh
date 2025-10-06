#!/bin/bash
# Start All HayStack Multi-Tenant Services
# Usage: ./start_services.sh [environment]
# Environment: dev (default) | prod | test

set -e

ENVIRONMENT=${1:-dev}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PID_DIR="$SCRIPT_DIR/.pids"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create directories
mkdir -p "$LOG_DIR" "$PID_DIR"

echo -e "${BLUE}🚀 Starting HayStack Multi-Tenant Services (${ENVIRONMENT})${NC}"
echo "================================================================="

# Function to start a service
start_service() {
    local service_name=$1
    local command=$2
    local log_file="$LOG_DIR/${service_name}.log"
    local pid_file="$PID_DIR/${service_name}.pid"
    
    echo -e "${YELLOW}Starting ${service_name}...${NC}"
    
    # Check if already running
    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo -e "${GREEN}✅ ${service_name} already running (PID: $(cat "$pid_file"))${NC}"
        return 0
    fi
    
    # Start the service
    nohup $command > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$pid_file"
    
    # Verify it started
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${GREEN}✅ ${service_name} started successfully (PID: $pid)${NC}"
        echo "   Log: $log_file"
    else
        echo -e "${RED}❌ Failed to start ${service_name}${NC}"
        cat "$log_file"
        return 1
    fi
}

# Function to check if virtual environment exists and activate it
setup_python_env() {
    if [ -d "venv" ]; then
        echo -e "${YELLOW}🐍 Activating Python virtual environment...${NC}"
        source venv/bin/activate
        echo -e "${GREEN}✅ Python environment activated${NC}"
    elif [ -d ".venv" ]; then
        echo -e "${YELLOW}🐍 Activating Python virtual environment...${NC}"
        source .venv/bin/activate
        echo -e "${GREEN}✅ Python environment activated${NC}"
    else
        echo -e "${YELLOW}⚠️  No virtual environment found, using system Python${NC}"
    fi
}

# Function to install dependencies if needed
check_dependencies() {
    echo -e "${YELLOW}📦 Checking Python dependencies...${NC}"
    
    if ! python -c "import sqlalchemy, pandas, dataclasses" 2>/dev/null; then
        echo -e "${YELLOW}Installing missing dependencies...${NC}"
        pip install -r requirements.txt
        echo -e "${GREEN}✅ Dependencies installed${NC}"
    else
        echo -e "${GREEN}✅ Dependencies satisfied${NC}"
    fi
}

# Function to run tests before starting services
run_health_checks() {
    if [ "$ENVIRONMENT" != "prod" ]; then
        echo -e "${YELLOW}🔍 Running health checks...${NC}"
        
        # Test RBAC service
        if python test_universal_rbac.py >/dev/null 2>&1; then
            echo -e "${GREEN}✅ RBAC service tests passed${NC}"
        else
            echo -e "${RED}❌ RBAC service tests failed${NC}"
            return 1
        fi
        
        # Test multi-tenant connector
        if python test_multitenant_connector.py >/dev/null 2>&1; then
            echo -e "${GREEN}✅ Multi-tenant connector tests passed${NC}"
        else
            echo -e "${RED}❌ Multi-tenant connector tests failed${NC}"
            return 1
        fi
        
        echo -e "${GREEN}✅ All health checks passed${NC}"
    fi
}

# Main execution
main() {
    echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
    echo -e "${BLUE}Script directory: ${SCRIPT_DIR}${NC}"
    echo -e "${BLUE}Logs directory: ${LOG_DIR}${NC}"
    echo ""
    
    # Setup environment
    setup_python_env
    check_dependencies
    
    # Run health checks (except in production)
    if [ "$ENVIRONMENT" != "prod" ]; then
        run_health_checks
    fi
    
    echo ""
    echo -e "${BLUE}📋 Starting Services...${NC}"
    echo "--------------------------------"
    
    # Start Architecture Demo Service (for development/testing)
    if [ "$ENVIRONMENT" = "dev" ] || [ "$ENVIRONMENT" = "test" ]; then
        start_service "architecture_demo" "python demo_architecture.py"
    fi
    
    # Start MCP Tools Service 
    start_service "mcp_tools" "python -c \"
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
\""
    
    # Start Test Runners (for continuous testing in dev)
    if [ "$ENVIRONMENT" = "dev" ]; then
        start_service "test_runner" "python -c \"
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
\""
    fi
    
    echo ""
    echo -e "${GREEN}🎉 All services started successfully!${NC}"
    echo "================================================================="
    echo -e "${BLUE}📋 Service Status:${NC}"
    
    # Show running services
    for pid_file in "$PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            service_name=$(basename "$pid_file" .pid)
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "   ${GREEN}✅ ${service_name} (PID: $pid)${NC}"
            else
                echo -e "   ${RED}❌ ${service_name} (stopped)${NC}"
            fi
        fi
    done
    
    echo ""
    echo -e "${BLUE}📁 Log Files:${NC}"
    ls -la "$LOG_DIR"/*.log 2>/dev/null || echo "   No log files yet"
    
    echo ""
    echo -e "${BLUE}🛑 To stop all services: ./stop_services.sh${NC}"
    echo -e "${BLUE}📊 To check status: ./status_services.sh${NC}"
    echo -e "${BLUE}📋 To view logs: tail -f logs/[service_name].log${NC}"
}

# Handle interruption
trap 'echo -e "\n${RED}🛑 Startup interrupted${NC}"; exit 1' INT

# Run main function
main