#!/bin/bash
# Stop All HayStack Multi-Tenant Services
# Usage: ./stop_services.sh [--force]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"
LOG_DIR="$SCRIPT_DIR/logs"
FORCE_STOP=${1}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Stopping HayStack Multi-Tenant Services${NC}"
echo "=============================================="

# Function to stop a service gracefully
stop_service() {
    local service_name=$1
    local pid_file="$PID_DIR/${service_name}.pid"
    
    if [ ! -f "$pid_file" ]; then
        echo -e "${YELLOW}⚠️  ${service_name}: No PID file found${NC}"
        return 0
    fi
    
    local pid=$(cat "$pid_file")
    
    if ! kill -0 "$pid" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  ${service_name}: Process not running (PID: $pid)${NC}"
        rm -f "$pid_file"
        return 0
    fi
    
    echo -e "${YELLOW}🛑 Stopping ${service_name} (PID: $pid)...${NC}"
    
    # Try graceful shutdown first
    if [ "$FORCE_STOP" = "--force" ]; then
        kill -9 "$pid" 2>/dev/null
        echo -e "${RED}💀 Force killed ${service_name}${NC}"
    else
        kill -TERM "$pid" 2>/dev/null
        
        # Wait up to 10 seconds for graceful shutdown
        local count=0
        while kill -0 "$pid" 2>/dev/null && [ $count -lt 10 ]; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}   Graceful shutdown timeout, force killing...${NC}"
            kill -9 "$pid" 2>/dev/null
            echo -e "${RED}💀 Force killed ${service_name}${NC}"
        else
            echo -e "${GREEN}✅ ${service_name} stopped gracefully${NC}"
        fi
    fi
    
    rm -f "$pid_file"
}

# Function to cleanup resources
cleanup_resources() {
    echo -e "${YELLOW}🧹 Cleaning up resources...${NC}"
    
    # Clear any remaining PID files
    rm -f "$PID_DIR"/*.pid 2>/dev/null
    
    # Archive old logs if they exist
    if [ -d "$LOG_DIR" ] && [ "$(ls -A "$LOG_DIR" 2>/dev/null)" ]; then
        local timestamp=$(date +"%Y%m%d_%H%M%S")
        local archive_dir="$LOG_DIR/archive_$timestamp"
        mkdir -p "$archive_dir"
        mv "$LOG_DIR"/*.log "$archive_dir/" 2>/dev/null || true
        echo -e "${GREEN}✅ Logs archived to $archive_dir${NC}"
    fi
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Function to show final status
show_final_status() {
    echo ""
    echo -e "${BLUE}📊 Final Status Check:${NC}"
    echo "------------------------"
    
    local any_running=false
    
    # Check for any remaining processes
    if [ -d "$PID_DIR" ]; then
        for pid_file in "$PID_DIR"/*.pid; do
            if [ -f "$pid_file" ]; then
                local service_name=$(basename "$pid_file" .pid)
                local pid=$(cat "$pid_file")
                if kill -0 "$pid" 2>/dev/null; then
                    echo -e "${RED}❌ ${service_name} still running (PID: $pid)${NC}"
                    any_running=true
                else
                    echo -e "${GREEN}✅ ${service_name} stopped${NC}"
                fi
            fi
        done
    fi
    
    if [ "$any_running" = true ]; then
        echo ""
        echo -e "${RED}⚠️  Some services are still running. Use --force to kill them:${NC}"
        echo -e "${YELLOW}   ./stop_services.sh --force${NC}"
        return 1
    else
        echo -e "${GREEN}✅ All services stopped successfully${NC}"
        return 0
    fi
}

# Main execution
main() {
    if [ ! -d "$PID_DIR" ]; then
        echo -e "${YELLOW}⚠️  No services appear to be running (.pids directory not found)${NC}"
        exit 0
    fi
    
    echo -e "${BLUE}PID directory: ${PID_DIR}${NC}"
    if [ "$FORCE_STOP" = "--force" ]; then
        echo -e "${RED}🚨 Force stop mode enabled${NC}"
    fi
    echo ""
    
    # Stop all services
    local services_found=false
    for pid_file in "$PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            services_found=true
            service_name=$(basename "$pid_file" .pid)
            stop_service "$service_name"
        fi
    done
    
    if [ "$services_found" = false ]; then
        echo -e "${YELLOW}⚠️  No running services found${NC}"
    fi
    
    # Cleanup and show status
    cleanup_resources
    show_final_status
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}🎉 All HayStack services stopped successfully!${NC}"
        echo "=============================================="
    else
        echo ""
        echo -e "${RED}⚠️  Some services may still be running${NC}"
        echo "=============================================="
        exit 1
    fi
}

# Handle interruption
trap 'echo -e "\n${RED}🛑 Stop script interrupted${NC}"; exit 1' INT

# Run main function
main