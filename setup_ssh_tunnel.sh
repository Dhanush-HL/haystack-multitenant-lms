#!/bin/bash

# HayStack Server Configuration with SSH Tunnel
# Secure connection to Moodle database via SSH tunnel
# Server: hldevlms.westeurope.cloudapp.azure.com

set -e

echo "🔧 Setting up HayStack with SSH Tunnel to Moodle Database"
echo "========================================================="

# Configuration
MOODLE_HOST="hldevlms.westeurope.cloudapp.azure.com"
MOODLE_PORT="65022"
MOODLE_USER="demos"
MOODLE_PASS="A9CC1DA8"
LOCAL_PORT="3307"  # Use 3307 to avoid conflicts with local MySQL

echo "📋 Configuration:"
echo "   Moodle Server: $MOODLE_HOST:$MOODLE_PORT"
echo "   SSH User: $MOODLE_USER"
echo "   Local Tunnel Port: $LOCAL_PORT"
echo ""

# Step 1: Test SSH connection
echo "🔧 Step 1: Testing SSH connection..."
if sshpass -p "$MOODLE_PASS" ssh -p "$MOODLE_PORT" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$MOODLE_USER@$MOODLE_HOST" "echo 'SSH connection successful'" 2>/dev/null; then
    echo "✅ SSH connection working"
else
    echo "❌ SSH connection failed. Check credentials and network."
    exit 1
fi

# Step 2: Create SSH tunnel
echo "🔧 Step 2: Creating SSH tunnel..."

# Kill any existing tunnel on the port
pkill -f "ssh.*$LOCAL_PORT:localhost:3306.*$MOODLE_HOST" 2>/dev/null || true

# Create new SSH tunnel in background
sshpass -p "$MOODLE_PASS" ssh -L $LOCAL_PORT:localhost:3306 -p "$MOODLE_PORT" -N -f \
    -o StrictHostKeyChecking=no \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    "$MOODLE_USER@$MOODLE_HOST"

# Wait for tunnel to establish
sleep 3

# Check if tunnel is running
if netstat -tln | grep ":$LOCAL_PORT " >/dev/null; then
    echo "✅ SSH tunnel established on port $LOCAL_PORT"
else
    echo "❌ SSH tunnel failed to establish"
    exit 1
fi

# Step 3: Create .env file
echo "🔧 Step 3: Creating .env configuration..."

cat > .env << EOF
# HayStack Configuration with SSH Tunnel
# =====================================

# Database Configuration (via SSH tunnel)
DB_HOST=localhost
DB_PORT=$LOCAL_PORT
DB_NAME=demos7_moodle40
DB_USER=demos7
DB_PASSWORD=AF306_A0452

# Multi-tenant Configuration
TENANT_KEY=moodle_demos7_tunnel
ADMIN_USERS=admin,manager,demos

# HayStack Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=2
LOG_LEVEL=info

# Security Configuration
ENABLE_RBAC=true
SESSION_TIMEOUT=30

# Vector Store Configuration
VECTOR_STORE_PATH=./chromadb_data
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Logging Configuration
LOG_DIR=./logs
DEBUG_MODE=false

# SSH Tunnel Configuration (for monitoring)
TUNNEL_LOCAL_PORT=$LOCAL_PORT
TUNNEL_REMOTE_HOST=$MOODLE_HOST
TUNNEL_REMOTE_PORT=$MOODLE_PORT
TUNNEL_USER=$MOODLE_USER
EOF

echo "✅ .env file created"

# Step 4: Test database connection through tunnel
echo "🔧 Step 4: Testing database connection through tunnel..."

python3 -c "
import pymysql
import sys

try:
    connection = pymysql.connect(
        host='localhost',
        port=$LOCAL_PORT,
        user='demos7',
        password='AF306_A0452',
        database='demos7_moodle40'
    )
    print('✅ Database connection through tunnel successful!')
    
    with connection.cursor() as cursor:
        cursor.execute('SELECT COUNT(*) FROM mdl_user')
        count = cursor.fetchone()[0]
        print(f'✅ Found {count} users in Moodle database')
        
        cursor.execute('SELECT shortname, fullname FROM mdl_course WHERE visible=1 LIMIT 5')
        courses = cursor.fetchall()
        print(f'✅ Sample courses: {len(courses)} visible courses found')
        
        cursor.execute('SELECT COUNT(*) FROM mdl_role')
        roles = cursor.fetchone()[0]
        print(f'✅ Found {roles} roles in system')
    
    connection.close()
    print('✅ Database test completed successfully')
    
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    sys.exit(1)
" || {
    echo "❌ Database connection test failed"
    exit 1
}

# Step 5: Test HayStack components
echo "🔧 Step 5: Testing HayStack components..."

python3 -c "
import sys
sys.path.append('./src')

try:
    from database_connector_multitenant import DBConfig
    print('✅ HayStack imports working')
    
    # Test DBConfig creation
    db_config = DBConfig(
        host='localhost',
        port=$LOCAL_PORT,
        user='demos7',
        password='AF306_A0452',
        database='demos7_moodle40'
    )
    print('✅ DBConfig created successfully')
    
except Exception as e:
    print(f'❌ HayStack component test failed: {e}')
    sys.exit(1)
"

echo ""
echo "🎉 SSH Tunnel Setup Complete!"
echo "================================"
echo "✅ SSH tunnel running on port $LOCAL_PORT"  
echo "✅ Database connection verified"
echo "✅ HayStack configuration ready"
echo ""
echo "📋 Next Steps:"
echo "   1. Start HayStack services:"
echo "      ./start_services.sh prod"
echo ""
echo "   2. Test HayStack API:"
echo "      curl http://localhost:8000/health"
echo ""
echo "   3. Monitor tunnel connection:"
echo "      netstat -tln | grep :$LOCAL_PORT"
echo ""
echo "📊 Tunnel Management:"
echo "   Check: ps aux | grep ssh"
echo "   Kill: pkill -f 'ssh.*$LOCAL_PORT:localhost:3306'"
echo "   Restart: bash setup_ssh_tunnel.sh"
echo ""
echo "⚠️  Note: Keep this terminal session alive or run tunnel in background"

# Create tunnel monitoring script
cat > monitor_tunnel.sh << 'EOF'
#!/bin/bash
# Monitor and restart SSH tunnel if needed

TUNNEL_PORT=3307
MOODLE_HOST="hldevlms.westeurope.cloudapp.azure.com"
MOODLE_PORT="65022"
MOODLE_USER="demos"
MOODLE_PASS="A9CC1DA8"

check_tunnel() {
    if netstat -tln | grep ":$TUNNEL_PORT " >/dev/null; then
        echo "✅ Tunnel is running"
        return 0
    else
        echo "❌ Tunnel is down"
        return 1
    fi
}

restart_tunnel() {
    echo "🔧 Restarting SSH tunnel..."
    pkill -f "ssh.*$TUNNEL_PORT:localhost:3306.*$MOODLE_HOST" 2>/dev/null || true
    sleep 2
    
    sshpass -p "$MOODLE_PASS" ssh -L $TUNNEL_PORT:localhost:3306 -p "$MOODLE_PORT" -N -f \
        -o StrictHostKeyChecking=no \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=3 \
        "$MOODLE_USER@$MOODLE_HOST"
    
    sleep 3
    if check_tunnel; then
        echo "✅ Tunnel restarted successfully"
    else
        echo "❌ Failed to restart tunnel"
        exit 1
    fi
}

# Main monitoring loop
if [ "$1" = "--monitor" ]; then
    echo "🔍 Starting tunnel monitoring (Ctrl+C to stop)..."
    while true; do
        if ! check_tunnel; then
            restart_tunnel
        fi
        sleep 30
    done
else
    check_tunnel || restart_tunnel
fi
EOF

chmod +x monitor_tunnel.sh
echo "📝 Created monitor_tunnel.sh for tunnel management"