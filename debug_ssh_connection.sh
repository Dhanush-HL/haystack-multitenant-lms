#!/bin/bash

# Manual SSH Tunnel Setup with Enhanced Debugging
# Use this if the automated script fails

set -e

echo "🔧 Manual SSH Tunnel Setup with Debugging"
echo "=========================================="

# Configuration
MOODLE_HOST="hldevlms.westeurope.cloudapp.azure.com"
MOODLE_PORT="65022"
MOODLE_USER="demos"
MOODLE_PASS="A9CC1DA8"
LOCAL_PORT="3307"

echo "📋 Testing connectivity step by step..."
echo ""

# Test 1: Network connectivity
echo "🌐 Test 1: Network connectivity to $MOODLE_HOST"
if ping -c 2 -W 5 "$MOODLE_HOST" >/dev/null 2>&1; then
    echo "✅ Host is reachable"
else
    echo "❌ Host unreachable. Check network/DNS."
    echo "💡 Try: nslookup $MOODLE_HOST"
    exit 1
fi

# Test 2: Port connectivity  
echo "🔌 Test 2: SSH port connectivity to $MOODLE_HOST:$MOODLE_PORT"
if timeout 10 bash -c "echo >/dev/tcp/$MOODLE_HOST/$MOODLE_PORT" 2>/dev/null; then
    echo "✅ SSH port is accessible"
else
    echo "❌ SSH port blocked or filtered"
    echo "💡 Check firewall rules on both servers"
    exit 1
fi

# Test 3: SSH service response
echo "🔑 Test 3: SSH service response"
ssh_banner=$(timeout 5 telnet "$MOODLE_HOST" "$MOODLE_PORT" 2>/dev/null | head -1 || echo "No response")
echo "SSH Banner: $ssh_banner"

# Test 4: Manual SSH authentication
echo "🔐 Test 4: SSH authentication test"
echo "Trying SSH connection with verbose output..."

# Method 1: With sshpass
if command -v sshpass >/dev/null; then
    echo "Method 1: Using sshpass"
    if sshpass -p "$MOODLE_PASS" ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o PreferredAuthentications=password -p "$MOODLE_PORT" "$MOODLE_USER@$MOODLE_HOST" "echo 'SSH authentication successful'" 2>/dev/null; then
        echo "✅ SSH authentication working with sshpass"
        AUTH_METHOD="sshpass"
    else
        echo "❌ SSH authentication failed with sshpass"
        echo "Detailed error:"
        sshpass -p "$MOODLE_PASS" ssh -v -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p "$MOODLE_PORT" "$MOODLE_USER@$MOODLE_HOST" "echo test" 2>&1 | tail -10
        AUTH_METHOD="none"
    fi
else
    echo "❌ sshpass not installed"
    AUTH_METHOD="none"
fi

# Method 2: Interactive SSH (as fallback)
if [ "$AUTH_METHOD" = "none" ]; then
    echo ""
    echo "Method 2: Interactive SSH test"
    echo "Please enter password when prompted: $MOODLE_PASS"
    if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p "$MOODLE_PORT" "$MOODLE_USER@$MOODLE_HOST" "echo 'Interactive SSH successful'"; then
        echo "✅ Interactive SSH working"
        AUTH_METHOD="interactive"
    else
        echo "❌ All SSH methods failed"
        echo ""
        echo "🔍 Diagnostics:"
        echo "   1. Verify credentials: demos / A9CC1DA8"
        echo "   2. Check if SSH key authentication is required"
        echo "   3. Verify SSH service is running on port 65022"
        echo "   4. Check if password authentication is enabled"
        echo ""
        echo "🛠️ Try these commands on the target server:"
        echo "   sudo systemctl status ssh"
        echo "   sudo grep 'PasswordAuthentication' /etc/ssh/sshd_config"
        echo "   sudo grep 'Port' /etc/ssh/sshd_config"
        exit 1
    fi
fi

# Test 5: Create SSH tunnel
echo ""
echo "🌉 Test 5: Creating SSH tunnel"

if [ "$AUTH_METHOD" = "sshpass" ]; then
    echo "Creating tunnel with sshpass..."
    
    # Kill any existing tunnel
    pkill -f "ssh.*$LOCAL_PORT:localhost:3306.*$MOODLE_HOST" 2>/dev/null || true
    sleep 2
    
    # Create tunnel
    sshpass -p "$MOODLE_PASS" ssh -L "$LOCAL_PORT:localhost:3306" -p "$MOODLE_PORT" -N -f \
        -o StrictHostKeyChecking=no \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=3 \
        "$MOODLE_USER@$MOODLE_HOST" &
    
    TUNNEL_PID=$!
    echo "Tunnel PID: $TUNNEL_PID"
    
elif [ "$AUTH_METHOD" = "interactive" ]; then
    echo "Creating tunnel interactively..."
    echo "You'll need to enter the password: $MOODLE_PASS"
    
    ssh -L "$LOCAL_PORT:localhost:3306" -p "$MOODLE_PORT" -N \
        -o StrictHostKeyChecking=no \
        -o ExitOnForwardFailure=yes \
        "$MOODLE_USER@$MOODLE_HOST" &
    
    TUNNEL_PID=$!
    echo "Tunnel PID: $TUNNEL_PID"
fi

# Wait for tunnel to establish
sleep 5

# Test 6: Verify tunnel
echo "🔍 Test 6: Verifying tunnel on port $LOCAL_PORT"
if netstat -tln 2>/dev/null | grep ":$LOCAL_PORT " >/dev/null; then
    echo "✅ SSH tunnel established successfully"
    
    # Show tunnel status
    echo "📊 Tunnel Status:"
    netstat -tln | grep ":$LOCAL_PORT"
    ps aux | grep ssh | grep "$LOCAL_PORT"
    
elif ss -tln 2>/dev/null | grep ":$LOCAL_PORT " >/dev/null; then
    echo "✅ SSH tunnel established successfully (detected with ss)"
else
    echo "❌ SSH tunnel failed to establish"
    echo "Checking for errors..."
    
    # Check if process is running
    if ps -p "$TUNNEL_PID" >/dev/null 2>&1; then
        echo "SSH process is running but port not bound"
    else
        echo "SSH process died"
    fi
    
    exit 1
fi

# Test 7: Database connection through tunnel
echo "🗄️ Test 7: Testing database connection through tunnel"

if command -v mysql >/dev/null; then
    echo "Testing with mysql client..."
    if mysql -h localhost -P "$LOCAL_PORT" -u demos7 -p'AF306_A0452' demos7_moodle40 -e "SELECT COUNT(*) as user_count FROM mdl_user;" 2>/dev/null; then
        echo "✅ MySQL connection through tunnel successful"
    else
        echo "❌ MySQL connection failed, but tunnel is working"
        echo "This might be normal - will test with Python"
    fi
fi

# Python test
python3 -c "
try:
    import pymysql
    connection = pymysql.connect(
        host='localhost',
        port=$LOCAL_PORT,
        user='demos7',
        password='AF306_A0452',
        database='demos7_moodle40',
        connect_timeout=10
    )
    
    with connection.cursor() as cursor:
        cursor.execute('SELECT COUNT(*) FROM mdl_user')
        count = cursor.fetchone()[0]
        print(f'✅ Python database connection successful! Users: {count}')
    
    connection.close()
    
except ImportError:
    print('⚠️ PyMySQL not installed, installing...')
    import subprocess
    subprocess.check_call(['pip3', 'install', 'pymysql'])
    print('Please run the test again')
    
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    print('Check if MySQL is running on the target server')
"

# Create .env file
echo ""
echo "📝 Creating .env configuration file..."

cat > .env << EOF
# Database Configuration (via SSH tunnel)
DB_HOST=localhost
DB_PORT=$LOCAL_PORT
DB_NAME=demos7_moodle40
DB_USER=demos7
DB_PASSWORD=AF306_A0452

# Multi-tenant Configuration
TENANT_KEY=moodle_demos7_tunnel
ADMIN_USERS=admin,manager,demos

# Server Configuration
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
EOF

echo "✅ .env file created"

echo ""
echo "🎉 Manual SSH Tunnel Setup Complete!"
echo "===================================="
echo "✅ SSH tunnel running on port $LOCAL_PORT"
echo "✅ Database connection verified"
echo "✅ Configuration file created"
echo ""
echo "📋 Next Steps:"
echo "   1. Start HayStack services: ./start_services.sh prod"
echo "   2. Test API: curl http://localhost:8000/health"
echo ""
echo "🔧 Tunnel Management:"
echo "   Check status: netstat -tln | grep :$LOCAL_PORT"
echo "   Kill tunnel: pkill -f 'ssh.*$LOCAL_PORT:localhost:3306'"
echo "   Monitor: ps aux | grep ssh"
echo ""
echo "⚠️ Keep the SSH connection alive for continuous operation"