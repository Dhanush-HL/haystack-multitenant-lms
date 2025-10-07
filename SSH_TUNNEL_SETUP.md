# SSH Tunnel Approach - Quick Setup Guide

## 🎯 Why SSH Tunnel is Better for Your Setup

Based on your environment:
- ✅ **Secure**: No MySQL exposed to internet
- ✅ **No server changes needed**: Uses existing SSH access
- ✅ **Production ready**: Enterprise-grade security
- ✅ **Easy maintenance**: Simple to monitor and restart

## 🚀 Quick Setup (Copy-Paste Ready)

### Step 1: Run the Setup Script
```bash
# On your HayStack server
cd ~/Haystack/Haystack_new/haystack-multitenant-lms

# Download and run the setup script
curl -O https://raw.githubusercontent.com/Dhanush-HL/haystack-multitenant-lms/main/setup_ssh_tunnel.sh
chmod +x setup_ssh_tunnel.sh
bash setup_ssh_tunnel.sh
```

### Step 2: Start HayStack Services
```bash
# After tunnel is established
./start_services.sh prod

# Test API
curl http://localhost:8000/health
```

### Step 3: Get Your Server IP for Moodle
```bash
# Get public IP to configure in Moodle
curl ifconfig.me
echo "Configure Moodle to use: http://$(curl -s ifconfig.me):8000"
```

## 🔧 Manual Setup (If Needed)

### Create SSH Tunnel
```bash
# Install sshpass if not available
sudo apt-get update && sudo apt-get install -y sshpass

# Create tunnel (port 3307 to avoid conflicts)
sshpass -p "A9CC1DA8" ssh -L 3307:localhost:3306 -p 65022 -N -f \
    -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=60 \
    demos@hldevlms.westeurope.cloudapp.azure.com

# Verify tunnel
netstat -tln | grep :3307
```

### Create .env File
```bash
cat > .env << 'EOF'
# Database via SSH Tunnel
DB_HOST=localhost
DB_PORT=3307
DB_NAME=demos7_moodle40
DB_USER=demos7
DB_PASSWORD=AF306_A0452

# Tenant Configuration  
TENANT_KEY=moodle_demos7_tunnel
ADMIN_USERS=admin,manager,demos

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=2
LOG_LEVEL=info
ENABLE_RBAC=true
SESSION_TIMEOUT=30
VECTOR_STORE_PATH=./chromadb_data
EMBEDDING_MODEL=all-MiniLM-L6-v2
LOG_DIR=./logs
DEBUG_MODE=false
EOF
```

### Test Database Connection
```bash
python3 -c "
import pymysql
connection = pymysql.connect(
    host='localhost', port=3307,
    user='demos7', password='AF306_A0452',
    database='demos7_moodle40'
)
with connection.cursor() as cursor:
    cursor.execute('SELECT COUNT(*) FROM mdl_user')
    print(f'Users: {cursor.fetchone()[0]}')
connection.close()
print('✅ Database connection successful!')
"
```

## 🔍 Troubleshooting

### Tunnel Issues
```bash
# Check if tunnel is running
ps aux | grep ssh
netstat -tln | grep :3307

# Restart tunnel
pkill -f "ssh.*3307:localhost:3306"
# Then recreate tunnel command above
```

### Database Issues
```bash
# Test SSH connection first
ssh -p 65022 demos@hldevlms.westeurope.cloudapp.azure.com "echo 'SSH OK'"

# Test MySQL on Moodle server
ssh -p 65022 demos@hldevlms.westeurope.cloudapp.azure.com "mysql -u demos7 -p'AF306_A0452' demos7_moodle40 -e 'SELECT COUNT(*) FROM mdl_user;'"
```

### HayStack Service Issues
```bash
# Check logs
tail -f logs/haystack_production.log

# Test individual components
python3 test_universal_rbac.py
python3 test_multitenant_connector.py
```

## 🎯 Expected Results

Once working:
1. ✅ SSH tunnel on port 3307
2. ✅ HayStack API at `http://your-ip:8000/health` 
3. ✅ Database queries working: `curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"How many users?","user_id":"admin","tenant_key":"moodle_demos7_tunnel"}'`
4. ✅ Ready for Moodle integration

## 🔄 Production Automation

The setup script creates `monitor_tunnel.sh` for production:
```bash
# Run in background to auto-restart tunnel
./monitor_tunnel.sh --monitor &

# Or check/restart manually
./monitor_tunnel.sh
```

This approach gives you maximum security while keeping setup simple! 🛡️