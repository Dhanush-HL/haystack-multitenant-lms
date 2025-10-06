# HayStack Multi-Tenant Setup Guide

## 🚀 Quick Production Setup

### 1. Server Requirements

```bash
# Ubuntu 20.04+ recommended
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git

# For PostgreSQL support (Totara)
sudo apt install -y libpq-dev

# For MySQL support (Moodle)  
sudo apt install -y default-libmysqlclient-dev
```

### 2. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd haystack-multitenant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create `.env` file:

```bash
# Totara Configuration
TOTARA_DB_HOST=your-totara-host
TOTARA_DB_PORT=5432
TOTARA_DB_NAME=totara_db
TOTARA_DB_USER=totara_user
TOTARA_DB_PASSWORD=your-password

# Moodle Configuration  
MOODLE_DB_HOST=your-moodle-host
MOODLE_DB_PORT=3306
MOODLE_DB_NAME=moodle_db
MOODLE_DB_USER=moodle_user
MOODLE_DB_PASSWORD=your-password

# Application Settings
LOG_LEVEL=INFO
CACHE_TTL_SECONDS=120
MAX_QUERY_RESULTS=1000
```

### 4. Quick Test

```bash
# Test the installation
python demo_architecture.py

# Run comprehensive tests
python test_universal_rbac.py
python test_multitenant_connector.py
```

### 5. Production Usage

```python
from src.tenant_aware_mcp_tools import create_totara_mcp_tools

# Initialize for your LMS
tools = create_totara_mcp_tools(
    db_host=os.getenv('TOTARA_DB_HOST'),
    db_name=os.getenv('TOTARA_DB_NAME'),
    db_user=os.getenv('TOTARA_DB_USER'),
    db_password=os.getenv('TOTARA_DB_PASSWORD'),
    tenant_key="main_tenant",
    admin_users={2}  # Configure your admin user IDs
)

# Get LLM context
context = tools.get_llm_schema_context()

# Execute canonical queries
result = tools.execute_sql(
    "SELECT id, username FROM canon_user LIMIT 10",
    user_id=2
)
```

## 🔧 Advanced Configuration

### Multi-Tenant Setup

```python
from src.database_connector_multitenant import DatabaseConnector, DBConfig
from src.universal_rbac import TenantConfig
from src.tenant_aware_mcp_tools import TenantAwareMCPTools, MCPConfig

# Configure multiple tenants
mcp_config = MCPConfig(default_tenant="tenant_a")
tools = TenantAwareMCPTools(mcp_config)

# Add Totara tenant
totara_db = DBConfig(host="totara.edu", port=5432, database="totara_db", 
                    user="user", password="pass")
totara_tenant = TenantConfig(tenant_key="tenant_a", admin_users={2})
tools.configure_tenant("tenant_a", totara_db, totara_tenant)

# Add Moodle tenant
moodle_db = DBConfig(host="moodle.edu", port=3306, database="moodle_db",
                    user="user", password="pass") 
moodle_tenant = TenantConfig(tenant_key="tenant_b", admin_users={1})
tools.configure_tenant("tenant_b", moodle_db, moodle_tenant)

# Runtime switching
tools.switch_tenant("tenant_a")  # Now using Totara
tools.switch_tenant("tenant_b")  # Now using Moodle
```

## 🛡️ Security Configuration

### RBAC Settings

```python
# Custom tenant configuration
tenant_config = TenantConfig(
    tenant_key="secure_tenant",
    admin_user_ids={1, 2, 10},  # Your admin users
    privileged_roles={"admin", "manager", "editingteacher"}
)
```

### Database Security

- Use read-only database users when possible
- Configure connection SSL/TLS
- Set up database firewall rules
- Regular security auditing

## 📊 Monitoring & Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Monitor cache performance
stats = tools.rbac_service.get_cache_stats()
print(f"RBAC cache: {stats['valid_entries']} entries")
```

## 🚨 Troubleshooting

### Common Issues

1. **Connection Errors**
   ```bash
   # Check database connectivity
   python -c "from src.database_connector_multitenant import *; print('OK')"
   ```

2. **Schema Discovery Issues**
   - Ensure database user has SELECT permissions on INFORMATION_SCHEMA
   - Check table naming conventions match expected patterns

3. **RBAC Permission Errors**
   - Verify user exists in role assignment tables
   - Check admin_user_ids configuration

### Performance Tuning

```python
# Adjust cache settings
rbac_service = UniversalRBACService(
    engine=engine,
    synonyms_map=synonyms,
    tenant_config=tenant_config,
    cache_ttl_seconds=300  # Increase for better performance
)

# Clear caches when needed
tools.clear_caches()
```

## 🔄 Deployment Strategies

### Blue-Green Deployment

1. Deploy new version to green environment
2. Test with `demo_architecture.py`
3. Switch traffic to green environment
4. Keep blue as rollback option

### Rolling Updates

1. Update one tenant at a time
2. Use `tools.switch_tenant()` for gradual migration
3. Monitor performance and errors

---

**Ready for production! 🚀**