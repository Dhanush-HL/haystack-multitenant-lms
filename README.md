# HayStack Multi-Tenant LMS Analytics Platform

🚀 **True database portability**: One codebase works across Totara, Moodle, and custom LMS platforms with runtime tenant switching.

## 🌟 Key Features

- **🔄 Runtime Database Switching**: Change databases without application restarts
- **🛡️ Universal RBAC**: Schema-agnostic role-based access control
- **📊 Canonical SQL**: LLM-friendly database abstraction
- **🏢 Multi-Tenant**: Full tenant isolation with shared infrastructure  
- **🔍 Dynamic Discovery**: Automatic adaptation to new LMS platforms
- **⚡ High Performance**: Per-tenant caching and connection pooling

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Interface Layer                      │
│  (Writes canonical SQL using canon_* table names)          │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│              Tenant-Aware MCP Tools                        │
│  • Schema context generation for LLMs                      │
│  • Canonical SQL execution with RBAC                       │
│  • Multi-tenant query orchestration                        │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│              Universal RBAC Service                        │
│  • Schema-agnostic permission enforcement                  │
│  • SQL security filter injection                          │
│  • Column masking for PII protection                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│          Multi-Tenant Database Connector                   │
│  • Runtime database switching                              │
│  • Schema introspection & canonical mapping                │
│  • SQL rewriting (canon_user → ttl_user/mdl_user)         │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                Database Layer                              │
│  Totara (ttl_*)  │  Moodle (mdl_*)  │  Custom (users)    │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd haystack-multitenant

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```python
from src.tenant_aware_mcp_tools import create_totara_mcp_tools, create_moodle_mcp_tools

# Configure Totara tenant
totara_tools = create_totara_mcp_tools(
    db_host="totara.example.com",
    db_name="totara_db", 
    db_user="totara_user",
    db_password="password",
    tenant_key="university_a",
    admin_users={2}
)

# Configure Moodle tenant  
moodle_tools = create_moodle_mcp_tools(
    db_host="moodle.example.com",
    db_name="moodle_db",
    db_user="moodle_user", 
    db_password="password",
    tenant_key="university_b",
    admin_users={1}
)
```

### 3. Usage Examples

```python
# Get schema context for LLM
context = totara_tools.get_llm_schema_context()
print("Available canonical entities:", context["canonical_schema"]["available_entities"])

# Execute canonical SQL (works on any LMS platform)
result = totara_tools.execute_sql(
    sql="SELECT id, username, email FROM canon_user WHERE deleted = 0 LIMIT 10",
    user_id=2
)

# Switch tenant at runtime (no restart needed!)
moodle_tools.switch_tenant("university_b")
result = moodle_tools.execute_sql(
    sql="SELECT id, username, email FROM canon_user WHERE deleted = 0 LIMIT 10", 
    user_id=1
)
# Same SQL, different database - it just works! 🎉
```

## 📊 Canonical Schema

The system uses canonical table names that work across all LMS platforms:

| Canonical Entity | Totara | Moodle | Custom LMS |
|------------------|--------|--------|------------|
| `canon_user` | `ttl_user` | `mdl_user` | `users` |
| `canon_course` | `ttl_course` | `mdl_course` | `courses` |
| `canon_enrollment` | `ttl_user_enrolments` | `mdl_user_enrolments` | `enrollments` |
| `canon_role_assignment` | `ttl_role_assignments` | `mdl_role_assignments` | `user_roles` |

## 🔒 Security Features

- **Row-Level Security**: Automatic filtering based on user permissions
- **Column Masking**: PII protection for non-privileged users
- **Role-Based Access**: Students see own data, teachers see their courses, admins see all
- **SQL Injection Prevention**: Parameterized queries and validation
- **Tenant Isolation**: Complete separation between tenant data

## 🧪 Testing

```bash
# Run comprehensive RBAC tests
python test_universal_rbac.py

# Run multi-tenant connector tests  
python test_multitenant_connector.py

# Run architecture demo
python demo_architecture.py
```

## 📁 Project Structure

```
haystack-multitenant/
├── src/
│   ├── database_connector_multitenant.py  # Multi-tenant DB connector
│   ├── universal_rbac.py                  # Schema-agnostic RBAC
│   ├── tenant_aware_mcp_tools.py         # MCP integration layer
│   └── config.py                         # Configuration
├── test_universal_rbac.py                # RBAC test suite
├── test_multitenant_connector.py         # DB connector tests
├── demo_architecture.py                 # Architecture demonstration
├── requirements.txt                      # Python dependencies
└── README.md                            # This file
```

## 🌍 Real-World Use Cases

- **🏫 Multi-University SaaS**: Each university keeps their LMS, shared analytics
- **🔀 LMS Migration**: Switch from Totara to Moodle without code changes
- **🏢 Corporate Training**: Support multiple subsidiaries with different LMS platforms
- **☁️ Cloud Deployment**: Dynamic tenant provisioning with different backends
- **📈 Unified Reporting**: Single dashboard for data from multiple LMS platforms

## 🎯 Benefits

✅ **True Database Portability** - One codebase works everywhere  
✅ **Zero-Downtime Switching** - Runtime tenant changes  
✅ **LLM-Friendly** - Canonical schema abstraction  
✅ **Enterprise Security** - Universal RBAC with tenant isolation  
✅ **High Performance** - Connection pooling and caching  
✅ **Future-Proof** - Easy adaptation to new LMS platforms  

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**🚀 Achievement: "Change DB and it works!" - True database portability realized.**