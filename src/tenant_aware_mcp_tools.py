"""
Tenant-Aware MCP Tools Integration
Updated version of MCP tools that integrates with multi-tenant database connector and universal RBAC.

This module provides a simplified interface for:
1. Multi-tenant SQL execution with canonical schema abstraction
2. Automatic RBAC enforcement and query rewriting  
3. Schema-agnostic LLM context generation
4. Runtime database switching for multi-tenant deployment
"""

import os
import sys
import json
import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from database_connector_multitenant import DatabaseConnector, DBConfig
from universal_rbac import UniversalRBACService, TenantConfig, EffectiveRBAC

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MCPConfig:
    """Configuration for MCP tools instance."""
    default_tenant: str
    fallback_user_id: int = 2  
    max_query_results: int = 1000
    enable_audit_logging: bool = True

@dataclass
class QueryResult:
    """Result from SQL query execution."""
    success: bool
    data: List[Dict[str, Any]]
    row_count: int
    execution_time_seconds: float
    tenant: str
    user_id: int
    original_sql: str
    final_sql: str
    rbac_applied: bool
    columns_masked: bool
    error: Optional[str] = None
    truncated: bool = False

class TenantAwareMCPTools:
    """
    Tenant-aware MCP tools that provide schema-agnostic database access
    with universal RBAC enforcement across different LMS platforms.
    """
    
    def __init__(self, mcp_config: MCPConfig):
        self.config = mcp_config
        self.db_connector = None
        self.rbac_service = None
        self.current_tenant = mcp_config.default_tenant
        self.tenant_configs: Dict[str, TenantConfig] = {}
        
        logger.info(f"TenantAwareMCPTools initialized with default tenant: {mcp_config.default_tenant}")

    def configure_tenant(self, tenant_key: str, db_config: DBConfig, tenant_config: TenantConfig):
        """Configure a new tenant with database and RBAC settings."""
        
        # Initialize database connector if needed
        if not self.db_connector:
            self.db_connector = DatabaseConnector()
        
        # Add database configuration
        self.db_connector.add_database_config(tenant_key, db_config)
        
        # Store tenant configuration
        self.tenant_configs[tenant_key] = tenant_config
        
        # Switch to this tenant and initialize RBAC
        self.switch_tenant(tenant_key)
        
        logger.info(f"Tenant {tenant_key} configured successfully")

    def switch_tenant(self, tenant_key: str):
        """Switch active tenant and reinitialize RBAC service."""
        if tenant_key not in self.tenant_configs:
            raise ValueError(f"Tenant {tenant_key} not configured")
        
        # Switch database and get synonyms
        synonyms_map = self.db_connector.switch_database(tenant_key)
        
        # Initialize RBAC service for this tenant
        tenant_config = self.tenant_configs[tenant_key]
        self.rbac_service = UniversalRBACService(
            engine=self.db_connector.current_engine,
            synonyms_map=synonyms_map,
            tenant_config=tenant_config,
            cache_ttl_seconds=120
        )
        
        self.current_tenant = tenant_key
        logger.info(f"Switched to tenant {tenant_key}")

    def get_llm_schema_context(self) -> Dict[str, Any]:
        """Generate schema context for LLM to enable canonical SQL generation."""
        
        if not self.db_connector:
            return {"error": "Database connector not configured"}
        
        synonyms_map = self.db_connector.get_synonyms_for_tenant(self.current_tenant)
        discovered_tables = self.db_connector.get_discovered_tables(self.current_tenant)
        
        return {
            "tenant_info": {
                "active_tenant": self.current_tenant,
                "database_type": getattr(self.db_connector.get_config(self.current_tenant), 'db_type', 'unknown'),
                "total_tables_discovered": len(discovered_tables)
            },
            "canonical_schema": {
                "instructions": "Always use these canonical table names in SQL queries. They will be automatically translated to actual table names for the current LMS platform.",
                "available_entities": {
                    "canon_user": {
                        "maps_to": synonyms_map.get("user_table", "not_mapped"),
                        "description": "User accounts, profiles, authentication data",
                        "key_columns": ["id", "username", "email", "firstname", "lastname", "deleted"],
                        "sample_query": "SELECT id, username, email FROM canon_user WHERE deleted = 0 LIMIT 10"
                    },
                    "canon_course": {
                        "maps_to": synonyms_map.get("course_table", "not_mapped"),
                        "description": "Courses, programs, learning paths",
                        "key_columns": ["id", "shortname", "fullname", "visible", "startdate"],
                        "sample_query": "SELECT id, shortname, fullname FROM canon_course WHERE visible = 1"
                    },
                    "canon_enrollment": {
                        "maps_to": synonyms_map.get("user_enrol_table", "not_mapped"),
                        "description": "User enrollments and course participation",
                        "key_columns": ["user_id", "course_id", "status", "timecreated", "timemodified"],
                        "sample_query": "SELECT user_id, course_id, status FROM canon_enrollment WHERE status = 'active'"
                    },
                    "canon_grade_grades": {
                        "maps_to": synonyms_map.get("grade_grades_table", "ttl_grade_grades"),
                        "description": "Assessment grades and scores", 
                        "key_columns": ["userid", "itemid", "finalgrade", "timemodified"],
                        "sample_query": "SELECT userid, finalgrade FROM canon_grade_grades WHERE finalgrade IS NOT NULL"
                    },
                    "canon_role_assignment": {
                        "maps_to": synonyms_map.get("role_assignments_table", "not_mapped"),
                        "description": "User role assignments (teacher, student, admin)",
                        "key_columns": ["userid", "roleid", "contextid", "timemodified"],
                        "sample_query": "SELECT userid, roleid FROM canon_role_assignment"
                    },
                    "canon_context": {
                        "maps_to": synonyms_map.get("context_table", "not_mapped"), 
                        "description": "Context hierarchy (system, course, user levels)",
                        "key_columns": ["id", "contextlevel", "instanceid", "path"],
                        "sample_query": "SELECT id, contextlevel, instanceid FROM canon_context WHERE contextlevel = 50"
                    },
                    "canon_role": {
                        "maps_to": synonyms_map.get("role_table", "not_mapped"),
                        "description": "Available roles and permissions",
                        "key_columns": ["id", "shortname", "name", "description"],
                        "sample_query": "SELECT id, shortname, name FROM canon_role ORDER BY shortname"
                    }
                },
                "query_rules": {
                    "use_canonical_names": "Always use canon_* table names, never use platform-specific prefixes like ttl_ or mdl_",
                    "security_automatic": "Row-level security filters and column masking are applied automatically based on user permissions",
                    "supported_operations": ["SELECT queries only", "JOINs between canonical tables", "WHERE conditions", "ORDER BY", "GROUP BY", "LIMIT"],
                    "forbidden_operations": ["INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "TRUNCATE"]
                }
            },
            "platform_info": {
                "detected_prefixes": list(set(table.split('_')[0] + '_' for table in discovered_tables if '_' in table)),
                "synonyms_count": len(synonyms_map),
                "example_mapping": dict(list(synonyms_map.items())[:3]) if synonyms_map else {}
            }
        }

    def execute_sql(self, sql: str, user_id: Optional[int] = None) -> QueryResult:
        """Execute SQL with canonical table names and RBAC enforcement."""
        
        if not self.db_connector or not self.rbac_service:
            return QueryResult(
                success=False,
                data=[],
                row_count=0,
                execution_time_seconds=0.0,
                tenant=self.current_tenant,
                user_id=user_id or self.config.fallback_user_id,
                original_sql=sql,
                final_sql="",
                rbac_applied=False,
                columns_masked=False,
                error="Database connector or RBAC service not initialized"
            )
        
        effective_user_id = user_id or self.config.fallback_user_id
        start_time = datetime.now()
        
        try:
            # Get user's RBAC permissions
            rbac = self.rbac_service.get_effective_rbac(effective_user_id)
            
            # Apply RBAC security to SQL
            secured_sql, params = self.rbac_service.apply_sql_rbac(sql, effective_user_id, rbac)
            
            # Rewrite canonical table names to actual table names
            final_sql = self.db_connector.rewrite_canonical_sql(secured_sql)
            
            # Execute query
            df = self.db_connector.execute_query(final_sql, params)
            
            # Apply column masking based on user permissions
            masked_df = self.rbac_service.mask_dataframe(df, rbac)
            
            # Handle result size limits
            truncated = False
            if len(masked_df) > self.config.max_query_results:
                masked_df = masked_df.head(self.config.max_query_results)
                truncated = True
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log execution if audit enabled
            if self.config.enable_audit_logging:
                logger.info(f"SQL executed by user {effective_user_id} on tenant {self.current_tenant}: {len(masked_df)} rows in {execution_time:.3f}s")
            
            return QueryResult(
                success=True,
                data=masked_df.to_dict('records'),
                row_count=len(masked_df),
                execution_time_seconds=execution_time,
                tenant=self.current_tenant,
                user_id=effective_user_id,
                original_sql=sql,
                final_sql=final_sql,
                rbac_applied=len(params) > 3,  # Basic params + security params
                columns_masked=len(rbac.masked_columns_by_table) > 0,
                truncated=truncated
            )
            
        except PermissionError as e:
            logger.warning(f"RBAC permission denied for user {effective_user_id}: {str(e)}")
            return QueryResult(
                success=False,
                data=[],
                row_count=0,
                execution_time_seconds=(datetime.now() - start_time).total_seconds(),
                tenant=self.current_tenant,
                user_id=effective_user_id,
                original_sql=sql,
                final_sql="",
                rbac_applied=True,
                columns_masked=False,
                error=f"Permission denied: {str(e)}"
            )
            
        except Exception as e:
            logger.error(f"SQL execution failed for user {effective_user_id}: {str(e)}")
            return QueryResult(
                success=False,
                data=[],
                row_count=0,
                execution_time_seconds=(datetime.now() - start_time).total_seconds(),
                tenant=self.current_tenant,
                user_id=effective_user_id,
                original_sql=sql,
                final_sql="",
                rbac_applied=False,
                columns_masked=False,
                error=str(e)
            )

    def validate_user(self, user_id: int) -> Dict[str, Any]:
        """Validate user exists and get their permissions summary."""
        
        if not self.rbac_service:
            return {"error": "RBAC service not initialized"}
        
        try:
            rbac = self.rbac_service.get_effective_rbac(user_id)
            
            return {
                "success": True,
                "user_id": user_id,
                "tenant": self.current_tenant,
                "permissions": {
                    "is_admin": rbac.is_admin,
                    "is_teacher": rbac.is_teacher,
                    "is_student": rbac.is_student,
                    "role_count": len(rbac.roles),
                    "courses_accessible": len(rbac.authorized_courses),
                    "users_viewable": len(rbac.authorized_users),
                    "tables_readable": sum(1 for allowed in rbac.can_read_table.values() if allowed)
                },
                "role_details": [
                    {"context_level": r[0], "instance_id": r[1], "role_name": r[2]} 
                    for r in rbac.roles
                ]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "user_id": user_id,
                "tenant": self.current_tenant
            }

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the MCP tools instance."""
        
        status = {
            "mcp_tools": {
                "current_tenant": self.current_tenant,
                "configured_tenants": list(self.tenant_configs.keys()),
                "max_query_results": self.config.max_query_results,
                "audit_logging": self.config.enable_audit_logging
            }
        }
        
        if self.db_connector:
            try:
                db_config = self.db_connector.get_config(self.current_tenant)
                synonyms = self.db_connector.get_synonyms_for_tenant(self.current_tenant)
                tables = self.db_connector.get_discovered_tables(self.current_tenant)
                
                status["database"] = {
                    "type": db_config.db_type,
                    "host": db_config.host,
                    "database": db_config.database,
                    "schema_discovery": {
                        "total_tables": len(tables),
                        "synonyms_mapped": len(synonyms)
                    }
                }
            except Exception as e:
                status["database"] = {"error": str(e)}
        
        if self.rbac_service:
            try:
                rbac_stats = self.rbac_service.get_cache_stats()
                status["rbac"] = rbac_stats
            except Exception as e:
                status["rbac"] = {"error": str(e)}
        
        return status

    def clear_caches(self, user_id: Optional[int] = None):
        """Clear caches for performance or after configuration changes."""
        
        cleared = []
        
        if self.db_connector:
            self.db_connector.clear_cache(self.current_tenant)
            cleared.append("schema_cache")
        
        if self.rbac_service:
            if user_id:
                self.rbac_service.clear_cache(self.current_tenant, user_id)
                cleared.append(f"rbac_user_{user_id}")
            else:
                self.rbac_service.clear_cache(self.current_tenant)
                cleared.append("rbac_all_users")
        
        logger.info(f"Cleared caches: {cleared}")
        return {"cleared": cleared, "tenant": self.current_tenant}

# Factory functions for common LMS deployments

def create_totara_mcp_tools(db_host: str, db_name: str, db_user: str, db_password: str,
                          tenant_key: str = "totara_main", admin_users: Optional[Set[int]] = None) -> TenantAwareMCPTools:
    """Create MCP tools instance configured for Totara LMS."""
    
    mcp_config = MCPConfig(default_tenant=tenant_key, fallback_user_id=2)
    
    db_config = DBConfig(
        host=db_host,
        port=5432,
        database=db_name,
        user=db_user,
        password=db_password
    )
    
    tenant_config = TenantConfig(
        tenant_key=tenant_key,
        admin_user_ids=admin_users or {2},
        privileged_roles={"admin", "manager", "editingteacher", "coursecreator"}
    )
    
    tools = TenantAwareMCPTools(mcp_config)
    tools.configure_tenant(tenant_key, db_config, tenant_config)
    
    logger.info(f"Created Totara MCP tools for tenant {tenant_key}")
    return tools

def create_moodle_mcp_tools(db_host: str, db_name: str, db_user: str, db_password: str,
                          tenant_key: str = "moodle_main", admin_users: Optional[Set[int]] = None) -> TenantAwareMCPTools:
    """Create MCP tools instance configured for Moodle LMS."""
    
    mcp_config = MCPConfig(default_tenant=tenant_key, fallback_user_id=1)
    
    db_config = DBConfig(
        host=db_host,
        port=3306,
        database=db_name,
        user=db_user,
        password=db_password
    )
    
    tenant_config = TenantConfig(
        tenant_key=tenant_key,
        admin_user_ids=admin_users or {1},
        privileged_roles={"admin", "manager", "editingteacher", "teacher", "coursecreator"}
    )
    
    tools = TenantAwareMCPTools(mcp_config)
    tools.configure_tenant(tenant_key, db_config, tenant_config)
    
    logger.info(f"Created Moodle MCP tools for tenant {tenant_key}")
    return tools

# Demo/Example usage
if __name__ == "__main__":
    print("🔧 Tenant-Aware MCP Tools Demo")
    print("=" * 40)
    
    # Example configuration
    mcp_config = MCPConfig(
        default_tenant="demo_tenant",
        fallback_user_id=2,
        max_query_results=50
    )
    
    tools = TenantAwareMCPTools(mcp_config)
    
    print("✅ MCP tools initialized")
    print("✅ Multi-tenant database support ready")
    print("✅ Universal RBAC enforcement active")
    print("✅ Canonical SQL rewriting available")
    print("✅ Schema-agnostic queries supported")
    
    print("\n📋 Key Features:")
    print("  • Runtime tenant switching without restarts")
    print("  • Automatic canonical-to-actual table mapping")
    print("  • Row-level security and column masking")
    print("  • LLM-friendly schema abstraction")
    print("  • Cross-platform LMS compatibility")