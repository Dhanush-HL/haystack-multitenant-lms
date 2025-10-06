"""
Universal RBAC Service for Multi-Tenant LMS Deployment
Implements dynamic role-based access control using canonical schema abstraction.
Works with Totara, Moodle, and custom LMS platforms through synonyms mapping.

This module provides:
1. Schema-agnostic role computation using canonical entities
2. Tenant-aware authorized course/user scope calculation 
3. Universal SQL query rewriting with row-level security filters
4. Platform-independent column masking for PII protection
5. Multi-tenant audit logging and caching
"""

from typing import Dict, Any, List, Set, Tuple, Optional, Union
from dataclasses import dataclass
import re
import pandas as pd
import sqlalchemy
import time
import logging
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EffectiveRBAC:
    """Container for a user's effective RBAC permissions and scopes."""
    roles: List[Tuple[int, int, str]]  # (contextlevel, instanceid, role_shortname)
    authorized_courses: Set[int]
    authorized_users: Set[int]  # Users visible to this user
    can_read_table: Dict[str, bool]
    masked_columns_by_table: Dict[str, Set[str]]
    is_admin: bool
    is_teacher: bool
    is_student: bool
    tenant_key: str  # New: track which tenant this RBAC applies to

@dataclass
class RBACCacheEntry:
    """Cache entry for RBAC data with TTL."""
    rbac: EffectiveRBAC
    timestamp: datetime
    ttl_seconds: int
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)

@dataclass 
class TenantConfig:
    """Configuration for tenant-specific RBAC settings"""
    tenant_key: str
    admin_user_ids: Optional[Set[int]] = None  # Configurable admin users per tenant
    privileged_roles: Optional[Set[str]] = None  # Configurable privileged roles per tenant
    
    def __post_init__(self):
        if self.admin_user_ids is None:
            self.admin_user_ids = set()
        if self.privileged_roles is None:
            self.privileged_roles = {"admin", "manager", "editingteacher", "teacher", "coursecreator"}

class UniversalRBACService:
    """
    Universal Role-Based Access Control service for any LMS platform.
    
    Uses canonical schema abstraction to work with Totara, Moodle, and custom LMS.
    Provides dynamic, data-driven RBAC with tenant isolation and schema portability.
    """
    
    def __init__(self, engine, synonyms_map: Dict[str, str], tenant_config: TenantConfig, cache_ttl_seconds: int = 120):
        self.engine = engine
        self.synonyms_map = synonyms_map
        self.tenant_config = tenant_config
        self.cache: Dict[Tuple[str, int], RBACCacheEntry] = {}  # (tenant_key, user_id) -> cache
        self.cache_ttl = cache_ttl_seconds
        
        # Build canonical table access policy (schema-independent)
        self.base_canonical_policy = {
            "canon_course": True,
            "canon_user": True,                 # with masking for non-privileged roles
            "canon_enrollment": True,
            "canon_role_assignment": True,      # scoped to authorized contexts
            "canon_context": True,
            "canon_role": True,
            "canon_grade_grades": True,         # scoped by user/course
            "canon_grade_items": True,
            "canon_course_modules": True,
            "canon_forum": True,
            "canon_forum_posts": True,
            "canon_quiz": True,
            "canon_quiz_attempts": True,
            "canon_log": False,                 # Admin only
            "canon_sessions": False,            # Admin only  
            "canon_config": False,              # Admin only
        }
        
        # Canonical column masking policy (PII protection) 
        self.canonical_column_masking = {
            "canon_user": {"email", "phone1", "phone2", "address", "city", "country", "lastip", "secret", "password"},
            "canon_enrollment": {"timemodified", "modifierid"},
            "canon_grade_grades": {"information"},  # Grade feedback may contain PII
            "canon_log": {"ip"},
        }
        
        logger.info(f"UniversalRBACService initialized for tenant {tenant_config.tenant_key} with cache TTL: {cache_ttl_seconds}s")

    def get_effective_rbac(self, user_id: int) -> EffectiveRBAC:
        """
        Compute effective RBAC permissions for a user using canonical schema.
        
        Uses cached results when available and not expired.
        Computes dynamic scopes from live database queries via synonyms.
        """
        cache_key = (self.tenant_config.tenant_key, user_id)
        
        # Check cache first
        if cache_key in self.cache and not self.cache[cache_key].is_expired:
            logger.debug(f"RBAC cache hit for tenant {self.tenant_config.tenant_key}, user {user_id}")
            return self.cache[cache_key].rbac
            
        logger.info(f"Computing effective RBAC for user {user_id} on tenant {self.tenant_config.tenant_key}")
        start_time = time.time()
        
        try:
            # Compute roles and scopes using canonical queries
            roles = self._get_user_roles(user_id)
            authorized_courses = self._get_authorized_courses(user_id)
            authorized_users = self._get_authorized_users(user_id, roles, authorized_courses)
            
            # Determine role types from canonical roles
            role_names = {role[2].lower() for role in roles}
            privileged_roles = self.tenant_config.privileged_roles or set()
            admin_user_ids = self.tenant_config.admin_user_ids or set()
            is_admin = bool(role_names & privileged_roles) or user_id in admin_user_ids
            is_teacher = bool(role_names & {"editingteacher", "teacher", "coursecreator"})
            is_student = "student" in role_names
            
            # Apply canonical table access policy  
            can_read_table = self._get_canonical_table_access_policy(is_admin, is_teacher, is_student)
            
            # Apply canonical column masking policy
            masked_columns = self._get_canonical_column_masking_policy(is_admin, is_teacher)
            
            rbac = EffectiveRBAC(
                roles=roles,
                authorized_courses=authorized_courses,
                authorized_users=authorized_users,
                can_read_table=can_read_table,
                masked_columns_by_table=masked_columns,
                is_admin=is_admin,
                is_teacher=is_teacher,
                is_student=is_student,
                tenant_key=self.tenant_config.tenant_key
            )
            
            # Cache the result
            self.cache[cache_key] = RBACCacheEntry(
                rbac=rbac,
                timestamp=datetime.now(),
                ttl_seconds=self.cache_ttl
            )
            
            elapsed = time.time() - start_time
            logger.info(f"RBAC computed for user {user_id} in {elapsed:.3f} seconds. Roles: {role_names}, Authorized courses: {len(authorized_courses)}")
            
            return rbac
            
        except Exception as e:
            logger.error(f"RBAC computation failed for user {user_id} on tenant {self.tenant_config.tenant_key}: {e}")
            
            # Fallback RBAC for critical users (configurable per tenant)
            admin_user_ids = self.tenant_config.admin_user_ids or set()
            if user_id in admin_user_ids:
                logger.warning(f"Providing fallback admin RBAC for user {user_id}")
                fallback_rbac = EffectiveRBAC(
                    roles=[(10, 1, "admin")],  # System admin role
                    authorized_courses=set(range(1, 10000)),  # All courses
                    authorized_users=set(range(1, 100000)),   # All users  
                    can_read_table={table: True for table in self.base_canonical_policy.keys()},
                    masked_columns_by_table={},
                    is_admin=True, is_teacher=False, is_student=False,
                    tenant_key=self.tenant_config.tenant_key
                )
                return fallback_rbac
            
            # For other users, re-raise the exception
            raise

    def _get_user_roles(self, user_id: int) -> List[Tuple[int, int, str]]:
        """Get all role assignments for a user using canonical schema."""
        # Build SQL using synonyms mapping to actual table names
        user_table = self.synonyms_map.get("user_table", "canon_user")
        role_assignments_table = self.synonyms_map.get("role_assignments_table", "canon_role_assignment") 
        context_table = self.synonyms_map.get("context_table", "canon_context")
        role_table = self.synonyms_map.get("role_table", "canon_role")
        
        roles_sql = f"""
            SELECT ctx.contextlevel, 
                   COALESCE(ctx.instanceid, 0) as instanceid, 
                   r.shortname
            FROM {role_assignments_table} ra
            JOIN {context_table} ctx ON ctx.id = ra.contextid
            JOIN {role_table} r ON r.id = ra.roleid
            WHERE ra.userid = :user_id
            ORDER BY ctx.contextlevel, ctx.instanceid, r.shortname
        """
        
        with self.engine.begin() as conn:
            result = conn.execute(sqlalchemy.text(roles_sql), {"user_id": user_id})
            roles = [(row[0], row[1], row[2]) for row in result]
            
        logger.debug(f"User {user_id} has {len(roles)} role assignments")
        return roles

    def _get_authorized_courses(self, user_id: int) -> Set[int]:
        """Get all courses a user has access to using canonical schema."""
        
        # Get table names from synonyms
        course_table = self.synonyms_map.get("course_table", "canon_course")
        enrol_table = self.synonyms_map.get("enrol_table", "canon_enrollment") 
        user_enrol_table = self.synonyms_map.get("user_enrol_table", "canon_enrollment")
        role_assignments_table = self.synonyms_map.get("role_assignments_table", "canon_role_assignment")
        context_table = self.synonyms_map.get("context_table", "canon_context")
        role_table = self.synonyms_map.get("role_table", "canon_role")
        
        # Enrolled courses (as student) - try both enrol patterns
        if "ttl_" in user_enrol_table:  # Totara pattern
            enrolled_sql = f"""
                SELECT DISTINCT c.id
                FROM {course_table} c
                JOIN ttl_enrol e ON e.courseid = c.id AND e.status = 0
                JOIN {user_enrol_table} ue ON ue.enrolid = e.id AND ue.status = 0
                WHERE ue.userid = :user_id AND c.visible = 1
            """
        else:  # Direct enrollment pattern (Moodle/Custom)
            enrolled_sql = f"""
                SELECT DISTINCT course_id as id
                FROM {user_enrol_table}
                WHERE user_id = :user_id AND status IN ('active', 'enrolled', '0', 0)
            """
        
        # Course-level role assignments (as teacher/admin)
        role_courses_sql = f"""
            SELECT DISTINCT ctx.instanceid AS course_id
            FROM {role_assignments_table} ra
            JOIN {context_table} ctx ON ctx.id = ra.contextid
            WHERE ra.userid = :user_id AND ctx.contextlevel = 50
        """
        
        # System-level admin roles get access to all visible courses
        system_admin_sql = f"""
            SELECT COUNT(*) as admin_count
            FROM {role_assignments_table} ra
            JOIN {context_table} ctx ON ctx.id = ra.contextid  
            JOIN {role_table} r ON r.id = ra.roleid
            WHERE ra.userid = :user_id 
              AND ctx.contextlevel = 10 
              AND r.shortname IN ('admin', 'manager')
        """
        
        all_courses_sql = f"""
            SELECT DISTINCT id FROM {course_table} WHERE visible = 1
        """
        
        with self.engine.begin() as conn:
            # Get enrolled courses
            try:
                enrolled_result = conn.execute(sqlalchemy.text(enrolled_sql), {"user_id": user_id})
                enrolled = {row[0] for row in enrolled_result}
            except Exception as e:
                logger.warning(f"Enrolled courses query failed: {e}")
                enrolled = set()
            
            # Get role-based course access
            try:
                role_result = conn.execute(sqlalchemy.text(role_courses_sql), {"user_id": user_id})
                role_courses = {row[0] for row in role_result}
            except Exception as e:
                logger.warning(f"Role courses query failed: {e}")
                role_courses = set()
            
            # Check for system admin
            try:
                admin_result = conn.execute(sqlalchemy.text(system_admin_sql), {"user_id": user_id})
                is_system_admin = admin_result.fetchone()[0] > 0
            except Exception as e:
                logger.warning(f"Admin check query failed: {e}")
                is_system_admin = False
            
            # Check tenant-specific admin config
            admin_user_ids = self.tenant_config.admin_user_ids or set()
            if user_id in admin_user_ids:
                is_system_admin = True
                logger.info(f"User {user_id} granted admin access via tenant config")
            
            if is_system_admin:
                # System admins get all courses
                try:
                    all_result = conn.execute(sqlalchemy.text(all_courses_sql))
                    authorized_courses = {row[0] for row in all_result}
                    logger.debug(f"User {user_id} is system admin, authorized for all {len(authorized_courses)} courses")
                except Exception as e:
                    logger.error(f"All courses query failed: {e}")
                    authorized_courses = enrolled | role_courses
            else:
                # Regular users get enrolled + role courses
                authorized_courses = enrolled | role_courses
                logger.debug(f"User {user_id} authorized for {len(authorized_courses)} courses ({len(enrolled)} enrolled, {len(role_courses)} via roles)")
        
        return authorized_courses

    def _get_authorized_users(self, user_id: int, roles: List[Tuple[int, int, str]], 
                            authorized_courses: Set[int]) -> Set[int]:
        """Get all users this user is authorized to see data for using canonical schema."""
        
        role_names = {role[2].lower() for role in roles}
        user_table = self.synonyms_map.get("user_table", "canon_user")
        user_enrol_table = self.synonyms_map.get("user_enrol_table", "canon_enrollment")
        
        # System admins can see all users
        privileged_roles = self.tenant_config.privileged_roles or set()
        admin_user_ids = self.tenant_config.admin_user_ids or set()
        if (role_names & privileged_roles) or user_id in admin_user_ids:
            all_users_sql = f"SELECT DISTINCT id FROM {user_table} WHERE deleted = 0"
            with self.engine.begin() as conn:
                result = conn.execute(sqlalchemy.text(all_users_sql))
                authorized_users = {row[0] for row in result}
                logger.debug(f"User {user_id} (admin) authorized for all {len(authorized_users)} users")
                return authorized_users
        
        # Teachers can see students in their courses
        authorized_users = {user_id}  # Always see yourself
        
        if role_names & {"editingteacher", "teacher"} and authorized_courses:
            # Try different enrollment table patterns
            if "ttl_" in user_enrol_table:  # Totara pattern
                students_sql = f"""
                    SELECT DISTINCT ue.userid
                    FROM {user_enrol_table} ue
                    JOIN ttl_enrol e ON e.id = ue.enrolid AND e.status = 0
                    WHERE ue.status = 0 AND e.courseid IN :course_list
                """
            else:  # Direct pattern (Moodle/Custom)
                students_sql = f"""
                    SELECT DISTINCT user_id as userid
                    FROM {user_enrol_table}
                    WHERE status IN ('active', 'enrolled', '0', 0) AND course_id IN :course_list
                """
            
            course_list = tuple(authorized_courses) if authorized_courses else (0,)
            try:
                with self.engine.begin() as conn:
                    result = conn.execute(sqlalchemy.text(students_sql), {"course_list": course_list})
                    students = {row[0] for row in result}
                    authorized_users.update(students)
            except Exception as e:
                logger.warning(f"Students query failed: {e}")
                
        logger.debug(f"User {user_id} authorized for {len(authorized_users)} users")
        return authorized_users

    def _get_canonical_table_access_policy(self, is_admin: bool, is_teacher: bool, is_student: bool) -> Dict[str, bool]:
        """Get table access permissions based on user roles (canonical tables)."""
        policy = self.base_canonical_policy.copy()
        
        # Admin-only canonical tables
        admin_only_tables = {"canon_log", "canon_sessions", "canon_config"}
        if is_admin:
            # Admins get access to admin-only tables
            for table in admin_only_tables:
                policy[table] = True
        else:
            # Non-admins don't get access to admin-only tables
            for table in admin_only_tables:
                policy[table] = False
                
        return policy

    def _get_canonical_column_masking_policy(self, is_admin: bool, is_teacher: bool) -> Dict[str, Set[str]]:
        """Get column masking policy based on user privileges (canonical columns)."""
        if is_admin:
            # Admins see everything
            return {}
        elif is_teacher:
            # Teachers see some PII but not all
            policy = self.canonical_column_masking.copy()
            # Teachers can see student emails in their courses
            if "canon_user" in policy:
                policy["canon_user"] = policy["canon_user"] - {"email"}
            return policy
        else:
            # Students see minimal PII
            return self.canonical_column_masking.copy()

    def apply_sql_rbac(self, sql: str, user_id: int, rbac: EffectiveRBAC) -> Tuple[str, Dict[str, Any]]:
        """
        Apply RBAC enforcement to SQL query using canonical schema.
        
        Returns modified SQL with injected security filters and parameter bindings.
        Raises PermissionError for unauthorized table access.
        """
        logger.debug(f"Applying RBAC to SQL for user {user_id} on tenant {rbac.tenant_key}")
        
        sql_clean = sql.strip()
        
        # 1. Validate SQL is SELECT only (prevent DML/DDL)
        self._validate_sql_type(sql_clean, user_id)
        
        # 2. Extract and validate table access (canonical tables)
        tables = self._extract_canonical_tables_from_sql(sql_clean)
        self._validate_canonical_table_access(tables, rbac, user_id)
        
        # 3. Prepare RBAC parameters
        params = {
            "current_user_id": user_id,
            "authorized_courses": list(rbac.authorized_courses) if rbac.authorized_courses else [0],
            "authorized_users": list(rbac.authorized_users) if rbac.authorized_users else [user_id]
        }
        
        # 4. Inject security filters for canonical tables
        sql_secured = self._inject_canonical_security_filters(sql_clean, tables, rbac, user_id)
        
        logger.info(f"RBAC applied to SQL for user {user_id}. Tables: {tables}, Filters injected")
        return sql_secured, params

    def _extract_canonical_tables_from_sql(self, sql: str) -> Set[str]:
        """Extract canonical table names from SQL query."""
        # Match FROM and JOIN clauses - look for canonical names
        table_pattern = r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(table_pattern, sql, re.IGNORECASE)
        
        # Filter to canonical table names only
        canonical_tables = set()
        for match in matches:
            table_lower = match.lower()
            # Check if it's a canonical table
            if table_lower.startswith('canon_') or table_lower in self.base_canonical_policy:
                canonical_tables.add(table_lower)
        
        logger.debug(f"Extracted canonical tables from SQL: {canonical_tables}")
        return canonical_tables

    def _validate_sql_type(self, sql: str, user_id: int):
        """Validate SQL is SELECT only - prevent DML/DDL operations."""
        sql_upper = sql.upper().strip()
        
        dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'REPLACE']
        
        for keyword in dangerous_keywords:
            if sql_upper.startswith(keyword + ' ') or f';{keyword} ' in sql_upper or f'; {keyword} ' in sql_upper:
                logger.warning(f"User {user_id} attempted DML/DDL operation: {keyword}")
                raise PermissionError(f"Operation '{keyword}' is not allowed. Only SELECT queries are permitted.")
        
        if ';' in sql and not sql_upper.rstrip().endswith(';'):
            logger.warning(f"User {user_id} attempted multiple SQL statements")
            raise PermissionError("Multiple SQL statements are not allowed for security reasons.")

    def _validate_canonical_table_access(self, tables: Set[str], rbac: EffectiveRBAC, user_id: int):
        """Validate user has access to all canonical tables in query."""
        for table in tables:
            if table not in rbac.can_read_table or not rbac.can_read_table[table]:
                logger.warning(f"User {user_id} attempted to access unauthorized canonical table: {table}")
                raise PermissionError(f"Access to table '{table}' is not allowed for your role")

    def _inject_canonical_security_filters(self, sql: str, tables: Set[str], rbac: EffectiveRBAC, user_id: int) -> str:
        """Inject row-level security filters for canonical tables into SQL query."""
        logger.debug(f"Injecting security filters for canonical tables: {tables}")
        
        where_conditions = []
        
        # Course scoping for canonical course table
        if rbac.authorized_courses and "canon_course" in tables:
            alias = self._find_table_alias(sql, "canon_course") or "canon_course"
            where_conditions.append(f"{alias}.id IN :authorized_courses")
            
        # User scoping for canonical grade tables (students see only their own)
        if rbac.is_student and "canon_grade_grades" in tables:
            alias = self._find_table_alias(sql, "canon_grade_grades") or "canon_grade_grades"
            where_conditions.append(f"{alias}.userid = :current_user_id")
            
        # User scoping for canonical enrollment table (students see only their own)
        if rbac.is_student and "canon_enrollment" in tables:
            alias = self._find_table_alias(sql, "canon_enrollment") or "canon_enrollment"
            where_conditions.append(f"{alias}.user_id = :current_user_id")
            
        # User scoping for canonical user table (limit to authorized users)
        if "canon_user" in tables and not rbac.is_admin:
            alias = self._find_table_alias(sql, "canon_user") or "canon_user"
            where_conditions.append(f"{alias}.id IN :authorized_users")
        
        # Integrity predicates for canonical tables
        if "canon_user" in tables:
            alias = self._find_table_alias(sql, "canon_user") or "canon_user"
            if f"{alias}.deleted" not in sql.lower():
                where_conditions.append(f"{alias}.deleted = 0")
        
        if "canon_course" in tables:
            alias = self._find_table_alias(sql, "canon_course") or "canon_course"
            if f"{alias}.visible" not in sql.lower():
                where_conditions.append(f"{alias}.visible = 1")
        
        # Apply all WHERE conditions
        if where_conditions:
            return self._inject_where_conditions(sql, where_conditions)
        
        return sql

    def _inject_where_conditions(self, sql: str, conditions: List[str]) -> str:
        """Inject WHERE conditions into SQL, handling proper placement."""
        if not conditions:
            return sql
            
        combined_condition = " AND ".join(conditions)
        
        if re.search(r'\bWHERE\b', sql, re.IGNORECASE):
            # Add to existing WHERE clause
            sql_modified = re.sub(
                r'\bWHERE\b', 
                f"WHERE {combined_condition} AND ", 
                sql, count=1, flags=re.IGNORECASE
            )
        else:
            # Insert WHERE clause before ORDER BY/GROUP BY/HAVING/LIMIT
            insert_pattern = r'(\s+)(ORDER\s+BY|GROUP\s+BY|HAVING|LIMIT)\b'
            match = re.search(insert_pattern, sql, re.IGNORECASE)
            if match:
                insertion_point = match.start()
                sql_modified = (sql[:insertion_point] + 
                              f" WHERE {combined_condition}" + 
                              sql[insertion_point:])
            else:
                sql_modified = sql.rstrip() + f" WHERE {combined_condition}"
        
        return sql_modified

    def _find_table_alias(self, sql: str, table: str) -> Optional[str]:
        """Find table alias in SQL query."""
        import re
        
        # Pattern to find table alias: FROM table AS alias or FROM table alias
        patterns = [
            rf'\bFROM\s+{re.escape(table)}\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b',
            rf'\bJOIN\s+{re.escape(table)}\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sql, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

    def mask_dataframe(self, df: pd.DataFrame, rbac: EffectiveRBAC) -> pd.DataFrame:
        """Apply column masking to DataFrame results based on canonical schema."""
        if not rbac.masked_columns_by_table:
            return df
        
        masked_df = df.copy()
        
        for canonical_table, masked_columns in rbac.masked_columns_by_table.items():
            # Apply masking if any of the masked columns exist in the DataFrame
            for column in masked_columns:
                if column in masked_df.columns:
                    masked_df[column] = "***MASKED***"
                    logger.debug(f"Masked column '{column}' for user on tenant {rbac.tenant_key}")
        
        return masked_df

    def clear_cache(self, tenant_key: Optional[str] = None, user_id: Optional[int] = None):
        """Clear RBAC cache entries."""
        if tenant_key and user_id:
            # Clear specific user cache
            cache_key = (tenant_key, user_id)
            self.cache.pop(cache_key, None)
            logger.info(f"Cleared RBAC cache for tenant {tenant_key}, user {user_id}")
        elif tenant_key:
            # Clear all cache for tenant
            keys_to_remove = [k for k in self.cache.keys() if k[0] == tenant_key]
            for key in keys_to_remove:
                del self.cache[key]
            logger.info(f"Cleared RBAC cache for tenant {tenant_key} ({len(keys_to_remove)} entries)")
        else:
            # Clear all cache
            cache_size = len(self.cache)
            self.cache.clear()
            logger.info(f"Cleared all RBAC cache ({cache_size} entries)")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get RBAC cache statistics."""
        now = datetime.now()
        valid_entries = sum(1 for entry in self.cache.values() if not entry.is_expired)
        expired_entries = len(self.cache) - valid_entries
        
        tenant_stats = {}
        for (tenant_key, user_id), entry in self.cache.items():
            if tenant_key not in tenant_stats:
                tenant_stats[tenant_key] = {"users": 0, "expired": 0}
            tenant_stats[tenant_key]["users"] += 1
            if entry.is_expired:
                tenant_stats[tenant_key]["expired"] += 1
        
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cache_ttl_seconds": self.cache_ttl,
            "tenant_stats": tenant_stats
        }