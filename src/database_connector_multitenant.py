"""
Multi-Tenant Database Connector Module
Handles runtime database switching and schema introspection for HayStack
Supports Totara, Moodle, and custom LMS platforms through canonical mapping
"""

import os
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
from sqlalchemy import create_engine, text, pool
from urllib.parse import quote_plus
from dataclasses import dataclass
from functools import lru_cache
from collections import Counter
import threading
import re

@dataclass
class DBConfig:
    """Database configuration for a tenant"""
    host: str
    port: int
    database: str
    user: str
    password: str
    charset: str = "utf8mb4"

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseConnector:
    """
    Multi-tenant database connector with runtime DB switching capability.
    
    Features:
    - Runtime database switching without process restarts
    - Schema introspection for any LMS platform (Totara, Moodle, Custom)
    - Canonical-to-actual table name mapping and SQL rewriting
    - Per-tenant engine caching for performance
    - Schema-agnostic SQL validation
    """
    # Class-level engine cache for multi-tenant support
    _engine_lock = threading.Lock()
    _engines_by_key: Dict[str, Any] = {}
    
    def __init__(self, default_config: Optional[DBConfig] = None):
        if default_config is None:
            # Load from existing config for backward compatibility
            try:
                from .config import DB_CONFIG
            except ImportError:
                from config import DB_CONFIG
            default_config = DBConfig(
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'], 
                database=DB_CONFIG['database'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password']
            )
        
        self.logger = logging.getLogger(__name__)
        self.default_config = default_config
        self.current_key = self._key_from_config(default_config)
        self.table_prefix = None  # discovered dynamically per-DB
        
        # Cache for schema introspection
        self._schema_cache = {}
        self._synonyms_cache = {}
        
        # Initialize engine after logger is set
        self.engine = self._get_or_create_engine(default_config)
    
    def _key_from_config(self, cfg: DBConfig) -> str:
        """Generate unique key for tenant engine cache"""
        return f"{cfg.user}@{cfg.host}:{cfg.port}/{cfg.database}"
    
    def _build_url(self, cfg: DBConfig) -> str:
        """Build SQLAlchemy connection URL from config"""
        return (
            f"mysql+pymysql://{cfg.user}:{quote_plus(cfg.password)}@"
            f"{cfg.host}:{cfg.port}/{cfg.database}?charset={cfg.charset}"
        )
    
    def _get_or_create_engine(self, cfg: DBConfig):
        """Get or create engine for tenant (thread-safe)"""
        key = self._key_from_config(cfg)
        with self._engine_lock:
            if key in self._engines_by_key:
                self.logger.debug(f"Reusing engine for tenant: {key}")
                return self._engines_by_key[key]
            
            self.logger.info(f"Creating new engine for tenant: {key}")
            engine = create_engine(
                self._build_url(cfg),
                poolclass=pool.QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={"connect_timeout": 30}
            )
            
            # Test connection
            try:
                with engine.begin() as conn:
                    conn.execute(text("SELECT 1"))
                self.logger.info(f"Engine created successfully for tenant: {key}")
            except Exception as e:
                self.logger.error(f"Failed to create engine for tenant {key}: {e}")
                raise
            
            self._engines_by_key[key] = engine
            return engine
    
    def switch_database(self, cfg: DBConfig):
        """Switch current database engine without restarting (runtime DB switching)"""
        old_key = self.current_key
        new_key = self._key_from_config(cfg)
        
        if old_key == new_key:
            self.logger.debug(f"Already connected to tenant: {new_key}")
            return
        
        self.logger.info(f"Switching database from {old_key} to {new_key}")
        self.engine = self._get_or_create_engine(cfg)
        self.current_key = new_key
        
        # Reset schema memoization for new DB
        self.table_prefix = None
        if new_key in self._schema_cache:
            del self._schema_cache[new_key]
        if new_key in self._synonyms_cache:
            del self._synonyms_cache[new_key]
        
        self.logger.info(f"Successfully switched to tenant: {new_key}")
    
    def current_tenant_key(self) -> str:
        """Get current tenant key"""
        return self.current_key
    
    @lru_cache(maxsize=64)
    def _introspect_tables_cached(self, key: str) -> Dict[str, List[str]]:
        """Cached table introspection for performance"""
        sql = text("""
            SELECT TABLE_NAME, COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """)
        
        by_table: Dict[str, List[str]] = {}
        try:
            with self.engine.connect() as conn:
                for table_name, column_name in conn.execute(sql):
                    by_table.setdefault(table_name.lower(), []).append(column_name.lower())
            self.logger.debug(f"Discovered {len(by_table)} tables for tenant {key}")
        except Exception as e:
            self.logger.error(f"Schema introspection failed for {key}: {e}")
            raise
        
        return by_table
    
    def get_schema_map(self) -> Dict[str, List[str]]:
        """Get lowercase table->columns mapping for current DB"""
        if self.current_key not in self._schema_cache:
            self._schema_cache[self.current_key] = self._introspect_tables_cached(self.current_key)
        return self._schema_cache[self.current_key]
    
    def detect_prefix(self) -> Optional[str]:
        """Detect dominant table prefix (ttl_, mdl_, lms_, etc.) - optional heuristic"""
        if self.table_prefix is not None:
            return self.table_prefix
        
        tables = list(self.get_schema_map().keys())
        prefixes = [t.split('_', 1)[0] + '_' for t in tables if '_' in t]
        
        if not prefixes:
            self.table_prefix = ""
            return None
        
        # Choose most common prefix
        most_common = Counter(prefixes).most_common(1)
        self.table_prefix = most_common[0][0] if most_common else ""
        
        self.logger.info(f"Detected table prefix for {self.current_key}: '{self.table_prefix}'")
        return self.table_prefix if self.table_prefix else None
    
    def build_synonyms(self) -> Dict[str, str]:
        """Build canonical->actual table mapping for current DB"""
        if self.current_key in self._synonyms_cache:
            return self._synonyms_cache[self.current_key]
        
        tables = self.get_schema_map()
        
        def choose_table(*candidate_names) -> Optional[str]:
            """Choose best matching table from candidates"""
            for table_name in tables.keys():
                for candidate in candidate_names:
                    if table_name == candidate or table_name.endswith('_' + candidate):
                        return table_name
            return None
        
        synonyms = {
            "user_table": choose_table("user", "users"),
            "course_table": choose_table("course", "courses"), 
            "enrol_table": choose_table("enrol", "enroll", "enrollment", "enrolments"),
            "user_enrol_table": choose_table("user_enrolments", "user_enrollments", "enrollments"),
            "role_assignments_table": choose_table("role_assignments", "role_assignment", "roles"),
            "context_table": choose_table("context", "contexts"),
            "role_table": choose_table("role", "roles"),
            "grade_grades_table": choose_table("grade_grades", "grades"),
            "grade_items_table": choose_table("grade_items", "grade_item")
        }
        
        # Filter out None values
        synonyms = {k: v for k, v in synonyms.items() if v is not None}
        
        self.logger.info(f"Built synonyms map for {self.current_key}: {synonyms}")
        self._synonyms_cache[self.current_key] = synonyms
        return synonyms
    
    def rewrite_canonical_sql(self, sql: str) -> str:
        """Rewrite canonical table names to actual table names before execution"""
        synonyms = self.build_synonyms()
        rewritten_sql = sql
        
        for canonical, actual in synonyms.items():
            if actual:
                # Replace canonical name with actual table name (word boundaries)
                canonical_name = canonical.replace('_table', '')
                pattern = r'\b' + re.escape(canonical_name) + r'\b'
                rewritten_sql = re.sub(pattern, actual, rewritten_sql, flags=re.IGNORECASE)
        
        if rewritten_sql != sql:
            self.logger.debug(f"Rewrote SQL: {sql[:100]}... -> {rewritten_sql[:100]}...")
        
        return rewritten_sql
    
    def validate_sql_query(self, query: str) -> bool:
        """
        Validate SQL query for security and compliance
        Allows SELECT statements on discovered tables + canonical names
        """
        query_upper = query.strip().upper()
        
        # Check if query starts with SELECT
        if not query_upper.startswith('SELECT'):
            self.logger.warning(f"Non-SELECT query blocked: {query[:50]}...")
            return False
        
        # Check for dangerous keywords
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                self.logger.warning(f"Dangerous keyword '{keyword}' found in query")
                return False
        
        # Check for multiple statements (basic protection against SQL injection)
        if ';' in query.strip()[:-1]:  # Allow trailing semicolon
            self.logger.warning("Multiple statements detected in query")
            return False
        
        # Get allowed tables from schema introspection + canonical names
        discovered_tables = set(self.get_schema_map().keys())
        canonical_names = set(self.build_synonyms().keys())
        allowed_tables = discovered_tables | canonical_names
        
        # Extract tables from query
        table_matches = re.findall(r'\\b(?:FROM|JOIN)\\s+([a-zA-Z_][a-zA-Z0-9_]*)', query, re.IGNORECASE)
        
        for table in table_matches:
            table_lower = table.lower()
            # Allow if it's a discovered table or canonical name
            if table_lower not in discovered_tables:
                # Check if it's a canonical name that will be rewritten
                canonical_found = False
                for canonical in canonical_names:
                    if canonical.replace('_table', '') == table_lower:
                        canonical_found = True
                        break
                if not canonical_found:
                    self.logger.warning(f"Access to unknown table blocked: {table}")
                    return False
        
        return True
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame"""
        try:
            # Rewrite canonical names to actual table names before execution
            rewritten_query = self.rewrite_canonical_sql(query)
            
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(rewritten_query), params)
                else:
                    result = conn.execute(text(rewritten_query))
                
                # Convert to pandas DataFrame
                columns = list(result.keys())
                rows = result.fetchall()
                
                if rows:
                    df = pd.DataFrame(rows, columns=columns)
                    self.logger.info(f"Query executed successfully, returned {len(df)} rows")
                    return df
                else:
                    self.logger.info("Query executed successfully, no rows returned")
                    return pd.DataFrame()
                    
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            self.logger.error(f"Query: {query}")
            raise
    
    def get_llm_schema_context(self) -> Dict[str, Any]:
        """Get schema context for LLM prompts (canonical + synonyms)"""
        synonyms = self.build_synonyms()
        schema_map = self.get_schema_map()
        
        # Build canonical schema description
        canonical_entities = {
            "user": "Users/students/learners in the system",
            "course": "Courses/classes/modules offered", 
            "enrol": "Enrollment methods available",
            "user_enrol": "User enrollments in courses",
            "role_assignments": "User role assignments with context",
            "context": "Permission contexts (system, course, etc.)",
            "role": "Available roles in the system",
            "grade_grades": "Student grades for assignments/activities",
            "grade_items": "Gradeable items/activities"
        }
        
        return {
            "tenant_key": self.current_key,
            "table_prefix": self.detect_prefix(),
            "canonical_entities": canonical_entities,
            "synonyms_map": synonyms,
            "discovered_tables": list(schema_map.keys()),
            "instructions": (
                "Always use canonical entity names in your SQL (user, course, user_enrol, etc.). "
                "The system will automatically rewrite them to actual table names. "
                "Only use SELECT statements. No DML/DDL operations allowed."
            )
        }
    
    def clear_tenant_cache(self, tenant_key: Optional[str] = None):
        """Clear schema/synonyms cache for tenant (or all if None)"""
        if tenant_key:
            self._schema_cache.pop(tenant_key, None)
            self._synonyms_cache.pop(tenant_key, None)
            self.logger.info(f"Cleared cache for tenant: {tenant_key}")
        else:
            self._schema_cache.clear()
            self._synonyms_cache.clear()
            self.logger.info("Cleared all tenant caches")
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()
            self.logger.info("Database connection closed")