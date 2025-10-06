"""
Database Connector Module
Handles database connections and query execution for HayStack
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, text, pool
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus
from src.config import DB_CONFIG

class DatabaseConnector:
    """Database connection and query management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connection_string = self._build_connection_string()
        self.engine = None
        self._initialize_connection()
    
    def _build_connection_string(self) -> str:
        """Build MySQL connection string from config"""
        return (
            f"mysql+pymysql://{DB_CONFIG['user']}:{quote_plus(DB_CONFIG['password'])}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            f"?charset=utf8mb4"
        )
    
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
    
    def _initialize_connection(self):
        """Initialize database engine and test connection"""
        try:
            self.engine = create_engine(
                self.connection_string,
                poolclass=pool.QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Test connection
            with self.engine.begin() as conn:
                conn.execute(text("SELECT 1"))
            
            self.logger.info("Database connection established successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
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
                pattern = r'\b' + re.escape(canonical.replace('_table', '')) + r'\b'
                rewritten_sql = re.sub(pattern, actual, rewritten_sql, flags=re.IGNORECASE)
        
        if rewritten_sql != sql:
            self.logger.debug(f"Rewrote SQL: {sql[:100]}... -> {rewritten_sql[:100]}...")
        
        return rewritten_sql
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame"""
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                
                # Convert to DataFrame
                df = pd.DataFrame(result.fetchall())
                if not df.empty:
                    df.columns = list(result.keys())
                
                self.logger.info(f"Query executed successfully, returned {len(df)} rows")
                return df
                
        except SQLAlchemyError as e:
            self.logger.error(f"Database query failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in query execution: {e}")
            raise
    
    def execute_raw_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute query and return raw results as list of dictionaries"""
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                
                # Convert to list of dicts
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                
                self.logger.info(f"Raw query executed successfully, returned {len(rows)} rows")
                return rows
                
        except SQLAlchemyError as e:
            self.logger.error(f"Database query failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in raw query execution: {e}")
            raise
    
    def execute_query_df(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute query and return DataFrame - alias for execute_query"""
        return self.execute_query(query, params)
    
    def get_connection(self):
        """Get a database connection for manual use"""
        if self.engine:
            return self.engine.connect()
        else:
            raise Exception("Database engine not initialized")
    
    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            if self.engine:
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get basic database information"""
        try:
            if self.engine:
                with self.engine.connect() as conn:
                    version_result = conn.execute(text("SELECT VERSION() as version")).fetchone()
                    db_result = conn.execute(text("SELECT DATABASE() as db_name")).fetchone()
                    
                    return {
                        "mysql_version": version_result[0] if version_result else "Unknown",
                        "database_name": db_result[0] if db_result else "Unknown",
                        "host": DB_CONFIG['host'],
                        "port": DB_CONFIG['port'],
                        "user": DB_CONFIG['user']
                    }
            else:
                return {"error": "Engine not initialized"}
        except Exception as e:
            self.logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connection closed")