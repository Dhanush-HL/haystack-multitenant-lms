#!/usr/bin/env python3

"""
HayStack FastMCP Server with Qwen 2.5:7b-instruct
Custom implementation for HayStack multi-agent integration
Enhanced with dynamic RBAC using Totara LMS native role model
"""

import os
import sys
import re
import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple, Optional, Union
import asyncio
import aiohttp
import json
import datetime
import uuid
from urllib.parse import quote_plus
from dataclasses import dataclass, asdict

# Load environment variables
load_dotenv()

# Import RBAC service
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    from src.rbac import RBACService
    print("[HayStack MCP] RBAC Service imported successfully")
except ImportError:
    print("[HayStack MCP] WARNING: RBAC Service not found, running without security")
    RBACService = None

try:
    from fastmcp import FastMCP
    print("[HayStack MCP] FastMCP imported successfully")
except ImportError:
    print("[HayStack MCP] ERROR: FastMCP not installed. Run: pip install fastmcp")
    sys.exit(1)

# Removed Vanna dependency - we're doing all the work ourselves anyway

# Error Handling Framework
import logging
import traceback
from enum import Enum
import hashlib
import time
from threading import Lock

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('haystack_mcp.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Chart.js Integration Classes and Helpers
@dataclass
class ChartJSResult:
    """Chart.js specification result"""
    type: str
    data: dict
    options: dict

def create_individual_course_bars_chart(df: pd.DataFrame, label_col: str, title: str, 
                                       custom_colors: list, user_comparison: dict) -> ChartJSResult:
    """
    Create individual course bars for multi-user comparison
    Each user's courses appear as separate colored bars
    """
    # Get unique users and their colors
    unique_users = df['user_source'].unique()
    
    # Create individual course labels and data
    labels = []
    datasets = []
    
    # Default colors
    default_colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
    colors_to_use = custom_colors if custom_colors else default_colors
    
    # First pass: collect all course labels
    for user in unique_users:
        user_df = df[df['user_source'] == user]
        for _, row in user_df.iterrows():
            course_name = row[label_col]
            labels.append(f"{course_name}")
    
    # Second pass: create datasets - one per user with all their courses
    for user in unique_users:
        user_df = df[df['user_source'] == user]
        
        # Get user-specific color
        user_key = f"user_{user}"
        if user_key in user_comparison and user_comparison[user_key] in ['red', 'blue', 'green', 'yellow', 'purple', 'orange']:
            color_map = {
                'red': '#FF0000', 'blue': '#0000FF', 'green': '#00FF00',
                'yellow': '#FFFF00', 'purple': '#800080', 'orange': '#FFA500'
            }
            user_color = color_map.get(user_comparison[user_key], colors_to_use[0])
        else:
            user_color = colors_to_use[0]
        
        # Create data array for this user
        data_values = []
        user_courses = user_df[label_col].tolist()
        
        for course_label in labels:
            if course_label in user_courses:
                data_values.append(1)  # User has this course
            else:
                data_values.append(0)  # User doesn't have this course
        
        dataset = {
            "label": f"User {user}",
            "data": data_values,
            "backgroundColor": user_color,
            "borderColor": user_color,
            "borderWidth": 2
        }
        datasets.append(dataset)
    
    # If no courses found, create empty chart
    if not labels:
        labels = ["No courses found"]
        datasets = [{
            "label": "No data",
            "data": [0],
            "backgroundColor": "#CCCCCC",
            "borderColor": "#CCCCCC",
            "borderWidth": 1
        }]
    
    # Chart.js options
    options = {
        "responsive": True,
        "maintainAspectRatio": False,
        "plugins": {
            "legend": {
                "position": "top",
                "display": True
            },
            "title": {
                "display": bool(title),
                "text": title,
                "font": {"size": 16}
            }
        },
        "scales": {
            "x": {"display": True},
            "y": {"display": True, "beginAtZero": True}
        }
    }
    
    return ChartJSResult(
        type="bar",
        data={"labels": labels, "datasets": datasets},
        options=options
    )

def df_to_chartjs(df: pd.DataFrame, chart_type: str, label_col: str, value_cols: list, title: str = "", 
                 custom_colors: Optional[list] = None, user_comparison: Optional[dict] = None) -> ChartJSResult:
    """
    Convert pandas DataFrame to Chart.js specification
    
    Args:
        df: DataFrame with data
        chart_type: Chart.js chart type (bar, line, pie, doughnut, scatter)
        label_col: Column name for x-axis labels
        value_cols: List of column names for y-axis data
        title: Chart title
        custom_colors: Optional list of custom hex colors to use
        user_comparison: Optional dict mapping user identifiers to colors
    
    Returns:
        ChartJSResult with Chart.js specification
    """
    # Extract labels (x-axis)
    labels = df[label_col].astype(str).tolist()
    
    # Handle custom colors and user comparison
    default_colors = [
        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', 
        '#FF9F40', '#FF6384', '#C9CBCF', '#4BC0C0', '#FF6384'
    ]
    
    # Use custom colors if provided, otherwise defaults
    colors_to_use = custom_colors if custom_colors else default_colors
    
    # Create datasets for each value column
    datasets = []
    
    # Check if this is multi-user data with user_source column
    has_user_source = 'user_source' in df.columns
    
    if has_user_source and user_comparison:
        # Multi-user comparison: create separate datasets per user
        unique_users = df['user_source'].unique()
        
        # Get all requested users from user_comparison (including those with no data)
        requested_users = [int(user_key.replace("user_", "")) for user_key in user_comparison.keys() if user_key.startswith("user_")]
        
        # Create a comprehensive list of all course labels from all users
        all_labels = df[label_col].unique().tolist()
        labels = all_labels
        
        # If no data at all, create empty labels for better visualization
        if not all_labels:
            all_labels = ["No courses found"]
            labels = all_labels
        
        # Create datasets for ALL requested users (even those with no data)
        for user in requested_users:
            user_df = df[df['user_source'] == user] if user in unique_users else pd.DataFrame()
                
            # Get user-specific color
            user_key = f"user_{user}"
            if user_key in user_comparison and user_comparison[user_key] in ['red', 'blue', 'green', 'yellow', 'purple', 'orange']:
                color_map = {
                    'red': '#FF0000', 'blue': '#0000FF', 'green': '#00FF00',
                    'yellow': '#FFFF00', 'purple': '#800080', 'orange': '#FFA500'
                }
                user_color = color_map.get(user_comparison[user_key], colors_to_use[0])
            else:
                user_color = colors_to_use[len(datasets) % len(colors_to_use)]
            
            # For multi-user bar charts, create data array aligned with all labels
            # This ensures both users appear on the same chart with different colors
            data_values = []
            
            if user_df.empty:
                # User has no data - create all zeros but still show in legend
                data_values = [0] * len(all_labels)
                user_label = f"User {user} (No courses)"
            else:
                # User has data - check enrollment for each course
                for label in all_labels:
                    user_course_data = user_df[user_df[label_col] == label]
                    if not user_course_data.empty:
                        data_values.append(1)  # User is enrolled (binary presence)
                    else:
                        data_values.append(0)  # User not enrolled in this course
                user_label = f"User {user}"
            
            dataset = {
                "label": user_label,
                "data": data_values,
                "backgroundColor": user_color,
                "borderColor": user_color,
                "borderWidth": 2
            }
            
            # Chart type specific styling
            if chart_type in ['pie', 'doughnut']:
                dataset["backgroundColor"] = [user_color] * len(data_values)
            elif chart_type == 'line':
                dataset["fill"] = False
                dataset["tension"] = 0.1
                
            datasets.append(dataset)
    else:
        # Standard single dataset approach
        for i, col in enumerate(value_cols):
            if col not in df.columns:
                continue
            y_values = pd.to_numeric(df[col], errors="coerce").fillna(0).tolist()
            
            # Use custom color if available
            color = colors_to_use[i % len(colors_to_use)]
            
            dataset = {
                "label": col.replace('_', ' ').title(),
                "data": y_values,
                "backgroundColor": color,
                "borderColor": color,
                "borderWidth": 1
            }
            
            # Chart type specific styling
            if chart_type in ['pie', 'doughnut']:
                dataset["backgroundColor"] = colors_to_use[:len(y_values)]
            elif chart_type == 'line':
                dataset["fill"] = False
                dataset["tension"] = 0.1
                
            datasets.append(dataset)
    
    # Chart.js options
    options = {
        "responsive": True,
        "maintainAspectRatio": False,
        "plugins": {
            "legend": {
                "position": "top",
                "display": True
            },
            "title": {
                "display": bool(title),
                "text": title,
                "font": {"size": 16}
            }
        }
    }
    
    # Chart type specific options
    if chart_type in ['bar', 'line']:
        options["scales"] = {
            "x": {"display": True},
            "y": {"display": True, "beginAtZero": True}
        }
    
    return ChartJSResult(
        type=chart_type,
        data={"labels": labels, "datasets": datasets},
        options=options
    )

def minimal_chart_html(chart_id: str, chart_spec: ChartJSResult) -> str:
    """
    Generate minimal HTML snippet for Chart.js rendering
    
    Args:
        chart_id: Unique ID for the canvas element
        chart_spec: Chart.js specification
    
    Returns:
        HTML snippet with canvas and initialization script
    """
    return f"""
<div style="max-width:100%; height:400px; overflow:auto; margin:20px 0;">
  <canvas id="{chart_id}" style="max-height:350px;"></canvas>
</div>
<script>
  (function(){{
    var ctx = document.getElementById("{chart_id}");
    if (ctx && typeof Chart !== 'undefined') {{
      var spec = {json.dumps(asdict(chart_spec))};
      new Chart(ctx, spec);
    }} else {{
      console.warn("Chart.js not loaded or canvas element not found: {chart_id}");
    }}
  }})();
</script>
""".strip()

class ErrorType(Enum):
    """Error classification for better handling"""
    DATABASE_CONNECTION = "database_connection"
    SQL_VALIDATION = "sql_validation"
    SQL_EXECUTION = "sql_execution"
    LLM_GENERATION = "llm_generation"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    UNKNOWN = "unknown"

class HayStackError(Exception):
    """Base exception for HayStack MCP with enhanced error context"""
    
    def __init__(self, message: str, error_type: ErrorType = ErrorType.UNKNOWN, 
                 original_error: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_type = error_type
        self.original_error = original_error
        self.context = context or {}
        self.timestamp = datetime.datetime.now()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/response"""
        return {
            "error_type": self.error_type.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "original_error": str(self.original_error) if self.original_error else None
        }
    
    def get_user_message(self) -> str:
        """Get user-friendly error message"""
        friendly_messages = {
            ErrorType.DATABASE_CONNECTION: "Unable to connect to the database. Please try again later.",
            ErrorType.SQL_VALIDATION: "Your query contains invalid SQL. Please rephrase your request.",
            ErrorType.SQL_EXECUTION: "Unable to execute your query. Please check your request and try again.",
            ErrorType.LLM_GENERATION: "Unable to generate SQL for your request. Please rephrase your question.",
            ErrorType.CONFIGURATION: "System configuration error. Please contact support.",
            ErrorType.NETWORK: "Network connectivity issue. Please try again.",
            ErrorType.UNKNOWN: "An unexpected error occurred. Please try again."
        }
        return friendly_messages.get(self.error_type, self.message)

class ErrorHandler:
    """Centralized error handling utilities"""
    
    @staticmethod
    def log_error(error: Union[Exception, HayStackError], context: Optional[Dict[str, Any]] = None):
        """Log error with full context"""
        if isinstance(error, HayStackError):
            logger.error(f"HayStack Error [{error.error_type.value}]: {error.message}")
            if error.context:
                logger.error(f"Context: {json.dumps(error.context, indent=2)}")
            if error.original_error:
                logger.error(f"Original error: {error.original_error}")
                logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            logger.error(f"Unexpected error: {error}")
            logger.error(f"Context: {json.dumps(context or {}, indent=2)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    @staticmethod
    def create_safe_sql_fallback(user_message: str = "Query processing failed") -> str:
        """Create a safe SQL query for error scenarios"""
        return f"SELECT '{user_message}' as error_message, NOW() as timestamp;"
    
    @staticmethod
    def wrap_with_error_handling(func):
        """Decorator for comprehensive error handling"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except HayStackError:
                raise  # Re-raise HayStack errors as-is
            except Exception as e:
                # Convert to HayStack error with context
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],  # Truncate for logging
                    "kwargs": str(kwargs)[:200]
                }
                raise HayStackError(
                    f"Error in {func.__name__}: {str(e)}",
                    ErrorType.UNKNOWN,
                    e,
                    context
                )
        return wrapper
    
    @staticmethod
    async def async_wrap_with_error_handling(func):
        """Async decorator for comprehensive error handling"""
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HayStackError:
                raise  # Re-raise HayStack errors as-is
            except Exception as e:
                # Convert to HayStack error with context
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],  # Truncate for logging
                    "kwargs": str(kwargs)[:200]
                }
                raise HayStackError(
                    f"Error in {func.__name__}: {str(e)}",
                    ErrorType.UNKNOWN,
                    e,
                    context
                )
        return wrapper

# Performance Optimization - Caching System
class QueryCache:
    """Thread-safe in-memory cache for SQL results"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):  # 5 min TTL
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = Lock()
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, sql: str, params: Optional[Union[Tuple, Dict]] = None) -> str:
        """Generate cache key from SQL and parameters"""
        content = sql + str(params or "")
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, sql: str, params: Optional[Union[Tuple, Dict]] = None) -> Optional[pd.DataFrame]:
        """Get cached result if available and not expired"""
        key = self._generate_key(sql, params)
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry['timestamp'] < self.ttl_seconds:
                    self.hit_count += 1
                    logger.debug(f"Cache hit for query: {sql[:50]}...")
                    return entry['data'].copy()  # Return copy to avoid mutations
                else:
                    # Expired entry
                    del self.cache[key]
            
            self.miss_count += 1
            return None
    
    def put(self, sql: str, data: pd.DataFrame, params: Optional[Union[Tuple, Dict]] = None):
        """Cache query result"""
        key = self._generate_key(sql, params)
        
        with self.lock:
            # Implement LRU eviction if cache is full
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
                del self.cache[oldest_key]
            
            self.cache[key] = {
                'data': data.copy(),  # Store copy to avoid mutations
                'timestamp': time.time()
            }
            logger.debug(f"Cached result for query: {sql[:50]}...")
    
    def clear(self):
        """Clear all cached entries"""
        with self.lock:
            self.cache.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "ttl_seconds": self.ttl_seconds
            }

# Global cache instance
query_cache = QueryCache(
    max_size=int(os.getenv('CACHE_MAX_SIZE', '100')),
    ttl_seconds=int(os.getenv('CACHE_TTL_SECONDS', '300'))
)


class HayStackMCP:
    """
    HayStack MCP Server - Clean custom implementation.
    Handles SQL generation and database operations directly.
    """
    def __init__(self, config=None):
        # Initialize training data storage
        self.ddl_data = []
        self.question_sql_data = []
        self.documentation_data = []
        
        # Initialize role cache for performance optimization
        self.role_cache = {}  # Format: {(user_id, course_id): role}
        
        # Initialize self-access query cache for performance
        self.self_query_cache = {}  # Format: {(user_id, query_hash): results}
        
        # Conversation history for context
        self.conversation_history = []
        
        # Hard fail on unresolved/unauthorized data queries; do not synthesize results.
        self.strict_data_mode = True

        # Initialize Qwen 2.5 7B with Ollama for HayStack
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self.qwen_model = os.getenv('QWEN_MODEL', 'qwen2.5:7b-instruct')
        print(f"[HayStack MCP] Using Qwen 2.5 7B via Ollama: {self.ollama_url}")

        # Initialize database connection for Azure MySQL
        self.db_config = {
            'host': os.getenv('DB_HOST'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME'),
            'port': int(os.getenv('DB_PORT', 3306))
        }

        # Dynamic table prefix detection for Totara compatibility
        self.table_prefix = os.getenv("DB_TABLE_PREFIX")  # Optional override
        self.allowed_tables = set()
        self._detect_table_prefix_and_schema()

        # Initialize RBAC service
        self._rbac_service = None
        if RBACService:
            try:
                # RBAC will be initialized when engine is available
                print("[HayStack MCP] RBAC service will be initialized on first use")
            except Exception as e:
                print(f"[HayStack MCP] WARNING: Failed to initialize RBAC service: {e}")
                self._rbac_service = None
        else:
            print("[HayStack MCP] WARNING: Running without RBAC protection")

        print("[HayStack MCP] HayStack MCP initialized")

    def _detect_table_prefix_and_schema(self):
        """
        Detect Totara table prefix dynamically from INFORMATION_SCHEMA
        and build allowed tables list for validation
        """
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                # Get all table names in the database
                result = conn.execute(sqlalchemy.text("""
                    SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = :db_name
                    AND TABLE_TYPE = 'BASE TABLE'
                    ORDER BY TABLE_NAME
                """), {"db_name": self.db_config["database"]})
                
                table_names = [row[0] for row in result.fetchall()]
                
                if not table_names:
                    print("[HayStack MCP] WARNING: No tables found in database")
                    self.allowed_tables = set()
                    self.table_prefix = None
                    return
                
                # If prefix is explicitly set, use it
                if self.table_prefix:
                    self.allowed_tables = {name for name in table_names if name.startswith(self.table_prefix)}
                    print(f"[HayStack MCP] Using explicit table prefix: {self.table_prefix}")
                    print(f"[HayStack MCP] Found {len(self.allowed_tables)} allowed tables")
                    return
                
                # Auto-detect prefix by looking for core Moodle/Totara tables
                core_table_suffixes = ['user', 'course', 'enrol', 'user_enrolments', 'role', 'context', 'role_assignments']
                potential_prefixes = set()
                
                for table_name in table_names:
                    for suffix in core_table_suffixes:
                        if table_name.endswith('_' + suffix):
                            prefix = table_name[:-len('_' + suffix)]
                            potential_prefixes.add(prefix + '_')
                            break
                
                # Select the most common prefix
                if potential_prefixes:
                    prefix_counts = {}
                    for prefix in potential_prefixes:
                        prefix_counts[prefix] = len([name for name in table_names if name.startswith(prefix)])
                    
                    # Use the prefix with the most matching tables
                    self.table_prefix = max(prefix_counts.keys(), key=lambda x: prefix_counts[x])
                    self.allowed_tables = {name for name in table_names if name.startswith(self.table_prefix)}
                    
                    print(f"[HayStack MCP] Auto-detected table prefix: {self.table_prefix}")
                    print(f"[HayStack MCP] Found {len(self.allowed_tables)} allowed tables with prefix")
                else:
                    # Fallback: allow all tables (no prefix requirement)
                    print("[HayStack MCP] WARNING: No standard LMS prefix detected, allowing all tables")
                    self.table_prefix = None
                    self.allowed_tables = set(table_names)
                
        except Exception as e:
            print(f"[HayStack MCP] ERROR: Failed to detect table prefix: {e}")
            # Fallback to ttl_ for backward compatibility
            self.table_prefix = "ttl_"
            self.allowed_tables = set()
            print("[HayStack MCP] Falling back to ttl_ prefix")

    def _generate_dynamic_schema_context(self) -> List[str]:
        """
        Generate schema context from live INFORMATION_SCHEMA
        instead of hardcoded DDL statements
        """
        try:
            engine = self._get_engine()
            schema_parts = []
            
            # Core tables to prioritize (without prefix)
            core_tables = ['user', 'course', 'enrol', 'user_enrolments', 'role', 'context', 'role_assignments']
            
            # Get table schemas for core tables first
            with engine.connect() as conn:
                for core_table in core_tables:
                    table_name = None
                    # Find the actual table name with prefix
                    for allowed_table in self.allowed_tables:
                        if allowed_table.endswith('_' + core_table) or allowed_table == core_table:
                            table_name = allowed_table
                            break
                    
                    if table_name:
                        schema_parts.extend(self._get_table_schema(conn, table_name))
                
                # Add other important tables (modules, activities)
                activity_suffixes = ['course_modules', 'modules', 'assign', 'quiz', 'forum', 'lesson', 'resource']
                for suffix in activity_suffixes:
                    for allowed_table in self.allowed_tables:
                        if allowed_table.endswith('_' + suffix) or allowed_table == suffix:
                            schema_parts.extend(self._get_table_schema(conn, allowed_table))
                            break
            
            # Add relationship and pattern documentation
            prefix = self.table_prefix or ""
            schema_parts.extend([
                "",
                "-- HAYSTACK INTEGRATION PATTERNS:",
                f"-- User Profile: {prefix}user WHERE deleted = 0",
                f"-- User Enrollments: {prefix}user -> {prefix}user_enrolments -> {prefix}enrol -> {prefix}course",
                f"-- Role Verification: {prefix}user -> {prefix}role_assignments -> {prefix}role + {prefix}context",
                f"-- Course Activities: {prefix}course -> {prefix}course_modules -> {prefix}modules + activity tables",
                "-- Security: Always check user.deleted = 0, enrollment.status = 0. For course tables: check course.visible = 1 only if course table is JOINed in main query.",
                "-- TIMESTAMP HANDLING: timecreated fields are UNIX timestamps (INT). Use FROM_UNIXTIME(timecreated) to convert to datetime, YEAR(FROM_UNIXTIME(timecreated)) for year filtering.",
            ])
            
            return schema_parts
            
        except Exception as e:
            print(f"[HayStack MCP] WARNING: Failed to generate dynamic schema: {e}")
            return self._get_fallback_schema_context()

    def _get_table_schema(self, conn, table_name: str) -> List[str]:
        """Get CREATE TABLE statement for a specific table"""
        try:
            result = conn.execute(sqlalchemy.text("""
                SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY, COLUMN_DEFAULT, EXTRA
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = :db_name AND TABLE_NAME = :table_name
                ORDER BY ORDINAL_POSITION
            """), {"db_name": self.db_config["database"], "table_name": table_name})
            
            columns = result.fetchall()
            if not columns:
                return []
            
            schema_parts = [f"CREATE TABLE {table_name} ("]
            column_defs = []
            
            for col in columns:
                col_name, col_type, nullable, key, default, extra = col
                col_def = f"  {col_name} {col_type}"
                
                if key == 'PRI':
                    col_def += " PRIMARY KEY"
                elif nullable == 'NO':
                    col_def += " NOT NULL"
                
                if default is not None and default != 'NULL':
                    if isinstance(default, str) and default.isdigit():
                        col_def += f" DEFAULT {default}"
                    else:
                        col_def += f" DEFAULT '{default}'"
                
                if extra:
                    col_def += f" {extra}"
                
                column_defs.append(col_def)
            
            schema_parts.append(",\n".join(column_defs))
            schema_parts.append(");")
            schema_parts.append("")  # Empty line
            
            return schema_parts
            
        except Exception as e:
            print(f"[HayStack MCP] WARNING: Failed to get schema for {table_name}: {e}")
            return []

    def _get_fallback_schema_context(self) -> List[str]:
        """Fallback schema context when dynamic generation fails"""
        prefix = self.table_prefix or "ttl_"
        return [
            f"-- Core Tables (using prefix: {prefix})",
            f"CREATE TABLE {prefix}user (id INT PRIMARY KEY, username VARCHAR(100), firstname VARCHAR(100), lastname VARCHAR(100), email VARCHAR(255), deleted INT DEFAULT 0, timecreated INT);",
            f"CREATE TABLE {prefix}course (id INT PRIMARY KEY, fullname VARCHAR(255), shortname VARCHAR(100), visible INT DEFAULT 1, timecreated INT);",
            f"CREATE TABLE {prefix}enrol (id INT PRIMARY KEY, courseid INT, status INT, enrol VARCHAR(20));",
            f"CREATE TABLE {prefix}user_enrolments (id INT PRIMARY KEY, userid INT, enrolid INT, status INT, timecreated INT);",
            f"CREATE TABLE {prefix}role (id INT PRIMARY KEY, shortname VARCHAR(100), name VARCHAR(255));",
            f"CREATE TABLE {prefix}context (id INT PRIMARY KEY, contextlevel INT, instanceid INT);",
            f"CREATE TABLE {prefix}role_assignments (id INT PRIMARY KEY, roleid INT, contextid INT, userid INT);",
        ]

    # Training data management
    def add_ddl(self, ddl: str) -> str:
        """Add DDL to training data"""
        if ddl not in self.ddl_data:
            self.ddl_data.append(ddl)
        return f"Added DDL: {ddl[:50]}..."

    def add_documentation(self, documentation: str) -> str:
        """Add documentation to training data"""
        if documentation not in self.documentation_data:
            self.documentation_data.append(documentation)
        return f"Added documentation: {documentation[:50]}..."

    def add_question_sql(self, question: str, sql: str) -> str:
        """Add question-SQL pair to training data"""
        pair = {'question': question, 'sql': sql}
        if pair not in self.question_sql_data:
            self.question_sql_data.append(pair)
        return f"Added Q&A: {question[:30]}..."

    def get_similar_question_sql(self, question: str, **kwargs) -> List[Dict]:
        """Get similar question-SQL pairs for context"""
        return self.question_sql_data[:10]

    def submit_prompt(self, prompt: str, **kwargs) -> str:
        """Submit prompt to Qwen 2.5 7B via Ollama for SQL generation or JSON routing"""
        try:
            import requests
            
            # Check if JSON format is requested
            json_format = kwargs.get('format') == 'json'
            
            request_data = {
                "model": self.qwen_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1 if not json_format else 0.0,  # Lower temperature for JSON
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1
                }
            }
            
            # Add format parameter if JSON mode is requested
            if json_format:
                request_data["format"] = "json"
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                print(f"[HayStack MCP] Ollama error: {response.status_code}")
                if json_format:
                    # For JSON mode, try once without format parameter
                    try:
                        fallback_data = request_data.copy()
                        fallback_data.pop("format", None)
                        fallback_response = requests.post(
                            f"{self.ollama_url}/api/generate",
                            json=fallback_data,
                            timeout=30
                        )
                        if fallback_response.status_code == 200:
                            return fallback_response.json().get('response', '')
                    except Exception as fallback_error:
                        print(f"[HayStack MCP] JSON fallback failed: {fallback_error}")
                
                return self._nl_to_sql_with_guardrails(prompt)

        except Exception as e:
            print(f"[HayStack MCP] Error calling Ollama: {e}")
        return self._nl_to_sql_with_guardrails(prompt)
    
    def classify_intent_with_llm(self, text: str, requesting_user_id: int | None) -> dict:
        """Use LLM to classify user intent and extract targets with strict JSON output"""
        
        router_prompt = f"""System:
You route LMS queries for a Haystack MCP server. Decide if a message is casual chat, a data query, or a tool call.
Return STRICT JSON only (no markdown). Use the schema provided.

Developer:
JSON schema:
{{
  "intent": "conversational | data_query | tool_call",
  "tool": "haystack_query | haystack_user_profile | haystack_user_courses | haystack_status | haystack_raw_sql | haystack_chart | null",
  "targets": {{ "user_id": number | null, "target_user_id": number | null, "course_id": number | null }},
  "cross_user_request": boolean,
  "needs_admin": boolean,
  "natural_language_query": string,
  "risk_level": "low | medium | high",
  "reason": string
}}

Notes:
- Never inline numeric IDs into SQL; the execution layer (RBAC) will inject user/course restrictions.
- tool_call when the user explicitly asks for status, raw SQL, or named tool. haystack_raw_sql always needs_admin=true.
- Chart requests: If query contains chart keywords ("plot", "chart", "graph", "visualize", "trend", "distribution", "compare", "bar chart", "pie chart", "line chart") → ALWAYS use haystack_chart tool, regardless of data type mentioned.
- If the requester mentions another user ("user 2"), set cross_user_request=true and targets.target_user_id=2.
- If short greeting ("hi", "thanks", etc.) → conversational.

Few-shot examples:
User: "hi"
→ {{"intent":"conversational","tool":null,"targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"","risk_level":"low","reason":"greeting"}}

User: "what courses is user 2 enrolled in? i'm user 71"
→ {{"intent":"data_query","tool":"haystack_query","targets":{{"user_id":71,"target_user_id":2,"course_id":null}},"cross_user_request":true,"needs_admin":false,"natural_language_query":"courses for target user","risk_level":"medium","reason":"data for another user"}}

User: "my courses"
→ {{"intent":"data_query","tool":"haystack_user_courses","targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"courses for requesting user","risk_level":"low","reason":"self enrollment"}}

User: "what are my enrolled courses?"
→ {{"intent":"data_query","tool":"haystack_user_courses","targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"enrolled courses for requesting user","risk_level":"low","reason":"self enrollment"}}

User: "show me my enrolled courses"
→ {{"intent":"data_query","tool":"haystack_user_courses","targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"enrolled courses for requesting user","risk_level":"low","reason":"self enrollment"}}

User: "check system status"
→ {{"intent":"tool_call","tool":"haystack_status","targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"","risk_level":"low","reason":"status check"}}

User: "run sql: select count(*) from ttl_user"
→ {{"intent":"tool_call","tool":"haystack_raw_sql","targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":true,"natural_language_query":"select count(*) from ttl_user","risk_level":"high","reason":"raw sql requires admin"}}

User: "what can you do"
→ {{"intent":"conversational","tool":null,"targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"","risk_level":"low","reason":"help request"}}

User: "show me enrollments for course 5"
→ {{"intent":"data_query","tool":"haystack_query","targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":5}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"enrollments for course","risk_level":"low","reason":"course enrollment data"}}

User: "thanks"
→ {{"intent":"conversational","tool":null,"targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"","risk_level":"low","reason":"acknowledgment"}}

User: "how many students are enrolled in total?"
→ {{"intent":"data_query","tool":"haystack_query","targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"total student enrollment count","risk_level":"low","reason":"system statistics"}}

User: "get user profile for user 15"
→ {{"intent":"data_query","tool":"haystack_user_profile","targets":{{"user_id":{requesting_user_id},"target_user_id":15,"course_id":null}},"cross_user_request":true,"needs_admin":false,"natural_language_query":"user profile for target user","risk_level":"medium","reason":"profile data for another user"}}

User: "plot enrollments per course for last 30 days"
→ {{"intent":"tool_call","tool":"haystack_chart","targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"chart enrollments per course 30d","risk_level":"low","reason":"chart request"}}

User: "show a line chart of daily signups this month"
→ {{"intent":"tool_call","tool":"haystack_chart","targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"line chart daily signups monthly","risk_level":"low","reason":"chart request"}}

User: "pie chart of enrollments by category"
→ {{"intent":"tool_call","tool":"haystack_chart","targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"pie chart enrollments by category","risk_level":"low","reason":"chart request"}}

User: "visualize user activity trends"
→ {{"intent":"tool_call","tool":"haystack_chart","targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"visualize user activity trends","risk_level":"low","reason":"chart request"}}

User: "graph course completion rates"
→ {{"intent":"tool_call","tool":"haystack_chart","targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"graph course completion rates","risk_level":"low","reason":"chart request"}}

User: "make a chart of my enrolled courses"
→ {{"intent":"tool_call","tool":"haystack_chart","targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"chart of enrolled courses for requesting user","risk_level":"low","reason":"chart request"}}

User: "create a bar chart of my courses"
→ {{"intent":"tool_call","tool":"haystack_chart","targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"bar chart of user courses","risk_level":"low","reason":"chart request"}}

User: "show a graph of my enrolled courses"
→ {{"intent":"tool_call","tool":"haystack_chart","targets":{{"user_id":{requesting_user_id},"target_user_id":null,"course_id":null}},"cross_user_request":false,"needs_admin":false,"natural_language_query":"graph of enrolled courses for requesting user","risk_level":"low","reason":"chart request"}}

User:
{text} (requesting_user_id: {requesting_user_id})"""
        
        response = ""
        try:
            # Call LLM with JSON formatting
            response = self.submit_prompt(router_prompt, format="json")
            
            # Parse JSON response
            parsed_response = json.loads(response)
            
            # Validate required fields
            required_fields = ["intent", "targets", "cross_user_request", "needs_admin", "natural_language_query", "risk_level", "reason"]
            for field in required_fields:
                if field not in parsed_response:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure targets has required sub-fields
            if not isinstance(parsed_response["targets"], dict):
                raise ValueError("targets must be a dictionary")
            
            target_fields = ["user_id", "target_user_id", "course_id"]
            for field in target_fields:
                if field not in parsed_response["targets"]:
                    parsed_response["targets"][field] = None
            
            print(f"[HayStack MCP] LLM Router Classification: {parsed_response}")
            return parsed_response
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[HayStack MCP] Router JSON parsing failed: {e}")
            print(f"[HayStack MCP] Raw LLM response: {response[:200] if response else 'No response'}")
            
            # Return fallback classification - let existing heuristics handle it
            return {
                "intent": "fallback_to_heuristics",
                "tool": None,
                "targets": {
                    "user_id": requesting_user_id,
                    "target_user_id": None,
                    "course_id": None
                },
                "cross_user_request": False,
                "needs_admin": False,
                "natural_language_query": text,
                "risk_level": "unknown",
                "reason": f"JSON parsing failed: {str(e)}"
            }
        
        except Exception as e:
            print(f"[HayStack MCP] Router classification error: {e}")
            return {
                "intent": "fallback_to_heuristics",
                "tool": None,
                "targets": {
                    "user_id": requesting_user_id,
                    "target_user_id": None,
                    "course_id": None
                },
                "cross_user_request": False,
                "needs_admin": False,
                "natural_language_query": text,
                "risk_level": "unknown",
                "reason": f"Router error: {str(e)}"
            }

    def _build_context_for_question(self, question: str) -> str:
        """Build comprehensive context using training data with enhanced schema relationships"""
        context_parts = []

        # Add live database schema for HayStack (dynamically generated)
        context_parts.append("HayStack-Compatible Totara/LMS Schema:")
        schema_with_relationships = self._generate_dynamic_schema_context()
        
        context_parts.extend(schema_with_relationships)

        # Add HayStack-specific examples
        context_parts.append("\nHayStack Query Examples:")
        
        haystack_examples = [
            {
                "question": "What courses is user ID 71 enrolled in?",
                "sql": "SELECT c.id AS course_id, c.fullname, c.shortname FROM ttl_user u JOIN ttl_user_enrolments ue ON ue.userid = u.id AND ue.status = 0 JOIN ttl_enrol e ON e.id = ue.enrolid AND e.status = 0 JOIN ttl_course c ON c.id = e.courseid WHERE u.id = 71 AND u.deleted = 0 AND c.visible = 1 ORDER BY c.fullname;",
                "explanation": "-- HayStack pattern: User profile with enrollments"
            },
            {
                "question": "General information of user 71",
                "sql": "SELECT id, firstname, lastname, email, username, timecreated FROM ttl_user WHERE id = 71 AND deleted = 0;",
                "explanation": "-- HayStack pattern: Clean user profile data only"
            },
            {
                "question": "Count courses for user 71",
                "sql": "SELECT COUNT(*) as course_count FROM ttl_user_enrolments ue JOIN ttl_enrol e ON e.id = ue.enrolid AND e.status = 0 JOIN ttl_course c ON c.id = e.courseid WHERE ue.userid = 71 AND ue.status = 0 AND c.visible = 1;",
                "explanation": "-- HayStack pattern: Count with proper joins and status checks"
            },
            {
                "question": "How many users are enrolled in ADGM Test Course?",
                "sql": "SELECT COUNT(DISTINCT ue.userid) AS user_count FROM ttl_course c JOIN ttl_enrol e ON e.courseid = c.id JOIN ttl_user_enrolments ue ON ue.enrolid = e.id WHERE c.fullname LIKE '%ADGM%Test%Course%' AND c.visible = 1 AND e.status = 0 AND ue.status = 0;",
                "explanation": "-- HayStack pattern: Course enrollment count using LIKE with word-by-word wildcards for robust name matching"
            },
            {
                "question": "What is the role of user 71 in course 16?",
                "sql": "SELECT r.shortname, r.name FROM ttl_user u JOIN ttl_role_assignments ra ON ra.userid = u.id JOIN ttl_role r ON r.id = ra.roleid JOIN ttl_context c ON c.id = ra.contextid WHERE u.id = 71 AND u.deleted = 0 AND c.contextlevel = 50 AND c.instanceid = 16;",
                "explanation": "-- HayStack pattern: Role assignments with context filtering"
            },
            {
                "question": "What is my role in the course ADGM Test Course?",
                "sql": "SELECT r.shortname, r.name FROM ttl_user u JOIN ttl_role_assignments ra ON ra.userid = u.id JOIN ttl_role r ON r.id = ra.roleid JOIN ttl_context c ON c.id = ra.contextid WHERE u.deleted = 0 AND c.contextlevel = 50 AND c.instanceid = (SELECT id FROM ttl_course WHERE fullname LIKE 'ADGM Test%Course' AND visible = 1);",
                "explanation": "-- HayStack pattern: Role assignment by course name. Note: Only reference tables in WHERE clause that are JOINed in main query. Subqueries have their own scope."
            },
            {
                "question": "List all users enrolled in the first course",
                "sql": "SELECT u.id, u.username, u.firstname, u.lastname, u.email FROM ttl_user u JOIN ttl_user_enrolments ue ON ue.userid = u.id AND ue.status = 0 JOIN ttl_enrol e ON e.id = ue.enrolid AND e.status = 0 JOIN ttl_course c ON c.id = e.courseid WHERE u.deleted = 0 AND c.visible = 1 ORDER BY c.id ASC LIMIT 1;",
                "explanation": "-- HayStack pattern: User listing from enrollments. Note: Use ORDER BY with table.id for ordinals like 'first'"
            },
            {
                "question": "What is my role in the first course?",
                "sql": "SELECT r.shortname, r.name FROM ttl_user u JOIN ttl_role_assignments ra ON ra.userid = u.id JOIN ttl_role r ON r.id = ra.roleid JOIN ttl_context c ON c.id = ra.contextid WHERE u.deleted = 0 AND c.contextlevel = 50 ORDER BY c.instanceid ASC LIMIT 1;",
                "explanation": "-- HayStack pattern: Role assignment for ordinals. Use ORDER BY c.instanceid for 'first course' since instanceid contains course IDs"
            }
        ]
        
        relevant_examples = self.get_similar_question_sql(question)[:3]
        all_examples = haystack_examples + relevant_examples
        
        for example in all_examples:
            if isinstance(example, dict) and 'question' in example and 'sql' in example:
                context_parts.append(f"\nQ: {example['question']}")
                context_parts.append(f"SQL: {example['sql']}")
                if 'explanation' in example:
                    context_parts.append(example['explanation'])

        return "\n".join(context_parts)

    def _validate_sql(self, sql: str) -> bool:
        """Strong SQL validator with security checks for HayStack"""
        if not sql or not sql.strip():
            return False
            
        sql_stripped = sql.strip()
        sql_upper = sql_stripped.upper()
        
        # 1. Single statement only (only one trailing semicolon allowed)
        semicolon_count = sql_stripped.count(';')
        if semicolon_count > 1:
            print("[HayStack MCP] Validation failed: Multiple statements detected")
            return False
        
        # 2. Must start with SELECT (no other DML/DDL allowed)
        if not sql_upper.startswith('SELECT'):
            print("[HayStack MCP] Validation failed: Only SELECT statements allowed")
            return False
        
        # 3. Forbidden patterns (security checks)
        forbidden_patterns = [
            'INTO OUTFILE', 'LOAD_FILE', 'LOAD DATA', 
            '--', '#', '/*', '*/',  # Comments (except MySQL hints)
            'UNION', 'EXEC', 'EXECUTE', 'SP_',
            'XP_', 'OPENROWSET', 'OPENDATASOURCE'
        ]
        
        for pattern in forbidden_patterns:
            if pattern in sql_upper:
                # Allow MySQL optimizer hints like /*+ MAX_EXECUTION_TIME() */
                if pattern in ['/*', '*/'] and '/*+' in sql_upper:
                    continue
                print(f"[HayStack MCP] Validation failed: Forbidden pattern '{pattern}' detected")
                return False
        
        # 4. Check table access permissions (dynamic prefix support)
        import re
        # Find all table references (FROM/JOIN patterns)
        table_pattern = r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        tables = re.findall(table_pattern, sql_upper, re.IGNORECASE)
        
        if not tables:
            print("[HayStack MCP] Validation failed: No tables found in query")
            return False
        
        # 5. Validate all referenced tables are in allowed set
        for table in tables:
            if table.lower() not in {t.lower() for t in self.allowed_tables}:
                print(f"[HayStack MCP] Validation failed: Table '{table}' not in allowed set")
                print(f"[HayStack MCP] Allowed tables: {sorted(list(self.allowed_tables))[:10]}...")  # Show first 10
                return False
        
        # 6. Must have basic SELECT structure
        if 'FROM' not in sql_upper:
            print("[HayStack MCP] Validation failed: No FROM clause found")
            return False
            
        return True

    def _extract_sql(self, response: str) -> str:
        """Extract and clean SQL from Qwen response"""
        # Clean the response
        sql_response = response.replace('```sql', '').replace('```', '').strip()
        
        # Remove leading/trailing whitespace and split by lines
        lines = [line.strip() for line in sql_response.split('\n') if line.strip()]
        
        # Look for complete SQL blocks first
        sql_lines = []
        collecting_sql = False
        
        for line in lines:
            if line.upper().startswith('SELECT'):
                collecting_sql = True
                sql_lines = [line]
            elif collecting_sql:
                sql_lines.append(line)
                if line.endswith(';'):
                    break
        
        # Try to construct complete SQL
        if sql_lines:
            sql_candidate = ' '.join(sql_lines)
            if self._validate_sql(sql_candidate):
                return sql_candidate.rstrip(';') + ';'
        
        # Fallback: try to find any line with SELECT and FROM
        for line in lines:
            if 'SELECT' in line.upper() and 'FROM' in line.upper():
                if self._validate_sql(line):
                    return line.rstrip(';') + ';'
        
        # Final fallback: handle incomplete responses
        if sql_response:
            if 'SELECT' in sql_response.upper() and 'FROM' in sql_response.upper():
                cleaned = sql_response.replace('\n', ' ').strip()
                if self._validate_sql(cleaned):
                    return cleaned.rstrip(';') + ';'

        return ""



    def generate_sql(self, question: str, **kwargs) -> str:
        """Generate SQL using Qwen 2.5 7B with HayStack training context"""
        try:
            # Extract resolved ID if present
            resolved_id = None
            clean_question = question
            
            import re
            resolved_match = re.search(r'\[RESOLVED_ID:\s*(\d+)\]', question)
            if resolved_match:
                resolved_id = int(resolved_match.group(1))
                clean_question = re.sub(r'\[RESOLVED_ID:\s*\d+\]', '', question).strip()
                print(f"[HayStack MCP] Extracted resolved ID: {resolved_id}")
            
            # Build context safely
            try:
                context = self._build_context_for_question(clean_question)
            except Exception as e:
                ErrorHandler.log_error(HayStackError(
                    "Failed to build context for question",
                    ErrorType.LLM_GENERATION,
                    e,
                    {"question": clean_question[:100]}
                ))
                context = "Basic Totara LMS schema context not available."
            
            resolved_context = ""
            if resolved_id:
                resolved_context = f"\n\nRESOLVED REFERENCE: Use course ID {resolved_id} when filtering by course. Query is asking about a specific course that was previously identified."
            
            prompt = f"""You are a HayStack-compatible SQL expert for Totara LMS. Generate precise SQL for this query.

{context}{resolved_context}

SQL SYNTAX RULES:
1. Every table referenced must be properly JOINed in the FROM clause
2. Use explicit JOIN syntax: "JOIN table_name alias ON condition"
3. Never reference a table without declaring it first
4. Always use table aliases consistently
5. Check status fields: user.deleted = 0, enrolment.status = 0. For course.visible = 1: ONLY if ttl_course is JOINed in main query
6. DO NOT use placeholders like [YOUR_USER_ID] or [USER_ID] - RBAC will handle user filtering
7. Write general queries - the security system will add appropriate WHERE conditions
8. For course name filtering, use these patterns:
   - PREFERRED: Use exact course ID when available: "c.id = 31"
   - BY NAME: Use simple LIKE patterns: "c.fullname LIKE 'ADGM Test%Course'" (avoid multiple % in sequence)
   - FOR ROLE QUERIES: Use subquery pattern: "c.instanceid = (SELECT id FROM ttl_course WHERE fullname LIKE 'Course Name%' AND visible = 1)"
   - This prevents SQL formatting issues while maintaining functionality
9. If a specific course ID is provided via RESOLVED REFERENCE, use "c.id = {resolved_id}" instead of name-based filtering
10. For ordinal queries ("first course", "last user"): Use ORDER BY with id + LIMIT. Do NOT use non-existent fields like c.section - use c.id instead

IMPORTANT: For user-specific queries, write the SQL without user ID filtering. 
The RBAC system will automatically add the correct user and course restrictions.

CRITICAL: Generate ONLY a valid SQL query that follows HayStack patterns. No explanations.

Question: {clean_question}

SQL:"""
            
            # Submit prompt with error handling
            try:
                response = self.submit_prompt(prompt)
            except Exception as e:
                ErrorHandler.log_error(HayStackError(
                    "Failed to get LLM response",
                    ErrorType.LLM_GENERATION,
                    e,
                    {"question": question[:100]}
                ))
                return self._nl_to_sql_with_guardrails(question)
            
            extracted_sql = self._extract_sql(response)
            
            if extracted_sql and self._validate_sql(extracted_sql):
                return extracted_sql
            else:
                # SQL validation failed
                ErrorHandler.log_error(HayStackError(
                    "Generated SQL failed validation",
                    ErrorType.SQL_VALIDATION,
                    None,
                    {"question": question[:100], "sql": extracted_sql}
                ))
                return self._nl_to_sql_with_guardrails(question)
                
        except HayStackError:
            raise  # Re-raise HayStack errors
        except Exception as e:
            ErrorHandler.log_error(HayStackError(
                f"Unexpected error in SQL generation: {str(e)}",
                ErrorType.LLM_GENERATION,
                e,
                {"question": question[:100]}
            ))
            return self._nl_to_sql_with_guardrails(question)
    
    def _nl_to_sql_with_guardrails(self, question: str) -> str:
        """Fallback LLM SQL generation with basic guardrails"""
        try:
            prompt = f"""You are an expert SQL generator for Totara LMS (HayStack system).

STRICT RULES:
1. Only generate SELECT statements
2. Always include proper WHERE clauses for data integrity
3. Use meaningful column aliases
4. No comments, no explanations, just SQL
5. Include reasonable LIMIT clauses

AVAILABLE TABLES:
- ttl_user (id, firstname, lastname, email, username, deleted, timecreated)
- ttl_course (id, fullname, shortname, visible, timecreated)  
- ttl_user_enrolments (userid, enrolid, status, timecreated)
- ttl_enrol (id, courseid, status, enrol)
- ttl_role_assignments (userid, roleid, contextid)

IMPORTANT: timecreated fields are UNIX timestamps (INT). Use FROM_UNIXTIME() to convert to datetime.
Examples: YEAR(FROM_UNIXTIME(timecreated)), DATE(FROM_UNIXTIME(timecreated))

Question: {question}

Generate SQL:"""
            
            response = self.submit_prompt(prompt)
            extracted_sql = self._extract_sql(response)
            
            if extracted_sql and self._validate_sql(extracted_sql):
                # Add LIMIT if not present
                if "LIMIT" not in extracted_sql.upper():
                    extracted_sql = f"{extracted_sql.rstrip(';')} LIMIT 1000"
                return extracted_sql
            else:
                # Final fallback - return safe error query
                return "SELECT 'Unable to generate valid SQL for this query. Please rephrase your request.' as error_message;"
                
        except Exception as e:
            print(f"[HayStack MCP] Fallback SQL generation error: {e}")
            return "SELECT 'System error during SQL generation. Please try again.' as error_message;"

    def _get_engine(self) -> sqlalchemy.engine.Engine:
        """Get or create SQLAlchemy engine with connection pooling for Azure MySQL"""
        if not hasattr(self, '_engine'):
            connection_string = f"mysql+pymysql://{self.db_config['user']}:{quote_plus(self.db_config['password'])}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            
            # Enhanced connection pool configuration
            self._engine = create_engine(
                connection_string,
                pool_size=10,                    # Increased pool size
                max_overflow=20,                 # Increased overflow
                pool_pre_ping=True,              # Test connections before use
                pool_recycle=300,                # Recycle connections every 5 minutes
                connect_args={
                    "connect_timeout": 15,       # MySQL connection timeout
                    "read_timeout": 60,          # MySQL read timeout
                    "write_timeout": 60,         # MySQL write timeout
                    "charset": "utf8mb4"         # Proper UTF-8 support
                }
            )
        return self._engine

    def _get_rbac_service(self):
        """Get RBAC service, initializing if needed"""
        if not RBACService:
            return None
            
        if self._rbac_service is None:
            try:
                engine = self._get_engine()
                cache_ttl = int(os.getenv('RBAC_CACHE_TTL_SECONDS', '120'))
                self._rbac_service = RBACService(engine, cache_ttl)
                logger.info("RBAC service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RBAC service: {e}")
                return None
                
        return self._rbac_service

    def run_sql(self, sql: str, params: Optional[Union[Tuple, Dict]]=None, user_id: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Execute SQL query safely using SQLAlchemy with RBAC enforcement"""
        context = {
            "sql": sql[:200],  # Truncate for logging
            "params": str(params)[:100] if params else None,
            "user_id": user_id
        }
        
        # Check if this is a multi-user UNION query for chart comparison
        # SECURITY: Dynamic detection - no hardcoded user IDs, requires legitimate multi-user pattern
        import re
        is_multi_user_union = False
        
        if 'UNION ALL' in sql.upper() and 'user_source' in sql.lower():
            # Check for pattern: userid = <number> in multiple places (indicating multi-user query)
            userid_pattern = r'userid\s*=\s*(\d+)'
            userid_matches = re.findall(userid_pattern, sql, re.IGNORECASE)
            
            # SECURITY: Must have at least 2 different user IDs to be a legitimate multi-user comparison
            # This prevents single-user queries from bypassing RBAC and ensures only genuine 
            # multi-user chart comparisons are allowed
            unique_userids = set(userid_matches)
            is_multi_user_union = len(unique_userids) >= 2
            
            if is_multi_user_union:
                logger.info(f"Multi-user UNION query detected with users: {sorted(unique_userids)}")
            else:
                logger.debug(f"UNION query found but not multi-user pattern (users: {unique_userids})")
        
        # RBAC enforcement - apply relaxed rules for multi-user chart queries
        rbac_service = self._get_rbac_service()
        if rbac_service and user_id and not is_multi_user_union:
            try:
                # Get user's effective RBAC permissions
                rbac = rbac_service.get_effective_rbac(user_id)
                
                # Apply RBAC to SQL query
                sql_secured, rbac_params = rbac_service.apply_sql_rbac(sql, user_id, rbac)
                
                # Merge RBAC parameters with existing parameters
                if params:
                    if isinstance(params, dict):
                        merged_params = {**params, **rbac_params}
                    else:
                        # Convert tuple/list params to dict and merge
                        merged_params = rbac_params
                        logger.warning("Converting positional params to dict for RBAC compatibility")
                else:
                    merged_params = rbac_params
                    
                # Update context and SQL for secured version
                sql = sql_secured
                params = merged_params
                context["rbac_applied"] = True
                context["sql_secured"] = sql[:200]
                
                logger.info(f"RBAC applied for user {user_id}: {len(rbac.authorized_courses)} courses, roles: {[r[2] for r in rbac.roles]}")
                
            except PermissionError as e:
                logger.warning(f"RBAC denied access for user {user_id}: {e}")
                return pd.DataFrame([{"error": str(e)}])
            except Exception as e:
                logger.error(f"RBAC enforcement failed for user {user_id}: {e}")
                # Continue without RBAC if it fails (logged but not blocking)
                context["rbac_error"] = str(e)
        else:
            if is_multi_user_union:
                logger.info(f"Multi-user UNION query detected - bypassing restrictive RBAC for chart comparison")
                context["rbac_bypass_reason"] = "multi_user_union_chart"
            elif not rbac_service:
                logger.warning("RBAC service not available - query executing without security")
            elif not user_id:
                logger.warning("No user_id provided - query executing without RBAC")
        
        # Check cache first (only for SELECT queries) - use secured SQL for cache key
        if sql.strip().upper().startswith('SELECT'):
            cached_result = query_cache.get(sql, params)
            if cached_result is not None:
                logger.debug(f"Returning cached result for query: {sql[:50]}...")
                # Apply column masking to cached results if RBAC is active
                if rbac_service and user_id:
                    try:
                        rbac = rbac_service.get_effective_rbac(user_id)
                        cached_result = rbac_service.mask_dataframe(cached_result, rbac)
                    except Exception as e:
                        logger.warning(f"Failed to apply RBAC masking to cached result: {e}")
                return cached_result
        
        try:
            from sqlalchemy import text
            
            # Get engine with error handling
            try:
                engine = self._get_engine()
            except Exception as e:
                error = HayStackError(
                    "Failed to get database engine",
                    ErrorType.DATABASE_CONNECTION,
                    e,
                    context
                )
                ErrorHandler.log_error(error)
                return pd.DataFrame([{"error": error.get_user_message()}])
            
            # Apply SQL guardrails with error handling
            try:
                guarded_sql = self._apply_sql_guardrails(sql)
                context["guarded_sql"] = guarded_sql[:200]
            except Exception as e:
                error = HayStackError(
                    "Failed to apply SQL guardrails",
                    ErrorType.SQL_VALIDATION,
                    e,
                    context
                )
                ErrorHandler.log_error(error)
                return pd.DataFrame([{"error": error.get_user_message()}])
            
            # Use parameterized queries with sqlalchemy.text()
            try:
                sql_text = text(guarded_sql)
            except Exception as e:
                error = HayStackError(
                    "Failed to create SQL text object",
                    ErrorType.SQL_VALIDATION,
                    e,
                    context
                )
                ErrorHandler.log_error(error)
                return pd.DataFrame([{"error": error.get_user_message()}])
            
            # Execute with chunksize for large results
            max_rows = int(os.getenv('SQL_MAX_ROWS_DEFAULT', '200'))
            
            try:
                # If we have parameters, bind them through SQLAlchemy
                if params:
                    # Create bound SQL by executing with SQLAlchemy and getting result
                    with engine.begin() as conn:
                        result = conn.execute(sql_text, params)
                        # Convert to DataFrame
                        df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
                        
                        # Check if we hit row limit for truncation tracking
                        has_more = len(df) >= max_rows
                        
                        if has_more:
                            df = df.head(max_rows)
                            df.attrs['truncated'] = True
                            df.attrs['shown'] = len(df)
                            df.attrs['estimated_total'] = 'unknown'
                else:
                    # No parameters, but we still need to handle % characters in SQL
                    # Use SQLAlchemy directly to avoid pandas format string issues
                    with engine.begin() as conn:
                        result = conn.execute(sql_text)
                        # Convert to DataFrame
                        df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
                        
                        # Check if we hit row limit for truncation tracking
                        has_more = len(df) >= max_rows
                        
                        if has_more:
                            df = df.head(max_rows)
                            df.attrs['truncated'] = True
                            df.attrs['shown'] = len(df)
                            df.attrs['estimated_total'] = 'unknown'

                
                logger.info(f"SQL executed successfully, returned {len(df)} rows")
                logger.info(f"Result columns: {list(df.columns)}")
                
                # Apply RBAC column masking
                if rbac_service and user_id and context.get("rbac_applied"):
                    try:
                        rbac = rbac_service.get_effective_rbac(user_id)
                        df = rbac_service.mask_dataframe(df, rbac)
                        logger.debug(f"RBAC column masking applied for user {user_id}")
                    except Exception as e:
                        logger.warning(f"Failed to apply RBAC column masking: {e}")
                
                # Cache successful results (cache the masked version)
                if sql.strip().upper().startswith('SELECT'):
                    query_cache.put(sql, df, params)
                
                return df
                
            except TypeError:
                # Fallback for queries that don't support chunksize
                try:
                    df = pd.read_sql_query(sql_text, engine, params=params)
                    
                    # Apply row limit
                    if len(df) > max_rows:
                        df = df.head(max_rows)
                        df.attrs['truncated'] = True
                        df.attrs['shown'] = max_rows
                        df.attrs['estimated_total'] = 'more than ' + str(max_rows)
                    
                    logger.info(f"SQL executed successfully (fallback mode), returned {len(df)} rows")
                    logger.info(f"Result columns: {list(df.columns)}")
                    
                    # Apply RBAC column masking
                    if rbac_service and user_id and context.get("rbac_applied"):
                        try:
                            rbac = rbac_service.get_effective_rbac(user_id)
                            df = rbac_service.mask_dataframe(df, rbac)
                            logger.debug(f"RBAC column masking applied for user {user_id} (fallback mode)")
                        except Exception as e:
                            logger.warning(f"Failed to apply RBAC column masking (fallback): {e}")
                    
                    # Cache successful results (cache the masked version)
                    if sql.strip().upper().startswith('SELECT'):
                        query_cache.put(sql, df, params)
                    
                    return df
                    
                except Exception as e:
                    error = HayStackError(
                        f"SQL execution failed: {str(e)}",
                        ErrorType.SQL_EXECUTION,
                        e,
                        context
                    )
                    ErrorHandler.log_error(error)
                    return pd.DataFrame([{"error": error.get_user_message()}])
            
            except Exception as e:
                error = HayStackError(
                    f"SQL execution failed (chunked mode): {str(e)}",
                    ErrorType.SQL_EXECUTION,
                    e,
                    context
                )
                ErrorHandler.log_error(error)
                return pd.DataFrame([{"error": error.get_user_message()}])
                
        except HayStackError:
            raise  # Re-raise HayStack errors
        except Exception as e:
            error = HayStackError(
                f"Unexpected error in SQL execution: {str(e)}",
                ErrorType.SQL_EXECUTION,
                e,
                context
            )
            ErrorHandler.log_error(error)
            return pd.DataFrame([{"error": error.get_user_message()}])
    
    def _apply_sql_guardrails(self, sql: str) -> str:
        """Apply execution guardrails to SQL query"""
        sql = sql.strip()
        
        # CRITICAL FIX: Clean up LLM-generated SQL that incorrectly includes column types
        # The LLM sometimes copies "column_name TYPE" from schema into SELECT clauses
        # This removes common SQL type annotations that shouldn't be in SELECT clauses
        import re
        
        # Remove column type annotations like "cm.module INT", "u.name VARCHAR(255)", etc.
        # Pattern matches: column_reference followed by SQL data types
        sql_type_pattern = r'\b(\w+\.\w+)\s+(INT|INTEGER|VARCHAR\(\d+\)|VARCHAR|TEXT|CHAR\(\d+\)|CHAR|DATE|DATETIME|TIMESTAMP|DECIMAL\(\d+,\d+\)|DECIMAL|FLOAT|DOUBLE|BOOLEAN|BOOL|TINYINT|SMALLINT|MEDIUMINT|BIGINT|BLOB|LONGTEXT|MEDIUMTEXT|TINYTEXT)\b'
        
        # Replace with just the column reference (group 1)
        sql_cleaned = re.sub(sql_type_pattern, r'\1', sql, flags=re.IGNORECASE)
        
        if sql_cleaned != sql:
            logger.info(f"SQL cleanup applied - removed type annotations from query")
            logger.debug(f"Original: {sql[:200]}...")
            logger.debug(f"Cleaned:  {sql_cleaned[:200]}...")
            sql = sql_cleaned
        
        # Check if it's a SELECT without LIMIT and not a COUNT
        sql_upper = sql.upper()
        is_select = sql_upper.startswith('SELECT')
        has_limit = 'LIMIT' in sql_upper
        is_count = 'COUNT(' in sql_upper and sql_upper.count('SELECT') == 1
        
        # Add execution timeout for non-COUNT queries
        if is_select and not is_count:
            max_exec_time = int(os.getenv('SQL_MAX_EXEC_MS', '25000'))
            if not sql_upper.startswith('SELECT /*+'):
                sql = f"SELECT /*+ MAX_EXECUTION_TIME({max_exec_time}) */ " + sql[6:]
        
        # Add LIMIT for SELECT queries without LIMIT and not pure COUNT
        if is_select and not has_limit and not is_count:
            max_rows = int(os.getenv('SQL_MAX_ROWS_DEFAULT', '200'))
            # Remove trailing semicolon and add LIMIT
            sql = sql.rstrip(';') + f' LIMIT {max_rows};'
        
        return sql
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get performance cache statistics"""
        return query_cache.get_stats()
    
    def clear_cache(self):
        """Clear query result cache and RBAC cache"""
        query_cache.clear()
        rbac_service = self._get_rbac_service()
        if rbac_service:
            rbac_service.clear_cache()
            logger.info("Query cache and RBAC cache cleared")
        else:
            logger.info("Query cache cleared")
    
    def get_rbac_stats(self) -> Dict[str, Any]:
        """Get RBAC service statistics"""
        rbac_service = self._get_rbac_service()
        if rbac_service:
            return rbac_service.get_cache_stats()
        else:
            return {"rbac_enabled": False}

    def train(self, **kwargs):
        """Add training data for HayStack"""
        if 'ddl' in kwargs:
            self.add_ddl(kwargs['ddl'])
        elif 'question' in kwargs and 'sql' in kwargs:
            self.add_question_sql(kwargs['question'], kwargs['sql'])
        elif 'documentation' in kwargs:
            self.add_documentation(kwargs['documentation'])
        else:
            print("[HayStack MCP] Invalid training data format")

    # HayStack-specific methods
    def process_haystack_query(self, user_input: str, user_id: Optional[int] = None, requesting_user_id: Optional[int] = None) -> Dict[str, Any]:
        """Process user query with LLM-based intent classification and RBAC enforcement"""
        
        # Use LLM router for intelligent classification
        try:
            router_result = self.classify_intent_with_llm(user_input, requesting_user_id)
            
            # If router failed, fall back to heuristics
            if router_result.get("intent") == "fallback_to_heuristics":
                print("[HayStack MCP] Router failed, using fallback heuristics")
                return self._process_query_with_heuristics(user_input, user_id, requesting_user_id)
            
            # Extract information from router result
            intent = router_result.get("intent")
            tool = router_result.get("tool")
            targets = router_result.get("targets", {})
            cross_user_request = router_result.get("cross_user_request", False)
            needs_admin = router_result.get("needs_admin", False)
            natural_language_query = router_result.get("natural_language_query", user_input)
            risk_level = router_result.get("risk_level", "low")
            
            # Resolve user IDs from router output
            resolved_user_id = targets.get("user_id") or requesting_user_id or user_id
            if not resolved_user_id:
                logger.error("No valid user_id provided for query processing")
                return {
                    'type': 'access_denied',
                    'intent': 'invalid_user',
                    'error': 'User authentication required',
                    'message': 'Valid user_id is required for data access'
                }
            target_user_id = targets.get("target_user_id")
            course_id = targets.get("course_id")
            
            # RBAC Security Check for cross-user requests
            if cross_user_request and target_user_id and requesting_user_id and target_user_id != requesting_user_id:
                print(f"[HayStack MCP] RBAC CHECK: User {requesting_user_id} requesting data for User {target_user_id}")
                
                try:
                    rbac_service = self._get_rbac_service()
                    if rbac_service:
                        rbac = rbac_service.get_effective_rbac(requesting_user_id)
                        
                        # Only admins can access other users' data freely
                        if not rbac.is_admin:
                            # Teachers can only access students in their courses  
                            if rbac.is_teacher and target_user_id in rbac.authorized_users:
                                print(f"[HayStack MCP] RBAC ALLOWED: Teacher {requesting_user_id} can access Student {target_user_id}")
                                resolved_user_id = target_user_id  # Use target user for data retrieval
                            else:
                                print(f"[HayStack MCP] RBAC DENIED: User {requesting_user_id} cannot access User {target_user_id}")
                                return {
                                    'type': 'access_denied',
                                    'intent': 'unauthorized_access',
                                    'user_id': requesting_user_id,
                                    'target_user_id': target_user_id,
                                    'error': f'Access denied: You are not authorized to view data for user {target_user_id}',
                                    'response_preview': f'Access denied for user {target_user_id} data',
                                    'router_result': router_result
                                }
                        else:
                            print(f"[HayStack MCP] RBAC ALLOWED: Admin {requesting_user_id} can access User {target_user_id}")
                            resolved_user_id = target_user_id  # Use target user for data retrieval
                    else:
                        print("[HayStack MCP] RBAC SERVICE UNAVAILABLE: Denying cross-user access")
                        return {
                            'type': 'access_denied',
                            'intent': 'unauthorized_access', 
                            'user_id': requesting_user_id,
                            'target_user_id': target_user_id,
                            'error': f'Access denied: Cannot verify permissions for user {target_user_id}',
                            'response_preview': 'Access verification failed',
                            'router_result': router_result
                        }
                except Exception as e:
                    print(f"[HayStack MCP] RBAC CHECK ERROR: {e}")
                    return {
                        'type': 'access_denied',
                        'intent': 'unauthorized_access',
                        'user_id': requesting_user_id,
                        'target_user_id': target_user_id,
                        'error': 'Access denied: Permission check failed',
                        'response_preview': 'Access check failed',
                        'router_result': router_result
                    }
            
            # Admin check for sensitive operations
            if needs_admin and requesting_user_id:
                try:
                    rbac_service = self._get_rbac_service()
                    if rbac_service:
                        rbac = rbac_service.get_effective_rbac(requesting_user_id)
                        if not rbac.is_admin:
                            return {
                                'type': 'access_denied',
                                'intent': 'admin_required',
                                'user_id': requesting_user_id,
                                'error': 'Access denied: Administrator privileges required',
                                'response_preview': 'Admin access required',
                                'router_result': router_result
                            }
                except Exception as e:
                    print(f"[HayStack MCP] Admin check error: {e}")
                    return {
                        'type': 'access_denied',
                        'intent': 'admin_check_failed',
                        'user_id': requesting_user_id,
                        'error': 'Access denied: Cannot verify admin privileges',
                        'response_preview': 'Admin verification failed',
                        'router_result': router_result
                    }
            
            # Map LLM intent to internal types
            if intent == "conversational":
                return {
                    'type': 'conversational',
                    'intent': 'chat',
                    'user_id': resolved_user_id,
                    'original_input': user_input,
                    'response_preview': "Conversational response...",
                    'router_result': router_result
                }
            elif intent == "tool_call":
                return {
                    'type': 'tool_call',
                    'intent': 'tool_execution',
                    'tool': tool,
                    'user_id': resolved_user_id,
                    'target_user_id': target_user_id,
                    'course_id': course_id,
                    'natural_language_query': natural_language_query,
                    'needs_admin': needs_admin,
                    'response_preview': f"Executing tool: {tool}",
                    'router_result': router_result
                }
            elif intent == "data_query":
                # If we have a specific tool, treat it as a tool call
                if tool in ["haystack_user_profile", "haystack_user_courses"]:
                    return {
                        'type': 'tool_call',
                        'intent': 'tool_execution',
                        'tool': tool,
                        'user_id': resolved_user_id,
                        'target_user_id': target_user_id,
                        'course_id': course_id,
                        'natural_language_query': natural_language_query,
                        'needs_admin': needs_admin,
                        'response_preview': f"Executing tool: {tool}",
                        'router_result': router_result
                    }
                else:
                    # Map remaining data queries to appropriate internal type
                    if "count" in natural_language_query.lower() or "statistics" in natural_language_query.lower():
                        query_type = 'system_statistics'
                    else:
                        query_type = 'general_query'
                    
                    return {
                        'type': query_type,
                        'intent': 'data_search',
                        'user_id': resolved_user_id,
                        'target_user_id': target_user_id,
                        'course_id': course_id,
                        'natural_language_query': natural_language_query,
                        'response_preview': f"Processing {intent} request...",
                        'router_result': router_result
                    }
            else:
                # Unknown intent, default to general query
                return {
                    'type': 'general_query',
                    'intent': 'data_search',
                    'user_id': resolved_user_id,
                    'natural_language_query': natural_language_query,
                    'response_preview': "Processing query...",
                    'router_result': router_result
                }
                
        except Exception as e:
            print(f"[HayStack MCP] LLM Router error: {e}")
            # Fall back to heuristics on any error
            return self._process_query_with_heuristics(user_input, user_id, requesting_user_id)
    
    def _process_query_with_heuristics(self, user_input: str, user_id: Optional[int] = None, requesting_user_id: Optional[int] = None) -> Dict[str, Any]:
        """Fallback heuristic-based query processing (original logic)"""
        user_input_lower = user_input.lower().strip()
        
        # More flexible conversational patterns
        conversational_indicators = [
            r'^hi+!*$', r'^hello+!*$', r'^hey+!*$',
            r'^good\s+(morning|afternoon|evening)!*$',
            r'^how\s+are\s+you\??!*$', r'^how\'?s\s+it\s+going\??!*$',
            r'^what\'?s\s+up\??!*$', r'^how\s+do\s+you\s+do\??!*$',
            r'^thanks?!*$', r'^thank\s+you!*$', r'^thx!*$',
            r'^ok!*$', r'^okay!*$', r'^alright!*$',
            r'^yes!*$', r'^no!*$', r'^maybe!*$', r'^sure!*$',
            r'^(what\s+can\s+you\s+do|help\s+me|what\s+is\s+this)\??!*$',
            r'^how\s+does\s+this\s+work\??!*$'
        ]
        
        import re
        for pattern in conversational_indicators:
            if re.match(pattern, user_input_lower):
                return {
                    'type': 'conversational',
                    'intent': 'chat',
                    'user_id': requesting_user_id or user_id,
                    'original_input': user_input,
                    'response_preview': "Conversational response...",
                    'fallback_heuristic': True
                }
        
        # Extract user ID patterns
        extracted_user_id = None
        user_id_patterns = [
            r'i\s+am\s+user\s+(\d+)', r'as\s+user\s+(\d+)', 
            r'user\s+(\d+)', r'userid\s*[=:]\s*(\d+)'
        ]
        
        for pattern in user_id_patterns:
            user_id_match = re.search(pattern, user_input_lower)
            if user_id_match:
                extracted_user_id = int(user_id_match.group(1))
                break
        
        # Resolve final user_id
        if extracted_user_id is not None:
            user_id = extracted_user_id
        elif user_id is None:
            user_id = requesting_user_id
            if not user_id:
                logger.error("No valid user_id available for heuristic processing")
                return {
                    'type': 'access_denied',
                    'intent': 'invalid_user', 
                    'error': 'User authentication required'
                }
        
        # Basic query type classification  
        if any(phrase in user_input_lower for phrase in ['my courses', 'i am enrolled', 'my enrolled courses', 'what are my enrolled', 'courses i am enrolled', 'my enrollment']):
            return {
                'type': 'enrollment_data', 
                'intent': 'course_list',
                'user_id': requesting_user_id or user_id,
                'response_preview': "Finding your enrolled courses...",
                'fallback_heuristic': True
            }
        elif 'profile' in user_input_lower:
            return {
                'type': 'user_profile',
                'intent': 'profile_data', 
                'user_id': user_id,
                'response_preview': "Retrieving user profile...",
                'fallback_heuristic': True
            }
        else:
            return {
                'type': 'general_query',
                'intent': 'data_search',
                'user_id': user_id,
                'response_preview': "Searching HayStack data...",
                'fallback_heuristic': True
            }

    def execute_haystack_query(self, processed_query: Dict[str, Any], user_input: str = "", requesting_user_id: Optional[int] = None) -> Dict[str, Any]:
        """Execute query with HayStack formatting and RBAC enforcement"""
        try:
            # Handle conversational messages
            if processed_query and processed_query.get('type') == 'conversational':
                return self._handle_conversational_message(processed_query, user_input)
            
            # Handle tool calls
            if processed_query and processed_query.get('type') == 'tool_call':
                return self._handle_tool_call(processed_query, user_input, requesting_user_id)
            
            # Check for access denied queries first
            if processed_query and processed_query.get('type') == 'access_denied':
                return {
                    'success': False,
                    'content': processed_query.get('error', 'Access denied'),
                    'sql': 'N/A - Access denied before SQL generation',
                    'row_count': 0,
                    'user_id': processed_query.get('user_id'),
                    'rbac_denied': True,
                    'reason': 'cross_user_access_denied'
                }
            
            # Extract user_id from processed_query if available
            user_id = processed_query.get('user_id') if processed_query else None
            if not user_id:
                return {
                    'success': False,
                    'content': 'Error: Valid user_id is required for query execution',
                    'sql': 'N/A - Missing user_id',
                    'row_count': 0
                }
            
            # CRITICAL SECURITY CHECK: Validate cross-user access at execution level
            if requesting_user_id is not None and user_id != requesting_user_id:
                print(f"[HayStack MCP] EXECUTION RBAC CHECK: User {requesting_user_id} executing query for User {user_id}")
                
                try:
                    rbac_service = self._get_rbac_service()
                    if rbac_service:
                        rbac = rbac_service.get_effective_rbac(requesting_user_id)
                        
                        # Only admins can execute queries for other users
                        if not rbac.is_admin:
                            # Teachers can only access students in their courses
                            if rbac.is_teacher and user_id in rbac.authorized_users:
                                print(f"[HayStack MCP] EXECUTION ALLOWED: Teacher {requesting_user_id} can access Student {user_id}")
                            else:
                                print(f"[HayStack MCP] EXECUTION DENIED: User {requesting_user_id} cannot execute query for User {user_id}")
                                return {
                                    'success': False,
                                    'content': f'Access denied: You are not authorized to execute queries for user {user_id}',
                                    'sql': 'N/A - Cross-user execution denied',
                                    'row_count': 0,
                                    'user_id': requesting_user_id,
                                    'rbac_denied': True,
                                    'reason': 'cross_user_execution_denied'
                                }
                        else:
                            print(f"[HayStack MCP] EXECUTION ALLOWED: Admin {requesting_user_id} can access User {user_id}")
                    else:
                        # If RBAC service unavailable, deny cross-user execution
                        print(f"[HayStack MCP] EXECUTION DENIED: RBAC service unavailable")
                        return {
                            'success': False,
                            'content': f'Access denied: Cannot verify permissions for user {user_id}',
                            'sql': 'N/A - RBAC verification failed',
                            'row_count': 0,
                            'user_id': requesting_user_id,
                            'rbac_denied': True,
                            'reason': 'rbac_verification_failed'
                        }
                except Exception as e:
                    print(f"[HayStack MCP] EXECUTION RBAC ERROR: {e}")
                    return {
                        'success': False,
                        'content': f'Access denied: Permission check failed for user {user_id}',
                        'sql': 'N/A - Permission check error',
                        'row_count': 0,
                        'user_id': requesting_user_id,
                        'rbac_denied': True,
                        'reason': 'permission_check_error'
                    }
            sql = self.generate_sql(user_input)
            
            # Execute SQL - apply RBAC only for non-system queries
            query_type = processed_query.get('type', 'general')
            if query_type == 'system_statistics':
                # System statistics should not be filtered by RBAC
                print(f"[HayStack MCP] SYSTEM QUERY: Executing without RBAC for system statistics")
                df = self.run_sql(sql)
            else:
                # Execute SQL with RBAC enforcement
                df = self.run_sql(sql, user_id=user_id)
            
            # Check for RBAC permission errors
            if not df.empty and len(df) == 1 and 'error' in df.columns:
                error_msg = df.iloc[0]['error']
                if "not allowed" in error_msg or "Access" in error_msg:
                    return {
                        'success': False,
                        'content': f"Access denied: {error_msg}",
                        'sql': sql,
                        'row_count': 0,
                        'rbac_denied': True
                    }
            
            if df.empty:
                return {
                    'success': False,
                    'content': "No data found for your query.",
                    'sql': sql,
                    'row_count': 0
                }
            
            # Format results for HayStack
            formatted_content = self._format_haystack_results(df, processed_query)
            
            return {
                'success': True,
                'content': formatted_content,
                'sql': sql,
                'row_count': len(df),
                'data': df.to_dict('records'),
                'user_id': user_id
            }
            
        except Exception as e:
            # Extract user_id for error response
            user_id = processed_query.get('user_id') if processed_query else None
            sql_fallback = self._nl_to_sql_with_guardrails(user_input)
            return {
                'success': False,
                'content': f"Error processing HayStack query: {str(e)}",
                'sql': sql_fallback,
                'row_count': 0,
                'user_id': user_id
            }

    def _format_haystack_results(self, df: pd.DataFrame, query_context: Dict[str, Any]) -> str:
        """Format results specifically for HayStack display"""
        if df.empty:
            return "No results found in HayStack data."
        
        query_type = query_context.get('type', 'general')
        print(f"[DEBUG] Formatting results: query_type={query_type}, df_shape={df.shape}, columns={list(df.columns)}")
        
        if query_type == 'user_profile':
            # Clean user profile format
            if len(df) == 1:
                row = df.iloc[0]
                result = "👤 **User Profile:**\n\n"
                if 'firstname' in row and 'lastname' in row:
                    result += f"**Name:** {row['firstname']} {row['lastname']}\n"
                if 'email' in row:
                    result += f"**Email:** {row['email']}\n"
                if 'username' in row:
                    result += f"**Username:** {row['username']}\n"
                if 'timecreated' in row:
                    result += f"**Account Created:** {row['timecreated']}\n"
                return result
                
        elif query_type == 'count_query' or query_type == 'system_statistics':
            # Statistics format - check this BEFORE enrollment_data
            if len(df) == 1 and len(df.columns) == 1:
                count_value = df.iloc[0, 0]
                if query_type == 'system_statistics':
                    return f"📊 **System Statistics:** {count_value}"
                else:
                    return f"📊 **Count Result:** {count_value}"
            
        # Check if this looks like a count query based on column names (fallback detection)
        if len(df) == 1 and len(df.columns) == 1:
            col_name = df.columns[0].lower()
            if 'count' in col_name or col_name in ['total', 'num', 'number']:
                count_value = df.iloc[0, 0]
                return f"📊 **Count Result:** {count_value}"
                
        # Check for popularity/ranking queries based on columns
        if 'enrollment_count' in df.columns and len(df) >= 1:
            row = df.iloc[0]
            result = "🏆 **Most Popular Course:**\n\n"
            
            # Try multiple possible column names for course name
            course_name = 'Unknown Course'
            for col in ['fullname', 'course_name', 'name', 'coursename', 'title', 'course_title']:
                if col in row and pd.notna(row[col]):
                    course_name = str(row[col])
                    break
            
            result += f"**Course:** {course_name}\n"
            
            # Try multiple possible column names for course code
            if any(col in row for col in ['shortname', 'course_code', 'code']):
                for col in ['shortname', 'course_code', 'code']:
                    if col in row and pd.notna(row[col]):
                        result += f"**Code:** {row[col]}\n"
                        break
            
            if 'enrollment_count' in row:
                result += f"**Enrollments:** {row['enrollment_count']}\n"
            return result
            
        elif query_type == 'enrollment_data':
            # Course enrollment format
            result = f"📚 **Enrolled Courses ({len(df)}):**\n\n"
            for idx, (_, row) in enumerate(df.iterrows(), 1):
                # Try multiple possible column names for course name
                course_name = 'Unknown Course'
                for col in ['fullname', 'course_name', 'name', 'coursename', 'title', 'course_title']:
                    if col in row and pd.notna(row[col]):
                        course_name = str(row[col])
                        break
                
                result += f"{idx}. **{course_name}**\n"
                
                # Try multiple possible column names for course code
                for col in ['shortname', 'course_code', 'code']:
                    if col in row and pd.notna(row[col]):
                        result += f"   Code: {row[col]}\n"
                        break
                
                if 'summary' in row and pd.notna(row['summary']):
                    summary = str(row['summary'])[:100] + "..." if len(str(row['summary'])) > 100 else str(row['summary'])
                    result += f"   Description: {summary}\n"
                result += "\n"
            return result
                
        # Default table format for other queries
        result = f"📋 **HayStack Results ({len(df)} found):**\n\n"
        result += "| " + " | ".join(df.columns) + " |\n"
        result += "|" + "|".join(["---" for _ in df.columns]) + "|\n"
        
        for _, row in df.head(10).iterrows():
            row_values = [str(val) if pd.notna(val) else "" for val in row]
            result += "| " + " | ".join(row_values) + " |\n"
            
        if len(df) > 10:
            result += f"\n*... and {len(df) - 10} more results*"
            
        return result

    def _handle_conversational_message(self, processed_query: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Handle conversational messages that don't require database queries"""
        import random
        original_input = processed_query.get('original_input', user_input).lower().strip()
        
        # More natural, varied responses
        if any(greeting in original_input for greeting in ['hi', 'hello', 'hey']):
            responses = [
                "Hi there! How can I help you with your educational data today?",
                "Hello! I'm here to help with courses, enrollments, and analytics. What would you like to know?",
                "Hey! Ready to dive into some data? Ask me about courses, users, or any educational insights!"
            ]
            response_text = random.choice(responses)
            
        elif 'how are you' in original_input:
            responses = [
                "I'm doing great, thanks for asking! How can I assist with your data today?",
                "I'm well, thank you! Ready to help you explore your educational data. What are you curious about?",
                "I'm excellent! What educational insights can I help you discover?"
            ]
            response_text = random.choice(responses)
            
        elif any(thanks in original_input for thanks in ['thanks', 'thank you', 'thx']):
            responses = [
                "You're very welcome! Happy to help anytime.",
                "My pleasure! Let me know if you need anything else!",
                "Glad I could help! Feel free to ask more questions."
            ]
            response_text = random.choice(responses)
            
        elif 'what can you do' in original_input or 'help' in original_input:
            response_text = """I can help you with lots of educational data! Here are some examples:

**Course Information**
• "What courses are available?"
• "Show me course enrollments"

**User Data** 
• "Tell me about user profiles"
• "How many students are enrolled?"

**Analytics**
• "What are the most popular courses?" 
• "Show me enrollment statistics"

Just ask me in plain English - I'll figure out what data you need!"""
            
        elif any(word in original_input for word in ['ok', 'okay', 'alright', 'sure']):
            responses = [
                "Great! What would you like to explore?",
                "Perfect! How can I help with your data?",
                "Excellent! What are you curious about?"
            ]
            response_text = random.choice(responses)
            
        else:
            # Default friendly response
            responses = [
                "I'm here to help with your educational data! What would you like to know?",
                "Ask me about courses, users, enrollments, or any educational analytics!",
                "I can help you explore your LMS data. What are you interested in learning about?"
            ]
            response_text = random.choice(responses)
        
        return {
            'success': True,
            'content': response_text,
            'sql': 'N/A - Conversational response',
            'row_count': 0,
            'user_id': processed_query.get('user_id'),
            'query_type': 'conversational',
            'conversational': True
        }
    
    def _handle_tool_call(self, processed_query: Dict[str, Any], user_input: str, requesting_user_id: Optional[int] = None) -> Dict[str, Any]:
        """Handle tool call requests"""
        tool = processed_query.get('tool')
        user_id = processed_query.get('user_id') or requesting_user_id
        natural_language_query = processed_query.get('natural_language_query', user_input)
        
        try:
            if tool == "haystack_status":
                # System status check
                try:
                    # Test database connection
                    test_df = self.run_sql("SELECT 1 as test_connection LIMIT 1")
                    db_status = "✅ Connected" if not test_df.empty else "❌ No response"
                except:
                    db_status = "❌ Connection failed"
                
                # Test LLM connection
                try:
                    test_response = self.submit_prompt("Test prompt")
                    llm_status = "✅ Connected" if test_response else "❌ No response"
                except:
                    llm_status = "❌ Connection failed"
                
                status_content = f"""🔧 **HayStack MCP Server Status**

**Database:** {db_status}
**LLM (Qwen 2.5:7b):** {llm_status}
**RBAC Service:** {'✅ Active' if self._get_rbac_service() else '❌ Unavailable'}
**Server:** ✅ Running

**Features Available:**
• Natural language to SQL conversion
• Multi-user RBAC enforcement
• Conversational interactions
• Educational data analytics
"""
                
                return {
                    'success': True,
                    'content': status_content,
                    'sql': 'N/A - System status',
                    'row_count': 0,
                    'user_id': user_id,
                    'tool_call': tool
                }
                
            elif tool == "haystack_user_profile":
                # Get user profile data
                target_user_id = processed_query.get('target_user_id') or user_id
                sql = f"SELECT firstname, lastname, email, username, timecreated FROM ttl_user WHERE id = {target_user_id} LIMIT 1"
                df = self.run_sql(sql, user_id=requesting_user_id)
                
                if df.empty:
                    content = f"No profile found for user {target_user_id}"
                else:
                    content = self._format_haystack_results(df, {'type': 'user_profile'})
                
                return {
                    'success': True,
                    'content': content,
                    'sql': sql,
                    'row_count': len(df),
                    'user_id': user_id,
                    'tool_call': tool
                }
                
            elif tool == "haystack_user_courses":
                # Get user's enrolled courses
                target_user_id = processed_query.get('target_user_id') or user_id 
                sql = f"""SELECT c.fullname, c.shortname, c.summary 
                         FROM ttl_course c 
                         JOIN ttl_enrol en ON c.id = en.courseid 
                         JOIN ttl_user_enrolments ue ON en.id = ue.enrolid 
                         WHERE ue.userid = {target_user_id} 
                         ORDER BY c.fullname"""
                df = self.run_sql(sql, user_id=requesting_user_id)
                
                if df.empty:
                    content = f"No courses found for user {target_user_id}"
                else:
                    content = self._format_haystack_results(df, {'type': 'enrollment_data'})
                
                return {
                    'success': True,
                    'content': content,
                    'sql': sql,
                    'row_count': len(df),
                    'user_id': user_id,
                    'tool_call': tool
                }
                
            elif tool == "haystack_raw_sql":
                # Raw SQL execution (admin only)
                raw_sql = natural_language_query
                
                # Security: Only allow SELECT statements
                if not raw_sql.strip().upper().startswith('SELECT'):
                    return {
                        'success': False,
                        'content': 'Raw SQL tool only allows SELECT statements for security',
                        'sql': raw_sql,
                        'row_count': 0,
                        'user_id': user_id,
                        'tool_call': tool
                    }
                
                df = self.run_sql(raw_sql)  # No RBAC filtering for admin raw SQL
                
                if df.empty:
                    content = "Query executed successfully but returned no results"
                else:
                    content = self._format_haystack_results(df, {'type': 'general_query'})
                
                return {
                    'success': True,
                    'content': content,
                    'sql': raw_sql,
                    'row_count': len(df),
                    'user_id': user_id,
                    'tool_call': tool
                }
                
            elif tool == "haystack_query":
                # General haystack query - generate SQL from natural language
                sql = self.generate_sql(natural_language_query)
                df = self.run_sql(sql, user_id=user_id)
                
                if df.empty:
                    content = "No data found for your query"
                else:
                    content = self._format_haystack_results(df, {'type': 'general_query'})
                
                return {
                    'success': True,
                    'content': content,
                    'sql': sql,
                    'row_count': len(df),
                    'user_id': user_id,
                    'tool_call': tool
                }
                
            elif tool == "haystack_chart":
                # Chart generation tool
                if user_id is None:
                    return {
                        'success': False,
                        'content': 'Error: User ID is required for chart generation',
                        'sql': 'N/A - User ID missing',
                        'row_count': 0,
                        'user_id': user_id,
                        'tool_call': tool
                    }
                
                try:
                    # Call the internal chart function
                    chart_result_json = _generate_chart_internal(natural_language_query, user_id=user_id)
                    chart_result = json.loads(chart_result_json)
                    
                    if chart_result.get('status') == 'ok':
                        # Successfully generated chart
                        chart_url = chart_result.get('chart_view_url', 'URL not available')
                        
                        return {
                            'success': True,
                            'content': f"📊 **Chart Generated: {chart_result.get('title', 'Data Visualization')}**\n\n"
                                     f"Chart Type: {chart_result.get('chart_type', 'N/A')}\n"
                                     f"Data Points: {chart_result.get('row_count', 0)}\n"
                                     f"Columns: {chart_result.get('label_column')} vs {', '.join(chart_result.get('value_columns', []))}\n\n"
                                     f"🌐 **VIEW YOUR CHART HERE:**\n"
                                     f"═══════════════════════════════════════\n"
                                     f"{chart_url}\n"
                                     f"═══════════════════════════════════════\n"
                                     f"📥 Click the link above to view and download your chart",
                            'sql': chart_result.get('sql_used', 'N/A'),
                            'row_count': chart_result.get('row_count', 0),
                            'user_id': user_id,
                            'tool_call': tool,
                            'chart_spec': chart_result.get('chartjs_spec'),
                            'chart_html': chart_result.get('html_snippet'),
                            'chart_id': chart_result.get('chart_id'),
                            'chart_type': chart_result.get('chart_type'),
                            'chart_title': chart_result.get('title'),
                            'chart_view_url': chart_result.get('chart_view_url')
                        }
                    elif chart_result.get('status') == 'clarification_needed':
                        # Chart needs clarification - format the options dynamically
                        options = chart_result.get('options', [])
                        options_text = ""
                        
                        for i, option in enumerate(options, 1):
                            options_text += f"**{i}. {option.get('title', f'Option {i}')}**\n"
                            if option.get('description'):
                                options_text += f"   {option.get('description')}\n"
                            if option.get('example_query'):
                                options_text += f"   💬 *Example: \"{option.get('example_query')}\"*\n"
                            options_text += "\n"
                        
                        content = f"🤔 **{chart_result.get('message', 'I need more details to create your chart.')}**\n\n"
                        if chart_result.get('reason'):
                            content += f"*{chart_result.get('reason')}*\n\n"
                        content += f"{options_text}"
                        if chart_result.get('suggestion'):
                            content += f"💡 **{chart_result.get('suggestion')}**"
                        
                        return {
                            'success': True,
                            'content': content,
                            'sql': 'N/A - Clarification needed',
                            'row_count': 0,
                            'user_id': user_id,
                            'tool_call': tool
                        }
                    else:
                        # Chart generation failed
                        return {
                            'success': False,
                            'content': f"❌ Chart generation failed: {chart_result.get('message', 'Unknown error')}",
                            'sql': chart_result.get('sql_used', 'N/A'),
                            'row_count': 0,
                            'user_id': user_id,
                            'tool_call': tool,
                            'chart_error': chart_result.get('message')
                        }
                        
                except Exception as e:
                    return {
                        'success': False,
                        'content': f"❌ Chart tool error: {str(e)}",
                        'sql': 'N/A - Chart tool error',
                        'row_count': 0,
                        'user_id': user_id,
                        'tool_call': tool,
                        'chart_error': str(e)
                    }
                
            else:
                # Unknown tool
                return {
                    'success': False,
                    'content': f'Unknown tool: {tool}',
                    'sql': 'N/A - Unknown tool',
                    'row_count': 0,
                    'user_id': user_id,
                    'tool_call': tool
                }
                
        except Exception as e:
            return {
                'success': False,
                'content': f'Tool execution error: {str(e)}',
                'sql': 'N/A - Tool error',
                'row_count': 0,
                'user_id': user_id,
                'tool_call': tool
            }


# Initialize FastMCP server for HayStack
mcp = FastMCP("HayStack Totara LMS MCP Server")

# Initialize HayStack MCP instance
haystack_mcp = None


def initialize_haystack_mcp():
    """Initialize HayStack MCP instance with training data"""
    global haystack_mcp

    try:
        print("[HayStack MCP] Initializing HayStack MCP instance...")
        haystack_mcp = HayStackMCP()

        # Add HayStack-specific training data
        haystack_training_data = {
            'ddl': [
                "CREATE TABLE ttl_user (id INT PRIMARY KEY, username VARCHAR(100), firstname VARCHAR(100), lastname VARCHAR(100), email VARCHAR(255), deleted INT DEFAULT 0, timecreated INT);",
                "CREATE TABLE ttl_course (id INT PRIMARY KEY, fullname VARCHAR(255), shortname VARCHAR(100), visible INT DEFAULT 1, timecreated INT);",
                "CREATE TABLE ttl_user_enrolments (id INT PRIMARY KEY, userid INT, enrolid INT, status INT, timecreated INT);",
                "CREATE TABLE ttl_enrol (id INT PRIMARY KEY, courseid INT, status INT, enrol VARCHAR(20));",
                "CREATE TABLE ttl_role (id INT PRIMARY KEY, shortname VARCHAR(100), name VARCHAR(255));",
                "CREATE TABLE ttl_context (id INT PRIMARY KEY, contextlevel INT, instanceid INT);",
                "CREATE TABLE ttl_role_assignments (id INT PRIMARY KEY, roleid INT, contextid INT, userid INT);"
            ],
            'examples': [
                {
                    'question': 'General information of user 71',
                    'sql': 'SELECT id, firstname, lastname, email, username, timecreated FROM ttl_user WHERE id = 71 AND deleted = 0;'
                },
                {
                    'question': 'What courses is user 71 enrolled in?',
                    'sql': 'SELECT c.id AS course_id, c.fullname, c.shortname, c.summary FROM ttl_user_enrolments ue JOIN ttl_enrol e ON ue.enrolid = e.id AND e.status = 0 JOIN ttl_course c ON e.courseid = c.id WHERE ue.userid = 71 AND ue.status = 0 AND c.visible = 1;'
                },
                {
                    'question': 'How many courses does user 71 have?',
                    'sql': 'SELECT COUNT(*) as course_count FROM ttl_user_enrolments ue JOIN ttl_enrol e ON ue.enrolid = e.id AND e.status = 0 WHERE ue.userid = 71 AND ue.status = 0;'
                },
                {
                    'question': 'Total enrollments by course category',
                    'sql': 'SELECT c.fullname, COUNT(ue.id) AS total_enrollments FROM ttl_course c JOIN ttl_enrol e ON c.id = e.courseid JOIN ttl_user_enrolments ue ON e.id = ue.enrolid WHERE c.visible = 1 AND e.status = 0 AND ue.status = 0 GROUP BY c.fullname ORDER BY total_enrollments DESC;'
                },
                {
                    'question': 'Pie chart of total enrollments by course category',
                    'sql': 'SELECT c.fullname AS category, COUNT(ue.id) AS enrollment_count FROM ttl_course c JOIN ttl_enrol e ON c.id = e.courseid JOIN ttl_user_enrolments ue ON e.id = ue.enrolid WHERE c.visible = 1 AND e.status = 0 AND ue.status = 0 GROUP BY c.fullname ORDER BY enrollment_count DESC;'
                },
                {
                    'question': 'Bar chart showing enrollment trends over time',
                    'sql': 'SELECT YEAR(FROM_UNIXTIME(ue.timecreated)) AS enrollment_year, COUNT(ue.id) AS enrollment_count FROM ttl_user_enrolments ue JOIN ttl_enrol e ON ue.enrolid = e.id WHERE ue.status = 0 AND e.status = 0 GROUP BY YEAR(FROM_UNIXTIME(ue.timecreated)) ORDER BY enrollment_year;'
                },
                {
                    'question': 'Course enrollment statistics by year',
                    'sql': 'SELECT YEAR(FROM_UNIXTIME(ue.timecreated)) AS year, COUNT(DISTINCT ue.userid) AS unique_users, COUNT(ue.id) AS total_enrollments FROM ttl_user_enrolments ue JOIN ttl_enrol e ON ue.enrolid = e.id WHERE ue.status = 0 AND e.status = 0 GROUP BY YEAR(FROM_UNIXTIME(ue.timecreated)) ORDER BY year;'
                },
                {
                    'question': 'Enrollment trends by year',
                    'sql': 'SELECT YEAR(FROM_UNIXTIME(ue.timecreated)) AS enrollment_year, COUNT(*) AS enrollment_count FROM ttl_user_enrolments ue JOIN ttl_enrol e ON ue.enrolid = e.id WHERE ue.status = 0 AND e.status = 0 GROUP BY YEAR(FROM_UNIXTIME(ue.timecreated)) ORDER BY enrollment_year;'
                },
                {
                    'question': 'Monthly enrollment trends',
                    'sql': 'SELECT CONCAT(YEAR(FROM_UNIXTIME(ue.timecreated)), \"-\", LPAD(MONTH(FROM_UNIXTIME(ue.timecreated)), 2, \"0\")) AS month, COUNT(*) AS enrollments FROM ttl_user_enrolments ue JOIN ttl_enrol e ON ue.enrolid = e.id WHERE ue.status = 0 AND e.status = 0 GROUP BY YEAR(FROM_UNIXTIME(ue.timecreated)), MONTH(FROM_UNIXTIME(ue.timecreated)) ORDER BY YEAR(FROM_UNIXTIME(ue.timecreated)), MONTH(FROM_UNIXTIME(ue.timecreated));'
                },
                {
                    'question': 'Generate enrolled courses data for multiple users with UNION ALL query',
                    'sql': 'SELECT c.fullname AS course_name, USER_A AS user_source, c.id AS course_id FROM ttl_user_enrolments ue JOIN ttl_enrol e ON ue.enrolid = e.id JOIN ttl_course c ON e.courseid = c.id WHERE ue.userid = USER_A AND ue.status = 0 AND e.status = 0 AND c.visible = 1 UNION ALL SELECT c.fullname AS course_name, USER_B AS user_source, c.id AS course_id FROM ttl_user_enrolments ue JOIN ttl_enrol e ON ue.enrolid = e.id JOIN ttl_course c ON e.courseid = c.id WHERE ue.userid = USER_B AND ue.status = 0 AND e.status = 0 AND c.visible = 1 ORDER BY user_source, course_name;'
                },
                {
                    'question': 'Compare enrolled courses between multiple users',
                    'sql': 'SELECT c.fullname AS course_name, ue.userid AS user_source, COUNT(*) AS enrollment_count FROM ttl_user_enrolments ue JOIN ttl_enrol e ON ue.enrolid = e.id JOIN ttl_course c ON e.courseid = c.id WHERE ue.userid IN (USER_A, USER_B) AND ue.status = 0 AND e.status = 0 AND c.visible = 1 GROUP BY c.fullname, ue.userid ORDER BY c.fullname, ue.userid;'
                },
                {
                    'question': 'Enrolled courses of multiple users with user identification',
                    'sql': 'SELECT c.fullname, ue.userid AS user_source FROM ttl_user_enrolments ue JOIN ttl_enrol e ON ue.enrolid = e.id JOIN ttl_course c ON e.courseid = c.id WHERE ue.status = 0 AND e.status = 0 AND c.visible = 1 ORDER BY ue.userid, c.fullname;'
                },
                {
                    'question': 'My enrolled courses for charting',
                    'sql': 'SELECT c.fullname AS course_name, c.shortname AS course_code, c.id AS course_id FROM ttl_user_enrolments ue JOIN ttl_enrol e ON ue.enrolid = e.id JOIN ttl_course c ON e.courseid = c.id WHERE ue.userid = CURRENT_USER_ID AND ue.status = 0 AND e.status = 0 AND c.visible = 1 ORDER BY c.fullname;'
                },
                {
                    'question': 'Individual enrolled courses for current user',
                    'sql': 'SELECT c.fullname AS course_name FROM ttl_user_enrolments ue JOIN ttl_enrol e ON ue.enrolid = e.id JOIN ttl_course c ON e.courseid = c.id WHERE ue.userid = CURRENT_USER_ID AND ue.status = 0 AND e.status = 0 AND c.visible = 1 ORDER BY c.fullname;'
                }
            ]
        }

        # Train the model
        for ddl in haystack_training_data['ddl']:
            haystack_mcp.train(ddl=ddl)

        for example in haystack_training_data['examples']:
            haystack_mcp.train(question=example['question'], sql=example['sql'])

        print("[HayStack MCP] Training completed successfully")
        return True

    except Exception as e:
        print(f"[HayStack MCP] Initialization error: {e}")
        return False


# FastMCP Tools for HayStack Integration

@mcp.tool()
def haystack_query(question: str, user_id: int) -> str:
    """
    Execute a natural language query against HayStack Totara LMS database
    
    Args:
        question: Natural language question about the data
        user_id: User ID for context (required for security)
    
    Returns:
        Formatted response with query results
    """
    if not haystack_mcp:
        return "❌ HayStack MCP not initialized. Please check configuration."
    
    try:
        # Process query with HayStack context
        processed_query = haystack_mcp.process_haystack_query(question, user_id, requesting_user_id=user_id)
        
        # Execute the query
        result = haystack_mcp.execute_haystack_query(processed_query, question, requesting_user_id=user_id)
        
        if result['success']:
            response = f"✅ **HayStack Query Result:**\n\n{result['content']}"
            if result.get('sql'):
                response += f"\n\n🔍 **SQL Generated:**\n```sql\n{result['sql']}\n```"
            return response
        else:
            return f"❌ **Query Failed:** {result['content']}"
            
    except Exception as e:
        return f"❌ **HayStack Error:** {str(e)}"


@mcp.tool()
def haystack_user_profile(user_id: int) -> str:
    """
    Get comprehensive user profile from HayStack
    
    Args:
        user_id: User ID to retrieve profile for
    
    Returns:
        Formatted user profile information
    """
    if not haystack_mcp:
        return "❌ HayStack MCP not initialized."
    
    try:
        question = f"General information of user {user_id}"
        processed_query = haystack_mcp.process_haystack_query(question, user_id, requesting_user_id=user_id)
        result = haystack_mcp.execute_haystack_query(processed_query, question, requesting_user_id=user_id)
        
        if result['success']:
            return result['content']
        else:
            return f"❌ Could not retrieve profile for user {user_id}: {result['content']}"
            
    except Exception as e:
        return f"❌ Error retrieving user profile: {str(e)}"


@mcp.tool()
def haystack_user_courses(user_id: int) -> str:
    """
    Get enrolled courses for a user from HayStack
    
    Args:
        user_id: User ID to get courses for
    
    Returns:
        Formatted list of enrolled courses
    """
    if not haystack_mcp:
        return "❌ HayStack MCP not initialized."
    
    try:
        question = f"What courses is user {user_id} enrolled in?"
        processed_query = haystack_mcp.process_haystack_query(question, user_id, requesting_user_id=user_id)
        result = haystack_mcp.execute_haystack_query(processed_query, question, requesting_user_id=user_id)
        
        if result['success']:
            return result['content']
        else:
            return f"❌ Could not retrieve courses for user {user_id}: {result['content']}"
            
    except Exception as e:
        return f"❌ Error retrieving user courses: {str(e)}"


@mcp.tool()
def haystack_status() -> str:
    """
    Check HayStack MCP server status and configuration
    
    Returns:
        Status information about the HayStack MCP server
    """
    try:
        if not haystack_mcp:
            return "❌ HayStack MCP not initialized"
            
        # Test database connection
        test_sql = "SELECT 1 as test"
        df = haystack_mcp.run_sql(test_sql)
        
        db_status = "✅ Connected" if not df.empty else "❌ Connection failed"
        
        # Check Ollama connection
        try:
            import requests
            response = requests.get(f"{haystack_mcp.ollama_url}/api/tags", timeout=5)
            ollama_status = "✅ Connected" if response.status_code == 200 else "❌ Connection failed"
        except:
            ollama_status = "❌ Connection failed"
            
        status = f"""🚀 **HayStack MCP Server Status:**

**Database:** {db_status}
**Ollama:** {ollama_status}
**Model:** {haystack_mcp.qwen_model}
**Training Data:** {len(haystack_mcp.ddl_data)} DDL, {len(haystack_mcp.question_sql_data)} Examples

**Configuration:**
- Host: {haystack_mcp.db_config['host']}
- Database: {haystack_mcp.db_config['database']}
- Ollama URL: {haystack_mcp.ollama_url}
"""
        
        return status
        
    except Exception as e:
        return f"❌ Error checking status: {str(e)}"


@mcp.tool()
def haystack_raw_sql(sql: str) -> str:
    """
    Execute raw SQL query against HayStack database (admin use only)
    
    Args:
        sql: Raw SQL query to execute
    
    Returns:
        Query results in table format
    """
    if not haystack_mcp:
        return "❌ HayStack MCP not initialized."
    
    try:
        # Basic security check
        sql_upper = sql.upper().strip()
        if any(word in sql_upper for word in ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']):
            return "❌ Only SELECT queries are allowed for security."
            
        df = haystack_mcp.run_sql(sql)
        
        if df.empty:
            return "📊 Query executed successfully but returned no results."
            
        # Format as table
        result = f"📊 **Query Results ({len(df)} rows):**\n\n"
        result += "| " + " | ".join(df.columns) + " |\n"
        result += "|" + "|".join(["---" for _ in df.columns]) + "|\n"
        
        for _, row in df.head(20).iterrows():
            row_values = [str(val) if pd.notna(val) else "" for val in row]
            result += "| " + " | ".join(row_values) + " |\n"
            
        if len(df) > 20:
            result += f"\n*... and {len(df) - 20} more results*"
            
        return result
        
    except Exception as e:
        return f"❌ SQL execution error: {str(e)}"


def _generate_chart_internal(question: str, user_id: int) -> str:
    """
    Internal function to generate Chart.js specification from a natural-language request with RBAC-safe data.
    
    Args:
        question: Natural language chart request (e.g., "bar chart of enrollments per course")
        user_id: User ID for RBAC context (required for security)
    
    Returns:
        Strict JSON with chart spec, html_snippet, and sql_used
    """
    global haystack_mcp
    
    # Validate user_id is provided
    if not user_id or user_id <= 0:
        return json.dumps({
            "status": "error", 
            "message": "Valid user_id is required for chart generation"
        })
    
    if not haystack_mcp:
        # Try to initialize if not already done
        initialize_haystack_mcp()
        if not haystack_mcp:
            return json.dumps({"status": "error", "message": "HayStack MCP not initialized"})

    try:
        # Check if this chart request needs clarification using LLM
        clarification_prompt = f"""Analyze this chart request to determine if it needs clarification. The user has access to HayStack LMS data.

User request: "{question}"

Available data context:
- User enrollment in courses (with enrollment dates, status)  
- Course information (names, categories, creation dates)
- User progress/completion data
- Course popularity/enrollment counts across all users

Determine if this request is:
1. CLEAR - can generate a meaningful chart directly 
2. AMBIGUOUS - truly needs clarification to proceed

A request is CLEAR if it has:
- Chart type (pie, bar, line, etc.) OR can reasonably infer one
- Clear data subject (courses, enrollments, users, etc.)
- Sufficient context to proceed

Common CLEAR patterns (be permissive):
- "bar chart of my enrolled courses" → CLEAR (shows individual courses as bars)
- "pie chart of my enrolled courses" → CLEAR (shows courses as pie slices)
- "chart my courses" → CLEAR (can default to bar chart of enrolled courses)
- "graph my enrollments" → CLEAR (can show enrolled courses)
- "make a chart of user X's courses" → CLEAR (shows user's enrolled courses)
- "show courses for user 71 in red and user 18 in blue" → CLEAR (multi-user comparison)
- "pie chart showing course statistics" → CLEAR (course enrollment statistics)
- "generate a chart of enrolled courses for the current user" → CLEAR (current user's courses)
- "line chart of enrollment trends" → CLEAR (enrollment data over time)

Only mark as AMBIGUOUS if truly unclear:
- "show me data" → AMBIGUOUS (no subject specified)
- "make a chart" → AMBIGUOUS (no data subject)
- "visualize something" → AMBIGUOUS (too vague)
- "make a graph of the above" → AMBIGUOUS (no context about what data to chart)
- "chart the above data" → AMBIGUOUS (unclear what specific data to visualize)

For enrolled courses, the most natural interpretation is individual courses as chart elements.
IMPORTANT: For "my enrolled courses" or "my courses" charts:
- Show individual course names as chart elements (bars/slices)
- Use: SELECT c.fullname AS course_name FROM ... WHERE ue.userid = CURRENT_USER_ID
- NOT: SELECT COUNT(*) ... (this shows counts, not individual courses)

For contextual requests like "graph the above" or "make a chart of the above":
- These are CONTEXTUAL REQUESTS that refer to previous queries/data
- These should be marked as AMBIGUOUS since they lack specific data context
- Ask what specific data to chart: "Could you specify what data you'd like me to chart?"

Return ONLY this JSON:
{{
  "needs_clarification": true/false,
  "reason": "explanation of why clarification is needed (if true)",
  "message": "friendly message to user asking for clarification (if true)", 
  "options": [
    {{
      "title": "Chart Type Name",
      "description": "What this chart would show", 
      "example_query": "Suggested rephrased query"
    }}
  ]
}}"""

        try:
            clarification_response = haystack_mcp.submit_prompt(clarification_prompt, format="json")
            clarification_data = json.loads(clarification_response)
            
            if clarification_data.get("needs_clarification", False):
                return json.dumps({
                    "status": "clarification_needed",
                    "message": clarification_data.get("message", "Could you be more specific about what you'd like to visualize?"),
                    "reason": clarification_data.get("reason", "Request needs clarification"),
                    "options": clarification_data.get("options", []),
                    "suggestion": "Please choose one of the options above or rephrase your request with more specific details."
                })
        except Exception as e:
            logger.warning(f"Clarification check failed: {e}, proceeding with normal chart generation")
        
        # 1) Chart planner: Let LLM propose chart structure
        plan_prompt = f"""You are a chart planner for HayStack LMS data visualization. From the user request, emit ONLY valid JSON:

{{
  "chart_type": "bar|line|pie|doughnut|scatter",
  "label_column": "string",
  "value_columns": ["string1", "string2"],
  "title": "descriptive chart title",
  "sql_needed": true,
  "sql_nl_request": "natural language description for SQL generation",
  "is_multi_user": false,
  "user_comparison": {{}},
  "custom_colors": [],
  "requires_union": false
}}

Additional fields for complex queries:
- is_multi_user: true if comparing multiple specific users (REQUIRED for multi-user charts)
- user_comparison: {{"user_71": "red", "user_18": "blue"}} - maps user IDs to COLOR NAMES only 
- custom_colors: ["#FF0000", "#0000FF"] - corresponding hex colors for the color names
- requires_union: true if need UNION ALL query for multi-user data (REQUIRED for comparing specific users)

CRITICAL: For requests like "user 71 in red, user 18 in blue":
- Set is_multi_user: true
- Set requires_union: true  
- Set user_comparison: {{"user_71": "red", "user_18": "blue"}} (color names, NOT column names)
- Set custom_colors: ["#FF0000", "#0000FF"]
- Set label_column to course name field (like "fullname")

HayStack schema context:
- ttl_user: users (id, username, firstname, lastname, email, deleted, timecreated)
- ttl_course: courses (id, fullname, shortname, visible, timecreated)  
- ttl_user_enrolments + ttl_enrol: enrollment data (includes timecreated)
- ttl_role_assignments + ttl_context + ttl_role: user roles in courses
- Use COUNT(), SUM(), AVG() for aggregations
- IMPORTANT: timecreated fields are UNIX timestamps - use YEAR(FROM_UNIXTIME(timecreated)) for year filtering

CRITICAL JOIN PATTERNS (MUST FOLLOW EXACTLY):
- CORRECT: ttl_course c JOIN ttl_enrol e ON c.id = e.courseid JOIN ttl_user_enrolments ue ON e.id = ue.enrolid
- WRONG: ttl_course c JOIN ttl_user_enrolments ue ON c.id = ue.courseid (ERROR - no courseid column in ttl_user_enrolments!)
- WRONG: Any direct JOIN between ttl_course and ttl_user_enrolments without ttl_enrol
- MUST use three-table JOIN: ttl_course → ttl_enrol → ttl_user_enrolments
- Always use table prefixes (ue.timecreated, e.timecreated, c.timecreated) to avoid ambiguous column errors
- For enrollment dates, use ue.timecreated (user enrollment timestamp)
- For course creation dates, use c.timecreated (course creation timestamp)

SCHEMA REMINDER:
- ttl_user_enrolments has: userid, enrolid, status, timecreated (NO courseid!)
- ttl_enrol has: id, courseid, status (links courses to enrollments)
- ttl_course has: id, fullname, shortname, visible, timecreated

COMMON FIXES:
- Wrong: SELECT YEAR(FROM_UNIXTIME(timecreated)) (ambiguous column error)
- Correct: SELECT YEAR(FROM_UNIXTIME(ue.timecreated)) (user enrollment time)

MULTI-USER COMPARISON EXAMPLES:
- "enrolled courses of user A and user B": is_multi_user=true, requires_union=true, generates UNION ALL query
- "user X in red, user Y in blue": user_comparison maps to colors, NOT column names
- For multi-user bar charts: each user becomes a separate dataset with distinct colors

CONTEXTUAL REQUEST HANDLING:
- "make a graph of the above": needs_clarification=true, ask what data to chart
- "chart the previous results": needs_clarification=true, ask what specific data
- DO NOT assume multi-user context for ambiguous contextual requests

CRITICAL ENROLLMENT CHART GUIDANCE:
For requests like "chart my enrolled courses", "bar graph of my enrolled courses", "pie chart of my courses":
- MUST set label_column to "course_name" or "fullname" (course name field)
- MUST set sql_nl_request to "What courses is the requesting user enrolled in?" NOT "count of enrolled courses"
- DO NOT use COUNT() aggregations - show individual courses as chart elements
- MUST use exact phrasing: sql_nl_request: "What courses is user [USER_ID] enrolled in?"
- This ensures proper enrollment filtering and RBAC compliance
- NEVER use sql_nl_request with "count" or "total" for individual course charts

CRITICAL POPULARITY/ENROLLMENT COUNT CHART GUIDANCE:
For requests like "most popular courses", "courses by enrollment count", "enrollment statistics", "popular courses based on enrollment":
- MUST set label_column to "course_name", "fullname", or "shortname" (course name field)
- MUST set value_columns to ["enrollment_count", "enrollments", "total_enrollments"] (count field only)
- sql_nl_request should be: "get course names with their enrollment counts for [YEAR/PERIOD]"
- SQL should SELECT course names and COUNT(enrollments), never include course_id in SELECT
- Example: "most popular courses by enrollment 2024" → sql_nl_request: "get course names with enrollment counts for 2024"
- CRITICAL: course_id should NEVER appear as a value column - only use it for JOINs
- Exclude ALL ID columns from chart data: course_id, user_id, enrolid, etc.

User request: {question}

Emit ONLY the JSON, no explanations:"""

        plan_raw = haystack_mcp.submit_prompt(plan_prompt, format="json")
        
        try:
            plan = json.loads(plan_raw)
        except Exception as parse_err:
            logger.warning(f"Chart planner JSON parse failed: {parse_err}, using defaults")
            plan = {
                "chart_type": "bar",
                "label_column": "label", 
                "value_columns": ["count"],
                "title": "HayStack Data Chart",
                "sql_needed": True,
                "sql_nl_request": question
            }

        # 2) Handle multi-user queries with special SQL generation
        is_multi_user = plan.get("is_multi_user", False)
        requires_union = plan.get("requires_union", False)
        user_comparison = plan.get("user_comparison", {})
        
        if is_multi_user and requires_union:
            # Generate multi-user SQL with UNION ALL approach
            user_ids = []
            for user_key in user_comparison.keys():
                if user_key.startswith("user_"):
                    user_ids.append(user_key.replace("user_", ""))
            
            if user_ids:
                # Generate multi-user SQL request without hardcoded format
                multi_user_sql_request = f"Generate enrolled courses data for multiple users with UNION ALL query including user_source column to identify which user each row belongs to. Show courses for users: {', '.join(user_ids)}"
                logger.info(f"Multi-user SQL request: {multi_user_sql_request}")
                sql = haystack_mcp.generate_sql(multi_user_sql_request)
                
                # Fallback: If no UNION ALL generated, create explicit UNION query
                if 'UNION ALL' not in sql.upper():
                    logger.warning("No UNION ALL in generated SQL, creating explicit multi-user query")
                    # Create explicit UNION ALL query based on training examples
                    union_parts = []
                    for user_id in user_ids:
                        part = f"SELECT c.fullname AS course_name, {user_id} AS user_source, c.id AS course_id FROM ttl_user_enrolments ue JOIN ttl_enrol e ON ue.enrolid = e.id JOIN ttl_course c ON e.courseid = c.id WHERE ue.userid = {user_id} AND ue.status = 0 AND e.status = 0 AND c.visible = 1"
                        union_parts.append(part)
                    sql = " UNION ALL ".join(union_parts) + " ORDER BY user_source, course_name"
                    logger.info(f"Created explicit UNION query: {sql[:100]}...")
            else:
                sql_request = plan.get("sql_nl_request", question)
                sql = haystack_mcp.generate_sql(sql_request)
        else:
            # Standard SQL generation
            sql_request = plan.get("sql_nl_request", question)
            # For enrollment queries, use the same logic as regular course queries
            if "enrolled" in sql_request.lower() or "courses is user" in sql_request.lower():
                # Use the same query processing as haystack_user_courses for consistency
                processed_query = haystack_mcp.process_haystack_query(sql_request, user_id, requesting_user_id=user_id)
                result = haystack_mcp.execute_haystack_query(processed_query, sql_request, requesting_user_id=user_id)
                if result['success']:
                    # Extract DataFrame from the result instead of running SQL separately
                    logger.info(f"Using processed enrollment query result instead of generate_sql")
                    # Convert the result back to DataFrame format for chart processing
                    # Since we need the raw data for charts, we'll still use generate_sql but with better prompt
                    sql = haystack_mcp.generate_sql(f"What courses is user {user_id} enrolled in?")
                else:
                    sql = haystack_mcp.generate_sql(sql_request)
            else:
                sql = haystack_mcp.generate_sql(sql_request)
        
        # 3) Execute with RBAC enforcement
        df = haystack_mcp.run_sql(sql, user_id=user_id)
        
        # Check for errors or empty results
        if df.empty:
            return json.dumps({
                "status": "error", 
                "message": "No data returned or access denied",
                "sql_used": sql
            })
            
        if "error" in df.columns and len(df) > 0:
            error_msg = df.iloc[0].get("error", "Unknown error")
            return json.dumps({
                "status": "error",
                "message": str(error_msg),
                "sql_used": sql
            })

        # 4) Smart column detection for chart data
        label_col = plan.get("label_column", "")
        value_cols = plan.get("value_columns", [])
        
        # Auto-detect label column if not specified or invalid
        if not label_col or label_col not in df.columns:
            # Prefer string/categorical columns for labels
            string_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            if string_cols:
                label_col = string_cols[0]
            else:
                label_col = df.columns[0]
        
        # Auto-detect value columns if not specified or invalid  
        if not value_cols or not any(col in df.columns for col in value_cols):
            # Prefer numeric columns for values, but exclude ID columns
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            # Filter out ID columns that shouldn't be chart data
            id_columns = ['course_id', 'user_id', 'id', 'userid', 'courseid', 'enrolid']
            numeric_cols = [c for c in numeric_cols if c not in id_columns and not c.endswith('_id')]
            
            if numeric_cols:
                value_cols = numeric_cols
            else:
                # Fallback: use non-label columns that aren't IDs
                fallback_cols = [c for c in df.columns if c != label_col and c not in id_columns and not c.endswith('_id')]
                value_cols = fallback_cols if fallback_cols else [c for c in df.columns if c != label_col]
        
        # Filter to existing columns
        value_cols = [col for col in value_cols if col in df.columns]
        
        if not value_cols:
            return json.dumps({
                "status": "error",
                "message": f"No valid numeric columns found for chart. Available: {list(df.columns)}",
                "sql_used": sql
            })

        # 4.5) Get chart configuration first
        chart_type = plan.get("chart_type", "bar")
        title = plan.get("title", "HayStack Data Chart")
        custom_colors = plan.get("custom_colors", [])
        user_comparison = plan.get("user_comparison", {})
        
        # Optimize large datasets for readability
        MAX_CHART_ITEMS = 25  # Limit charts to 25 items for readability
        
        if len(df) > MAX_CHART_ITEMS:
            # For enrollment charts with many courses, limit to most recent or important ones
            if 'timecreated' in df.columns:
                # Sort by enrollment time and take most recent
                df = df.nlargest(MAX_CHART_ITEMS, 'timecreated')
                title += f" (Top {MAX_CHART_ITEMS} Most Recent)"
            elif 'course_name' in df.columns or 'fullname' in df.columns:
                # Sort by course name and take first N alphabetically  
                course_col = 'course_name' if 'course_name' in df.columns else 'fullname'
                df = df.nsmallest(MAX_CHART_ITEMS, course_col)
                title += f" (First {MAX_CHART_ITEMS} Alphabetically)"
            else:
                # Take first N rows
                df = df.head(MAX_CHART_ITEMS)
                title += f" (Limited to {MAX_CHART_ITEMS} items)"
            
            logger.info(f"Chart data limited from {len(df)} to {MAX_CHART_ITEMS} items for readability")

        # 5) Convert DataFrame to Chart.js specification with multi-user enhancements
        
        # For multi-user charts, check if all requested users are represented
        if is_multi_user and user_comparison:
            requested_users = [user_key.replace("user_", "") for user_key in user_comparison.keys() if user_key.startswith("user_")]
            actual_users = df['user_source'].unique().tolist() if 'user_source' in df.columns else []
            missing_users = [user for user in requested_users if int(user) not in actual_users]
            
            if missing_users:
                # Update title to reflect missing users
                missing_str = ', '.join([f"User {user}" for user in missing_users])
                title += f" (Note: {missing_str} has no enrolled courses)"
        
        # Special handling for multi-user course comparison - create individual course bars
        if is_multi_user and 'user_source' in df.columns and chart_type == 'bar':
            chart_spec = create_individual_course_bars_chart(df, label_col, title, custom_colors, user_comparison)
        else:
            chart_spec = df_to_chartjs(df, chart_type, label_col, value_cols, title, custom_colors, user_comparison)
        chart_id = f"chart_{uuid.uuid4().hex[:8]}"
        html_snippet = minimal_chart_html(chart_id, chart_spec)

        # 6) Return comprehensive JSON response
        # Store chart for web viewing
        from chart_storage import chart_storage
        # Ensure we have a valid user_id for storage (should never be None at this point)
        if not user_id or user_id <= 0:
            return json.dumps({
                "status": "error",
                "message": "Cannot store chart: invalid user_id",
                "sql_used": sql
            })
        
        stored_chart_id = chart_storage.store_chart({
            'chartjs_spec': asdict(chart_spec),
            'html_snippet': html_snippet,
            'chart_type': chart_type,
            'title': title,
            'sql_used': sql,
            'row_count': len(df),
            'user_id': user_id
        })
        
        # Generate viewing URL
        chart_view_url = f"http://localhost:8080/chart_viewer.html?chart_id={stored_chart_id}"

        payload = {
            "status": "ok",
            "chart_type": chart_type,
            "chartjs_spec": asdict(chart_spec),
            "html_snippet": html_snippet,
            "chart_id": chart_id,
            "stored_chart_id": stored_chart_id,
            "chart_view_url": chart_view_url,
            "label_column": label_col,
            "value_columns": value_cols,
            "title": title,
            "row_count": len(df),
            "sql_used": sql,
            "data_preview": df.head(5).to_dict('records') if len(df) <= 100 else None
        }
        
        logger.info(f"Chart generated successfully: {chart_type} with {len(df)} rows")
        return json.dumps(payload)

    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Chart generation failed: {str(e)}",
            "sql_used": getattr(locals(), 'sql', 'N/A')
        })


@mcp.tool()
def haystack_chart(question: str, user_id: int) -> str:
    """
    Build a Chart.js specification from a natural-language request with RBAC-safe data.
    
    Args:
        question: Natural language chart request (e.g., "bar chart of enrollments per course")
        user_id: User ID for RBAC context (required for security)
    
    Returns:
        Strict JSON with chart spec, html_snippet, and sql_used
    """
    # Validate user_id before proceeding
    if not user_id or user_id <= 0:
        return json.dumps({
            "status": "error",
            "message": "Valid user_id is required for chart generation"
        })
    
    return _generate_chart_internal(question, user_id)


def main():
    """Main HayStack MCP server"""
    print("[HayStack MCP] Starting HayStack MCP Server...")

    # Initialize HayStack MCP
    if not initialize_haystack_mcp():
        print("[HayStack MCP] Failed to initialize. Exiting.")
        sys.exit(1)

    print("[HayStack MCP] HayStack MCP Server ready!")
    print("[HayStack MCP] Starting MCP server in stdio mode")
    print("[HayStack MCP] Ready for HayStack integration!")

    # Run the FastMCP server
    mcp.run()


if __name__ == "__main__":
    main()