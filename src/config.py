"""
Enhanced Configuration System for HayStack LMS
Comprehensive configuration management with environment support and validation
"""

import os
from pathlib import Path
from typing import Dict, Any
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded environment from {env_path}")
else:
    logger.warning(f"No .env file found at {env_path}")

# Database Configuration - Updated for Azure MySQL Totara LMS
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'hl-uaen-mysql-flex-01.mysql.database.azure.com'),
    'user': os.getenv('DB_USER', 'dev_aiadgmuser'),
    'password': os.getenv('DB_PASSWORD', 'afdj%892jHgg'),
    'database': os.getenv('DB_NAME', 'dev_aiadgm'),
    'port': int(os.getenv('DB_PORT', 3306))
}

# JWT Configuration
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'haystack-secret-key-2024')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

# ChromaDB Configuration
CHROMA_CONFIG = {
    'host': os.getenv('CHROMA_HOST', 'localhost'),
    'port': int(os.getenv('CHROMA_PORT', 8000)),
    'collection_name': os.getenv('CHROMA_COLLECTION', 'haystack_conversations'),
    'persist_directory': os.getenv('CHROMADB_PERSIST_DIR', './chromadb_data')
}

# MCP Configuration
MCP_CONFIG = {
    'host': os.getenv('MCP_HOST', 'localhost'),
    'port': int(os.getenv('MCP_PORT', 3001)),
    'server_name': os.getenv('MCP_SERVER_NAME', 'totara-lms-chatbot'),
    'version': os.getenv('MCP_SERVER_VERSION', '1.0.0')
}

# Ollama Configuration
OLLAMA_CONFIG = {
    'base_url': os.getenv('OLLAMA_URL', 'http://localhost:11434'),
    'model': os.getenv('OLLAMA_MODEL', 'qwen2.5:7b-instruct'),
    'timeout': int(os.getenv('OLLAMA_TIMEOUT_SECONDS', 120))
}

# LLM Configuration
LLM_CONFIG = {
    'model': os.getenv('LLM_MODEL', 'qwen2.5:7b-instruct'),
    'qwen_model': os.getenv('QWEN_MODEL', 'qwen2.5:7b-instruct'),
    'ollama_url': os.getenv('OLLAMA_URL', 'http://localhost:11434'),
    'timeout_seconds': int(os.getenv('OLLAMA_TIMEOUT_SECONDS', 120))
}

# SQL Execution Configuration
SQL_CONFIG = {
    'max_rows_default': int(os.getenv('SQL_MAX_ROWS_DEFAULT', 200)),
    'max_exec_ms': int(os.getenv('SQL_MAX_EXEC_MS', 25000))
}

# Performance & Caching Configuration
PERFORMANCE_CONFIG = {
    'cache_max_size': int(os.getenv('CACHE_MAX_SIZE', 100)),
    'cache_ttl_seconds': int(os.getenv('CACHE_TTL_SECONDS', 300)),
    'agent_timeout_seconds': int(os.getenv('AGENT_TIMEOUT_SECONDS', 180))
}

# Session Management Configuration
SESSION_CONFIG = {
    'timeout_minutes': int(os.getenv('SESSION_TIMEOUT_MINUTES', 60)),
    'cleanup_interval_minutes': int(os.getenv('CLEANUP_INTERVAL_MINUTES', 10))
}

# Security Configuration
SECURITY_CONFIG = {
    'max_query_length': int(os.getenv('MAX_QUERY_LENGTH', 1000)),
    'max_conversation_history': int(os.getenv('MAX_CONVERSATION_HISTORY', 10))
}

# Azure Search Configuration
AZURE_SEARCH_CONFIG = {
    'endpoint': os.getenv('AZURE_SEARCH_ENDPOINT', ''),
    'admin_key': os.getenv('AZURE_SEARCH_ADMIN_KEY', '')
}

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

def load_env_config():
    """Load configuration from environment variables with backward compatibility"""
    return {
        # Database Configuration (backward compatible)
        'db_host': os.getenv('DB_HOST', 'localhost'),
        'db_port': int(os.getenv('DB_PORT', '3306')),
        'db_name': os.getenv('DB_NAME') or os.getenv('DB_DATABASE', 'haystack_db'),
        'db_username': os.getenv('DB_USER') or os.getenv('DB_USERNAME', 'root'),
        'db_password': os.getenv('DB_PASSWORD', ''),
        
        # AI/LLM Configuration
        'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY', ''),
        'openai_model': os.getenv('OPENAI_MODEL', 'gpt-4'),
        'anthropic_model': os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'),
        
        # Vector Store Configuration
        'vector_store_path': os.getenv('VECTOR_STORE_PATH', './chromadb_data'),
        'collection_name': os.getenv('COLLECTION_NAME', 'ultimate_haystack'),
        
        # Application Configuration
        'app_host': os.getenv('APP_HOST', '0.0.0.0'),
        'app_port': int(os.getenv('APP_PORT', '8000')),
        'debug_mode': os.getenv('DEBUG_MODE', 'true').lower() == 'true',
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        
        # Security Configuration
        'api_key': os.getenv('API_KEY', ''),
        'jwt_secret': os.getenv('JWT_SECRET', 'your-secret-key'),
        'cors_origins': os.getenv('CORS_ORIGINS', '*').split(','),
        
        # Performance Configuration
        'max_concurrent_requests': int(os.getenv('MAX_CONCURRENT_REQUESTS', '10')),
        'request_timeout': int(os.getenv('REQUEST_TIMEOUT', '300')),
        'cache_ttl': int(os.getenv('CACHE_TTL', '3600')),
        
        # AI Agent Configuration
        'max_reasoning_steps': int(os.getenv('MAX_REASONING_STEPS', '5')),
        'memory_limit': int(os.getenv('MEMORY_LIMIT', '1000')),
        
        # Analytics Configuration
        'enable_analytics': os.getenv('ENABLE_ANALYTICS', 'false').lower() == 'true',
        'analytics_endpoint': os.getenv('ANALYTICS_ENDPOINT', ''),
        
        # Notification Configuration
        'email_enabled': os.getenv('EMAIL_ENABLED', 'false').lower() == 'true',
        'smtp_host': os.getenv('SMTP_HOST', ''),
        'smtp_port': int(os.getenv('SMTP_PORT', '587')),
        'smtp_username': os.getenv('SMTP_USERNAME', ''),
        'smtp_password': os.getenv('SMTP_PASSWORD', ''),
        
        # Advanced Features
        'enable_experimental_features': os.getenv('ENABLE_EXPERIMENTAL_FEATURES', 'false').lower() == 'true',
        'feature_flags': os.getenv('FEATURE_FLAGS', '').split(',') if os.getenv('FEATURE_FLAGS') else [],
    }

# Export commonly used configurations
__all__ = [
    'DB_CONFIG',
    'JWT_SECRET_KEY', 'JWT_ALGORITHM', 'JWT_EXPIRATION_HOURS',
    'CHROMA_CONFIG',
    'MCP_CONFIG',
    'OLLAMA_CONFIG',
    'LLM_CONFIG',
    'SQL_CONFIG',
    'PERFORMANCE_CONFIG',
    'SESSION_CONFIG',
    'SECURITY_CONFIG',
    'AZURE_SEARCH_CONFIG',
    'LOG_LEVEL',
    'load_env_config'
]