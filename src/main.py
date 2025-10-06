"""
Main Application Entry Point for HayStack Multi-Tenant LMS
Provides unified interface for all HayStack functionality
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import our multi-tenant components
from .database_connector_multitenant import MultiTenantDatabaseConnector, DBConfig
from .universal_rbac import UniversalRBACService
from .tenant_aware_mcp_tools import TenantAwareMCPTools
from .haystack_pipeline import HayStackPipeline
from .universal_haystack_pipeline import UniversalHayStackPipeline

logger = logging.getLogger(__name__)

@dataclass
class HayStackConfig:
    """Configuration for HayStack application"""
    # Database settings
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = "root" 
    db_password: str = ""
    db_name: str = "totara_db"
    
    # Multi-tenant settings
    default_tenant: str = "main_tenant"
    
    # Security settings
    enable_rbac: bool = True
    admin_users: list = None
    
    # Service settings
    mcp_server_port: int = 8000
    enable_logging: bool = True

class HayStackApplication:
    """Main application class orchestrating all HayStack functionality"""
    
    def __init__(self, config: HayStackConfig):
        self.config = config
        self.db_connector = None
        self.rbac_service = None
        self.mcp_tools = None
        self.pipeline = None
        
        if config.enable_logging:
            self._setup_logging()
    
    def _setup_logging(self):
        """Configure application logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('haystack_app.log')
            ]
        )
    
    async def initialize(self) -> bool:
        """Initialize all application components"""
        try:
            logger.info("🚀 Initializing HayStack Multi-Tenant Application")
            
            # Initialize database connector
            db_config = DBConfig(
                host=self.config.db_host,
                port=self.config.db_port,
                username=self.config.db_user,
                password=self.config.db_password,
                database=self.config.db_name,
                tenant_key=self.config.default_tenant
            )
            
            self.db_connector = MultiTenantDatabaseConnector()
            await self.db_connector.add_tenant_config(self.config.default_tenant, db_config)
            logger.info("✅ Database connector initialized")
            
            # Initialize RBAC service
            if self.config.enable_rbac:
                self.rbac_service = UniversalRBACService(self.db_connector)
                if self.config.admin_users:
                    for tenant in [self.config.default_tenant]:
                        self.rbac_service.set_tenant_config(
                            tenant, 
                            admin_users=self.config.admin_users
                        )
                logger.info("✅ RBAC service initialized")
            
            # Initialize MCP tools
            self.mcp_tools = TenantAwareMCPTools(
                self.db_connector,
                self.rbac_service,
                self.config.default_tenant
            )
            logger.info("✅ MCP tools initialized")
            
            # Initialize pipeline
            self.pipeline = UniversalHayStackPipeline(
                self.db_connector,
                self.rbac_service
            )
            logger.info("✅ Pipeline initialized")
            
            logger.info("🎉 HayStack application fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize application: {e}")
            return False
    
    async def process_query(self, query: str, user_id: str, tenant_key: str = None) -> Dict[str, Any]:
        """Process a user query through the pipeline"""
        try:
            if not tenant_key:
                tenant_key = self.config.default_tenant
            
            # Switch to tenant context
            await self.db_connector.switch_database(tenant_key)
            
            # Process through pipeline
            result = await self.pipeline.process_query(
                query=query,
                user_id=user_id,
                tenant_key=tenant_key
            )
            
            return {
                "success": True,
                "result": result,
                "tenant": tenant_key,
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tenant": tenant_key,
                "user_id": user_id
            }
    
    async def add_tenant(self, tenant_key: str, db_config: DBConfig) -> bool:
        """Add a new tenant to the system"""
        try:
            await self.db_connector.add_tenant_config(tenant_key, db_config)
            
            if self.rbac_service:
                self.rbac_service.set_tenant_config(
                    tenant_key,
                    admin_users=self.config.admin_users or []
                )
            
            logger.info(f"✅ Tenant {tenant_key} added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add tenant {tenant_key}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get application status"""
        return {
            "initialized": all([
                self.db_connector is not None,
                self.mcp_tools is not None,
                self.pipeline is not None
            ]),
            "tenants": list(self.db_connector.tenant_configs.keys()) if self.db_connector else [],
            "rbac_enabled": self.rbac_service is not None,
            "config": {
                "default_tenant": self.config.default_tenant,
                "mcp_port": self.config.mcp_server_port,
                "rbac_enabled": self.config.enable_rbac
            }
        }

# Factory function for easy application creation
def create_haystack_app(
    db_host: str = "localhost",
    db_name: str = "totara_db", 
    db_user: str = "root",
    db_password: str = "",
    tenant_key: str = "main_tenant",
    admin_users: list = None
) -> HayStackApplication:
    """Factory function to create HayStack application with common settings"""
    
    config = HayStackConfig(
        db_host=db_host,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        default_tenant=tenant_key,
        admin_users=admin_users or ["admin", "manager"]
    )
    
    return HayStackApplication(config)

# CLI entry point
async def main():
    """Main entry point for CLI usage"""
    import asyncio
    
    # Create app with environment variables
    app = create_haystack_app(
        db_host=os.getenv("DB_HOST", "localhost"),
        db_name=os.getenv("DB_NAME", "totara_db"),
        db_user=os.getenv("DB_USER", "root"),
        db_password=os.getenv("DB_PASSWORD", ""),
        tenant_key=os.getenv("TENANT_KEY", "main_tenant")
    )
    
    # Initialize
    success = await app.initialize()
    if not success:
        logger.error("Failed to initialize application")
        return
    
    # Example usage
    logger.info("🎯 HayStack application ready for queries")
    
    # Print status
    status = app.get_status()
    logger.info(f"📊 Application Status: {status}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())