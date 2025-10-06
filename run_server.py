"""
Production Runner for HayStack Multi-Tenant LMS
Complete application orchestrator with all services
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import HayStackApplication, HayStackConfig, create_haystack_app
from src.database_connector_multitenant import DBConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('haystack_production.log')
    ]
)
logger = logging.getLogger(__name__)

# Global application instance
app_instance: HayStackApplication = None

# FastAPI application
api = FastAPI(
    title="HayStack Multi-Tenant LMS API",
    description="Production API for HayStack multi-tenant LMS functionality",
    version="1.0.0"
)

# CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    user_id: str
    tenant_key: str = "main_tenant"

class QueryResponse(BaseModel):
    success: bool
    result: Any = None
    error: str = None
    tenant: str = None
    user_id: str = None

class TenantConfigRequest(BaseModel):
    tenant_key: str
    db_host: str
    db_port: int = 3306
    db_name: str
    db_user: str
    db_password: str

class StatusResponse(BaseModel):
    status: str
    details: Dict[str, Any]

# Dependency to get app instance
async def get_app() -> HayStackApplication:
    global app_instance
    if not app_instance:
        raise HTTPException(status_code=503, detail="Application not initialized")
    return app_instance

# API Routes
@api.get("/", response_model=Dict[str, str])
async def root():
    """API root endpoint"""
    return {
        "message": "HayStack Multi-Tenant LMS API",
        "version": "1.0.0",
        "status": "running"
    }

@api.get("/health", response_model=StatusResponse)
async def health_check(app: HayStackApplication = Depends(get_app)):
    """Health check endpoint"""
    try:
        status = app.get_status()
        return StatusResponse(
            status="healthy" if status["initialized"] else "initializing",
            details=status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return StatusResponse(
            status="unhealthy",
            details={"error": str(e)}
        )

@api.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    app: HayStackApplication = Depends(get_app)
):
    """Process a user query"""
    try:
        result = await app.process_query(
            query=request.query,
            user_id=request.user_id,
            tenant_key=request.tenant_key
        )
        
        return QueryResponse(
            success=result["success"],
            result=result.get("result"),
            error=result.get("error"),
            tenant=result.get("tenant"),
            user_id=result.get("user_id")
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return QueryResponse(
            success=False,
            error=str(e),
            tenant=request.tenant_key,
            user_id=request.user_id
        )

@api.post("/tenant/add")
async def add_tenant(
    request: TenantConfigRequest,
    app: HayStackApplication = Depends(get_app)
):
    """Add a new tenant configuration"""
    try:
        db_config = DBConfig(
            host=request.db_host,
            port=request.db_port,
            user=request.db_user,
            password=request.db_password,
            database=request.db_name
        )
        
        success = await app.add_tenant(request.tenant_key, db_config)
        
        return {
            "success": success,
            "message": f"Tenant {request.tenant_key} {'added' if success else 'failed to add'}"
        }
        
    except Exception as e:
        logger.error(f"Failed to add tenant: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@api.get("/tenants")
async def list_tenants(app: HayStackApplication = Depends(get_app)):
    """List all configured tenants"""
    try:
        status = app.get_status()
        return {
            "success": True,
            "tenants": status.get("tenants", []),
            "count": len(status.get("tenants", []))
        }
    except Exception as e:
        logger.error(f"Failed to list tenants: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@api.get("/status/detailed", response_model=StatusResponse)
async def detailed_status(app: HayStackApplication = Depends(get_app)):
    """Get detailed application status"""
    try:
        status = app.get_status()
        
        # Add additional runtime information
        status["runtime"] = {
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "environment": dict(os.environ)
        }
        
        return StatusResponse(
            status="operational",
            details=status
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return StatusResponse(
            status="error",
            details={"error": str(e)}
        )

# Application lifecycle
@api.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global app_instance
    
    logger.info("🚀 Starting HayStack Multi-Tenant LMS Production Server...")
    
    try:
        # Create application with environment configuration
        app_instance = create_haystack_app(
            db_host=os.getenv("DB_HOST", "localhost"),
            db_name=os.getenv("DB_NAME", "totara_db"),
            db_user=os.getenv("DB_USER", "root"),
            db_password=os.getenv("DB_PASSWORD", ""),
            tenant_key=os.getenv("TENANT_KEY", "main_tenant"),
            admin_users=os.getenv("ADMIN_USERS", "admin,manager").split(",")
        )
        
        # Initialize application
        success = await app_instance.initialize()
        if not success:
            logger.error("❌ Failed to initialize HayStack application")
            raise Exception("Application initialization failed")
        
        logger.info("✅ HayStack Multi-Tenant LMS started successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@api.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    global app_instance
    
    logger.info("🛑 Shutting down HayStack Multi-Tenant LMS...")
    
    try:
        if app_instance:
            # Add any cleanup logic here
            logger.info("✅ Application shutdown completed")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Main runner
def main():
    """Main entry point for production server"""
    
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    logger.info(f"🎯 Starting production server on {host}:{port}")
    logger.info(f"⚙️ Configuration: workers={workers}, log_level={log_level}")
    
    # Start server
    uvicorn.run(
        "run_server:api",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=True,
        reload=False  # Disable for production
    )

if __name__ == "__main__":
    main()