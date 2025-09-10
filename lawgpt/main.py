"""
Main FastAPI application for LawGPT
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lawgpt.api.endpoint.chat import router as chat_router
from lawgpt.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting LawGPT application...")
    
    # Initialize session workflows storage
    
    logger.info("System initialized with session-based memory management")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LawGPT application...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="LawGPT API",
        description="Legal AI Assistant with RAG capabilities for Bangladesh Law",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
    
    @app.get("/")
    async def root():
        return {"message": "LawGPT API is running", "version": "1.0.0"}
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "lawgpt.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
