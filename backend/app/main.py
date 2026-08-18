"""
AutoDS — Autonomous Data Science & Machine Learning Platform
FastAPI Application Entrypoint
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.v1 import api_v1_router
from backend.app.core.config import settings
from backend.app.core.database import init_db
from backend.app.core.logging import logger
from backend.app.tools.ml_trainer import initialize_mlflow


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("Initializing AutoDS Database and Storage Directories...")
    await init_db()
    initialize_mlflow()
    settings.data_raw_dir
    settings.data_processed_dir
    settings.reports_artifacts_dir
    settings.experiments_artifacts_dir
    logger.info(f"AutoDS {settings.VERSION} backend startup complete (Environment: {settings.ENVIRONMENT})")
    yield
    logger.info("AutoDS backend shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Autonomous Data Science platform combining Gemini agents, deterministic ML pipelines, critic auditing, and reproducible experiments.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS Configuration
origins = settings.BACKEND_CORS_ORIGINS if isinstance(settings.BACKEND_CORS_ORIGINS, list) else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Artifacts for UI Visualizations
reports_artifacts_path = settings.reports_artifacts_dir
app.mount(
    "/reports/artifacts",
    StaticFiles(directory=str(reports_artifacts_path)),
    name="reports_artifacts"
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

# Include API Routers (supporting /api/ and /api/v1/ aliases)
app.include_router(api_v1_router, prefix="/api")
app.include_router(api_v1_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
