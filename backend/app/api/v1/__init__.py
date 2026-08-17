"""
AutoDS API v1 Router Aggregator
"""

from fastapi import APIRouter
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.datasets import router as datasets_router
from backend.app.api.v1.analysis import router as analysis_router
from backend.app.api.v1.experiments import router as experiments_router
from backend.app.api.v1.models import router as models_router
from backend.app.api.v1.reports import router as reports_router
from backend.app.api.v1.chat import router as chat_router
from backend.app.api.v1.query import router as query_router


api_v1_router = APIRouter()

api_v1_router.include_router(health_router)
api_v1_router.include_router(datasets_router)
api_v1_router.include_router(analysis_router)
api_v1_router.include_router(experiments_router)
api_v1_router.include_router(models_router)
api_v1_router.include_router(reports_router)
api_v1_router.include_router(chat_router)
api_v1_router.include_router(query_router)
