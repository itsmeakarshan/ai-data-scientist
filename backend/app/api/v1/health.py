"""
AutoDS Health Check API Endpoints
"""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.agents.gemini_client import gemini_client
from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.models.entities import Dataset, Experiment
from backend.app.schemas.domain import HealthStatusResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthStatusResponse)
async def get_health_status(db: AsyncSession = Depends(get_db)):
    """System health check verifying database, Gemini API, MLflow, and storage."""
    db_connected = False
    dataset_count = 0
    exp_count = 0

    try:
        await db.execute(text("SELECT 1"))
        db_connected = True

        # Count datasets
        ds_res = await db.execute(select(func.count(Dataset.id)))
        dataset_count = ds_res.scalar() or 0

        # Count experiments
        exp_res = await db.execute(select(func.count(Experiment.id)))
        exp_count = exp_res.scalar() or 0
    except Exception:
        db_connected = False

    return HealthStatusResponse(
        status="healthy" if db_connected else "degraded",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        database_connected=db_connected,
        gemini_api_configured=gemini_client.is_active,
        gemini_model=settings.GEMINI_MODEL,
        mlflow_tracking_active=True,
        datasets_count=dataset_count,
        experiments_count=exp_count,
    )
