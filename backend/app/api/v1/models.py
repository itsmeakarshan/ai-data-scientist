"""
AutoDS Model Registry API Endpoints
Manages trained champion models, SHAP interpretability values, and feature importances.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.core.database import get_db
from backend.app.models.entities import ModelRecord
from backend.app.schemas.domain import ModelRecordResponse


router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=List[ModelRecordResponse])
async def list_models(
    is_best: Optional[bool] = Query(None),
    task_type: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List trained models with optional filtering by champion status or task type."""
    query = select(ModelRecord).order_by(ModelRecord.created_at.desc())
    if is_best is not None:
        query = query.filter(ModelRecord.is_best == is_best)
    if task_type:
        query = query.filter(ModelRecord.task_type == task_type)

    query = query.offset(skip).limit(limit)
    res = await db.execute(query)
    return list(res.scalars().all())


@router.get("/{model_id}", response_model=ModelRecordResponse)
async def get_model(
    model_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get model metadata, SHAP summary attributions, and feature importance rankings."""
    res = await db.execute(select(ModelRecord).filter(ModelRecord.id == model_id))
    model_rec = res.scalar_one_or_none()
    if not model_rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model record not found.")
    return model_rec
