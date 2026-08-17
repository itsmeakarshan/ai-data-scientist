"""
AutoDS Analysis Run API Endpoints
Initiates and monitors autonomous Data Science workflow executions.
"""

from typing import List
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.agents.workflows import run_autonomous_datascience_pipeline
from backend.app.core.database import get_db
from backend.app.core.logging import logger
from backend.app.models.entities import AnalysisRun, Dataset
from backend.app.schemas.domain import AnalysisCreateRequest, AnalysisRunResponse


router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("", response_model=AnalysisRunResponse, status_code=status.HTTP_201_CREATED)
async def create_analysis_run(
    req: AnalysisCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Trigger autonomous Data Science pipeline on a registered dataset."""
    res = await db.execute(select(Dataset).filter(Dataset.id == req.dataset_id))
    dataset = res.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target dataset not found.")

    analysis_run = AnalysisRun(
        dataset_id=req.dataset_id,
        user_goal=req.user_goal,
        status="RUNNING",
        problem_type=req.problem_type or "classification",
        target_column=req.target_column,
        time_column=req.time_column,
        validation_strategy=req.validation_strategy or "stratified_kfold",
    )
    db.add(analysis_run)
    await db.commit()
    await db.refresh(analysis_run)

    # Execute synchronous or background pipeline
    try:
        run_autonomous_datascience_pipeline(
            analysis_id=analysis_run.id,
            dataset_id=req.dataset_id,
            user_goal=req.user_goal,
            target_column_override=req.target_column,
            time_column_override=req.time_column,
            problem_type_override=req.problem_type
        )
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}")

    # Re-fetch updated run
    updated_res = await db.execute(select(AnalysisRun).filter(AnalysisRun.id == analysis_run.id))
    return updated_res.scalar_one()


@router.get("", response_model=List[AnalysisRunResponse])
async def list_analysis_runs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List all autonomous analysis runs ordered by most recent."""
    res = await db.execute(
        select(AnalysisRun).order_by(AnalysisRun.created_at.desc()).offset(skip).limit(limit)
    )
    return list(res.scalars().all())


@router.get("/{analysis_id}", response_model=AnalysisRunResponse)
async def get_analysis_run(
    analysis_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get the full status, plan, critic findings, and results of an analysis run."""
    res = await db.execute(select(AnalysisRun).filter(AnalysisRun.id == analysis_id))
    run = res.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis run not found.")
    return run
