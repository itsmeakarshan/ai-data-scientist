"""
AutoDS Analysis Run API Endpoints
Initiates and monitors autonomous Data Science workflow executions.
"""

from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.agents.stage_tracker import get_stage_status, start_stage_tracking
from backend.app.agents.workflows import run_autonomous_datascience_pipeline
from backend.app.core.database import get_db
from backend.app.core.logging import logger
from backend.app.models.entities import AnalysisRun, Dataset
from backend.app.schemas.domain import (
    AnalysisCreateRequest,
    AnalysisRunResponse,
    AnalysisStatusResponse,
    WorkflowProgressResponse,
)


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

    # Initialize live stage tracker
    start_stage_tracking(analysis_run.id)

    # Schedule background autonomous pipeline execution
    background_tasks.add_task(
        run_autonomous_datascience_pipeline,
        analysis_id=analysis_run.id,
        dataset_id=req.dataset_id,
        user_goal=req.user_goal,
        target_column_override=req.target_column,
        time_column_override=req.time_column,
        problem_type_override=req.problem_type
    )

    return analysis_run


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


@router.get("/{analysis_id}/progress", response_model=WorkflowProgressResponse)
async def get_analysis_progress(
    analysis_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve structured real-time backend-driven workflow progress for a specific analysis run."""
    progress = get_stage_status(analysis_id)
    if not progress:
        res = await db.execute(select(AnalysisRun).filter(AnalysisRun.id == analysis_id))
        run = res.scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis run not found.")
        progress = get_stage_status(analysis_id)

    if not progress:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Progress tracking unavailable.")

    return progress


@router.get("/{analysis_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(
    analysis_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve real-time deterministic stage progression, elapsed time, and sub-stage model progress."""
    live_status = get_stage_status(analysis_id)
    if live_status:
        completed_stages_list = [s["number"] for s in live_status.get("stages", []) if s.get("status") == "COMPLETED"]
        return AnalysisStatusResponse(
            analysis_id=live_status["analysis_id"],
            status=live_status["status"],
            current_stage=live_status["current_stage_number"],
            current_stage_name=live_status["current_stage"],
            completed_stages=completed_stages_list,
            total_stages=live_status.get("total_stages", 9),
            progress_percent=int(live_status["progress_percentage"]),
            elapsed_seconds=live_status["elapsed_seconds"],
            models_evaluated=live_status.get("models_evaluated", []),
            current_model=live_status.get("current_model"),
            stage_details=live_status.get("stage_details"),
            error=live_status.get("error_message")
        )

    # Fallback to database record for historical runs
    res = await db.execute(select(AnalysisRun).filter(AnalysisRun.id == analysis_id))
    run = res.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis run not found.")

    if run.status == "COMPLETED":
        elapsed = int((run.completed_at - run.created_at).total_seconds()) if run.completed_at else 0
        return AnalysisStatusResponse(
            analysis_id=run.id,
            status="COMPLETED",
            current_stage=9,
            current_stage_name="Evidence-Backed Report Synthesis",
            completed_stages=list(range(1, 10)),
            total_stages=9,
            progress_percent=100,
            elapsed_seconds=max(elapsed, 0),
            models_evaluated=[],
            current_model=None,
            stage_details="Autonomous analysis pipeline completed successfully.",
            error=None
        )
    elif run.status == "FAILED":
        return AnalysisStatusResponse(
            analysis_id=run.id,
            status="FAILED",
            current_stage=1,
            current_stage_name="Dataset Inspection & Profiling",
            completed_stages=[],
            total_stages=9,
            progress_percent=0,
            elapsed_seconds=0,
            models_evaluated=[],
            current_model=None,
            stage_details="Analysis run failed.",
            error=run.error_message or "Execution failure."
        )
    else:
        return AnalysisStatusResponse(
            analysis_id=run.id,
            status=run.status,
            current_stage=1,
            current_stage_name="Dataset Inspection & Profiling",
            completed_stages=[],
            total_stages=9,
            progress_percent=0,
            elapsed_seconds=0,
            models_evaluated=[],
            current_model=None,
            stage_details="Analysis run is in progress.",
            error=None
        )


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

