"""
AutoDS Experiments API Endpoints
Tracks and compares ML experiments, cross-validation metrics, and model configurations.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from backend.app.core.database import get_db
from backend.app.models.entities import AnalysisRun, Experiment
from backend.app.schemas.domain import (
    ExperimentComparisonResponse,
    ExperimentResponse,
)


router = APIRouter(prefix="/experiments", tags=["Experiments"])


@router.get("", response_model=List[ExperimentResponse])
async def list_experiments(
    analysis_id: Optional[str] = Query(None),
    dataset_id: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List tracked experiments with optional filtering."""
    query = select(Experiment).options(selectinload(Experiment.metrics)).order_by(Experiment.created_at.desc())
    if analysis_id:
        query = query.filter(Experiment.analysis_id == analysis_id)
    if dataset_id:
        query = query.filter(Experiment.dataset_id == dataset_id)

    query = query.offset(skip).limit(limit)
    res = await db.execute(query)
    return list(res.scalars().all())


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve full experiment details, hyperparameters, and evaluated metrics."""
    res = await db.execute(
        select(Experiment).options(selectinload(Experiment.metrics)).filter(Experiment.id == experiment_id)
    )
    exp = res.scalar_one_or_none()
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found.")
    return exp


@router.get("/compare/{analysis_id}", response_model=ExperimentComparisonResponse)
async def compare_experiments(
    analysis_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Compare all candidate models for a specific analysis run."""
    run_res = await db.execute(select(AnalysisRun).filter(AnalysisRun.id == analysis_id))
    run = run_res.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis run not found.")

    exp_res = await db.execute(
        select(Experiment).options(selectinload(Experiment.metrics)).filter(Experiment.analysis_id == analysis_id)
    )
    experiments = list(exp_res.scalars().all())
    
    # Determine primary comparison metric
    primary_metric = "roc_auc" if run.problem_type == "classification" else ("wape" if run.problem_type == "forecasting" else "rmse")

    table_rows = []
    best_id = None
    best_val = -1.0 if run.problem_type == "classification" else 999999.0

    for e in experiments:
        test_m = e.metrics_json.get("test", {})
        score = test_m.get(primary_metric, 0.0)
        
        row = {
            "experiment_id": e.id,
            "model_name": e.model_name,
            "model_family": e.model_family,
            "primary_metric": primary_metric,
            "score": score,
            "cv_mean": e.metrics_json.get("cv_mean", 0.0),
            "cv_std": e.metrics_json.get("cv_std", 0.0),
            "train_time_sec": e.train_time_sec,
            "metrics": test_m,
        }
        table_rows.append(row)

        if run.problem_type == "classification":
            if score > best_val:
                best_val = score
                best_id = e.id
        else:
            if score < best_val:
                best_val = score
                best_id = e.id

    return ExperimentComparisonResponse(
        experiments=experiments,
        best_experiment_id=best_id,
        primary_metric=primary_metric,
        comparison_table=table_rows,
    )
