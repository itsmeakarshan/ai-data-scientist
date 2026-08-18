"""
AutoDS Model Registry API Endpoints
Manages trained champion models, SHAP interpretability values, and feature importances.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.core.database import get_db
from backend.app.models.entities import Experiment, ModelRecord
from backend.app.schemas.domain import ModelRecordResponse

router = APIRouter(prefix="/models", tags=["Models"])


def extract_model_metric_and_score(task_type: str, metrics_json: dict) -> tuple[str, float]:
    """Extract standard evaluation metric name and normalized score in [0.0, 1.0] range."""
    test_metrics = metrics_json.get("test", {}) if metrics_json else {}
    task = (task_type or "classification").lower()

    if task == "classification":
        roc = test_metrics.get("roc_auc")
        if roc is not None and isinstance(roc, (int, float)) and 0.0 <= float(roc) <= 1.0:
            return "Holdout ROC-AUC", round(float(roc), 4)
        bacc = test_metrics.get("balanced_accuracy")
        if bacc is not None and isinstance(bacc, (int, float)) and 0.0 <= float(bacc) <= 1.0:
            return "Holdout Balanced Accuracy", round(float(bacc), 4)
        acc = test_metrics.get("accuracy")
        if acc is not None and isinstance(acc, (int, float)) and 0.0 <= float(acc) <= 1.0:
            return "Holdout Accuracy", round(float(acc), 4)
        pr = test_metrics.get("pr_auc")
        if pr is not None and isinstance(pr, (int, float)) and 0.0 <= float(pr) <= 1.0:
            return "Holdout PR-AUC", round(float(pr), 4)
        f1 = test_metrics.get("f1_positive") or test_metrics.get("f1_macro") or test_metrics.get("f1")
        if f1 is not None and isinstance(f1, (int, float)) and 0.0 <= float(f1) <= 1.0:
            return "Holdout F1-Score", round(float(f1), 4)
        return "Holdout Score", 0.50

    # Regression / Forecasting
    r2 = test_metrics.get("r2")
    if r2 is not None and isinstance(r2, (int, float)):
        clamped_r2 = max(0.0, min(float(r2), 1.0))
        return "Holdout R²", round(clamped_r2, 4)

    wape = test_metrics.get("wape")
    if wape is not None and isinstance(wape, (int, float)):
        # Normalize WAPE (whether stored as decimal fraction 0.1813 or percentage 18.13) into [0, 1] accuracy
        wape_frac = float(wape) / 100.0 if float(wape) > 1.0 else float(wape)
        acc_score = max(0.0, min(1.0 - wape_frac, 1.0))
        return "Holdout Accuracy (1-WAPE)", round(acc_score, 4)

    return "Holdout Score", 0.50


def _build_model_response(model_rec: ModelRecord) -> ModelRecordResponse:
    """Helper to convert ModelRecord to ModelRecordResponse with populated metadata."""
    dataset_name = None
    dataset_id = None
    analysis_id = None

    if model_rec.experiment:
        dataset_id = model_rec.experiment.dataset_id
        analysis_id = model_rec.experiment.analysis_id
        if model_rec.experiment.dataset:
            dataset_name = model_rec.experiment.dataset.name

    metric_name, normalized_score = extract_model_metric_and_score(
        model_rec.task_type,
        model_rec.metrics_json
    )

    return ModelRecordResponse(
        id=model_rec.id,
        experiment_id=model_rec.experiment_id,
        name=model_rec.name,
        task_type=model_rec.task_type,
        is_best=model_rec.is_best,
        artifact_path=model_rec.artifact_path,
        feature_importance_json=model_rec.feature_importance_json or {},
        shap_summary_json=model_rec.shap_summary_json or {},
        metrics_json=model_rec.metrics_json or {},
        created_at=model_rec.created_at,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        analysis_id=analysis_id,
        metric_name=metric_name,
        normalized_score=normalized_score,
    )


@router.get("", response_model=List[ModelRecordResponse])
async def list_models(
    is_best: Optional[bool] = Query(None),
    task_type: Optional[str] = Query(None),
    latest_per_dataset: bool = Query(False),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List trained models with optional filtering by champion status or task type."""
    query = (
        select(ModelRecord)
        .options(
            selectinload(ModelRecord.experiment).selectinload(Experiment.dataset),
            selectinload(ModelRecord.experiment).selectinload(Experiment.analysis_run),
        )
        .order_by(ModelRecord.created_at.desc())
    )
    if is_best is not None:
        query = query.filter(ModelRecord.is_best == is_best)
    if task_type:
        query = query.filter(ModelRecord.task_type == task_type)

    if latest_per_dataset:
        res = await db.execute(query)
        all_models = list(res.scalars().all())
        seen_datasets = set()
        deduped = []
        for m in all_models:
            ds_key = (
                (m.experiment.dataset_id if m.experiment else None)
                or (m.experiment.dataset.name if m.experiment and m.experiment.dataset else None)
                or m.name
            )
            if ds_key not in seen_datasets:
                seen_datasets.add(ds_key)
                deduped.append(m)
        sliced = deduped[skip : skip + limit]
        return [_build_model_response(m) for m in sliced]

    query = query.offset(skip).limit(limit)
    res = await db.execute(query)
    models = list(res.scalars().all())
    return [_build_model_response(m) for m in models]


@router.get("/{model_id}", response_model=ModelRecordResponse)
async def get_model(
    model_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get model metadata, SHAP summary attributions, and feature importance rankings."""
    query = (
        select(ModelRecord)
        .options(
            selectinload(ModelRecord.experiment).selectinload(Experiment.dataset),
            selectinload(ModelRecord.experiment).selectinload(Experiment.analysis_run),
        )
        .filter(ModelRecord.id == model_id)
    )
    res = await db.execute(query)
    model_rec = res.scalar_one_or_none()
    if not model_rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model record not found.")
    return _build_model_response(model_rec)
