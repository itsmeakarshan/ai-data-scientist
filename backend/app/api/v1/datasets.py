"""
AutoDS Datasets API Endpoints
Handles dataset upload, schema inspection, profiling, and sampling.
"""

from pathlib import Path
import shutil
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.core.logging import logger
from backend.app.core.security import ALLOWED_EXTENSIONS, sanitize_filename
from backend.app.models.entities import Dataset, DatasetProfile
from backend.app.schemas.domain import (
    DatasetListResponse,
    DatasetProfileResponse,
    DatasetResponse,
    SampleRowsResponse,
)
from backend.app.tools.data_profiler import profile_dataset
from backend.app.tools.dataset_inspector import load_dataset_as_dataframe
from backend.app.tools.quality_detector import detect_data_quality


router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post("/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Upload, inspect, profile, and register a new dataset."""
    filename = file.filename or "dataset.csv"
    sanitized_name = sanitize_filename(filename)
    file_ext = Path(sanitized_name).suffix.lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{file_ext}'. Allowed formats: CSV, Parquet, Excel, JSON."
        )

    # Save to data/raw/
    raw_dir = settings.data_raw_dir
    unique_name = f"{uuid.uuid4().hex[:8]}_{sanitized_name}"
    dest_path = raw_dir / unique_name

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {e}"
        )

    # Inspect & Load
    try:
        df, meta = load_dataset_as_dataframe(dest_path)
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse dataset: {e}"
        )

    # Compute Profile & Quality Alerts
    try:
        profile_data = profile_dataset(df)
        quality_alerts = detect_data_quality(df, profile_data)
        profile_data["quality_alerts"] = quality_alerts
    except Exception as e:
        logger.warning(f"Profiling warning: {e}")
        profile_data = {
            "summary_stats": {},
            "missingness_report": {},
            "column_types": {},
            "correlations": {},
            "quality_alerts": [],
            "candidate_targets": [],
            "candidate_datetimes": [],
            "inferred_problem_type": "classification",
        }

    # Save to DB
    dataset = Dataset(
        name=sanitized_name,
        file_path=str(dest_path),
        file_type=meta.get("file_type", file_ext.replace(".", "")),
        size_bytes=meta.get("file_size_bytes", dest_path.stat().st_size),
        row_count=meta.get("row_count", len(df)),
        col_count=meta.get("col_count", len(df.columns)),
        checksum=meta.get("checksum", ""),
    )
    db.add(dataset)
    await db.flush()

    profile_obj = DatasetProfile(
        dataset_id=dataset.id,
        summary_stats=profile_data.get("summary_stats", {}),
        missingness_report=profile_data.get("missingness_report", {}),
        column_types=profile_data.get("column_types", {}),
        correlations=profile_data.get("correlations", {}),
        quality_alerts=profile_data.get("quality_alerts", []),
        candidate_targets=profile_data.get("candidate_targets", []),
        candidate_datetimes=profile_data.get("candidate_datetimes", []),
        inferred_problem_type=profile_data.get("inferred_problem_type"),
    )
    db.add(profile_obj)
    await db.commit()

    # Re-fetch with relationship
    res = await db.execute(
        select(Dataset).options(selectinload(Dataset.profile)).filter(Dataset.id == dataset.id)
    )
    saved_ds = res.scalar_one()
    return saved_ds


@router.get("", response_model=DatasetListResponse)
async def list_datasets(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """Retrieve all registered datasets with summary counts."""
    query = select(Dataset).options(selectinload(Dataset.profile)).order_by(Dataset.created_at.desc()).offset(skip).limit(limit)
    res = await db.execute(query)
    datasets = res.scalars().all()
    return DatasetListResponse(items=list(datasets), total=len(datasets))


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get dataset details, schema, statistical profile, and quality alerts."""
    res = await db.execute(
        select(Dataset).options(selectinload(Dataset.profile)).filter(Dataset.id == dataset_id)
    )
    dataset = res.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")
    return dataset


@router.get("/{dataset_id}/sample", response_model=SampleRowsResponse)
async def get_dataset_sample(
    dataset_id: str,
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    """Fetch sample rows for tabular preview."""
    res = await db.execute(select(Dataset).filter(Dataset.id == dataset_id))
    dataset = res.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")

    try:
        df, _ = load_dataset_as_dataframe(dataset.file_path, sample_rows=limit)
        clean_df = df.fillna("").head(limit)
        return SampleRowsResponse(
            columns=list(clean_df.columns),
            rows=clean_df.to_dict(orient="records"),
            total_rows=dataset.row_count,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read dataset sample: {e}"
        )
