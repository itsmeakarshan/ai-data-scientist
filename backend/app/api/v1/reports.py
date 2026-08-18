"""
AutoDS Reports API Endpoints
Serves evidence-backed Markdown and JSON reports.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.models.entities import Report
from backend.app.schemas.domain import ReportResponse

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("", response_model=List[ReportResponse])
async def list_reports(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """List all synthesized reports."""
    res = await db.execute(select(Report).order_by(Report.created_at.desc()).offset(skip).limit(limit))
    reports = list(res.scalars().all())
    # Exclude temporary test reports created during pytest runs
    valid_reports = [
        r for r in reports
        if not (r.artifact_paths and any("autods_test_reports" in str(p) for p in r.artifact_paths))
    ]
    return valid_reports


@router.get("/{identifier}", response_model=ReportResponse)
async def get_report(
    identifier: str,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve full report by Report ID or Analysis ID."""
    res = await db.execute(
        select(Report).filter(or_(Report.id == identifier, Report.analysis_id == identifier))
    )
    report = res.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found.")
    return report


@router.delete("/{identifier}", status_code=status.HTTP_200_OK)
async def delete_report(
    identifier: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a report by ID or Analysis ID."""
    res = await db.execute(
        select(Report).filter(or_(Report.id == identifier, Report.analysis_id == identifier))
    )
    report = res.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found.")

    await db.delete(report)
    await db.commit()
    return {"message": "Report deleted successfully."}

