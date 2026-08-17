"""
AutoDS Safe SQL Query API Endpoint
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.core.database import get_db
from backend.app.models.entities import Dataset
from backend.app.schemas.domain import QueryRequest, QueryResponse
from backend.app.tools.safe_query import execute_safe_sql_query


router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def run_safe_query(
    req: QueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """Execute safe read-only SQL query over a dataset using DuckDB."""
    res = await db.execute(select(Dataset).filter(Dataset.id == req.dataset_id))
    dataset = res.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")

    try:
        query_result = execute_safe_sql_query(
            file_path=dataset.file_path,
            sql_query=req.sql_query,
            limit=req.limit
        )
        return QueryResponse(**query_result)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Query execution error: {e}")
