"""
Regression & Security Tests for Dataset Path Resolution and Workspace Boundary Enforcement.
Verifies secure resolution of uploaded datasets (synthetic_test.csv) and rejection of traversal attempts.
"""

from pathlib import Path

import pytest
from fastapi import HTTPException
from httpx import AsyncClient

from backend.app.core.config import settings
from backend.app.core.security import validate_file_path


def test_validate_file_path_security_boundaries(tmp_path):
    """Verify validate_file_path accepts valid workspace paths and strictly rejects outside paths."""
    # 1. Valid workspace file
    test_file = Path(settings.STORAGE_DIR) / "raw" / "valid_test_file.csv"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("a,b,c\n1,2,3")

    resolved = validate_file_path(str(test_file))
    assert resolved == test_file.resolve()

    # 2. Valid relative path
    rel_path = f"data/raw/{test_file.name}"
    resolved_rel = validate_file_path(rel_path)
    assert resolved_rel.exists()

    # 3. Path Traversal rejection (e.g. /etc/passwd)
    with pytest.raises(HTTPException) as exc_info:
        validate_file_path("../../../../../etc/passwd")
    assert exc_info.value.status_code == 400
    assert "Security Error" in exc_info.value.detail

    # 4. Out of workspace absolute path rejection
    with pytest.raises(HTTPException) as exc_info:
        validate_file_path("/etc/hosts")
    assert exc_info.value.status_code == 400
    assert "Security Error" in exc_info.value.detail

    # 5. Non-existent file inside workspace raises clean 404
    with pytest.raises(HTTPException) as exc_info:
        validate_file_path("data/raw/completely_non_existent_file_12345.csv")
    assert exc_info.value.status_code == 404
    assert "Dataset file not found" in exc_info.value.detail


@pytest.mark.asyncio
async def test_synthetic_test_upload_and_analysis_lifecycle(async_client: AsyncClient, synthetic_csv_file):
    """
    Test uploading synthetic_test.csv, ensuring its path is safely resolved,
    and running POST /api/analysis without any path traversal security errors.
    """
    with open(synthetic_csv_file, "rb") as f:
        upload_res = await async_client.post(
            "/api/datasets/upload",
            files={"file": ("synthetic_test.csv", f, "text/csv")}
        )
    assert upload_res.status_code == 201
    ds_data = upload_res.json()
    dataset_id = ds_data["id"]
    assert ds_data["name"] == "synthetic_test.csv"
    assert ds_data["row_count"] == 200

    # 1. Fetch sample rows to confirm path resolution works
    sample_res = await async_client.get(f"/api/datasets/{dataset_id}/sample?limit=5")
    assert sample_res.status_code == 200
    assert len(sample_res.json()["rows"]) == 5

    # 2. Run POST /api/analysis and confirm zero security error
    analysis_payload = {
        "dataset_id": dataset_id,
        "user_goal": "Predict target from demographic features.",
        "target_column": "target",
        "problem_type": "classification"
    }
    analysis_res = await async_client.post("/api/analysis", json=analysis_payload)
    assert analysis_res.status_code == 201
    run_data = analysis_res.json()
    assert run_data["status"] in ("RUNNING", "COMPLETED")
    assert run_data["id"] is not None
    assert run_data["error_message"] is None

    # 3. Verify analysis details
    get_run_res = await async_client.get(f"/api/analysis/{run_data['id']}")
    assert get_run_res.status_code == 200
    fetched_run = get_run_res.json()
    assert fetched_run["dataset_id"] == dataset_id
    assert fetched_run["problem_type"] == "classification"
