"""
End-to-End API Integration Tests for AutoDS REST Endpoints
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(async_client: AsyncClient):
    """Test /api/health endpoint."""
    response = await async_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "version" in data
    assert "database_connected" in data


@pytest.mark.asyncio
async def test_dataset_upload_and_lifecycle(async_client: AsyncClient, synthetic_csv_file):
    """Test uploading a dataset, fetching profile, previewing rows, and listing."""
    with open(synthetic_csv_file, "rb") as f:
        response = await async_client.post(
            "/api/datasets/upload",
            files={"file": ("synthetic_test.csv", f, "text/csv")}
        )
    assert response.status_code == 201
    ds_data = response.json()
    dataset_id = ds_data["id"]
    assert ds_data["name"] == "synthetic_test.csv"
    assert ds_data["row_count"] == 200

    # 1. Fetch Dataset Details
    get_res = await async_client.get(f"/api/datasets/{dataset_id}")
    assert get_res.status_code == 200
    assert get_res.json()["profile"] is not None

    # 2. Fetch Sample Rows
    sample_res = await async_client.get(f"/api/datasets/{dataset_id}/sample?limit=10")
    assert sample_res.status_code == 200
    sample_data = sample_res.json()
    assert len(sample_data["rows"]) == 10
    assert "income" in sample_data["columns"]

    # 3. Execute Safe SQL Query
    query_payload = {
        "dataset_id": dataset_id,
        "sql_query": "SELECT job_category, COUNT(*) as count FROM dataset GROUP BY job_category;"
    }
    query_res = await async_client.post("/api/query", json=query_payload)
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert q_data["row_count"] > 0

    # 4. Trigger Autonomous Analysis
    analysis_payload = {
        "dataset_id": dataset_id,
        "user_goal": "Predict whether the customer meets the target threshold.",
        "target_column": "target",
        "problem_type": "classification"
    }
    analysis_res = await async_client.post("/api/analysis", json=analysis_payload)
    assert analysis_res.status_code == 201
    run_data = analysis_res.json()
    analysis_id = run_data["id"]
    assert run_data["status"] in ("RUNNING", "COMPLETED")

    # 5. List Models / Model Registry
    models_res = await async_client.get("/api/models")
    assert models_res.status_code == 200

    # 6. Fetch Analysis Progress & Status
    status_res = await async_client.get(f"/api/analysis/{analysis_id}/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["analysis_id"] == analysis_id

    # 7. Fetch Report
    rep_res = await async_client.get(f"/api/reports/{analysis_id}")
    assert rep_res.status_code == 200
    rep_data = rep_res.json()
    assert "Executive Summary" in rep_data["full_report_markdown"]

    # 8. Interactive Chat Message
    chat_payload = {
        "dataset_id": dataset_id,
        "content": "Which model performed best on this dataset?"
    }
    chat_res = await async_client.post("/api/agent/chat", json=chat_payload)
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert chat_data["role"] == "assistant"
    assert len(chat_data["content"]) > 10
