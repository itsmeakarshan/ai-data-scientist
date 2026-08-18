"""
Tests for AutoDS Stage Tracker, GET /api/analysis/{analysis_id}/progress, and GET /api/analysis/{analysis_id}/status endpoints.
Verifies production real-time autonomous workflow progress tracking, database persistence across refreshes, per-analysis isolation, and error handling.
"""

import pytest
from httpx import AsyncClient
from backend.app.agents.stage_tracker import (
    STAGE_DEFINITIONS,
    complete_stage_tracking,
    fail_stage_tracking,
    get_stage_status,
    start_stage_tracking,
    update_stage_progress,
)


@pytest.mark.asyncio
async def test_1_new_analysis_starts_at_0_percent():
    """1. New analysis starts at 0%."""
    run_id = "test-req-1-start-0"
    rec = start_stage_tracking(run_id)
    assert rec["analysis_id"] == run_id
    assert rec["status"] == "RUNNING"
    assert rec["overall_status"] == "RUNNING"
    assert rec["current_stage_number"] == 1
    assert rec["completed_stages"] == 0
    assert rec["progress_percentage"] == 0.0
    assert rec["stages"][0]["status"] == "RUNNING"
    assert rec["stages"][1]["status"] == "WAITING"


@pytest.mark.asyncio
async def test_2_stage_completion_updates_progress_correctly():
    """2. Stage completion updates progress correctly (e.g. stage 5 = 44.4%)."""
    run_id = "test-req-2-update-prog"
    start_stage_tracking(run_id)
    
    # Move to stage 5 (stages 1..4 completed)
    rec5 = update_stage_progress(run_id, stage_number=5)
    assert rec5["current_stage_number"] == 5
    assert rec5["completed_stages"] == 4
    assert rec5["progress_percentage"] == 44.4
    assert rec5["stages"][0]["status"] == "COMPLETED"
    assert rec5["stages"][3]["status"] == "COMPLETED"
    assert rec5["stages"][4]["status"] == "RUNNING"


@pytest.mark.asyncio
async def test_3_progress_reaches_exactly_100_percent_after_stage_9():
    """3. Progress reaches exactly 100% after stage 9."""
    run_id = "test-req-3-comp-100"
    start_stage_tracking(run_id)
    rec9 = complete_stage_tracking(run_id)
    assert rec9["status"] == "COMPLETED"
    assert rec9["overall_status"] == "COMPLETED"
    assert rec9["completed_stages"] == 9
    assert rec9["progress_percentage"] == 100.0
    assert all(s["status"] == "COMPLETED" for s in rec9["stages"])


@pytest.mark.asyncio
async def test_4_failed_stage_sets_failed_state():
    """4. Failed stage sets FAILED state."""
    run_id = "test-req-4-fail-state"
    start_stage_tracking(run_id)
    update_stage_progress(run_id, stage_number=5)
    
    fail_rec = fail_stage_tracking(run_id, "Data split failed due to imbalance.")
    assert fail_rec["status"] == "FAILED"
    assert fail_rec["overall_status"] == "FAILED"
    assert fail_rec["error_message"] == "Data split failed due to imbalance."
    assert fail_rec["stages"][4]["status"] == "FAILED"


@pytest.mark.asyncio
async def test_5_failed_run_does_not_continue_to_later_stages():
    """5. Failed run does not continue to later stages (stages 6..9 stay WAITING)."""
    run_id = "test-req-5-no-continue"
    start_stage_tracking(run_id)
    update_stage_progress(run_id, stage_number=5)
    fail_rec = fail_stage_tracking(run_id, "Model fit error.")
    
    # Completed stages before stage 5 remain 4
    assert fail_rec["completed_stages"] == 4
    # Stage 5 is FAILED
    assert fail_rec["stages"][4]["status"] == "FAILED"
    # Stages 6, 7, 8, 9 remain WAITING
    for idx in range(5, 9):
        assert fail_rec["stages"][idx]["status"] == "WAITING"


@pytest.mark.asyncio
async def test_6_progress_belongs_to_correct_analysis_id():
    """6. Progress belongs to the correct analysis_id without leaking into other runs."""
    run_a = "run-a-111"
    run_b = "run-b-222"
    
    start_stage_tracking(run_a)
    start_stage_tracking(run_b)
    
    update_stage_progress(run_a, stage_number=3)
    update_stage_progress(run_b, stage_number=7)
    
    status_a = get_stage_status(run_a)
    status_b = get_stage_status(run_b)
    
    assert status_a["analysis_id"] == run_a
    assert status_a["current_stage_number"] == 3
    assert status_a["completed_stages"] == 2
    assert status_a["progress_percentage"] == 22.2
    
    assert status_b["analysis_id"] == run_b
    assert status_b["current_stage_number"] == 7
    assert status_b["completed_stages"] == 6
    assert status_b["progress_percentage"] == 66.7


@pytest.mark.asyncio
async def test_7_get_progress_endpoint_returns_correct_structure(async_client: AsyncClient):
    """7. GET progress endpoint returns correct structured JSON response."""
    test_id = "api-progress-test-999"
    start_stage_tracking(test_id)
    update_stage_progress(test_id, stage_number=5)

    res = await async_client.get(f"/api/analysis/{test_id}/progress")
    assert res.status_code == 200
    data = res.json()

    assert data["analysis_id"] == test_id
    assert data["status"] == "RUNNING"
    assert data["overall_status"] == "RUNNING"
    assert data["current_stage"] == "Candidate Model Training & CV"
    assert data["current_stage_number"] == 5
    assert data["total_stages"] == 9
    assert data["completed_stages"] == 4
    assert data["progress_percentage"] == 44.4
    assert isinstance(data["elapsed_seconds"], int)
    assert isinstance(data["stages"], list)
    assert len(data["stages"]) == 9

    stage_1 = data["stages"][0]
    assert stage_1["number"] == 1
    assert stage_1["name"] == "Dataset Inspection & Profiling"
    assert stage_1["status"] == "COMPLETED"

    stage_5 = data["stages"][4]
    assert stage_5["number"] == 5
    assert stage_5["name"] == "Candidate Model Training & CV"
    assert stage_5["status"] == "RUNNING"


@pytest.mark.asyncio
async def test_8_existing_completed_analyses_remain_readable(async_client: AsyncClient):
    """8. Existing completed analyses remain readable via GET status & progress endpoints."""
    comp_id = "completed-run-history-555"
    start_stage_tracking(comp_id)
    complete_stage_tracking(comp_id)

    res = await async_client.get(f"/api/analysis/{comp_id}/progress")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert data["progress_percentage"] == 100.0
    assert data["completed_stages"] == 9

    res_status = await async_client.get(f"/api/analysis/{comp_id}/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert status_data["status"] == "COMPLETED"
    assert status_data["progress_percent"] == 100


@pytest.mark.asyncio
async def test_9_refresh_refetch_returns_persisted_progress():
    """9. Refresh/re-fetch returns persisted progress across state queries."""
    persisted_id = "persisted-refresh-run-777"
    start_stage_tracking(persisted_id)
    update_stage_progress(persisted_id, stage_number=4)

    # Simulate query on refetch
    st1 = get_stage_status(persisted_id)
    assert st1["current_stage_number"] == 4
    assert st1["completed_stages"] == 3
    assert st1["progress_percentage"] == 33.3

    # Subsequent re-fetch returns consistent persisted state
    st2 = get_stage_status(persisted_id)
    assert st2["current_stage_number"] == 4
    assert st2["completed_stages"] == 3
    assert st2["progress_percentage"] == 33.3


@pytest.mark.asyncio
async def test_10_no_fake_frontend_timer_controls_progress():
    """10. Backend progress is strictly calculated from stage execution state."""
    det_id = "deterministic-no-fake-timer"
    rec0 = start_stage_tracking(det_id)
    assert rec0["progress_percentage"] == 0.0

    rec1 = update_stage_progress(det_id, stage_number=2)
    assert rec1["progress_percentage"] == 11.1

    rec2 = update_stage_progress(det_id, stage_number=3)
    assert rec2["progress_percentage"] == 22.2

    rec8 = update_stage_progress(det_id, stage_number=9)
    assert rec8["progress_percentage"] == 88.9

    rec_final = complete_stage_tracking(det_id)
    assert rec_final["progress_percentage"] == 100.0
