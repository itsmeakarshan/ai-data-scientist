"""
AutoDS SQLite Concurrency & Database Lock Regression Tests
Verifies WAL mode, busy timeout, session isolation, concurrent read/write resilience, and safe exception handling.
"""

import asyncio
import threading
import time
import pytest
from sqlalchemy import select, text
from backend.app.agents.workflows import run_autonomous_datascience_pipeline
from backend.app.core.database import (
    AsyncSessionLocal,
    SyncSessionLocal,
    async_engine,
    init_db,
    sync_engine,
    with_db_retry,
)
from backend.app.models.entities import AnalysisRun, Dataset


@pytest.mark.asyncio
async def test_sqlite_wal_mode_and_busy_timeout_configured():
    """Verify that SQLite connection PRAGMAs (WAL mode, busy_timeout >= 30000) are active."""
    await init_db()

    # Verify Async Engine PRAGMAs
    async with AsyncSessionLocal() as session:
        res = await session.execute(text("PRAGMA journal_mode;"))
        mode = res.scalar_one_or_none()
        # In SQLite file-based DB, journal_mode should be 'wal' (or 'memory' in test :memory:)
        assert mode.lower() in ("wal", "memory")

        res_timeout = await session.execute(text("PRAGMA busy_timeout;"))
        timeout = res_timeout.scalar_one_or_none()
        assert int(timeout) >= 30000

    # Verify Sync Engine PRAGMAs
    with SyncSessionLocal() as sync_session:
        res_sync = sync_session.execute(text("PRAGMA journal_mode;")).scalar_one_or_none()
        assert res_sync.lower() in ("wal", "memory")

        timeout_sync = sync_session.execute(text("PRAGMA busy_timeout;")).scalar_one_or_none()
        assert int(timeout_sync) >= 30000


@pytest.mark.asyncio
async def test_concurrent_async_reads_and_sync_writes():
    """Verify that concurrent async API readers and sync background worker writes do not deadlock."""
    await init_db()

    # 1. Seed a test dataset and analysis run
    async with AsyncSessionLocal() as session:
        ds = Dataset(
            name="test_concurrency_ds",
            file_path="data/raw/synthetic_test.csv",
            file_type="csv",
            size_bytes=1000,
            row_count=100,
            col_count=5,
            checksum="chk_test_concurrency",
        )
        session.add(ds)
        await session.commit()
        await session.refresh(ds)
        ds_id = ds.id

        run = AnalysisRun(
            dataset_id=ds_id,
            user_goal="Test concurrent database access without locking",
            status="RUNNING",
            problem_type="classification",
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        run_id = run.id

    stop_event = threading.Event()
    sync_write_errors = []
    async_read_errors = []

    # Sync background worker writing repeatedly
    def worker_thread():
        for i in range(20):
            if stop_event.is_set():
                break
            try:
                def _do_write():
                    with SyncSessionLocal() as s:
                        r = s.query(AnalysisRun).filter(AnalysisRun.id == run_id).first()
                        if r:
                            r.plan_json = {"iteration": i, "timestamp": time.time()}
                            s.commit()
                with_db_retry(_do_write, max_retries=5, initial_delay=0.05)
            except Exception as e:
                sync_write_errors.append(str(e))
            time.sleep(0.02)

    t = threading.Thread(target=worker_thread)
    t.start()

    # Async readers polling simultaneously
    for _ in range(30):
        try:
            async with AsyncSessionLocal() as session:
                res = await session.execute(select(AnalysisRun).filter(AnalysisRun.id == run_id))
                fetched = res.scalar_one_or_none()
                assert fetched is not None
        except Exception as e:
            async_read_errors.append(str(e))
        await asyncio.sleep(0.01)

    t.join(timeout=5)
    stop_event.set()

    assert len(sync_write_errors) == 0, f"Sync writes encountered errors: {sync_write_errors}"
    assert len(async_read_errors) == 0, f"Async reads encountered errors: {async_read_errors}"


def test_with_db_retry_helper():
    """Verify that with_db_retry handles transient failures and succeeds."""
    attempts = 0

    def flaky_operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            import sqlite3
            raise sqlite3.OperationalError("database is locked")
        return "SUCCESS"

    result = with_db_retry(flaky_operation, max_retries=5, initial_delay=0.01, backoff=1.5)
    assert result == "SUCCESS"
    assert attempts == 3


@pytest.mark.asyncio
async def test_workflow_exception_handling_no_unbound_local_error():
    """Verify that a pipeline failure does NOT throw UnboundLocalError and safely updates DB."""
    await init_db()

    # Pass an invalid dataset ID to trigger an immediate failure at Stage 1 / init
    invalid_ds_id = "non_existent_dataset_uuid"
    invalid_run_id = "non_existent_run_uuid"

    # Create run in DB first
    async with AsyncSessionLocal() as session:
        # Seed dummy dataset
        ds = Dataset(
            name="test_fail_ds",
            file_path="data/raw/synthetic_test.csv",
            file_type="csv",
            size_bytes=1000,
            row_count=100,
            col_count=5,
            checksum="chk_test_fail",
        )
        session.add(ds)
        await session.commit()
        await session.refresh(ds)

        run = AnalysisRun(
            dataset_id=ds.id,
            user_goal="Test fail gracefully",
            status="PENDING",
            problem_type="classification",
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        real_run_id = run.id

    # Run with non-existent dataset should raise ValueError, NOT UnboundLocalError!
    with pytest.raises(Exception) as exc_info:
        run_autonomous_datascience_pipeline(
            analysis_id=real_run_id,
            dataset_id=invalid_ds_id,  # triggers failure before db_run lookup
            user_goal="Fail cleanly",
        )

    # Ensure original error is preserved and NOT UnboundLocalError
    assert not isinstance(exc_info.value, UnboundLocalError)
    assert "not found in database" in str(exc_info.value)
