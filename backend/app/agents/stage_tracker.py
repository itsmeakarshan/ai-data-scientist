"""
AutoDS Workflow Stage Tracker
Maintains deterministic stage progression, timing, stage objects, and status for active analysis runs.
"""

import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.app.core.database import SyncSessionLocal, with_db_retry
from backend.app.models.entities import AnalysisRun

STAGE_DEFINITIONS = [
    {
        "number": 1,
        "name": "Dataset Inspection & Profiling",
        "description": "Inspect schema, calculate distributions, detect missingness and column types"
    },
    {
        "number": 2,
        "name": "Problem Classification & Target Selection",
        "description": "Infer problem formulation (classification vs regression vs time-series), target, and split strategy"
    },
    {
        "number": 3,
        "name": "Autonomous Analysis Planning",
        "description": "Generate candidate model strategy and validation protocol via Gemini 3.1"
    },
    {
        "number": 4,
        "name": "Leak-Free Preprocessing & Splitting",
        "description": "Execute fit-on-train encoding, imputing, and leak-free train/test partition"
    },
    {
        "number": 5,
        "name": "Candidate Model Training & CV",
        "description": "Train candidate models with stratified k-fold CV and MLflow logging"
    },
    {
        "number": 6,
        "name": "Multi-Metric Leaderboard Ranking",
        "description": "Rank candidate model leaderboard using cross-validation performance on training portion"
    },
    {
        "number": 7,
        "name": "Methodological Critic Audit",
        "description": "Audit for prospective data leakage, severe overfitting, and execute corrective retraining if needed"
    },
    {
        "number": 8,
        "name": "SHAP Explainability & Feature Attribution",
        "description": "Compute TreeSHAP attributions and generate diagnostic visualizations"
    },
    {
        "number": 9,
        "name": "Evidence-Backed Report Synthesis",
        "description": "Synthesize 4-pillar business insights and compile final Markdown report"
    }
]

_LOCK = threading.RLock()
_ACTIVE_STAGES: Dict[str, Dict[str, Any]] = {}


def _init_stage_list() -> List[Dict[str, Any]]:
    stages = []
    for d in STAGE_DEFINITIONS:
        stages.append({
            "number": d["number"],
            "name": d["name"],
            "description": d["description"],
            "status": "WAITING",
            "started_at": None,
            "completed_at": None,
            "duration_seconds": None
        })
    return stages


def start_stage_tracking(analysis_id: str, total_stages: int = 9) -> Dict[str, Any]:
    """Initialize stage tracker for a new analysis run at Stage 1 RUNNING with 0% progress."""
    with _LOCK:
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()

        stages = _init_stage_list()
        # Stage 1 starts running
        stages[0]["status"] = "RUNNING"
        stages[0]["started_at"] = now_iso

        record = {
            "analysis_id": analysis_id,
            "status": "RUNNING",
            "overall_status": "RUNNING",
            "current_stage": STAGE_DEFINITIONS[0]["name"],
            "current_stage_number": 1,
            "total_stages": total_stages,
            "completed_stages": 0,
            "progress_percentage": 0.0,
            "progress_percent": 0.0,
            "stage_status": "RUNNING",
            "stage_started_at": now_iso,
            "stage_completed_at": None,
            "start_time_dt": now_dt,
            "elapsed_seconds": 0,
            "error_message": None,
            "error": None,
            "stages": stages,
            "models_evaluated": [],
            "current_model": None,
            "stage_details": STAGE_DEFINITIONS[0]["description"],
        }
        _ACTIVE_STAGES[analysis_id] = record
        _persist_to_db_if_possible(analysis_id, record)
        return _format_response(record)


def update_stage_progress(
    analysis_id: str,
    stage_number: int,
    current_model: Optional[str] = None,
    models_evaluated: Optional[List[str]] = None,
    stage_details: Optional[str] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """Update active stage progression deterministically."""
    with _LOCK:
        if analysis_id not in _ACTIVE_STAGES:
            start_stage_tracking(analysis_id)

        rec = _ACTIVE_STAGES[analysis_id]
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        start_t = rec.get("start_time_dt", now_dt)
        elapsed = int((now_dt - start_t).total_seconds())

        stages = rec["stages"]

        # Complete all previous stages < stage_number
        for idx in range(stage_number - 1):
            s = stages[idx]
            if s["status"] != "COMPLETED":
                s["status"] = "COMPLETED"
                if not s["started_at"]:
                    s["started_at"] = now_iso
                s["completed_at"] = now_iso
                try:
                    s_start = datetime.fromisoformat(s["started_at"])
                    s["duration_seconds"] = round((now_dt - s_start).total_seconds(), 2)
                except Exception:
                    s["duration_seconds"] = 0.0

        # Set stage_number to RUNNING if waiting
        current_idx = min(max(stage_number - 1, 0), len(stages) - 1)
        curr_s = stages[current_idx]
        if curr_s["status"] != "RUNNING" and curr_s["status"] != "COMPLETED":
            curr_s["status"] = "RUNNING"
            curr_s["started_at"] = now_iso

        completed_count = sum(1 for s in stages if s["status"] == "COMPLETED")
        prog_pct = round((completed_count / rec.get("total_stages", 9)) * 100.0, 1)

        rec.update({
            "status": "RUNNING",
            "overall_status": "RUNNING",
            "current_stage_number": stage_number,
            "current_stage": STAGE_DEFINITIONS[current_idx]["name"],
            "completed_stages": completed_count,
            "progress_percentage": prog_pct,
            "progress_percent": prog_pct,
            "stage_status": "RUNNING",
            "stage_started_at": curr_s["started_at"],
            "stage_completed_at": None,
            "elapsed_seconds": elapsed,
            "stage_details": stage_details or STAGE_DEFINITIONS[current_idx]["description"],
            "error_message": error,
            "error": error
        })

        if current_model is not None:
            rec["current_model"] = current_model
        if models_evaluated is not None:
            rec["models_evaluated"] = list(models_evaluated)

        _persist_to_db_if_possible(analysis_id, rec)
        return _format_response(rec)


def complete_stage_tracking(analysis_id: str, models_evaluated: Optional[List[str]] = None) -> Dict[str, Any]:
    """Mark all 9 stages completed successfully and set progress to 100%."""
    with _LOCK:
        if analysis_id not in _ACTIVE_STAGES:
            start_stage_tracking(analysis_id)

        rec = _ACTIVE_STAGES[analysis_id]
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        start_t = rec.get("start_time_dt", now_dt)
        elapsed = int((now_dt - start_t).total_seconds())

        stages = rec["stages"]
        for s in stages:
            if s["status"] != "COMPLETED":
                s["status"] = "COMPLETED"
                if not s["started_at"]:
                    s["started_at"] = now_iso
                s["completed_at"] = now_iso
                try:
                    s_start = datetime.fromisoformat(s["started_at"])
                    s["duration_seconds"] = round((now_dt - s_start).total_seconds(), 2)
                except Exception:
                    s["duration_seconds"] = 0.0

        rec.update({
            "status": "COMPLETED",
            "overall_status": "COMPLETED",
            "current_stage_number": 9,
            "current_stage": STAGE_DEFINITIONS[-1]["name"],
            "completed_stages": 9,
            "progress_percentage": 100.0,
            "progress_percent": 100.0,
            "stage_status": "COMPLETED",
            "stage_completed_at": now_iso,
            "elapsed_seconds": elapsed,
            "current_model": None,
            "stage_details": "Autonomous analysis pipeline completed successfully.",
            "error_message": None,
            "error": None
        })
        if models_evaluated is not None:
            rec["models_evaluated"] = list(models_evaluated)

        _persist_to_db_if_possible(analysis_id, rec)
        return _format_response(rec)


def fail_stage_tracking(analysis_id: str, error_message: str) -> Dict[str, Any]:
    """Mark workflow progress as failed and preserve completed stages."""
    with _LOCK:
        if analysis_id not in _ACTIVE_STAGES:
            start_stage_tracking(analysis_id)

        rec = _ACTIVE_STAGES[analysis_id]
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        start_t = rec.get("start_time_dt", now_dt)
        elapsed = int((now_dt - start_t).total_seconds())

        stages = rec["stages"]
        curr_num = rec.get("current_stage_number", 1)
        curr_idx = min(max(curr_num - 1, 0), len(stages) - 1)

        # Mark current running stage as FAILED
        stages[curr_idx]["status"] = "FAILED"
        stages[curr_idx]["completed_at"] = now_iso

        completed_count = sum(1 for s in stages if s["status"] == "COMPLETED")
        prog_pct = round((completed_count / rec.get("total_stages", 9)) * 100.0, 1)

        rec.update({
            "status": "FAILED",
            "overall_status": "FAILED",
            "stage_status": "FAILED",
            "completed_stages": completed_count,
            "progress_percentage": prog_pct,
            "progress_percent": prog_pct,
            "elapsed_seconds": elapsed,
            "error_message": error_message,
            "error": error_message,
            "stage_completed_at": now_iso
        })

        _persist_to_db_if_possible(analysis_id, rec)
        return _format_response(rec)


def get_stage_status(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve current live or persisted stage tracking record."""
    with _LOCK:
        rec = _ACTIVE_STAGES.get(analysis_id)
        if rec:
            now_dt = datetime.now(timezone.utc)
            start_t = rec.get("start_time_dt", now_dt)
            if rec["status"] == "RUNNING":
                rec["elapsed_seconds"] = int((now_dt - start_t).total_seconds())
            return _format_response(rec)

        # Reconstitute from DB if not in active memory
        db_rec = _load_from_db(analysis_id)
        if db_rec:
            _ACTIVE_STAGES[analysis_id] = db_rec
            return _format_response(db_rec)

        return None


def _persist_to_db_if_possible(analysis_id: str, rec: Dict[str, Any]) -> None:
    """Save stage tracking JSON to DB AnalysisRun.plan_json["_stage_progress"]."""
    def _do_persist():
        db = SyncSessionLocal()
        try:
            run = db.query(AnalysisRun).filter(AnalysisRun.id == analysis_id).first()
            if run:
                plan = dict(run.plan_json or {})
                formatted = _format_response(rec)
                plan["_stage_progress"] = formatted
                run.plan_json = plan
                if rec["status"] in ("COMPLETED", "FAILED"):
                    run.status = rec["status"]
                    if rec["status"] == "FAILED":
                        run.error_message = rec.get("error_message")
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    try:
        with_db_retry(_do_persist, max_retries=3, initial_delay=0.05, backoff=1.5)
    except Exception:
        pass


def _load_from_db(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Load persisted stage progress from DB."""
    def _do_load():
        db = SyncSessionLocal()
        try:
            run = db.query(AnalysisRun).filter(AnalysisRun.id == analysis_id).first()
            if not run:
                return None
            plan = run.plan_json or {}
            saved = plan.get("_stage_progress")
            if saved and isinstance(saved, dict):
                start_dt = run.created_at or datetime.now(timezone.utc)
                saved["start_time_dt"] = start_dt
                return saved

            # Synthesize fallback from DB AnalysisRun fields
            now_dt = datetime.now(timezone.utc)
            start_dt = run.created_at or now_dt
            comp_dt = run.completed_at or now_dt
            elapsed = int((comp_dt - start_dt).total_seconds()) if run.status == "COMPLETED" else int((now_dt - start_dt).total_seconds())

            is_comp = run.status == "COMPLETED"
            is_fail = run.status == "FAILED"

            stages = _init_stage_list()
            if is_comp:
                for s in stages:
                    s["status"] = "COMPLETED"
                    s["started_at"] = start_dt.isoformat()
                    s["completed_at"] = comp_dt.isoformat()
                    s["duration_seconds"] = round(elapsed / 9.0, 2)
                comp_count = 9
                prog_pct = 100.0
                curr_stage = STAGE_DEFINITIONS[-1]["name"]
                curr_num = 9
                stg_status = "COMPLETED"
            elif is_fail:
                stages[0]["status"] = "FAILED"
                stages[0]["completed_at"] = comp_dt.isoformat()
                comp_count = 0
                prog_pct = 0.0
                curr_stage = STAGE_DEFINITIONS[0]["name"]
                curr_num = 1
                stg_status = "FAILED"
            else:
                stages[0]["status"] = "RUNNING"
                stages[0]["started_at"] = start_dt.isoformat()
                comp_count = 0
                prog_pct = 0.0
                curr_stage = STAGE_DEFINITIONS[0]["name"]
                curr_num = 1
                stg_status = "RUNNING"

            return {
                "analysis_id": run.id,
                "status": run.status if run.status in ("COMPLETED", "FAILED", "RUNNING") else "RUNNING",
                "overall_status": run.status if run.status in ("COMPLETED", "FAILED", "RUNNING") else "RUNNING",
                "current_stage": curr_stage,
                "current_stage_number": curr_num,
                "total_stages": 9,
                "completed_stages": comp_count,
                "progress_percentage": prog_pct,
                "progress_percent": prog_pct,
                "stage_status": stg_status,
                "stage_started_at": start_dt.isoformat(),
                "stage_completed_at": comp_dt.isoformat() if (is_comp or is_fail) else None,
                "start_time_dt": start_dt,
                "elapsed_seconds": max(elapsed, 0),
                "error_message": run.error_message if is_fail else None,
                "error": run.error_message if is_fail else None,
                "stages": stages,
                "models_evaluated": [],
                "current_model": None,
                "stage_details": "Persisted DB Run"
            }
        finally:
            db.close()

    try:
        return with_db_retry(_do_load, max_retries=3, initial_delay=0.05, backoff=1.5)
    except Exception:
        return None


def _format_response(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Return clean dict conforming to API response schema."""
    resp = dict(rec)
    resp.pop("start_time_dt", None)
    return resp
