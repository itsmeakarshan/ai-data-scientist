"""
AutoDS End-to-End Verification Runner for Wine Quality Dataset
Leaves target_column and problem_type EMPTY to test autonomous detection.
Verifies all visual diagnostics (ROC, PR, Confusion Matrix, Feature Importance).
"""

import asyncio
import time
import sys
from pathlib import Path
import httpx
from backend.app.core.config import settings
from backend.app.core.database import SyncSessionLocal
from backend.app.main import app
from backend.app.models.entities import AnalysisRun, Dataset, Experiment, ModelRecord, Report, DatasetProfile


async def main():
    print("=" * 80)
    print("AUTODS REAL END-TO-END VERIFICATION: WINE QUALITY RED")
    print("=" * 80)

    # 1. Locate Dataset
    db = SyncSessionLocal()
    try:
        ds = db.query(Dataset).filter(Dataset.name == "winequality-red.csv").order_by(Dataset.created_at.desc()).first()
        if not ds:
            print("[ERROR] winequality-red.csv dataset not found in database.")
            sys.exit(1)
        dataset_id = ds.id
        print(f"[1/5] Target Dataset: {ds.name} (ID: {dataset_id})")
        print(f"      File Path: {ds.file_path} | Rows: {ds.row_count} | Cols: {ds.col_count}")
    finally:
        db.close()

    user_goal = (
        "Predict wine quality based on its chemical properties and identify the main factors "
        "associated with higher or lower quality. Automatically inspect the dataset, detect the "
        "appropriate target and problem type, identify data-quality issues such as missing values "
        "and duplicate rows, compare multiple machine learning models using cross-validation, "
        "evaluate the best model on an untouched holdout set, perform leakage and methodological audits, "
        "explain the model's predictive drivers, and generate evidence-backed insights and a final report."
    )

    # 2. Trigger Analysis Run with target_column=None and problem_type=None
    print("\n[2/5] Triggering Autonomous DS Pipeline via POST /api/v1/analysis (Target & Type EMPTY)...")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        res = await client.post(
            "/api/v1/analysis",
            json={
                "dataset_id": dataset_id,
                "user_goal": user_goal,
                "target_column": None, # LEAVE EMPTY
                "problem_type": None,  # LEAVE EMPTY
            }
        )
        if res.status_code != 201:
            print(f"[ERROR] Failed to create analysis run: {res.status_code} {res.text}")
            sys.exit(1)
        
        analysis_data = res.json()
        analysis_id = analysis_data["id"]
        print(f"[SUCCESS] Analysis Run Created: {analysis_id}")
        print(f"          Initial Status: {analysis_data['status']}")

        # 3. Concurrent Polling Simulation
        print("\n[3/5] Starting Real-Time Progress Polling...")
        start_time = time.time()
        last_stage = 0
        poll_count = 0
        final_status = None
        error_msg = None

        while True:
            await asyncio.sleep(1.0)
            poll_count += 1
            elapsed = time.time() - start_time

            prog_res = await client.get(f"/api/v1/analysis/{analysis_id}/progress")
            if prog_res.status_code == 200:
                prog = prog_res.json()
                current_stage = prog.get("current_stage_number", 0)
                stage_name = prog.get("current_stage", "")
                progress_pct = prog.get("progress_percentage", 0)
                overall_status = prog.get("status", "")
                evaluated = prog.get("models_evaluated", [])

                if current_stage != last_stage or poll_count % 5 == 0:
                    print(
                        f"  [Poll #{poll_count:02d} | {elapsed:5.1f}s] "
                        f"Stage {current_stage}/9: {stage_name:<40} | "
                        f"Progress: {progress_pct:3.0f}% | Status: {overall_status} | "
                        f"Evaluated: {len(evaluated)} models"
                    )
                    last_stage = current_stage

                if overall_status in ("COMPLETED", "FAILED"):
                    final_status = overall_status
                    error_msg = prog.get("error_message") or prog.get("error")
                    break

            if elapsed > 300:
                print("[ERROR] Analysis run timed out after 300 seconds.")
                final_status = "TIMEOUT"
                break

    # 4. Verify Final State
    print("\n[4/5] Pipeline Execution Result:")
    print(f"      Final Status: {final_status}")
    print(f"      Total Time:   {time.time() - start_time:.2f}s")

    if final_status != "COMPLETED":
        print(f"[FAILURE] Run did not complete successfully. Error: {error_msg}")
        sys.exit(1)

    # 5. Deep Forensic Inspection of Generated Visual Diagnostics
    print("\n[5/5] Forensic Inspection of Generated Visual Diagnostics & Report...")
    db = SyncSessionLocal()
    try:
        run = db.query(AnalysisRun).filter(AnalysisRun.id == analysis_id).first()
        report = db.query(Report).filter(Report.analysis_id == analysis_id).first()

        print(f"\n--- A. Autonomous Detection Verification ---")
        print(f"  Target Column:       {run.target_column} (Expected: quality)")
        print(f"  Problem Type:        {run.problem_type} (Expected: classification)")
        assert run.target_column == "quality", f"Expected target 'quality', got '{run.target_column}'"
        assert run.problem_type == "classification", f"Expected 'classification', got '{run.problem_type}'"
        print("  [PASSED] Auto-detection correctly identified Target='quality' and Type='classification'")

        print(f"\n--- B. Champion Model & Holdout Metrics ---")
        print(f"  Champion Model ID:   {run.final_model_id}")
        champ_exp = db.query(Experiment).filter(Experiment.analysis_id == analysis_id, Experiment.model_record != None).first()
        if champ_exp:
            print(f"  Champion Model Name: {champ_exp.model_name}")
            test_m = champ_exp.metrics_json.get("test", {})
            print(f"  Holdout Accuracy:    {test_m.get('accuracy')}")
            print(f"  Holdout Macro-AUC:   {test_m.get('roc_auc')}")
            print(f"  Holdout Macro PR-AUC:{test_m.get('pr_auc')}")
            print(f"  Holdout Confusion:   {test_m.get('confusion_matrix')}")

        print(f"\n--- C. Visual Diagnostics Artifacts Verification ---")
        print(f"  Total Artifacts in Report: {len(report.artifact_paths)}")
        for idx, p in enumerate(report.artifact_paths, 1):
            full_p = Path(settings.REPORTS_DIR).parent / p
            exists = full_p.exists()
            size = full_p.stat().st_size if exists else 0
            print(f"    [{idx}] Path: {p}")
            print(f"        Exists: {exists} | File Size: {size:,} bytes")
            assert exists, f"Artifact file does not exist: {full_p}"
            assert size > 1000, f"Artifact file is empty or corrupted: {full_p}"

        # Verify all four core classification diagnostics are present
        has_roc = any("_roc.png" in p for p in report.artifact_paths)
        has_pr = any("_pr.png" in p for p in report.artifact_paths)
        has_cm = any("_cm.png" in p for p in report.artifact_paths)
        has_imp = any("_feature_imp.png" in p for p in report.artifact_paths)

        print(f"\n--- D. Core Classification Diagnostics Checklist ---")
        print(f"  [ {'X' if has_roc else ' '} ] 1. ROC Curve (One-vs-Rest Multiclass AUC)")
        print(f"  [ {'X' if has_pr else ' '} ] 2. Precision-Recall Curve (One-vs-Rest Multiclass PR-AUC)")
        print(f"  [ {'X' if has_cm else ' '} ] 3. Confusion Matrix (Holdout Test Set)")
        print(f"  [ {'X' if has_imp else ' '} ] 4. Top Predictive Drivers (Feature Importance / SHAP)")

        assert has_roc, "Missing ROC Curve artifact"
        assert has_pr, "Missing Precision-Recall Curve artifact"
        assert has_cm, "Missing Confusion Matrix artifact"
        assert has_imp, "Missing Feature Importance artifact"

        print("\n" + "=" * 80)
        print("SUCCESS: ALL FOUR CORE CLASSIFICATION DIAGNOSTICS GENERATED AND VERIFIED!")
        print("=" * 80)
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
