"""
AutoDS OpenML-CC18 Benchmark Runner
Evaluates autonomous pipeline generalization across diverse standard benchmark datasets.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from backend.app.agents.workflows import run_autonomous_datascience_pipeline
from backend.app.core.config import settings
from backend.app.core.database import SyncSessionLocal, init_db
from backend.app.core.logging import logger
from backend.app.models.entities import AnalysisRun, Dataset, DatasetProfile
from backend.app.tools.data_profiler import profile_dataset
from backend.app.tools.dataset_inspector import compute_file_sha256
from backend.app.tools.quality_detector import detect_data_quality


def setup_benchmark_dataset(name: str, df: pd.DataFrame, target_col: str, problem_type: str) -> str:
    """Register benchmark dataset in DB and return dataset ID."""
    bench_dir = Path(settings.STORAGE_DIR) / "raw" / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    csv_path = bench_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)

    db = SyncSessionLocal()
    try:
        checksum = compute_file_sha256(csv_path)
        existing = db.query(Dataset).filter(Dataset.checksum == checksum).first()
        if existing:
            return existing.id

        profile_data = profile_dataset(df)
        quality_alerts = detect_data_quality(df, profile_data, target_column=target_col)
        profile_data["quality_alerts"] = quality_alerts

        ds = Dataset(
            name=f"Benchmark_{name}",
            file_path=str(csv_path),
            file_type="csv",
            size_bytes=csv_path.stat().st_size,
            row_count=len(df),
            col_count=len(df.columns),
            checksum=checksum,
        )
        db.add(ds)
        db.flush()

        profile = DatasetProfile(
            dataset_id=ds.id,
            summary_stats=profile_data.get("summary_stats", {}),
            missingness_report=profile_data.get("missingness_report", {}),
            column_types=profile_data.get("column_types", {}),
            correlations=profile_data.get("correlations", {}),
            quality_alerts=profile_data.get("quality_alerts", []),
            candidate_targets=[target_col],
            candidate_datetimes=[],
            inferred_problem_type=problem_type,
        )
        db.add(profile)
        db.commit()
        return ds.id
    finally:
        db.close()


def run_openml_benchmark_suite() -> List[Dict[str, Any]]:
    """Run standard benchmark suite across classification & regression datasets."""
    logger.info("Initializing OpenML & Standard Benchmark Evaluation Suite...")
    
    benchmark_tasks = []

    # 1. Breast Cancer (Binary Classification)
    bc = load_breast_cancer(as_frame=True)
    bc_df = bc.frame
    benchmark_tasks.append({
        "name": "BreastCancer_Wisconsin",
        "df": bc_df,
        "target": "target",
        "problem_type": "classification",
        "goal": "Classify malignant vs benign tumors."
    })

    # 2. Wine Recognition (Multiclass Classification)
    wine = load_wine(as_frame=True)
    wine_df = wine.frame
    benchmark_tasks.append({
        "name": "Wine_Recognition",
        "df": wine_df,
        "target": "target",
        "problem_type": "classification",
        "goal": "Classify wine cultivar origin."
    })

    # 3. Diabetes Progression (Tabular Regression)
    diab = load_diabetes(as_frame=True)
    diab_df = diab.frame
    benchmark_tasks.append({
        "name": "Diabetes_Progression",
        "df": diab_df,
        "target": "target",
        "problem_type": "regression",
        "goal": "Predict quantitative disease progression measure."
    })

    results = []
    db = SyncSessionLocal()

    try:
        for t in benchmark_tasks:
            logger.info(f"Running benchmark on {t['name']} ({len(t['df'])} rows)...")
            start_time = time.time()
            ds_id = setup_benchmark_dataset(t["name"], t["df"], t["target"], t["problem_type"])
            
            import uuid
            analysis_id = str(uuid.uuid4())
            run_rec = AnalysisRun(
                id=analysis_id,
                dataset_id=ds_id,
                user_goal=t["goal"],
                status="PENDING",
                problem_type=t["problem_type"],
                target_column=t["target"],
            )
            db.add(run_rec)
            db.commit()

            state = run_autonomous_datascience_pipeline(
                analysis_id=analysis_id,
                dataset_id=ds_id,
                user_goal=t["goal"],
                target_column_override=t["target"],
                problem_type_override=t["problem_type"],
                sync_db_session=db
            )
            
            runtime = round(time.time() - start_time, 2)
            best_m = state.best_experiment
            test_metrics = best_m.get("metrics", {}).get("test", {}) if best_m else {}

            res_entry = {
                "dataset": t["name"],
                "problem_type": t["problem_type"],
                "rows": len(t["df"]),
                "cols": len(t["df"].columns),
                "runtime_sec": runtime,
                "champion_model": best_m.get("model_name") if best_m else "N/A",
                "test_score": test_metrics.get("roc_auc", test_metrics.get("r2", test_metrics.get("accuracy", 0.0))),
                "metrics": test_metrics,
                "critic_audit": state.critic_findings.get("audit_status", "PASSED"),
                "status": state.status,
            }
            results.append(res_entry)
            logger.info(f"Finished {t['name']}: Champion={res_entry['champion_model']}, Score={res_entry['test_score']}, Time={runtime}s")

    finally:
        db.close()

    print("\n" + "="*85)
    print("OPENML & STANDARD BENCHMARK EVALUATION RESULTS")
    print("="*85)
    print(f"{'Dataset':<26} {'Type':<15} {'Champion Model':<20} {'Test Score':<12} {'Runtime':<10}")
    print("-"*85)
    for r in results:
        print(f"{r['dataset']:<26} {r['problem_type']:<15} {r['champion_model']:<20} {r['test_score']:<12.4f} {r['runtime_sec']}s")
    print("="*85 + "\n")
    return results


if __name__ == "__main__":
    run_openml_benchmark_suite()
