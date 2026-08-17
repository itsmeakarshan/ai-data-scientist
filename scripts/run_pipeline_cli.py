#!/usr/bin/env python3
"""
AutoDS Autonomous Pipeline CLI Runner
Registers datasets in database and triggers end-to-end autonomous analysis.
"""

import argparse
import sys
import uuid
from pathlib import Path
import pandas as pd

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.agents.workflows import run_autonomous_datascience_pipeline
from backend.app.core.config import settings
from backend.app.core.database import SyncSessionLocal, init_db
from backend.app.core.logging import logger
from backend.app.models.entities import AnalysisRun, Dataset, DatasetProfile
from backend.app.tools.data_profiler import profile_dataset
from backend.app.tools.dataset_inspector import compute_file_sha256, load_dataset_as_dataframe
from backend.app.tools.quality_detector import detect_data_quality


def register_dataset_file(file_path: Path, name: str) -> str:
    """Register a local dataset file into database and profile it."""
    db = SyncSessionLocal()
    try:
        # Check if already registered
        checksum = compute_file_sha256(file_path)
        existing = db.query(Dataset).filter(Dataset.checksum == checksum).first()
        if existing:
            logger.info(f"Dataset '{name}' already registered with ID: {existing.id}")
            return existing.id

        df, meta = load_dataset_as_dataframe(file_path)
        profile_data = profile_dataset(df)
        quality_alerts = detect_data_quality(df, profile_data)
        profile_data["quality_alerts"] = quality_alerts

        ds = Dataset(
            name=name,
            file_path=str(file_path),
            file_type=meta["file_type"],
            size_bytes=meta["file_size_bytes"],
            row_count=meta["row_count"],
            col_count=meta["col_count"],
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
            candidate_targets=profile_data.get("candidate_targets", []),
            candidate_datetimes=profile_data.get("candidate_datetimes", []),
            inferred_problem_type=profile_data.get("inferred_problem_type"),
        )
        db.add(profile)
        db.commit()
        logger.info(f"Successfully registered dataset '{name}' ({len(df)} rows) with ID: {ds.id}")
        return ds.id
    finally:
        db.close()


def run_cli_analysis(dataset_id: str, goal: str, target: str = None, p_type: str = None, time_col: str = None):
    """Run full autonomous pipeline via CLI."""
    db = SyncSessionLocal()
    try:
        analysis_id = str(uuid.uuid4())
        run_record = AnalysisRun(
            id=analysis_id,
            dataset_id=dataset_id,
            user_goal=goal,
            status="PENDING",
            problem_type=p_type or "classification",
            target_column=target,
            time_column=time_col,
        )
        db.add(run_record)
        db.commit()

        logger.info(f"Triggering Autonomous Pipeline (Analysis ID: {analysis_id})...")
        state = run_autonomous_datascience_pipeline(
            analysis_id=analysis_id,
            dataset_id=dataset_id,
            user_goal=goal,
            target_column_override=target,
            time_column_override=time_col,
            problem_type_override=p_type,
            sync_db_session=db
        )

        print("\n" + "="*80)
        print("AUTONOMOUS PIPELINE EXECUTION COMPLETED")
        print("="*80)
        print(f"Dataset:            {state.dataset_name}")
        print(f"Problem Type:       {state.problem_type} ({state.sub_type})")
        print(f"Target Column:      {state.target_column}")
        print(f"Champion Model:     {state.best_experiment.get('model_name')}")
        print(f"Champion Metrics:   {state.best_experiment.get('metrics', {}).get('test')}")
        print(f"Critic Audit:       {state.critic_findings.get('audit_status')}")
        print(f"Visual Artifacts:   {len(state.visual_artifacts)} generated")
        print("="*80)
        print("\n--- SAMPLE BUSINESS INSIGHTS ---")
        for ins in state.business_insights:
            print(f"• [{ins.get('category').upper()}] {ins.get('title')}: {ins.get('finding')} (Evidence: {ins.get('evidence')})")
        print("\n" + "="*80 + "\n")
        return state
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoDS CLI Runner")
    parser.add_argument("--task", choices=["bank_marketing", "housing", "m5", "all"], default="bank_marketing")
    args = parser.parse_args()

    raw_dir = settings.data_raw_dir

    if args.task in ("bank_marketing", "all"):
        bank_csv = raw_dir / "bank_marketing" / "bank-additional-full.csv"
        if not bank_csv.exists():
            bank_csv = raw_dir / "bank_marketing" / "bank-full.csv"
        if bank_csv.exists():
            ds_id = register_dataset_file(bank_csv, "Bank_Marketing_UCI")
            run_cli_analysis(ds_id, "Predict whether a telemarketing client will subscribe to a term deposit.", target="y", p_type="classification")

    if args.task in ("housing", "all"):
        housing_csv = raw_dir / "housing" / "housing_prices.csv"
        if housing_csv.exists():
            ds_id = register_dataset_file(housing_csv, "California_Housing")
            run_cli_analysis(ds_id, "Predict the median house value based on geographical and demographic features.", target="median_house_value", p_type="regression")

    if args.task in ("m5", "all"):
        m5_csv = raw_dir / "m5" / "m5_sales_sample.csv"
        if m5_csv.exists():
            ds_id = register_dataset_file(m5_csv, "M5_Sales_Retail")
            run_cli_analysis(ds_id, "Forecast daily sales demand across stores and categories.", target="sales", time_col="date", p_type="forecasting")
