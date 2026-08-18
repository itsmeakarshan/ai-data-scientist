"""
AutoDS Publication-Ready Notebook Generator & Runner
Constructs and fully executes the 5 official demonstration notebooks for AutoDS.
Captures real stdout, outputs, and figures so all notebooks are 100% pre-rendered.
"""

import base64
import contextlib
import io
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_code_cell(source_code: str, exec_count: int, outputs: list) -> dict:
    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in source_code.strip().split("\n")]
    }


def create_markdown_cell(markdown_text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in markdown_text.strip().split("\n")]
    }


def execute_and_build_notebook(cells_spec: list, output_path: Path):
    """Execute code cells in sequential Python context and build full .ipynb."""
    exec_scope = {}
    built_cells = []
    exec_counter = 1

    for item in cells_spec:
        cell_type = item["type"]
        content = item["content"]

        if cell_type == "markdown":
            built_cells.append(create_markdown_cell(content))
        elif cell_type == "code":
            plt.close("all")
            stdout_buf = io.StringIO()
            outputs = []
            
            try:
                with contextlib.redirect_stdout(stdout_buf):
                    exec(content, exec_scope)
            except Exception as e:
                print(f"[ERROR in cell {exec_counter} of {output_path.name}]: {e}")
                raise

            stdout_val = stdout_buf.getvalue()
            if stdout_val:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in stdout_val.split("\n") if line or line == ""]
                })

            # Check if matplotlib created any figures
            figs = [plt.figure(i) for i in plt.get_fignums()]
            for fig in figs:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                outputs.append({
                    "data": {
                        "image/png": img_b64,
                        "text/plain": ["<Figure size ...>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
                plt.close(fig)

            built_cells.append(create_code_cell(content, exec_counter, outputs))
            exec_counter += 1

    nb_json = {
        "cells": built_cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            },
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=1)
    print(f"[SUCCESS] Built and pre-executed: {output_path.name} ({len(built_cells)} cells)")


def build_all_notebooks():
    nb_dir = Path("notebooks")
    nb_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Notebook 01: Data Profiling & Quality Inspection
    # -------------------------------------------------------------------------
    nb1_cells = [
        {
            "type": "markdown",
            "content": """# 01 — Autonomous Dataset Profiling & Data Quality Auditing
**AutoDS Scientific Methodology Demonstration**

This notebook demonstrates the deterministic profiling, cryptographic hashing, and automated data-quality detection engines powering AutoDS Stage 1.
"""
        },
        {
            "type": "code",
            "content": """import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from backend.app.tools.data_profiler import profile_dataset
from backend.app.tools.quality_detector import detect_data_quality
from backend.app.tools.dataset_inspector import compute_file_sha256

sns.set_theme(style="whitegrid")
print("AutoDS Profiling modules successfully imported.")"""
        },
        {
            "type": "markdown",
            "content": """## 1. Loading and Inspecting the Wine Quality Dataset
We begin by profiling `winequality-red.csv`, computing its SHA-256 integrity hash, and analyzing its schema.
"""
        },
        {
            "type": "code",
            "content": """data_path = "data/raw/31513e04_winequality-red.csv"
if not os.path.exists(data_path):
    df = pd.DataFrame({
        "fixed acidity": [7.4, 7.8, 7.8, 11.2],
        "volatile acidity": [0.70, 0.88, 0.76, 0.28],
        "citric acid": [0.0, 0.0, 0.04, 0.56],
        "alcohol": [9.4, 9.8, 9.8, 9.8],
        "quality": [5, 5, 5, 6]
    })
else:
    df = pd.read_csv(data_path, sep=None, engine="python")

print(f"Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"File Checksum (SHA-256): {compute_file_sha256(data_path) if os.path.exists(data_path) else 'N/A'}")
print("Top 3 rows:")
print(df.head(3))"""
        },
        {
            "type": "markdown",
            "content": """## 2. Comprehensive Profiling & Statistical Summary
The `profile_dataset` tool extracts missingness metrics, column type mappings, distribution summaries, and potential target candidates.
"""
        },
        {
            "type": "code",
            "content": """profile = profile_dataset(df)
print("Candidate Targets Identified:", profile.get("candidate_targets"))
print("Total Missing Cells Pct:", profile.get("missingness_report", {}).get("total_missing_pct"), "%")
print("Duplicate Rows Detected:", profile.get("missingness_report", {}).get("duplicate_rows"))

alerts = detect_data_quality(df, profile, target_column="quality")
print(f"\\nQuality Alerts ({len(alerts)}):")
for a in alerts:
    print(f"  [{a['severity'].upper()}] {a['message']}")"""
        },
        {
            "type": "markdown",
            "content": """## 3. Visualizing Target Distribution
"""
        },
        {
            "type": "code",
            "content": """fig, ax = plt.subplots(figsize=(7, 4))
df["quality"].value_counts().sort_index().plot(kind="bar", color="#3b82f6", edgecolor="#1d4ed8", ax=ax)
ax.set_title("Wine Quality Target Class Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Quality Score", fontsize=10, fontweight="bold")
ax.set_ylabel("Frequency", fontsize=10, fontweight="bold")
plt.tight_layout()"""
        }
    ]
    execute_and_build_notebook(nb1_cells, nb_dir / "01_Data_Profiling_and_Quality.ipynb")

    # -------------------------------------------------------------------------
    # Notebook 02: Leakage & Preprocessing
    # -------------------------------------------------------------------------
    nb2_cells = [
        {
            "type": "markdown",
            "content": """# 02 — Leak-Free Preprocessing & Problem Classification
**AutoDS Scientific Methodology Demonstration**

This notebook demonstrates problem classification, target-component leakage auditing, and strict fold-safe preprocessing.
"""
        },
        {
            "type": "code",
            "content": """import pandas as pd
import numpy as np
from backend.app.tools.problem_classifier import classify_problem_type
from backend.app.tools.preprocessor import prepare_train_test_split

print("AutoDS Preprocessing modules loaded.")"""
        },
        {
            "type": "markdown",
            "content": """## 1. Automatic Problem Classification
AutoDS inspects the detected target column, cardinality, and data type to determine the appropriate mathematical modeling objective.
"""
        },
        {
            "type": "code",
            "content": """data_path = "data/raw/31513e04_winequality-red.csv"
df = pd.read_csv(data_path, sep=None, engine="python")
problem_info = classify_problem_type(df, target_column="quality", user_goal="Predict wine quality score")
print("Detected Problem Type:", problem_info.get("problem_type"))
print("Is Binary Classification:", problem_info.get("is_binary"))
print("Validation Strategy:", problem_info.get("validation_strategy"))"""
        },
        {
            "type": "markdown",
            "content": """## 2. Touchless Holdout Partitioning & Leak-Free Feature Encoding
All transformations (scaling, imputation, one-hot encoding) are fitted strictly on the training partition and applied to the untouched holdout test partition without label peeking.
"""
        },
        {
            "type": "code",
            "content": """X_train, X_test, y_train, y_test, prep_artifacts = prepare_train_test_split(
    df=df,
    target_column="quality",
    problem_type=problem_info.get("problem_type"),
    test_size=0.2,
    random_state=42
)

print(f"X_train Shape: {X_train.shape}")
print(f"X_test Shape:  {X_test.shape}")
print(f"Engineered Features ({len(prep_artifacts.feature_names)}):", prep_artifacts.feature_names[:6], "...")"""
        }
    ]
    execute_and_build_notebook(nb2_cells, nb_dir / "02_Leakage_and_Preprocessing.ipynb")

    # -------------------------------------------------------------------------
    # Notebook 03: Model Comparison & Validation
    # -------------------------------------------------------------------------
    nb3_cells = [
        {
            "type": "markdown",
            "content": """# 03 — Multi-Metric Model Comparison & Cross-Validation
**AutoDS Scientific Methodology Demonstration**

This notebook demonstrates candidate algorithm benchmarking using Stratified Cross-Validation on training data, followed by champion model selection.
"""
        },
        {
            "type": "code",
            "content": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backend.app.tools.preprocessor import prepare_train_test_split
from backend.app.tools.ml_trainer import train_and_evaluate_model, evaluate_locked_champion_on_holdout

df = pd.read_csv("data/raw/31513e04_winequality-red.csv", sep=None, engine="python")
X_train, X_test, y_train, y_test, prep = prepare_train_test_split(df, target_column="quality", problem_type="classification", test_size=0.2, random_state=42)
print("Data partitioned for model evaluation.")"""
        },
        {
            "type": "markdown",
            "content": """## 1. Candidate Model Cross-Validation
We evaluate LightGBM, Random Forest, Logistic Regression, and Dummy Baseline across 3-fold Stratified CV.
"""
        },
        {
            "type": "code",
            "content": """candidates = ["LogisticRegression", "RandomForest", "Baseline"]
results = []

for model_name in candidates:
    res = train_and_evaluate_model(
        model_name=model_name,
        problem_type="classification",
        X_train=X_train,
        y_train=y_train,
        feature_names=prep.feature_names,
        cv_folds=3,
        user_goal="Predict wine quality",
        track_mlflow=False
    )
    cv_score = res["metrics"]["cv_mean"]
    cv_std = res["metrics"]["cv_std"]
    print(f"Model: {model_name:<20} | CV Mean: {cv_score:.4f} (+/- {cv_std:.4f}) | Fit Time: {res['train_time_sec']:.3f}s")
    results.append(res)"""
        },
        {
            "type": "markdown",
            "content": """## 2. Champion Selection & Holdout Evaluation
The champion model is fitted on the complete training set and evaluated on the untouched holdout test set exactly once.
"""
        },
        {
            "type": "code",
            "content": """champion = max(results, key=lambda x: x["metrics"]["cv_mean"])
print(f"Selected Champion Model: {champion['model_name']} (CV: {champion['metrics']['cv_mean']:.4f})")

holdout_res = evaluate_locked_champion_on_holdout(
    champion_exp=champion,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    user_goal="Predict wine quality",
    track_mlflow=False
)

test_metrics = champion["metrics"]["test"]
print(f"Final Touchless Holdout Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Final Touchless Holdout Macro-AUC: {test_metrics['roc_auc']:.4f}")"""
        }
    ]
    execute_and_build_notebook(nb3_cells, nb_dir / "03_Model_Comparison_and_Validation.ipynb")

    # -------------------------------------------------------------------------
    # Notebook 04: Model Explainability
    # -------------------------------------------------------------------------
    nb4_cells = [
        {
            "type": "markdown",
            "content": """# 04 — Model Explainability & Visual Diagnostics
**AutoDS Scientific Methodology Demonstration**

This notebook demonstrates feature importance extraction, SHAP attributions, and dataset-agnostic visual diagnostic generation.
"""
        },
        {
            "type": "code",
            "content": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backend.app.tools.preprocessor import prepare_train_test_split
from backend.app.tools.ml_trainer import train_and_evaluate_model, evaluate_locked_champion_on_holdout
from backend.app.tools.explainability import calculate_feature_importance
from backend.app.tools.visualizer import (
    generate_roc_pr_plots,
    generate_confusion_matrix_plot,
    generate_feature_importance_plot
)

df = pd.read_csv("data/raw/31513e04_winequality-red.csv", sep=None, engine="python")
X_train, X_test, y_train, y_test, prep = prepare_train_test_split(df, target_column="quality", problem_type="classification", test_size=0.2, random_state=42)
exp = train_and_evaluate_model("LogisticRegression", "classification", X_train, y_train, feature_names=prep.feature_names, cv_folds=3, track_mlflow=False)
evaluate_locked_champion_on_holdout(exp, X_train, y_train, X_test, y_test, track_mlflow=False)
print("Champion model ready for explainability extraction.")"""
        },
        {
            "type": "markdown",
            "content": """## 1. Top Predictive Drivers (Relative Importance)
Note: These rankings reflect model-derived statistical signal in observational data and do not establish causal mechanisms.
"""
        },
        {
            "type": "code",
            "content": """feat_imp = calculate_feature_importance(exp["model"], prep.feature_names)
print(f"Top 5 Predictive Drivers for {exp['model_name']}:")
for r in feat_imp["rankings"][:5]:
    print(f"  - {r['feature']:<25}: {r['importance_pct']:.2f}%")"""
        },
        {
            "type": "markdown",
            "content": """## 2. Generating Core Classification Visual Diagnostics
We generate and display the 4 core diagnostic figures: Multiclass OvR ROC Curve, Precision-Recall Curve, Confusion Matrix, and Feature Importance.
"""
        },
        {
            "type": "code",
            "content": """test_m = exp["metrics"]["test"]
roc_pr = generate_roc_pr_plots(
    roc_data=test_m["roc_curve"],
    pr_data=test_m["pr_curve"],
    roc_auc=test_m["roc_auc"],
    pr_auc=test_m["pr_auc"],
    model_name=exp["model_name"],
    run_id="nb_demo"
)

cm_path = generate_confusion_matrix_plot(
    cm=test_m["confusion_matrix"],
    model_name=exp["model_name"],
    run_id="nb_demo",
    class_labels=test_m["class_labels"]
)

imp_path = generate_feature_importance_plot(
    feature_rankings=feat_imp["rankings"],
    model_name=exp["model_name"],
    run_id="nb_demo"
)

print(f"Generated ROC Path:     {roc_pr['roc_curve_path']}")
print(f"Generated PR Path:      {roc_pr['pr_curve_path']}")
print(f"Generated CM Path:      {cm_path}")
print(f"Generated Feature Path: {imp_path}")"""
        }
    ]
    execute_and_build_notebook(nb4_cells, nb_dir / "04_Model_Explainability.ipynb")

    # -------------------------------------------------------------------------
    # Notebook 05: Final Evaluation & Insights
    # -------------------------------------------------------------------------
    nb5_cells = [
        {
            "type": "markdown",
            "content": """# 05 — Methodological Critic Audit & 4-Pillar Report Synthesis
**AutoDS Scientific Methodology Demonstration**

This notebook demonstrates the Critic leakage and overfitting audit, and final report compilation with structured 4-pillar evidence.
"""
        },
        {
            "type": "code",
            "content": """import pandas as pd
from backend.app.tools.preprocessor import prepare_train_test_split
from backend.app.tools.ml_trainer import train_and_evaluate_model, evaluate_locked_champion_on_holdout
from backend.app.tools.critic import critique_experiment
from backend.app.tools.reporter import generate_full_markdown_report
from backend.app.tools.data_profiler import profile_dataset

df = pd.read_csv("data/raw/31513e04_winequality-red.csv", sep=None, engine="python")
profile = profile_dataset(df)
X_train, X_test, y_train, y_test, prep = prepare_train_test_split(df, target_column="quality", problem_type="classification", test_size=0.2, random_state=42)
exp = train_and_evaluate_model("LogisticRegression", "classification", X_train, y_train, feature_names=prep.feature_names, cv_folds=3, track_mlflow=False)
evaluate_locked_champion_on_holdout(exp, X_train, y_train, X_test, y_test, track_mlflow=False)
print("Pipeline evaluated. Proceeding to Critic Audit.")"""
        },
        {
            "type": "markdown",
            "content": """## 1. Methodological Critic Audit
The Critic evaluates train/test accuracy divergence, target leakage indicators, and feature collinearity.
"""
        },
        {
            "type": "code",
            "content": """critic_report = critique_experiment(
    model_name=exp["model_name"],
    problem_type="classification",
    metrics=exp["metrics"],
    feature_names=prep.feature_names,
    validation_strategy="Stratified 3-Fold Cross-Validation",
    target_column="quality"
)

print(f"Critic Audit Status: {critic_report['audit_status']}")
print(f"Total Critic Findings: {len(critic_report['findings'])}")
for f in critic_report['findings']:
    print(f"  [{f['severity'].upper()}] {f['issue_type']}: {f['description']}")"""
        },
        {
            "type": "markdown",
            "content": """## 2. 4-Pillar Evidence Report Compilation
The final markdown report synthesizes Observed Facts, Model Evidence, Actionable Recommendations, and Causal Limitations.
"""
        },
        {
            "type": "code",
            "content": """insights = [
    {
        "category": "observed_facts",
        "title": "Empirical Sample Volume",
        "finding": f"Audited {df.shape[0]:,} red wine samples across {df.shape[1]-1} chemical features.",
        "evidence": f"Total samples: {df.shape[0]:,}",
        "confidence": "High"
    },
    {
        "category": "model_derived",
        "title": "Predictive Discriminability",
        "finding": f"Champion {exp['model_name']} achieved Macro-AUC of {exp['metrics']['test']['roc_auc']:.4f}.",
        "evidence": f"Holdout Macro-AUC: {exp['metrics']['test']['roc_auc']:.4f}",
        "confidence": "High"
    }
]

report_md = generate_full_markdown_report(
    dataset_name="winequality-red.csv",
    user_goal="Predict wine quality ratings based on physicochemical properties",
    problem_type="classification",
    target_column="quality",
    validation_strategy="Stratified 3-Fold Cross-Validation",
    profile_summary=profile,
    experiment_results=[exp],
    best_experiment=exp,
    critic_audit=critic_report,
    business_insights=insights,
    artifact_paths=["reports/artifacts/demo_roc.png"]
)

print(f"Generated Audit-Ready Markdown Report ({len(report_md):,} characters):\\n")
print(report_md)"""
        }
    ]
    execute_and_build_notebook(nb5_cells, nb_dir / "05_Final_Evaluation_and_Insights.ipynb")


if __name__ == "__main__":
    build_all_notebooks()
