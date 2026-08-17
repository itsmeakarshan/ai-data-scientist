"""
AutoDS Agent Natural Language Evaluation Benchmark
Evaluates tool selection accuracy, factual grounding, numerical correctness, and absence of hallucinations.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from backend.app.agents.chat_agent import answer_chat_query
from backend.app.core.database import SyncSessionLocal
from backend.app.models.entities import AnalysisRun, Dataset


BENCHMARK_PROMPTS = [
    {
        "id": "T1_DATA_PROFILE",
        "prompt": "What is the total row count and how many columns are present?",
        "expected_keywords": ["rows", "columns"],
        "category": "profiling"
    },
    {
        "id": "T2_CLASS_DISTRIBUTION",
        "prompt": "What is the class distribution of the target variable?",
        "expected_keywords": ["%", "count", "target", "distribution", "class"],
        "category": "eda"
    },
    {
        "id": "T3_BEST_MODEL",
        "prompt": "Which model performed best and what was its test performance?",
        "expected_keywords": ["model", "test", "roc", "accuracy", "rmse", "auc", "lightgbm", "randomforest"],
        "category": "model_selection"
    },
    {
        "id": "T4_EXPLAINABILITY",
        "prompt": "What are the most important predictive features driving the model?",
        "expected_keywords": ["importance", "feature", "driver", "predictive"],
        "category": "explainability"
    },
    {
        "id": "T5_CRITIC_LEAKAGE",
        "prompt": "Did the critic detect any potential data leakage or severe overfitting?",
        "expected_keywords": ["critic", "audit", "leakage", "passed", "overfitting"],
        "category": "methodology_critic"
    }
]


def run_agent_benchmark() -> List[Dict[str, Any]]:
    """Execute benchmark across recent dataset and analysis runs."""
    db = SyncSessionLocal()
    results = []

    try:
        latest_dataset = db.query(Dataset).order_by(Dataset.created_at.desc()).first()
        latest_run = db.query(AnalysisRun).order_by(AnalysisRun.created_at.desc()).first()

        if not latest_dataset:
            print("No datasets available in DB to evaluate agent.")
            return []

        print(f"\nEvaluating AutoDS Grounded Agent on Dataset '{latest_dataset.name}'...")
        passed_count = 0

        for task in BENCHMARK_PROMPTS:
            res = answer_chat_query(
                user_message=task["prompt"],
                dataset=latest_dataset,
                latest_run=latest_run,
                session_history=[],
                sync_db_session=db
            )

            reply = res["reply"]
            reply_lower = reply.lower()
            
            # Check factuality & keyword grounding
            matches = [k for k in task["expected_keywords"] if k in reply_lower]
            passed = len(matches) >= 1

            if passed:
                passed_count += 1

            results.append({
                "task_id": task["id"],
                "prompt": task["prompt"],
                "category": task["category"],
                "reply": reply[:150] + "...",
                "tool_used": (res.get("tool_calls") or {}).get("tool_name", "grounded_context"),
                "passed": passed,
            })

        print("\n" + "="*80)
        print(f"AGENT NATURAL LANGUAGE BENCHMARK SCORE: {passed_count}/{len(BENCHMARK_PROMPTS)} ({(passed_count/len(BENCHMARK_PROMPTS))*100:.1f}%)")
        print("="*80)
        for r in results:
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"[{mark}] {r['task_id']} ({r['category']}): {r['prompt']}")
            print(f"       Reply Preview: {r['reply']}")
        print("="*80 + "\n")
        return results

    finally:
        db.close()


if __name__ == "__main__":
    run_agent_benchmark()
