"""
AutoDS Agent State Architecture
Explicit state tracking across the entire autonomous Data Science lifecycle.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class AgentState:
    analysis_id: str
    dataset_id: str
    dataset_name: str
    file_path: str
    user_goal: str
    
    # Inferred Metadata
    problem_type: str = "classification"  # classification, regression, forecasting, eda
    sub_type: str = "binary_classification"
    target_column: Optional[str] = None
    time_column: Optional[str] = None
    validation_strategy: str = "stratified_kfold"
    
    # Execution Artifacts & Data
    profile_summary: Dict[str, Any] = field(default_factory=dict)
    quality_alerts: List[Dict[str, Any]] = field(default_factory=list)
    analysis_plan: Dict[str, Any] = field(default_factory=dict)
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    best_experiment_id: Optional[str] = None
    best_experiment: Optional[Dict[str, Any]] = None
    
    # Critic & Iteration
    critic_findings: Dict[str, Any] = field(default_factory=dict)
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Explainability & Insights
    explainability: Dict[str, Any] = field(default_factory=dict)
    business_insights: List[Dict[str, Any]] = field(default_factory=list)
    
    # Visuals & Reports
    visual_artifacts: List[str] = field(default_factory=list)
    final_report_markdown: str = ""
    
    # Execution status
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED
    error_message: Optional[str] = None
    current_step: str = "INITIALIZED"
    logs: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    def log(self, message: str):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{ts}] [{self.current_step}] {message}"
        self.logs.append(entry)
