"""
AutoDS Pydantic Schemas
Comprehensive request, response, and domain models for datasets, analyses, experiments, metrics, agents, and queries.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


# ==========================================
# Common / Base Schemas
# ==========================================
class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ==========================================
# Dataset Schemas
# ==========================================
class DatasetSummaryStats(BaseSchema):
    numerical_columns: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    categorical_columns: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    datetime_columns: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    id_columns: List[str] = Field(default_factory=list)
    constant_columns: List[str] = Field(default_factory=list)


class QualityAlert(BaseSchema):
    type: str  # missing_values, high_cardinality, target_imbalance, potential_leakage, constant_feature
    column: Optional[str] = None
    severity: str  # info, warning, critical
    message: str
    suggested_action: str


class DatasetProfileResponse(BaseSchema):
    id: str
    dataset_id: str
    summary_stats: Dict[str, Any]
    missingness_report: Dict[str, Any]
    column_types: Dict[str, str]
    correlations: Dict[str, Any]
    quality_alerts: List[Dict[str, Any]]
    candidate_targets: List[str]
    candidate_datetimes: List[str]
    inferred_problem_type: Optional[str]
    created_at: datetime


class DatasetResponse(BaseSchema):
    id: str
    name: str
    file_path: str
    file_type: str
    size_bytes: int
    row_count: int
    col_count: int
    checksum: str
    created_at: datetime
    updated_at: datetime
    profile: Optional[DatasetProfileResponse] = None


class DatasetListResponse(BaseSchema):
    items: List[DatasetResponse]
    total: int


class SampleRowsResponse(BaseSchema):
    columns: List[str]
    rows: List[Dict[str, Any]]
    total_rows: int


# ==========================================
# Analysis & Agent Schemas
# ==========================================
class AnalysisCreateRequest(BaseSchema):
    dataset_id: str
    user_goal: str
    target_column: Optional[str] = None
    problem_type: Optional[str] = None  # classification, regression, forecasting, eda
    time_column: Optional[str] = None
    validation_strategy: Optional[str] = None


class AnalysisPlanStep(BaseSchema):
    step_number: int
    tool_name: str
    description: str
    status: str = "pending"  # pending, running, completed, failed, skipped
    result_summary: Optional[str] = None


class AnalysisPlan(BaseSchema):
    problem_type: str
    target_column: Optional[str] = None
    time_column: Optional[str] = None
    validation_strategy: str
    candidate_models: List[str]
    feature_engineering_steps: List[str] = Field(default_factory=list)
    steps: List[AnalysisPlanStep] = Field(default_factory=list)


class CriticFinding(BaseSchema):
    issue_type: str  # data_leakage, severe_overfitting, improper_validation, class_imbalance, calibration_error
    severity: str  # warning, critical
    description: str
    affected_components: List[str]
    remediation: str
    action_taken: Optional[str] = None


class AnalysisRunResponse(BaseSchema):
    id: str
    dataset_id: str
    user_goal: str
    status: str
    problem_type: str
    target_column: Optional[str]
    time_column: Optional[str]
    validation_strategy: str
    plan_json: Dict[str, Any]
    critic_findings_json: Dict[str, Any]
    final_model_id: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]


# ==========================================
# Experiment & Metrics Schemas
# ==========================================
class ExperimentMetricResponse(BaseSchema):
    id: str
    metric_name: str
    metric_value: float
    split_type: str


class ExperimentResponse(BaseSchema):
    id: str
    analysis_id: str
    dataset_id: str
    model_name: str
    model_family: str
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    preprocessing_config: Dict[str, Any]
    validation_strategy: str
    cv_folds: int
    train_time_sec: float
    metrics_json: Dict[str, Any]
    artifacts_path: Optional[str]
    mlflow_run_id: Optional[str]
    status: str
    created_at: datetime
    metrics: List[ExperimentMetricResponse] = Field(default_factory=list)


class ExperimentComparisonResponse(BaseSchema):
    experiments: List[ExperimentResponse]
    best_experiment_id: Optional[str]
    primary_metric: str
    comparison_table: List[Dict[str, Any]]


# ==========================================
# Model Schemas
# ==========================================
class ModelRecordResponse(BaseSchema):
    id: str
    experiment_id: str
    name: str
    task_type: str
    is_best: bool
    artifact_path: str
    feature_importance_json: Dict[str, Any]
    shap_summary_json: Dict[str, Any]
    metrics_json: Dict[str, Any]
    created_at: datetime


# ==========================================
# Report Schemas
# ==========================================
class BusinessInsight(BaseSchema):
    category: str  # observed_facts, model_derived, agent_interpretation, business_recommendation
    title: str
    finding: str
    evidence: str
    confidence: str


class ReportResponse(BaseSchema):
    id: str
    analysis_id: str
    title: str
    summary_markdown: str
    full_report_markdown: str
    business_insights_json: Dict[str, Any]
    methodology_json: Dict[str, Any]
    artifact_paths: List[str]
    created_at: datetime


# ==========================================
# Safe SQL / Data Query Schemas
# ==========================================
class QueryRequest(BaseSchema):
    dataset_id: str
    sql_query: str
    limit: Optional[int] = 500


class QueryResponse(BaseSchema):
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: float
    sql_executed: str


# ==========================================
# Chat Schemas
# ==========================================
class ChatMessageCreate(BaseSchema):
    session_id: Optional[str] = None
    dataset_id: Optional[str] = None
    content: str


class ChatMessageResponse(BaseSchema):
    id: str
    session_id: str
    role: str
    content: str
    tool_calls_json: Optional[Dict[str, Any]] = None
    tool_results_json: Optional[Dict[str, Any]] = None
    created_at: datetime


class ChatSessionResponse(BaseSchema):
    id: str
    dataset_id: Optional[str]
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessageResponse] = Field(default_factory=list)


# ==========================================
# Health Check Schemas
# ==========================================
class HealthStatusResponse(BaseSchema):
    status: str
    version: str
    environment: str
    database_connected: bool
    gemini_api_configured: bool
    gemini_model: str
    mlflow_tracking_active: bool
    datasets_count: int
    experiments_count: int
