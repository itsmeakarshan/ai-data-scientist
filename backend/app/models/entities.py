"""
AutoDS SQLAlchemy ORM Database Models
Defines relational schemas for datasets, profiles, analyses, experiments, metrics, models, reports, and chats.
"""

from datetime import datetime, timezone
import uuid
from typing import Optional, List
from sqlalchemy import (
    String, Text, Integer, Float, Boolean, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.core.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    col_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    profile: Mapped[Optional["DatasetProfile"]] = relationship("DatasetProfile", back_populates="dataset", uselist=False, cascade="all, delete-orphan")
    analysis_runs: Mapped[List["AnalysisRun"]] = relationship("AnalysisRun", back_populates="dataset", cascade="all, delete-orphan")
    experiments: Mapped[List["Experiment"]] = relationship("Experiment", back_populates="dataset", cascade="all, delete-orphan")


class DatasetProfile(Base):
    __tablename__ = "dataset_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), unique=True, nullable=False)
    summary_stats: Mapped[dict] = mapped_column(JSON, default=dict)
    missingness_report: Mapped[dict] = mapped_column(JSON, default=dict)
    column_types: Mapped[dict] = mapped_column(JSON, default=dict)
    correlations: Mapped[dict] = mapped_column(JSON, default=dict)
    quality_alerts: Mapped[list] = mapped_column(JSON, default=list)
    candidate_targets: Mapped[list] = mapped_column(JSON, default=list)
    candidate_datetimes: Mapped[list] = mapped_column(JSON, default=list)
    inferred_problem_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    # Relationships
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="profile")


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    user_goal: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="PENDING")  # PENDING, RUNNING, COMPLETED, FAILED, CRITIQUE_REVISION
    problem_type: Mapped[str] = mapped_column(String(50), nullable=False, default="classification")
    target_column: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    time_column: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    validation_strategy: Mapped[str] = mapped_column(String(100), default="stratified_kfold")
    plan_json: Mapped[dict] = mapped_column(JSON, default=dict)
    critic_findings_json: Mapped[dict] = mapped_column(JSON, default=dict)
    final_model_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="analysis_runs")
    experiments: Mapped[List["Experiment"]] = relationship("Experiment", back_populates="analysis_run", cascade="all, delete-orphan")
    report: Mapped[Optional["Report"]] = relationship("Report", back_populates="analysis_run", uselist=False, cascade="all, delete-orphan")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    analysis_id: Mapped[str] = mapped_column(String(36), ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_family: Mapped[str] = mapped_column(String(100), nullable=False)  # tree, linear, ensemble, naive
    hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)
    feature_names: Mapped[list] = mapped_column(JSON, default=list)
    preprocessing_config: Mapped[dict] = mapped_column(JSON, default=dict)
    validation_strategy: Mapped[str] = mapped_column(String(100), nullable=False)
    cv_folds: Mapped[int] = mapped_column(Integer, default=5)
    train_time_sec: Mapped[float] = mapped_column(Float, default=0.0)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    artifacts_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="COMPLETED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    # Relationships
    analysis_run: Mapped["AnalysisRun"] = relationship("AnalysisRun", back_populates="experiments")
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="experiments")
    metrics: Mapped[List["ExperimentMetric"]] = relationship("ExperimentMetric", back_populates="experiment", cascade="all, delete-orphan")
    model_record: Mapped[Optional["ModelRecord"]] = relationship("ModelRecord", back_populates="experiment", uselist=False, cascade="all, delete-orphan")


class ExperimentMetric(Base):
    __tablename__ = "experiment_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    experiment_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)  # roc_auc, f1, pr_auc, accuracy, rmse, mae, r2
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    split_type: Mapped[str] = mapped_column(String(50), default="test")  # train, val, test, cv_mean, cv_std
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    # Relationships
    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="metrics")


class ModelRecord(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    experiment_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiments.id", ondelete="CASCADE"), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    is_best: Mapped[bool] = mapped_column(Boolean, default=False)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    feature_importance_json: Mapped[dict] = mapped_column(JSON, default=dict)
    shap_summary_json: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    # Relationships
    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="model_record")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    analysis_id: Mapped[str] = mapped_column(String(36), ForeignKey("analysis_runs.id", ondelete="CASCADE"), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    summary_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    full_report_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    business_insights_json: Mapped[dict] = mapped_column(JSON, default=dict)
    methodology_json: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_paths: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    # Relationships
    analysis_run: Mapped["AnalysisRun"] = relationship("AnalysisRun", back_populates="report")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    dataset_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    title: Mapped[str] = mapped_column(String(255), default="Data Science Session")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    messages: Mapped[List["ChatMessage"]] = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    session_id: Mapped[str] = mapped_column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False)  # user, assistant, system, tool
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tool_calls_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    tool_results_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    # Relationships
    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")


