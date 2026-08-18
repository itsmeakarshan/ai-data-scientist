"""
AutoDS Deterministic Tools Export
"""

from backend.app.tools.critic import (
    critique_experiment,
)
from backend.app.tools.data_profiler import (
    is_candidate_datetime,
    profile_dataset,
)
from backend.app.tools.dataset_inspector import (
    compute_file_sha256,
    detect_csv_delimiter,
    load_dataset_as_dataframe,
)
from backend.app.tools.evaluator import (
    evaluate_classification,
    evaluate_forecasting,
    evaluate_regression,
)
from backend.app.tools.explainability import (
    calculate_feature_importance,
    compute_shap_explanations,
)
from backend.app.tools.ml_trainer import (
    get_model_instance,
    train_and_evaluate_model,
)
from backend.app.tools.preprocessor import (
    PreprocessingArtifacts,
    clean_dataframe,
    create_forecasting_features,
    prepare_train_test_split,
)
from backend.app.tools.problem_classifier import (
    classify_problem_type,
)
from backend.app.tools.quality_detector import (
    detect_data_quality,
)
from backend.app.tools.reporter import (
    generate_full_markdown_report,
)
from backend.app.tools.safe_query import (
    execute_safe_sql_query,
)
from backend.app.tools.visualizer import (
    generate_actual_vs_predicted_plot,
    generate_confusion_matrix_plot,
    generate_correlation_heatmap,
    generate_feature_importance_plot,
    generate_residual_plot,
    generate_roc_pr_plots,
)

__all__ = [
    "compute_file_sha256",
    "detect_csv_delimiter",
    "load_dataset_as_dataframe",
    "profile_dataset",
    "is_candidate_datetime",
    "detect_data_quality",
    "classify_problem_type",
    "clean_dataframe",
    "create_forecasting_features",
    "prepare_train_test_split",
    "PreprocessingArtifacts",
    "get_model_instance",
    "train_and_evaluate_model",
    "evaluate_classification",
    "evaluate_regression",
    "evaluate_forecasting",
    "calculate_feature_importance",
    "compute_shap_explanations",
    "critique_experiment",
    "execute_safe_sql_query",
    "generate_roc_pr_plots",
    "generate_confusion_matrix_plot",
    "generate_feature_importance_plot",
    "generate_actual_vs_predicted_plot",
    "generate_residual_plot",
    "generate_correlation_heatmap",
    "generate_full_markdown_report",
]
