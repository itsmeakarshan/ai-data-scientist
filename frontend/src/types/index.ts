export interface DatasetProfile {
  id: string;
  dataset_id: string;
  summary_stats: {
    numerical_columns?: Record<string, any>;
    categorical_columns?: Record<string, any>;
    datetime_columns?: Record<string, any>;
    id_columns?: string[];
    constant_columns?: string[];
  };
  missingness_report: {
    total_missing_cells: number;
    total_missing_pct: number;
    missing_by_column: Record<string, number>;
    missing_pct_by_column: Record<string, number>;
    duplicate_rows: number;
    duplicate_pct: number;
  };
  column_types: Record<string, string>;
  correlations: {
    matrix?: Record<string, Record<string, number>>;
    top_positive?: Array<{ feature_1: string; feature_2: string; correlation: number }>;
    top_negative?: Array<{ feature_1: string; feature_2: string; correlation: number }>;
  };
  quality_alerts: Array<{
    type: string;
    column?: string;
    severity: 'info' | 'warning' | 'critical';
    message: string;
    suggested_action: string;
  }>;
  candidate_targets: string[];
  candidate_datetimes: string[];
  inferred_problem_type?: string;
  created_at: string;
}

export interface Dataset {
  id: string;
  name: string;
  file_path: string;
  file_type: string;
  size_bytes: number;
  row_count: number;
  col_count: number;
  checksum: string;
  created_at: string;
  updated_at: string;
  profile?: DatasetProfile;
}

export interface AnalysisRun {
  id: string;
  dataset_id: string;
  user_goal: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  problem_type: string;
  target_column?: string;
  time_column?: string;
  validation_strategy: string;
  plan_json: Record<string, any>;
  critic_findings_json: {
    audit_status?: string;
    requires_iteration?: boolean;
    findings?: Array<{
      issue_type: string;
      severity: string;
      description: string;
      remediation: string;
    }>;
    remediation_actions?: string[];
  };
  final_model_id?: string;
  error_message?: string;
  created_at: string;
  completed_at?: string;
}

export interface ExperimentMetric {
  id: string;
  metric_name: string;
  metric_value: number;
  split_type: string;
}

export interface Experiment {
  id: string;
  analysis_id: string;
  dataset_id: string;
  model_name: string;
  model_family: string;
  hyperparameters: Record<string, any>;
  feature_names: string[];
  preprocessing_config: Record<string, any>;
  validation_strategy: string;
  cv_folds: number;
  train_time_sec: number;
  metrics_json: {
    test?: Record<string, any>;
    train?: Record<string, any>;
    cv_mean?: number;
    cv_std?: number;
  };
  artifacts_path?: string;
  mlflow_run_id?: string;
  status: string;
  created_at: string;
  metrics?: ExperimentMetric[];
}

export interface ModelRecord {
  id: string;
  experiment_id: string;
  name: string;
  task_type: string;
  is_best: boolean;
  artifact_path: string;
  feature_importance_json: {
    rankings?: Array<{ feature: string; importance_pct: number; raw_importance: number }>;
    top_features?: string[];
  };
  shap_summary_json: {
    shap_available?: boolean;
    top_shap_features?: Array<{ feature: string; mean_abs_shap: number }>;
  };
  metrics_json: Record<string, any>;
  created_at: string;
}

export interface Report {
  id: string;
  analysis_id: string;
  title: string;
  summary_markdown: string;
  full_report_markdown: string;
  business_insights_json: {
    insights?: Array<{
      category: string;
      title: string;
      finding: string;
      evidence: string;
      confidence: string;
    }>;
  };
  methodology_json: Record<string, any>;
  artifact_paths: string[];
  created_at: string;
}

export interface ChatMessage {
  id: string;
  session_id: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  tool_calls_json?: Record<string, any>;
  tool_results_json?: Record<string, any>;
  created_at: string;
}

export interface ChatSession {
  id: string;
  dataset_id?: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages?: ChatMessage[];
}

export interface HealthStatus {
  status: string;
  version: string;
  environment: string;
  database_connected: boolean;
  gemini_api_configured: boolean;
  gemini_model: string;
  mlflow_tracking_active: boolean;
  datasets_count: number;
  experiments_count: number;
}
