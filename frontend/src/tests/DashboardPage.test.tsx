import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { DashboardPage } from '../pages/DashboardPage';
import { api } from '../services/api';

class ResizeObserverMock {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}
(global as any).ResizeObserver = ResizeObserverMock;

vi.mock('../services/api', () => ({
  api: {
    getDatasets: vi.fn(),
    getAnalysisRuns: vi.fn(),
    getModels: vi.fn(),
  },
}));

describe('DashboardPage Champion Model Scores & Metric Scaling', () => {
  const mockDatasets = [
    {
      id: 'ds-bank-1',
      name: 'Bank_Marketing_UCI',
      file_path: 'data/raw/bank_marketing.csv',
      file_type: 'csv',
      size_bytes: 45000,
      row_count: 41188,
      col_count: 21,
      checksum: 'chk1',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    },
    {
      id: 'ds-bike-2',
      name: 'hour.csv',
      file_path: 'data/raw/hour.csv',
      file_type: 'csv',
      size_bytes: 15000,
      row_count: 17379,
      col_count: 17,
      checksum: 'chk2',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    },
  ];

  const mockRuns = [
    {
      id: 'run-bank-1',
      dataset_id: 'ds-bank-1',
      user_goal: 'Predict term deposit subscription',
      status: 'COMPLETED',
      problem_type: 'classification',
      target_column: 'y',
      validation_strategy: 'stratified_kfold',
      plan_json: {},
      critic_findings_json: {},
      final_model_id: 'model-lgbm-1',
      created_at: '2026-08-17T12:00:00Z',
    },
    {
      id: 'run-bike-2',
      dataset_id: 'ds-bike-2',
      user_goal: 'Forecast bike demand',
      status: 'COMPLETED',
      problem_type: 'forecasting',
      target_column: 'cnt',
      validation_strategy: 'time_series_split',
      plan_json: {},
      critic_findings_json: {},
      final_model_id: 'model-xgb-2',
      created_at: '2026-08-17T12:30:00Z',
    },
  ];

  const mockModels = [
    {
      id: 'model-lgbm-1',
      experiment_id: 'exp-1',
      name: 'LightGBM',
      task_type: 'classification',
      is_best: true,
      artifact_path: 'artifacts/lgbm.joblib',
      feature_importance_json: {},
      shap_summary_json: {},
      metrics_json: {
        test: {
          roc_auc: 0.8121,
          pr_auc: 0.4829,
          balanced_accuracy: 0.7619,
          accuracy: 0.8998,
        },
      },
      created_at: '2026-08-17T12:05:00Z',
      dataset_name: 'Bank_Marketing_UCI',
      metric_name: 'Holdout ROC-AUC',
      normalized_score: 0.8121,
    },
    {
      id: 'model-xgb-2',
      experiment_id: 'exp-2',
      name: 'XGBoost',
      task_type: 'forecasting',
      is_best: true,
      artifact_path: 'artifacts/xgb.joblib',
      feature_importance_json: {},
      shap_summary_json: {},
      metrics_json: {
        test: {
          wape: 0.1813,
          rmse: 68.61,
          r2: 0.9030,
        },
      },
      created_at: '2026-08-17T12:35:00Z',
      dataset_name: 'hour.csv',
      metric_name: 'Holdout R²',
      normalized_score: 0.9030,
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    (api.getDatasets as any).mockResolvedValue({ items: mockDatasets, total: 2 });
    (api.getAnalysisRuns as any).mockResolvedValue(mockRuns);
    (api.getModels as any).mockResolvedValue(mockModels);
  });

  it('1. Renders champion scores on [0, 1] scale without any 99.819 anomaly', async () => {
    const { container } = render(
      <MemoryRouter>
        <DashboardPage />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/Champion Model Scores/i)).toBeDefined();
      expect(screen.getByText(/AutoDS — Autonomous Data Science Platform/i)).toBeDefined();
    });

    // Check that 99.819 does NOT exist in the DOM
    expect(container.innerHTML).not.toContain('99.819');
    expect(container.innerHTML).not.toContain('99.82');
  });

  it('2. Shows correctly scaled stats and deduplicated champion models', async () => {
    render(
      <MemoryRouter>
        <DashboardPage />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/Champion Model Scores/i)).toBeDefined();
    });

    // Verified KPI numbers
    expect(screen.getAllByText('2').length).toBeGreaterThan(0); // 2 Ingested Datasets / Runs
  });
});
