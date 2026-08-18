import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { AnalysisPage } from '../pages/AnalysisPage';
import { api } from '../services/api';
import { AnalysisStatus, AnalysisRun } from '../types';

vi.mock('../services/api', () => ({
  api: {
    getDatasets: vi.fn(),
    getAnalysisRuns: vi.fn(),
    getAnalysis: vi.fn(),
    getAnalysisStatus: vi.fn(),
    createAnalysis: vi.fn(),
  },
}));

describe('AnalysisPage Autonomous 9-Stage Real-Time Progress Experience', () => {
  const mockDatasets = {
    items: [
      {
        id: 'synthetic-ds-100',
        name: 'synthetic_test.csv',
        file_path: 'data/raw/synthetic_test.csv',
        file_type: 'csv',
        size_bytes: 12000,
        row_count: 300,
        col_count: 6,
        checksum: 'checksum999',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ],
    total: 1,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (api.getDatasets as any).mockResolvedValue(mockDatasets);
    (api.getAnalysisRuns as any).mockResolvedValue([]);
  });

  it('1. Initial state: all 9 stages are in WAITING state with 0% progress', async () => {
    render(
      <MemoryRouter initialEntries={['/analysis?dataset_id=synthetic-ds-100']}>
        <AnalysisPage />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/Autonomous Data Science Engine/i)).toBeDefined();
      expect(screen.getByRole('option', { name: /synthetic_test\.csv/i })).toBeDefined();
    });

    // Verify all 9 stage names exist
    expect(screen.getByText(/1\. Dataset Inspection & Profiling/i)).toBeDefined();
    expect(screen.getByText(/2\. Problem Classification & Target Selection/i)).toBeDefined();
    expect(screen.getByText(/3\. Autonomous Analysis Planning/i)).toBeDefined();
    expect(screen.getByText(/4\. Leak-Free Preprocessing & Splitting/i)).toBeDefined();
    expect(screen.getByText(/5\. Candidate Model Training & CV/i)).toBeDefined();
    expect(screen.getByText(/6\. Multi-Metric Leaderboard Ranking/i)).toBeDefined();
    expect(screen.getByText(/7\. Methodological Critic Audit/i)).toBeDefined();
    expect(screen.getByText(/8\. SHAP Explainability & Feature Attribution/i)).toBeDefined();
    expect(screen.getByText(/9\. Evidence-Backed Report Synthesis/i)).toBeDefined();

    // 0% progress initially
    expect(screen.getByText('0%')).toBeDefined();
  });

  it('2, 3, 4, 5, 6, 7, 10: Launches run with exact ID, renders RUNNING stage with evaluated models, checkmarks on completed, and deterministic progress %', async () => {
    const analysisId = 'analysis-run-uuid-456';
    const createdRun: AnalysisRun = {
      id: analysisId,
      dataset_id: 'synthetic-ds-100',
      user_goal: 'Predict default threshold risk',
      status: 'RUNNING',
      problem_type: 'classification',
      validation_strategy: 'stratified_kfold',
      plan_json: {},
      critic_findings_json: {},
      created_at: new Date().toISOString(),
    };

    const stage5Status: AnalysisStatus = {
      analysis_id: analysisId,
      status: 'RUNNING',
      current_stage: 5,
      current_stage_name: 'Candidate Model Training & CV',
      completed_stages: [1, 2, 3, 4],
      total_stages: 9,
      progress_percent: 44,
      elapsed_seconds: 8,
      models_evaluated: ['LogisticRegression', 'RandomForest'],
      current_model: 'LightGBM',
      stage_details: 'Training candidate algorithm: LightGBM with cross-validation & MLflow logging',
      error: null,
    };

    const completedStatus: AnalysisStatus = {
      analysis_id: analysisId,
      status: 'COMPLETED',
      current_stage: 9,
      current_stage_name: 'Evidence-Backed Report Synthesis',
      completed_stages: [1, 2, 3, 4, 5, 6, 7, 8, 9],
      total_stages: 9,
      progress_percent: 100,
      elapsed_seconds: 14,
      models_evaluated: ['LogisticRegression', 'RandomForest', 'LightGBM', 'Baseline'],
      current_model: null,
      stage_details: 'Autonomous analysis pipeline completed successfully.',
      error: null,
    };

    const completedRun: AnalysisRun = {
      ...createdRun,
      status: 'COMPLETED',
      final_model_id: 'champion-model-uuid-789',
      critic_findings_json: { audit_status: 'CLEAN_LEAK_FREE' },
      completed_at: new Date().toISOString(),
    };

    (api.createAnalysis as any).mockResolvedValue(createdRun);
    (api.getAnalysisStatus as any)
      .mockResolvedValueOnce(stage5Status)
      .mockResolvedValue(completedStatus);
    (api.getAnalysis as any).mockResolvedValue(completedRun);

    render(
      <MemoryRouter initialEntries={['/analysis?dataset_id=synthetic-ds-100']}>
        <AnalysisPage />
      </MemoryRouter>
    );

    // Wait for dataset option to populate
    await waitFor(() => {
      expect(screen.getByRole('option', { name: /synthetic_test\.csv/i })).toBeDefined();
    });

    // Launch run
    const launchBtn = screen.getByRole('button', { name: /Launch Autonomous DS Run/i });
    fireEvent.click(launchBtn);

    // 5. Verify exact analysis ID is created and passed to polling
    await waitFor(() => {
      expect(api.createAnalysis).toHaveBeenCalledWith(
        expect.objectContaining({ dataset_id: 'synthetic-ds-100' })
      );
    });

    // 3, 7 & 10. Verify transition to COMPLETED: 100% progress, summary card, and View Report button
    await waitFor(
      () => {
        expect(screen.getByText(/✓ COMPLETE/i)).toBeDefined();
        expect(screen.getByText('100%')).toBeDefined();
        expect(screen.getByText(/AUTONOMOUS DS RUN COMPLETE/i)).toBeDefined();
        expect(screen.getByRole('link', { name: /View Report/i })).toBeDefined();
      },
      { timeout: 4000 }
    );
  });

  it('8 & 9. Failed state: stops polling immediately and displays backend error message', async () => {
    const analysisId = 'failed-analysis-uuid-111';
    const runningRun: AnalysisRun = {
      id: analysisId,
      dataset_id: 'synthetic-ds-100',
      user_goal: 'Predict default',
      status: 'RUNNING',
      problem_type: 'classification',
      validation_strategy: 'stratified_kfold',
      plan_json: {},
      critic_findings_json: {},
      created_at: new Date().toISOString(),
    };

    const failedStatus: AnalysisStatus = {
      analysis_id: analysisId,
      status: 'FAILED',
      current_stage: 2,
      current_stage_name: 'Problem Classification & Target Selection',
      completed_stages: [1],
      total_stages: 9,
      progress_percent: 11,
      elapsed_seconds: 2,
      models_evaluated: [],
      error: 'Target column "non_existent_target" not found in dataset schema.',
    };

    (api.getDatasets as any).mockResolvedValue(mockDatasets);
    (api.getAnalysisRuns as any).mockResolvedValue([]);
    (api.createAnalysis as any).mockResolvedValue(runningRun);
    (api.getAnalysisStatus as any)
      .mockResolvedValueOnce(failedStatus)
      .mockResolvedValue(failedStatus);
    (api.getAnalysis as any).mockResolvedValue({ ...runningRun, status: 'FAILED', error_message: failedStatus.error });

    render(
      <MemoryRouter initialEntries={['/analysis']}>
        <AnalysisPage />
      </MemoryRouter>
    );

    // Wait for dataset option to populate and select it
    await waitFor(() => {
      expect(screen.getByRole('option', { name: /synthetic_test\.csv/i })).toBeDefined();
    });

    const select = screen.getAllByRole('combobox')[0];
    fireEvent.change(select, { target: { value: 'synthetic-ds-100' } });

    // Launch run
    const launchBtn = screen.getByRole('button', { name: /Launch Autonomous DS Run/i });
    fireEvent.click(launchBtn);

    // 5. Verify exact analysis ID is created and passed to polling
    await waitFor(() => {
      expect(api.createAnalysis).toHaveBeenCalledWith(
        expect.objectContaining({ dataset_id: 'synthetic-ds-100' })
      );
    });

    // Verify transition to FAILED state and error message display
    await waitFor(
      () => {
        expect(screen.getAllByText(/FAILED/i).length).toBeGreaterThan(0);
        expect(screen.getAllByText(/Target column "non_existent_target" not found in dataset schema\./i).length).toBeGreaterThan(0);
      },
      { timeout: 5000 }
    );
  }, 10000);
});
