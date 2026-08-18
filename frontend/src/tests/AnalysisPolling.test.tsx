import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { AnalysisPage } from '../pages/AnalysisPage';
import { api } from '../services/api';
import { AnalysisStatus, AnalysisRun } from '../types';

// Mock API methods
vi.mock('../services/api', () => ({
  api: {
    getDatasets: vi.fn(),
    getAnalysisRuns: vi.fn(),
    getAnalysis: vi.fn(),
    getAnalysisStatus: vi.fn(),
    createAnalysis: vi.fn(),
  },
}));

describe('AnalysisPage State Synchronization & Polling Flow', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('captures created analysis ID on POST and transitions RUNNING -> COMPLETED via status polling', async () => {
    const mockDatasets = {
      items: [
        {
          id: 'bank-ds-123',
          name: 'Bank_Marketing_UCI',
          file_path: '/path/bank.csv',
          file_type: 'csv',
          size_bytes: 500000,
          row_count: 41188,
          col_count: 21,
          checksum: 'abc123sha',
          created_at: new Date().toISOString(),
        },
      ],
      total: 1,
    };

    const initialRunningAnalysis: AnalysisRun = {
      id: 'analysis-uuid-999',
      dataset_id: 'bank-ds-123',
      user_goal: 'Predict term deposit subscription',
      status: 'RUNNING',
      problem_type: 'classification',
      validation_strategy: 'stratified_kfold',
      plan_json: {},
      critic_findings_json: {},
      created_at: new Date().toISOString(),
    };

    const runningStatus: AnalysisStatus = {
      analysis_id: 'analysis-uuid-999',
      status: 'RUNNING',
      current_stage: 5,
      current_stage_name: 'Candidate Model Training & CV',
      completed_stages: [1, 2, 3, 4],
      total_stages: 9,
      progress_percent: 44,
      elapsed_seconds: 5,
      models_evaluated: ['LogisticRegression'],
      current_model: 'RandomForest',
    };

    const completedStatus: AnalysisStatus = {
      analysis_id: 'analysis-uuid-999',
      status: 'COMPLETED',
      current_stage: 9,
      current_stage_name: 'Evidence-Backed Report Synthesis',
      completed_stages: [1, 2, 3, 4, 5, 6, 7, 8, 9],
      total_stages: 9,
      progress_percent: 100,
      elapsed_seconds: 12,
      models_evaluated: ['LogisticRegression', 'RandomForest', 'LightGBM'],
    };

    const completedAnalysis: AnalysisRun = {
      ...initialRunningAnalysis,
      status: 'COMPLETED',
      completed_at: new Date().toISOString(),
      final_model_id: 'model-champion-1',
    };

    // 1. Initial page load mocks
    (api.getDatasets as any).mockResolvedValue(mockDatasets);
    (api.getAnalysisRuns as any).mockResolvedValue([]);

    // 2. Mock POST /api/analysis
    (api.createAnalysis as any).mockResolvedValue(initialRunningAnalysis);

    // 3. Mock Polling GET /api/analysis/{id}/status
    (api.getAnalysisStatus as any)
      .mockResolvedValueOnce(runningStatus)
      .mockResolvedValue(completedStatus);
    (api.getAnalysis as any).mockResolvedValue(completedAnalysis);

    render(
      <MemoryRouter initialEntries={['/analysis']}>
        <AnalysisPage />
      </MemoryRouter>
    );

    // Wait for dataset to be loaded and option visible
    await waitFor(() => {
      expect(screen.getByRole('option', { name: /Bank_Marketing_UCI/i })).toBeDefined();
    });

    // Click Launch Autonomous Run
    const launchButton = screen.getByRole('button', { name: /Launch Autonomous DS Run/i });
    fireEvent.click(launchButton);

    // Verify POST /api/analysis was called with dataset_id
    await waitFor(() => {
      expect(api.createAnalysis).toHaveBeenCalledWith(
        expect.objectContaining({
          dataset_id: 'bank-ds-123',
        })
      );
    });

    // Verify status transitions to COMPLETED when polling detects completed state
    await waitFor(
      () => {
        expect(screen.getByText(/✓ COMPLETE/i)).toBeDefined();
        expect(screen.getByText(/Autonomous DS Run Complete/i)).toBeDefined();
      },
      { timeout: 4000 }
    );
  });
});
