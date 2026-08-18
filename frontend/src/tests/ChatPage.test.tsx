import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { ChatPage } from '../pages/ChatPage';
import { ReportViewer } from '../components/ReportViewer';
import { api } from '../services/api';

vi.mock('../services/api', () => ({
  api: {
    getDatasets: vi.fn(),
    getReports: vi.fn(),
    getAgentContext: vi.fn(),
    sendChatMessage: vi.fn(),
    getChatSessions: vi.fn(),
  },
}));

describe('AutoDS Grounded AI Agent Chat Experience', () => {
  const mockDataset = {
    id: 'ds-bank-1',
    name: 'Bank_Marketing_UCI',
    file_path: 'data/raw/bank_marketing.csv',
    file_type: 'csv',
    size_bytes: 45000,
    row_count: 41188,
    col_count: 21,
    checksum: 'chk123',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };

  const mockReport = {
    id: 'rep-bank-1',
    analysis_id: 'run-bank-1',
    dataset_id: 'ds-bank-1',
    title: 'AutoDS Executive Report: Bank Marketing',
    summary_markdown: 'LightGBM champion model.',
    full_report_markdown: '## 8. Model Limitations & Operational Risk Analysis\n1. **Class Asymmetry**: Skewed target.',
    business_insights_json: { insights: [] },
    methodology_json: {},
    artifact_paths: [],
    created_at: new Date().toISOString(),
  };

  const mockContext = {
    has_context: true,
    context_type: 'analysis',
    dataset: {
      id: 'ds-bank-1',
      name: 'Bank_Marketing_UCI',
      row_count: 41188,
      col_count: 21,
      total_missing_pct: 0.0,
      column_types: { age: 'numeric', y: 'categorical' },
    },
    analysis: {
      id: 'run-bank-1',
      problem_type: 'classification',
      target_column: 'y',
    },
    champion_model: {
      id: 'model-lgbm-1',
      name: 'LightGBM',
      task_type: 'classification',
      holdout_metrics: {
        roc_auc: 0.7937,
        positive_recall: 0.6853,
      },
    },
    threshold_analysis: {
      selected_threshold: 0.10,
      recall_gain_pts: 43.4,
      tp_gain_over_default: 403,
    },
    critic_audit: {
      audit_status: 'PASSED (Remediated)',
      leakage_remediated: true,
      remediated_features: ['duration'],
    },
    leaderboard: [
      { model_name: 'LightGBM', cv_mean: 0.7977, status: 'Champion' },
      { model_name: 'XGBoost', cv_mean: 0.7912, status: 'Candidate' },
    ],
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (api.getDatasets as any).mockResolvedValue({ items: [mockDataset], total: 1 });
    (api.getReports as any).mockResolvedValue([mockReport]);
    (api.getAgentContext as any).mockResolvedValue(mockContext);
    (api.sendChatMessage as any).mockResolvedValue({
      id: 'msg-1',
      session_id: 'session-1',
      role: 'assistant',
      content: 'LightGBM won with CV ROC-AUC of 0.7977.\n\n> [Evidence: Model Leaderboard]',
      created_at: new Date().toISOString(),
    });
  });

  it('1. Renders context indicator with dataset name, analysis ID, and champion model', async () => {
    render(
      <MemoryRouter initialEntries={['/chat?analysisId=run-bank-1&reportId=rep-bank-1']}>
        <ChatPage />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/AutoDS Grounded Data Science Agent/i)).toBeDefined();
      expect(screen.getAllByText(/Bank_Marketing_UCI/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/LightGBM/i).length).toBeGreaterThan(0);
    });

    expect(screen.getByText(/Grounded in Computed Evidence/i)).toBeDefined();
    expect(screen.getAllByText(/Why did this model win\?/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Explain the threshold/i).length).toBeGreaterThan(0);
  });

  it('2. Clicking a suggested question sends the query and renders evidence badge', async () => {
    render(
      <MemoryRouter initialEntries={['/chat?analysisId=run-bank-1']}>
        <ChatPage />
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByText(/Why did this model win\?/i)).toBeDefined();
    });

    const chip = screen.getByText(/Why did this model win\?/i);
    fireEvent.click(chip);

    await waitFor(() => {
      expect(api.sendChatMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          content: 'Why did this model win?',
          analysis_id: 'run-bank-1',
        })
      );
    });

    await waitFor(() => {
      expect(screen.getByText(/Model Leaderboard/i)).toBeDefined();
      expect(screen.getByText(/Evidence Source/i)).toBeDefined();
    });
  });

  it('3. ReportViewer contains Ask AutoDS Agent button', () => {
    render(
      <MemoryRouter>
        <ReportViewer report={mockReport as any} />
      </MemoryRouter>
    );

    const askBtn = screen.getByRole('button', { name: /Ask AutoDS Agent/i });
    expect(askBtn).toBeDefined();
  });
});
