import {
  AnalysisRun,
  AnalysisStatus,
  ChatMessage,
  ChatSession,
  Dataset,
  Experiment,
  HealthStatus,
  ModelRecord,
  Report,
  WorkflowProgressResponse,
} from '../types';

const API_BASE = '/api';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    let errorDetail = res.statusText;
    try {
      const errObj = await res.json();
      errorDetail = errObj.detail || JSON.stringify(errObj);
    } catch {
      // ignore
    }
    throw new Error(`API Error (${res.status}): ${errorDetail}`);
  }
  return res.json();
}

export const api = {
  // Health
  getHealth: () => fetchJson<HealthStatus>(`${API_BASE}/health`),

  // Datasets
  getDatasets: () => fetchJson<{ items: Dataset[]; total: number }>(`${API_BASE}/datasets`),
  getDataset: (id: string) => fetchJson<Dataset>(`${API_BASE}/datasets/${id}`),
  getDatasetSample: (id: string, limit = 50) =>
    fetchJson<{ columns: string[]; rows: Record<string, any>[]; total_rows: number }>(
      `${API_BASE}/datasets/${id}/sample?limit=${limit}`
    ),
  uploadDataset: async (file: File): Promise<Dataset> => {
    const formData = new FormData();
    formData.append('file', file);
    return fetchJson<Dataset>(`${API_BASE}/datasets/upload`, {
      method: 'POST',
      body: formData,
    });
  },
  deleteDataset: (id: string) =>
    fetchJson<{ message: string; id: string }>(`${API_BASE}/datasets/${id}`, {
      method: 'DELETE',
    }),

  // Analysis
  createAnalysis: (payload: {
    dataset_id: string;
    user_goal: string;
    target_column?: string;
    problem_type?: string;
    time_column?: string;
  }) =>
    fetchJson<AnalysisRun>(`${API_BASE}/analysis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  getAnalysisRuns: () => fetchJson<AnalysisRun[]>(`${API_BASE}/analysis`),
  getAnalysis: (id: string) => fetchJson<AnalysisRun>(`${API_BASE}/analysis/${id}`),
  getAnalysisStatus: (id: string) => fetchJson<AnalysisStatus>(`${API_BASE}/analysis/${id}/status`),
  getAnalysisProgress: (id: string) => fetchJson<WorkflowProgressResponse>(`${API_BASE}/analysis/${id}/progress`),

  // Experiments
  getExperiments: (params?: { analysis_id?: string; dataset_id?: string }) => {
    const query = new URLSearchParams();
    if (params?.analysis_id) query.append('analysis_id', params.analysis_id);
    if (params?.dataset_id) query.append('dataset_id', params.dataset_id);
    return fetchJson<Experiment[]>(`${API_BASE}/experiments?${query.toString()}`);
  },
  getExperiment: (id: string) => fetchJson<Experiment>(`${API_BASE}/experiments/${id}`),
  compareExperiments: (analysisId: string) =>
    fetchJson<{
      experiments: Experiment[];
      best_experiment_id: string;
      primary_metric: string;
      comparison_table: any[];
    }>(`${API_BASE}/experiments/compare/${analysisId}`),

  // Models
  getModels: (params?: { is_best?: boolean; task_type?: string; latest_per_dataset?: boolean }) => {
    const query = new URLSearchParams();
    if (params?.is_best !== undefined) query.append('is_best', String(params.is_best));
    if (params?.task_type) query.append('task_type', params.task_type);
    if (params?.latest_per_dataset !== undefined) query.append('latest_per_dataset', String(params.latest_per_dataset));
    return fetchJson<ModelRecord[]>(`${API_BASE}/models?${query.toString()}`);
  },
  getModel: (id: string) => fetchJson<ModelRecord>(`${API_BASE}/models/${id}`),

  // Reports
  getReports: () => fetchJson<Report[]>(`${API_BASE}/reports`),
  getReport: (idOrAnalysisId: string) => fetchJson<Report>(`${API_BASE}/reports/${idOrAnalysisId}`),
  deleteReport: (id: string) =>
    fetchJson<{ message: string }>(`${API_BASE}/reports/${id}`, {
      method: 'DELETE',
    }),

  // Agent Chat
  sendChatMessage: (payload: {
    session_id?: string;
    dataset_id?: string;
    analysis_id?: string;
    report_id?: string;
    comparison_analysis_id?: string;
    content: string;
  }) =>
    fetchJson<ChatMessage>(`${API_BASE}/agent/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  getAgentContext: (params?: {
    analysis_id?: string;
    report_id?: string;
    dataset_id?: string;
    comparison_analysis_id?: string;
  }) => {
    const query = new URLSearchParams();
    if (params?.analysis_id) query.append('analysis_id', params.analysis_id);
    if (params?.report_id) query.append('report_id', params.report_id);
    if (params?.dataset_id) query.append('dataset_id', params.dataset_id);
    if (params?.comparison_analysis_id) query.append('comparison_analysis_id', params.comparison_analysis_id);
    return fetchJson<Record<string, any>>(`${API_BASE}/agent/context?${query.toString()}`);
  },
  getChatSessions: () => fetchJson<ChatSession[]>(`${API_BASE}/agent/sessions`),
  getChatSession: (id: string) => fetchJson<ChatSession>(`${API_BASE}/agent/sessions/${id}`),

  // Safe SQL Query Tool
  runQuery: (payload: { dataset_id: string; sql_query: string; limit?: number }) =>
    fetchJson<{
      columns: string[];
      rows: Record<string, any>[];
      row_count: number;
      execution_time_ms: number;
      sql_executed: string;
    }>(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
};

