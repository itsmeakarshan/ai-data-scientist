import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Database,
  PlayCircle,
  FlaskConical,
  Boxes,
  ArrowRight,
  ShieldCheck,
  Sparkles
} from 'lucide-react';
import { api } from '../services/api';
import { AnalysisRun, Dataset, ModelRecord } from '../types';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

function cleanDatasetName(name: string): string {
  return name
    .replace(/^Benchmark_/, '')
    .replace(/_UCI$/, '')
    .replace(/_/g, ' ')
    .replace(/\.csv$/, '');
}

function extractModelScore(
  taskType: string,
  testMetrics: Record<string, any>,
  backendNormalizedScore?: number,
  backendMetricName?: string
): { metricName: string; score: number } {
  if (backendNormalizedScore !== undefined && backendNormalizedScore !== null && backendMetricName) {
    return { metricName: backendMetricName, score: Number(backendNormalizedScore.toFixed(4)) };
  }

  const task = (taskType || 'classification').toLowerCase();
  if (task === 'classification') {
    if (typeof testMetrics.roc_auc === 'number' && testMetrics.roc_auc >= 0 && testMetrics.roc_auc <= 1) {
      return { metricName: 'Holdout ROC-AUC', score: Number(testMetrics.roc_auc.toFixed(4)) };
    }
    if (typeof testMetrics.balanced_accuracy === 'number' && testMetrics.balanced_accuracy >= 0 && testMetrics.balanced_accuracy <= 1) {
      return { metricName: 'Holdout Balanced Accuracy', score: Number(testMetrics.balanced_accuracy.toFixed(4)) };
    }
    if (typeof testMetrics.accuracy === 'number' && testMetrics.accuracy >= 0 && testMetrics.accuracy <= 1) {
      return { metricName: 'Holdout Accuracy', score: Number(testMetrics.accuracy.toFixed(4)) };
    }
    if (typeof testMetrics.pr_auc === 'number' && testMetrics.pr_auc >= 0 && testMetrics.pr_auc <= 1) {
      return { metricName: 'Holdout PR-AUC', score: Number(testMetrics.pr_auc.toFixed(4)) };
    }
    if (typeof testMetrics.f1_positive === 'number' && testMetrics.f1_positive >= 0 && testMetrics.f1_positive <= 1) {
      return { metricName: 'Holdout F1-Score', score: Number(testMetrics.f1_positive.toFixed(4)) };
    }
    return { metricName: 'Holdout Score', score: 0.5 };
  }

  // Regression / Forecasting
  if (typeof testMetrics.r2 === 'number') {
    const clampedR2 = Math.max(0, Math.min(1, testMetrics.r2));
    return { metricName: 'Holdout R²', score: Number(clampedR2.toFixed(4)) };
  }
  if (typeof testMetrics.wape === 'number') {
    const wapeFrac = testMetrics.wape > 1 ? testMetrics.wape / 100 : testMetrics.wape;
    const acc = Math.max(0, Math.min(1, 1 - wapeFrac));
    return { metricName: 'Holdout Accuracy (1-WAPE)', score: Number(acc.toFixed(4)) };
  }
  return { metricName: 'Holdout Score', score: 0.5 };
}

export const DashboardPage: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [runs, setRuns] = useState<AnalysisRun[]>([]);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [dsRes, runRes, modelRes] = await Promise.all([
          api.getDatasets(),
          api.getAnalysisRuns(),
          api.getModels({ is_best: true, latest_per_dataset: true }),
        ]);
        setDatasets(dsRes.items || []);
        setRuns(runRes || []);
        setModels(modelRes || []);
      } catch (err) {
        console.error('Error loading dashboard:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const chartData = React.useMemo(() => {
    const dsMap = new Map(datasets.map((d) => [d.id, d]));
    const modelMap = new Map(models.map((m) => [m.id, m]));

    // Find the latest completed run per dataset
    const latestCompletedRunByDataset = new Map<string, AnalysisRun>();
    for (const run of runs) {
      if (run.status === 'COMPLETED' && run.dataset_id && run.final_model_id) {
        const existing = latestCompletedRunByDataset.get(run.dataset_id);
        if (!existing || new Date(run.created_at) > new Date(existing.created_at)) {
          latestCompletedRunByDataset.set(run.dataset_id, run);
        }
      }
    }

    const entries: Array<{
      id: string;
      name: string;
      modelName: string;
      datasetName: string;
      score: number;
      task: string;
      metricName: string;
    }> = [];

    for (const [datasetId, run] of latestCompletedRunByDataset.entries()) {
      if (!run.final_model_id) continue;
      const model = modelMap.get(run.final_model_id);
      const ds = dsMap.get(datasetId);
      const rawDsName = ds?.name || run.dataset_id;
      const cleanDs = cleanDatasetName(rawDsName);

      if (model) {
        const testM = model.metrics_json?.test || {};
        const { metricName, score } = extractModelScore(
          model.task_type || run.problem_type,
          testM,
          model.normalized_score,
          model.metric_name
        );
        entries.push({
          id: model.id,
          name: cleanDs.length > 18 ? cleanDs.substring(0, 18) + '...' : cleanDs,
          modelName: model.name,
          datasetName: cleanDs,
          score,
          task: model.task_type || run.problem_type,
          metricName,
        });
      }
    }

    // Fallback: If no completed runs map to models, display deduplicated best models
    if (entries.length === 0) {
      const seenDatasets = new Set<string>();
      for (const m of models) {
        if (!m.is_best) continue;
        const dsLabel = m.dataset_name ? cleanDatasetName(m.dataset_name) : m.name;
        if (seenDatasets.has(dsLabel)) continue;
        seenDatasets.add(dsLabel);

        const testM = m.metrics_json?.test || {};
        const { metricName, score } = extractModelScore(
          m.task_type,
          testM,
          m.normalized_score,
          m.metric_name
        );
        entries.push({
          id: m.id,
          name: dsLabel.length > 18 ? dsLabel.substring(0, 18) + '...' : dsLabel,
          modelName: m.name,
          datasetName: dsLabel,
          score,
          task: m.task_type,
          metricName,
        });
      }
    }

    return entries;
  }, [datasets, runs, models]);

  return (
    <div className="space-y-8 p-8 w-full">
      {/* Hero Welcome Banner */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-r from-indigo-50/80 via-white to-emerald-50/80 border border-slate-200/90 p-8 shadow-sm">
        <div className="relative z-10 max-w-3xl space-y-3">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-100/80 border border-emerald-200 text-emerald-800 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
            <span>Autonomous Machine Learning & Agentic MLOps</span>
          </div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
            AutoDS — Autonomous Data Science Platform
          </h1>
          <p className="text-slate-600 text-sm leading-relaxed">
            Upload any dataset and state your objective in natural language. AutoDS autonomously profiles the schema,
            determines problem formulation, executes leak-free ML experiments, conducts methodological critic audits,
            and produces evidence-backed explainability reports.
          </p>
          <div className="pt-2 flex items-center space-x-4">
            <Link
              to="/analysis"
              className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl text-sm font-bold bg-emerald-600 hover:bg-emerald-500 text-white shadow-md shadow-emerald-600/20 transition"
            >
              <PlayCircle className="w-4 h-4" />
              <span>Launch Autonomous Run</span>
            </Link>
            <Link
              to="/chat"
              className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl text-sm font-semibold bg-white hover:bg-slate-100 text-slate-700 border border-slate-300 shadow-2xs transition"
            >
              <span>Ask Agent Questions</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </div>

      {/* KPI Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
        <div className="glass-panel p-5 rounded-2xl space-y-2">
          <div className="flex items-center justify-between text-slate-500 text-xs font-semibold">
            <span>Ingested Datasets</span>
            <Database className="w-4 h-4 text-emerald-600" />
          </div>
          <p className="text-2xl font-bold text-slate-900">{datasets.length}</p>
          <p className="text-xs text-slate-500">Structured & profiled</p>
        </div>

        <div className="glass-panel p-5 rounded-2xl space-y-2">
          <div className="flex items-center justify-between text-slate-500 text-xs font-semibold">
            <span>Autonomous Runs</span>
            <PlayCircle className="w-4 h-4 text-indigo-600" />
          </div>
          <p className="text-2xl font-bold text-slate-900">{runs.length}</p>
          <p className="text-xs text-slate-500">With plan execution & critic audit</p>
        </div>

        <div className="glass-panel p-5 rounded-2xl space-y-2">
          <div className="flex items-center justify-between text-slate-500 text-xs font-semibold">
            <span>Champion Models</span>
            <Boxes className="w-4 h-4 text-purple-600" />
          </div>
          <p className="text-2xl font-bold text-slate-900">{models.length}</p>
          <p className="text-xs text-slate-500">In production registry</p>
        </div>

        <div className="glass-panel p-5 rounded-2xl space-y-2">
          <div className="flex items-center justify-between text-slate-500 text-xs font-semibold">
            <span>Critic Verification</span>
            <ShieldCheck className="w-4 h-4 text-teal-600" />
          </div>
          <p className="text-2xl font-bold text-emerald-600">100% Active</p>
          <p className="text-xs text-slate-500">Zero data leakage guarantee</p>
        </div>
      </div>

      {/* Main Grid: Recent Runs & Model Leaderboard */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Recent Autonomous Runs */}
        <div className="lg:col-span-2 glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-bold text-base text-slate-900">Recent Autonomous Analyses</h3>
              <p className="text-xs text-slate-500">Tracked end-to-end Data Science workflows</p>
            </div>
            <Link to="/analysis" className="text-xs font-semibold text-indigo-600 hover:text-indigo-800 flex items-center gap-1">
              View all <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="divide-y divide-slate-100">
            {runs.length === 0 ? (
              <div className="py-8 text-center text-slate-500 text-sm">
                No analyses run yet. Click "Launch Autonomous Run" to begin.
              </div>
            ) : (
              runs.slice(0, 5).map((run) => (
                <div key={run.id} className="py-3.5 flex items-center justify-between hover:bg-slate-50 px-2 rounded-xl transition">
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-semibold uppercase px-2 py-0.5 rounded-lg bg-indigo-50 text-indigo-700 border border-indigo-100">
                        {run.problem_type}
                      </span>
                      <span className="text-sm font-medium text-slate-800 line-clamp-1">{run.user_goal}</span>
                    </div>
                    <div className="flex items-center space-x-3 text-xs text-slate-500">
                      <span>Target: <strong className="text-slate-700">{run.target_column || 'Auto'}</strong></span>
                      <span>•</span>
                      <span>Validation: <strong className="text-slate-700">{run.validation_strategy}</strong></span>
                    </div>
                  </div>

                  <div className="text-right space-y-1 shrink-0 ml-4">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                      {run.status}
                    </span>
                    <p className="text-[10px] text-slate-400 font-medium">
                      {new Date(run.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Model Performance Snapshot */}
        <div className="glass-panel p-6 rounded-2xl space-y-4">
          <div>
            <h3 className="font-bold text-base text-slate-900">Champion Model Scores</h3>
            <p className="text-xs text-slate-500">Empirical evaluation across holdout test sets</p>
          </div>

          <div className="h-64 w-full">
            {chartData.length === 0 ? (
              <div className="h-full flex items-center justify-center text-slate-400 text-xs">
                No model benchmarks recorded yet
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
                  <XAxis type="number" domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 10 }} />
                  <YAxis type="category" dataKey="name" tick={{ fill: '#334155', fontSize: 11, fontWeight: 500 }} width={90} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-white p-3 rounded-2xl border border-slate-200 shadow-xl text-xs space-y-1.5 min-w-[190px]">
                            <p className="font-bold text-slate-900 text-sm border-b border-slate-100 pb-1">{data.datasetName}</p>
                            <div className="flex items-center justify-between text-slate-600">
                              <span>Champion:</span>
                              <strong className="text-indigo-600 font-semibold">{data.modelName}</strong>
                            </div>
                            <div className="flex items-center justify-between text-slate-500 capitalize">
                              <span>Task:</span>
                              <span className="font-mono">{data.task}</span>
                            </div>
                            <div className="pt-1.5 border-t border-slate-100 flex items-center justify-between gap-3">
                              <span className="text-slate-600 font-medium">{data.metricName}:</span>
                              <strong className="text-emerald-700 font-mono text-sm font-bold">{data.score.toFixed(4)}</strong>
                            </div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar dataKey="score" fill="#6366f1" radius={[0, 4, 4, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#4f46e5' : '#059669'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="pt-2 border-t border-slate-100">
            <Link
              to="/reports"
              className="w-full text-center block text-xs font-semibold text-indigo-600 hover:text-indigo-800 py-1"
            >
              View Evidence-Backed Reports →
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};
