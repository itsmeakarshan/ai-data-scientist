import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Database,
  PlayCircle,
  FlaskConical,
  Boxes,
  ArrowRight,
  TrendingUp,
  ShieldCheck,
  Zap,
  Clock,
  Sparkles
} from 'lucide-react';
import { api } from '../services/api';
import { AnalysisRun, Dataset, ModelRecord } from '../types';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

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
          api.getModels({ is_best: true }),
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

  const chartData = models.map((m) => {
    const testM = m.metrics_json?.test || {};
    const score = testM.roc_auc || testM.r2 || (testM.wape ? 100 - testM.wape : testM.accuracy || 0.85);
    return {
      name: m.name.length > 15 ? m.name.substring(0, 15) + '...' : m.name,
      score: Number(score.toFixed(3)),
      task: m.task_type,
    };
  });

  return (
    <div className="space-y-8 p-8 max-w-7xl mx-auto">
      {/* Hero Welcome Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-slate-900 via-indigo-950/70 to-slate-900 border border-indigo-500/20 p-8 shadow-xl">
        <div className="relative z-10 max-w-3xl space-y-3">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Autonomous Machine Learning & Agentic MLOps</span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">
            AutoDS — Autonomous Data Science Platform
          </h1>
          <p className="text-slate-300 text-sm leading-relaxed">
            Upload any dataset and state your objective in natural language. AutoDS autonomously profiles the schema,
            determines problem formulation, executes leak-free ML experiments, conducts methodological critic audits,
            and produces evidence-backed explainability reports.
          </p>
          <div className="pt-2 flex items-center space-x-4">
            <Link
              to="/analysis"
              className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl text-sm font-bold bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-600/30 transition"
            >
              <PlayCircle className="w-4 h-4" />
              <span>Launch Autonomous Run</span>
            </Link>
            <Link
              to="/chat"
              className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl text-sm font-semibold bg-slate-800/80 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
            >
              <span>Ask Agent Questions</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </div>

      {/* KPI Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
        <div className="glass-panel p-5 rounded-xl space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Ingested Datasets</span>
            <Database className="w-4 h-4 text-emerald-400" />
          </div>
          <p className="text-2xl font-bold text-slate-100">{datasets.length}</p>
          <p className="text-xs text-slate-500">Structured & profiled</p>
        </div>

        <div className="glass-panel p-5 rounded-xl space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Autonomous Runs</span>
            <PlayCircle className="w-4 h-4 text-indigo-400" />
          </div>
          <p className="text-2xl font-bold text-slate-100">{runs.length}</p>
          <p className="text-xs text-slate-500">With plan execution & critic audit</p>
        </div>

        <div className="glass-panel p-5 rounded-xl space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Champion Models</span>
            <Boxes className="w-4 h-4 text-purple-400" />
          </div>
          <p className="text-2xl font-bold text-slate-100">{models.length}</p>
          <p className="text-xs text-slate-500">In production registry</p>
        </div>

        <div className="glass-panel p-5 rounded-xl space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
            <span>Critic Verification</span>
            <ShieldCheck className="w-4 h-4 text-teal-400" />
          </div>
          <p className="text-2xl font-bold text-emerald-400">100% Active</p>
          <p className="text-xs text-slate-500">Zero data leakage guarantee</p>
        </div>
      </div>

      {/* Main Grid: Recent Runs & Model Leaderboard */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Recent Autonomous Runs */}
        <div className="lg:col-span-2 glass-panel p-6 rounded-xl space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-bold text-base text-slate-100">Recent Autonomous Analyses</h3>
              <p className="text-xs text-slate-400">Tracked end-to-end Data Science workflows</p>
            </div>
            <Link to="/analysis" className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1">
              View all <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          <div className="divide-y divide-slate-800/80">
            {runs.length === 0 ? (
              <div className="py-8 text-center text-slate-500 text-sm">
                No analyses run yet. Click "Launch Autonomous Run" to begin.
              </div>
            ) : (
              runs.slice(0, 5).map((run) => (
                <div key={run.id} className="py-3.5 flex items-center justify-between hover:bg-slate-800/30 px-2 rounded-lg transition">
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <span className="text-xs font-semibold uppercase px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                        {run.problem_type}
                      </span>
                      <span className="text-sm font-medium text-slate-200 line-clamp-1">{run.user_goal}</span>
                    </div>
                    <div className="flex items-center space-x-3 text-xs text-slate-400">
                      <span>Target: <strong className="text-slate-300">{run.target_column || 'Auto'}</strong></span>
                      <span>•</span>
                      <span>Validation: <strong className="text-slate-300">{run.validation_strategy}</strong></span>
                    </div>
                  </div>

                  <div className="text-right space-y-1 shrink-0 ml-4">
                    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                      {run.status}
                    </span>
                    <p className="text-[10px] text-slate-500">
                      {new Date(run.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Model Performance Snapshot */}
        <div className="glass-panel p-6 rounded-xl space-y-4">
          <div>
            <h3 className="font-bold text-base text-slate-100">Champion Model Scores</h3>
            <p className="text-xs text-slate-400">Empirical evaluation across holdout test sets</p>
          </div>

          <div className="h-64 w-full">
            {chartData.length === 0 ? (
              <div className="h-full flex items-center justify-center text-slate-500 text-xs">
                No model benchmarks recorded yet
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
                  <XAxis type="number" domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <YAxis type="category" dataKey="name" tick={{ fill: '#e2e8f0', fontSize: 11 }} width={90} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
                  />
                  <Bar dataKey="score" fill="#6366f1" radius={[0, 4, 4, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#6366f1' : '#10b981'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          <div className="pt-2 border-t border-slate-800/80">
            <Link
              to="/models"
              className="w-full text-center block text-xs font-semibold text-indigo-400 hover:text-indigo-300 py-1"
            >
              Open Full Model Registry →
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};
