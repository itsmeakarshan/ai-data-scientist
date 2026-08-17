import React, { useEffect, useState } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import {
  PlayCircle,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  FileText
} from 'lucide-react';
import { api } from '../services/api';
import { AnalysisRun, Dataset } from '../types';

export const AnalysisPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const initialDatasetId = searchParams.get('dataset_id') || '';

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState(initialDatasetId);
  const [userGoal, setUserGoal] = useState('Build the best model to predict target outcome with high accuracy and explainability.');
  const [targetColumn, setTargetColumn] = useState('');
  const [problemType, setProblemType] = useState('classification');
  const [timeColumn, setTimeColumn] = useState('');
  const [running, setRunning] = useState(false);
  const [activeRun, setActiveRun] = useState<AnalysisRun | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function init() {
      try {
        const [dsRes] = await Promise.all([
          api.getDatasets(),
          api.getAnalysisRuns(),
        ]);
        setDatasets(dsRes.items || []);
        if (!selectedDatasetId && dsRes.items?.length > 0) {
          setSelectedDatasetId(dsRes.items[0].id);
        }
      } catch (err) {
        console.error('Failed to load initial data:', err);
      }
    }
    init();
  }, []);

  const handleStartAnalysis = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedDatasetId) {
      setError('Please select a dataset.');
      return;
    }

    setRunning(true);
    setError(null);
    try {
      const run = await api.createAnalysis({
        dataset_id: selectedDatasetId,
        user_goal: userGoal,
        target_column: targetColumn.trim() || undefined,
        problem_type: problemType || undefined,
        time_column: timeColumn.trim() || undefined,
      });
      setActiveRun(run);
      await api.getAnalysisRuns();
    } catch (err: any) {
      setError(err.message || 'Analysis failed to execute.');
    } finally {
      setRunning(false);
    }
  };

  const stepsList = [
    { num: 1, name: 'Dataset Inspection & Profiling', desc: 'Inspect schema, calculate distributions, and detect types' },
    { num: 2, name: 'Problem Classification & Target Selection', desc: 'Infer classification vs regression vs time-series' },
    { num: 3, name: 'Autonomous Analysis Planning', desc: 'Generate multi-model strategy and validation protocol' },
    { num: 4, name: 'Leak-Free Preprocessing & Splitting', desc: 'Fit-on-train only scaling, encoding, and chronological/stratified split' },
    { num: 5, name: 'Candidate Model Training & CV', desc: 'Train LightGBM, Random Forest, Linear, and Baselines' },
    { num: 6, name: 'Multi-Metric Leaderboard Ranking', desc: 'Evaluate ROC-AUC, PR-AUC, F1, RMSE, and train times' },
    { num: 7, name: 'Methodological Critic Audit', desc: 'Deep audit for data leakage, severe overfitting, and invalid splits' },
    { num: 8, name: 'SHAP Explainability & Feature Attribution', desc: 'Extract TreeSHAP values and key predictive drivers' },
    { num: 9, name: 'Evidence-Backed Report Synthesis', desc: 'Compile final audited Data Science report' },
  ];

  return (
    <div className="space-y-8 p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Autonomous Data Science Engine</h1>
        <p className="text-sm text-slate-500">
          State your high-level objective. AutoDS will autonomously plan, preprocess, train candidate ML models, audit methodology, and produce evidence-backed reports.
        </p>
      </div>

      {/* Main Grid: Launcher & Stepper */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Run Form */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-3xl space-y-5 h-fit">
          <h2 className="font-bold text-base text-slate-900 flex items-center gap-2">
            <PlayCircle className="w-5 h-5 text-emerald-600" />
            Configure Autonomous Run
          </h2>

          <form onSubmit={handleStartAnalysis} className="space-y-4">
            {/* Dataset Selector */}
            <div className="space-y-1.5">
              <label className="block text-xs font-bold text-slate-700">Target Dataset</label>
              <select
                value={selectedDatasetId}
                onChange={(e) => setSelectedDatasetId(e.target.value)}
                className="w-full bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-2xl p-3 focus:outline-none focus:border-emerald-500 focus:bg-white transition"
              >
                {datasets.map((ds) => (
                  <option key={ds.id} value={ds.id}>
                    {ds.name} ({ds.row_count.toLocaleString()} rows, {ds.col_count} cols)
                  </option>
                ))}
              </select>
            </div>

            {/* Natural Language Goal */}
            <div className="space-y-1.5">
              <label className="block text-xs font-bold text-slate-700">Natural Language Goal</label>
              <textarea
                value={userGoal}
                onChange={(e) => setUserGoal(e.target.value)}
                rows={3}
                className="w-full bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-2xl p-3 focus:outline-none focus:border-emerald-500 focus:bg-white transition"
                placeholder="e.g. Predict customer subscription to term deposit and identify key drivers."
              />
            </div>

            {/* Problem Type & Target Grid */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label className="block text-xs font-bold text-slate-700">Problem Type</label>
                <select
                  value={problemType}
                  onChange={(e) => setProblemType(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-2xl p-2.5 focus:outline-none focus:border-emerald-500 focus:bg-white transition"
                >
                  <option value="classification">Classification</option>
                  <option value="regression">Regression</option>
                  <option value="forecasting">Forecasting (Time Series)</option>
                </select>
              </div>

              <div className="space-y-1.5">
                <label className="block text-xs font-bold text-slate-700">Target Column (Optional)</label>
                <input
                  type="text"
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  placeholder="Auto-detected if empty"
                  className="w-full bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-2xl p-2.5 focus:outline-none focus:border-emerald-500 focus:bg-white font-mono transition"
                />
              </div>
            </div>

            {problemType === 'forecasting' && (
              <div className="space-y-1.5">
                <label className="block text-xs font-bold text-slate-700">Time / Date Column</label>
                <input
                  type="text"
                  value={timeColumn}
                  onChange={(e) => setTimeColumn(e.target.value)}
                  placeholder="e.g. date, timestamp"
                  className="w-full bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-2xl p-2.5 focus:outline-none focus:border-emerald-500 focus:bg-white font-mono transition"
                />
              </div>
            )}

            {error && (
              <div className="p-3 rounded-2xl bg-red-50 border border-red-200 text-red-700 text-xs flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 shrink-0 text-red-600" />
                <span>{error}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={running}
              className="w-full py-3.5 px-4 rounded-2xl text-xs font-bold bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-md shadow-emerald-600/20 transition flex items-center justify-center space-x-2 disabled:opacity-60"
            >
              {running ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Autonomous Pipeline Running...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Launch Autonomous DS Run</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Live Stepper & Active Run Details */}
        <div className="lg:col-span-7 space-y-6">
          <div className="glass-panel p-6 rounded-3xl space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="font-bold text-base text-slate-900">Autonomous Workflow Execution Plan</h2>
                <p className="text-xs text-slate-500">9-stage automated agent orchestration</p>
              </div>
              {activeRun && (
                <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                  Status: {activeRun.status}
                </span>
              )}
            </div>

            {/* Stepper list */}
            <div className="space-y-2.5">
              {stepsList.map((step) => {
                const isCompleted = activeRun && activeRun.status === 'COMPLETED';
                const isRunning = running;
                return (
                  <div
                    key={step.num}
                    className={`p-3.5 rounded-2xl border flex items-start space-x-3 transition ${
                      isCompleted
                        ? 'bg-emerald-50/70 border-emerald-200'
                        : isRunning
                        ? 'bg-indigo-50/70 border-indigo-200'
                        : 'bg-slate-50/70 border-slate-200/80'
                    }`}
                  >
                    <div className="w-6 h-6 rounded-full bg-white border border-slate-300 flex items-center justify-center text-xs font-bold text-slate-700 shrink-0 mt-0.5 shadow-2xs">
                      {isCompleted ? <CheckCircle2 className="w-4 h-4 text-emerald-600" /> : step.num}
                    </div>
                    <div className="space-y-0.5 flex-1">
                      <p className="text-xs font-bold text-slate-800">{step.name}</p>
                      <p className="text-[11px] text-slate-500">{step.desc}</p>
                    </div>
                  </div>
                );
              })}
            </div>

            {activeRun && activeRun.status === 'COMPLETED' && (
              <div className="pt-4 border-t border-slate-100 flex items-center justify-between">
                <span className="text-xs text-slate-500">Analysis completed successfully.</span>
                <div className="flex items-center space-x-3">
                  <Link
                    to={`/experiments?analysis_id=${activeRun.id}`}
                    className="px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition"
                  >
                    View Model Experiments
                  </Link>
                  <Link
                    to={`/reports/${activeRun.id}`}
                    className="px-3.5 py-1.5 rounded-xl text-xs font-bold bg-indigo-600 hover:bg-indigo-500 text-white shadow-2xs transition flex items-center space-x-1.5"
                  >
                    <FileText className="w-3.5 h-3.5" />
                    <span>Open Evidence Report</span>
                  </Link>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
