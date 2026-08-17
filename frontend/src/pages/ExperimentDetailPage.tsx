import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  FlaskConical,
  ArrowLeft,
  Clock,
  Sliders,
  TrendingUp,
  BarChart3,
  Layers,
  Cpu
} from 'lucide-react';
import { api } from '../services/api';
import { Experiment } from '../types';

export const ExperimentDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      if (!id) return;
      try {
        const exp = await api.getExperiment(id);
        setExperiment(exp);
      } catch (err) {
        console.error('Failed to load experiment:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [id]);

  if (loading || !experiment) {
    return (
      <div className="p-8 text-center text-slate-500">
        <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
        Loading experiment diagnostics...
      </div>
    );
  }

  const testM = experiment.metrics_json?.test || {};
  const trainM = experiment.metrics_json?.train || {};
  const cvMean = experiment.metrics_json?.cv_mean || 0;
  const cvStd = experiment.metrics_json?.cv_std || 0;
  const cm = testM.confusion_matrix;

  return (
    <div className="space-y-8 p-8 max-w-7xl mx-auto">
      {/* Top Header */}
      <div className="space-y-2">
        <Link to="/experiments" className="inline-flex items-center space-x-1.5 text-xs text-slate-400 hover:text-slate-200">
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Experiments</span>
        </Link>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <FlaskConical className="w-7 h-7 text-indigo-400" />
            <div>
              <h1 className="text-2xl font-extrabold text-slate-100">{experiment.model_name}</h1>
              <p className="text-xs text-slate-400 capitalize">
                Family: <strong className="text-slate-200">{experiment.model_family}</strong> • Validation: <strong className="text-slate-200">{experiment.validation_strategy}</strong>
              </p>
            </div>
          </div>
          <span className="text-xs font-mono text-slate-500 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800">
            MLflow Run: {experiment.mlflow_run_id ? experiment.mlflow_run_id.substring(0, 12) + '...' : 'Tracked'}
          </span>
        </div>
      </div>

      {/* Metrics Scorecards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Object.entries(testM)
          .filter(([k, v]) => typeof v === 'number' && !['is_binary'].includes(k))
          .slice(0, 4)
          .map(([key, val]) => (
            <div key={key} className="glass-panel p-4 rounded-xl space-y-1">
              <span className="text-slate-400 text-xs font-medium uppercase tracking-wider">{key.replace('_', ' ')}</span>
              <p className="text-2xl font-bold text-emerald-400 font-mono">{(val as number).toFixed(4)}</p>
              <span className="text-[10px] text-slate-500">Holdout test score</span>
            </div>
          ))}
      </div>

      {/* Main Grid: Confusion Matrix / Residuals & Hyperparameters */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left: Confusion Matrix & Diagnostic Visuals */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-2xl space-y-5">
          <h2 className="font-bold text-base text-slate-100 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-indigo-400" />
            Diagnostic Evaluation Artifacts
          </h2>

          {cm && cm.length === 2 && (
            <div className="space-y-3">
              <p className="text-xs font-semibold text-slate-300">Confusion Matrix (Test Split)</p>
              <div className="grid grid-cols-2 gap-3 max-w-sm font-mono text-center text-xs">
                <div className="p-4 rounded-xl bg-indigo-950/40 border border-indigo-500/30">
                  <span className="text-[10px] text-slate-400 block mb-1">True Negative</span>
                  <strong className="text-lg text-slate-100">{cm[0][0]}</strong>
                </div>
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-400 block mb-1">False Positive</span>
                  <strong className="text-lg text-amber-400">{cm[0][1]}</strong>
                </div>
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-400 block mb-1">False Negative</span>
                  <strong className="text-lg text-amber-400">{cm[1][0]}</strong>
                </div>
                <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-500/30">
                  <span className="text-[10px] text-slate-400 block mb-1">True Positive</span>
                  <strong className="text-lg text-emerald-400">{cm[1][1]}</strong>
                </div>
              </div>
            </div>
          )}

          {/* Cross Validation Stability */}
          <div className="pt-4 border-t border-slate-800 space-y-2">
            <h3 className="text-xs font-semibold text-slate-300">Cross-Validation Stability ({experiment.cv_folds} folds)</h3>
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 flex items-center justify-between text-xs font-mono">
              <span className="text-slate-400">CV Score (Mean ± Std):</span>
              <span className="text-slate-200 font-bold">{cvMean.toFixed(4)} ± {cvStd.toFixed(4)}</span>
            </div>
          </div>
        </div>

        {/* Right: Hyperparameters & Feature Dimensions */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-2xl space-y-5">
          <h2 className="font-bold text-base text-slate-100 flex items-center gap-2">
            <Sliders className="w-5 h-5 text-indigo-400" />
            Config & Hyperparameters
          </h2>

          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">Training Latency:</span>
              <span className="font-mono text-slate-200">{experiment.train_time_sec?.toFixed(3)} seconds</span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">Features Evaluated:</span>
              <span className="font-mono text-slate-200">{experiment.feature_names?.length || 0} features</span>
            </div>
          </div>

          <div className="space-y-1.5 pt-2 border-t border-slate-800">
            <span className="text-xs font-semibold text-slate-400">Hyperparameters JSON:</span>
            <pre className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-[11px] font-mono text-indigo-300 overflow-x-auto max-h-56">
              {JSON.stringify(experiment.hyperparameters, null, 2)}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
