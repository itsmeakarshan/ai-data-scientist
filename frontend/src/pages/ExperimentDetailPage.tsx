import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  FlaskConical,
  ArrowLeft,
  Sliders,
  BarChart3,
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
        <div className="w-8 h-8 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
        Loading experiment diagnostics...
      </div>
    );
  }

  const testM = experiment.metrics_json?.test || {};
  const cvMean = experiment.metrics_json?.cv_mean || 0;
  const cvStd = experiment.metrics_json?.cv_std || 0;
  const cm = testM.confusion_matrix;

  return (
    <div className="space-y-8 p-8 max-w-7xl mx-auto">
      {/* Top Header */}
      <div className="space-y-2">
        <Link to="/experiments" className="inline-flex items-center space-x-1.5 text-xs text-slate-500 hover:text-slate-800">
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Experiments</span>
        </Link>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <FlaskConical className="w-7 h-7 text-indigo-600" />
            <div>
              <h1 className="text-2xl font-extrabold text-slate-900">{experiment.model_name}</h1>
              <p className="text-xs text-slate-500 capitalize">
                Family: <strong className="text-slate-800">{experiment.model_family}</strong> • Validation: <strong className="text-slate-800">{experiment.validation_strategy}</strong>
              </p>
            </div>
          </div>
          <span className="text-xs font-mono text-slate-600 bg-slate-100 px-3 py-1.5 rounded-xl border border-slate-200">
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
            <div key={key} className="glass-panel p-4 rounded-2xl space-y-1">
              <span className="text-slate-500 text-xs font-semibold uppercase tracking-wider">{key.replace('_', ' ')}</span>
              <p className="text-2xl font-bold text-emerald-700 font-mono">{(val as number).toFixed(4)}</p>
              <span className="text-[10px] text-slate-400">Holdout test score</span>
            </div>
          ))}
      </div>

      {/* Main Grid: Confusion Matrix & Hyperparameters */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left: Confusion Matrix & Diagnostic Visuals */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-3xl space-y-5">
          <h2 className="font-bold text-base text-slate-900 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-indigo-600" />
            Diagnostic Evaluation Artifacts
          </h2>

          {cm && cm.length === 2 && (
            <div className="space-y-3">
              <p className="text-xs font-semibold text-slate-700">Confusion Matrix (Test Split)</p>
              <div className="grid grid-cols-2 gap-3 max-w-sm font-mono text-center text-xs">
                <div className="p-4 rounded-2xl bg-indigo-50 border border-indigo-200">
                  <span className="text-[10px] text-slate-500 block mb-1">True Negative</span>
                  <strong className="text-lg text-slate-900">{cm[0][0]}</strong>
                </div>
                <div className="p-4 rounded-2xl bg-amber-50 border border-amber-200">
                  <span className="text-[10px] text-slate-500 block mb-1">False Positive</span>
                  <strong className="text-lg text-amber-700">{cm[0][1]}</strong>
                </div>
                <div className="p-4 rounded-2xl bg-amber-50 border border-amber-200">
                  <span className="text-[10px] text-slate-500 block mb-1">False Negative</span>
                  <strong className="text-lg text-amber-700">{cm[1][0]}</strong>
                </div>
                <div className="p-4 rounded-2xl bg-emerald-50 border border-emerald-200">
                  <span className="text-[10px] text-slate-500 block mb-1">True Positive</span>
                  <strong className="text-lg text-emerald-700">{cm[1][1]}</strong>
                </div>
              </div>
            </div>
          )}

          {/* Cross Validation Stability */}
          <div className="pt-4 border-t border-slate-100 space-y-2">
            <h3 className="text-xs font-semibold text-slate-700">Cross-Validation Stability ({experiment.cv_folds} folds)</h3>
            <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200 flex items-center justify-between text-xs font-mono">
              <span className="text-slate-600">CV Score (Mean ± Std):</span>
              <span className="text-slate-900 font-bold">{cvMean.toFixed(4)} ± {cvStd.toFixed(4)}</span>
            </div>
          </div>
        </div>

        {/* Right: Hyperparameters & Feature Dimensions */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-3xl space-y-5">
          <h2 className="font-bold text-base text-slate-900 flex items-center gap-2">
            <Sliders className="w-5 h-5 text-indigo-600" />
            Config & Hyperparameters
          </h2>

          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-500">Training Latency:</span>
              <span className="font-mono text-slate-800">{experiment.train_time_sec?.toFixed(3)} seconds</span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-500">Features Evaluated:</span>
              <span className="font-mono text-slate-800">{experiment.feature_names?.length || 0} features</span>
            </div>
          </div>

          <div className="space-y-1.5 pt-2 border-t border-slate-100">
            <span className="text-xs font-semibold text-slate-600">Hyperparameters JSON:</span>
            <pre className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200 text-[11px] font-mono text-indigo-800 overflow-x-auto max-h-56">
              {JSON.stringify(experiment.hyperparameters, null, 2)}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
