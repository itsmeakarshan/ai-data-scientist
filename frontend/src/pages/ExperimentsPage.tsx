import React, { useEffect, useState } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import {
  FlaskConical,
  ArrowRight,
  TrendingUp,
} from 'lucide-react';
import { api } from '../services/api';
import { Experiment } from '../types';

export const ExperimentsPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const analysisIdFilter = searchParams.get('analysis_id') || undefined;

  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [comparison, setComparison] = useState<any | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const exps = await api.getExperiments({ analysis_id: analysisIdFilter });
        setExperiments(exps || []);

        if (analysisIdFilter) {
          const comp = await api.compareExperiments(analysisIdFilter);
          setComparison(comp);
        }
      } catch (err) {
        console.error('Error loading experiments:', err);
      }
    }
    load();
  }, [analysisIdFilter]);

  return (
    <div className="space-y-8 p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Experiment Tracking & Benchmarks</h1>
          <p className="text-sm text-slate-500">
            Reproducible experiments tracked with MLflow. Inspect hyperparameters, cross-validation metrics, and holdout scores.
          </p>
        </div>
      </div>

      {/* Comparison Leaderboard Card (if analysis selected) */}
      {comparison && comparison.comparison_table?.length > 0 && (
        <div className="glass-panel p-6 rounded-3xl space-y-4 border-indigo-200 bg-indigo-50/40">
          <div className="flex items-center justify-between">
            <h2 className="font-bold text-base text-slate-900 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-indigo-600" />
              Candidate Comparison Leaderboard
            </h2>
            <span className="text-xs font-bold px-2.5 py-0.5 rounded-full bg-indigo-100 text-indigo-800 border border-indigo-200 uppercase">
              Ranked by {comparison.primary_metric}
            </span>
          </div>

          <div className="overflow-x-auto border border-indigo-100 rounded-2xl bg-white">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-200 bg-slate-50 text-slate-600 font-semibold">
                <tr>
                  <th className="py-2.5 px-3">Model</th>
                  <th className="py-2.5 px-3">Family</th>
                  <th className="py-2.5 px-3">Test Score ({comparison.primary_metric})</th>
                  <th className="py-2.5 px-3">CV Stability (Mean ± Std)</th>
                  <th className="py-2.5 px-3">Train Time</th>
                  <th className="py-2.5 px-3 text-right">Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-slate-700">
                {comparison.comparison_table.map((row: any) => {
                  const isBest = row.experiment_id === comparison.best_experiment_id;
                  return (
                    <tr key={row.experiment_id} className={`hover:bg-slate-50 ${isBest ? 'bg-emerald-50/60' : ''}`}>
                      <td className="py-3 px-3 font-sans font-bold text-slate-900 flex items-center gap-2">
                        {row.model_name}
                        {isBest && (
                          <span className="text-[10px] uppercase font-bold bg-emerald-100 text-emerald-800 px-1.5 py-0.2 rounded border border-emerald-200">
                            Champion
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-3 capitalize font-sans">{row.model_family}</td>
                      <td className="py-3 px-3 text-emerald-700 font-bold">{row.score?.toFixed(4)}</td>
                      <td className="py-3 px-3">{row.cv_mean?.toFixed(4)} ± {row.cv_std?.toFixed(4)}</td>
                      <td className="py-3 px-3">{row.train_time_sec?.toFixed(2)}s</td>
                      <td className="py-3 px-3 text-right font-sans">
                        <Link
                          to={`/experiments/${row.experiment_id}`}
                          className="text-xs font-semibold text-indigo-600 hover:text-indigo-800 inline-flex items-center gap-1"
                        >
                          Diagnostics <ArrowRight className="w-3 h-3" />
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Experiments Table */}
      <div className="glass-panel p-6 rounded-3xl space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="font-bold text-base text-slate-900">All Tracked Runs ({experiments.length})</h2>
        </div>

        {experiments.length === 0 ? (
          <div className="py-12 text-center text-slate-500 text-sm">
            No experiments recorded yet.
          </div>
        ) : (
          <div className="overflow-x-auto border border-slate-200 rounded-2xl">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-200 bg-slate-50 text-slate-600 font-semibold">
                <tr>
                  <th className="py-3 px-4">Model Name</th>
                  <th className="py-3 px-4">Family</th>
                  <th className="py-3 px-4">Validation Protocol</th>
                  <th className="py-3 px-4">Test Performance</th>
                  <th className="py-3 px-4">Train Time</th>
                  <th className="py-3 px-4">Timestamp</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700">
                {experiments.map((exp) => {
                  const testM = exp.metrics_json?.test || {};
                  const scoreStr = Object.entries(testM)
                    .filter(([k, v]) => typeof v === 'number' && ['roc_auc', 'rmse', 'f1_macro', 'r2', 'wape', 'accuracy'].includes(k))
                    .map(([k, v]) => `${k}: ${(v as number).toFixed(3)}`)
                    .join(' | ');

                  return (
                    <tr key={exp.id} className="hover:bg-slate-50">
                      <td className="py-3.5 px-4 font-bold text-slate-900 flex items-center space-x-2">
                        <FlaskConical className="w-4 h-4 text-indigo-600 shrink-0" />
                        <span>{exp.model_name}</span>
                      </td>
                      <td className="py-3.5 px-4 capitalize">{exp.model_family}</td>
                      <td className="py-3.5 px-4 font-mono text-slate-500">{exp.validation_strategy}</td>
                      <td className="py-3.5 px-4 font-mono text-emerald-700 font-semibold">{scoreStr || 'N/A'}</td>
                      <td className="py-3.5 px-4 font-mono">{exp.train_time_sec?.toFixed(2)}s</td>
                      <td className="py-3.5 px-4 text-slate-400 font-mono">
                        {new Date(exp.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <Link
                          to={`/experiments/${exp.id}`}
                          className="px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition"
                        >
                          View Plots
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
