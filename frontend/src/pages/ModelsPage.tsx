import React, { useEffect, useState } from 'react';
import {
  Boxes,
  CheckCircle2,
  Sparkles,
  Layers,
  ArrowRight,
  TrendingUp,
  Award
} from 'lucide-react';
import { api } from '../services/api';
import { ModelRecord } from '../types';

export const ModelsPage: React.FC = () => {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const res = await api.getModels();
        setModels(res || []);
      } catch (err) {
        console.error('Error loading model registry:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="space-y-8 p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-100 tracking-tight">Production Model Registry</h1>
          <p className="text-sm text-slate-400">
            Validated champion models serialized with SHAP feature attributions and empirical metrics.
          </p>
        </div>
      </div>

      {/* Model Cards Grid */}
      {models.length === 0 ? (
        <div className="glass-panel p-12 rounded-xl text-center text-slate-500">
          No champion models registered yet. Complete an autonomous analysis run to register models.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {models.map((m) => {
            const testM = m.metrics_json?.test || {};
            const rankings = m.feature_importance_json?.rankings || [];
            const topDrivers = rankings.slice(0, 4);

            return (
              <div
                key={m.id}
                className="glass-panel glass-panel-hover p-6 rounded-2xl flex flex-col justify-between space-y-5"
              >
                <div className="space-y-4">
                  <div className="flex items-start justify-between">
                    <div className="space-y-1">
                      <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 inline-flex items-center gap-1">
                        <Award className="w-3 h-3" /> Champion Model
                      </span>
                      <h3 className="font-bold text-lg text-slate-100">{m.name}</h3>
                      <p className="text-xs text-slate-400 capitalize">Task: {m.task_type}</p>
                    </div>
                  </div>

                  {/* Metrics Badge Row */}
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    {Object.entries(testM)
                      .filter(([k, v]) => typeof v === 'number' && ['roc_auc', 'rmse', 'f1_macro', 'r2', 'wape', 'accuracy'].includes(k))
                      .slice(0, 2)
                      .map(([k, v]) => (
                        <div key={k} className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                          <span className="text-[10px] text-slate-500 block uppercase font-sans">{k}</span>
                          <strong className="text-emerald-400 text-sm">{(v as number).toFixed(4)}</strong>
                        </div>
                      ))}
                  </div>

                  {/* Top SHAP Drivers */}
                  {topDrivers.length > 0 && (
                    <div className="space-y-2 pt-2 border-t border-slate-800/80">
                      <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                        <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                        Top Predictive Drivers (SHAP)
                      </span>
                      <div className="space-y-1.5">
                        {topDrivers.map((d, i) => (
                          <div key={i} className="flex items-center justify-between text-xs">
                            <span className="text-slate-400 truncate max-w-[180px]">{d.feature}</span>
                            <span className="font-mono text-indigo-300 font-bold">{d.importance_pct}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <div className="pt-3 border-t border-slate-800/80 text-xs text-slate-500 font-mono flex items-center justify-between">
                  <span>Registered: {new Date(m.created_at).toLocaleDateString()}</span>
                  <span className="text-emerald-400 font-sans font-semibold flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" /> Ready
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
