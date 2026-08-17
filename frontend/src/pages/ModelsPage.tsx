import React, { useEffect, useState } from 'react';
import {
  CheckCircle2,
  Sparkles,
  Award
} from 'lucide-react';
import { api } from '../services/api';
import { ModelRecord } from '../types';

export const ModelsPage: React.FC = () => {
  const [models, setModels] = useState<ModelRecord[]>([]);

  useEffect(() => {
    async function load() {
      try {
        const res = await api.getModels();
        setModels(res || []);
      } catch (err) {
        console.error('Error loading model registry:', err);
      }
    }
    load();
  }, []);

  return (
    <div className="space-y-8 p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Production Model Registry</h1>
          <p className="text-sm text-slate-500">
            Validated champion models serialized with SHAP feature attributions and empirical metrics.
          </p>
        </div>
      </div>

      {/* Model Cards Grid */}
      {models.length === 0 ? (
        <div className="glass-panel p-12 rounded-3xl text-center text-slate-500">
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
                className="glass-panel glass-panel-hover p-6 rounded-3xl flex flex-col justify-between space-y-5"
              >
                <div className="space-y-4">
                  <div className="flex items-start justify-between">
                    <div className="space-y-1">
                      <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 inline-flex items-center gap-1">
                        <Award className="w-3 h-3" /> Champion Model
                      </span>
                      <h3 className="font-bold text-lg text-slate-900">{m.name}</h3>
                      <p className="text-xs text-slate-500 capitalize">Task: {m.task_type}</p>
                    </div>
                  </div>

                  {/* Metrics Badge Row */}
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    {Object.entries(testM)
                      .filter(([k, v]) => typeof v === 'number' && ['roc_auc', 'rmse', 'f1_macro', 'r2', 'wape', 'accuracy'].includes(k))
                      .slice(0, 2)
                      .map(([k, v]) => (
                        <div key={k} className="p-2.5 rounded-2xl bg-slate-50 border border-slate-200">
                          <span className="text-[10px] text-slate-500 block uppercase font-sans font-semibold">{k}</span>
                          <strong className="text-emerald-700 text-sm">{(v as number).toFixed(4)}</strong>
                        </div>
                      ))}
                  </div>

                  {/* Top SHAP Drivers */}
                  {topDrivers.length > 0 && (
                    <div className="space-y-2 pt-2 border-t border-slate-100">
                      <span className="text-xs font-semibold text-slate-800 flex items-center gap-1.5">
                        <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
                        Top Predictive Drivers (SHAP)
                      </span>
                      <div className="space-y-1.5">
                        {topDrivers.map((d, i) => (
                          <div key={i} className="flex items-center justify-between text-xs">
                            <span className="text-slate-600 truncate max-w-[180px]">{d.feature}</span>
                            <span className="font-mono text-indigo-700 font-bold">{d.importance_pct}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <div className="pt-3 border-t border-slate-100 text-xs text-slate-500 font-mono flex items-center justify-between">
                  <span>Registered: {new Date(m.created_at).toLocaleDateString()}</span>
                  <span className="text-emerald-700 font-sans font-semibold flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Ready
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
