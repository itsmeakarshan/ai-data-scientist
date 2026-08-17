import React, { useEffect, useState } from 'react';
import {
  Database,
  Sparkles,
  CheckCircle2,
  AlertTriangle
} from 'lucide-react';
import { api } from '../services/api';
import { HealthStatus } from '../types';

export const SettingsPage: React.FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await api.getHealth();
        setHealth(res);
      } catch (err) {
        console.error('Error fetching settings/health:', err);
      }
    }
    load();
  }, []);

  return (
    <div className="space-y-8 p-8 max-w-5xl mx-auto">
      <div>
        <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">System & Platform Settings</h1>
        <p className="text-sm text-slate-500">
          Inspect engine configuration, official Gemini API integrations, MLflow experiment tracking, and database status.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Gemini AI Settings */}
        <div className="glass-panel p-6 rounded-3xl space-y-4">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-2xl bg-gradient-to-tr from-emerald-600 to-teal-600 flex items-center justify-center text-white shadow-2xs">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-sm text-slate-900">Google Gemini AI Engine</h3>
              <p className="text-xs text-slate-500">Official google-genai SDK</p>
            </div>
          </div>

          <div className="space-y-3 pt-2 text-xs">
            <div className="flex items-center justify-between p-3.5 rounded-2xl bg-slate-50 border border-slate-200">
              <span className="text-slate-600">Integration Status:</span>
              <span className={`font-semibold flex items-center gap-1.5 ${health?.gemini_api_configured ? 'text-emerald-700' : 'text-amber-800'}`}>
                {health?.gemini_api_configured ? (
                  <>
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" /> Active API Key
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-600" /> Deterministic Mode Active
                  </>
                )}
              </span>
            </div>

            <div className="flex items-center justify-between p-3.5 rounded-2xl bg-slate-50 border border-slate-200">
              <span className="text-slate-600">Configured Model:</span>
              <span className="font-mono text-indigo-700 font-bold">{health?.gemini_model || 'gemini-2.5-flash'}</span>
            </div>

            <p className="text-[11px] text-slate-500 leading-relaxed">
              AutoDS uses Gemini for strategic planning, methodological review, and natural language explanation. When no API key is supplied, AutoDS seamlessly falls back to deterministic heuristic intelligence.
            </p>
          </div>
        </div>

        {/* Database & MLOps Infrastructure */}
        <div className="glass-panel p-6 rounded-3xl space-y-4">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-2xl bg-gradient-to-tr from-indigo-600 to-purple-600 flex items-center justify-center text-white shadow-2xs">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-sm text-slate-900">Storage & MLOps Infrastructure</h3>
              <p className="text-xs text-slate-500">PostgreSQL / SQLite & MLflow</p>
            </div>
          </div>

          <div className="space-y-3 pt-2 text-xs">
            <div className="flex items-center justify-between p-3.5 rounded-2xl bg-slate-50 border border-slate-200">
              <span className="text-slate-600">Database Engine:</span>
              <span className="text-emerald-700 font-semibold flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" /> {health?.database_connected ? 'Connected' : 'Offline'}
              </span>
            </div>

            <div className="flex items-center justify-between p-3.5 rounded-2xl bg-slate-50 border border-slate-200">
              <span className="text-slate-600">MLflow Tracking:</span>
              <span className="text-emerald-700 font-semibold flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" /> Active (./mlruns)
              </span>
            </div>

            <div className="flex items-center justify-between p-3.5 rounded-2xl bg-slate-50 border border-slate-200">
              <span className="text-slate-600">Environment:</span>
              <span className="font-mono text-slate-800 capitalize">{health?.environment || 'development'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
