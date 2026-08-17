import React, { useEffect, useState } from 'react';
import {
  Settings as SettingsIcon,
  ShieldCheck,
  Cpu,
  Database,
  Layers,
  Sparkles,
  CheckCircle2,
  AlertTriangle
} from 'lucide-react';
import { api } from '../services/api';
import { HealthStatus } from '../types';

export const SettingsPage: React.FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const res = await api.getHealth();
        setHealth(res);
      } catch (err) {
        console.error('Error fetching settings/health:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="space-y-8 p-8 max-w-5xl mx-auto">
      <div>
        <h1 className="text-2xl font-extrabold text-slate-100 tracking-tight">System & Platform Settings</h1>
        <p className="text-sm text-slate-400">
          Inspect engine configuration, official Gemini API integrations, MLflow experiment tracking, and database status.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Gemini AI Settings */}
        <div className="glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-500 flex items-center justify-center text-white">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-sm text-slate-100">Google Gemini AI Engine</h3>
              <p className="text-xs text-slate-400">Official google-genai SDK</p>
            </div>
          </div>

          <div className="space-y-3 pt-2 text-xs">
            <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-slate-400">Integration Status:</span>
              <span className={`font-semibold flex items-center gap-1 ${health?.gemini_api_configured ? 'text-emerald-400' : 'text-amber-400'}`}>
                {health?.gemini_api_configured ? (
                  <>
                    <CheckCircle2 className="w-3.5 h-3.5" /> Active API Key
                  </>
                ) : (
                  <>
                    <AlertTriangle className="w-3.5 h-3.5" /> Deterministic Mode Active
                  </>
                )}
              </span>
            </div>

            <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-slate-400">Configured Model:</span>
              <span className="font-mono text-indigo-300 font-bold">{health?.gemini_model || 'gemini-2.5-flash'}</span>
            </div>

            <p className="text-[11px] text-slate-400 leading-relaxed">
              AutoDS uses Gemini for strategic planning, methodological review, and natural language explanation. When no API key is supplied, AutoDS seamlessly falls back to deterministic heuristic intelligence.
            </p>
          </div>
        </div>

        {/* Database & MLOps Infrastructure */}
        <div className="glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center text-white">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-sm text-slate-100">Storage & MLOps Infrastructure</h3>
              <p className="text-xs text-slate-400">PostgreSQL / SQLite & MLflow</p>
            </div>
          </div>

          <div className="space-y-3 pt-2 text-xs">
            <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-slate-400">Database Engine:</span>
              <span className="text-emerald-400 font-semibold flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> {health?.database_connected ? 'Connected' : 'Offline'}
              </span>
            </div>

            <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-slate-400">MLflow Tracking:</span>
              <span className="text-emerald-400 font-semibold flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> Active (./mlruns)
              </span>
            </div>

            <div className="flex items-center justify-between p-3 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-slate-400">Environment:</span>
              <span className="font-mono text-slate-200 capitalize">{health?.environment || 'development'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
