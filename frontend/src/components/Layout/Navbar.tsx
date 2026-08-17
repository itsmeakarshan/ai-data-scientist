import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Play, Database, Sparkles } from 'lucide-react';

export const Navbar: React.FC = () => {
  const location = useLocation();

  const getPageTitle = () => {
    const path = location.pathname;
    if (path.startsWith('/dashboard')) return 'Platform Dashboard';
    if (path.startsWith('/datasets')) return 'Dataset Ingestion & Profiling';
    if (path.startsWith('/analysis')) return 'Autonomous Data Science Workflow';
    if (path.startsWith('/experiments')) return 'Experiment Tracking & Benchmarks';
    if (path.startsWith('/models')) return 'Model Registry & SHAP Explainability';
    if (path.startsWith('/reports')) return 'Evidence-Backed Reports';
    if (path.startsWith('/chat')) return 'AutoDS Grounded Chat Agent';
    if (path.startsWith('/settings')) return 'System Settings';
    return 'Autonomous Data Science Platform';
  };

  return (
    <header className="h-16 border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md px-8 flex items-center justify-between sticky top-0 z-10">
      <div>
        <h2 className="text-lg font-bold text-slate-100 tracking-tight">{getPageTitle()}</h2>
      </div>

      <div className="flex items-center space-x-3">
        <Link
          to="/datasets"
          className="flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
        >
          <Database className="w-3.5 h-3.5" />
          <span>Upload Data</span>
        </Link>
        <Link
          to="/analysis"
          className="flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-md shadow-emerald-600/20 transition"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          <span>Run Autonomous DS</span>
        </Link>
      </div>
    </header>
  );
};
