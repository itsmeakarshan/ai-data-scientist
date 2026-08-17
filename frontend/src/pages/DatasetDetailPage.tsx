import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  Database,
  Play,
  ArrowLeft,
  AlertTriangle,
  Terminal,
  Layers,
  Hash,
  HardDrive,
  Sparkles,
  Search,
  CheckCircle2,
  Table as TableIcon
} from 'lucide-react';
import { api } from '../services/api';
import { Dataset } from '../types';

export const DatasetDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [sampleRows, setSampleRows] = useState<{ columns: string[]; rows: any[] }>({ columns: [], rows: [] });
  const [activeTab, setActiveTab] = useState<'profile' | 'sample' | 'sql'>('profile');
  const [loading, setLoading] = useState(true);

  // SQL Console State
  const [sqlQuery, setSqlQuery] = useState('SELECT * FROM dataset LIMIT 10;');
  const [queryResult, setQueryResult] = useState<any | null>(null);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      if (!id) return;
      try {
        const [ds, sample] = await Promise.all([
          api.getDataset(id),
          api.getDatasetSample(id, 25),
        ]);
        setDataset(ds);
        setSampleRows({ columns: sample.columns || [], rows: sample.rows || [] });
      } catch (err) {
        console.error('Error loading dataset detail:', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [id]);

  const handleRunSql = async () => {
    if (!id || !sqlQuery.trim()) return;
    setQueryLoading(true);
    setQueryError(null);
    try {
      const res = await api.runQuery({ dataset_id: id, sql_query: sqlQuery });
      setQueryResult(res);
    } catch (err: any) {
      setQueryError(err.message || 'Query execution failed.');
    } finally {
      setQueryLoading(false);
    }
  };

  if (loading || !dataset) {
    return (
      <div className="p-8 text-center text-slate-500">
        <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
        Loading dataset profile...
      </div>
    );
  }

  const profile = dataset.profile;
  const numCols = profile?.summary_stats?.numerical_columns || {};
  const catCols = profile?.summary_stats?.categorical_columns || {};
  const alerts = profile?.quality_alerts || [];

  return (
    <div className="space-y-8 p-8 max-w-7xl mx-auto">
      {/* Top Breadcrumb & Actions */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div className="space-y-1">
          <Link to="/datasets" className="inline-flex items-center space-x-1.5 text-xs text-slate-400 hover:text-slate-200">
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back to Datasets</span>
          </Link>
          <div className="flex items-center space-x-3">
            <h1 className="text-2xl font-extrabold text-slate-100">{dataset.name}</h1>
            <span className="text-xs font-semibold uppercase px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              {profile?.inferred_problem_type || 'Classification'}
            </span>
          </div>
          <p className="text-xs font-mono text-slate-500">SHA-256: {dataset.checksum}</p>
        </div>

        <Link
          to={`/analysis?dataset_id=${dataset.id}`}
          className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white shadow-md shadow-emerald-600/20 transition"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          <span>Launch Autonomous Pipeline</span>
        </Link>
      </div>

      {/* Overview Stats Bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="glass-panel p-4 rounded-xl space-y-1">
          <span className="text-slate-400 text-xs">Total Records</span>
          <p className="text-xl font-bold text-slate-100">{dataset.row_count.toLocaleString()}</p>
        </div>
        <div className="glass-panel p-4 rounded-xl space-y-1">
          <span className="text-slate-400 text-xs">Total Features</span>
          <p className="text-xl font-bold text-slate-100">{dataset.col_count}</p>
        </div>
        <div className="glass-panel p-4 rounded-xl space-y-1">
          <span className="text-slate-400 text-xs">Missing Cells</span>
          <p className="text-xl font-bold text-slate-100">{profile?.missingness_report?.total_missing_pct || 0}%</p>
        </div>
        <div className="glass-panel p-4 rounded-xl space-y-1">
          <span className="text-slate-400 text-xs">Target Column</span>
          <p className="text-xl font-bold text-emerald-400">{profile?.candidate_targets?.[0] || 'Inferred'}</p>
        </div>
      </div>

      {/* Quality Alerts */}
      {alerts.length > 0 && (
        <div className="glass-panel p-6 rounded-2xl space-y-3 border-amber-500/30 bg-amber-500/5">
          <div className="flex items-center space-x-2 text-amber-400 font-bold text-sm">
            <AlertTriangle className="w-4 h-4" />
            <span>Data Quality & Hygiene Audit ({alerts.length} alerts)</span>
          </div>
          <div className="divide-y divide-slate-800/80">
            {alerts.map((a, idx) => (
              <div key={idx} className="py-2.5 flex flex-col md:flex-row md:items-center justify-between text-xs gap-2">
                <div className="space-y-0.5">
                  <div className="flex items-center space-x-2">
                    <span
                      className={`uppercase font-bold px-1.5 py-0.5 rounded text-[10px] ${
                        a.severity === 'critical'
                          ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                          : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                      }`}
                    >
                      {a.severity}
                    </span>
                    <strong className="text-slate-200">{a.column ? `Column '${a.column}'` : 'Dataset-wide'}:</strong>
                    <span className="text-slate-300">{a.message}</span>
                  </div>
                </div>
                <span className="text-slate-400 font-mono bg-slate-900 px-2 py-1 rounded border border-slate-800 shrink-0">
                  Action: {a.suggested_action}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tabs Navigation */}
      <div className="border-b border-slate-800 flex space-x-6">
        <button
          onClick={() => setActiveTab('profile')}
          className={`pb-3 text-sm font-semibold border-b-2 transition ${
            activeTab === 'profile'
              ? 'border-emerald-500 text-emerald-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          Statistical Feature Profile
        </button>
        <button
          onClick={() => setActiveTab('sample')}
          className={`pb-3 text-sm font-semibold border-b-2 transition ${
            activeTab === 'sample'
              ? 'border-emerald-500 text-emerald-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          Sample Data Table ({sampleRows.rows.length} rows)
        </button>
        <button
          onClick={() => setActiveTab('sql')}
          className={`pb-3 text-sm font-semibold border-b-2 transition ${
            activeTab === 'sql'
              ? 'border-emerald-500 text-emerald-400'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          Safe DuckDB SQL Console
        </button>
      </div>

      {/* Tab 1: Statistical Profile */}
      {activeTab === 'profile' && (
        <div className="space-y-6">
          {/* Numerical Features */}
          {Object.keys(numCols).length > 0 && (
            <div className="glass-panel p-6 rounded-2xl space-y-4">
              <h3 className="font-bold text-base text-slate-100">Numerical Features ({Object.keys(numCols).length})</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-slate-800 text-slate-400 font-semibold">
                    <tr>
                      <th className="py-2 px-3">Column</th>
                      <th className="py-2 px-3">Mean</th>
                      <th className="py-2 px-3">Std Dev</th>
                      <th className="py-2 px-3">Min</th>
                      <th className="py-2 px-3">Median</th>
                      <th className="py-2 px-3">Max</th>
                      <th className="py-2 px-3">Skewness</th>
                      <th className="py-2 px-3">Outliers</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                    {Object.entries(numCols).map(([col, stats]: [string, any]) => (
                      <tr key={col} className="hover:bg-slate-800/30">
                        <td className="py-2.5 px-3 font-semibold font-sans text-slate-100">{col}</td>
                        <td className="py-2.5 px-3">{stats.mean}</td>
                        <td className="py-2.5 px-3">{stats.std}</td>
                        <td className="py-2.5 px-3">{stats.min}</td>
                        <td className="py-2.5 px-3">{stats.median}</td>
                        <td className="py-2.5 px-3">{stats.max}</td>
                        <td className="py-2.5 px-3">{stats.skewness}</td>
                        <td className="py-2.5 px-3 text-amber-400">{stats.outlier_count} ({stats.outlier_pct}%)</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Categorical Features */}
          {Object.keys(catCols).length > 0 && (
            <div className="glass-panel p-6 rounded-2xl space-y-4">
              <h3 className="font-bold text-base text-slate-100">Categorical Features ({Object.keys(catCols).length})</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-slate-800 text-slate-400 font-semibold">
                    <tr>
                      <th className="py-2 px-3">Column</th>
                      <th className="py-2 px-3">Unique Values</th>
                      <th className="py-2 px-3">Top Category</th>
                      <th className="py-2 px-3">Top Frequency</th>
                      <th className="py-2 px-3">Distribution Breakdown</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300">
                    {Object.entries(catCols).map(([col, stats]: [string, any]) => (
                      <tr key={col} className="hover:bg-slate-800/30">
                        <td className="py-2.5 px-3 font-semibold text-slate-100">{col}</td>
                        <td className="py-2.5 px-3 font-mono">{stats.unique_count}</td>
                        <td className="py-2.5 px-3 font-mono">{stats.top_value}</td>
                        <td className="py-2.5 px-3 font-mono">{stats.top_freq} ({stats.top_freq_pct}%)</td>
                        <td className="py-2.5 px-3 text-[11px] text-slate-400 truncate max-w-xs">
                          {JSON.stringify(stats.top_categories)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Sample Data Table */}
      {activeTab === 'sample' && (
        <div className="glass-panel p-6 rounded-2xl space-y-4 overflow-hidden">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-base text-slate-100">Preview Top 25 Rows</h3>
            <span className="text-xs text-slate-400">Total in file: {dataset.row_count.toLocaleString()} rows</span>
          </div>
          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900 border-b border-slate-800 text-slate-300 sticky top-0 font-semibold">
                <tr>
                  {sampleRows.columns.map((col) => (
                    <th key={col} className="py-2.5 px-3 whitespace-nowrap">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                {sampleRows.rows.map((row, rIdx) => (
                  <tr key={rIdx} className="hover:bg-slate-800/40">
                    {sampleRows.columns.map((col) => (
                      <td key={col} className="py-2 px-3 whitespace-nowrap">{String(row[col] ?? '')}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 3: Safe SQL Query Console */}
      {activeTab === 'sql' && (
        <div className="glass-panel p-6 rounded-2xl space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <h3 className="font-bold text-base text-slate-100 flex items-center gap-2">
                <Terminal className="w-4 h-4 text-emerald-400" />
                DuckDB Safe SQL Query Engine
              </h3>
              <p className="text-xs text-slate-400">
                Table is mapped to <code className="text-emerald-400 bg-slate-900 px-1 py-0.5 rounded">dataset</code> and <code className="text-emerald-400 bg-slate-900 px-1 py-0.5 rounded">data</code>. Read-only analytical queries.
              </p>
            </div>
            <button
              onClick={handleRunSql}
              disabled={queryLoading}
              className="px-4 py-2 rounded-xl text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white shadow-md transition flex items-center space-x-2"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>{queryLoading ? 'Executing...' : 'Execute SQL'}</span>
            </button>
          </div>

          <textarea
            value={sqlQuery}
            onChange={(e) => setSqlQuery(e.target.value)}
            rows={3}
            className="w-full font-mono text-xs p-3.5 rounded-xl bg-slate-950 border border-slate-700 text-emerald-300 focus:outline-none focus:border-emerald-500"
            placeholder="SELECT job, COUNT(*) as count FROM dataset GROUP BY job ORDER BY count DESC;"
          />

          {queryError && (
            <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-xs">
              {queryError}
            </div>
          )}

          {queryResult && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Returned <strong>{queryResult.row_count}</strong> rows in <strong>{queryResult.execution_time_ms}ms</strong></span>
                <span className="font-mono text-[10px] text-slate-500">{queryResult.sql_executed}</span>
              </div>
              <div className="overflow-x-auto max-h-72 rounded-xl border border-slate-800">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-slate-900 text-slate-300 border-b border-slate-800">
                    <tr>
                      {queryResult.columns.map((c: string) => (
                        <th key={c} className="py-2 px-3">{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300">
                    {queryResult.rows.map((row: any, i: number) => (
                      <tr key={i} className="hover:bg-slate-800/30">
                        {queryResult.columns.map((c: string) => (
                          <td key={c} className="py-1.5 px-3">{String(row[c] ?? '')}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
