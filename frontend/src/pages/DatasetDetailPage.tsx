import React, { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import {
  Play,
  ArrowLeft,
  AlertTriangle,
  Terminal,
  Trash2,
} from 'lucide-react';
import { api } from '../services/api';
import { Dataset } from '../types';

export const DatasetDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [sampleRows, setSampleRows] = useState<{ columns: string[]; rows: any[] }>({ columns: [], rows: [] });
  const [activeTab, setActiveTab] = useState<'profile' | 'sample' | 'sql'>('profile');
  const [loading, setLoading] = useState(true);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

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

  const confirmDelete = async () => {
    if (!dataset) return;
    setIsDeleting(true);
    try {
      await api.deleteDataset(dataset.id);
      navigate('/datasets');
    } catch (err: any) {
      console.error('Failed to delete dataset:', err);
      alert(err.message || 'Failed to delete dataset.');
    } finally {
      setIsDeleting(false);
    }
  };

  if (loading || !dataset) {
    return (
      <div className="p-8 text-center text-slate-500">
        <div className="w-8 h-8 border-2 border-emerald-600 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
        Loading dataset profile...
      </div>
    );
  }

  const profile = dataset.profile;
  const numCols = profile?.summary_stats?.numerical_columns || {};
  const catCols = profile?.summary_stats?.categorical_columns || {};
  const alerts = profile?.quality_alerts || [];

  return (
    <div className="space-y-8 p-8 w-full">
      {/* Top Breadcrumb & Actions */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div className="space-y-1">
          <Link to="/datasets" className="inline-flex items-center space-x-1.5 text-xs text-slate-500 hover:text-slate-800">
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back to Datasets</span>
          </Link>
          <div className="flex items-center space-x-3">
            <h1 className="text-2xl font-extrabold text-slate-900">{dataset.name}</h1>
            <span className="text-xs font-semibold uppercase px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
              {profile?.inferred_problem_type || 'Classification'}
            </span>
          </div>
          <p className="text-xs font-mono text-slate-500">SHA-256: {dataset.checksum}</p>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setShowDeleteModal(true)}
            className="inline-flex items-center space-x-1.5 px-4 py-2.5 rounded-xl text-xs font-bold bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200/80 shadow-2xs transition cursor-pointer"
          >
            <Trash2 className="w-4 h-4 text-rose-600" />
            <span>Delete Dataset</span>
          </button>

          <Link
            to={`/analysis?dataset_id=${dataset.id}`}
            className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm shadow-emerald-600/20 transition"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>Launch Autonomous Pipeline</span>
          </Link>
        </div>
      </div>

      {/* Overview Stats Bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="glass-panel p-4 rounded-2xl space-y-1">
          <span className="text-slate-500 text-xs font-medium">Total Records</span>
          <p className="text-xl font-bold text-slate-900">{dataset.row_count.toLocaleString()}</p>
        </div>
        <div className="glass-panel p-4 rounded-2xl space-y-1">
          <span className="text-slate-500 text-xs font-medium">Total Features</span>
          <p className="text-xl font-bold text-slate-900">{dataset.col_count}</p>
        </div>
        <div className="glass-panel p-4 rounded-2xl space-y-1">
          <span className="text-slate-500 text-xs font-medium">Missing Cells</span>
          <p className="text-xl font-bold text-slate-900">{profile?.missingness_report?.total_missing_pct || 0}%</p>
        </div>
        <div className="glass-panel p-4 rounded-2xl space-y-1">
          <span className="text-slate-500 text-xs font-medium">Target Column</span>
          <p className="text-xl font-bold text-emerald-700">{profile?.candidate_targets?.[0] || 'Inferred'}</p>
        </div>
      </div>

      {/* Quality Alerts */}
      {alerts.length > 0 && (
        <div className="glass-panel p-6 rounded-3xl space-y-3 border-amber-200 bg-amber-50/60 shadow-2xs">
          <div className="flex items-center space-x-2 text-amber-800 font-bold text-sm">
            <AlertTriangle className="w-4 h-4 text-amber-600" />
            <span>Data Quality & Hygiene Audit ({alerts.length} alerts)</span>
          </div>
          <div className="divide-y divide-amber-200/60">
            {alerts.map((a, idx) => (
              <div key={idx} className="py-2.5 flex flex-col md:flex-row md:items-center justify-between text-xs gap-2">
                <div className="space-y-0.5">
                  <div className="flex items-center space-x-2">
                    <span
                      className={`uppercase font-bold px-1.5 py-0.5 rounded text-[10px] ${
                        a.severity === 'critical'
                          ? 'bg-red-100 text-red-700 border border-red-200'
                          : 'bg-amber-100 text-amber-800 border border-amber-200'
                      }`}
                    >
                      {a.severity}
                    </span>
                    <strong className="text-slate-800">{a.column ? `Column '${a.column}'` : 'Dataset-wide'}:</strong>
                    <span className="text-slate-700">{a.message}</span>
                  </div>
                </div>
                <span className="text-slate-600 font-mono bg-white px-2.5 py-1 rounded-lg border border-amber-200/80 shrink-0">
                  Action: {a.suggested_action}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tabs Navigation */}
      <div className="border-b border-slate-200 flex space-x-6">
        <button
          onClick={() => setActiveTab('profile')}
          className={`pb-3 text-sm font-semibold border-b-2 transition ${
            activeTab === 'profile'
              ? 'border-emerald-600 text-emerald-700'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          Statistical Feature Profile
        </button>
        <button
          onClick={() => setActiveTab('sample')}
          className={`pb-3 text-sm font-semibold border-b-2 transition ${
            activeTab === 'sample'
              ? 'border-emerald-600 text-emerald-700'
              : 'border-transparent text-slate-500 hover:text-slate-800'
          }`}
        >
          Sample Data Table ({sampleRows.rows.length} rows)
        </button>
        <button
          onClick={() => setActiveTab('sql')}
          className={`pb-3 text-sm font-semibold border-b-2 transition ${
            activeTab === 'sql'
              ? 'border-emerald-600 text-emerald-700'
              : 'border-transparent text-slate-500 hover:text-slate-800'
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
            <div className="glass-panel p-6 rounded-3xl space-y-4">
              <h3 className="font-bold text-base text-slate-900">Numerical Features ({Object.keys(numCols).length})</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-slate-200 bg-slate-50 text-slate-600 font-semibold">
                    <tr>
                      <th className="py-2.5 px-3">Column</th>
                      <th className="py-2.5 px-3">Mean</th>
                      <th className="py-2.5 px-3">Std Dev</th>
                      <th className="py-2.5 px-3">Min</th>
                      <th className="py-2.5 px-3">Median</th>
                      <th className="py-2.5 px-3">Max</th>
                      <th className="py-2.5 px-3">Skewness</th>
                      <th className="py-2.5 px-3">Outliers</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono text-slate-700">
                    {Object.entries(numCols).map(([col, stats]: [string, any]) => (
                      <tr key={col} className="hover:bg-slate-50">
                        <td className="py-2.5 px-3 font-semibold font-sans text-slate-900">{col}</td>
                        <td className="py-2.5 px-3">{stats.mean}</td>
                        <td className="py-2.5 px-3">{stats.std}</td>
                        <td className="py-2.5 px-3">{stats.min}</td>
                        <td className="py-2.5 px-3">{stats.median}</td>
                        <td className="py-2.5 px-3">{stats.max}</td>
                        <td className="py-2.5 px-3">{stats.skewness}</td>
                        <td className="py-2.5 px-3 text-amber-700 font-semibold">{stats.outlier_count} ({stats.outlier_pct}%)</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Categorical Features */}
          {Object.keys(catCols).length > 0 && (
            <div className="glass-panel p-6 rounded-3xl space-y-4">
              <h3 className="font-bold text-base text-slate-900">Categorical Features ({Object.keys(catCols).length})</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-slate-200 bg-slate-50 text-slate-600 font-semibold">
                    <tr>
                      <th className="py-2.5 px-3">Column</th>
                      <th className="py-2.5 px-3">Unique Values</th>
                      <th className="py-2.5 px-3">Top Category</th>
                      <th className="py-2.5 px-3">Top Frequency</th>
                      <th className="py-2.5 px-3">Distribution Breakdown</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-slate-700">
                    {Object.entries(catCols).map(([col, stats]: [string, any]) => (
                      <tr key={col} className="hover:bg-slate-50">
                        <td className="py-2.5 px-3 font-semibold text-slate-900">{col}</td>
                        <td className="py-2.5 px-3 font-mono">{stats.unique_count}</td>
                        <td className="py-2.5 px-3 font-mono">{stats.top_value}</td>
                        <td className="py-2.5 px-3 font-mono">{stats.top_freq} ({stats.top_freq_pct}%)</td>
                        <td className="py-2.5 px-3 text-[11px] text-slate-500 truncate max-w-xs">
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
        <div className="glass-panel p-6 rounded-3xl space-y-4 overflow-hidden">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-base text-slate-900">Preview Top 25 Rows</h3>
            <span className="text-xs text-slate-500">Total in file: {dataset.row_count.toLocaleString()} rows</span>
          </div>
          <div className="overflow-x-auto max-h-96 border border-slate-200 rounded-2xl">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-100 border-b border-slate-200 text-slate-700 sticky top-0 font-semibold">
                <tr>
                  {sampleRows.columns.map((col) => (
                    <th key={col} className="py-2.5 px-3 whitespace-nowrap">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono text-slate-700">
                {sampleRows.rows.map((row, rIdx) => (
                  <tr key={rIdx} className="hover:bg-slate-50">
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
        <div className="glass-panel p-6 rounded-3xl space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <h3 className="font-bold text-base text-slate-900 flex items-center gap-2">
                <Terminal className="w-4 h-4 text-emerald-600" />
                DuckDB Safe SQL Query Engine
              </h3>
              <p className="text-xs text-slate-500">
                Table is mapped to <code className="text-emerald-700 bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">dataset</code> and <code className="text-emerald-700 bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">data</code>. Read-only analytical queries.
              </p>
            </div>
            <button
              onClick={handleRunSql}
              disabled={queryLoading}
              className="px-4 py-2 rounded-xl text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm transition flex items-center space-x-2"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>{queryLoading ? 'Executing...' : 'Execute SQL'}</span>
            </button>
          </div>

          <textarea
            value={sqlQuery}
            onChange={(e) => setSqlQuery(e.target.value)}
            rows={3}
            className="w-full font-mono text-xs p-3.5 rounded-2xl bg-slate-50 border border-slate-300 text-slate-900 focus:outline-none focus:border-emerald-500 focus:bg-white transition"
            placeholder="SELECT job, COUNT(*) as count FROM dataset GROUP BY job ORDER BY count DESC;"
          />

          {queryError && (
            <div className="p-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs">
              {queryError}
            </div>
          )}

          {queryResult && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-slate-500">
                <span>Returned <strong>{queryResult.row_count}</strong> rows in <strong>{queryResult.execution_time_ms}ms</strong></span>
                <span className="font-mono text-[10px] text-slate-400">{queryResult.sql_executed}</span>
              </div>
              <div className="overflow-x-auto max-h-72 rounded-2xl border border-slate-200">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-slate-100 text-slate-700 border-b border-slate-200">
                    <tr>
                      {queryResult.columns.map((c: string) => (
                        <th key={c} className="py-2.5 px-3">{c}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-slate-700">
                    {queryResult.rows.map((row: any, i: number) => (
                      <tr key={i} className="hover:bg-slate-50">
                        {queryResult.columns.map((c: string) => (
                          <td key={c} className="py-2 px-3">{String(row[c] ?? '')}</td>
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

      {/* Delete Confirmation Modal */}
      {showDeleteModal && (
        <div
          className="fixed inset-0 z-50 bg-slate-900/80 backdrop-blur-md flex items-center justify-center p-4 animate-in fade-in duration-150"
          onClick={() => setShowDeleteModal(false)}
        >
          <div
            className="bg-white rounded-3xl p-6 max-w-md w-full space-y-5 shadow-2xl border border-slate-200"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-3 text-rose-600">
              <div className="w-10 h-10 rounded-2xl bg-rose-50 border border-rose-200 flex items-center justify-center shrink-0">
                <AlertTriangle className="w-5 h-5 text-rose-600" />
              </div>
              <div>
                <h3 className="font-extrabold text-slate-900 text-lg">Delete Dataset?</h3>
                <p className="text-xs text-slate-500">This action cannot be undone.</p>
              </div>
            </div>

            <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200 text-xs space-y-1.5">
              <span className="text-[10px] font-extrabold uppercase text-slate-400 block">Dataset Name</span>
              <p className="font-bold text-slate-900 leading-snug break-words">{dataset.name}</p>
              <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-500 font-mono pt-1">
                <span>Rows: {dataset.row_count.toLocaleString()}</span>
                <span>Cols: {dataset.col_count}</span>
                <span>Size: {(dataset.size_bytes / (1024 * 1024)).toFixed(2)} MB</span>
                <span>Type: {dataset.file_type.toUpperCase()}</span>
              </div>
            </div>

            <p className="text-xs text-slate-600 leading-relaxed">
              Are you sure you want to delete this dataset? This will completely remove the dataset, its statistical profiles, and all associated metadata from the database.
            </p>

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setShowDeleteModal(false)}
                disabled={isDeleting}
                className="px-4 py-2.5 rounded-xl text-xs font-bold bg-slate-100 hover:bg-slate-200 text-slate-700 transition cursor-pointer disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={confirmDelete}
                disabled={isDeleting}
                className="px-5 py-2.5 rounded-xl text-xs font-extrabold bg-rose-600 hover:bg-rose-700 text-white shadow-md shadow-rose-600/20 transition cursor-pointer disabled:opacity-50 inline-flex items-center gap-2"
              >
                <Trash2 className="w-4 h-4" />
                <span>{isDeleting ? 'Deleting...' : 'Yes, Delete Dataset'}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
