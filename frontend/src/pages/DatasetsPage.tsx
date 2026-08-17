import React, { useEffect, useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  Upload,
  AlertTriangle,
  Play,
  HardDrive,
  Hash,
  Layers,
  Sparkles,
  CheckCircle2
} from 'lucide-react';
import { api } from '../services/api';
import { Dataset } from '../types';

export const DatasetsPage: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchDatasets = async () => {
    try {
      const res = await api.getDatasets();
      setDatasets(res.items || []);
    } catch (err) {
      console.error('Failed to load datasets:', err);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const newDataset = await api.uploadDataset(file);
      setUploadSuccess(`Dataset "${newDataset.name}" successfully uploaded and profiled!`);
      await fetchDatasets();
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err: any) {
      setUploadError(err.message || 'Failed to upload dataset.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-8 p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">Dataset Ingestion & Management</h1>
          <p className="text-sm text-slate-500">
            Upload CSV, Parquet, or Excel files. AutoDS automatically profiles distributions, checks quality, and infers schemas.
          </p>
        </div>
      </div>

      {/* Upload Dropzone */}
      <div className="border-2 border-dashed border-slate-300 hover:border-emerald-500 bg-white rounded-3xl p-8 text-center transition group shadow-2xs">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileUpload}
          accept=".csv,.parquet,.pq,.xlsx,.xls,.json"
          className="hidden"
          id="dataset-upload-input"
          disabled={uploading}
        />
        <label
          htmlFor="dataset-upload-input"
          className="cursor-pointer flex flex-col items-center justify-center space-y-3"
        >
          <div className="w-14 h-14 rounded-2xl bg-emerald-50 flex items-center justify-center text-emerald-600 group-hover:scale-105 group-hover:bg-emerald-100 transition shadow-2xs">
            <Upload className="w-7 h-7" />
          </div>
          <div>
            <p className="text-base font-bold text-slate-800">
              {uploading ? 'Ingesting and profiling dataset...' : 'Click to browse or drag and drop dataset'}
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Supports CSV, Parquet, Excel (.xlsx), and JSON (up to 100MB)
            </p>
          </div>
        </label>

        {uploadSuccess && (
          <div className="mt-4 p-3 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center justify-center space-x-2 max-w-md mx-auto">
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            <span>{uploadSuccess}</span>
          </div>
        )}

        {uploadError && (
          <div className="mt-4 p-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-xs flex items-center justify-center space-x-2 max-w-md mx-auto">
            <AlertTriangle className="w-4 h-4 text-red-600" />
            <span>{uploadError}</span>
          </div>
        )}
      </div>

      {/* Datasets Grid */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-slate-800">Registered Datasets ({datasets.length})</h2>
        </div>

        {datasets.length === 0 ? (
          <div className="glass-panel p-12 rounded-2xl text-center text-slate-500">
            No datasets uploaded yet. Upload a dataset or load the reference Bank Marketing data.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {datasets.map((ds) => {
              const profile = ds.profile;
              const probType = profile?.inferred_problem_type || 'classification';
              const target = profile?.candidate_targets?.[0] || 'Unknown';
              const alertsCount = profile?.quality_alerts?.length || 0;

              return (
                <div
                  key={ds.id}
                  className="glass-panel glass-panel-hover p-6 rounded-3xl flex flex-col justify-between space-y-5"
                >
                  <div className="space-y-3">
                    <div className="flex items-start justify-between">
                      <div className="space-y-1">
                        <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-md bg-slate-100 text-slate-700 border border-slate-200">
                          {ds.file_type.toUpperCase()}
                        </span>
                        <h3 className="font-bold text-base text-slate-900 line-clamp-1" title={ds.name}>
                          {ds.name}
                        </h3>
                      </div>
                      <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 capitalize">
                        {probType}
                      </span>
                    </div>

                    {/* Metadata Badges */}
                    <div className="grid grid-cols-2 gap-2 text-xs text-slate-600 pt-1">
                      <div className="flex items-center space-x-1.5">
                        <Layers className="w-3.5 h-3.5 text-indigo-600" />
                        <span><strong>{ds.row_count.toLocaleString()}</strong> rows</span>
                      </div>
                      <div className="flex items-center space-x-1.5">
                        <Hash className="w-3.5 h-3.5 text-indigo-600" />
                        <span><strong>{ds.col_count}</strong> columns</span>
                      </div>
                      <div className="flex items-center space-x-1.5">
                        <HardDrive className="w-3.5 h-3.5 text-indigo-600" />
                        <span>{(ds.size_bytes / (1024 * 1024)).toFixed(2)} MB</span>
                      </div>
                      <div className="flex items-center space-x-1.5">
                        <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
                        <span className="truncate">Target: <strong>{target}</strong></span>
                      </div>
                    </div>

                    {alertsCount > 0 && (
                      <div className="flex items-center space-x-1.5 text-xs text-amber-800 bg-amber-50 border border-amber-200 px-2.5 py-1.5 rounded-xl">
                        <AlertTriangle className="w-3.5 h-3.5 shrink-0 text-amber-600" />
                        <span>{alertsCount} quality alerts identified</span>
                      </div>
                    )}
                  </div>

                  {/* Action Buttons */}
                  <div className="pt-4 border-t border-slate-100 flex items-center justify-between space-x-2">
                    <Link
                      to={`/datasets/${ds.id}`}
                      className="flex-1 text-center px-3 py-2 rounded-xl text-xs font-semibold bg-slate-100 hover:bg-slate-200/80 text-slate-700 border border-slate-200 transition"
                    >
                      View Profile & SQL
                    </Link>
                    <Link
                      to={`/analysis?dataset_id=${ds.id}`}
                      className="flex items-center justify-center space-x-1.5 px-4 py-2 rounded-xl text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white shadow-2xs transition"
                    >
                      <Play className="w-3 h-3 fill-current" />
                      <span>Run DS</span>
                    </Link>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
