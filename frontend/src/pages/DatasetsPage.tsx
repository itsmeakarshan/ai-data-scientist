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
  CheckCircle2,
  Trash2
} from 'lucide-react';
import { api } from '../services/api';
import { Dataset } from '../types';

export const DatasetsPage: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [datasetToDelete, setDatasetToDelete] = useState<Dataset | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
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

  const confirmDeleteDataset = async () => {
    if (!datasetToDelete) return;
    setIsDeleting(true);
    try {
      await api.deleteDataset(datasetToDelete.id);
      setDatasets((prev) => prev.filter((d) => d.id !== datasetToDelete.id));
      setDatasetToDelete(null);
    } catch (err: any) {
      console.error('Failed to delete dataset:', err);
      alert(err.message || 'Failed to delete dataset. Please try again.');
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="space-y-8 p-6 md:p-8 w-full">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-slate-900 tracking-tight">Dataset Ingestion & Management</h1>
          <p className="text-sm text-slate-500 mt-1">
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
          <div className="glass-panel p-12 rounded-3xl text-center text-slate-500">
            No datasets uploaded yet. Upload a dataset to begin autonomous data science profiling.
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
                  className="glass-panel glass-panel-hover p-6 rounded-3xl flex flex-col justify-between space-y-5 relative group/card"
                >
                  <div className="space-y-3">
                    <div className="flex items-start justify-between gap-2">
                      <div className="space-y-1 flex-1 min-w-0">
                        <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-md bg-slate-100 text-slate-700 border border-slate-200">
                          {ds.file_type.toUpperCase()}
                        </span>
                        <h3 className="font-extrabold text-base text-slate-900 leading-snug break-words" title={ds.name}>
                          {ds.name}
                        </h3>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 capitalize">
                          {probType}
                        </span>
                        <button
                          type="button"
                          onClick={() => setDatasetToDelete(ds)}
                          className="p-1.5 rounded-xl hover:bg-rose-100/80 text-slate-400 hover:text-rose-600 transition cursor-pointer"
                          title="Delete dataset"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>

                    {/* Metadata Badges */}
                    <div className="grid grid-cols-2 gap-2 text-xs text-slate-600 pt-1 font-sans">
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

      {/* Delete Confirmation Modal */}
      {datasetToDelete && (
        <div
          className="fixed inset-0 z-50 bg-slate-900/80 backdrop-blur-md flex items-center justify-center p-4 animate-in fade-in duration-150"
          onClick={() => setDatasetToDelete(null)}
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
              <p className="font-bold text-slate-900 leading-snug break-words">{datasetToDelete.name}</p>
              <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-500 font-mono pt-1">
                <span>Rows: {datasetToDelete.row_count.toLocaleString()}</span>
                <span>Cols: {datasetToDelete.col_count}</span>
                <span>Size: {(datasetToDelete.size_bytes / (1024 * 1024)).toFixed(2)} MB</span>
                <span>Type: {datasetToDelete.file_type.toUpperCase()}</span>
              </div>
            </div>

            <p className="text-xs text-slate-600 leading-relaxed">
              Are you sure you want to delete this dataset? This will completely remove the dataset, its statistical profiles, and all associated metadata from the database.
            </p>

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setDatasetToDelete(null)}
                disabled={isDeleting}
                className="px-4 py-2.5 rounded-xl text-xs font-bold bg-slate-100 hover:bg-slate-200 text-slate-700 transition cursor-pointer disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={confirmDeleteDataset}
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
