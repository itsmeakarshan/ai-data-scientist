import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Trash2, AlertTriangle, FileText, Calendar, ChevronRight } from 'lucide-react';
import { api } from '../services/api';
import { Report } from '../types';
import { ReportViewer } from '../components/ReportViewer';

export const ReportsPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [reports, setReports] = useState<Report[]>([]);
  const [activeReport, setActiveReport] = useState<Report | null>(null);
  const [reportToDelete, setReportToDelete] = useState<Report | null>(null);
  const [isDeleting, setIsDeleting] = useState<boolean>(false);

  useEffect(() => {
    async function load() {
      try {
        if (id) {
          const rep = await api.getReport(id);
          setActiveReport(rep);
        } else {
          const list = await api.getReports();
          setReports(list || []);
          if (list && list.length > 0) {
            setActiveReport(list[0]);
          }
        }
      } catch (err) {
        console.error('Failed to load reports:', err);
      }
    }
    load();
  }, [id]);

  const confirmDeleteReport = async () => {
    if (!reportToDelete) return;
    setIsDeleting(true);
    try {
      await api.deleteReport(reportToDelete.id);
      const updatedReports = reports.filter((r) => r.id !== reportToDelete.id);
      setReports(updatedReports);

      if (activeReport?.id === reportToDelete.id) {
        setActiveReport(updatedReports.length > 0 ? updatedReports[0] : null);
      }
      setReportToDelete(null);
    } catch (err) {
      console.error('Failed to delete report:', err);
      alert('Failed to delete report. Please try again.');
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="space-y-8 p-6 md:p-8 w-full">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-slate-900 tracking-tight">Evidence-Backed Data Science Reports</h1>
          <p className="text-sm text-slate-500 mt-1">
            Audit-ready Machine Learning evaluation reports separating Observed Facts, Model Evidence, Actionable Recommendations, and Causal Constraints.
          </p>
        </div>
      </div>

      {reports.length === 0 && !activeReport ? (
        <div className="glass-panel p-12 rounded-3xl text-center text-slate-500 space-y-3">
          <FileText className="w-10 h-10 text-slate-400 mx-auto" />
          <h3 className="text-base font-bold text-slate-800">No Reports Available</h3>
          <p className="text-xs text-slate-500">Execute an autonomous Data Science run to generate evidence-backed reports.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Left: Reports Selector Sidebar (if viewing list) */}
          {!id && reports.length > 1 && (
            <div className="lg:col-span-4 space-y-4">
              <div className="flex items-center justify-between px-1">
                <h2 className="text-xs font-extrabold uppercase tracking-wider text-slate-500">
                  Available Reports ({reports.length})
                </h2>
              </div>
              <div className="space-y-3 max-h-[calc(100vh-220px)] overflow-y-auto pr-1">
                {reports.map((rep) => {
                  const isActive = activeReport?.id === rep.id;
                  return (
                    <div
                      key={rep.id}
                      onClick={() => setActiveReport(rep)}
                      className={`p-4 rounded-2xl border cursor-pointer transition flex flex-col justify-between space-y-3 ${
                        isActive
                          ? 'bg-gradient-to-r from-indigo-50/90 via-white to-indigo-50/40 border-indigo-300 shadow-md shadow-indigo-500/10 text-slate-900 ring-1 ring-indigo-200'
                          : 'bg-white border-slate-200 text-slate-700 hover:border-slate-300 hover:shadow-2xs'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="space-y-1 flex-1 min-w-0">
                          <h3 className="font-extrabold text-sm text-slate-900 leading-snug break-words">
                            {rep.title}
                          </h3>
                          <p className="text-xs text-slate-500 leading-relaxed line-clamp-2 break-words">
                            {rep.summary_markdown || 'Autonomous data science execution report'}
                          </p>
                        </div>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setReportToDelete(rep);
                          }}
                          className="p-1.5 rounded-xl hover:bg-rose-100/70 text-slate-400 hover:text-rose-600 transition shrink-0 cursor-pointer"
                          title="Delete report"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>

                      <div className="flex items-center justify-between pt-2 border-t border-slate-100 text-[11px] text-slate-400 font-mono">
                        <span className="flex items-center gap-1 text-slate-500">
                          <Calendar className="w-3 h-3 text-slate-400" />
                          {new Date(rep.created_at).toLocaleDateString()} • {new Date(rep.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                        {isActive && (
                          <span className="inline-flex items-center gap-0.5 text-xs font-bold text-indigo-600 font-sans">
                            Viewing <ChevronRight className="w-3 h-3" />
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Right: Active Report Viewer */}
          <div className={`${!id && reports.length > 1 ? 'lg:col-span-8' : 'lg:col-span-12'}`}>
            {activeReport ? (
              <ReportViewer
                report={activeReport}
                onDeleteReport={(rep) => setReportToDelete(rep)}
              />
            ) : (
              <div className="glass-panel p-12 rounded-3xl text-center text-slate-500">
                Select a report from the list to view its complete evidence-backed audit details.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {reportToDelete && (
        <div
          className="fixed inset-0 z-50 bg-slate-900/80 backdrop-blur-md flex items-center justify-center p-4 animate-in fade-in duration-150"
          onClick={() => setReportToDelete(null)}
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
                <h3 className="font-extrabold text-slate-900 text-lg">Delete Report?</h3>
                <p className="text-xs text-slate-500">This action cannot be undone.</p>
              </div>
            </div>

            <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200 text-xs space-y-1.5">
              <span className="text-[10px] font-extrabold uppercase text-slate-400 block">Report Target</span>
              <p className="font-bold text-slate-900 leading-snug break-words">{reportToDelete.title}</p>
              <p className="text-[11px] font-mono text-slate-500">
                ID: {reportToDelete.id.substring(0, 8)} • {new Date(reportToDelete.created_at).toLocaleString()}
              </p>
            </div>

            <p className="text-xs text-slate-600 leading-relaxed">
              Are you sure you want to delete this report? It will be permanently removed from the database.
            </p>

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setReportToDelete(null)}
                disabled={isDeleting}
                className="px-4 py-2.5 rounded-xl text-xs font-bold bg-slate-100 hover:bg-slate-200 text-slate-700 transition cursor-pointer disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={confirmDeleteReport}
                disabled={isDeleting}
                className="px-5 py-2.5 rounded-xl text-xs font-extrabold bg-rose-600 hover:bg-rose-700 text-white shadow-md shadow-rose-600/20 transition cursor-pointer disabled:opacity-50 inline-flex items-center gap-2"
              >
                <Trash2 className="w-4 h-4" />
                <span>{isDeleting ? 'Deleting...' : 'Yes, Delete Report'}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
