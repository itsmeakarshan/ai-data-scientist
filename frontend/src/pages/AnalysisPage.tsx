import React, { useEffect, useState, useRef } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import {
  PlayCircle,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Clock,
  ArrowRight,
  Loader2,
  XCircle,
  ShieldCheck,
  Award,
  Cpu,
  BarChart3,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { api } from '../services/api';
import { AnalysisRun, AnalysisStatus, Dataset, WorkflowProgressResponse } from '../types';

export const WORKFLOW_STAGES = [
  {
    num: 1,
    name: 'Dataset Inspection & Profiling',
    desc: 'Inspect schema, calculate distributions, detect missingness and column types',
    details: 'Sniff delimiter, compute column missingness, determine statistical profiles',
  },
  {
    num: 2,
    name: 'Problem Classification & Target Selection',
    desc: 'Infer problem formulation (classification vs regression vs time-series), target, and split strategy',
    details: 'Analyze target distribution, detect temporal sequencing, formulate validation protocol',
  },
  {
    num: 3,
    name: 'Autonomous Analysis Planning',
    desc: 'Generate candidate model strategy and validation protocol via Gemini 3.1',
    details: 'Synthesize multi-model candidate plan and quality audit protocol via Gemini 3.1',
  },
  {
    num: 4,
    name: 'Leak-Free Preprocessing & Splitting',
    desc: 'Execute fit-on-train encoding, imputing, and leak-free train/test partition',
    details: 'Fit imputers and encoders strictly on train set with zero data leakage',
  },
  {
    num: 5,
    name: 'Candidate Model Training & CV',
    desc: 'Train candidate models with stratified k-fold CV and MLflow logging',
    details: 'Execute stratified cross-validation and log metrics to SQLite MLflow store',
  },
  {
    num: 6,
    name: 'Multi-Metric Leaderboard Ranking',
    desc: 'Rank candidate models using cross-validation performance on the training portion',
    details: 'Rank candidate models using cross-validation performance on the training portion',
  },
  {
    num: 7,
    name: 'Methodological Critic Audit',
    desc: 'Audit for prospective data leakage, severe overfitting, and execute corrective retraining if needed',
    details: 'Audit prospective target leakage, test generalization gaps, and trigger corrective retrains',
  },
  {
    num: 8,
    name: 'SHAP Explainability & Feature Attribution',
    desc: 'Compute TreeSHAP attributions and generate diagnostic visualizations',
    details: 'Compute TreeSHAP attributions and generate ROC/PR and feature importance curves',
  },
  {
    num: 9,
    name: 'Evidence-Backed Report Synthesis',
    desc: 'Synthesize 4-pillar business insights and compile final Markdown report',
    details: 'Synthesize executive business insights and compile comprehensive audited report',
  },
];

const formatElapsed = (sec: number): string => {
  const mins = Math.floor(sec / 60);
  const secs = sec % 60;
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

export const AnalysisPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialDatasetId = searchParams.get('dataset_id') || '';
  const initialAnalysisId = searchParams.get('analysis_id') || '';

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetRuns, setDatasetRuns] = useState<AnalysisRun[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState(initialDatasetId);
  const [userGoal, setUserGoal] = useState('Predict target variable and identify primary predictive drivers.');
  const [targetColumn, setTargetColumn] = useState('');
  const [problemType, setProblemType] = useState('classification');
  const [timeColumn, setTimeColumn] = useState('');

  // Real-time workflow state
  const [running, setRunning] = useState(false);
  const [activeAnalysisId, setActiveAnalysisId] = useState<string | null>(initialAnalysisId || null);
  const [activeRun, setActiveRun] = useState<AnalysisRun | null>(null);
  const [liveStatus, setLiveStatus] = useState<AnalysisStatus | null>(null);
  const [workflowProgress, setWorkflowProgress] = useState<WorkflowProgressResponse | null>(null);
  const [status, setStatus] = useState<'READY' | 'IDLE' | 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED'>('READY');
  const [error, setError] = useState<string | null>(null);
  const [showErrorDetails, setShowErrorDetails] = useState(false);

  // Polling timer ref for safe cleanup
  const pollTimerRef = useRef<NodeJS.Timeout | null>(null);
  const hasInitializedRef = useRef(false);

  // Initial Data Load
  useEffect(() => {
    async function init() {
      if (hasInitializedRef.current) return;
      hasInitializedRef.current = true;
      try {
        const dsRes = await api.getDatasets();
        const items = dsRes?.items || [];
        setDatasets(items);
        if (items.length > 0) {
          const match = items.find((d) => d.id === initialDatasetId);
          setSelectedDatasetId(match ? match.id : items[0].id);
        }

        const runs = await api.getAnalysisRuns();
        setDatasetRuns(runs || []);

        if (initialAnalysisId) {
          try {
            const [progressData, runData] = await Promise.all([
              api.getAnalysisProgress(initialAnalysisId).catch(() => null),
              api.getAnalysis(initialAnalysisId).catch(() => null),
            ]);
            if (progressData) {
              setWorkflowProgress(progressData);
              setStatus(progressData.status as any);
            }
            if (runData) {
              setActiveRun(runData);
              if (!progressData) setStatus(runData.status as any);
            }
          } catch (err) {
            console.error('[AutoDS] Error loading initial analysis:', err);
          }
        }
      } catch (err) {
        console.error('[AutoDS] Failed to load initial data:', err);
      }
    }
    init();
  }, []);

  // Update form presets when dataset changes
  useEffect(() => {
    if (!selectedDatasetId) return;
    const selectedDs = datasets.find((d) => d.id === selectedDatasetId);
    if (!selectedDs) return;
    const nameLower = selectedDs.name.toLowerCase();
    if (nameLower.includes('bank')) {
      setUserGoal('Predict whether a telemarketing client will subscribe to a term deposit.');
      setTargetColumn('y');
      setProblemType('classification');
    } else if (nameLower.includes('cancer') || nameLower.includes('breast')) {
      setUserGoal('Classify breast cancer tumor samples as malignant or benign.');
      setTargetColumn('target');
      setProblemType('classification');
    } else if (nameLower.includes('housing')) {
      setUserGoal('Predict the median house value based on geographical and demographic features.');
      setTargetColumn('median_house_value');
      setProblemType('regression');
    } else if (nameLower.includes('m5')) {
      setUserGoal('Forecast daily sales demand across stores and categories.');
      setTargetColumn('sales');
      setTimeColumn('date');
      setProblemType('forecasting');
    } else if (nameLower.includes('diabetes')) {
      setUserGoal('Predict quantitative disease progression measure one year after baseline.');
      setTargetColumn('target');
      setProblemType('regression');
    } else if (nameLower.includes('wine')) {
      setUserGoal('Classify wine cultivar origin based on chemical constituents analysis.');
      setTargetColumn('target');
      setProblemType('classification');
    }
  }, [selectedDatasetId, datasets]);

  // Form Submit: Trigger Autonomous DS Pipeline Run
  const handleStartAnalysis = async (e: React.FormEvent) => {
    e.preventDefault();
    const effectiveDatasetId = selectedDatasetId || (datasets.length > 0 ? datasets[0].id : '');
    if (!effectiveDatasetId) {
      setError('Please select a dataset.');
      return;
    }

    if (!selectedDatasetId && effectiveDatasetId) {
      setSelectedDatasetId(effectiveDatasetId);
    }

    setRunning(true);
    setError(null);
    setShowErrorDetails(false);
    setStatus('RUNNING');
    setActiveRun(null);
    setLiveStatus(null);
    setWorkflowProgress(null);

    try {
      const response = await api.createAnalysis({
        dataset_id: effectiveDatasetId,
        user_goal: userGoal,
        target_column: targetColumn.trim() || undefined,
        problem_type: problemType || undefined,
        time_column: timeColumn.trim() || undefined,
      });

      console.log('[AutoDS] Created analysis ID:', response.id);
      setActiveAnalysisId(response.id);
      setActiveRun(response);
      setStatus((response.status as any) || 'RUNNING');

      setSearchParams({ dataset_id: effectiveDatasetId, analysis_id: response.id });
      if (typeof api.getAnalysisRuns === 'function') {
        api.getAnalysisRuns().then((runs) => setDatasetRuns(runs || [])).catch(() => {});
      }
    } catch (err: any) {
      console.error('[AutoDS] Error creating analysis:', err);
      setError(err.message || 'Analysis failed to execute.');
      setRunning(false);
      setStatus('FAILED');
    }
  };

  // Real-time Backend-Driven Polling: queries GET /api/analysis/{id}/progress every 1.5s
  useEffect(() => {
    if (!activeAnalysisId) return;
    if (status === 'COMPLETED' || status === 'FAILED') {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      return;
    }

    const checkBackendProgress = async () => {
      try {
        let progressData: any = null;
        if (api.getAnalysisProgress) {
          progressData = await api.getAnalysisProgress(activeAnalysisId).catch(() => null);
        }
        if (!progressData && api.getAnalysisStatus) {
          const st = await api.getAnalysisStatus(activeAnalysisId).catch(() => null);
          if (st) {
            setLiveStatus(st);
            progressData = {
              analysis_id: st.analysis_id,
              status: st.status,
              overall_status: st.status,
              current_stage: st.current_stage_name,
              current_stage_number: st.current_stage,
              total_stages: st.total_stages,
              completed_stages: st.completed_stages?.length || 0,
              progress_percentage: st.progress_percent,
              progress_percent: st.progress_percent,
              stage_status: st.status,
              elapsed_seconds: st.elapsed_seconds,
              error_message: st.error,
              error: st.error,
              stages: []
            };
          }
        }

        if (!progressData) return;

        setWorkflowProgress(progressData);
        setStatus(progressData.status as any);

        if (progressData.status === 'COMPLETED') {
          if (pollTimerRef.current) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
          }
          setRunning(false);
          const finalRun = await api.getAnalysis(activeAnalysisId).catch(() => null);
          if (finalRun) setActiveRun(finalRun);
          api.getAnalysisRuns().then((runs) => setDatasetRuns(runs || [])).catch(() => {});
        } else if (progressData.status === 'FAILED') {
          if (pollTimerRef.current) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
          }
          setRunning(false);
          setError(progressData.error_message || progressData.error || 'Autonomous analysis execution failed.');
          const finalRun = await api.getAnalysis(activeAnalysisId).catch(() => null);
          if (finalRun) setActiveRun(finalRun);
        }
      } catch (pollErr: any) {
        console.error('[AutoDS] Polling progress error:', pollErr);
      }
    };

    checkBackendProgress();
    pollTimerRef.current = setInterval(checkBackendProgress, 1500);

    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [activeAnalysisId, status]);

  const isCompleted = status === 'COMPLETED';
  const isFailed = status === 'FAILED';
  const isRunning = (status === 'RUNNING' || running) && !isCompleted && !isFailed;

  const selectedDs = datasets.find((d) => d.id === selectedDatasetId);
  const relevantRuns = datasetRuns.filter((r) => !selectedDatasetId || r.dataset_id === selectedDatasetId);

  // Backend-Driven progress properties (NO fake animation beyond backend report)
  const currentStageNum = workflowProgress?.current_stage_number || liveStatus?.current_stage || 1;
  const currentStageName = workflowProgress?.current_stage || liveStatus?.current_stage_name || WORKFLOW_STAGES[0].name;
  const completedStagesCount = workflowProgress?.completed_stages ?? liveStatus?.completed_stages?.length ?? 0;
  const progressPercentage = isCompleted
    ? 100
    : workflowProgress?.progress_percentage ?? workflowProgress?.progress_percent ?? liveStatus?.progress_percent ?? 0;
  const elapsedSeconds = workflowProgress?.elapsed_seconds ?? liveStatus?.elapsed_seconds ?? 0;

  const stageList = WORKFLOW_STAGES.map((def) => {
    const backendStage = workflowProgress?.stages?.find((s) => s.number === def.num);
    const stageStatus = backendStage?.status || (
      isCompleted
        ? 'COMPLETED'
        : isFailed && currentStageNum === def.num
        ? 'FAILED'
        : currentStageNum > def.num
        ? 'COMPLETED'
        : currentStageNum === def.num && isRunning
        ? 'RUNNING'
        : 'WAITING'
    );
    return {
      ...def,
      status: stageStatus,
      started_at: backendStage?.started_at,
      completed_at: backendStage?.completed_at,
      duration_seconds: backendStage?.duration_seconds,
    };
  });

  return (
    <div className="space-y-8 p-8 w-full max-w-7xl mx-auto">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight flex items-center gap-2.5">
          <Sparkles className="w-6 h-6 text-emerald-600" />
          Autonomous Data Science Engine
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          State your natural-language objective. AutoDS will autonomously plan, inspect, train multi-model candidate portfolios, audit methodology, and synthesize evidence-backed reports.
        </p>
      </div>

      {/* Main Grid: Config Form & Workflow Progress Stepper */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Config Form & Recent History */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-sm space-y-5">
            <h2 className="font-bold text-base text-slate-900 flex items-center gap-2">
              <PlayCircle className="w-5 h-5 text-emerald-600" />
              Configure Autonomous Run
            </h2>

            <form onSubmit={handleStartAnalysis} className="space-y-4">
              {/* Dataset Selector */}
              <div className="space-y-1.5">
                <label className="block text-xs font-bold text-slate-700">Target Dataset</label>
                <select
                  value={selectedDatasetId}
                  onChange={(e) => setSelectedDatasetId(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-2xl p-3 focus:outline-none focus:border-emerald-500 focus:bg-white transition"
                >
                  {datasets.map((ds) => (
                    <option key={ds.id} value={ds.id}>
                      {ds.name} ({ds.row_count?.toLocaleString()} rows, {ds.col_count} cols)
                    </option>
                  ))}
                </select>
              </div>

              {/* Natural Language Goal */}
              <div className="space-y-1.5">
                <label className="block text-xs font-bold text-slate-700">Natural Language Goal</label>
                <textarea
                  value={userGoal}
                  onChange={(e) => setUserGoal(e.target.value)}
                  rows={3}
                  className="w-full bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-2xl p-3 focus:outline-none focus:border-emerald-500 focus:bg-white transition"
                  placeholder="e.g. Predict customer subscription to term deposit and identify key drivers."
                />
              </div>

              {/* Problem Type & Target Grid */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label className="block text-xs font-bold text-slate-700">Problem Type</label>
                  <select
                    value={problemType}
                    onChange={(e) => setProblemType(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-2xl p-2.5 focus:outline-none focus:border-emerald-500 focus:bg-white transition"
                  >
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                    <option value="forecasting">Forecasting (Time Series)</option>
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="block text-xs font-bold text-slate-700">Target Column (Optional)</label>
                  <input
                    type="text"
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                    placeholder="Auto-detected if empty"
                    className="w-full bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-2xl p-2.5 focus:outline-none focus:border-emerald-500 focus:bg-white font-mono transition"
                  />
                </div>
              </div>

              {problemType === 'forecasting' && (
                <div className="space-y-1.5">
                  <label className="block text-xs font-bold text-slate-700">Time / Date Column</label>
                  <input
                    type="text"
                    value={timeColumn}
                    onChange={(e) => setTimeColumn(e.target.value)}
                    placeholder="e.g. date, timestamp"
                    className="w-full bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-2xl p-2.5 focus:outline-none focus:border-emerald-500 focus:bg-white font-mono transition"
                  />
                </div>
              )}

              {error && (
                <div className="p-3.5 rounded-2xl bg-red-50 border border-red-200 text-red-700 text-xs flex flex-col space-y-2">
                  <div className="flex items-center space-x-2 font-semibold">
                    <AlertTriangle className="w-4 h-4 shrink-0 text-red-600" />
                    <span>Autonomous Execution Error</span>
                  </div>
                  <p className="text-xs text-red-600 pl-6">{error}</p>
                </div>
              )}

              <button
                type="submit"
                onClick={handleStartAnalysis}
                disabled={isRunning}
                className="w-full py-3.5 px-4 rounded-2xl text-xs font-bold bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-md shadow-emerald-600/20 transition flex items-center justify-center space-x-2 disabled:opacity-60 cursor-pointer"
              >
                {isRunning ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Autonomous Pipeline Running...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Launch Autonomous DS Run</span>
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Historical Runs List */}
          {relevantRuns.length > 0 && (
            <div className="bg-white border border-slate-200 p-5 rounded-3xl space-y-3 shadow-sm">
              <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-slate-500" />
                <span>Historical Analyses for Selected Dataset ({relevantRuns.length})</span>
              </h3>
              <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                {relevantRuns.slice(0, 6).map((r) => (
                  <button
                    key={r.id}
                    type="button"
                    onClick={() => {
                      setActiveAnalysisId(r.id);
                      setActiveRun(r);
                      setStatus(r.status as any);
                      setSearchParams({ dataset_id: selectedDatasetId, analysis_id: r.id });
                    }}
                    className={`w-full text-left p-3 rounded-xl border text-xs transition flex items-center justify-between ${
                      activeAnalysisId === r.id
                        ? 'bg-emerald-50/80 border-emerald-300 text-emerald-950 font-semibold ring-1 ring-emerald-400/40'
                        : 'bg-slate-50/70 border-slate-200 text-slate-700 hover:bg-slate-100'
                    }`}
                  >
                    <div className="truncate pr-2">
                      <p className="truncate font-medium text-slate-800">{r.user_goal}</p>
                      <p className="text-[10px] text-slate-400 font-mono mt-0.5">
                        {new Date(r.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} • {r.problem_type}
                      </p>
                    </div>
                    <span
                      className={`text-[10px] font-bold px-2.5 py-0.5 rounded-full uppercase shrink-0 ${
                        r.status === 'COMPLETED'
                          ? 'bg-emerald-100 text-emerald-800'
                          : r.status === 'RUNNING'
                          ? 'bg-emerald-100 text-emerald-800 animate-pulse'
                          : 'bg-red-100 text-red-800'
                      }`}
                    >
                      {r.status}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Production Real-Time Autonomous Progress System */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-white border border-slate-200 rounded-3xl p-6 shadow-sm space-y-6">
            {/* Top Workflow Panel Header */}
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <span className="text-[11px] font-extrabold uppercase tracking-widest text-slate-400">
                    WORKFLOW EXECUTION PLAN
                  </span>
                  <h2 className="text-lg font-bold text-slate-900 mt-0.5">
                    {selectedDs?.name || 'Dataset Analysis'}
                  </h2>
                </div>

                {/* Status Badge */}
                <div className="flex items-center gap-2">
                  {isRunning ? (
                    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-extrabold bg-emerald-100 text-emerald-800 border border-emerald-300">
                      <span className="w-2 h-2 rounded-full bg-emerald-600 animate-ping" />
                      ▶ RUNNING
                    </span>
                  ) : isCompleted ? (
                    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-extrabold bg-emerald-100 text-emerald-800 border border-emerald-300">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                      ✓ COMPLETED
                    </span>
                  ) : isFailed ? (
                    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-extrabold bg-red-100 text-red-800 border border-red-300">
                      <XCircle className="w-3.5 h-3.5 text-red-600" />
                      ✕ FAILED
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-slate-100 text-slate-600 border border-slate-200">
                      ○ READY
                    </span>
                  )}
                </div>
              </div>

              {/* Natural Language Goal / Objective */}
              <div className="p-3.5 rounded-2xl bg-slate-50 border border-slate-200 text-xs text-slate-700 flex items-start gap-2.5">
                <span className="font-bold text-slate-900 shrink-0">Objective:</span>
                <span className="italic">{userGoal}</span>
              </div>

              {/* Real-Time Backend-Driven Progress Card */}
              <div className="space-y-2 bg-slate-50/80 p-4 rounded-2xl border border-slate-200">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2 font-bold text-slate-800">
                    <span>Overall Progress</span>
                    <span className="text-emerald-700 font-mono text-sm">{Math.round(progressPercentage)}%</span>
                  </div>
                  <div className="flex items-center gap-3 text-slate-500 font-mono text-[11px]">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3 text-slate-400" />
                      Elapsed: {formatElapsed(elapsedSeconds)}
                    </span>
                    <span>Stage {currentStageNum} / 9</span>
                  </div>
                </div>

                {/* Progress Bar (Strict Backend State, smooth transition between updates) */}
                <div
                  role="progressbar"
                  aria-valuenow={progressPercentage}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  className="w-full bg-slate-200 h-2.5 rounded-full overflow-hidden relative shadow-inner"
                >
                  <div
                    className={`h-full transition-all duration-500 rounded-full ${
                      isCompleted
                        ? 'bg-emerald-600'
                        : isFailed
                        ? 'bg-red-500'
                        : 'bg-gradient-to-r from-emerald-500 to-teal-500'
                    }`}
                    style={{ width: `${Math.max(progressPercentage, isRunning ? 2 : 0)}%` }}
                  />
                </div>

                <div className="flex items-center justify-between text-[11px] text-slate-500 pt-0.5">
                  <span className="truncate max-w-[80%] font-medium text-slate-700">
                    Current stage: <strong className="text-slate-900">{isCompleted ? 'Evidence-Backed Report Synthesized' : currentStageName}</strong>
                  </span>
                  <span className="shrink-0 text-slate-400 font-mono">
                    {completedStagesCount} of 9 completed
                  </span>
                </div>
              </div>
            </div>

            {/* Error Banner when FAILED */}
            {isFailed && (
              <div className="p-4 rounded-2xl bg-red-50 border border-red-200 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-red-800 font-bold text-xs">
                    <XCircle className="w-4 h-4 text-red-600" />
                    <span>AUTONOMOUS DS RUN FAILED</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowErrorDetails(!showErrorDetails)}
                    className="text-[11px] font-semibold text-red-700 hover:text-red-900 flex items-center gap-1"
                  >
                    <span>{showErrorDetails ? 'Hide Diagnostics' : 'Inspect Error'}</span>
                    {showErrorDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                  </button>
                </div>

                <div className="text-xs text-red-800 font-semibold pt-1">
                  Stage {currentStageNum} of 9: <span className="font-bold">{currentStageName}</span>
                </div>

                <p className="text-xs text-red-700 bg-white/60 p-2.5 rounded-xl border border-red-200">
                  {error || workflowProgress?.error_message || workflowProgress?.error || activeRun?.error_message || 'An unexpected error occurred during execution.'}
                </p>

                {showErrorDetails && (
                  <div className="p-3 rounded-xl bg-white border border-red-200 font-mono text-[11px] text-red-900 overflow-x-auto space-y-1">
                    <p>Analysis ID: {activeAnalysisId}</p>
                    <p>Failed Stage Number: {currentStageNum}</p>
                    <p>Failed Stage Name: {currentStageName}</p>
                    <p>Error Message: {error || workflowProgress?.error_message || activeRun?.error_message}</p>
                  </div>
                )}
              </div>
            )}

            {/* 9 Backend-Driven Workflow Stages */}
            <div className="space-y-3" aria-live="polite">
              {stageList.map((step) => {
                const isStepCompleted = step.status === 'COMPLETED';
                const isStepRunning = step.status === 'RUNNING';
                const isStepFailed = step.status === 'FAILED';
                const isStepWaiting = step.status === 'WAITING';

                return (
                  <div
                    key={step.num}
                    className={`p-4 rounded-2xl border transition-all duration-300 ${
                      isStepRunning
                        ? 'bg-emerald-50/90 border-emerald-400 ring-1 ring-emerald-400/40 shadow-sm'
                        : isStepCompleted
                        ? 'bg-emerald-50/40 border-emerald-200 text-slate-800'
                        : isStepFailed
                        ? 'bg-red-50/80 border-red-300 text-red-900'
                        : 'bg-slate-50/50 border-slate-200/80 text-slate-400'
                    }`}
                  >
                    <div className="flex items-start space-x-3.5">
                      {/* State Indicator Icon */}
                      <div className="shrink-0 mt-0.5">
                        {isStepCompleted ? (
                          <div className="w-6 h-6 rounded-full bg-emerald-600 text-white flex items-center justify-center shadow-xs font-bold text-xs">
                            ✓
                          </div>
                        ) : isStepRunning ? (
                          <div className="w-6 h-6 rounded-full bg-emerald-100 border border-emerald-400 text-emerald-700 flex items-center justify-center font-bold text-xs animate-pulse">
                            ▶
                          </div>
                        ) : isStepFailed ? (
                          <div className="w-6 h-6 rounded-full bg-red-600 text-white flex items-center justify-center font-bold text-xs">
                            ✕
                          </div>
                        ) : (
                          <div className="w-6 h-6 rounded-full bg-white border border-slate-300 flex items-center justify-center text-[11px] font-medium text-slate-400 shadow-2xs">
                            ○
                          </div>
                        )}
                      </div>

                      {/* Stage Content */}
                      <div className="flex-1 space-y-1">
                        <div className="flex items-center justify-between">
                          <h4
                            className={`text-xs font-bold ${
                              isStepRunning
                                ? 'text-emerald-950'
                                : isStepCompleted
                                ? 'text-slate-900'
                                : isStepFailed
                                ? 'text-red-900'
                                : 'text-slate-500'
                            }`}
                          >
                            {step.num}. {step.name}
                            {step.duration_seconds !== undefined && step.duration_seconds !== null && (
                              <span className="text-[10px] text-slate-400 font-mono ml-2 font-normal">
                                ({step.duration_seconds}s)
                              </span>
                            )}
                          </h4>

                          {/* Stage Status Badge */}
                          {isStepRunning && (
                            <span className="text-[10px] font-extrabold px-2 py-0.5 rounded-full bg-emerald-200/80 text-emerald-900 uppercase tracking-wide flex items-center gap-1 animate-pulse">
                              <span className="w-1.5 h-1.5 rounded-full bg-emerald-700 animate-ping" />
                              ● LIVE
                            </span>
                          )}
                          {isStepCompleted && (
                            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 uppercase tracking-wide">
                              Completed
                            </span>
                          )}
                          {isStepFailed && (
                            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-red-100 text-red-800 uppercase tracking-wide">
                              Failed
                            </span>
                          )}
                          {isStepWaiting && (
                            <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-slate-100 text-slate-400 uppercase tracking-wide">
                              Waiting
                            </span>
                          )}
                        </div>

                        <p className={`text-[11px] ${isStepRunning ? 'text-emerald-900/80 font-medium' : 'text-slate-500'}`}>
                          {step.desc}
                        </p>

                        {isStepRunning && (
                          <div className="space-y-1.5 pt-1.5">
                            <div className="text-[11px] text-emerald-900 font-semibold flex items-center gap-1.5">
                              <Loader2 className="w-3 h-3 animate-spin text-emerald-700" />
                              <span>{liveStatus?.stage_details || step.details}</span>
                            </div>
                            {liveStatus?.models_evaluated && liveStatus.models_evaluated.length > 0 && (
                              <div className="flex flex-wrap gap-1.5 pt-1">
                                {liveStatus.models_evaluated.map((m) => (
                                  <span key={m} className="px-2 py-0.5 rounded-md text-[10px] font-mono bg-emerald-100/90 text-emerald-900 border border-emerald-300">
                                    {m}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Completion Summary Banner */}
            {isCompleted && (
              <div className="p-5 rounded-3xl bg-gradient-to-br from-emerald-50 via-teal-50/50 to-white border border-emerald-200 space-y-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Award className="w-5 h-5 text-emerald-600" />
                    <h3 className="text-sm font-bold text-slate-900">AUTONOMOUS DS RUN COMPLETE</h3>
                  </div>
                  <span className="text-[11px] font-extrabold px-2.5 py-0.5 rounded-full bg-emerald-200 text-emerald-900 font-mono">
                    100% AUDITED
                  </span>
                </div>

                {activeRun && (
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
                    <div className="p-2.5 rounded-xl bg-white border border-slate-200">
                      <span className="text-[10px] font-bold uppercase text-slate-400 block">Problem Type</span>
                      <span className="font-semibold text-slate-800 capitalize">{activeRun.problem_type}</span>
                    </div>

                    <div className="p-2.5 rounded-xl bg-white border border-slate-200">
                      <span className="text-[10px] font-bold uppercase text-slate-400 block">Target Column</span>
                      <span className="font-mono font-semibold text-slate-800">{activeRun.target_column || 'None (EDA)'}</span>
                    </div>

                    <div className="p-2.5 rounded-xl bg-white border border-slate-200">
                      <span className="text-[10px] font-bold uppercase text-slate-400 block">Validation Strategy</span>
                      <span className="font-semibold text-slate-800 font-mono text-[11px]">{activeRun.validation_strategy}</span>
                    </div>
                  </div>
                )}

                {/* Executive Action Links */}
                <div className="pt-2 flex flex-wrap items-center justify-end gap-2.5">
                  <Link
                    to={`/reports/${activeAnalysisId}`}
                    className="px-4 py-2 rounded-xl text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm transition flex items-center gap-1.5 cursor-pointer"
                  >
                    <FileText className="w-3.5 h-3.5" />
                    <span>View Report</span>
                    <ArrowRight className="w-3 h-3" />
                  </Link>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
