import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { ReportViewer } from '../components/ReportViewer';
import { Report } from '../types';

const mockReport: Report = {
  id: 'rep-uuid-123',
  analysis_id: 'analysis-uuid-123',
  title: 'AutoDS Analysis Report — Bank_Marketing_UCI',
  summary_markdown: 'Completed classification pipeline for goal "Predict subscription". Best Model: LightGBM.',
  full_report_markdown: `# AutoDS Autonomous Data Science Report
**Dataset:** \`Bank_Marketing_UCI\`  
**Objective:** Predict customer subscription to term deposit.  

The champion model selected is **LightGBM**, achieving a Test **ROC-AUC of 0.8121**, **PR-AUC of 0.4829**, **Balanced Accuracy of 0.7619**, and **Positive-Class F1 of 0.4862**.

## 2. Dataset Overview & Data Quality Profile (Observed Facts)
Total rows: 4521

## 3. Model Leaderboard & Multi-Metric Evaluation
| Model | Task | Validation Strategy | Primary Metric | ROC-AUC | PR-AUC | Balanced Accuracy | F1 Positive |
|---|---|---|---|---|---|---|---|
| **LightGBM** (Champion) | classification | Stratified 5-Fold CV | 0.7977 | 0.8121 | 0.4829 | 0.7619 | 0.4862 |
| **RandomForest** | classification | Stratified 5-Fold CV | 0.7854 | 0.7950 | 0.4510 | 0.7320 | 0.4610 |

## 4. Threshold Optimization & Trade-Off Analysis (OOF Validated)
| Decision Cutoff | Objective / Role | Positive Precision | Positive Recall | Positive F1 |
|---|---|---|---|---|
| **0.32** | Selected operating threshold: 0.32 | 45.2% | 68.4% | 0.5440 |

## 5. Methodological Critic Audit
All Methodological Critic Rules Passed Cleanly.

## 6. Model Explainability & Feature Importance
| Rank | Feature Name | Relative Importance | Attribution Direction |
|---|---|---|---|
| 1 | \`pdays\` | 34.2% | Negative |
| 2 | \`age\` | 28.5% | Positive |

## 7. Evidence-Backed Business Insights
Observed fact: Direct contact marketing campaigns.

## 8. Operational Risk Analysis & Deployment Boundaries
- Data Drift: Monitor age distribution quarterly.
`,
  business_insights_json: {
    insights: [
      {
        category: 'fact',
        title: 'Observed Campaign Volume',
        finding: '4,521 total client contact records were audited with 11.7% historical positive response rate.',
        evidence: 'Dataset summary and profile metrics',
        confidence: 'HIGH',
      },
      {
        category: 'model_evidence',
        title: 'Model Discriminative Power',
        finding: 'LightGBM achieved ROC-AUC 0.8121 and PR-AUC 0.4829 on the untouched holdout set.',
        evidence: 'Holdout evaluation metrics',
        confidence: 'HIGH',
      },
      {
        category: 'recommendation',
        title: 'Calibrated Prospect Prioritization',
        finding: 'Deploy operating threshold of 0.32 to capture 68.4% of total potential conversions.',
        evidence: 'Out-of-fold threshold optimization',
        confidence: 'HIGH',
      },
      {
        category: 'causal_limitation',
        title: 'Non-Causal Statistical Association',
        finding: 'Feature attributions represent model predictive signal, not causal intervention guarantees.',
        evidence: 'Methodological critic audit',
        confidence: 'HIGH',
      },
    ],
  },
  methodology_json: {
    plan: 'Stratified 5-Fold Cross-Validation with OOF Threshold Tuning',
    critic: {
      audit_status: 'PASSED',
      findings: [],
    },
  },
  artifact_paths: [
    'reports/artifacts/analysis-uuid-123_LightGBM_roc.png',
    'reports/artifacts/analysis-uuid-123_LightGBM_pr.png',
    'reports/artifacts/analysis-uuid-123_LightGBM_cm.png',
    'reports/artifacts/analysis-uuid-123_LightGBM_feature_imp.png',
  ],
  created_at: '2026-08-17T12:00:00Z',
};

describe('ReportViewer Visual Diagnostics & 4-Pillar Architecture', () => {
  it('renders all four model diagnostic visualizations with champion model names in titles', () => {
    render(
      <BrowserRouter>
        <ReportViewer report={mockReport} />
      </BrowserRouter>
    );

    // 1. Verify "Generated Visual Diagnostics" subsection heading is present in default PDF view
    expect(screen.getByText(/5\. Generated Visual Diagnostics/i)).toBeDefined();

    // 2. Verify all 4 diagnostic plot titles (clean and concise)
    expect(screen.getByText(/LightGBM — ROC Curve/i)).toBeDefined();
    expect(screen.getByText(/LightGBM — Precision-Recall Curve/i)).toBeDefined();
    expect(screen.getByText(/LightGBM — Confusion Matrix/i)).toBeDefined();
    expect(screen.getByText(/LightGBM — Top Predictive Drivers/i)).toBeDefined();

    // 3. Verify non-causal disclaimer on feature importance
    expect(screen.getByText(/\* These are model-derived predictive associations, not causal effects\./i)).toBeDefined();
  });

  it('renders all four 4-Pillar business insights cards correctly', () => {
    render(
      <BrowserRouter>
        <ReportViewer report={mockReport} />
      </BrowserRouter>
    );

    // Switch to Interactive Tabs view
    const interactiveBtn = screen.getByRole('button', { name: /Interactive Tabs/i });
    fireEvent.click(interactiveBtn);

    // Verify all 4 pillars exist
    expect(screen.getByText(/1\. Observed Fact/i)).toBeDefined();
    expect(screen.getByText(/2\. Model-Derived Evidence/i)).toBeDefined();
    expect(screen.getByText(/3\. Actionable Recommendation/i)).toBeDefined();
    expect(screen.getByText(/4\. Causal Limitation/i)).toBeDefined();

    // Verify finding titles
    expect(screen.getByText(/Observed Campaign Volume/i)).toBeDefined();
    expect(screen.getByText(/Model Discriminative Power/i)).toBeDefined();
    expect(screen.getByText(/Calibrated Prospect Prioritization/i)).toBeDefined();
    expect(screen.getByText(/Non-Causal Statistical Association/i)).toBeDefined();
  });

  it('opens and closes fullscreen lightbox modal when user expands a plot', () => {
    render(
      <BrowserRouter>
        <ReportViewer report={mockReport} />
      </BrowserRouter>
    );

    const expandButtons = screen.getAllByRole('button', { name: /Expand/i });
    expect(expandButtons.length).toBe(4);

    // Click first expand button
    fireEvent.click(expandButtons[0]);

    // Modal should appear with Close and Download Image buttons
    expect(screen.getByRole('button', { name: /✕ Close/i })).toBeDefined();
    expect(screen.getByText(/Download Image/i)).toBeDefined();

    // Close modal
    fireEvent.click(screen.getByRole('button', { name: /✕ Close/i }));
    expect(screen.queryByRole('button', { name: /✕ Close/i })).toBeNull();
  });

  it('renders classification section for classification reports and hides it for regression and forecasting', () => {
    const regressionReport: Report = {
      ...mockReport,
      id: 'rep-reg-123',
      title: 'AutoDS Analysis Report — California_Housing',
      summary_markdown: 'Completed regression pipeline for goal "Predict median house value". Best Model: XGBoost.',
      full_report_markdown: `# AutoDS Autonomous Data Science Report
**Dataset:** \`California_Housing\`  
**Task Type:** regression  

The champion model selected is **XGBoost**, achieving a Test **RMSE of 0.45**, **MAE of 0.32**, and **R² of 0.81**.

## 2. Dataset Overview & Data Quality Profile (Observed Facts)
Total rows: 20640

## 3. Model Leaderboard & Multi-Metric Evaluation
| Model Name | Primary Loss Metric (CV RMSE) | CV Std | Train Time (s) | Model Family | Status |
|---|---|---|---|---|---|
| \`XGBoost\` | CV: 0.4612 | ±0.0120 | 2.10s | Gradient Boosting | **Champion** |

## 5. Methodological Critic Audit
All Methodological Critic Rules Passed Cleanly.
`,
      methodology_json: {
        problem_type: 'regression',
        plan: '5-Fold CV Regression',
      },
      artifact_paths: [
        'reports/artifacts/rep-reg-123_XGBoost_actual_vs_pred.png',
        'reports/artifacts/rep-reg-123_XGBoost_residuals.png',
        'reports/artifacts/rep-reg-123_XGBoost_feature_imp.png',
      ],
    };

    const forecastingReport: Report = {
      ...regressionReport,
      id: 'rep-fore-123',
      summary_markdown: 'Completed forecasting pipeline. Best Model: LightGBM.',
      full_report_markdown: `# AutoDS Autonomous Data Science Report
**Task Type:** forecasting  
The champion model selected is **LightGBM**.
`,
      methodology_json: {
        problem_type: 'forecasting',
      },
    };

    const multiclassReport: Report = {
      ...regressionReport,
      id: 'rep-multi-123',
      title: 'AutoDS Analysis Report — Wine_Quality',
      summary_markdown: 'Completed multiclass classification pipeline. Best Model: RandomForest.',
      full_report_markdown: `# AutoDS Autonomous Data Science Report
**Dataset:** \`Wine_Quality\`  
**Task Type:** classification  

The champion model selected is **RandomForest**, achieving a Test **Macro F1 of 0.7241**, **Macro ROC-AUC of 0.8842**, **Macro PR-AUC of 0.7102**, and **Balanced Accuracy of 0.7015**.

## 2. Dataset Overview & Data Quality Profile (Observed Facts)
Total rows: 4898
Target Column: \`quality\`

## 4. Multi-Class Diagnostic Evaluation & Class Breakdown
Evaluated Target Classes (6): 3, 4, 5, 6, 7, 8
`,
      methodology_json: {
        problem_type: 'classification',
        is_binary: false,
      },
      artifact_paths: [
        'reports/artifacts/rep-multi-123_RandomForest_cm.png',
        'reports/artifacts/rep-multi-123_RandomForest_feature_imp.png',
      ],
    };

    // A. Binary classification report: threshold section rendered
    const { unmount: unmountClass } = render(
      <BrowserRouter>
        <ReportViewer report={mockReport} />
      </BrowserRouter>
    );
    expect(screen.getByText(/3\. Classification Threshold Selection & Touchless Holdout Analysis/i)).toBeDefined();
    unmountClass();

    // B. Multiclass classification report: threshold section NOT rendered, multiclass metrics present
    const { unmount: unmountMulti } = render(
      <BrowserRouter>
        <ReportViewer report={multiclassReport} />
      </BrowserRouter>
    );
    expect(screen.queryByText(/3\. Classification Threshold Selection & Touchless Holdout Analysis/i)).toBeNull();
    expect(screen.queryByText(/0\.50 cutoff/i)).toBeNull();
    expect(screen.queryByText(/positive recall/i)).toBeNull();
    expect(screen.getByText(/Class Selection Strategy/i)).toBeDefined();
    unmountMulti();

    // C. Regression report: threshold section NOT rendered, titles correct, no classification terminology/stale values
    const { unmount: unmountReg } = render(
      <BrowserRouter>
        <ReportViewer report={regressionReport} />
      </BrowserRouter>
    );
    expect(screen.queryByText(/3\. Classification Threshold Selection & Touchless Holdout Analysis/i)).toBeNull();
    expect(screen.queryByText(/Target Base Rate \(Prevalence\)/i)).toBeNull();
    expect(screen.queryByText(/Class Imbalance Detected/i)).toBeNull();
    expect(screen.queryByText(/11\.27%/i)).toBeNull();
    expect(screen.queryByText(/Threshold Guarantee/i)).toBeNull();
    expect(screen.queryByText(/OOF Validation Selected → Touchless Holdout Eval/i)).toBeNull();

    // Verify regression actual_vs_pred artifact title is clean and concise
    expect(screen.getByText(/XGBoost — Actual vs Predicted/i)).toBeDefined();
    expect(screen.getByText(/XGBoost — Residual Diagnostics/i)).toBeDefined();
    expect(screen.getByText(/XGBoost — Top Predictive Drivers/i)).toBeDefined();
    unmountReg();

    // D. Forecasting report: threshold section NOT rendered, no classification terms
    render(
      <BrowserRouter>
        <ReportViewer report={forecastingReport} />
      </BrowserRouter>
    );
    expect(screen.queryByText(/3\. Classification Threshold Selection & Touchless Holdout Analysis/i)).toBeNull();
    expect(screen.queryByText(/Target Base Rate \(Prevalence\)/i)).toBeNull();
    expect(screen.queryByText(/11\.27%/i)).toBeNull();
  });

  it('dynamically renumbers sections sequentially with zero missing numbers when threshold section is omitted', () => {
    const regressionReport: Report = {
      ...mockReport,
      id: 'rep-reg-456',
      title: 'AutoDS Analysis Report — California_Housing',
      summary_markdown: 'Completed regression pipeline for goal "Predict median house value". Best Model: XGBoost.',
      full_report_markdown: `# AutoDS Autonomous Data Science Report
**Dataset:** \`California_Housing\`  
**Task Type:** regression  

The champion model selected is **XGBoost**.

## 3. Model Leaderboard & Multi-Metric Evaluation
| Model Name | Primary Loss Metric | CV Std | Train Time (s) | Model Family | Status |
|---|---|---|---|---|---|
| \`XGBoost\` | CV: 0.4612 | ±0.0120 | 2.10s | Gradient Boosting | Champion |

## 5. Methodological Critic Audit
All Methodological Critic Rules Passed Cleanly.

## 8. Model Limitations & Operational Risk Analysis
1. **Out-of-Bounds Sensitivity**: Sensitive to extreme outliers.
`,
      methodology_json: {
        problem_type: 'regression',
      },
      artifact_paths: [
        'reports/artifacts/rep-reg-456_XGBoost_actual_vs_pred.png',
      ],
    };

    render(
      <BrowserRouter>
        <ReportViewer report={regressionReport} />
      </BrowserRouter>
    );

    // Section 1: Executive Summary
    expect(screen.getByText(/1\. Executive Summary & Problem Formulation/i)).toBeDefined();
    // Section 2: Model Leaderboard
    expect(screen.getByText(/2\. Candidate Model Leaderboard & Multi-Metric Evaluation/i)).toBeDefined();
    // Section 3: Critic Audit (renumbered from 4 to 3!)
    expect(screen.getByText(/3\. Methodological Critic Audit & Leakage Safeguards/i)).toBeDefined();
    // Section 4: Visual Diagnostics (renumbered from 5 to 4!)
    expect(screen.getByText(/4\. Generated Visual Diagnostics/i)).toBeDefined();
    // Section 5: 4-Pillar Insights (renumbered from 6 to 5!)
    expect(screen.getByText(/5\. 4-Pillar Evidence-Backed Business Insights/i)).toBeDefined();
    // Section 6: Operational Risks (renumbered from 7 to 6!)
    expect(screen.getByText(/6\. Operational Risk Analysis & Deployment Boundaries/i)).toBeDefined();

    // Verify there are no duplicate or missing section numbers
    expect(screen.queryByText(/7\. Operational Risk Analysis/i)).toBeNull();
  });

  it('renders Markdown bold cleanly without literal asterisks in the UI and PDF views', () => {
    const reportWithBold: Report = {
      ...mockReport,
      id: 'rep-bold-789',
      title: 'AutoDS Analysis Report — **Bank_Marketing_UCI**',
      full_report_markdown: `# AutoDS Autonomous Data Science Report
**Dataset:** \`Bank_Marketing_UCI\`
**Objective:** **Maximize positive conversion** with high recall.

The champion model selected is **LightGBM**, with **ROC-AUC of 0.8121**.

## 8. Model Limitations & Operational Risk Analysis
1. **Correlation vs Causation**: Feature rankings are **statistical signals**, not causal guarantees.
`,
      business_insights_json: {
        insights: [
          {
            category: 'fact',
            title: '**Observed Campaign Volume**',
            finding: 'Total of **4,521** client records audited.',
            evidence: '**Dataset profile** metrics',
            confidence: 'HIGH',
          },
        ],
      },
    };

    const { container } = render(
      <BrowserRouter>
        <ReportViewer report={reportWithBold} />
      </BrowserRouter>
    );

    // Ensure raw ** asterisks do not appear literally in the rendered DOM text
    const allText = container.textContent || '';
    // Look for occurrences of "**" in rendered text
    expect(allText).not.toContain('**');
  });
});
