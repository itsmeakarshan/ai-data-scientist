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

    // 2. Verify all 4 diagnostic plot titles
    expect(screen.getByText(/LightGBM — ROC Curve \(Receiver Operating Characteristic\)/i)).toBeDefined();
    expect(screen.getByText(/LightGBM — Precision-Recall Curve \(PR-AUC\)/i)).toBeDefined();
    expect(screen.getByText(/LightGBM — Confusion Matrix \(Locked Operating Threshold\)/i)).toBeDefined();
    expect(screen.getByText(/LightGBM — Top Predictive Drivers \(Feature Importance\)/i)).toBeDefined();

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
});
