# AutoDS Diagnostic Visualization Artifacts

This directory stores generated high-resolution diagnostic charts synthesized during Stage 8 of the AutoDS pipeline.

## Diagnostic Plots
- **Classification**:
  - Receiver Operating Characteristic (ROC Curve with AUC / Multiclass OvR)
  - Precision-Recall Curve (PR-AUC with baseline prevalence reference)
  - Confusion Matrix (Evaluated on untouched holdout test set)
  - Top Predictive Drivers (Relative feature importance with non-causal labeling)
- **Regression / Forecasting**:
  - Actual vs Predicted / Forecast Trajectory
  - Residual Diagnostics & Error Distribution
  - Top Predictive Drivers
