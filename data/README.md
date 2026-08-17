# AutoDS Data Directory

This directory stores datasets used by AutoDS. Raw data is strictly preserved and never mutated directly.

## Directory Structure

- `raw/`: Immutable raw datasets downloaded from original sources.
- `interim/`: Intermediate cached transformations during agent exploration.
- `processed/`: Standardized, validated, and cleaned dataset versions.
- `external/`: External benchmarks, reference schemas, or lookup tables.

## Primary Datasets

### 1. Bank Marketing Dataset (Primary Classification Dataset)
- **Source**: UCI Machine Learning Repository
- **URL**: `https://archive.ics.uci.edu/dataset/222/bank+marketing`
- **Citation**: S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, In Press, 2014.
- **License**: CC BY 4.0
- **Target**: `y` (binary: "yes", "no" — term deposit subscription)
- **Location**: `data/raw/bank_marketing/`
- **Expected Files**: `bank-additional-full.csv` (41,188 rows, 21 columns) or `bank-full.csv` (45,211 rows, 17 columns)

### 2. M5 Forecasting Dataset (Forecasting Dataset)
- **Source**: Kaggle / Makridakis Open Forecasting Center (MOFC) M5 Competition
- **Official Repository**: `https://github.com/Mcompetitions/M5-methods`
- **License**: Competition dataset terms / open academic research
- **Location**: `data/raw/m5/`
- **Expected Files**: Calendar data, daily item sales series, price history (or standardized subset for local testing).

### 3. OpenML-CC18 Benchmark Suite (Generalization Benchmark)
- **Source**: OpenML Curated Classification Benchmark Suite 18
- **URL**: `https://www.openml.org/s/99`
- **Usage**: Evaluating agent generalization on diverse tabular classification tasks without dataset-specific bias.

## Download Instructions

You can download and verify datasets using the built-in CLI scripts:

```bash
# Download and verify Bank Marketing dataset
python scripts/download_data.py --dataset bank_marketing

# Download and verify M5 forecasting sample
python scripts/download_data.py --dataset m5_sample
```
