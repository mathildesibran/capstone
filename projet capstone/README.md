# Market Anomalies – Capstone Project (Advanced Programming 2025)

## Research Question

This project investigates whether well-known calendar-based market anomalies generate **systematic excess returns** for:

- the overall equity market,
- individual stocks,
- specific economic sectors.

The following anomalies are studied:

- Pre-holiday effect  
- Turn-of-the-month effect  
- Sell-in-May effect  
- Christmas effect  
- Thanksgiving effect  
- New Year effect  
- First day of quarter effect  

---

## Setup

### Create the Conda environment

```bash
conda env create -f environment.yml
conda activate capstone-project

## Usage 

To run the full data pipeline and model estimation:

bash
Copier le code
python main.py
This script executes the complete workflow:
data loading, feature engineering, anomaly analysis, sector analysis, and machine learning model training.

Project Structure
text
Copier le code
projet-capstone/
│
├── README.md                    # Project description and usage instructions
├── project_report.tex           # LaTeX academic report
├── project_report.pdf           # Compiled academic report
├── environment.yml              # Conda dependencies
│
├── main.py                      # Main entry point (full pipeline)
│
├── src/                         # Source code
│   ├── data_loader.py           # Load Excel data, download indices, data cleaning
│   ├── features.py              # Feature engineering (returns, anomalies, etc.)
│   ├── anomalies.py             # Day-of-week and January effect analysis
│   ├── anomaly_analysis.py      # Excess-return statistics, sector analysis, plots
│   ├── models.py                # Machine learning models (LogReg, RF, GB, MLP)
│   └── evaluation.py            # Evaluation helpers (metrics and reports)
│
├── data/
│   └── raw/
│       ├── market_anomalie.xlsx # DAILY and MONTHLY S&P 500 data
│       └── sector_mapping.csv   # Ticker-to-sector mapping
│
└── results/
    └── anomalies/
        ├── anomaly_A1_by_ticker.xlsx
        ├── anomaly_A1_global.xlsx
        ├── anomaly_A2_by_ticker.xlsx
        ├── anomaly_A2_global.xlsx
        ├── anomaly_global_returns.xlsx
        ├── anomaly_ticker_returns.xlsx
        ├── anomaly_sector_returns.xlsx
        ├── global_anomaly_diff.png
        └── sector_anomaly_heatmap.png
Results Summary
Baseline (majority class prediction)
Accuracy ≈ 0.52

Logistic Regression
Accuracy ≈ 0.515
AUC ≈ 0.514

Random Forest
Accuracy ≈ 0.507
AUC ≈ 0.515

Gradient Boosting
Accuracy ≈ 0.507
AUC ≈ 0.520
Best recall for excess-return days

Neural Network (MLPClassifier)
Accuracy ≈ 0.503
AUC ≈ 0.517

Selected model for interpretation:
Gradient Boosting provides the best trade-off between AUC and recall for excess-return days, despite overall accuracy remaining close to the baseline.

Requirements
All dependencies are specified in environment.yml:

Python 3.10

pandas

numpy

scipy

scikit-learn

matplotlib

seaborn

imbalanced-learn

yfinance

