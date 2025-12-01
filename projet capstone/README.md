# 📈 Market Anomalies – Capstone Project (Advanced Programming 2025)

## 🔎 Research Question

Do well-known calendar anomalies in equity markets generate **systematic excess returns** for:

- the overall market?
- individual stocks?
- specific sectors?

Studied anomalies:

- Pre-holiday effect  
- Turn-of-the-month effect  
- Sell-in-May effect  
- Christmas effect  
- Thanksgiving effect  
- New-Year effect  
- First-day-of-quarter effect  

---

## ⚙️ Setup

### Create environment

```bash
conda env create -f environment.yml
conda activate capstone-project

## Usage 

python main.py 

## Project Structure 

projet-capstone/
│
├── README.md                    # How to run the project
├── project_report.tex           # LaTeX academic report
├── project_report.pdf           # Compiled report
├── environment.yml              # Conda dependencies
│
├── main.py                      # Main entry point (full pipeline)
│
├── src/                         # Source code
│   ├── data_loader.py           # Load Excel + download indices + cleaning
│   ├── features.py              # Feature engineering (returns, anomalies, etc.)
│   ├── anomalies.py             # Day-of-week & January effect analysis
│   ├── anomaly_analysis.py      # Excess-return stats + sector analysis + plots
│   ├── models.py                # ML models (LogReg, RF, GB, MLP)
│   └── evaluation.py            # Evaluation helpers (metrics, reports)
│
├── data/
│   └── raw/
│       ├── market_anomalie.xlsx # DAILY & MONTHLY S&P500 data
│       └── sector_mapping.csv   # Ticker → sector mapping
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

#Results 
Baseline (always majority class)

- Accuracy ≈ 0.52

Logistic Regression

- Accuracy ≈ 0.515

AUC ≈ 0.514

- Random Forest

Accuracy ≈ 0.507

AUC ≈ 0.515

- Gradient Boosting

Accuracy ≈ 0.507

AUC ≈ 0.520

Better recall for positive (excess-return) days

Neural Network (MLPClassifier)

Accuracy ≈ 0.503

AUC ≈ 0.517

👉 Best model (for our goal): Gradient Boosting
It offers the best trade-off between AUC and recall for excess-return days, even if overall accuracy is close to the baseline.

## Requirements 
Main tools (all specified in environment.yml):

Python 3.10
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
imbalanced-learn
yfinance

#Academic Material 
project_report.tex – LaTeX source of the academic report

project_report.pdf – Final report to submit

All Python code in src/

Generated tables and figures in results/