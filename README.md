# Market Anomalies – Capstone Project (Advanced Programming)

## Research Question

This project investigates whether selected **calendar-based market anomalies** generate systematic excess returns and exhibit predictive power for the **direction of next-day excess stock returns** at three levels:

- Market level  
- Individual stock level  
- Sector level  

The analysis focuses on a panel of liquid U.S. equities and combines descriptive anomaly analysis with supervised machine learning models.

---

## Project Structure
```text
capstone/
├── README.md
├── PROPOSAL.md
├── environment.yml
├── requirements.txt
├── main.py                 
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── anomalies.py
│   ├── anomaly_analysis.py
│   ├── models.py
│   └── visualization.py
│
├── scripts/                 
│   ├── reduce_to_40_tickers.py
│   └── generate_sector_mapping_from_excel.py
│
├── data/
│   └── raw/
│       ├── market_anomalie_40.xlsx  (required)
│       └── sector_mapping.csv       (required)
│
├── results/                
│   ├── tables/
│   ├── models/
│   └── figures/
│
└── notebooks/
```

---

## Prerequisites

Before running the project, ensure you have:

- **Python 3.8+** installed
- **Conda** (recommended) or pip
- Required data files in `data/raw/`:
  - `market_anomalie_40.xlsx`
  - `sector_mapping.csv`

---

## Environment Setup

### Option 1 — Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate capstone-env
```

### Option 2 — pip
```bash
pip install -r requirements.txt
```

---

## Data Preparation (One-Time)

The following scripts were executed once to prepare the dataset and are **not part of the main pipeline**:
```bash
python scripts/generate_sector_mapping_from_excel.py
python scripts/reduce_to_40_tickers.py
```

They generate the following files in `data/raw/`:

- `sector_mapping.csv`
- `market_anomalie_40.xlsx`

**⚠️ Important**: These files are already included in the repository and **do not need to be re-generated** to run the project.

---

## Running the Project

From the root directory of the repository, run:
```bash
python main.py
```

### What the Pipeline Does

The main script executes the following steps:

1. **Data Loading and Cleaning**: Loads the daily panel from Excel and filters by tickers
2. **Feature Engineering**: Creates calendar-based features and technical indicators
3. **Calendar Anomaly Analysis**: Tests 5 market anomalies (A1-A5)
4. **Global ML Models**: Trains 4 models on the entire dataset
5. **Sector-Level ML Models**: Trains models separately for each of the 11 sectors
6. **Visualization**: Generates performance charts and anomaly plots

### Expected Runtime

- **Total execution time**: ~5-10 minutes (depending on hardware)
- Progress is printed to console for each step

### Output Structure

All results are automatically saved to the `results/` directory:
```text
results/
├── structured_dataset.csv
├── tables/
│   ├── anomaly_A1.xlsx (Day-of-Week effect)
│   ├── anomaly_A2.xlsx (January effect)
│   ├── anomaly_A3.xlsx (Turn-of-the-Month effect)
│   ├── anomaly_A4.xlsx (Sell-in-May effect)
│   └── anomaly_A5.xlsx (Pre-Holiday effect)
├── models/
│   ├── model_performance.csv
│   └── model_performance_[Sector].csv (one per sector)
└── figures/
    ├── day_of_week_global.png
    ├── monday_effect_by_sector.png
    ├── january_by_sector.png
    ├── turn_of_month_by_sector.png
    ├── sell_in_may_by_sector.png
    ├── pre_holiday_by_sector.png
    └── model_performance.png
```

---

## Models

The following supervised learning models are implemented:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **Neural Network (MLP)**

### Evaluation Metrics

Models are evaluated using:

- Accuracy
- ROC AUC
- Macro precision, recall, and F1-score

### Train/Test Split

- **Train period**: 2010–2018
- **Test period**: 2019–2025
- A strict time-based split is applied to avoid look-ahead bias

### Performance Summary

Global model accuracies range from **51.4% to 51.8%**, slightly above the baseline of 52.0% (majority class). Sector-level performance varies, with some sectors showing modest improvements.

---

## Dataset Information

### Data Source

- **40 liquid U.S. stocks** from 11 GICS sectors
- **Daily frequency**: 2010-2025
- **Total observations**: 149,076 rows

### Sector Distribution

| Sector                    | Observations |
|---------------------------|--------------|
| Information Technology    | 20,705       |
| Utilities                 | 20,705       |
| Financials                | 20,705       |
| Health Care               | 16,564       |
| Materials                 | 16,564       |
| Consumer Discretionary    | 16,564       |
| Industrials               | 12,423       |
| Real Estate               | 8,282        |
| Communication Services    | 8,282        |
| Consumer Staples          | 4,141        |
| Energy                    | 4,141        |

---

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure the environment is activated
```bash
   conda activate capstone-env
```

2. **FileNotFoundError**: Verify that `data/raw/market_anomalie_40.xlsx` exists

3. **Memory Error**: The pipeline requires ~4GB RAM for full execution

4. **Empty results folder**: The `results/` directory is created automatically on first run

---

## Notes

- `main.py` is the **single entry point** of the project
- Scripts in `scripts/` are utility scripts and are **not imported** by the pipeline
- The project is **fully reproducible** using the provided environment files
- All randomness is controlled via `random_state=42` for reproducibility

---

