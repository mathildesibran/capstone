# Market Anomalies – Capstone Project (Advanced Programming)

## Research Question

This project investigates whether selected **calendar-based market anomalies**
generate systematic excess returns and exhibit predictive power for the
**direction of next-day excess stock returns** at three levels:

- Market level  
- Individual stock level  
- Sector level  

The analysis focuses on a panel of liquid U.S. equities and combines
descriptive anomaly analysis with supervised machine learning models.

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
│
├── results/
└── notebooks/
Environment Setup
Option — Conda
bash
Copier le code
conda env create -f environment.yml
conda activate capstone-env
Data Preparation (One-Time)
The following scripts were executed once to prepare the dataset and are not
part of the main pipeline:

bash
Copier le code
python scripts/generate_sector_mapping_from_excel.py
python scripts/reduce_to_40_tickers.py
They generate the following files in data/raw/:

sector_mapping.csv

market_anomalie_40.xlsx

These files are already included in the repository and do not need to be
re-generated to run the project.

Running the Project
From the root directory of the repository, run:

bash
Copier le code
python main.py
This script performs:

Data loading and cleaning

Feature engineering

Calendar anomaly analysis

Machine learning model estimation (global and sector-level)

Result export and visualization

All outputs are saved in the results/ directory.

Models
The following supervised learning models are implemented:

Logistic Regression

Random Forest

Gradient Boosting

XGBoost

Neural Network (MLP)

Models are evaluated using:

Accuracy

ROC AUC

Macro precision, recall, and F1-score

A strict time-based train/test split is applied to avoid look-ahead bias.

Notes
main.py is the single entry point of the project.

Scripts in scripts/ are utility scripts and are not imported by the pipeline.

The project is fully reproducible using the provided environment files.
