# Market Anomalies – Capstone Project (Advanced Programming)

## Research Question

This project investigates whether selected **calendar-based market anomalies**
generate systematic excess returns and exhibit predictive power for the **direction
of next-day excess stock returns** at three levels:

- market level,
- individual stock level,
- sector level.

The analysis focuses on a panel of liquid U.S. equities and combines
descriptive anomaly analysis with supervised machine learning models.

---

## Project Structure

your-project/
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


## Environment Setup

### Option 1 — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate capstone-env

Option 2 — Pip
pip install -r requirements.txt


Data Preparation (One-Time)
The following scripts are executed once to prepare the data and are not part
of the main pipeline:

python scripts/generate_sector_mapping_from_excel.py
python scripts/reduce_to_40_tickers.py

They generate:
sector_mapping.csv
market_anomalie_40.xlsx

Running the Project
To run the full analysis pipeline, execute:
python main.py


This script performs:
1. Data loading and cleaning
2. Feature engineering
3. Calendar anomaly analysis
4. Machine learning model estimation
5. Result export and visualization
All outputs are saved in the results/ directory.


Models
The following supervised models are implemented:
Logistic Regression
Random Forest
XGBoost
Neural Network (MLP)

Models are evaluated using:
Accuracy
ROC AUC
Macro precision, recall, and F1-score
A strict time-based split is applied to avoid look-ahead bias.


Notes
The main.py file is the single entry point of the project.
Scripts in scripts/ are utility scripts and are not imported by the pipeline.
The project is fully reproducible using the provided environment files.