# Project Proposal  
**Predicting Stock Performance During Market Anomalies Using Machine Learning**

## Objective  
The goal of this project is to analyze how individual stocks behave during well-known market anomalies (Monday effect, January effect, and Holiday effect) and to develop machine learning models capable of predicting which stocks are most likely to outperform the market during these periods.

## Data  
The dataset is sourced from Refinitiv and includes approximately 40–50 stocks from a major equity index such as the S&P 500 or STOXX Europe 600, covering the period from 2010 to 2024.

From daily prices, the following features are constructed:
- Daily and weekly returns  
- Rolling volatility  
- Moving averages  
- Market beta  
- Correlation with the market  
- Calendar effects (Monday, January, Holidays)  
- Sector classification  

## Target Variable  
The target variable is binary:
- 1 if the stock outperforms the market  
- 0 otherwise  

## Machine Learning Models  
The following models are evaluated and compared:
- Logistic Regression  
- Random Forest  
- Gradient Boosting (XGBoost-type model)  
- Neural Network (MLP)

## Validation Strategy  
A temporal split is applied to prevent look-ahead bias:
- Training set: 2010–2018  
- Test set: 2019–2024  

This ensures that the models are evaluated strictly on future data.

## Class Imbalance  
If outperforming stocks are underrepresented, class imbalance is addressed using:
- Class weighting  
- SMOTE (Synthetic Minority Oversampling Technique)

## Evaluation Metrics  
Model performance is evaluated using:
- Accuracy  
- Recall  
- F1-score  
- AUC (Area Under the ROC Curve)  

The final objective is to identify the key factors associated with stock outperformance during anomaly periods.
