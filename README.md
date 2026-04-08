# Hospital Financial Distress Prediction

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](#)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![Scikit-learn](https://img.shields.io/badge/-scikit--learn-%23F7931E?logo=scikit-learn&logoColor=white)](#)
![SHAP](https://img.shields.io/badge/SHAP-222F29)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD)

## Project Overview
Hospital closures have been a significant concern in many communities, especially after the COVID-19 pandemic. This project builds machine learning systems to identify hospitals at risk of financial distress using data from annual CMS Medicare Cost Reports.  
  
Two models are created, one only on operational metrics and a combined model that includes financial data. The models are tuned to recall to prioritize identification of hospitals at risk over overall accuracy.  
  
## Dataset
- **Source**: [Centers for Medicaid & Medicare Services Hospital Provider Cost Report](https://catalog.data.gov/dataset/hospital-provider-cost-report)
- **Period**: 2019-2023
- **Scope**: Acute Care Hospitals: Short Term General and Critical Access
- **Records**: 18,534 after cleaning (Raw dataset: 30400)
- **Target**: Net Income transformed into a binary classification (< 0)

## Machine Learning Pipeline
- Data validation and handling of missing data and extreme outliers
- Feature engineering: normalized ratios to facilitate facility comparison
- Preprocessing using sklearn pipeline
- Train-test-split for evaluation
- Hyperparameter tuning using GridSearchCV optimized for recall for both Random Forest Classifier and XGBoost
- Evaluation using classification metrics
- Explainability using SHAP for the best model performing model on each dataset

## Model Performance

#### Operational Features 
| Metric | Random Forest | XGBoost |
|---|---|---|
| ROC-AUC | 0.6780 | 0.6897 |
| Net Loss Recall | 59% | 59% |
| Macro F1 | 0.62 | 0.61 |

#### Operational & Financial Features
| Metric | Random Forest | XGBoost |
|---|---|---|
| ROC-AUC | 0.7434 | 0.7774 |
| Net Loss Recall | 70% | 70% |
| Macro F1 | 0.65 | 0.68 |

## Key Findings
- **% of Uncompensated Care** is the strongest operational predictor of financial distress
- **Bed Utilization** and **Average Length of Stay** also play a significant role in facility distress
- 2020 and 2021 were significant pandemic years and government initiatives protected hospital finances despite pandemic related challenges
- 2022-2023 saw the end of COVID related initiatives despite the fact that costs remained high due to continued inflation
- Adding financial metrics: assets to costs and liabilities to costs improved model performance significantly as expected

## Limitations & Opportunities for Future Work
- The features used for this project were a small fraction of the available options in the CMS Cost Reports. Selection was made based on domain knowledge and quality of the selected metrics. Several that could be strong predictors had high null rates, and handling those were beyond the scope of this project. 
- This project treated each year as a separate category instead of evaluating specific facility performance over time. A longitudinal approach could identify deterioration trends that improve detection. 

## How to Run
```bash
git clone https://github.com/Sarah-Eden/hospital-financial-distress.git
cd hospital-financial-distress
pip install -r requirements.txt
jupyter notebook hospital-financial-status-prediction.ipynb
```

Data files should be placed in a `data/` directory with the naming convention `CostReport_{year}_Final.csv` for years 2019-2023. Source data is available from the [US Data Catalog](https://catalog.data.gov/dataset/hospital-provider-cost-report).

