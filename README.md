# HR Attrition Analysis — IBM Dataset

End-to-end HR analytics project. SQL pattern analysis + Python EDA + Logistic Regression model predicting employee attrition + Power BI dashboard with actionable business recommendations.

![Python](https://img.shields.io/badge/python-3.11-blue)
![SQL](https://img.shields.io/badge/SQL-SQLite-lightgrey)
![PowerBI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow)
![ScikitLearn](https://img.shields.io/badge/scikit--learn-ML-orange)

---

## Business Question
**"Which employees are most likely to leave, and what can HR do about it?"**

---

## Key Findings

| Finding | Insight |
|---|---|
| Overall attrition rate | 16.1% |
| Overtime employees | 3x more likely to leave |
| Under $3K/month salary band | 26% attrition rate |
| 0-1 year tenure | Highest risk group |
| Low job satisfaction (score 1) | 22.8% attrition vs 11.3% (score 4) |
| Sales department | Highest attrition at 20.6% |

---

## Project Structure

```
hr-attrition-analysis/
│
├── scripts/
│   └── attrition_analysis.py    # EDA + ML model + insights
│
├── sql/
│   └── attrition_queries.sql    # 6 business SQL queries
│
├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv  # raw IBM dataset
│   └── processed/               # EDA plots, feature importance, risk segments
│
├── dashboard/                   # Power BI .pbix file
└── README.md
```

---

## Quick Start

```bash
# 1. Download IBM HR dataset from Kaggle
# https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

# 2. Place in data/ folder
mv ~/Downloads/WA_Fn-UseC_-HR-Employee-Attrition.csv data/

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 4. Run full analysis
python scripts/attrition_analysis.py
```

Outputs saved to `data/processed/`:
- `eda_plots.png` — 6-panel EDA visualization
- `roc_curve.png` — model performance curve
- `feature_importance.csv` — top attrition predictors
- `high_risk_segments.csv` — departments needing HR attention

---

## ML Model Results

- **Algorithm:** Logistic Regression
- **ROC AUC:** 0.82
- **Top predictors:** OverTime, MonthlyIncome, YearsAtCompany, JobSatisfaction, Age

---

## Business Recommendations

1. **Cap overtime** — employees working overtime leave at 3x the rate. Implement overtime limits or compensation adjustments.
2. **Salary review for under $3K band** — 26% attrition rate, highest of any salary band.
3. **Focus on first 2 years** — new employees are highest risk. Structured onboarding and mentoring programs reduce early attrition.
4. **Sales department intervention** — 20.6% attrition rate needs targeted retention programs.

---

## Author
**Mallesh Putha** — Data Engineer
[LinkedIn](https://linkedin.com/in/malleshputha) · [GitHub](https://github.com/Malleshputha)
