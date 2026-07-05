# Credit Risk Modelling — Multiple Linear Regression

## Project Overview
Predicting the **credit amount** for loan applicants using **Multiple Linear Regression** on the German Credit Dataset. The project covers the full data science pipeline from exploratory data analysis through model evaluation and deployment as a Streamlit web application.

---

## Repository Structure
```
credit-risk-modelling/
   ├── data/
   │     └── german_credit_data.csv
   ├── notebook/
   │     └── Credit_Risk_Modelling_Documented.ipynb
   └── app/
         └── credit_risk_app_v2.py
```

---

## Dataset
| Property | Detail |
|----------|--------|
| Source | German Credit Dataset |
| Original size | 1,000 rows × 11 columns |
| Final size | 522 rows after preprocessing |
| Target variable | Credit Amount (log-transformed) |

---

## Pipeline
- ✅ Exploratory Data Analysis (EDA)
- ✅ Data Preprocessing (missing values, encoding)
- ✅ Log Transformation of target variable
- ✅ Feature Selection (f_regression + VIF)
- ✅ Train Test Split (80/20)
- ✅ Multiple Linear Regression
- ✅ Model Evaluation (R², MSE, RMSE)
- ✅ Residual Analysis
- ✅ Prediction and Validation

---

## Results
| Metric | Train | Test |
|--------|-------|------|
| R² | 0.5059 | 0.5628 |
| MSE | 0.3087 | 0.3205 |
| RMSE | 0.5556 | 0.5662 |
| Overfitting | None detected ✅ | — |

---

## Key Findings
- **Duration** is the strongest predictor of credit amount (F-statistic = 391.28)
- **Job level** is the second most important feature — higher skill levels request larger loans
- **Log transformation** reduced skewness from 2.06 to 0.10 — dramatically improving normality
- **All regression assumptions satisfied** — normality (Shapiro p = 0.39), homoscedasticity confirmed
- Model predicts within **1% of actual average** for similar customer profiles
- **No multicollinearity** — all VIF values below 6

---

## Feature Selection
### f_regression (p < 0.05) — 11 features selected
| Feature | F-Statistic | P-Value |
|---------|-------------|---------|
| Duration | 391.28 | 0.0000 |
| Job | 70.92 | 0.0000 |
| Purpose_radio/TV | 21.08 | 0.0000 |
| Purpose_vacation/others | 15.22 | 0.0001 |
| Housing_own | 12.59 | 0.0004 |
| Risk_good | 10.25 | 0.0015 |
| Checking account_rich | 7.57 | 0.0061 |
| Sex_male | 6.05 | 0.0142 |
| Purpose_domestic appliances | 5.76 | 0.0168 |
| Checking account_moderate | 4.65 | 0.0315 |
| Saving accounts_rich | 4.04 | 0.0450 |

### VIF — All features below threshold
| Feature | VIF | Status |
|---------|-----|--------|
| Job | 5.75 | ✅ Acceptable |
| Duration | 3.88 | ✅ Acceptable |
| All others | < 3.2 | ✅ Very low |

---

## Streamlit App
Run the app locally:

```bash
pip install streamlit
streamlit run app/credit_risk_app_v2.py
```

The app allows users to input a customer profile and receive an instant credit amount prediction.

---

## Libraries Used
```
pandas        — data manipulation
numpy         — numerical operations
matplotlib    — data visualization
seaborn       — statistical visualization
scikit-learn  — feature selection, modelling, evaluation
statsmodels   — VIF calculation
scipy         — Shapiro-Wilk normality test
streamlit     — web application
```

---
## Live App
🚀 [Click here to use the app](https://credit-risk-modelling-5o8zntcthfhdcncgaahffr.streamlit.app/)

## Author
**Prosper Amedu**
Chemical Engineering Student | Data Science Enthusiast

[GitHub](https://github.com/Prosper438) | [Portfolio](https://prosper438.github.io/prosper78.github.io/)

