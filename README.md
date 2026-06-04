# Customer-Lifetime-Value-Predictor
Customer Lifetime Value prediction and Customer Segmentation using BG/NBD Gamma-Gamma ,XGBoost, and interactive Streamlit dashboard 
# Customer Lifetime Value Prediction

A complete end-to-end Customer Lifetime Value (CLV) prediction system built on real retail transaction data using statistical modeling and machine learning.

---

## Overview

Customer Lifetime Value (CLV) measures how much revenue a customer is expected to generate over a given time period. This project predicts the 12-month CLV for each customer using a combination of statistical models and XGBoost, deployed as an interactive web application.

---

## Dataset

**Online Retail II** — UCI Machine Learning Repository
- 1,067,371 transactions
- 8 features: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
- Date range: December 2009 – December 2011
- Source: UK-based online retailer

---

## Methodology

### 1. Data Preprocessing
- Removed cancelled orders (Invoice starting with 'C')
- Filtered negative quantities and prices
- Handled missing Customer IDs
- Engineered Revenue column (Quantity × Price)
- Computed RFM metrics per customer

### 2. RFM Feature Engineering
| Feature | Description |
|---|---|
| Recency | Days since last purchase |
| Frequency | Number of unique invoices |
| Monetary | Average transaction value |
| T | Days since first purchase |

### 3. Statistical Modeling — BG/NBD + Gamma-Gamma
- **BG/NBD Model**: Predicts purchase probability and expected future transactions
- **Gamma-Gamma Model**: Estimates average monetary value of future transactions
- Combined to compute 12-month CLV per customer

### 4. Machine Learning — XGBoost
- Target: Log-transformed CLV (reduces right skewness)
- Features: Frequency, Recency, T, Monetary
- Hyperparameter tuning via GridSearchCV
- Final R² Score: **0.94**

---

## Results

| Metric | Value |
|---|---|
| R² Score | 0.94 |
| MAE | 0.15 |
| RMSE | 0.27 |
| Customers Analyzed | 4,176 |
| Max CLV Found | £14,505 |

### Customer Segments
| Segment | CLV Range | Share |
|---|---|---|
| VIP | > £500 | 14.6% |
| High Value | £200 – £500 | 22.6% |
| Medium Value | £100 – £200 | 21.5% |
| Low Value | < £100 | 41.3% |

### Geographic Insights
- Sweden has highest average CLV (£800) despite fewer customers
- Channel Islands (£670) and Lithuania (£530) outperform UK on per-customer basis
- UK dominates total CLV due to volume but not individual customer value

---

## Project Structure

```
├── project 2 - customer life value.ipynb   # Main notebook
├── app.py                                   # Streamlit web app
├── clv_model.pkl                            # Trained XGBoost model
└── README.md
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| Lifetimes | BG/NBD + Gamma-Gamma models |
| XGBoost | CLV prediction |
| Scikit-learn | Model evaluation |
| Streamlit | Web deployment |

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn lifetimes xgboost scikit-learn streamlit
```

### 2. Clone the repository
```bash
git clone https://github.com/yourusername/customer-lifetime-value.git
cd customer-lifetime-value
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## Web App

The Streamlit app takes 4 customer inputs:
- **Frequency** — Total number of purchases
- **Recency** — Days since last purchase
- **T** — Days since first purchase
- **Monetary** — Average spend per order (£)

And returns:
- Predicted 12-month CLV in £
- Customer segment (VIP / High / Medium / Low)
- Tailored business recommendation

---

## Business Impact

| Action | Insight | Expected Outcome |
|---|---|---|
| VIP Retention Program | 14.6% customers generate majority of revenue | Protect £500–£14,505 per customer |
| International Expansion | Sweden/Channel Islands 3x higher CLV than UK average | Higher ROI per acquired customer |
| Re-engagement Campaign | 41.3% customers are low value/churned | Recover 10% = significant revenue uplift |
| Increase Order Frequency | Average customer buys only 3 times | Each additional purchase directly increases CLV |
| Upsell Strategy | Average spend is £27 per transaction | Increasing to £35 would improve CLV significantly |

---

## Model Interpretation

### Feature Importance
Recency is the most dominant feature in predicting CLV — customers who purchased recently are far more likely to buy again and generate higher future value.

| Feature | Importance | Business Meaning |
|---|---|---|
| Recency | Highest | Recent buyers = higher CLV |
| Monetary | High | Higher spenders = higher CLV |
| T | Medium | Longer customers = more stable CLV |
| Frequency | Medium | More purchases = higher CLV |

### SHAP Analysis
SHAP values confirm that low recency (recent purchase) has the strongest positive impact on CLV prediction. Customers with high recency (inactive for long) are consistently predicted as low value regardless of their historical spending.

### Why Log Transformation
CLV distribution is highly right-skewed — majority of customers cluster near zero with few extreme high-value outliers. Log transformation normalizes this distribution enabling XGBoost to learn patterns more effectively, resulting in R² improvement from 0.50 to 0.94.

--


## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-AA0000?style=for-the-badge)
![Lifetimes](https://img.shields.io/badge/Lifetimes-00599C?style=for-the-badge)
![BG/NBD](https://img.shields.io/badge/BG%2FNBD-0066CC?style=for-the-badge)
![Gamma-Gamma](https://img.shields.io/badge/Gamma--Gamma-8A2BE2?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)

---

## Key Findings

- Classic Pareto Principle observed — top 14.6% VIP customers drive majority of future revenue
- Recency is the most important feature for CLV prediction
- International customers (Sweden, Channel Islands) show significantly higher per-customer value than UK
- 41.3% of customers are predicted as low value — indicating high churn rate typical of retail

##Live Application
Live Demo - [https://customer-lifetime-value-predictor-9knb7cd6yuhdoswmsktsgt.streamlit.app/]
