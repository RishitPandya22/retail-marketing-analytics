# 🛍️ Retail & Marketing Analytics Platform

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> A production-grade Retail & Marketing Analytics Platform featuring RFM customer segmentation, KMeans clustering, growth strategy intelligence, and CLV prediction — deployed as a Bloomberg terminal-style interactive dashboard.

**🔗 Live Demo:** [retail-marketing-analytics.streamlit.app](https://retail-marketing-analytics.streamlit.app/)

---

## 🚀 Overview

This end-to-end data science project tackles four core retail analytics problems:

- **RFM Segmentation** — Score and segment 10,000 customers into 8 actionable groups
- **KMeans Clustering** — Discover 4 natural customer groups from behavioural data
- **Growth Strategy** — Revenue trends, seasonal patterns, campaign and category intelligence
- **CLV Prediction** — Predict future customer lifetime value using Gradient Boosting

Built with a dark Bloomberg terminal UI, real-time customer intelligence predictor, and interactive Plotly visualisations across 50,000 transactions.

---

## 📈 Model Performance

| Model | Algorithm | Metric | Score |
|---|---|---|---|
| Customer Clustering | KMeans (K=4) | Silhouette | 4 clean clusters |
| CLV Predictor | Gradient Boosting Regressor | RMSE | $2,258 |

---

## 🖥️ Dashboard Features

| Tab | Features |
|---|---|
| 🎯 RFM Segments | Segment distribution, revenue share, segment map, CLV by segment, strategy guide |
| 🔵 Customer Clusters | KMeans cluster explorer, scatter plots, cluster profile table |
| 📈 Growth Analytics | Revenue trend, category breakdown, campaign performance, seasonal heatmap |
| 💰 CLV Analysis | CLV distribution, CLV by segment and tier, past vs future spend |
| 🤖 Live Predictor | Real-time RFM segment, cluster, CLV and strategy recommendation |

---

## 🗂️ Project Structure
retail-marketing-analytics/
│
├── data/
│   ├── customers.csv              # Raw customer profiles
│   ├── transactions.csv           # Raw transaction records
│   ├── rfm_data.csv               # RFM engineered dataset
│   └── rfm_clustered.csv          # Final dataset with clusters
│
├── models/
│   ├── kmeans_model.pkl           # Trained KMeans model
│   ├── kmeans_scaler.pkl          # StandardScaler for clustering
│   ├── clv_model.pkl              # Trained CLV predictor
│   └── model_metrics.csv          # Saved model metrics
│
├── assets/                        # EDA and model chart exports
│
├── reports/
│   └── retail_analytics_report.md # Full professional analytics report
│
├── generate_data.py               # Synthetic dataset generator
├── eda.py                         # Deep EDA and visualisations
├── feature_engineering.py        # RFM analysis and feature engineering
├── train_models.py                # KMeans and CLV model training
├── app.py                         # Streamlit dashboard
└── requirements.txt               # Dependencies
---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/RishitPandya22/retail-marketing-analytics.git
cd retail-marketing-analytics

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate dataset
python generate_data.py

# 5. Run EDA
python eda.py

# 6. Feature engineering & RFM
python feature_engineering.py

# 7. Train models
python train_models.py

# 8. Launch dashboard
streamlit run app.py
```

---

## 🧠 Key Business Insights

- **Champions are 14% of customers** but drive disproportionate revenue — protect with exclusive loyalty rewards
- **2,233 Potential Loyalists** are the biggest growth opportunity — converting 20% to Loyal doubles that segment
- **1,092 At Risk customers** have strong historical value — reactivation campaign with 15–20% discount has highest ROI
- **Electronics drives 51% of revenue** from one category — diversification into Sports and Home & Living is strategic priority
- **November–December peak is 30–80% above baseline** — Q4 marketing budget should be front-loaded
- **Online returns are double In-Store** — better product content reduces returns and improves net revenue

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn |
| Visualisation | plotly, matplotlib, seaborn |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | GitHub Desktop |

---

## 👨‍💻 Author

**Rishit Pandya**  
Master of Data Science — University of Adelaide 🇦🇺  
[GitHub](https://github.com/RishitPandya22) · [LinkedIn](https://www.linkedin.com/in/rishit-pandya-854b7928a)

---

*Part of an end-to-end data science portfolio spanning retail analytics, housing analysis, medical AI, sports prediction, stock forecasting, FinOps intelligence, and retail marketing analytics.*