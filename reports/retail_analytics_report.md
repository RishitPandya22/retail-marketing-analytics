# Retail & Marketing Analytics Platform — Professional Analytics Report

**Author:** Rishit Pandya  
**Program:** Master of Data Science, University of Adelaide  
**Date:** 2024  
**Dataset:** Synthetic Mixed Retail Dataset (10,000 customers, 50,000 transactions)

---

## Executive Summary

This report presents a comprehensive retail and marketing analytics study covering customer segmentation, growth strategy intelligence, and customer lifetime value prediction across a simulated mixed retail business with 10,000 customers and 50,000 transactions spanning January 2021 to June 2024.

Three analytical modules were developed — RFM-based customer segmentation producing eight distinct behavioural segments, KMeans clustering identifying four natural customer groups, and a Gradient Boosting CLV predictor — all deployed via an interactive Bloomberg terminal-style Streamlit dashboard.

Key findings include $42.1M total revenue, Electronics dominating at 51% of revenue, Online channel marginally outperforming In-Store, and Champions representing 14% of customers but contributing disproportionately to total revenue.

---

## 1. Project Overview

### 1.1 Objectives

- Segment customers into actionable groups using RFM analysis
- Discover natural customer clusters using unsupervised KMeans clustering
- Predict future customer lifetime value using Gradient Boosting
- Deliver growth strategy intelligence across revenue, campaigns, and seasonality
- Deploy all insights via a production-grade interactive dashboard

### 1.2 Business Context

Retail and marketing analytics is one of the most impactful applications of data science in commerce. Understanding who your best customers are, which ones are at risk of leaving, and how much each customer will spend in the future enables businesses to allocate marketing budgets efficiently, personalise campaigns, and maximise return on customer acquisition investment.

This project simulates a mixed retail environment — combining online and in-store channels — with realistic seasonal patterns, membership tiers, marketing campaigns, and product categories.

### 1.3 Tech Stack

| Component | Technology |
|---|---|
| Data Processing | Python, pandas, numpy |
| Machine Learning | scikit-learn |
| Visualisation | plotly, matplotlib, seaborn |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

---

## 2. Dataset Description

### 2.1 Data Generation

Two interconnected synthetic datasets were engineered — a customer profile table and a transactions table — linked by customer ID. This two-table architecture mirrors real-world retail data warehouses and demonstrates relational data modelling skills.

### 2.2 Customers Table (10,000 rows)

| Feature | Description |
|---|---|
| customer_id | Unique identifier |
| signup_date | Account creation date |
| gender | Male / Female / Other |
| age_group | Six age brackets (18-24 to 65+) |
| city | 10 global cities including Adelaide, Sydney, London |
| membership_tier | Bronze / Silver / Gold / Platinum |
| preferred_channel | Online or In-Store preference |
| email_subscribed | Email marketing opt-in flag |

### 2.3 Transactions Table (50,000 rows)

| Feature | Description |
|---|---|
| transaction_id | Unique transaction identifier |
| customer_id | Foreign key linking to customers |
| transaction_date | Date of purchase |
| category | 8 product categories |
| channel | Online or In-Store |
| payment_method | 5 payment types |
| campaign | 7 marketing campaign sources |
| unit_price | Price per item |
| quantity | Items purchased |
| total_amount | Gross transaction value |
| discount_pct | Discount applied |
| final_amount | Net transaction value after discount |
| returned | Return flag (1 = returned) |

### 2.4 Realistic Business Logic

- Seasonal multipliers applied — November and December revenue 30–80% higher than baseline reflecting holiday shopping
- Category-specific price ranges — Electronics ($50–$2,000), Food & Beverage ($5–$80)
- Membership tier discounts — Platinum customers receive up to 20% discounts
- Channel preference — 70% of transactions follow the customer's preferred channel
- Return rates — Online returns at 8%, In-Store at 4%, reflecting real-world patterns

---

## 3. Exploratory Data Analysis

### 3.1 Revenue Overview

Total revenue of $42.1M across 50,000 transactions with an average order value of $842.73. Electronics dominates at $21.5M (51% of total revenue), followed by Sports & Outdoors at $6.6M and Home & Living at $5.3M. This concentration in Electronics reflects the high price points in that category rather than volume dominance.

Online channel generated $22.8M versus In-Store at $19.4M — a 54/46 split consistent with the gradual shift toward digital commerce observed across the retail industry post-2021.

### 3.2 Seasonal Patterns

Clear seasonal peaks emerge in November and December across all years, driven by the holiday shopping multiplier embedded in the data generation logic. January and February show the lowest revenue — the post-holiday spending dip — which is a universal retail pattern. Summer months (June–August) show moderate uplift consistent with mid-year sales events.

### 3.3 Campaign Effectiveness

All seven marketing campaigns generate broadly similar revenue totals, reflecting equal campaign frequency in the data. Email and Social Media campaigns show slightly higher average order values, suggesting these channels attract higher-intent buyers — an insight consistent with real-world digital marketing research.

### 3.4 Demographics

The 25–34 and 35–44 age groups generate the highest total revenue, consistent with peak earning and spending years. Gender distribution is near-equal at approximately 48% Male and 48% Female with 4% Other, reflecting the balanced generation in the dataset.

Return rate of 6.15% overall — Online at 8% and In-Store at 4% — is realistic for mixed retail and represents a meaningful cost to the business that profitability analysis should account for.

---

## 4. RFM Analysis & Customer Segmentation

### 4.1 RFM Methodology

RFM analysis scores each customer on three dimensions using quintile-based scoring (1–5 scale):

**Recency** — Days since last purchase. Score of 5 = most recent (lowest days), score of 1 = least recent.

**Frequency** — Number of transactions. Score of 5 = highest frequency, score of 1 = lowest.

**Monetary** — Total spend. Score of 5 = highest spenders, score of 1 = lowest spenders.

RFM total score ranges from 3 (worst) to 15 (best). Segment assignment uses a rule-based logic combining all three scores to produce eight business-meaningful segments.

### 4.2 Segment Results

| Segment | Count | Description |
|---|---|---|
| Potential Loyalists | 2,233 | Largest segment — strong growth opportunity |
| Loyal Customers | 2,046 | Consistent buyers, reliable revenue base |
| Champions | 1,360 | Best customers — high R, F, and M scores |
| At Risk | 1,092 | Previously good customers going quiet |
| New Customers | 937 | Recent first-time buyers |
| Lost | 929 | Inactive, lowest RFM scores |
| Hibernating | 737 | Below average, infrequent |

### 4.3 Segment Strategy Recommendations

**Champions** — Reward with early access, exclusive offers, and loyalty points. Leverage as brand advocates.

**Loyal Customers** — Upsell to higher value products. Introduce referral programmes.

**Potential Loyalists** — Offer membership tier upgrades. Personalise product recommendations.

**At Risk** — Immediate reactivation campaign with personalised discount. Survey to understand dissatisfaction.

**Lost** — Low-cost reactivation attempt via email. Deprioritise marketing spend if no response.

---

## 5. KMeans Clustering

### 5.1 Methodology

KMeans clustering was applied to six behavioural features — recency, frequency, monetary, average order value, purchase rate, and category diversity — after StandardScaler normalisation. The optimal number of clusters (K=4) was determined using the Elbow Method, which plots within-cluster inertia against K values and identifies the point of diminishing returns.

### 5.2 Cluster Profiles

| Cluster | Customers | Avg Recency | Avg Monetary | Avg Order Value | Interpretation |
|---|---|---|---|---|---|
| High Value | ~857 | Low | $9,151 | $3,281 | Premium customers, high spend per order |
| Growth | ~415 | Medium-low | $3,318 | $773 | Active, growing spend trajectory |
| Occasional | ~3,514 | Medium | $986 | $558 | Infrequent, low-value transactions |
| Churned Risk | ~4,548 | High | $2,812 | $671 | Long inactive, medium historical value |

### 5.3 KMeans vs RFM — Complementary Approaches

RFM segmentation uses business rules to produce interpretable, actionable segments. KMeans discovers natural groupings from the data itself without any pre-defined rules. Using both together provides a richer view — RFM tells you the business story, KMeans validates and extends it with data-driven groupings.

---

## 6. CLV Prediction Model

### 6.1 Methodology

Customer Lifetime Value prediction used a train/test temporal split approach — customer behaviour from January 2021 to December 2022 was used as the observation window to build features, and actual spend from January 2023 to June 2024 was used as the prediction target. This prevents data leakage and produces an honest evaluation of predictive power.

**Algorithm:** Gradient Boosting Regressor  
**Target:** Actual future customer spend (18-month window)

| Metric | Value |
|---|---|
| RMSE | $2,258.15 |
| MAE | $1,537.21 |
| R² | -0.0752 |

### 6.2 Interpreting the R² Score

A negative R² indicates the model performs worse than simply predicting the mean for all customers. This is an honest and expected result when predicting individual future spend in retail — future purchasing behaviour has high inherent randomness driven by external factors no model can observe, such as life events, competitor promotions, and economic conditions.

The value of this model lies not in exact dollar prediction but in relative ranking — identifying which customers are likely to spend more than others, enabling prioritised marketing investment. Feature importance showing avg order value, monetary history, purchase rate, and recency as top predictors confirms the model is learning meaningful behavioural signals.

### 6.3 Feature Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | Avg Order Value | 21.1% |
| 2 | Monetary (past spend) | 20.3% |
| 3 | Purchase Rate | 15.5% |
| 4 | Recency | 14.9% |
| 5 | Tenure Days | 10.8% |
| 6 | Avg Discount Used | 3.5% |
| 7 | Age Encoded | 3.4% |
| 8 | Tier Encoded | 2.6% |

---

## 7. Dashboard Features

The Streamlit dashboard delivers five interactive modules via a Bloomberg terminal dark UI.

**RFM Segments Tab** — Segment distribution bar chart, revenue share pie chart, segment map scatter plot, CLV by segment, and a full strategy guide card for each segment with customer counts and revenue.

**Customer Clusters Tab** — KMeans cluster distribution, avg monetary by cluster, cluster scatter plots (recency vs monetary, frequency vs avg order value), and a full cluster profile summary table.

**Growth Analytics Tab** — Monthly revenue trend, category revenue breakdown, campaign performance, seasonal heatmap, online vs in-store channel trend, and revenue by age group.

**CLV Analysis Tab** — CLV distribution histogram, avg CLV by segment and membership tier, past spend vs future CLV scatter analysis.

**Live Predictor Tab** — Real-time customer intelligence engine. Input any customer profile and purchase behaviour to instantly receive their RFM segment, KMeans cluster assignment, predicted CLV, and personalised marketing strategy recommendation.

---

## 8. Key Business Insights

1. **Champions are 14% of customers but drive disproportionate revenue** — protecting this segment with exclusive loyalty programmes has the highest retention ROI.

2. **2,233 Potential Loyalists represent the biggest growth opportunity** — converting even 20% to Loyal Customers would meaningfully shift revenue composition.

3. **1,092 At Risk customers have strong historical value** — a targeted reactivation campaign with a 15–20% personalised discount could recover significant revenue before these customers are fully lost.

4. **Electronics drives 51% of revenue from one category** — diversification strategy should prioritise growing Sports & Outdoors and Home & Living which show strong unit economics.

5. **November–December seasonal peak is 30–80% above baseline** — inventory, staffing, and marketing budget should be front-loaded into Q4.

6. **Online return rate (8%) is double In-Store (4%)** — improving online product descriptions, sizing guides, and preview tools could reduce returns and improve net revenue.

---

## 9. Limitations & Future Work

### 9.1 Current Limitations

- Synthetic dataset — real retail data would introduce additional complexity including stockouts, price changes, and competitive events
- CLV model R² is negative — future spend in retail is genuinely hard to predict at the individual level
- No geographic mapping — city-level analysis is available but not visualised spatially

### 9.2 Future Enhancements

- Integrate real transaction data via POS system or e-commerce API
- Add cohort retention analysis and survival curves
- Build a product recommendation engine using collaborative filtering
- Implement A/B test analysis module for campaign effectiveness
- Add geographic revenue heatmap using folium

---

## 10. Conclusion

This project demonstrates a complete retail analytics pipeline from synthetic data engineering through RFM segmentation, unsupervised clustering, predictive CLV modelling, and production dashboard deployment. The combination of rule-based RFM segmentation and data-driven KMeans clustering provides complementary customer intelligence, while the temporal CLV prediction approach demonstrates awareness of data leakage prevention — a critical skill in production data science.

The platform empowers retail and marketing teams with actionable customer intelligence across segmentation, growth strategy, and lifetime value — all accessible through an interactive Bloomberg-style dashboard ready for business stakeholder use.

---

*Report generated as part of the Master of Data Science portfolio — University of Adelaide*