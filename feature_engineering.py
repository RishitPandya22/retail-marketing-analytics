import pandas as pd
import numpy as np

customers = pd.read_csv('data/customers.csv')
transactions = pd.read_csv('data/transactions.csv')
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
customers['signup_date'] = pd.to_datetime(customers['signup_date'])

print("=" * 60)
print("STAGE 3 - FEATURE ENGINEERING & RFM ANALYSIS")
print("=" * 60)

# ── Step 1: Remove returned transactions ──────────────────────────
print(f"\nTransactions before removing returns: {len(transactions):,}")
transactions_clean = transactions[transactions['returned'] == 0].copy()
print(f"Transactions after removing returns:  {len(transactions_clean):,}")

# ── Step 2: Split into observation & prediction window ────────────
# Observation window: Jan 2021 - Dec 2022 (we learn from this)
# Prediction window:  Jan 2023 - Jun 2024 (we predict this)
obs_end = pd.Timestamp('2023-01-01')
pred_end = pd.Timestamp('2024-07-01')

obs_txn = transactions_clean[transactions_clean['transaction_date'] < obs_end].copy()
pred_txn = transactions_clean[transactions_clean['transaction_date'] >= obs_end].copy()

print(f"\nObservation window transactions: {len(obs_txn):,}")
print(f"Prediction window transactions:  {len(pred_txn):,}")

# ── Step 3: Build RFM from observation window ──────────────────────
print("\nBuilding RFM table...")
snapshot_date = obs_end

rfm = obs_txn.groupby('customer_id').agg(
    recency=('transaction_date', lambda x: (snapshot_date - x.max()).days),
    frequency=('transaction_id', 'count'),
    monetary=('final_amount', 'sum')
).reset_index()

# ── Step 4: RFM Scoring (1-5) ──────────────────────────────────────
print("Calculating RFM scores...")
rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm['m_score'] = pd.qcut(rfm['monetary'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm['rfm_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
rfm['rfm_segment_code'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)

# ── Step 5: Customer Segments ──────────────────────────────────────
def assign_segment(row):
    r = row['r_score']
    f = row['f_score']
    m = row['m_score']
    score = row['rfm_score']
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'New Customers'
    elif r >= 3 and f >= 2 and m >= 2:
        return 'Potential Loyalists'
    elif r <= 2 and f >= 3 and m >= 3:
        return 'At Risk'
    elif r <= 2 and f >= 4 and m >= 4:
        return 'Cant Lose Them'
    elif r == 1 and f <= 2 and m <= 2:
        return 'Lost'
    elif score >= 9:
        return 'Loyal Customers'
    elif score >= 6:
        return 'Potential Loyalists'
    else:
        return 'Hibernating'

rfm['segment'] = rfm.apply(assign_segment, axis=1)

print("\nRFM Segment Distribution:")
print(rfm['segment'].value_counts())

# ── Step 6: CLV = actual future spend (prediction window) ─────────
print("\nCalculating actual future CLV from prediction window...")
future_spend = pred_txn.groupby('customer_id')['final_amount'].sum().reset_index()
future_spend.columns = ['customer_id', 'clv']
rfm = rfm.merge(future_spend, on='customer_id', how='left')
rfm['clv'] = rfm['clv'].fillna(0)
print(f"  Customers with future purchases: {(rfm['clv'] > 0).sum():,}")
print(f"  Avg future CLV: ${rfm['clv'].mean():,.2f}")

# ── Step 7: Additional Features ────────────────────────────────────
print("\nEngineering additional features...")

rfm['avg_order_value'] = (rfm['monetary'] / rfm['frequency']).round(2)

customer_tenure = customers[['customer_id', 'signup_date']].copy()
customer_tenure['tenure_days'] = (snapshot_date - customer_tenure['signup_date']).dt.days
rfm = rfm.merge(customer_tenure[['customer_id', 'tenure_days']], on='customer_id', how='left')
rfm['tenure_days'] = rfm['tenure_days'].clip(lower=30)

rfm['purchase_rate'] = (rfm['frequency'] / (rfm['tenure_days'] / 30)).round(4)

cat_diversity = obs_txn.groupby('customer_id')['category'].nunique().reset_index()
cat_diversity.columns = ['customer_id', 'category_diversity']
rfm = rfm.merge(cat_diversity, on='customer_id', how='left')

channel_pref = obs_txn.groupby(['customer_id', 'channel']).size().reset_index(name='count')
channel_pref = channel_pref.sort_values('count', ascending=False).drop_duplicates('customer_id')
channel_pref = channel_pref[['customer_id', 'channel']].rename(columns={'channel': 'preferred_channel_txn'})
rfm = rfm.merge(channel_pref, on='customer_id', how='left')

discount_usage = obs_txn.groupby('customer_id')['discount_pct'].mean().reset_index()
discount_usage.columns = ['customer_id', 'avg_discount_used']
rfm = rfm.merge(discount_usage, on='customer_id', how='left')

rfm = rfm.merge(customers[['customer_id', 'membership_tier', 'age_group',
                             'city', 'gender', 'email_subscribed']], on='customer_id', how='left')

# ── Step 8: Encode Categoricals ───────────────────────────────────
tier_map = {'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4}
rfm['tier_encoded'] = rfm['membership_tier'].map(tier_map)

age_map = {'18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55-64': 5, '65+': 6}
rfm['age_encoded'] = rfm['age_group'].map(age_map)

channel_map = {'Online': 1, 'In-Store': 0}
rfm['channel_encoded'] = rfm['preferred_channel_txn'].map(channel_map).fillna(0)

gender_dummies = pd.get_dummies(rfm['gender'], prefix='gender')
rfm = pd.concat([rfm, gender_dummies], axis=1)

# ── Step 9: Save ──────────────────────────────────────────────────
rfm.to_csv('data/rfm_data.csv', index=False)
print(f"\nFinal RFM dataset shape: {rfm.shape}")
print(f"\nSample RFM data:")
print(rfm[['customer_id', 'recency', 'frequency', 'monetary',
           'rfm_score', 'segment', 'clv']].head(8).to_string())
print("\nKey RFM Stats:")
print(f"  Avg Recency:   {rfm['recency'].mean():.0f} days")
print(f"  Avg Frequency: {rfm['frequency'].mean():.1f} orders")
print(f"  Avg Monetary:  ${rfm['monetary'].mean():,.2f}")
print(f"  Avg CLV:       ${rfm['clv'].mean():,.2f}")
print(f"  Champions:     {(rfm['segment'] == 'Champions').sum():,} customers")
print(f"  At Risk:       {(rfm['segment'] == 'At Risk').sum():,} customers")
print(f"  Lost:          {(rfm['segment'] == 'Lost').sum():,} customers")
print("\nStage 3 complete!")