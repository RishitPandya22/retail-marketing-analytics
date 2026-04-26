import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N_CUSTOMERS = 10000
N_TRANSACTIONS = 50000

# ── Reference Data ─────────────────────────────────────────────────
genders = ['Male', 'Female', 'Other']
age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
cities = ['Sydney', 'Melbourne', 'Brisbane', 'Adelaide', 'Perth', 'Auckland', 'London', 'Toronto', 'New York', 'Singapore']
channels = ['Online', 'In-Store']
categories = ['Electronics', 'Clothing', 'Food & Beverage', 'Home & Living', 'Beauty & Health', 'Sports & Outdoors', 'Toys & Games', 'Books & Media']
payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Cash', 'Buy Now Pay Later']
campaigns = ['Email', 'Social Media', 'TV Ad', 'Influencer', 'No Campaign', 'SMS', 'Google Ads']
membership_tiers = ['Bronze', 'Silver', 'Gold', 'Platinum']

category_price_range = {
    'Electronics': (50, 2000),
    'Clothing': (20, 300),
    'Food & Beverage': (5, 80),
    'Home & Living': (15, 500),
    'Beauty & Health': (10, 200),
    'Sports & Outdoors': (25, 600),
    'Toys & Games': (10, 150),
    'Books & Media': (5, 60)
}

start_date = datetime(2021, 1, 1)
end_date = datetime(2024, 6, 1)

def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

# ── Generate Customers ─────────────────────────────────────────────
print("Generating customers...")
customers = []
for i in range(N_CUSTOMERS):
    signup = random_date(start_date, datetime(2023, 1, 1))
    tier = np.random.choice(membership_tiers, p=[0.45, 0.30, 0.17, 0.08])
    age_group = random.choice(age_groups)
    city = random.choice(cities)
    gender = np.random.choice(genders, p=[0.48, 0.48, 0.04])
    preferred_channel = np.random.choice(channels, p=[0.55, 0.45])

    customers.append({
        'customer_id': f'CUST-{20000 + i}',
        'signup_date': signup.strftime('%Y-%m-%d'),
        'gender': gender,
        'age_group': age_group,
        'city': city,
        'membership_tier': tier,
        'preferred_channel': preferred_channel,
        'email_subscribed': np.random.choice([1, 0], p=[0.72, 0.28])
    })

customers_df = pd.DataFrame(customers)

# ── Generate Transactions ──────────────────────────────────────────
print("Generating transactions...")
transactions = []
for i in range(N_TRANSACTIONS):
    customer = customers_df.sample(1).iloc[0]
    tx_date = random_date(start_date, end_date)

    # Seasonal multiplier
    month = tx_date.month
    if month in [11, 12]:
        seasonal_mult = np.random.uniform(1.3, 1.8)
    elif month in [6, 7, 8]:
        seasonal_mult = np.random.uniform(1.1, 1.4)
    elif month in [1, 2]:
        seasonal_mult = np.random.uniform(0.7, 1.0)
    else:
        seasonal_mult = np.random.uniform(0.9, 1.2)

    category = random.choice(categories)
    price_min, price_max = category_price_range[category]
    unit_price = round(np.random.uniform(price_min, price_max) * seasonal_mult, 2)
    quantity = np.random.randint(1, 6)
    total_amount = round(unit_price * quantity, 2)

    discount_pct = 0
    if customer['membership_tier'] == 'Platinum':
        discount_pct = np.random.choice([0, 10, 15, 20], p=[0.3, 0.3, 0.2, 0.2])
    elif customer['membership_tier'] == 'Gold':
        discount_pct = np.random.choice([0, 5, 10], p=[0.4, 0.35, 0.25])
    elif customer['membership_tier'] == 'Silver':
        discount_pct = np.random.choice([0, 5], p=[0.6, 0.4])

    discount_amount = round(total_amount * discount_pct / 100, 2)
    final_amount = round(total_amount - discount_amount, 2)

    channel = customer['preferred_channel'] if random.random() < 0.70 else random.choice(channels)
    campaign = random.choice(campaigns)

    # Return rate
    returned = 1 if random.random() < (0.08 if channel == 'Online' else 0.04) else 0

    transactions.append({
        'transaction_id': f'TXN-{100000 + i}',
        'customer_id': customer['customer_id'],
        'transaction_date': tx_date.strftime('%Y-%m-%d'),
        'category': category,
        'channel': channel,
        'payment_method': payment_methods[random.randint(0, len(payment_methods)-1)],
        'campaign': campaign,
        'unit_price': unit_price,
        'quantity': quantity,
        'total_amount': total_amount,
        'discount_pct': discount_pct,
        'discount_amount': discount_amount,
        'final_amount': final_amount,
        'returned': returned
    })

transactions_df = pd.DataFrame(transactions)

# ── Save Both Datasets ─────────────────────────────────────────────
customers_df.to_csv('data/customers.csv', index=False)
transactions_df.to_csv('data/transactions.csv', index=False)

print("\nDataset generated successfully!")
print(f"Customers:    {len(customers_df):,}")
print(f"Transactions: {len(transactions_df):,}")
print(f"Date Range:   {transactions_df['transaction_date'].min()} to {transactions_df['transaction_date'].max()}")
print(f"Total Revenue: ${transactions_df['final_amount'].sum():,.2f}")
print(f"Avg Order Value: ${transactions_df['final_amount'].mean():,.2f}")
print(f"Return Rate: {transactions_df['returned'].mean():.2%}")
print(f"\nMembership Distribution:")
print(customers_df['membership_tier'].value_counts())