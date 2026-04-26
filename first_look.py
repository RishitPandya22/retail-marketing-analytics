import pandas as pd
import numpy as np

customers = pd.read_csv('data/customers.csv')
transactions = pd.read_csv('data/transactions.csv')

print("=" * 60)
print("RETAIL DATASET — FIRST LOOK")
print("=" * 60)

print(f"\nCUSTOMERS TABLE")
print(f"  Shape:   {customers.shape}")
print(f"  Columns: {list(customers.columns)}")
print(f"\n  Missing Values:")
print(customers.isnull().sum())
print(f"\n  First 3 rows:")
print(customers.head(3))

print(f"\nTRANSACTIONS TABLE")
print(f"  Shape:   {transactions.shape}")
print(f"  Columns: {list(transactions.columns)}")
print(f"\n  Missing Values:")
print(transactions.isnull().sum())
print(f"\n  First 3 rows:")
print(transactions.head(3))

print(f"\nKEY METRICS")
print(f"  Total Revenue:     ${transactions['final_amount'].sum():,.2f}")
print(f"  Total Transactions: {len(transactions):,}")
print(f"  Unique Customers:   {transactions['customer_id'].nunique():,}")
print(f"  Avg Order Value:    ${transactions['final_amount'].mean():,.2f}")
print(f"  Return Rate:        {transactions['returned'].mean():.2%}")

print(f"\nREVENUE BY CATEGORY:")
print(transactions.groupby('category')['final_amount'].sum().sort_values(ascending=False).round(2))

print(f"\nREVENUE BY CHANNEL:")
print(transactions.groupby('channel')['final_amount'].sum().round(2))

print(f"\nMEMBERSHIP DISTRIBUTION:")
print(customers['membership_tier'].value_counts())

print(f"\nTOP 5 CITIES:")
print(customers['city'].value_counts().head(5))