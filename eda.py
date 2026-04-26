import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('assets', exist_ok=True)

plt.style.use('dark_background')
BG = '#0A0E1A'
CARD = '#111827'
TEXT = '#E2E8F0'
COLORS = ['#00D4FF', '#FF6B6B', '#00FF88', '#FFD93D', '#C77DFF', '#FF9F43', '#48DBFB', '#FF6B9D']

def style_ax(ax, title):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1E293B')

customers = pd.read_csv('data/customers.csv')
transactions = pd.read_csv('data/transactions.csv')
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
transactions['month'] = transactions['transaction_date'].dt.to_period('M')
transactions['month_num'] = transactions['transaction_date'].dt.month
transactions['year'] = transactions['transaction_date'].dt.year

merged = transactions.merge(customers, on='customer_id', how='left')

print("Running EDA... generating all charts")

# ── Chart 1: Monthly Revenue Trend ────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
monthly_rev = transactions.groupby('month')['final_amount'].sum().reset_index()
monthly_rev['month_str'] = monthly_rev['month'].astype(str)
ax.plot(monthly_rev['month_str'], monthly_rev['final_amount'],
        color=COLORS[0], linewidth=2.5, marker='o', markersize=4)
ax.fill_between(monthly_rev['month_str'], monthly_rev['final_amount'],
                alpha=0.15, color=COLORS[0])
style_ax(ax, 'Monthly Revenue Trend (2021-2024)')
ax.set_xlabel('Month')
ax.set_ylabel('Revenue ($)')
tick_step = max(1, len(monthly_rev) // 10)
ax.set_xticks(range(0, len(monthly_rev), tick_step))
ax.set_xticklabels(monthly_rev['month_str'].iloc[::tick_step], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('assets/chart1_monthly_revenue.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Chart 1 done - Monthly Revenue Trend")

# ── Chart 2: Revenue by Category ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
cat_rev = transactions.groupby('category')['final_amount'].sum().sort_values(ascending=True)
bars = ax.barh(cat_rev.index, cat_rev.values, color=COLORS[:len(cat_rev)], edgecolor='none', height=0.5)
for bar, val in zip(bars, cat_rev.values):
    ax.text(val + 50000, bar.get_y() + bar.get_height()/2,
            f'${val/1e6:.1f}M', va='center', color=TEXT, fontsize=10)
style_ax(ax, 'Total Revenue by Category')
ax.set_xlabel('Revenue ($)')
plt.tight_layout()
plt.savefig('assets/chart2_revenue_by_category.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Chart 2 done - Revenue by Category")

# ── Chart 3: Revenue by Channel ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
channel_rev = transactions.groupby('channel')['final_amount'].sum()
axes[0].pie(channel_rev.values, labels=channel_rev.index, colors=[COLORS[0], COLORS[2]],
            autopct='%1.1f%%', textprops={'color': TEXT}, startangle=90)
axes[0].set_facecolor(CARD)
axes[0].set_title('Revenue Split: Online vs In-Store', color=TEXT, fontsize=13, fontweight='bold')

channel_monthly = transactions.groupby(['month', 'channel'])['final_amount'].sum().reset_index()
channel_monthly['month_str'] = channel_monthly['month'].astype(str)
for i, ch in enumerate(channel_monthly['channel'].unique()):
    data = channel_monthly[channel_monthly['channel'] == ch]
    axes[1].plot(data['month_str'], data['final_amount'],
                 color=COLORS[i], linewidth=2, label=ch, marker='o', markersize=3)
style_ax(axes[1], 'Monthly Revenue by Channel')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Revenue ($)')
axes[1].legend(facecolor=CARD, labelcolor=TEXT)
tick_step = max(1, len(data) // 8)
axes[1].set_xticks(range(0, len(data), tick_step))
axes[1].set_xticklabels(data['month_str'].iloc[::tick_step], rotation=45, ha='right')
fig.patch.set_facecolor(BG)
plt.tight_layout()
plt.savefig('assets/chart3_channel_analysis.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Chart 3 done - Channel Analysis")

# ── Chart 4: Seasonal Revenue Heatmap ─────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
seasonal = transactions.groupby(['year', 'month_num'])['final_amount'].sum().reset_index()
seasonal_pivot = seasonal.pivot(index='year', columns='month_num', values='final_amount')
seasonal_pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
sns.heatmap(seasonal_pivot, ax=ax, cmap='YlOrRd', annot=True, fmt='.0f',
            linewidths=0.5, linecolor='#1E293B', annot_kws={'size': 8},
            cbar_kws={'shrink': 0.8})
style_ax(ax, 'Seasonal Revenue Heatmap (Year x Month)')
plt.tight_layout()
plt.savefig('assets/chart4_seasonal_heatmap.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Chart 4 done - Seasonal Heatmap")

# ── Chart 5: Membership Tier Analysis ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
tier_order = ['Bronze', 'Silver', 'Gold', 'Platinum']
tier_rev = merged.groupby('membership_tier')['final_amount'].sum().reindex(tier_order)
bars = axes[0].bar(tier_rev.index, tier_rev.values,
                   color=[COLORS[5], COLORS[3], COLORS[2], COLORS[0]],
                   edgecolor='none', width=0.5)
for bar, val in zip(bars, tier_rev.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20000,
                 f'${val/1e6:.1f}M', ha='center', color=TEXT, fontsize=10, fontweight='bold')
style_ax(axes[0], 'Total Revenue by Membership Tier')
axes[0].set_ylabel('Revenue ($)')

tier_aov = merged.groupby('membership_tier')['final_amount'].mean().reindex(tier_order)
bars2 = axes[1].bar(tier_aov.index, tier_aov.values,
                    color=[COLORS[5], COLORS[3], COLORS[2], COLORS[0]],
                    edgecolor='none', width=0.5)
for bar, val in zip(bars2, tier_aov.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'${val:.0f}', ha='center', color=TEXT, fontsize=10, fontweight='bold')
style_ax(axes[1], 'Avg Order Value by Membership Tier')
axes[1].set_ylabel('Avg Order Value ($)')
fig.patch.set_facecolor(BG)
plt.tight_layout()
plt.savefig('assets/chart5_membership_analysis.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Chart 5 done - Membership Analysis")

# ── Chart 6: Campaign Effectiveness ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
camp_rev = transactions.groupby('campaign')['final_amount'].sum().sort_values(ascending=True)
bars = axes[0].barh(camp_rev.index, camp_rev.values, color=COLORS[:len(camp_rev)],
                    edgecolor='none', height=0.5)
for bar, val in zip(bars, camp_rev.values):
    axes[0].text(val + 10000, bar.get_y() + bar.get_height()/2,
                 f'${val/1e6:.1f}M', va='center', color=TEXT, fontsize=9)
style_ax(axes[0], 'Revenue by Marketing Campaign')
axes[0].set_xlabel('Revenue ($)')

camp_aov = transactions.groupby('campaign')['final_amount'].mean().sort_values(ascending=True)
bars2 = axes[1].barh(camp_aov.index, camp_aov.values, color=COLORS[:len(camp_aov)],
                     edgecolor='none', height=0.5)
for bar, val in zip(bars2, camp_aov.values):
    axes[1].text(val + 5, bar.get_y() + bar.get_height()/2,
                 f'${val:.0f}', va='center', color=TEXT, fontsize=9)
style_ax(axes[1], 'Avg Order Value by Campaign')
axes[1].set_xlabel('Avg Order Value ($)')
fig.patch.set_facecolor(BG)
plt.tight_layout()
plt.savefig('assets/chart6_campaign_effectiveness.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Chart 6 done - Campaign Effectiveness")

# ── Chart 7: Age Group & Gender Analysis ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
age_rev = merged.groupby('age_group')['final_amount'].sum()
age_order = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
age_rev = age_rev.reindex(age_order)
bars = axes[0].bar(age_rev.index, age_rev.values, color=COLORS[:6], edgecolor='none', width=0.5)
for bar, val in zip(bars, age_rev.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20000,
                 f'${val/1e6:.1f}M', ha='center', color=TEXT, fontsize=9, fontweight='bold')
style_ax(axes[0], 'Revenue by Age Group')
axes[0].set_ylabel('Revenue ($)')

gender_rev = merged.groupby('gender')['final_amount'].sum()
axes[1].pie(gender_rev.values, labels=gender_rev.index,
            colors=[COLORS[0], COLORS[1], COLORS[2]],
            autopct='%1.1f%%', textprops={'color': TEXT}, startangle=90)
axes[1].set_title('Revenue Split by Gender', color=TEXT, fontsize=13, fontweight='bold')
fig.patch.set_facecolor(BG)
plt.tight_layout()
plt.savefig('assets/chart7_demographics.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Chart 7 done - Demographics")

# ── Chart 8: Return Rate Analysis ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
return_cat = transactions.groupby('category')['returned'].mean().sort_values(ascending=True) * 100
bars = axes[0].barh(return_cat.index, return_cat.values, color=COLORS[1],
                    edgecolor='none', height=0.5)
for bar, val in zip(bars, return_cat.values):
    axes[0].text(val + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', color=TEXT, fontsize=9)
style_ax(axes[0], 'Return Rate by Category (%)')
axes[0].set_xlabel('Return Rate (%)')

return_channel = transactions.groupby('channel')['returned'].mean() * 100
bars2 = axes[1].bar(return_channel.index, return_channel.values,
                    color=[COLORS[0], COLORS[2]], edgecolor='none', width=0.4)
for bar, val in zip(bars2, return_channel.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.1f}%', ha='center', color=TEXT, fontsize=11, fontweight='bold')
style_ax(axes[1], 'Return Rate by Channel (%)')
axes[1].set_ylabel('Return Rate (%)')
fig.patch.set_facecolor(BG)
plt.tight_layout()
plt.savefig('assets/chart8_return_analysis.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Chart 8 done - Return Rate Analysis")

# ── Chart 9: Payment Method Analysis ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
pay_rev = transactions.groupby('payment_method')['final_amount'].sum().sort_values(ascending=False)
bars = ax.bar(pay_rev.index, pay_rev.values, color=COLORS[:len(pay_rev)],
              edgecolor='none', width=0.5)
for bar, val in zip(bars, pay_rev.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20000,
            f'${val/1e6:.1f}M', ha='center', color=TEXT, fontsize=10, fontweight='bold')
style_ax(ax, 'Revenue by Payment Method')
ax.set_ylabel('Revenue ($)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('assets/chart9_payment_methods.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Chart 9 done - Payment Methods")

print(f"\nAll 9 charts saved to assets/ folder!")
print(f"\nKey EDA Insights:")
print(f"  Top Category:      {cat_rev.idxmax()} (${cat_rev.max():,.0f})")
print(f"  Top Campaign:      {camp_rev.idxmax()}")
print(f"  Best AOV Channel:  {transactions.groupby('channel')['final_amount'].mean().idxmax()}")
print(f"  Peak Month:        {transactions.groupby('month_num')['final_amount'].sum().idxmax()}")
print(f"  Highest Return Cat: {return_cat.idxmax()} ({return_cat.max():.1f}%)")