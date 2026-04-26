import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

os.makedirs('models', exist_ok=True)
os.makedirs('assets', exist_ok=True)

BG = '#0A0E1A'
CARD = '#111827'
TEXT = '#E2E8F0'
COLORS = ['#00D4FF', '#FF6B6B', '#00FF88', '#FFD93D', '#C77DFF', '#FF9F43']

df = pd.read_csv('data/rfm_data.csv')

print("=" * 60)
print("STAGE 4 - MODEL BUILDING")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════
# MODEL 1 — KMEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════
print("\n[MODEL 1] KMeans Customer Clustering")
print("-" * 60)

cluster_features = ['recency', 'frequency', 'monetary',
                    'avg_order_value', 'purchase_rate', 'category_diversity']

cluster_df = df[cluster_features].fillna(0)

scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_df)

# ── Find optimal K using Elbow Method ─────────────────────────────
print("  Finding optimal number of clusters...")
inertias = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled)
    inertias.append(km.inertia_)

# Plot elbow curve
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
ax.set_facecolor(CARD)
ax.plot(list(k_range), inertias, color=COLORS[0], linewidth=2.5,
        marker='o', markersize=8)
ax.axvline(x=4, color=COLORS[1], linestyle='--', linewidth=1.5, label='Optimal K=4')
ax.set_title('KMeans Elbow Curve — Optimal Clusters', color=TEXT, fontsize=13, fontweight='bold')
ax.set_xlabel('Number of Clusters (K)', color=TEXT)
ax.set_ylabel('Inertia', color=TEXT)
ax.tick_params(colors=TEXT)
ax.legend(facecolor=CARD, labelcolor=TEXT)
for spine in ax.spines.values():
    spine.set_edgecolor('#1E293B')
plt.tight_layout()
plt.savefig('assets/chart10_elbow_curve.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Elbow curve saved!")

# ── Train final KMeans with K=4 ────────────────────────────────────
print("  Training KMeans with K=4...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled)

# ── Cluster Profiling ──────────────────────────────────────────────
cluster_profile = df.groupby('cluster')[cluster_features].mean().round(2)
print("\n  Cluster Profiles:")
print(cluster_profile.to_string())

# ── Name clusters based on profile ────────────────────────────────
cluster_monetary = df.groupby('cluster')['monetary'].mean()
cluster_recency = df.groupby('cluster')['recency'].mean()
cluster_frequency = df.groupby('cluster')['frequency'].mean()

cluster_labels = {}
sorted_by_value = cluster_monetary.sort_values(ascending=False)
for rank, (cluster_id, _) in enumerate(sorted_by_value.items()):
    rec = cluster_recency[cluster_id]
    freq = cluster_frequency[cluster_id]
    mon = cluster_monetary[cluster_id]
    if rank == 0:
        cluster_labels[cluster_id] = 'High Value'
    elif rank == 1 and rec < cluster_recency.mean():
        cluster_labels[cluster_id] = 'Growth'
    elif rec > cluster_recency.mean() * 1.2:
        cluster_labels[cluster_id] = 'Churned Risk'
    else:
        cluster_labels[cluster_id] = 'Occasional'

df['cluster_label'] = df['cluster'].map(cluster_labels)

print("\n  Cluster Label Distribution:")
print(df['cluster_label'].value_counts())

# ── Cluster Scatter Plot ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
cluster_colors = [COLORS[i] for i in range(4)]
for cid in range(4):
    mask = df['cluster'] == cid
    label = cluster_labels.get(cid, f'Cluster {cid}')
    axes[0].scatter(df[mask]['recency'], df[mask]['monetary'],
                    color=cluster_colors[cid], label=label, alpha=0.5, s=15)
axes[0].set_facecolor(CARD)
axes[0].set_title('Clusters: Recency vs Monetary', color=TEXT, fontsize=12, fontweight='bold')
axes[0].set_xlabel('Recency (days)', color=TEXT)
axes[0].set_ylabel('Monetary ($)', color=TEXT)
axes[0].tick_params(colors=TEXT)
axes[0].legend(facecolor=CARD, labelcolor=TEXT, fontsize=9)
for spine in axes[0].spines.values():
    spine.set_edgecolor('#1E293B')

for cid in range(4):
    mask = df['cluster'] == cid
    label = cluster_labels.get(cid, f'Cluster {cid}')
    axes[1].scatter(df[mask]['frequency'], df[mask]['monetary'],
                    color=cluster_colors[cid], label=label, alpha=0.5, s=15)
axes[1].set_facecolor(CARD)
axes[1].set_title('Clusters: Frequency vs Monetary', color=TEXT, fontsize=12, fontweight='bold')
axes[1].set_xlabel('Frequency (orders)', color=TEXT)
axes[1].set_ylabel('Monetary ($)', color=TEXT)
axes[1].tick_params(colors=TEXT)
axes[1].legend(facecolor=CARD, labelcolor=TEXT, fontsize=9)
for spine in axes[1].spines.values():
    spine.set_edgecolor('#1E293B')
fig.patch.set_facecolor(BG)
plt.tight_layout()
plt.savefig('assets/chart11_cluster_scatter.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Cluster scatter plots saved!")

# ── Save KMeans model and scaler ───────────────────────────────────
joblib.dump(kmeans, 'models/kmeans_model.pkl')
joblib.dump(scaler, 'models/kmeans_scaler.pkl')
joblib.dump(cluster_labels, 'models/cluster_labels.pkl')
joblib.dump(cluster_features, 'models/cluster_features.pkl')
print("  KMeans model saved!")

# ══════════════════════════════════════════════════════════════════
# MODEL 2 — CLV PREDICTOR (Gradient Boosting Regressor)
# ══════════════════════════════════════════════════════════════════
print("\n[MODEL 2] CLV Predictor — Gradient Boosting Regressor")
print("-" * 60)

clv_features = [
    'recency', 'frequency', 'monetary', 'avg_order_value',
    'purchase_rate', 'category_diversity', 'tenure_days',
    'avg_discount_used', 'tier_encoded', 'age_encoded',
    'channel_encoded', 'email_subscribed'
]

gender_cols = [c for c in df.columns if c.startswith('gender_')]
clv_features = clv_features + gender_cols

X = df[clv_features].fillna(0)
y = df['clv']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

clv_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.85,
    random_state=42
)

clv_model.fit(X_train, y_train)
y_pred = clv_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"  RMSE: ${rmse:,.2f}")
print(f"  MAE:  ${mae:,.2f}")
print(f"  R²:   {r2:.4f}")

# ── Feature Importance ─────────────────────────────────────────────
print("\n  Top 8 CLV Feature Importances:")
importance_df = pd.DataFrame({
    'feature': clv_features,
    'importance': clv_model.feature_importances_
}).sort_values('importance', ascending=False).head(8)
for _, row in importance_df.iterrows():
    bar = '█' * int(row['importance'] * 300)
    print(f"  {row['feature']:<25} {bar} {row['importance']:.4f}")

# ── Feature Importance Chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
ax.set_facecolor(CARD)
top_features = importance_df.head(8)
bars = ax.barh(top_features['feature'], top_features['importance'],
               color=COLORS[0], edgecolor='none', height=0.5)
for bar, val in zip(bars, top_features['importance']):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', color=TEXT, fontsize=9)
ax.set_title('CLV Model — Top Feature Importances', color=TEXT, fontsize=13, fontweight='bold')
ax.set_xlabel('Importance', color=TEXT)
ax.tick_params(colors=TEXT)
for spine in ax.spines.values():
    spine.set_edgecolor('#1E293B')
plt.tight_layout()
plt.savefig('assets/chart12_feature_importance.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  Feature importance chart saved!")

joblib.dump(clv_model, 'models/clv_model.pkl')
joblib.dump(clv_features, 'models/clv_features.pkl')
print("  CLV model saved!")

# ══════════════════════════════════════════════════════════════════
# SAVE ENRICHED DATASET & METRICS
# ══════════════════════════════════════════════════════════════════
df.to_csv('data/rfm_clustered.csv', index=False)

metrics = {
    'clv_rmse': round(rmse, 2),
    'clv_mae': round(mae, 2),
    'clv_r2': round(r2, 4),
    'n_clusters': 4,
    'cluster_labels': str(cluster_labels)
}
pd.DataFrame([metrics]).to_csv('models/model_metrics.csv', index=False)

print("\n" + "=" * 60)
print("ALL MODELS TRAINED & SAVED!")
print("=" * 60)
print(f"\nFinal dataset with clusters saved to data/rfm_clustered.csv")