import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

st.set_page_config(
    page_title="Retail & Marketing Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { background-color: #0A0E1A; color: #E2E8F0; }
.main { background-color: #0A0E1A; }
section[data-testid="stSidebar"] { background-color: #0D1117; border-right: 1px solid #1E293B; }
.kpi-card { background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 20px 24px; text-align: center; }
.kpi-label { font-size: 11px; color: #64748B; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 8px; }
.kpi-value { font-size: 26px; font-weight: 700; color: #00D4FF; }
.kpi-delta { font-size: 11px; color: #00FF88; margin-top: 4px; }
.section-header { background: #111827; border-left: 3px solid #00D4FF; border-radius: 0 8px 8px 0; padding: 10px 16px; margin: 24px 0 16px 0; }
.section-title { font-size: 13px; font-weight: 600; color: #00D4FF; text-transform: uppercase; letter-spacing: 1.5px; margin: 0; }
.segment-card { background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 16px 20px; margin-bottom: 10px; }
.segment-name { font-size: 15px; font-weight: 600; color: #E2E8F0; margin-bottom: 4px; }
.segment-desc { font-size: 12px; color: #64748B; margin-bottom: 8px; }
.segment-stat { font-size: 13px; color: #00D4FF; }
.model-badge { background: #0F2027; border: 1px solid #00D4FF; border-radius: 20px; padding: 4px 14px; font-size: 12px; color: #00D4FF; display: inline-block; margin: 4px; }
.stTabs [data-baseweb="tab-list"] { background-color: #111827; border-bottom: 1px solid #1E293B; gap: 4px; }
.stTabs [data-baseweb="tab"] { background-color: transparent; color: #64748B; border-radius: 8px 8px 0 0; padding: 10px 20px; font-size: 13px; font-weight: 500; }
.stTabs [aria-selected="true"] { background-color: #0A0E1A; color: #00D4FF; border-bottom: 2px solid #00D4FF; }
div[data-testid="stMetric"] { background: #111827; border: 1px solid #1E293B; border-radius: 12px; padding: 16px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor='#0A0E1A',
    plot_bgcolor='#111827',
    font=dict(color='#E2E8F0'),
    xaxis=dict(gridcolor='#1E293B', linecolor='#1E293B'),
    yaxis=dict(gridcolor='#1E293B', linecolor='#1E293B'),
    colorway=['#00D4FF', '#FF6B6B', '#00FF88', '#FFD93D', '#C77DFF', '#FF9F43', '#48DBFB', '#FF6B9D']
)

SEGMENT_COLORS = {
    'Champions': '#00D4FF',
    'Loyal Customers': '#00FF88',
    'Potential Loyalists': '#FFD93D',
    'New Customers': '#C77DFF',
    'At Risk': '#FF9F43',
    'Cant Lose Them': '#FF6B6B',
    'Hibernating': '#64748B',
    'Lost': '#374151'
}

SEGMENT_DESCRIPTIONS = {
    'Champions': 'Bought recently, buy often, spend the most',
    'Loyal Customers': 'Buy regularly, respond well to promotions',
    'Potential Loyalists': 'Recent customers with good frequency',
    'New Customers': 'Bought most recently but not often',
    'At Risk': 'Above average RFM but havent bought recently',
    'Cant Lose Them': 'Made biggest purchases but long ago',
    'Hibernating': 'Below average RFM, low recency',
    'Lost': 'Lowest RFM scores, inactive longest'
}

CLUSTER_COLORS = {
    'High Value': '#00D4FF',
    'Growth': '#00FF88',
    'Occasional': '#FFD93D',
    'Churned Risk': '#FF6B6B'
}

# ── Load Data & Models ─────────────────────────────────────────────
@st.cache_data
def load_data():
    rfm = pd.read_csv('data/rfm_clustered.csv')
    customers = pd.read_csv('data/customers.csv')
    transactions = pd.read_csv('data/transactions.csv')
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
    transactions['month'] = transactions['transaction_date'].dt.to_period('M').astype(str)
    transactions['month_num'] = transactions['transaction_date'].dt.month
    transactions['year'] = transactions['transaction_date'].dt.year
    merged = transactions.merge(customers, on='customer_id', how='left')
    return rfm, customers, transactions, merged

@st.cache_resource
def load_models():
    models = {}
    try:
        models['kmeans'] = joblib.load('models/kmeans_model.pkl')
        models['scaler'] = joblib.load('models/kmeans_scaler.pkl')
        models['cluster_labels'] = joblib.load('models/cluster_labels.pkl')
        models['cluster_features'] = joblib.load('models/cluster_features.pkl')
        models['clv'] = joblib.load('models/clv_model.pkl')
        models['clv_features'] = joblib.load('models/clv_features.pkl')
        if os.path.exists('models/model_metrics.csv'):
            models['metrics'] = pd.read_csv('models/model_metrics.csv').iloc[0]
    except Exception as e:
        st.error(f"Model loading error: {e}")
    return models

rfm, customers, transactions, merged = load_data()
models = load_models()
transactions_clean = transactions[transactions['returned'] == 0]

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='text-align:center;padding:16px 0 8px'><span style='font-size:28px'>🛍️</span><br><span style='font-size:16px;font-weight:700;color:#00D4FF'>Retail Analytics</span><br><span style='font-size:11px;color:#64748B'>MARKETING PLATFORM v1.0</span></div>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1E293B;margin:12px 0'>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Filters</p>", unsafe_allow_html=True)
    segment_filter = st.multiselect("RFM Segment", options=sorted(rfm['segment'].unique().tolist()), default=sorted(rfm['segment'].unique().tolist()))
    tier_filter = st.multiselect("Membership Tier", options=['Bronze', 'Silver', 'Gold', 'Platinum'], default=['Bronze', 'Silver', 'Gold', 'Platinum'])
    channel_filter = st.multiselect("Channel", options=['Online', 'In-Store'], default=['Online', 'In-Store'])

    st.markdown("<hr style='border-color:#1E293B;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>Model Info</p>", unsafe_allow_html=True)
    st.markdown("<div class='model-badge'>KMeans K=4</div>", unsafe_allow_html=True)
    st.markdown("<div class='model-badge'>GB CLV Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='model-badge'>RFM Segmentation</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E293B;margin:12px 0'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:10px;color:#334155;text-align:center'>Built by Rishit Pandya<br>MDS · University of Adelaide</p>", unsafe_allow_html=True)

# ── Apply Filters ──────────────────────────────────────────────────
filtered_rfm = rfm[rfm['segment'].isin(segment_filter) & rfm['membership_tier'].isin(tier_filter)]
filtered_txn = transactions_clean[transactions_clean['channel'].isin(channel_filter)]
filtered_merged = merged[merged['channel'].isin(channel_filter) & merged['membership_tier'].isin(tier_filter)]

# ── Header ─────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size:26px;font-weight:700;color:#E2E8F0;margin-bottom:4px'>Retail & Marketing Analytics Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:13px;color:#64748B;margin-bottom:24px'>Customer Segmentation · RFM Analysis · Growth Strategy · CLV Prediction</p>", unsafe_allow_html=True)

# ── KPI Row ────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Total Revenue</div><div class='kpi-value'>${filtered_txn['final_amount'].sum()/1e6:.2f}M</div><div class='kpi-delta'>↑ All Channels</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Total Customers</div><div class='kpi-value' style='color:#C77DFF'>{len(filtered_rfm):,}</div><div class='kpi-delta'>↑ Active Base</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Avg Order Value</div><div class='kpi-value' style='color:#FFD93D'>${filtered_txn['final_amount'].mean():,.0f}</div><div class='kpi-delta'>↑ Per Transaction</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Champions</div><div class='kpi-value' style='color:#00FF88'>{(filtered_rfm['segment']=='Champions').sum():,}</div><div class='kpi-delta'>↑ Top Customers</div></div>", unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='kpi-card'><div class='kpi-label'>At Risk</div><div class='kpi-value' style='color:#FF6B6B'>{(filtered_rfm['segment']=='At Risk').sum():,}</div><div class='kpi-delta'>↓ Need Attention</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 RFM Segments", "🔵 Customer Clusters", "📈 Growth Analytics", "💰 CLV Analysis", "🤖 Live Predictor"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — RFM SEGMENTS
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'><p class='section-title'>RFM Customer Segmentation</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        seg_counts = filtered_rfm['segment'].value_counts().reset_index()
        seg_counts.columns = ['segment', 'count']
        seg_counts['color'] = seg_counts['segment'].map(SEGMENT_COLORS)
        fig = px.bar(seg_counts.sort_values('count', ascending=True),
                     x='count', y='segment', orientation='h',
                     title='Customer Count by RFM Segment',
                     color='segment',
                     color_discrete_map=SEGMENT_COLORS)
        fig.update_layout(**PLOTLY_THEME, showlegend=False, title_font_color='#E2E8F0')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        seg_rev = filtered_rfm.groupby('segment')['monetary'].sum().reset_index()
        fig = px.pie(seg_rev, names='segment', values='monetary',
                     title='Revenue Share by RFM Segment',
                     color='segment', color_discrete_map=SEGMENT_COLORS)
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        fig.update_traces(textfont_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        seg_rfm = filtered_rfm.groupby('segment')[['recency', 'frequency', 'monetary']].mean().reset_index()
        fig = px.scatter(seg_rfm, x='recency', y='monetary', size='frequency',
                         color='segment', color_discrete_map=SEGMENT_COLORS,
                         title='Segment Map: Recency vs Monetary (size = Frequency)',
                         text='segment')
        fig.update_traces(textposition='top center', textfont_size=9)
        fig.update_layout(**PLOTLY_THEME, showlegend=False, title_font_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        seg_clv = filtered_rfm.groupby('segment')['clv'].mean().reset_index().sort_values('clv', ascending=True)
        fig = px.bar(seg_clv, x='clv', y='segment', orientation='h',
                     title='Avg Future CLV by Segment ($)',
                     color='clv', color_continuous_scale=[[0, '#1E293B'], [1, '#00D4FF']])
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, title_font_color='#E2E8F0')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'><p class='section-title'>Segment Strategy Guide</p></div>", unsafe_allow_html=True)
    strategy = {
        'Champions': '🏆 Reward them. Early access to new products, loyalty rewards, ask for reviews.',
        'Loyal Customers': '💚 Upsell higher value products. Engage them. Ask for referrals.',
        'Potential Loyalists': '⭐ Offer membership or loyalty programs. Recommend related products.',
        'New Customers': '🆕 Provide onboarding support, early discounts, build the relationship.',
        'At Risk': '⚠️ Send personalised reactivation emails. Offer special discounts. Survey them.',
        'Cant Lose Them': '🚨 Win them back via renewals or newer products. Dont lose these to competitors.',
        'Hibernating': '😴 Offer relevant products and special discounts. Recreate brand value.',
        'Lost': '💔 Revive interest with reach out campaign. Ignore if no ROI.'
    }
    col1, col2 = st.columns(2)
    for i, (seg, desc) in enumerate(strategy.items()):
        col = col1 if i % 2 == 0 else col2
        count = len(filtered_rfm[filtered_rfm['segment'] == seg])
        rev = filtered_rfm[filtered_rfm['segment'] == seg]['monetary'].sum()
        color = SEGMENT_COLORS.get(seg, '#64748B')
        with col:
            st.markdown(f"<div class='segment-card' style='border-left: 3px solid {color}'><div class='segment-name' style='color:{color}'>{seg}</div><div class='segment-desc'>{desc}</div><div class='segment-stat'>{count:,} customers · ${rev:,.0f} revenue</div></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER CLUSTERS
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'><p class='section-title'>KMeans Customer Clusters (K=4)</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        cluster_counts = filtered_rfm['cluster_label'].value_counts().reset_index()
        cluster_counts.columns = ['cluster', 'count']
        fig = px.pie(cluster_counts, names='cluster', values='count',
                     title='Customer Distribution by Cluster',
                     color='cluster', color_discrete_map=CLUSTER_COLORS)
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        fig.update_traces(textfont_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cluster_rev = filtered_rfm.groupby('cluster_label')['monetary'].mean().reset_index()
        fig = px.bar(cluster_rev.sort_values('monetary', ascending=False),
                     x='cluster_label', y='monetary',
                     title='Avg Monetary Value by Cluster',
                     color='cluster_label', color_discrete_map=CLUSTER_COLORS)
        fig.update_layout(**PLOTLY_THEME, showlegend=False, title_font_color='#E2E8F0')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(filtered_rfm, x='recency', y='monetary',
                         color='cluster_label', color_discrete_map=CLUSTER_COLORS,
                         title='Cluster Map: Recency vs Monetary',
                         opacity=0.5, size_max=8)
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.scatter(filtered_rfm, x='frequency', y='avg_order_value',
                         color='cluster_label', color_discrete_map=CLUSTER_COLORS,
                         title='Cluster Map: Frequency vs Avg Order Value',
                         opacity=0.5, size_max=8)
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'><p class='section-title'>Cluster Profile Summary</p></div>", unsafe_allow_html=True)
    cluster_summary = filtered_rfm.groupby('cluster_label').agg(
        customers=('customer_id', 'count'),
        avg_recency=('recency', 'mean'),
        avg_frequency=('frequency', 'mean'),
        avg_monetary=('monetary', 'mean'),
        avg_order_value=('avg_order_value', 'mean'),
        avg_clv=('clv', 'mean')
    ).round(2).reset_index()
    cluster_summary.columns = ['Cluster', 'Customers', 'Avg Recency (days)', 'Avg Frequency', 'Avg Monetary ($)', 'Avg Order Value ($)', 'Avg CLV ($)']
    st.dataframe(cluster_summary.set_index('Cluster'),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — GROWTH ANALYTICS
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'><p class='section-title'>Revenue & Growth Intelligence</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        monthly = filtered_txn.groupby('month')['final_amount'].sum().reset_index().sort_values('month')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly['month'], y=monthly['final_amount'],
                                  mode='lines+markers', line=dict(color='#00D4FF', width=2.5),
                                  marker=dict(size=4), fill='tozeroy',
                                  fillcolor='rgba(0,212,255,0.1)', name='Revenue'))
        fig.update_layout(**PLOTLY_THEME, title='Monthly Revenue Trend', title_font_color='#E2E8F0')
        tick_step = max(1, len(monthly) // 10)
        fig.update_xaxes(tickvals=monthly['month'].iloc[::tick_step],
                         tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cat_rev = filtered_txn.groupby('category')['final_amount'].sum().reset_index().sort_values('final_amount', ascending=True)
        fig = px.bar(cat_rev, x='final_amount', y='category', orientation='h',
                     title='Revenue by Product Category',
                     color='final_amount', color_continuous_scale=[[0, '#1E293B'], [1, '#C77DFF']])
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, title_font_color='#E2E8F0')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        camp_rev = filtered_txn.groupby('campaign')['final_amount'].sum().reset_index().sort_values('final_amount', ascending=False)
        fig = px.bar(camp_rev, x='campaign', y='final_amount',
                     title='Revenue by Marketing Campaign',
                     color='final_amount', color_continuous_scale=[[0, '#1E293B'], [1, '#00FF88']])
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, title_font_color='#E2E8F0', xaxis_tickangle=-20)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        seasonal = filtered_txn.groupby(['year', 'month_num'])['final_amount'].sum().reset_index()
        seasonal_pivot = seasonal.pivot(index='year', columns='month_num', values='final_amount').fillna(0)
        seasonal_pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        fig = px.imshow(seasonal_pivot, color_continuous_scale='YlOrRd',
                        title='Seasonal Revenue Heatmap')
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        channel_monthly = filtered_txn.groupby(['month', 'channel'])['final_amount'].sum().reset_index().sort_values('month')
        fig = px.line(channel_monthly, x='month', y='final_amount', color='channel',
                      title='Online vs In-Store Revenue Trend',
                      color_discrete_map={'Online': '#00D4FF', 'In-Store': '#FF6B6B'})
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        fig.update_traces(line_width=2.5)
        tick_step = max(1, len(channel_monthly['month'].unique()) // 8)
        fig.update_xaxes(tickvals=sorted(channel_monthly['month'].unique())[::tick_step], tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        age_rev = filtered_merged.groupby('age_group')['final_amount'].sum().reset_index()
        age_order = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        age_rev['age_group'] = pd.Categorical(age_rev['age_group'], categories=age_order, ordered=True)
        age_rev = age_rev.sort_values('age_group')
        fig = px.bar(age_rev, x='age_group', y='final_amount',
                     title='Revenue by Age Group',
                     color='final_amount', color_continuous_scale=[[0, '#1E293B'], [1, '#FFD93D']])
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, title_font_color='#E2E8F0')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — CLV ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'><p class='section-title'>Customer Lifetime Value Intelligence</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(filtered_rfm, x='clv', nbins=50,
                           title='CLV Distribution Across All Customers',
                           color_discrete_sequence=['#00D4FF'])
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        clv_seg = filtered_rfm.groupby('segment')['clv'].mean().reset_index().sort_values('clv', ascending=False)
        fig = px.bar(clv_seg, x='segment', y='clv',
                     title='Avg CLV by RFM Segment',
                     color='segment', color_discrete_map=SEGMENT_COLORS)
        fig.update_layout(**PLOTLY_THEME, showlegend=False, title_font_color='#E2E8F0', xaxis_tickangle=-20)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        clv_tier = filtered_rfm.groupby('membership_tier')['clv'].mean().reset_index()
        tier_order = ['Bronze', 'Silver', 'Gold', 'Platinum']
        clv_tier['membership_tier'] = pd.Categorical(clv_tier['membership_tier'], categories=tier_order, ordered=True)
        clv_tier = clv_tier.sort_values('membership_tier')
        fig = px.bar(clv_tier, x='membership_tier', y='clv',
                     title='Avg CLV by Membership Tier',
                     color='membership_tier',
                     color_discrete_sequence=['#FF9F43', '#C0C0C0', '#FFD700', '#00D4FF'])
        fig.update_layout(**PLOTLY_THEME, showlegend=False, title_font_color='#E2E8F0')
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.scatter(filtered_rfm, x='monetary', y='clv',
                         color='segment', color_discrete_map=SEGMENT_COLORS,
                         title='Past Spend vs Future CLV',
                         opacity=0.5, size_max=6)
        fig.update_layout(**PLOTLY_THEME, title_font_color='#E2E8F0')
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 5 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'><p class='section-title'>Live Customer Intelligence Predictor</p></div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B;font-size:13px;margin-bottom:20px'>Enter customer behaviour metrics to get their RFM segment, KMeans cluster, and predicted CLV instantly.</p>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<p style='color:#00D4FF;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px'>Customer Profile</p>", unsafe_allow_html=True)
        pred_tier = st.selectbox("Membership Tier", ['Bronze', 'Silver', 'Gold', 'Platinum'])
        pred_age = st.selectbox("Age Group", ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
        pred_gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
        pred_channel = st.selectbox("Preferred Channel", ['Online', 'In-Store'])
        pred_email = st.selectbox("Email Subscribed", ['Yes', 'No'])

    with col_r:
        st.markdown("<p style='color:#00D4FF;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:1px'>Purchase Behaviour</p>", unsafe_allow_html=True)
        pred_recency = st.slider("Recency (days since last purchase)", 1, 900, 90)
        pred_frequency = st.slider("Frequency (number of orders)", 1, 30, 5)
        pred_monetary = st.slider("Monetary (total spend $)", 10, 30000, 2000)
        pred_aov = st.slider("Avg Order Value ($)", 10, 5000, 400)
        pred_diversity = st.slider("Category Diversity (1-8)", 1, 8, 3)

    if st.button("🚀 Analyse Customer", use_container_width=True):
        tier_map = {'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4}
        age_map = {'18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55-64': 5, '65+': 6}
        channel_map = {'Online': 1, 'In-Store': 0}

        tenure_days = max(30, pred_recency * 2)
        purchase_rate = pred_frequency / (tenure_days / 30)

        # RFM Scoring
        r_score = 5 if pred_recency <= 30 else 4 if pred_recency <= 90 else 3 if pred_recency <= 180 else 2 if pred_recency <= 365 else 1
        f_score = 5 if pred_frequency >= 10 else 4 if pred_frequency >= 7 else 3 if pred_frequency >= 4 else 2 if pred_frequency >= 2 else 1
        m_score = 5 if pred_monetary >= 8000 else 4 if pred_monetary >= 4000 else 3 if pred_monetary >= 2000 else 2 if pred_monetary >= 500 else 1
        rfm_total = r_score + f_score + m_score

        if r_score >= 4 and f_score >= 4 and m_score >= 4:
            pred_segment = 'Champions'
        elif r_score >= 3 and f_score >= 3 and m_score >= 3:
            pred_segment = 'Loyal Customers'
        elif r_score >= 4 and f_score <= 2:
            pred_segment = 'New Customers'
        elif r_score >= 3 and f_score >= 2 and m_score >= 2:
            pred_segment = 'Potential Loyalists'
        elif r_score <= 2 and f_score >= 3 and m_score >= 3:
            pred_segment = 'At Risk'
        elif r_score == 1 and f_score <= 2 and m_score <= 2:
            pred_segment = 'Lost'
        elif rfm_total >= 9:
            pred_segment = 'Loyal Customers'
        elif rfm_total >= 6:
            pred_segment = 'Potential Loyalists'
        else:
            pred_segment = 'Hibernating'

        # KMeans Cluster
        cluster_input = pd.DataFrame([[pred_recency, pred_frequency, pred_monetary,
                                        pred_aov, purchase_rate, pred_diversity]],
                                      columns=models['cluster_features'])
        scaled_input = models['scaler'].transform(cluster_input)
        cluster_id = models['kmeans'].predict(scaled_input)[0]
        cluster_label = models['cluster_labels'].get(cluster_id, f'Cluster {cluster_id}')

        # CLV Prediction
        clv_input_dict = {
            'recency': pred_recency,
            'frequency': pred_frequency,
            'monetary': pred_monetary,
            'avg_order_value': pred_aov,
            'purchase_rate': purchase_rate,
            'category_diversity': pred_diversity,
            'tenure_days': tenure_days,
            'avg_discount_used': 5.0,
            'tier_encoded': tier_map[pred_tier],
            'age_encoded': age_map[pred_age],
            'channel_encoded': channel_map[pred_channel],
            'email_subscribed': 1 if pred_email == 'Yes' else 0,
            'gender_Female': 1 if pred_gender == 'Female' else 0,
            'gender_Male': 1 if pred_gender == 'Male' else 0,
            'gender_Other': 1 if pred_gender == 'Other' else 0
        }
        clv_input_df = pd.DataFrame([clv_input_dict])
        clv_features = models['clv_features']
        clv_input_df = clv_input_df.reindex(columns=clv_features, fill_value=0)
        predicted_clv = models['clv'].predict(clv_input_df)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'><p class='section-title'>Customer Intelligence Results</p></div>", unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)
        seg_color = SEGMENT_COLORS.get(pred_segment, '#00D4FF')
        cluster_color = CLUSTER_COLORS.get(cluster_label, '#00D4FF')

        with r1:
            st.markdown(f"<div class='kpi-card'><div class='kpi-label'>RFM Segment</div><div class='kpi-value' style='font-size:18px;color:{seg_color}'>{pred_segment}</div><div class='kpi-delta'>Score: {rfm_total}/15</div></div>", unsafe_allow_html=True)
        with r2:
            st.markdown(f"<div class='kpi-card'><div class='kpi-label'>KMeans Cluster</div><div class='kpi-value' style='font-size:18px;color:{cluster_color}'>{cluster_label}</div><div class='kpi-delta'>K=4 Model</div></div>", unsafe_allow_html=True)
        with r3:
            st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Predicted CLV</div><div class='kpi-value'>${predicted_clv:,.0f}</div><div class='kpi-delta'>Future 18 months</div></div>", unsafe_allow_html=True)
        with r4:
            st.markdown(f"<div class='kpi-card'><div class='kpi-label'>RFM Score</div><div class='kpi-value' style='color:#FFD93D'>{r_score} · {f_score} · {m_score}</div><div class='kpi-delta'>R · F · M</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        strategy_text = strategy.get(pred_segment, 'Analyse customer further.')
        st.markdown(f"<div class='segment-card' style='border-left:3px solid {seg_color}'><div class='segment-name' style='color:{seg_color}'>Recommended Strategy — {pred_segment}</div><div class='segment-desc' style='font-size:14px;color:#E2E8F0;margin-top:8px'>{strategy_text}</div></div>", unsafe_allow_html=True)