import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------------------------------------------------------
# 1. Page Configuration & Aesthetic Theme Engine
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Sales Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import Inter font look & feel */
    @import url('fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }

    /* Sleek Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }

    /* Refined Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 10px 25px -5px rgba(56, 189, 248, 0.1);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }
    .metric-delta {
        font-size: 0.78rem;
        font-weight: 500;
        margin-top: 6px;
        color: #38BDF8;
    }

    /* Glassmorphism Brief Box */
    .executive-brief {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-left: 4px solid #38BDF8;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 28px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    /* Customizing Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B;
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 500;
        padding: 8px 16px;
        border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #38BDF8 !important;
        color: #0F172A !important;
        font-weight: 600;
        border: 1px solid #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Data Ingestion & Synthetic Fallback
# -----------------------------------------------------------------------------
def generate_synthetic_superstore():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")
    n_samples = 3000
    
    random_dates = np.random.choice(dates, size=n_samples)
    categories = ["Furniture", "Office Supplies", "Technology"]
    regions = ["East", "West", "Central", "South"]
    customers = [f"CUST-{i:04d}" for i in range(1, 251)]
    
    sales = np.random.exponential(scale=220, size=n_samples) + 12
    profit = sales * np.random.uniform(-0.12, 0.42, size=n_samples)
    
    return pd.DataFrame({
        "Order Date": random_dates,
        "Customer ID": np.random.choice(customers, size=n_samples),
        "Region": np.random.choice(regions, size=n_samples),
        "Category": np.random.choice(categories, size=n_samples),
        "Sales": sales,
        "Profit": profit
    })

@st.cache_data
def load_and_preprocess_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, encoding="latin1")
    else:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sample - Superstore.csv")
        if os.path.exists(data_path):
            data = pd.read_csv(data_path, encoding="latin1")
        else:
            data = generate_synthetic_superstore()

    req_cols = ["Order Date", "Customer ID", "Region", "Category", "Sales", "Profit"]
    missing = [c for c in req_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {', '.join(missing)}")

    data["Order Date"] = pd.to_datetime(data["Order Date"], errors="coerce")
    data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")
    data["Profit"] = pd.to_numeric(data["Profit"], errors="coerce")
    data = data.dropna(subset=req_cols).copy()
    data["Customer ID"] = data["Customer ID"].astype(str)
    return data

# -----------------------------------------------------------------------------
# 3. Machine Learning Engines
# -----------------------------------------------------------------------------
def run_forecasting_engine(df, horizon_days=30, model_type="Random Forest"):
    daily = df.groupby("Order Date")["Sales"].sum().reset_index().sort_values("Order Date")
    
    daily["DayOfWeek"] = daily["Order Date"].dt.dayofweek
    daily["Month"] = daily["Order Date"].dt.month
    daily["Quarter"] = daily["Order Date"].dt.quarter
    daily["Year"] = daily["Order Date"].dt.year
    daily["Lag_7"] = daily["Sales"].shift(7)
    daily["Lag_30"] = daily["Sales"].shift(30)
    daily["Rolling_7_Mean"] = daily["Sales"].shift(1).rolling(7).mean()
    
    daily_clean = daily.dropna().copy()
    if len(daily_clean) < 35:
        raise ValueError("Insufficient history for ML pipeline. Expand your date range filter.")
    
    features = ["DayOfWeek", "Month", "Quarter", "Year", "Lag_7", "Lag_30", "Rolling_7_Mean"]
    X, y = daily_clean[features], daily_clean["Sales"]
    
    split_idx = int(len(daily_clean) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42) if model_type == "Random Forest" else LinearRegression()
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)
    
    metrics = {
        "R2": r2_score(y_test, preds_test),
        "MAE": mean_absolute_error(y_test, preds_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds_test)),
        "MAPE": np.mean(np.abs((y_test - preds_test) / np.maximum(y_test, 1e-5))) * 100
    }
    
    last_date = daily_clean["Order Date"].max()
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, horizon_days + 1)]
    future_preds = []
    curr_series = daily_clean["Sales"].tolist()
    
    for dt in future_dates:
        feat_df = pd.DataFrame([{
            "DayOfWeek": dt.dayofweek, "Month": dt.month, "Quarter": dt.quarter, "Year": dt.year,
            "Lag_7": curr_series[-7], "Lag_30": curr_series[-30], "Rolling_7_Mean": np.mean(curr_series[-7:])
        }])
        pred_val = max(0.0, float(model.predict(feat_df)[0]))
        future_preds.append(pred_val)
        curr_series.append(pred_val)
        
    return daily_clean, pd.DataFrame({"Order Date": future_dates, "Forecasted Sales": future_preds}), metrics

def detect_sales_anomalies(df, contamination=0.03):
    daily = df.groupby("Order Date").agg({"Sales": "sum", "Profit": "sum"}).reset_index()
    scaled_feats = StandardScaler().fit_transform(daily[["Sales", "Profit"]])
    daily["Is_Anomaly"] = (IsolationForest(contamination=contamination, random_state=42).fit_predict(scaled_feats) == -1)
    return daily

def compute_rfm_segmentation(df):
    ref_date = df["Order Date"].max() + datetime.timedelta(days=1)
    rfm = df.groupby("Customer ID").agg({"Order Date": lambda x: (ref_date - x.max()).days, "Sales": ["count", "sum"]})
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    
    rfm_scaled = StandardScaler().fit_transform(rfm)
    rfm["Cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(rfm_scaled)
    
    ranks = rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False).index
    rfm["Segment"] = rfm["Cluster"].map({ranks[0]: "VIP Champions", ranks[1]: "Core Loyalists", ranks[2]: "At-Risk / Low Volume"})
    return rfm

# -----------------------------------------------------------------------------
# 4. Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Control Suite")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

try:
    df_raw = load_and_preprocess_data(uploaded_file)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

st.sidebar.success(f"Loaded: {len(df_raw):,} records")
st.sidebar.markdown("---")

min_d, max_d = df_raw["Order Date"].min().date(), df_raw["Order Date"].max().date()
date_range = st.sidebar.date_input("Date Range Filter", value=(min_d, max_d), min_value=min_d, max_value=max_d)
start_d, end_d = date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (min_d, max_d)

selected_regions = st.sidebar.multiselect("Region", df_raw["Region"].unique(), default=list(df_raw["Region"].unique()))
selected_categories = st.sidebar.multiselect("Category", df_raw["Category"].unique(), default=list(df_raw["Category"].unique()))

model_choice = st.sidebar.selectbox("Forecasting Algorithm", ["Random Forest", "Linear Regression"])
forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", 14, 90, 30, step=7)

filtered_df = df_raw[
    (df_raw["Order Date"].dt.date >= start_d) &
    (df_raw["Order Date"].dt.date <= end_d) &
    (df_raw["Region"].isin(selected_regions)) &
    (df_raw["Category"].isin(selected_categories))
]

if filtered_df.empty:
    st.warning("No records match your filters. Please expand your selection range.")
    st.stop()

# -----------------------------------------------------------------------------
# 5. Main Layout & Tabs
# -----------------------------------------------------------------------------
st.markdown("## ⚡ Enterprise Sales Intelligence")
st.markdown("<p style='color: #94A3B8; margin-top: -10px;'>Advanced predictive demand analytics & operational risk monitoring suite.</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Executive Overview & ML Forecast",
    "🚨 Anomaly Detection Engine",
    "👥 Customer RFM Segmentation",
    "🎛️ Dynamic Profit Simulator"
])

# --- TAB 1 ---
with tab1:
    tot_sales = filtered_df["Sales"].sum()
    tot_profit = filtered_df["Profit"].sum()
    margin_pct = (tot_profit / tot_sales * 100) if tot_sales > 0 else 0
    
    try:
        hist_daily, forecast_df, metrics = run_forecasting_engine(filtered_df, horizon_days=forecast_horizon, model_type=model_choice)
        projected_rev = forecast_df["Forecasted Sales"].sum()
    except Exception as err:
        st.error(f"Error: {err}")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-label">Historical Revenue</div><div class="metric-val">${tot_sales:,.0f}</div><div class="metric-delta">Active Selection Filter</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-label">Net Profit Margin</div><div class="metric-val">{margin_pct:.1f}%</div><div class="metric-delta">Net Profit: ${tot_profit:,.0f}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-label">Projected Rev ({forecast_horizon}D)</div><div class="metric-val">${projected_rev:,.0f}</div><div class="metric-delta">Model: {model_choice}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-label">Validation Score</div><div class="metric-val">R² {metrics["R2"]:.2f}</div><div class="metric-delta">MAPE: {metrics["MAPE"]:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    top_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()
    top_cat = filtered_df.groupby("Category")["Sales"].sum().idxmax()
    
    st.markdown(f"""
    <div class="executive-brief">
        <h4 style="margin: 0 0 8px 0; color: #38BDF8; font-size: 1rem;">💡 Automated Executive Summary</h4>
        • Projected revenue for the next <b>{forecast_horizon} days</b> stands at <b>${projected_rev:,.2f}</b> using <b>{model_choice}</b>.<br>
        • Operating margins maintain stability at <b>{margin_pct:.1f}%</b>, with peak regional contribution anchored in <b>{top_region}</b>.<br>
        • Key Inventory Directive: Prioritize stock levels for the <b>{top_cat}</b> category to capture primary revenue velocity.
    </div>
    """, unsafe_allow_html=True)

    fig_f = go.Figure()
    recent_hist = hist_daily.tail(120)
    fig_f.add_trace(go.Scatter(x=recent_hist["Order Date"], y=recent_hist["Sales"], name="Historical Sales", line=dict(color="#38BDF8", width=2)))
    fig_f.add_trace(go.Scatter(x=forecast_df["Order Date"], y=forecast_df["Forecasted Sales"], name="ML Projection", line=dict(color="#F43F5E", width=2, dash="dash")))
    
    fig_f.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_f, use_container_width=True)

# --- TAB 2 ---
with tab2:
    st.markdown("### 🚨 Unsupervised Anomaly Engine")
    anom_data = detect_sales_anomalies(filtered_df)
    flagged = anom_data[anom_data["Is_Anomaly"]]
    
    col_a1, col_a2 = st.columns([1, 3])
    with col_a1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Flagged Outliers</div><div class="metric-val" style="color: #EF4444;">{len(flagged)}</div><div class="metric-delta">Significant Volume Breaks</div></div>', unsafe_allow_html=True)
    with col_a2:
        st.info("Isolation Forest detects multidimensional sales and profit anomalies to highlight operational volatility or system data errors.")
        
    fig_a = px.scatter(
        anom_data, x="Order Date", y="Sales", color="Is_Anomaly",
        color_discrete_map={False: "#38BDF8", True: "#EF4444"},
        template="plotly_dark", size=anom_data["Sales"].clip(lower=10), height=380
    )
    fig_a.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_a, use_container_width=True)
    
    if not flagged.empty:
        st.markdown("#### Exception Logs")
        st.dataframe(flagged[["Order Date", "Sales", "Profit"]].sort_values("Sales", ascending=False).style.format({"Sales": "${:,.2f}", "Profit": "${:,.2f}"}), use_container_width=True)

# --- TAB 3 ---
with tab3:
    st.markdown("### 👥 Customer Value Segmentation (RFM)")
    rfm_df = compute_rfm_segmentation(filtered_df)
    
    r_col1, r_col2 = st.columns([2, 1])
    with r_col1:
        fig_rfm = px.scatter_3d(
            rfm_df.reset_index(), x="Recency", y="Frequency", z="Monetary",
            color="Segment", opacity=0.85, template="plotly_dark",
            color_discrete_sequence=["#10B981", "#3B82F6", "#F59E0B"]
        )
        fig_rfm.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0), height=450)
        st.plotly_chart(fig_rfm, use_container_width=True)
    with r_col2:
        st.markdown("#### Segment Tiers")
        st.dataframe(rfm_df.groupby("Segment").agg({"Recency": "mean", "Frequency": "mean", "Monetary": "mean"}).reset_index().style.format({"Recency": "{:.0f}d", "Frequency": "{:.1f}", "Monetary": "${:,.0f}"}), use_container_width=True)

# --- TAB 4 ---
with tab4:
    st.markdown("### 🎛️ Strategic What-If Simulator")
    s1, s2, s3 = st.columns(3)
    price_shift = s1.slider("Price Adjustment (%)", -25, 25, 0, step=1)
    discount_shift = s2.slider("Discount Offset (%)", -10, 15, 0, step=1)
    marketing_exp = s3.number_input("Incremental Marketing OpEx ($)", min_value=0, max_value=100000, value=5000, step=1000)
    
    base_proj = forecast_df["Forecasted Sales"].sum()
    sim_sales = base_proj * (1 + (price_shift / 100.0))
    sim_net = (sim_sales * (margin_pct / 100.0)) - (sim_sales * (discount_shift / 100.0)) - marketing_exp
    base_profit = base_proj * (margin_pct / 100.0)
    delta = sim_net - base_profit
    
    st.markdown("---")
    res1, res2, res3 = st.columns(3)
    res1.markdown(f'<div class="metric-card"><div class="metric-label">Baseline Projected Sales</div><div class="metric-val">${base_proj:,.0f}</div></div>', unsafe_allow_html=True)
    res2.markdown(f'<div class="metric-card"><div class="metric-label">Simulated Revenue</div><div class="metric-val">${sim_sales:,.0f}</div><div class="metric-delta">Shift: {price_shift:+} %</div></div>', unsafe_allow_html=True)
    res3.markdown(f'<div class="metric-card"><div class="metric-label">Simulated Net Profit</div><div class="metric-val">${sim_net:,.0f}</div><div class="metric-delta" style="color: {"#10B981" if delta >= 0 else "#EF4444"};">Variance: ${delta:+,.0f}</div></div>', unsafe_allow_html=True)