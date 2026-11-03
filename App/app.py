import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------------------------------------------------------
# 1. Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Sales Intelligence & Risk Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0f172a; color: #f8fafc; }
    div[data-testid="stMetricValue"] { font-weight: 700; color: #38bdf8; }
    .stMetric { background-color: #1e293b; padding: 16px; border-radius: 12px; border: 1px solid #334155; }
    .insight-card { background-color: #1e293b; border-left: 4px solid #38bdf8; padding: 14px; border-radius: 6px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Load the original Sample Superstore dataset
# -----------------------------------------------------------------------------
@st.cache_data
def load_superstore_data(uploaded_file=None):
    """Load the real Sample Superstore CSV (not generated/sample data)."""
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, encoding="latin1")
    else:
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sample - Superstore.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                "Sample - Superstore.csv was not found. Place it in the same folder as app.py "
                "or upload it using the sidebar."
            )
        data = pd.read_csv(data_path, encoding="latin1")

    required_columns = ["Order Date", "Customer ID", "Region", "Category", "Sales", "Profit"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing_columns))

    # The original Sample Superstore file stores dates in month/day/year format.
    data["Order Date"] = pd.to_datetime(data["Order Date"], errors="coerce")
    data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")
    data["Profit"] = pd.to_numeric(data["Profit"], errors="coerce")
    data = data.dropna(subset=required_columns).copy()
    data["Customer ID"] = data["Customer ID"].astype(str)
    return data

# -----------------------------------------------------------------------------
# 3. Machine Learning Engines
# -----------------------------------------------------------------------------

# --- A. Time Series & Ensemble Engine ---
def run_forecast_pipeline(df, horizon_days=30, selected_model_name="Random Forest"):
    daily = df.groupby("Order Date")["Sales"].sum().reset_index().sort_values("Order Date")
    
    daily["DayOfWeek"] = daily["Order Date"].dt.dayofweek
    daily["Month"] = daily["Order Date"].dt.month
    daily["Quarter"] = daily["Order Date"].dt.quarter
    daily["Year"] = daily["Order Date"].dt.year
    daily["Lag_7"] = daily["Sales"].shift(7)
    daily["Lag_30"] = daily["Sales"].shift(30)
    daily["Rolling_7_Mean"] = daily["Sales"].shift(1).rolling(7).mean()
    
    daily = daily.dropna()
    if len(daily) < 10:
        raise ValueError("Not enough daily sales history for forecasting. Select a wider date range.")
    
    X = daily[["DayOfWeek", "Month", "Quarter", "Year", "Lag_7", "Lag_30", "Rolling_7_Mean"]]
    y = daily["Sales"]
    
    split_idx = int(len(daily) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression()
    }
    
    model = models[selected_model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAPE": np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    
    # Recursive Forecast
    last_date = daily["Order Date"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]
    future_preds = []
    curr_df = daily.copy()
    
    for date in future_dates:
        l7 = curr_df.iloc[-7]["Sales"]
        l30 = curr_df.iloc[-30]["Sales"]
        r7 = curr_df.iloc[-7:]["Sales"].mean()
        
        feat = pd.DataFrame([{
            "DayOfWeek": date.dayofweek,
            "Month": date.month,
            "Quarter": date.quarter,
            "Year": date.year,
            "Lag_7": l7,
            "Lag_30": l30,
            "Rolling_7_Mean": r7
        }])
        
        pred = max(0, model.predict(feat)[0])
        future_preds.append(pred)
        curr_df = pd.concat([curr_df, pd.DataFrame([{"Order Date": date, "Sales": pred}])], ignore_index=True)
        
    forecast_df = pd.DataFrame({"Order Date": future_dates, "Forecasted Sales": future_preds})
    return daily, forecast_df, metrics

# --- B. Isolation Forest Anomaly Detection ---
def detect_anomalies(df):
    daily = df.groupby("Order Date").agg({"Sales": "sum", "Profit": "sum"}).reset_index()
    iso = IsolationForest(contamination=0.03, random_state=42)
    daily["Anomaly_Score"] = iso.fit_predict(daily[["Sales", "Profit"]])
    daily["Is_Anomaly"] = daily["Anomaly_Score"] == -1
    return daily

# --- C. RFM Customer Segmentation ---
def compute_rfm_clusters(df):
    ref_date = df["Order Date"].max() + timedelta(days=1)
    rfm = df.groupby("Customer ID").agg({
        "Order Date": lambda x: (ref_date - x.max()).days,
        "Sales": ["count", "sum"]
    })
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm)
    
    # Label Clusters
    cluster_means = rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False)
    rank_map = {cluster_means.index[0]: "VIP Champions", cluster_means.index[1]: "Loyalists", cluster_means.index[2]: "At-Risk / Low Spenders"}
    rfm["Segment"] = rfm["Cluster"].map(rank_map)
    
    return rfm

# -----------------------------------------------------------------------------
# 4. Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.markdown("⚡ **AI Control Panel**")

file_upload = st.sidebar.file_uploader("Upload CSV Dataset (optional)", type=["csv"])
try:
    df = load_superstore_data(file_upload)
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.info("Keep 'Sample - Superstore.csv' in the same folder as app.py, or upload the CSV in the sidebar.")
    st.stop()

st.sidebar.success(f"Loaded actual dataset: {len(df):,} records | {len(df.columns)} columns")

st.sidebar.markdown("---")
st.sidebar.markdown("🔍 **Filters & ML Config**")

min_d, max_d = df["Order Date"].min().date(), df["Order Date"].max().date()
start_d = st.sidebar.date_input("Start Date", min_d, min_value=min_d, max_value=max_d)
end_d = st.sidebar.date_input("End Date", max_d, min_value=min_d, max_value=max_d)

regions = st.sidebar.multiselect("Region", df["Region"].unique(), default=df["Region"].unique())
categories = st.sidebar.multiselect("Category", df["Category"].unique(), default=df["Category"].unique())

model_choice = st.sidebar.selectbox("Forecasting Algorithm", ["Random Forest", "Linear Regression"])
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 14, 90, 30, step=7)

# Apply Filters
filtered = df[
    (df["Order Date"].dt.date >= start_d) &
    (df["Order Date"].dt.date <= end_d) &
    (df["Region"].isin(regions)) &
    (df["Category"].isin(categories))
]

if filtered.empty:
    st.warning("No records match the selected filters. Please widen the date range or select more regions/categories.")
    st.stop()

# -----------------------------------------------------------------------------
# 5. Dashboard Layout (Tabs)
# -----------------------------------------------------------------------------
st.title("⚡ AI Sales Intelligence Suite")
st.caption("Next-Gen Demand Forecasting, Anomaly Detection & Customer Intelligence")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Executive Overview & ML Forecast",
    "🚨 Anomaly Detection Engine",
    "👥 Customer RFM Segmentation",
    "🎛️ What-If Profit Simulator"
])

# -----------------------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW & FORECASTING
# -----------------------------------------------------------------------------
with tab1:
    tot_sales = filtered["Sales"].sum()
    tot_profit = filtered["Profit"].sum()
    margin = (tot_profit / tot_sales) * 100 if tot_sales > 0 else 0
    
    hist_daily, forecast_df, metrics = run_forecast_pipeline(filtered, horizon_days=forecast_days, selected_model_name=model_choice)
    proj_sales = forecast_df["Forecasted Sales"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${tot_sales:,.2f}")
    c2.metric("Net Profit Margin", f"{margin:.1f}%")
    c3.metric(f"Projected Revenue ({forecast_days}D)", f"${proj_sales:,.2f}")
    c4.metric("Model R² Score", f"{metrics['R2']:.3f}")
    
    st.markdown("---")
    
    # Natural Language Auto-Insight Generator
    st.subheader("💡 Automated Executive Brief")
    st.markdown(f"""
    <div class="insight-card">
        • <b>Revenue Trajectory:</b> Projected revenue over the next {forecast_days} days is <b>${proj_sales:,.2f}</b> using the {model_choice} model (R²: {metrics['R2']:.2f}, MAPE: {metrics['MAPE']:.1f}%).<br>
        • <b>Margin Status:</b> Current profit margins stand at <b>{margin:.1f}%</b>. Top sales performance is centered in the <b>{filtered.groupby('Region')['Sales'].sum().idxmax()}</b> region.<br>
        • <b>Action Item:</b> Keep inventory elevated for the <b>{filtered.groupby('Category')['Sales'].sum().idxmax()}</b> category, which represents the highest volume driver.
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive Plotly Chart
    fig_f = go.Figure()
    recent = hist_daily.tail(90)
    fig_f.add_trace(go.Scatter(x=recent["Order Date"], y=recent["Sales"], name="Historical Sales", line=dict(color="#38bdf8", width=2)))
    fig_f.add_trace(go.Scatter(x=forecast_df["Order Date"], y=forecast_df["Forecasted Sales"], name="ML Projection", line=dict(color="#f43f5e", width=2, dash="dash")))
    fig_f.update_layout(title=f"Historical vs Forecasted Revenue ({forecast_days} Days)", template="plotly_dark", height=400)
    st.plotly_chart(fig_f, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: ANOMALY DETECTION
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🚨 Unsupervised Isolation Forest Anomaly Detection")
    st.write("Identifies irregular sales spikes, operational glitches, or sudden demand collapses.")
    
    anom_df = detect_anomalies(filtered)
    anomalies_found = anom_df[anom_df["Is_Anomaly"]]
    
    st.warning(f"Detected **{len(anomalies_found)} suspicious date(s)** requiring audit.")
    
    fig_a = px.scatter(
        anom_df, x="Order Date", y="Sales", color="Is_Anomaly",
        color_discrete_map={False: "#38bdf8", True: "#ef4444"},
        title="Sales Anomaly Distribution", template="plotly_dark",
        size=anom_df["Sales"].clip(lower=10)
    )
    st.plotly_chart(fig_a, use_container_width=True)
    
    if not anomalies_found.empty:
        st.write("📋 **Flagged Anomaly Logs:**")
        st.dataframe(anomalies_found[["Order Date", "Sales", "Profit"]].sort_values("Sales", ascending=False), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: CUSTOMER RFM SEGMENTATION
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("👥 Recency, Frequency & Monetary (RFM) Segmentation")
    st.write("K-Means machine learning clustering categorizes customers based on transaction history.")
    
    rfm_df = compute_rfm_clusters(filtered)
    
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        fig_rfm = px.scatter_3d(
            rfm_df.reset_index(), x="Recency", y="Frequency", z="Monetary",
            color="Segment", opacity=0.8, template="plotly_dark",
            title="3D Customer Clusters"
        )
        st.plotly_chart(fig_rfm, use_container_width=True)
    
    with sc2:
        st.write("📊 **Segment Summary:**")
        seg_summary = rfm_df.groupby("Segment").agg({
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean"
        }).reset_index()
        st.dataframe(seg_summary, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: WHAT-IF SCENARIO SIMULATOR
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("🎛️ Dynamic Profit & Pricing Simulator")
    st.write("Simulate strategic business changes on projected revenue and margins.")
    
    sim_c1, sim_c2, sim_c3 = st.columns(3)
    price_adj = sim_c1.slider("Price Adjustment (%)", -20, 30, 0, step=5)
    discount_adj = sim_c2.slider("Discount Strategy Change (%)", -10, 15, 0, step=1)
    marketing_boost = sim_c3.slider("Marketing Spend ($)", 0, 50000, 5000, step=2500)
    
    # Calculate Impacted Metrics
    base_rev = forecast_df["Forecasted Sales"].sum()
    simulated_rev = base_rev * (1 + (price_adj / 100.0))
    simulated_profit = (simulated_rev * (margin / 100.0)) - (simulated_rev * (discount_adj / 100.0)) - marketing_boost
    
    res_c1, res_c2, res_c3 = st.columns(3)
    res_c1.metric("Baseline Forecast Revenue", f"${base_rev:,.2f}")
    res_c2.metric("Simulated Net Revenue", f"${simulated_rev:,.2f}", delta=f"{price_adj}% Price Shift")
    res_c3.metric("Simulated Net Profit", f"${simulated_profit:,.2f}", delta=f"${simulated_profit - (base_rev * (margin/100)):,.2f}")