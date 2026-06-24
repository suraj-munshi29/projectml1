import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. APP CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="HHS Care Load Forecaster", page_icon="📈", layout="wide")
st.title("📊 Predictive Forecasting: HHS Care Load & Discharge Demand")
st.markdown("Interactive dashboard for tracking and predicting unaccompanied children system capacity.")

# ==========================================
# 2. DATA LOADING & PREPARATION (Cached for Speed)
# ==========================================
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")
    
    # Clean numeric columns
    numeric_cols = ['Children in HHS Care', 'Children discharged from HHS Care']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'].str.replace('"', ''))
    df = df.set_index('Date').sort_index()
    
    # Ensure continuity (fill weekend gaps)
    df = df.resample('D').interpolate(method='time').dropna(subset=numeric_cols)
    return df

data = load_and_prep_data()

# ==========================================
# 3. SIDEBAR CONTROLS (User Capabilities)
# ==========================================
st.sidebar.header("⚙️ Forecast Settings")

# Forecast Horizon Selector
horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=60, value=14, step=7)

# Model Toggle
model_choice = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["Gradient Boosting", "Random Forest", "Baseline (7-day MA)"]
)

# Scenario Comparison (Surge Multiplier)
st.sidebar.markdown("---")
st.sidebar.header("🌪️ Scenario Planning")
scenario_multiplier = st.sidebar.slider(
    "System Shock (Demand Surge/Drop %)", 
    min_value=-30, max_value=50, value=0, step=5,
    help="Simulates a sudden % increase or decrease in capacity constraints."
) / 100.0

# ==========================================
# 4. MODELING & FORECASTING ENGINE
# ==========================================
def engineer_features(df, target):
    df_feat = pd.DataFrame(index=df.index)
    df_feat[target] = df[target]
    df_feat['DayOfWeek'] = df.index.dayofweek
    df_feat['Month'] = df.index.month
    df_feat['Lag_1'] = df[target].shift(1)
    df_feat['Lag_7'] = df[target].shift(7)
    df_feat['Roll_Mean_7'] = df[target].shift(1).rolling(7).mean()
    return df_feat.dropna()

def generate_forecast(df, target, horizon, model_name, multiplier):
    df_feat = engineer_features(df, target)
    
    # Train/Test Split logic for error estimation
    X = df_feat[['DayOfWeek', 'Month', 'Lag_1', 'Lag_7', 'Roll_Mean_7']]
    y = df_feat[target]
    
    # Select Model
    if model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = None # Baseline handled manually
        
    # Fit Model and calculate historical error (for Confidence Intervals)
    if model:
        model.fit(X, y)
        train_preds = model.predict(X)
        mae = mean_absolute_error(y, train_preds)
    else:
        mae = np.std(y[-14:]) # Baseline error approximation
        
    # Walk-forward forecasting
    last_known = df_feat.iloc[-1].copy()
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=horizon)
    predictions = []
    
    current_val = last_known[target]
    history = list(y.values)
    
    for date in future_dates:
        if model:
            # Rebuild features iteratively
            next_X = pd.DataFrame([{
                'DayOfWeek': date.dayofweek,
                'Month': date.month,
                'Lag_1': current_val,
                'Lag_7': history[-7],
                'Roll_Mean_7': np.mean(history[-7:])
            }])
            pred = model.predict(next_X)[0]
        else:
            # Baseline: 7-day moving average
            pred = np.mean(history[-7:])
            
        # Apply scenario multiplier
        pred = pred * (1 + multiplier)
        predictions.append(pred)
        
        # Update state for next step
        current_val = pred
        history.append(pred)
        
    # Generate Confidence Intervals (widening over time)
    uncertainty_growth = np.linspace(1.0, 2.5, horizon) # Uncertainty grows as we look further out
    lower_bound = predictions - (mae * 1.96 * uncertainty_growth)
    upper_bound = predictions + (mae * 1.96 * uncertainty_growth)
    
    return future_dates, predictions, lower_bound, upper_bound

# ==========================================
# 5. VISUALIZATION (Plotly Dashboards)
# ==========================================
def plot_forecast(historical_dates, historical_data, future_dates, predictions, lower, upper, title, y_label, color):
    fig = go.Figure()
    
    # Historical Data
    fig.add_trace(go.Scatter(
        x=historical_dates[-60:], y=historical_data[-60:], # Show last 60 days
        mode='lines+markers', name='Historical',
        line=dict(color='black', width=2)
    ))
    
    # Forecast Data
    fig.add_trace(go.Scatter(
        x=future_dates, y=predictions,
        mode='lines+markers', name='Forecast',
        line=dict(color=color, width=3, dash='dash')
    ))
    
    # Confidence Interval (Upper & Lower bound fill)
    fig.add_trace(go.Scatter(
        x=np.concatenate([future_dates, future_dates[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill='toself',
        fillcolor=f'rgba{color[3:-1]}, 0.2)' if color.startswith('rgb') else 'rgba(128, 128, 128, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='95% Confidence Interval'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        template="plotly_white",
        height=400
    )
    return fig

# --- RENDER DASHBOARD ---
tab1, tab2 = st.tabs(["Care Load Forecast", "Discharge Demand Forecast"])

with tab1:
    st.subheader(f"Future Care Load ({model_choice})")
    f_dates, preds, lb, ub = generate_forecast(data, 'Children in HHS Care', horizon, model_choice, scenario_multiplier)
    
    fig1 = plot_forecast(
        data.index, data['Children in HHS Care'], 
        f_dates, preds, lb, ub, 
        "HHS Capacity Forecast (Next {} Days)".format(horizon), 
        "Children in Care", "rgb(31, 119, 180)"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Care Load", f"{int(data['Children in HHS Care'].iloc[-1]):,}")
    col2.metric(f"Forecasted Load (Day {horizon})", f"{int(preds[-1]):,}", f"{int(preds[-1] - data['Children in HHS Care'].iloc[-1]):,} vs today")
    if np.any(preds > 3000):
        col3.error("⚠️ Capacity Breach Warning (>3000)")
    else:
        col3.success("✅ System Stable")

with tab2:
    st.subheader(f"Discharge Demand Panel ({model_choice})")
    f_dates_d, preds_d, lb_d, ub_d = generate_forecast(data, 'Children discharged from HHS Care', horizon, model_choice, scenario_multiplier)
    
    fig2 = plot_forecast(
        data.index, data['Children discharged from HHS Care'], 
        f_dates_d, preds_d, lb_d, ub_d, 
        "Expected Daily Discharges (Next {} Days)".format(horizon), 
        "Discharges per Day", "rgb(44, 160, 44)"
    )
    st.plotly_chart(fig2, use_container_width=True)