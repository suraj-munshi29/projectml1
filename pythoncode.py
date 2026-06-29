# ==========================================
# 1. TIME-SERIES PREPARATION
# ==========================================
def prepare_time_series(file_path):
    """Loads, cleans, and ensures continuity of daily observations."""
    df = pd.read_csv(file_path)

    # Clean numeric columns
    numeric_cols = [
        'Children apprehended and placed in CBP custody*',
        'Children in CBP custody',
        'Children transferred out of CBP custody',
        'Children in HHS Care',
        'Children discharged from HHS Care'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert Date to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'].str.replace('"', ''))
    df = df.sort_values('Date')

    # Aggregate duplicate dates by taking the mean of numeric columns
    # This handles cases where multiple entries exist for the same date
    df = df.groupby('Date').mean().reset_index()

    # Set Date as index after ensuring uniqueness
    df = df.set_index('Date')

    # Ensure continuity: Resample to Daily ('D') and interpolate missing days
    df_daily = df.resample('D').interpolate(method='time')

    return df_daily

def decompose_series(df, target_col='Children in HHS Care'):
    """Decomposes series into trend, seasonality, and residuals."""
    # Period 7 for weekly seasonality on daily data
    decomposition = seasonal_decompose(df[target_col].dropna(), model='additive', period=7)

    # Plotting decomposition
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle('Time-Series Decomposition', fontsize=14)
    plt.tight_layout()
    plt.show()

    return decomposition

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    """Generates lags, rolling metrics, flow signals, and calendar effects."""
    df_feat = df.copy()
    target = 'Children in HHS Care'

    # Flow-Based Signals (Net Pressure)
    if 'Children transferred out of CBP custody' in df.columns and 'Children discharged from HHS Care' in df.columns:
        df_feat['Net_Pressure'] = df_feat['Children transferred out of CBP custody'] - df_feat['Children discharged from HHS Care']

    # Calendar Effects
    df_feat['DayOfWeek'] = df_feat.index.dayofweek
    df_feat['Month'] = df_feat.index.month

    # Lags (t-1, t-7, t-14)
    for lag in [1, 7, 14]:
        df_feat[f'{target}_lag_{lag}'] = df_feat[target].shift(lag)

    # Rolling Averages & Variances (7-day and 14-day)
    for window in [7, 14]:
        df_feat[f'{target}_roll_mean_{window}'] = df_feat[target].shift(1).rolling(window=window).mean()
        df_feat[f'{target}_roll_var_{window}'] = df_feat[target].shift(1).rolling(window=window).var()

    # Drop rows with NaNs introduced by shifting
    df_feat = df_feat.dropna()
    return df_feat

# ==========================================
# 3. TRAIN-TEST STRATEGY & MODELING
# ==========================================
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_and_forecast(df, target_col='Children in HHS Care'):
    """Executes time-based split and trains multiple forecasting models."""

    # Time-based split (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    y_train = train[target_col]
    y_test = test[target_col]

    features = [c for c in df.columns if c != target_col and df[c].dtype in [np.float64, np.int32, np.int64]]
    X_train, X_test = train[features], test[features]

    predictions = {}

    # --- Baseline Models ---
    # Naive Persistence (t-1)
    predictions['Naive'] = test[f'{target_col}_lag_1']

    # Moving Average Forecast (7-day)
    predictions['Moving Average (7d)'] = test[f'{target_col}_roll_mean_7']

    # --- Statistical Models ---
    # SARIMA (using statsmodels)
    try:
        sarima_model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7))
        sarima_fit = sarima_model.fit(disp=False)
        predictions['SARIMA'] = sarima_fit.forecast(steps=len(y_test))
    except:
        print("SARIMA convergence failed; skipping.")

    # Exponential Smoothing (Holt-Winters)
    try:
        ets_model = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=7)
        ets_fit = ets_model.fit()
        predictions['ETS'] = ets_fit.forecast(steps=len(y_test))
    except:
        print("ETS failed; skipping.")

    # --- Machine Learning Models ---
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    predictions['Random Forest'] = rf.predict(X_test)

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    predictions['Gradient Boosting'] = gb.predict(X_test)

    return y_test, predictions

# ==========================================
# 4. EVALUATION & KPIs
# ==========================================
def calculate_kpis(y_test, predictions, capacity_threshold=3000):
    """Calculates standard metrics and domain-specific KPIs."""

    results = []

    for model_name, y_pred in predictions.items():
        # Standard Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # Custom KPIs
        forecast_accuracy = max(0, 100 - mape)  # Reliability of predictions

        # Capacity Breach Probability (Percentage of test days forecast exceeds capacity)
        breach_prob = np.mean(y_pred > capacity_threshold) * 100

        # Forecast Stability Index (Inverse of standard deviation of errors)
        errors = y_test - y_pred
        stability_index = 1 / (np.std(errors) + 1e-5)

        results.append({
            'Model': model_name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE (%)': round(mape, 2),
            'Forecast Accuracy (%)': round(forecast_accuracy, 2),
            'Breach Prob (%)': round(breach_prob, 2),
            'Stability Index': round(stability_index, 4)
        })

    results_df = pd.DataFrame(results).set_index('Model')
    return results_df
# ==========================================
# EXECUTION WORKFLOW
# ==========================================
if __name__ == "__main__":
    file_path = "HHS_Unaccompanied_Alien_Children_Program.csv"

    print("1. Preparing Time-Series...")
    df_daily = prepare_time_series(file_path)

    print("2. Decomposing Series...")
    # decompose_series(df_daily) # Uncomment to view the trend/seasonality plots

    print("3. Engineering Features...")
    df_features = engineer_features(df_daily)

    print("4. Training Models & Forecasting...")
    y_test, predictions = train_and_forecast(df_features)

    print("5. Evaluating KPIs...\n")
    # Assuming system capacity is ~3000 for demonstration
    evaluation_matrix = calculate_kpis(y_test, predictions, capacity_threshold=3000)

    print(evaluation_matrix.to_string())

    # Visualizing Top Performers
    plt.figure(figsize=(14, 6))
    plt.plot(y_test.index, y_test, label='Actual Care Load', color='black', linewidth=2)
    plt.plot(y_test.index, predictions['Gradient Boosting'], label='Gradient Boosting', linestyle='--')
    plt.plot(y_test.index, predictions['Random Forest'], label='Random Forest', linestyle='-.')
    if 'SARIMA' in predictions:
        plt.plot(y_test.index, predictions['SARIMA'], label='SARIMA', linestyle=':')

    plt.title('Multi-Model Forecast Evaluation: Children in HHS Care')
    plt.ylabel('Number of Children')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
