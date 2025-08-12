# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
from io import BytesIO

from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="RainPulse — Stock + Rain ML", layout="wide")

# -------------------------
# Helper utils
# -------------------------
@st.cache_data(show_spinner=False)
def fetch_stock(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch stock data and flatten columns if necessary."""
    stock = yf.download(ticker, start=start, end=end, auto_adjust=False)
    if stock.empty:
        return pd.DataFrame()
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = [' '.join(map(str, c)).strip() for c in stock.columns.values]
    # detect close-like column, rename to Close
    if "Close" not in stock.columns:
        close_candidates = [c for c in stock.columns if "close" in str(c).lower()]
        if close_candidates:
            stock.rename(columns={close_candidates[0]: "Close"}, inplace=True)
    # detect volume-like column, rename to Volume
    if "Volume" not in stock.columns:
        vol_candidates = [c for c in stock.columns if "volume" in str(c).lower()]
        if vol_candidates:
            stock.rename(columns={vol_candidates[0]: "Volume"}, inplace=True)
    return stock

@st.cache_data(show_spinner=False)
def fetch_rain_open_meteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch daily precipitation_sum from Open-Meteo archive."""
    base = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "precipitation_sum",
        "timezone": "UTC"
    }
    r = requests.get(base, params=params, timeout=15)
    r.raise_for_status()
    js = r.json()
    if "daily" not in js or "time" not in js["daily"]:
        raise RuntimeError("Open-Meteo returned unexpected structure.")
    df = pd.DataFrame({
        "Date": pd.to_datetime(js["daily"]["time"]),
        "Rain": js["daily"]["precipitation_sum"]
    }).sort_values("Date")
    df["Rain_lag1"] = df["Rain"].shift(1)
    df["Rain_lag2"] = df["Rain"].shift(2)
    return df

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    b = BytesIO()
    df.to_csv(b, index=False)
    return b.getvalue()

# -------------------------
# UI: Sidebar inputs
# -------------------------
st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Stock ticker (Yahoo)", value="RCF.NS")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime(datetime.today().date()))
# Location: default Karnataka coords, allow user override
loc_choice = st.sidebar.selectbox("Rainfall location", ["Custom lat/lon", "Karnataka (default)"])
if loc_choice == "Custom lat/lon":
    lat_default, lon_default = 20.593700, 78.962900
    lat = st.sidebar.number_input("Latitude", value=float(lat_default), format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=float(lon_default), format="%.6f")
else:
    lat = st.sidebar.number_input("Latitude", value=15.3173, format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=75.7139, format="%.6f")

st.sidebar.markdown("---")
show_raw = st.sidebar.checkbox("Show raw merged data", value=False)
show_metrics = st.sidebar.checkbox("Show model metrics", value=True)
show_charts = st.sidebar.checkbox("Show charts", value=True)
show_feature = st.sidebar.checkbox("Show feature importance", value=True)
download_data = st.sidebar.checkbox("Enable CSV download", value=True)

# Color palette
ACCENT = "#00CC96"
DANGER = "#FF4C4C"
BG = "#0E1117"
CARD_BG = "#111720"

st.markdown(f"""
<style>
/* simple dark card look */
.reportview-container .main .block-container{{padding-top:1rem}}
.stButton>button{{background-color:{ACCENT}; color: white}}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Main app
# -------------------------
st.title("🌧️📈 RainPulse — Stock + Rainfall ML Predictor")
st.write("Runs the full pipeline (indicators → rainfall merge → XGBoost + calibration) every time inputs change.")

status = st.empty()

try:
    status.info("Fetching stock data...")
    stock_raw = fetch_stock(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if stock_raw.empty:
        st.error("No stock data returned. Check ticker and date range.")
        st.stop()

    # make local copy to modify
    stock = stock_raw.copy()

    # ensure Close & Volume exist and are numeric
    if "Close" not in stock.columns or "Volume" not in stock.columns:
        missing = [c for c in ["Close", "Volume"] if c not in stock.columns]
        st.error(f"Missing required columns from yfinance: {missing}")
        st.stop()

    stock["Close"] = pd.to_numeric(stock["Close"], errors="coerce")
    stock["Volume"] = pd.to_numeric(stock["Volume"], errors="coerce")

    stock.reset_index(inplace=True)  # Date becomes column
    stock = stock.dropna(subset=["Close"])

    status.info("Calculating technical indicators...")
    close_series = stock["Close"].astype(float)

    stock["EMA8"] = EMAIndicator(close=close_series, window=8).ema_indicator().squeeze()
    stock["SMA10"] = SMAIndicator(close=close_series, window=10).sma_indicator().squeeze()
    stock["SMA20"] = SMAIndicator(close=close_series, window=20).sma_indicator().squeeze()
    stock["MACD"] = MACD(close=close_series).macd().squeeze()
    stock["RSI"] = RSIIndicator(close=close_series, window=14).rsi().squeeze()
    bb = BollingerBands(close=close_series, window=20, window_dev=2)
    stock["BB_wband"] = bb.bollinger_wband().squeeze()

    # safe volume change and returns (avoid deprecation warning by explicit fill_method=None)
    stock["Volume_Change"] = stock["Volume"].replace(0, np.nan).pct_change(fill_method=None)
    stock["Return_1"] = stock["Close"].pct_change(fill_method=None)

    status.info("Fetching rainfall data...")
    rain_df = fetch_rain_open_meteo(lat, lon, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    # normalize date columns and merge
    stock["Date"] = pd.to_datetime(stock["Date"]).dt.normalize()
    rain_df["Date"] = pd.to_datetime(rain_df["Date"]).dt.normalize()
    merged = stock.merge(rain_df[["Date", "Rain_lag1", "Rain_lag2"]], on="Date", how="left")

    # target creation
    merged["Target"] = (merged["Close"].shift(-5) > merged["Close"]).astype(int)
    merged.dropna(inplace=True)

    # show raw merged data if asked
    if show_raw:
        st.subheader("Merged data (sample)")
        st.dataframe(merged.tail(50))

    # training pipeline (identical logic)
    status.info("Training model (this may take a little while)...")
    features = ["MACD", "EMA8", "SMA10", "SMA20", "RSI", "BB_wband",
                "Volume_Change", "Return_1", "Rain_lag1", "Rain_lag2"]

    X = merged[features]
    y = merged["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # balance training data (upsample minority)
    train_df = pd.concat([X_train, y_train], axis=1)
    minority = train_df[train_df.Target == 1]
    majority = train_df[train_df.Target == 0]
    if len(minority) == 0 or len(majority) == 0:
        st.error("One class missing in training data after split — adjust date range.")
        st.stop()
    minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    train_balanced = pd.concat([majority, minority_upsampled])
    X_train_bal = train_balanced[features]
    y_train_bal = train_balanced["Target"]

    # scaler fitted on DataFrame so transform receives DataFrame with same column order
    scaler = StandardScaler()
    X_train_bal = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss"
    )
    xgb_model.fit(X_train_bal, y_train_bal)

    # calibration
    calibrated = CalibratedClassifierCV(estimator=xgb_model, cv=3, method="sigmoid")
    calibrated.fit(X_train_bal, y_train_bal)

    # evaluate
    y_pred = calibrated.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # metrics display
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision", f"{prec:.3f}")
    col3.metric("Recall", f"{rec:.3f}")
    col4.metric("F1", f"{f1:.3f}")

    if show_metrics:
        st.subheader("Confusion matrix")
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Viridis")
        fig.update_layout(width=500, height=400)
        st.plotly_chart(fig, use_container_width=False)

    # feature importances
    fi = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=False)
    if show_feature:
        st.subheader("Feature importances")
        fig = px.bar(fi.reset_index().rename(columns={"index": "feature", 0: "importance"}),
                     x="feature", y="importance", color="feature",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True)

    # charts: price + indicators
    if show_charts:
        st.subheader("Price & Indicators")
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=merged["Date"], y=merged["Close"], name="Close", line=dict(color="#00CC96")))
        price_fig.add_trace(go.Scatter(x=merged["Date"], y=merged["EMA8"], name="EMA8", line=dict(color="#FFA15A")))
        price_fig.add_trace(go.Scatter(x=merged["Date"], y=merged["SMA20"], name="SMA20", line=dict(color="#636EFA")))
        price_fig.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(price_fig, use_container_width=True)

        st.subheader("Rainfall (daily) & Rain lags")
        rain_fig = go.Figure()
        rain_fig.add_trace(go.Bar(x=rain_df["Date"], y=rain_df["Rain"], name="Rain (mm)", marker_color="#19F32F"))
        rain_fig.add_trace(go.Scatter(x=rain_df["Date"], y=rain_df["Rain"].rolling(7).mean(), name="7d avg", line=dict(color="#EF553B")))
        rain_fig.update_layout(height=350, template="plotly_dark")
        st.plotly_chart(rain_fig, use_container_width=True)

    # latest prediction
    latest_row = merged[features].iloc[-1].values.reshape(1, -1)
    latest_scaled = scaler.transform(latest_row)
    pred_proba = calibrated.predict_proba(latest_scaled)[0]
    pred_class = calibrated.predict(latest_scaled)[0]
    direction = "UP" if pred_class == 1 else "DOWN/flat"
    confidence = pred_proba[pred_class] * 100

    # show prediction badge
    st.markdown("### Latest model prediction")
    pred_col1, pred_col2 = st.columns([2, 4])
    color = ACCENT if pred_class == 1 else DANGER
    pred_col1.markdown(
        f"""
        <div style="background:{color};padding:18px;border-radius:8px">
            <h2 style="color:white;text-align:center;margin:0"> {direction} </h2>
            <p style="color:white;text-align:center;margin:0">Confidence: {confidence:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    pred_col2.metric("Latest Close", f"₹{merged['Close'].iloc[-1]:,.2f}", delta=None)

    # CSV download
    if download_data:
        csv_bytes = df_to_csv_bytes(merged)
        st.download_button("📥 Download merged CSV", data=csv_bytes, file_name=f"{ticker}_merged.csv", mime="text/csv")

    status.success("Done ✅")

except Exception as err:
    st.error(f"Fatal error: {err}")
    st.stop()
