from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime
import numpy as np
import plotly.graph_objects as go # type: ignore

from sklearn.ensemble import GradientBoostingRegressor # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type: ignore

SEED = 42
np.random.seed(SEED)

load_dotenv()
API_KEY = os.getenv("API_KEY")

st.set_page_config(page_title="Real-Time Stock Forecast Dashboard (GBM)", layout="wide")
st.title("📈 Real-Time Stock Forecast System — Gradient Boosting Version")

from streamlit_autorefresh import st_autorefresh # type: ignore
st_autorefresh(interval=60 * 1000, key="realtime_refresh")

# Initialize session state
if "symbols_text" not in st.session_state: st.session_state.symbols_text = " "
if "window" not in st.session_state: st.session_state.window = 60
if "future_steps" not in st.session_state: st.session_state.future_steps = 5
if "gbm_models" not in st.session_state: st.session_state.gbm_models = {}
if "stocks_data" not in st.session_state: st.session_state.stocks_data = {}
if "metrics" not in st.session_state: st.session_state.metrics = {}

# Sidebar inputs
symbols_text = st.sidebar.text_input("Symbols (comma separated)", value=st.session_state.symbols_text)
st.session_state.symbols_text = symbols_text

window = st.sidebar.number_input("Lookback Window (hours)", 20, 2000, value=st.session_state.window)
st.session_state.window = window

future_steps = st.sidebar.number_input("Forecast Steps (hours)", 1, 24, value=st.session_state.future_steps)
st.session_state.future_steps = future_steps

# -------------------------
# Metrics
# -------------------------
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    accuracy = 100 - mape
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE%": mape, "Total Accuracy%": accuracy}

# -------------------------
# Gradient Boosting Training
# -------------------------
@st.cache_resource(show_spinner=False)
def train_gbm(symbol: str, prices: np.ndarray, window: int, steps: int):
    if len(prices) <= window + steps:
        return None, None

    X, y = [], []
    for i in range(window, len(prices) - steps + 1):
        X.append(prices[i-window:i])
        y.append(prices[i:i+steps])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    models = []
    for step in range(steps):
        gbm = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED)
        gbm.fit(X, y[:, step])
        models.append(gbm)

    # Evaluate first step
    y_pred_first = np.array([m.predict(X) for m in models[:1]]).T[:,0]
    metrics = evaluate_model(y[:,0], y_pred_first)

    return models, metrics

# -------------------------
# Multi-step forecast
# -------------------------
def gbm_forecast(symbol: str, prices: np.ndarray, window: int, steps: int):
    if len(prices) <= window:
        return np.array([float(prices[-1])] * steps)

    window_use = min(window, len(prices)-steps)
    if window_use <= 0:
        return np.array([float(prices[-1])] * steps)

    models, metrics = train_gbm(symbol, prices, window_use, steps)
    if models is None:
        return np.array([float(prices[-1])] * steps)

    st.session_state.metrics[symbol] = metrics

    last_seq = prices[-window_use:].reshape(1, -1)
    forecast = []
    for step in range(steps):
        pred = models[step].predict(last_seq)
        forecast.append(pred[0])
        last_seq = np.append(last_seq[:,1:], pred).reshape(1, -1)

    return np.array(forecast, dtype=np.float32)

# -------------------------
# Fetch Historical Data
# -------------------------
@st.cache_resource(show_spinner=False)
def fetch_historical_data(symbol: str, size: int = 4000):
    if not API_KEY:
        return pd.DataFrame(columns=["datetime", "close"])
    try:
        res = requests.get(
            f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&outputsize={size}&apikey={API_KEY}"
        ).json()
        if "values" not in res:
            return pd.DataFrame(columns=["datetime", "close"])
        df = pd.DataFrame(res["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        return df.sort_values("datetime").reset_index(drop=True)
    except:
        return pd.DataFrame(columns=["datetime", "close"])

# -------------------------
# Fetch latest price
# -------------------------
def fetch_price(symbol: str):
    if not API_KEY:
        return None
    try:
        res = requests.get(f"https://api.twelvedata.com/price?symbol={symbol}&apikey={API_KEY}").json()
        return float(res.get("price", None)) if res.get("price", None) is not None else None
    except:
        return None

# -------------------------
# Main Execution
# -------------------------
if not API_KEY:
    st.error("API_KEY not found in .env file.")
    st.stop()

symbols = [s.strip().upper() for s in st.session_state.symbols_text.split(",") if s.strip()]
if not symbols:
    st.info("Enter at least one ticker symbol in the sidebar.")
    st.stop()

for symbol in symbols:
    st.header(symbol)

    df = fetch_historical_data(symbol)
    if df.empty:
        st.error(f"No historical data for {symbol}.")
        continue

    latest = fetch_price(symbol)
    if latest is not None and df["datetime"].iloc[-1] < pd.Timestamp.now():
        df = pd.concat([df, pd.DataFrame({"datetime":[pd.Timestamp.now()], "close":[latest]})], ignore_index=True)

    st.session_state.stocks_data[symbol] = df
    max_points = min(2000, window * 2)
    prices = df['close'].dropna().values.astype(np.float32)[-max_points:]

    if len(prices) <= window:
        st.warning(f"Not enough data for {symbol} to train GBM. Showing last prices only.")
        flat_forecast = np.array([float(prices[-1])] * future_steps) if len(prices) > 0 else np.array([0.0]*future_steps)
        plot_df = df.tail(300)
        future_time = pd.date_range(plot_df["datetime"].iloc[-1] if not plot_df.empty else pd.Timestamp.now(),
                                    periods=future_steps+1, freq="H")[1:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df["datetime"], y=plot_df["close"], mode="lines", name=f"{symbol} Price"))
        fig.add_trace(go.Scatter(x=future_time, y=flat_forecast, mode="lines", name="Flat Forecast", line=dict(dash="dash")))
        st.plotly_chart(fig, use_container_width=True)
        continue

    forecast = gbm_forecast(symbol, prices, window, future_steps)

    plot_df = df.tail(300)
    future_time = pd.date_range(plot_df["datetime"].iloc[-1], periods=future_steps+1, freq="H")[1:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["datetime"], y=plot_df["close"], mode="lines",
                             name=f"{symbol} Price", line=dict(width=2, color="green")))
    fig.add_trace(go.Scatter(x=future_time, y=forecast, mode="lines+markers",
                             name="GBM Forecast", line=dict(width=2, dash="dash", color="red")))
    fig.update_layout(title=f"{symbol} — Price & GBM Forecast", hovermode="x unified", height=450)

    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**Latest Price ({symbol}):** {prices[-1]:.2f}")
    st.write(f"**Next {future_steps}-hour Forecast:** {np.round(forecast, 2)}")

    if symbol in st.session_state.metrics:
        accuracy = st.session_state.metrics[symbol].get("Total Accuracy%", None)
        st.write(f"**Accuracy:** {accuracy:.2f}%" if accuracy else "**Accuracy:** Not available")
    else:
        st.write("**Accuracy:** Not available")
