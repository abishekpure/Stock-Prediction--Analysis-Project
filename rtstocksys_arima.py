from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error # type: ignore
from statsmodels.tsa.arima.model import ARIMA # type: ignore
import plotly.graph_objects as go # type: ignore

load_dotenv()
API_KEY = os.getenv("API_KEY")

st.set_page_config(page_title="Real-Time Multi-Stock Dashboard", layout="wide")
st.title("📈 Multi-Stock Real-Time Dashboard with ARIMA (Statsmodels Only)")

from streamlit_autorefresh import st_autorefresh # type: ignore
st_autorefresh(interval=60 * 1000, key="realtime_refresh")  

if "symbols_text" not in st.session_state:
    st.session_state.symbols_text = " "

if "window" not in st.session_state:
    st.session_state.window = 300

if "future_steps" not in st.session_state:
    st.session_state.future_steps = 5

if "use_ma" not in st.session_state:
    st.session_state.use_ma = False

symbols_text = st.sidebar.text_input(
    "Symbols (comma separated)", 
    value=st.session_state.symbols_text
)
st.session_state.symbols_text = symbols_text

window = st.sidebar.number_input(
    "ARIMA Window Size", 30, 1700, 
    value=st.session_state.window
)
st.session_state.window = window

future_steps = st.sidebar.number_input(
    "Forecast Steps (hours)", 1, 10,
    value=st.session_state.future_steps
)
st.session_state.future_steps = future_steps

use_ma = st.sidebar.checkbox(
    "Show Moving Averages (SMA & EMA)", 
    value=st.session_state.use_ma
)
st.session_state.use_ma = use_ma


def safe_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def direction_accuracy(test, forecast, threshold=0.2):
    test_diff = np.diff(test)
    forecast_diff = np.diff(forecast)
    test_dir = np.sign(np.where(np.abs(test_diff) < threshold, 0, test_diff))
    forecast_dir = np.sign(np.where(np.abs(forecast_diff) < threshold, 0, forecast_diff))
    matches = np.sum(test_dir == forecast_dir)
    return float(matches) / len(test_dir) if len(test_dir) > 0 else 0.0

def improved_total_accuracy(actual, forecast):
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    
    dir_score = direction_accuracy(actual, forecast)
    
    mae = mean_absolute_error(actual, forecast)
    rmse = safe_rmse(actual, forecast)
    avg_price = np.mean(actual)
    
    mae_score = max(0, 1 - mae / avg_price)
    rmse_score = max(0, 1 - rmse / avg_price)
    price_score = 0.5 * mae_score + 0.5 * rmse_score

    vol = np.std(np.diff(actual))
    forecast_vol = np.std(np.diff(forecast))
    vol_ratio = forecast_vol / (vol + 1e-9)
    vol_penalty = max(0, 1 - abs(vol_ratio - 1))

    final_score = (0.4 * dir_score + 0.4 * price_score + 0.2 * vol_penalty) * 100
    return final_score

@st.cache_data
def fetch_history(symbol: str, size: int = 4000) -> pd.DataFrame:
    """Fetch historical hourly data from Twelve Data"""
    try:
        resp = requests.get(
            f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&outputsize={size}&apikey={API_KEY}"
        ).json()
        if "values" not in resp:
            return pd.DataFrame(columns=["datetime", "close"])
        df = pd.DataFrame(resp["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["close"] = pd.to_numeric(df["close"])
        return df.sort_values("datetime")
    except Exception:
        return pd.DataFrame(columns=["datetime", "close"])

def arima_forecast(prices, window, steps):
    n = len(prices)
    if n < window + steps:
        return np.array([prices[-1]]*steps)
    train = prices[-window:]
    try:
        model = ARIMA(train, order=(5,1,0)).fit()
        forecast = model.forecast(steps=steps)
        return np.asarray(forecast).astype(float)
    except Exception:
        return np.array([train[-1]]*steps)

def fetch_price(symbol):
    """Fetch latest price (not cached)"""
    try:
        resp = requests.get(f"https://api.twelvedata.com/price?symbol={symbol}&apikey={API_KEY}").json()
        return float(resp["price"]) if "price" in resp else None
    except Exception:
        return None


symbols = [s.strip().upper() for s in st.session_state.symbols_text.split(",") if s.strip()]

if "stocks_data" not in st.session_state:
    st.session_state.stocks_data = {}

for symbol in symbols:
    df = fetch_history(symbol)
    if df.empty:
        st.warning(f"No historical data available for {symbol}")
        continue

    latest = fetch_price(symbol)
    if latest is not None and (len(df) == 0 or df["datetime"].iloc[-1] < datetime.now()):
        df = pd.concat([df, pd.DataFrame({"datetime":[datetime.now()], "close":[latest]})], ignore_index=True)

    st.session_state.stocks_data[symbol] = df
    prices = df['close'].dropna().values
    forecast = arima_forecast(prices, window, future_steps)

    plot_df = df.tail(300).dropna(subset=["datetime", "close"])
    future_time = pd.date_range(plot_df["datetime"].iloc[-1], periods=future_steps+1, freq="H")[1:]
    future_time = future_time[:len(forecast)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["datetime"], y=plot_df["close"], mode="lines", name=f"{symbol} Price", line=dict(width=2,color="blue"),
        hovertemplate="%{y:.2f} USD<br>%{x|%Y-%m-%d %H:%M}"
    ))
    fig.add_trace(go.Scatter(
        x=future_time, y=forecast, mode="lines", name="ARIMA Forecast", line=dict(width=2,dash="dash",color="red"),
        hovertemplate="%{y:.2f} USD<br>%{x|%Y-%m-%d %H:%M}"
    ))

    if use_ma and len(plot_df) >= 20:
        plot_df["SMA"] = plot_df["close"].rolling(window=20).mean()
        plot_df["EMA"] = plot_df["close"].ewm(span=20, adjust=False).mean()
        fig.add_trace(go.Scatter(x=plot_df["datetime"], y=plot_df["SMA"], mode="lines", name="SMA(20)", line=dict(width=2,color="orange")))
        fig.add_trace(go.Scatter(x=plot_df["datetime"], y=plot_df["EMA"], mode="lines", name="EMA(20)", line=dict(width=2,color="green")))

    fig.update_layout(title=f"{symbol} - Real-Time Price & ARIMA Forecast", hovermode="x unified", height=500,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**Latest Price ({symbol}):** {df['close'].iloc[-1]:.2f}")
    st.write(f"**Next {future_steps}-hour Forecast:** {forecast}")

    actual_for_acc = df['close'].tail(len(forecast)).values
    if len(actual_for_acc) < len(forecast):
        actual_for_acc = np.pad(actual_for_acc, (len(forecast)-len(actual_for_acc),0), 'edge')
    total_acc = improved_total_accuracy(actual_for_acc, forecast)
    st.write(f"**Total Model Accuracy:** {total_acc:.2f}%")
