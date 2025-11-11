from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import requests
import os
from io import StringIO
from datetime import datetime, timedelta
import numpy as np
import json
import plotly.graph_objects as go # type: ignore

import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.preprocessing import MinMaxScaler

from google.cloud import bigquery

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

load_dotenv()
API_KEY = os.getenv("API_KEY")
FINNHUB_TOKEN = os.getenv("FINNHUB_TOKEN")

# BigQuery config
BQ_SA_FILE = "rapid-access-435612-b8-af4746783507.json"
BQ_PROJECT = os.getenv("BQ_PROJECT")
BQ_DATASET = os.getenv("BQ_DATASET")
BQ_TABLE = os.getenv("BQ_TABLE")

st.set_page_config(page_title="Real-Time Stock Forecast Dashboard", layout="wide")
st.title("Real-Time Stock Forecast System Dashboard")

from streamlit_autorefresh import st_autorefresh # type: ignore
st_autorefresh(interval=60 * 1000, key="realtime_refresh")

# -------------------------
# Session State Defaults
# -------------------------
if "symbols_text" not in st.session_state: st.session_state.symbols_text = " "
if "window" not in st.session_state: st.session_state.window = 60
if "future_steps" not in st.session_state: st.session_state.future_steps = 5
if "lstm_models" not in st.session_state: st.session_state.lstm_models = {}
if "scalers" not in st.session_state: st.session_state.scalers = {}
if "stocks_data" not in st.session_state: st.session_state.stocks_data = {}
if "metrics" not in st.session_state: st.session_state.metrics = {}
if "submitted" not in st.session_state: st.session_state.submitted = False
if "news_fetched" not in st.session_state: st.session_state.news_fetched = []

# -------------------------
# Sidebar Inputs
# -------------------------
symbols_text = st.sidebar.text_input("Symbols (comma separated)", value=st.session_state.symbols_text)
st.session_state.symbols_text = symbols_text

window = st.sidebar.number_input("LSTM Lookback Window (hours)", 20, 2000, value=st.session_state.window)
st.session_state.window = window

future_steps = st.sidebar.number_input("Forecast Steps (hours)", 1, 24, value=st.session_state.future_steps)
st.session_state.future_steps = future_steps

submit = st.sidebar.button("Submit")
if submit:
    st.session_state.submitted = True

symbols = [s.strip().upper() for s in st.session_state.symbols_text.split(",") if s.strip()]
if not symbols:
    st.info("Enter at least one ticker symbol in the sidebar.")
    st.stop()

# -------------------------
# API Checks
# -------------------------
if not API_KEY:
    st.error("API_KEY not found in .env file.")
    st.stop()
if not FINNHUB_TOKEN:
    st.error("FINNHUB_TOKEN not found in .env file.")
    st.stop()

# -------------------------
# Helper Functions
# -------------------------
def fetch_finnhub_news(symbol: str):
    try:
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_TOKEN}"
        res = requests.get(url)
        if res.status_code != 200:
            return []
        data = res.json()
        if not isinstance(data, list) or len(data) == 0:
            return []
        news_entries = []
        for item in data[:10]:
            news_entries.append({
                "symbol": symbol,
                "title": item.get("headline", "N/A"),
                "source": item.get("source", "Unknown"),
                "published_at_str": datetime.fromtimestamp(item.get("datetime")).strftime("%Y-%m-%d %H:%M:%S"),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "image": item.get("image", "")
            })
        return news_entries
    except:
        return []

def write_news_to_bq_json(news_entries):
    if not news_entries:
        return
    client = bigquery.Client.from_service_account_json(BQ_SA_FILE, project=BQ_PROJECT)
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    json_lines = [json.dumps(entry, default=str) for entry in news_entries]
    file_obj = StringIO("\n".join(json_lines))
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        autodetect=True
    )
    load_job = client.load_table_from_file(file_obj, table_id, job_config=job_config)
    load_job.result()

def prep_data(prices: np.ndarray, window: int):
    if len(prices) <= window:
        return np.empty((0, window, 1)), np.empty((0, 1)), None
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1,1).astype(np.float32))
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1,1), scaler

def build_lstm_model(window: int):
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=(window,1)),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='mse')
    return model

def evaluate_model(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/(y_true+1e-8)))*100
    accuracy = 100 - mape
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE%": mape, "Total Accuracy%": accuracy}

@st.cache_resource(show_spinner=False)
def train_lstm(symbol, prices, window, epochs=8, batch_size=32):
    X, y, scaler = prep_data(prices, window)
    if X.shape[0] == 0:
        return None, None, None
    model = build_lstm_model(window)
    es = EarlyStopping(monitor="loss", patience=2, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
    y_pred = model.predict(X, verbose=0)
    y_true_inv = scaler.inverse_transform(y)
    y_pred_inv = scaler.inverse_transform(y_pred)
    metrics = evaluate_model(y_true_inv.flatten(), y_pred_inv.flatten())
    return model, scaler, metrics

def lstm_forecast(symbol, prices, window, steps):
    if len(prices)==0:
        return np.array([0.0]*steps)
    window_use = min(window, len(prices)-1, 2000)
    if window_use<=0:
        return np.array([float(prices[-1])]*steps)
    model, scaler, metrics = train_lstm(symbol, prices, window_use)
    if model is None:
        return np.array([float(prices[-1])]*steps)
    st.session_state.metrics[symbol] = metrics
    last_seq = prices[-window_use:].astype(np.float32)
    forecast=[]
    for _ in range(steps):
        scaled = scaler.transform(last_seq.reshape(-1,1))
        pred = model.predict(scaled.reshape(1,window_use,1), verbose=0)
        pred_val = float(scaler.inverse_transform(pred)[0][0])
        forecast.append(pred_val)
        last_seq = np.append(last_seq[1:], pred_val)
    return np.array(forecast, dtype=np.float32)

@st.cache_resource(show_spinner=False)
def fetch_historical_data(symbol, size=4000):
    try:
        res = requests.get(
            f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&outputsize={size}&apikey={API_KEY}"
        ).json()
        if "values" not in res:
            return pd.DataFrame(columns=["datetime","close"])
        df = pd.DataFrame(res["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        return df.sort_values("datetime").reset_index(drop=True)
    except:
        return pd.DataFrame(columns=["datetime","close"])

def fetch_price(symbol):
    try:
        res = requests.get(f"https://api.twelvedata.com/price?symbol={symbol}&apikey={API_KEY}").json()
        return float(res.get("price", None)) if res.get("price", None) is not None else None
    except:
        return None

# -------------------------
# Backend Population (Run Once per Submit)
# -------------------------
if st.session_state.submitted:
    new_symbols = [s for s in symbols if s not in st.session_state.news_fetched]
    if new_symbols:
        with st.spinner("Fetching & storing company news in BigQuery..."):
            for sym in new_symbols:
                news_data = fetch_finnhub_news(sym)
                if news_data:
                    write_news_to_bq_json(news_data)
        st.session_state.news_fetched.extend(new_symbols)

# -------------------------
# Dashboard: Live Prices & LSTM Forecasts
# -------------------------
for symbol in symbols:
    st.header(symbol)
    df = fetch_historical_data(symbol)
    if df.empty:
        st.error(f"No historical data for {symbol}")
        continue

    latest = fetch_price(symbol)
    if latest is not None and df["datetime"].iloc[-1] < pd.Timestamp.now():
        df = pd.concat([df, pd.DataFrame({"datetime":[pd.Timestamp.now()],"close":[latest]})], ignore_index=True)

    st.session_state.stocks_data[symbol] = df
    max_points = min(2000, window*2)
    prices = df['close'].dropna().values.astype(np.float32)[-max_points:]

    if len(prices)<=window:
        st.warning(f"Not enough data for {symbol} to train LSTM")
        flat_forecast = np.array([float(prices[-1])]*future_steps if len(prices)>0 else [0.0]*future_steps)
        plot_df = df.tail(300)
        future_time = pd.date_range(plot_df["datetime"].iloc[-1] if not plot_df.empty else pd.Timestamp.now(),
                                    periods=future_steps+1, freq="H")[1:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df["datetime"], y=plot_df["close"], mode="lines", name=f"{symbol} Price"))
        fig.add_trace(go.Scatter(x=future_time, y=flat_forecast, mode="lines", name="Flat Forecast", line=dict(dash="dash")))
        st.plotly_chart(fig, use_container_width=True)
        continue

    forecast = lstm_forecast(symbol, prices, window, future_steps)
    plot_df = df.tail(300)
    future_time = pd.date_range(plot_df["datetime"].iloc[-1], periods=future_steps+1, freq="H")[1:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["datetime"], y=plot_df["close"], mode="lines", name=f"{symbol} Price", line=dict(width=2,color="green")))
    fig.add_trace(go.Scatter(x=future_time, y=forecast, mode="lines", name="LSTM Forecast", line=dict(width=2,dash="dash",color="red")))
    fig.update_layout(title=f"{symbol} — Price & LSTM Forecast", hovermode="x unified", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.write(f"**Latest Price ({symbol}):** {prices[-1]:.2f}")
    st.write(f"**Next {future_steps}-hour Forecast:** {np.round(forecast,2)}")
    accuracy = st.session_state.metrics.get(symbol, {}).get("Total Accuracy%", None)
    st.write(f"**Accuracy:** {accuracy:.2f}%" if accuracy is not None else "**Accuracy:** Not available")
