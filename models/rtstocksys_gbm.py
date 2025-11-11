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

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from google.cloud import bigquery

SEED = 42
np.random.seed(SEED)

load_dotenv()
API_KEY = os.getenv("API_KEY")
FINNHUB_TOKEN = os.getenv("FINNHUB_TOKEN")

# BigQuery config
BQ_SA_FILE = "rapid-access-435612-b8-af4746783507.json"
BQ_PROJECT = os.getenv("BQ_PROJECT")
BQ_DATASET = os.getenv("BQ_DATASET")
BQ_TABLE = os.getenv("BQ_TABLE")

st.set_page_config(page_title="Real-Time Stock Forecast Dashboard (GBM)", layout="wide")
st.title("Real-Time Stock Forecast System Dashboard")

from streamlit_autorefresh import st_autorefresh # type: ignore
st_autorefresh(interval=60 * 1000, key="realtime_refresh")

# -------------------------
# Session State Defaults
# -------------------------
if "symbols_text" not in st.session_state: st.session_state.symbols_text = " "
if "window" not in st.session_state: st.session_state.window = 60
if "future_steps" not in st.session_state: st.session_state.future_steps = 5
if "gbm_models" not in st.session_state: st.session_state.gbm_models = {}
if "stocks_data" not in st.session_state: st.session_state.stocks_data = {}
if "metrics" not in st.session_state: st.session_state.metrics = {}
if "submitted" not in st.session_state: st.session_state.submitted = False
if "news_fetched" not in st.session_state: st.session_state.news_fetched = False

# -------------------------
# Sidebar Inputs
# -------------------------
symbols_text = st.sidebar.text_input("Symbols (comma separated)", value=st.session_state.symbols_text)
st.session_state.symbols_text = symbols_text

window = st.sidebar.number_input("Lookback Window (hours)", 20, 2000, value=st.session_state.window)
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
# API Key Checks
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
    except Exception:
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

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    accuracy = 100 - mape
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE%": mape, "Total Accuracy%": accuracy}

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
        gbm = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                        max_depth=3, random_state=SEED)
        gbm.fit(X, y[:, step])
        models.append(gbm)
    y_pred_first = np.array([m.predict(X) for m in models[:1]]).T[:,0]
    metrics = evaluate_model(y[:,0], y_pred_first)
    return models, metrics

def gbm_forecast(symbol: str, prices: np.ndarray, window: int, steps: int):
    if len(prices) <= window:
        return np.array([float(prices[-1])] * steps)
    window_use = min(window, len(prices))
    last_window = prices[-window_use:].reshape(1, -1)
    models, metrics = train_gbm(symbol, prices, window_use, steps)
    if models is None:
        return np.array([float(prices[-1])] * steps)
    st.session_state.metrics[symbol] = metrics
    forecast = [model.predict(last_window)[0] for model in models]
    return np.array(forecast, dtype=np.float32)

@st.cache_resource(show_spinner=False)
def fetch_historical_data(symbol: str, size: int = 4000):
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

def fetch_price(symbol: str):
    try:
        res = requests.get(f"https://api.twelvedata.com/price?symbol={symbol}&apikey={API_KEY}").json()
        return float(res.get("price", None)) if res.get("price", None) is not None else None
    except:
        return None

# -------------------------
# Backend Population (Run Once)
# -------------------------
if st.session_state.submitted and not st.session_state.news_fetched:
    with st.spinner("Fetching and storing company news in BigQuery..."):
        for sym in symbols:
            news_data = fetch_finnhub_news(sym)
            if news_data:
                write_news_to_bq_json(news_data)
    st.session_state.news_fetched = True

# -------------------------
# Main Dashboard (Autorefresh OK)
# -------------------------
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
        flat_forecast = np.array([float(prices[-1])] * future_steps if len(prices) > 0 else [0.0]*future_steps)
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
    accuracy = st.session_state.metrics.get(symbol, {}).get("Total Accuracy%", None)
    st.write(f"**Accuracy:** {accuracy:.2f}%" if accuracy else "**Accuracy:** Not available")
