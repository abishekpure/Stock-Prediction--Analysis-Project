from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go  # type: ignore
import json
from io import StringIO

from statsmodels.tsa.arima.model import ARIMA  # type: ignore
from google.cloud import bigquery

# -------------------------
# Initialization
# -------------------------
SEED = 42
np.random.seed(SEED)

load_dotenv()
API_KEY = os.getenv("API_KEY")
FINNHUB_TOKEN = os.getenv("FINNHUB_TOKEN")
BQ_SA_FILE = "rapid-access-435612-b8-af4746783507.json"
BQ_PROJECT = os.getenv("BQ_PROJECT")
BQ_DATASET = os.getenv("BQ_DATASET")
BQ_TABLE = os.getenv("BQ_TABLE")

st.set_page_config(page_title="Real-Time Stock Forecast Dashboard (ARIMA)", layout="wide")
st.title("📈 Real-Time Stock Forecast System Dashboard")

from streamlit_autorefresh import st_autorefresh # type: ignore
st_autorefresh(interval=60 * 1000, key="realtime_refresh")

# -------------------------
# Session defaults
# -------------------------
if "symbols_text" not in st.session_state: st.session_state.symbols_text = ""
if "start_date" not in st.session_state: st.session_state.start_date = (datetime.now() - timedelta(days=30)).date()
if "future_steps" not in st.session_state: st.session_state.future_steps = 5
if "stocks_data" not in st.session_state: st.session_state.stocks_data = {}
if "submitted" not in st.session_state: st.session_state.submitted = False
if "news_fetched" not in st.session_state: st.session_state.news_fetched = False
if "metrics" not in st.session_state: st.session_state.metrics = {}

# -------------------------
# Sidebar inputs
# -------------------------
symbols_text = st.sidebar.text_input("Symbols (comma separated)", value=st.session_state.symbols_text)
st.session_state.symbols_text = symbols_text

start_date = st.sidebar.date_input("Start Date", value=st.session_state.start_date)
st.session_state.start_date = start_date

future_steps = st.sidebar.number_input("Forecast Steps (hours)", 1, 24, value=st.session_state.future_steps)
st.session_state.future_steps = future_steps

submit = st.sidebar.button("Submit")
if submit:
    st.session_state.submitted = True

# -------------------------
# Helper functions
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
    except Exception:
        return pd.DataFrame(columns=["datetime", "close"])

def fetch_price(symbol: str):
    if not API_KEY:
        return None
    try:
        res = requests.get(f"https://api.twelvedata.com/price?symbol={symbol}&apikey={API_KEY}").json()
        return float(res.get("price", None)) if res.get("price", None) is not None else None
    except Exception:
        return None

# -------------------------
# ARIMA Forecast Function
# -------------------------
@st.cache_resource(show_spinner=False)
def arima_forecast(symbol: str, prices: np.ndarray, steps: int = 5, order=(5, 1, 0)):
    if len(prices) < 10:
        return np.array([float(prices[-1])] * steps if len(prices) > 0 else [0.0] * steps)
    try:
        model = ARIMA(prices, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)

        y_true = prices[-steps:] if len(prices) >= steps else prices
        y_pred = forecast[:len(y_true)]
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        accuracy = 100 - mape

        st.session_state.metrics[symbol] = {"Total Accuracy%": accuracy}
        return np.array(forecast, dtype=np.float32)
    except Exception:
        return np.array([float(prices[-1])] * steps)

# -------------------------
# Main Execution
# -------------------------
if not API_KEY:
    st.error("API_KEY not found in .env file.")
    st.stop()
if not FINNHUB_TOKEN:
    st.error("FINNHUB_TOKEN not found in .env file.")
    st.stop()

symbols = [s.strip().upper() for s in st.session_state.symbols_text.split(",") if s.strip()]
if not symbols:
    st.info("Enter at least one ticker symbol.")
    st.stop()

if not st.session_state.submitted:
    st.info("Set a start date and press Submit to fetch and forecast data.")
    st.stop()

# -------------------------
# Fetch & store news
# -------------------------
if not st.session_state.news_fetched:
    with st.spinner("Fetching and storing company news in BigQuery..."):
        for sym in symbols:
            news_data = fetch_finnhub_news(sym)
            if news_data:
                write_news_to_bq_json(news_data)
    st.session_state.news_fetched = True

# -------------------------
# Render charts
# -------------------------
for symbol in symbols:
    st.header(symbol)
    df = fetch_historical_data(symbol)
    if df.empty:
        st.error(f"No historical data for {symbol}")
        continue

    # Filter based on start date
    df = df[df["datetime"].dt.date >= start_date]

    latest = fetch_price(symbol)
    if latest is not None and df["datetime"].iloc[-1] < pd.Timestamp.now():
        df = pd.concat([df, pd.DataFrame({"datetime":[pd.Timestamp.now()], "close":[latest]})], ignore_index=True)

    st.session_state.stocks_data[symbol] = df

    prices = df['close'].dropna().values.astype(np.float32)
    if len(prices) == 0:
        st.warning(f"No price data for {symbol}.")
        continue

    forecast = arima_forecast(symbol, prices, steps=future_steps)
    future_time = pd.date_range(df["datetime"].iloc[-1], periods=future_steps + 1, freq="H")[1:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["close"], mode="lines",
                             name=f"{symbol} Price", line=dict(width=2, color="green")))
    fig.add_trace(go.Scatter(x=future_time, y=forecast, mode="lines",
                             name="ARIMA Forecast", line=dict(width=2, dash="dash", color="red")))
    fig.update_layout(title=f"{symbol} — Price & ARIMA Forecast", hovermode="x unified", height=450)

    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**Latest Price ({symbol}):** {prices[-1]:.2f}")
    st.write(f"**Next {future_steps}-hour Forecast:** {np.round(forecast, 2)}")

    if symbol in st.session_state.metrics:
        acc = st.session_state.metrics[symbol].get("Total Accuracy%", None)
        if acc is not None:
            st.write(f"**Accuracy:** {acc:.2f}%")
