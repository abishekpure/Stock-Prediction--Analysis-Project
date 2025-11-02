from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA  # type: ignore
import plotly.graph_objects as go  # type: ignore
from streamlit_autorefresh import st_autorefresh  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error # type: ignore
import numpy as np

load_dotenv()

API_KEY = os.getenv("API_KEY")


st.set_page_config(page_title="Real-Time Multi-Stock Dashboard", layout="wide")
st.title("📈 Multi-Stock Real-Time Dashboard with ARIMA Evaluation")

# --- Sidebar ---
symbols_text = st.sidebar.text_input("Symbols", "AAPL,MSFT,GOOGL")
refresh = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 15)
window = st.sidebar.slider("ARIMA Window Size", 30, 1700, 1000)
future_steps = st.sidebar.slider("Forecast Steps (hours)", 1, 10, 5)
use_ma = st.sidebar.checkbox("Show Moving Averages (SMA & EMA)", value=True)

symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
st_autorefresh(interval=refresh * 1000, key="stock_refresh")

# --- Fetch current price ---
def fetch_price(symbol):
    try:
        return float(requests.get(f"https://api.twelvedata.com/price?symbol={symbol}&apikey={API_KEY}").json()["price"])
    except:
        return None

# --- Fetch historical data ---
def fetch_history(symbol, size=4000):  # fetch more data to ensure enough for large windows
    try:
        resp = requests.get(
            f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&outputsize={size}&apikey={API_KEY}"
        ).json()
        df = pd.DataFrame(resp.get("values", []))
        if df.empty:
            return pd.DataFrame(columns=["datetime", "close"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["close"] = pd.to_numeric(df["close"])
        return df.sort_values("datetime")
    except:
        return pd.DataFrame(columns=["datetime", "close"])

# --- ARIMA forecast with evaluation ---
def arima_forecast_with_metrics(df, window, steps):
    if len(df) < 30 + steps:
        return None, None, None

    # Adjust window if too large
    window = min(window, len(df) - steps)

    train = df["close"].iloc[-(window + steps):-steps]
    test = df["close"].iloc[-steps:]

    try:
        model = ARIMA(train, order=(5, 1, 0)).fit()
        forecast = model.forecast(steps=steps)

        mae = mean_absolute_error(test, forecast)
        rmse = mean_squared_error(test, forecast, squared=False)

        return forecast, mae, rmse
    except:
        return None, None, None

# --- Initialize session state ---
if "stocks_data" not in st.session_state:
    st.session_state.stocks_data = {}

for symbol in symbols:
    if symbol not in st.session_state.stocks_data or st.session_state.stocks_data[symbol].empty:
        st.session_state.stocks_data[symbol] = fetch_history(symbol, size=4000)

for symbol in symbols:
    latest = fetch_price(symbol)
    if latest is not None:
        st.session_state.stocks_data[symbol] = pd.concat(
            [st.session_state.stocks_data[symbol], pd.DataFrame({"datetime": [datetime.now()], "close": [latest]})],
            ignore_index=True
        )

# --- Plot per symbol ---
for symbol in symbols:
    full_df = st.session_state.stocks_data.get(symbol, pd.DataFrame()).copy().sort_values("datetime")
    if full_df.empty:
        st.warning(f"No data for {symbol}")
        continue

    # --- Forecast & metrics ---
    forecast, mae, rmse = arima_forecast_with_metrics(full_df, window, future_steps)
    future_time = pd.date_range(full_df["datetime"].iloc[-1], periods=future_steps + 1, freq="H")[1:] if forecast is not None else []

    # --- Plotting only last 300 points for clarity ---
    plot_df = full_df.tail(300)
    fig = go.Figure()

    # Main price
    fig.add_trace(go.Scatter(
        x=plot_df["datetime"], y=plot_df["close"], mode="lines",
        name=f"{symbol} Price", line=dict(width=2, color="blue"),
        hovertemplate="%{y:.2f} USD<br>%{x|%H:%M:%S}",
    ))

    # Forecast
    if forecast is not None and len(forecast) > 0:
        fig.add_trace(go.Scatter(
            x=future_time, y=forecast, mode="lines",
            name=f"{symbol} Forecast", line=dict(dash="dash", color="red", width=2),
            hovertemplate="%{y:.2f} USD<br>%{x|%H:%M:%S}",
        ))

    # Moving averages
    if use_ma and len(plot_df) >= 20:
        plot_df["SMA"] = plot_df["close"].rolling(window=20).mean()
        plot_df["EMA"] = plot_df["close"].ewm(span=20, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=plot_df["datetime"], y=plot_df["SMA"], mode="lines",
            name="SMA(20)", line=dict(color="orange", width=2),
            hovertemplate="%{y:.2f} USD<br>%{x|%H:%M:%S}"
        ))
        fig.add_trace(go.Scatter(
            x=plot_df["datetime"], y=plot_df["EMA"], mode="lines",
            name="EMA(20)", line=dict(color="green", width=2),
            hovertemplate="%{y:.2f} USD<br>%{x|%H:%M:%S}"
        ))

    # Layout
    fig.update_layout(
        title=f"{symbol} - Real-Time Price & ARIMA Forecast",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis=dict(showgrid=True, gridcolor="#e6e6e6", tickformat="%H:%M"),
        yaxis=dict(showgrid=True, gridcolor="#e6e6e6"),
        hovermode="x unified",
        height=500,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Latest price
    if not full_df.empty:
        st.write(f"**Latest Price ({symbol}):** {full_df['close'].iloc[-1]:.2f}")
    else:
        st.write(f"**Latest Price ({symbol}):** N/A")

    # Forecast metrics
    if forecast is not None and len(forecast) > 0:
        st.write(f"**Next {future_steps}-hour Forecast:** {forecast.values}")
        st.write(f"**Evaluation Metrics:** MAE={mae:.2f}, RMSE={rmse:.2f}")
    else:
        st.write(f"**Next {future_steps}-hour Forecast:** Forecast not available")
        st.write(f"**Evaluation Metrics:** Not enough data")
