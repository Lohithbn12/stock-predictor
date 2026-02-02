from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from yfinance import Search
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMA as ARMA_MODEL

app = FastAPI(title="Stock Predictor API")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://stock-predictor-1-72h2.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Resolve company → symbol
# -----------------------------
def resolve_symbol(company_name: str):
    search = Search(company_name)
    results = search.quotes

    if not results:
        return None

    for item in results:
        if item.get("exchange") in ["NSI", "BSE"]:
            return item.get("symbol")

    return results[0].get("symbol")


# -----------------------------
# API Endpoint
# -----------------------------
@app.get("/stock")
def get_stock_data(
    company: str = Query(...),
    days: int = Query(30, ge=1, le=365),
    model: str = Query("Linear"),
    range: str = Query("120d")   # ✅ NEW (SAFE ADDITION)
):

    symbol = resolve_symbol(company)
    if not symbol:
        return {"error": "Stock not found"}

    # ==================================================
    # 1️⃣ HOURLY DATA (FOR CHART) – RANGE CONTROLLED
    # ==================================================
    hourly_df = yf.download(
        symbol,
        period=range,        # ✅ ONLY CHANGE HERE
        interval="1h",
        progress=False
    )

    if hourly_df.empty:
        return {"error": "No hourly data found"}

    if isinstance(hourly_df.columns, pd.MultiIndex):
        hourly_df.columns = hourly_df.columns.get_level_values(0)

    hourly_df = hourly_df.reset_index()

    hourly_prices = (
        hourly_df[["Datetime", "Close"]]
        .tail(200)
        .to_dict(orient="records")
    )

    # ==================================================
    # 2️⃣ DAILY DATA (PRICE + VOLUME)
    # ==================================================
    daily_df = yf.download(
        symbol,
        period="3y",
        interval="1d",
        progress=False
    )

    if daily_df.empty:
        return {"error": "No daily data found"}

    if isinstance(daily_df.columns, pd.MultiIndex):
        daily_df.columns = daily_df.columns.get_level_values(0)

    daily_df = daily_df.reset_index()

    last_close = round(daily_df["Close"].iloc[-1], 2)

    last_4_weeks = (
        daily_df[["Date", "Open", "Close"]]
        .tail(20)
        .to_dict(orient="records")
    )

    model = model.upper()
    prediction_result = {}

    # ================= LINEAR REGRESSION =================
    if model == "LINEAR":
        daily_df["DayIndex"] = np.arange(len(daily_df))
        X = daily_df[["DayIndex", "Volume"]]
        y = daily_df["Close"]

        lr = LinearRegression()
        lr.fit(X, y)

        future_days = np.arange(len(daily_df), len(daily_df) + days)
        avg_volume = daily_df["Volume"].tail(30).mean()

        future_X = pd.DataFrame({
            "DayIndex": future_days,
            "Volume": avg_volume
        })

        predicted_price = lr.predict(future_X)[-1]

        prediction_result = {
            "model": "Linear Regression (Price + Volume)",
            "expected_price": round(predicted_price, 2)
        }

    # ================= EWMA =================
    elif model == "EWMA":
        vol_weight = daily_df["Volume"] / daily_df["Volume"].mean()
        weighted_price = daily_df["Close"] * vol_weight

        ewma_price = weighted_price.ewm(span=20, adjust=False).mean().iloc[-1]

        prediction_result = {
            "model": "EWMA (Volume-Weighted)",
            "expected_price": round(ewma_price, 2)
        }

    # ================= ARIMA =================
    elif model == "ARIMA":
        arima = ARIMA(daily_df["Close"], order=(5, 1, 0))
        forecast = arima.fit().forecast(steps=days)

        prediction_result = {
            "model": "ARIMA",
            "expected_price": round(forecast.iloc[-1], 2)
        }

    # ================= ARMA =================
    elif model == "ARMA":
        arma = ARMA_MODEL(daily_df["Close"], order=(2, 1))
        forecast = arma.fit().forecast(steps=days)[0]

        prediction_result = {
            "model": "ARMA",
            "expected_price": round(forecast[-1], 2)
        }

    # ================= ARCH =================
    elif model == "ARCH":
        returns = daily_df["Close"].pct_change().dropna() * 100
        garch = arch_model(returns, vol="Garch", p=1, q=1)

        volatility = np.sqrt(
            garch.fit(disp="off").forecast(horizon=days).variance.values[-1]
        ).mean()

        prediction_result = {
            "model": "ARCH (Volatility)",
            "volatility_percent": round(volatility, 2)
        }

    else:
        return {"error": "Invalid model selected"}

    return {
        "company": company,
        "symbol": symbol,
        "prediction_days": days,
        "last_close": last_close,
        "prediction": prediction_result,
        "hourly_prices": hourly_prices,
        "last_4_weeks": last_4_weeks
    }
