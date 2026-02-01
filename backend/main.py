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
    model: str = Query("Linear")  # NEW
):

    symbol = resolve_symbol(company)
    if not symbol:
        return {"error": "Stock not found"}

    # ==================================================
    # 1️⃣ HOURLY DATA
    # ==================================================
    hourly_df = yf.download(
        symbol,
        period="30d",
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
    # 2️⃣ DAILY DATA
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

    last_4_weeks = (
        daily_df[["Date", "Open", "Close"]]
        .tail(20)
        .to_dict(orient="records")
    )

    last_close = round(daily_df["Close"].iloc[-1].item(), 2)

    # ==================================================
    # 3️⃣ MODEL-SPECIFIC PREDICTION
    # ==================================================
    prediction_result = {}

    model = model.upper()

    # ---------- LINEAR REGRESSION ----------
    if model == "LINEAR":
        daily_df["DayIndex"] = np.arange(len(daily_df))

        lr = LinearRegression()
        lr.fit(daily_df[["DayIndex"]], daily_df["Close"])

        future_index = pd.DataFrame({
            "DayIndex": np.arange(len(daily_df), len(daily_df) + days)
        })

        predicted_price = lr.predict(future_index)[-1].item()

        prediction_result = {
            "model": "Linear Regression",
            "expected_price": round(predicted_price, 2)
        }

    # ---------- ARCH / GARCH ----------
    elif model == "ARCH":
        returns = daily_df["Close"].pct_change().dropna() * 100

        garch = arch_model(returns, vol="Garch", p=1, q=1)
        garch_fit = garch.fit(disp="off")

        forecast = garch_fit.forecast(horizon=days)
        volatility = np.sqrt(forecast.variance.values[-1]).mean()

        prediction_result = {
            "model": "ARCH (GARCH)",
            "volatility_percent": round(volatility, 2)
        }

    # ---------- ARMA ----------
    elif model == "ARMA":
        series = daily_df["Close"]

        arma = ARMA_MODEL(series, order=(2, 1))
        arma_fit = arma.fit()

        forecast = arma_fit.forecast(steps=days)[0]
        predicted_price = forecast[-1]

        prediction_result = {
            "model": "ARMA",
            "expected_price": round(predicted_price, 2)
        }

    # ---------- ARIMA ----------
    elif model == "ARIMA":
        series = daily_df["Close"]

        arima = ARIMA(series, order=(5, 1, 0))
        arima_fit = arima.fit()

        forecast = arima_fit.forecast(steps=days)
        predicted_price = forecast.iloc[-1]

        prediction_result = {
            "model": "ARIMA",
            "expected_price": round(predicted_price, 2)
        }

    else:
        return {"error": "Invalid model selected"}

    # ==================================================
    # RESPONSE
    # ==================================================
    return {
        "company": company,
        "symbol": symbol,
        "prediction_days": days,
        "last_close": last_close,

        "prediction": prediction_result,

        "hourly_prices": hourly_prices,
        "last_4_weeks": last_4_weeks
    }
