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
    range: str = Query("120d")
):

    symbol = resolve_symbol(company)
    if not symbol:
        return {"error": "Stock not found"}

    # ==================================================
    # 1️⃣ CHART DATA – SMART INTERVAL SWITCH
    # ==================================================
    if range in ["1m", "3m"]:
        interval = "1h"
        date_col = "Datetime"
    else:
        interval = "1d"
        date_col = "Date"

    chart_df = yf.download(
        symbol,
        period=range,
        interval=interval,
        progress=False
    )

    if chart_df.empty:
        return {"error": "No chart data found"}

    if isinstance(chart_df.columns, pd.MultiIndex):
        chart_df.columns = chart_df.columns.get_level_values(0)

    chart_df = chart_df.reset_index()

    # ✅ SAFE KEY HANDLING FOR FRONTEND
    hourly_prices = (
        chart_df[[date_col, "Close"]]
        .rename(columns={date_col: "Datetime"})
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

    last_close = round(float(daily_df["Close"].iloc[-1]), 2)

    last_4_weeks = (
        daily_df[["Date", "Open", "Close"]]
        .tail(20)
        .to_dict(orient="records")
    )

    model = model.upper()
    prediction_result = {}

    # ==================================================
    # 3️⃣ LINEAR REGRESSION (PRICE + VOLUME) – FIXED
    # ==================================================
    if model == "LINEAR":
        daily_df = daily_df.dropna(subset=["Close", "Volume"]).copy()

        daily_df["DayIndex"] = np.arange(len(daily_df))

        X = daily_df[["DayIndex", "Volume"]].astype(float)
        y = daily_df["Close"].astype(float)

        lr = LinearRegression()
        lr.fit(X, y)

        future_days = np.arange(len(daily_df), len(daily_df) + days)

        avg_volume = float(daily_df["Volume"].tail(30).mean())

        future_X = pd.DataFrame({
            "DayIndex": future_days,
            "Volume": avg_volume
        })

        predicted_price = float(lr.predict(future_X)[-1])

        prediction_result = {
            "model": "Linear Regression (Price + Volume)",
            "expected_price": round(predicted_price, 2)
        }

    # ==================================================
    # 4️⃣ EWMA
    # ==================================================
    elif model == "EWMA":
        vol_weight = daily_df["Volume"] / daily_df["Volume"].mean()
        weighted_price = daily_df["Close"] * vol_weight

        ewma_price = weighted_price.ewm(span=20, adjust=False).mean().iloc[-1]

        prediction_result = {
            "model": "EWMA (Volume-Weighted)",
            "expected_price": round(float(ewma_price), 2)
        }

    # ==================================================
    # 5️⃣ ARIMA
    # ==================================================
    elif model == "ARIMA":
        arima = ARIMA(daily_df["Close"], order=(5, 1, 0))
        forecast = arima.fit().forecast(steps=days)

        prediction_result = {
            "model": "ARIMA",
            "expected_price": round(float(forecast.iloc[-1]), 2)
        }

    # ==================================================
    # 6️⃣ ARMA
    # ==================================================
    elif model == "ARMA":
        arma = ARMA_MODEL(daily_df["Close"], order=(2, 1))
        forecast = arma.fit().forecast(steps=days)[0]

        prediction_result = {
            "model": "ARMA",
            "expected_price": round(float(forecast[-1]), 2)
        }

    # ==================================================
    # 7️⃣ ARCH
    # ==================================================
    elif model == "ARCH":
        returns = daily_df["Close"].pct_change().dropna() * 100
        garch = arch_model(returns, vol="Garch", p=1, q=1)

        volatility = np.sqrt(
            garch.fit(disp="off").forecast(horizon=days).variance.values[-1]
        ).mean()

        prediction_result = {
            "model": "ARCH (Volatility)",
            "volatility_percent": round(float(volatility), 2)
        }

    else:
        return {"error": "Invalid model selected"}

    return {
        "company": company,
        "symbol": symbol,
        "prediction_days": days,
        "last_close": last_close,
        "prediction": prediction_result,

        # ✅ FRONTEND CRITICAL
        "hourly_prices": hourly_prices,
        "last_4_weeks": last_4_weeks
    }
