from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from yfinance import Search
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from arch import arch_model

app = FastAPI(title="Stock Predictor API (Linear + GARCH)")

# -----------------------------
# CORS (for React - CRA)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Resolve company name → symbol
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
def get_stock_data(company: str = Query(...)):

    symbol = resolve_symbol(company)
    if not symbol:
        return {"error": "Stock not found"}

    # ==================================================
    # 1️⃣ HOURLY DATA (FOR CHARTS)
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
    # 2️⃣ DAILY DATA (FOR MODELS)
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

    # -----------------------------
    # Last 4 weeks (Daily)
    # -----------------------------
    last_4_weeks = (
        daily_df[["Date", "Open", "Close"]]
        .tail(20)
        .to_dict(orient="records")
    )

    prices = daily_df["Close"]
    last_price = prices.iloc[-1]

    # ==================================================
    # 3️⃣ LINEAR REGRESSION (TREND)
    # ==================================================
    daily_df["DayIndex"] = np.arange(len(daily_df))

    X = daily_df[["DayIndex"]]
    y = prices

    lr_model = LinearRegression()
    lr_model.fit(X, y)

    future_days = pd.DataFrame({
        "DayIndex": np.arange(len(daily_df), len(daily_df) + 30)
    })

    lr_prediction = lr_model.predict(future_days)[-1].item()

    # ==================================================
    # 4️⃣ GARCH MODEL (VOLATILITY)
    # ==================================================
    log_returns = np.log(prices / prices.shift(1)).dropna() * 100

    garch = arch_model(
        log_returns,
        vol="Garch",
        p=1,
        q=1,
        mean="Zero",
        dist="normal"
    )

    garch_fit = garch.fit(disp="off")

    forecast = garch_fit.forecast(horizon=30)
    volatility = np.sqrt(forecast.variance.values[-1]).mean()

    expected_move = last_price * (volatility / 100)

    garch_lower = last_price - expected_move
    garch_upper = last_price + expected_move

    # ==================================================
    # 5️⃣ RESPONSE
    # ==================================================
    return {
        "company": company,
        "symbol": symbol,
        "last_close": round(last_price, 2),

        "linear_regression_prediction": {
            "expected_price_1_month": round(lr_prediction, 2)
        },

        "garch_prediction": {
            "volatility_30d_percent": round(volatility, 2),
            "price_range": {
                "lower": round(garch_lower, 2),
                "upper": round(garch_upper, 2)
            }
        },

        "hourly_prices": hourly_prices,
        "last_4_weeks": last_4_weeks
    }
