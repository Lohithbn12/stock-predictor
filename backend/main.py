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


@app.get("/stock")
def get_stock_data(
    company: str = Query(...),
    days: int = Query(30, ge=1, le=365),
    model: str = Query("Linear")
):

    symbol = resolve_symbol(company)
    if not symbol:
        return {"error": "Stock not found"}

    # ===============================
    # DAILY DATA (PRICE + VOLUME)
    # ===============================
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

    model = model.upper()
    prediction_result = {}

    # ==================================================
    # 1️⃣ LINEAR REGRESSION (PRICE + VOLUME)
    # ==================================================
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
            "Volume": avg_volume  # volume influences future price
        })

        predicted_price = lr.predict(future_X)[-1]

        prediction_result = {
            "model": "Linear Regression (Price + Volume)",
            "expected_price": round(predicted_price, 2)
        }

    # ==================================================
    # 2️⃣ EWMA (VOLUME-WEIGHTED)
    # ==================================================
    elif model == "EWMA":
        price = daily_df["Close"]
        volume = daily_df["Volume"]

        vol_weight = volume / volume.mean()
        weighted_price = price * vol_weight

        ewma_price = weighted_price.ewm(span=20, adjust=False).mean()
        predicted_price = ewma_price.iloc[-1]

        prediction_result = {
            "model": "EWMA (Volume-Weighted)",
            "expected_price": round(predicted_price, 2)
        }

    # ==================================================
    # 3️⃣ ARIMA (PRICE ONLY – CORRECT)
    # ==================================================
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

    # ==================================================
    # 4️⃣ ARMA (PRICE ONLY)
    # ==================================================
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

    # ==================================================
    # 5️⃣ ARCH / GARCH (VOLATILITY MODEL)
    # ==================================================
    elif model == "ARCH":
        returns = daily_df["Close"].pct_change().dropna() * 100

        garch = arch_model(returns, vol="Garch", p=1, q=1)
        garch_fit = garch.fit(disp="off")

        forecast = garch_fit.forecast(horizon=days)
        volatility = np.sqrt(forecast.variance.values[-1]).mean()

        prediction_result = {
            "model": "ARCH (Volatility)",
            "volatility_percent": round(volatility, 2)
        }

    else:
        return {"error": "Invalid model selected"}

    # ===============================
    # RESPONSE
    # ===============================
    return {
        "company": company,
        "symbol": symbol,
        "prediction_days": days,
        "last_close": last_close,
        "prediction": prediction_result
    }
