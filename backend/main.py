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

    # ---- DATA SAFETY ----
    if daily_df["Volume"].isna().sum() > 0:
     daily_df["Volume"] = daily_df["Volume"].fillna(
        daily_df["Volume"].median()
    )


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
    "expected_price": round(predicted_price, 2),

    "explanation": {
        "method": "Linear regression using time trend + trading volume",
        "inputs_used": ["DayIndex (trend)", "Volume"],
        "avg_volume_used": round(avg_volume, 2),
        "coefficients": {
            "trend_weight": float(lr.coef_[0]),
            "volume_weight": float(lr.coef_[1])
        },
        "interpretation":
            "Price is estimated using a best-fit line where both "
            "time progression and trading volume influence the future price."
    }
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
    "expected_price": round(ewma_price, 2),

    "explanation": {
        "method": "Exponentially weighted moving average adjusted by volume",
        "concept":
            "Recent prices get higher weight; volume amplifies strong moves.",
        "span": 20
    }
}

    # ==================================================
    # 5️⃣ ARIMA
    # ==================================================
    elif model == "ARIMA":
     try:
        series = daily_df["Close"]

        if len(series) < 100:
            raise Exception("Need 100+ days for ARIMA")

        arima = ARIMA(series, order=(3,1,2))
        fit = arima.fit()

        forecast = fit.forecast(steps=days)

        prediction_result = {
            "model": "ARIMA",
            "expected_price": round(float(forecast.iloc[-1]), 2),
            "explanation": {
                "method": "ARIMA with safer order (3,1,2)",
                "concept": "Auto differenced price series"
            }
        }

     except Exception as e:
        prediction_result = {
            "model": "ARIMA",
            "error": str(e),
            "expected_price": None
        }



    # ==================================================
    # 6️⃣ ARMA
    # ==================================================
    elif model == "ARMA":
     try:
        # Use RETURNS instead of prices
        series = daily_df["Close"].pct_change().dropna()

        if len(series) < 60:
            raise Exception("Not enough data for ARMA")

        arma = ARMA_MODEL(series, order=(2, 1))
        arma_fit = arma.fit()

        forecast = arma_fit.forecast(steps=days)

        # Convert return → price
        last_price = daily_df["Close"].iloc[-1]
        predicted_price = last_price * (1 + forecast.iloc[-1])

        prediction_result = {
            "model": "ARMA",
            "expected_price": round(float(predicted_price), 2),
            "explanation": {
                "method": "ARMA on RETURNS (stable version)",
                "concept": "Model built on percentage changes instead of raw price",
                "warning": "Short-term only"
            }
        }

     except Exception as e:
        prediction_result = {
            "model": "ARMA",
            "error": str(e),
            "expected_price": None
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
    "volatility_percent": round(volatility, 2),

    "explanation": {
        "method": "GARCH volatility estimation",
        "meaning":
            "Measures expected fluctuation range, not direction.",
        "interpretation":
            "Higher % = higher risk and price swings."
    }
}

    else:
        return {"error": "Invalid model selected"}
    

    # ==================================================
# STEP 3 – MARKET STYLE CONFIDENCE ENGINE
# ==================================================

    all_models = []

# -------- HELPER METRICS --------
    price_std = daily_df["Close"].pct_change().std()
    vol_std = daily_df["Volume"].pct_change().std()

    trend = daily_df["Close"].iloc[-1] - daily_df["Close"].iloc[-10]

# ==================================================
# LINEAR CONFIDENCE
# ==================================================
    if "expected_price" in prediction_result and model == "LINEAR":
      try:
        r2 = lr.score(X, y)

        score = 0

        # Model fit
        if r2 > 0.55:
            score += 2
        elif r2 > 0.35:
            score += 1

        # Trend strength
        if abs(trend) > price_std * 5:
            score += 1

        # Volume confirmation
        if vol_std < 0.8:
            score += 1

        confidence = (
            "High" if score >= 3
            else "Medium" if score == 2
            else "Low"
        )

      except:
        confidence = "Medium"

      all_models.append({
        "name": "Linear",
        "price": prediction_result.get("expected_price"),
        "confidence": confidence,
        "reason": "Trend + volume based reliability"
    })


# ==================================================
# EWMA CONFIDENCE
# ==================================================
    if model == "EWMA":

     momentum = daily_df["Close"].pct_change().tail(5).mean()

     confidence = (
        "High" if abs(momentum) > 0.01 and vol_std < 0.9
        else "Medium" if abs(momentum) > 0.005
        else "Low"
    )

     all_models.append({
        "name": "EWMA",
        "price": prediction_result.get("expected_price"),
        "confidence": confidence,
        "reason": "Recent momentum strength"
    })


# ==================================================
# ARIMA / ARMA CONFIDENCE
# ==================================================
    if model in ["ARIMA", "ARMA"]:

     error = daily_df["Close"].pct_change().std()

     confidence = (
        "High" if error < 0.015
        else "Medium" if error < 0.03
        else "Low"
    )

     all_models.append({
        "name": model,
        "price": prediction_result.get("expected_price"),
        "confidence": confidence,
        "reason": "Forecast stability"
    })


# ==================================================
# ARCH – RISK BASED
# ==================================================
    if model == "ARCH":

     vol = prediction_result.get("volatility_percent", 0)

     confidence = (
        "Safe" if vol < 2
        else "Risky" if vol < 4
        else "Very Risky"
    )

     all_models.append({
        "name": "ARCH",
        "volatility": vol,
        "confidence": confidence,
        "reason": "Market risk level"
    })


    prediction_result["comparison"] = all_models


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
