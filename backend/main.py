from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from yfinance import Search
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMA as ARIMA_MODEL


from xgboost import XGBRegressor


app = FastAPI(title="Stock Predictor API")

# -----------------------------
# CORS
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://stock-predictor-1-72h2.onrender.com",
        "http://localhost:3000",
        "http://localhost:5173",
        "*"   # temporary safety
    ],
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS"
    ],
    allow_headers=[
        "*",
        "Content-Type",
        "Authorization"
    ],
    expose_headers=["*"]
)


# ==========================================
# FEATURE ENGINEERING (SAFE ADDITION)
# ==========================================
def enhance(df):
    d = df.copy()

    d["return"] = d["Close"].pct_change()

    # EMA
    d["ema20"] = d["Close"].ewm(span=20).mean()
    d["ema50"] = d["Close"].ewm(span=50).mean()

    # RSI
    delta = d["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    d["rsi"] = 100 - (100 / (1 + rs))

    # Volume z
    d["vol_z"] = (
        d["Volume"] - d["Volume"].rolling(20).mean()
    ) / d["Volume"].rolling(20).std()

    return d.dropna()


# ==========================================
# XGBOOST MODEL
# ==========================================
def xgb_predict(df, days):

    if len(df) < 40:
        return None

    f = df.copy()

    f["DayIndex"] = np.arange(len(f))

    X = f[["DayIndex", "Volume", "rsi", "vol_z"]]
    y = f["Close"]

    model = XGBRegressor(n_estimators=80, learning_rate=0.05)
    model.fit(X, y)

    last = X.iloc[[-1]].copy()
    last["DayIndex"] += days

    return float(model.predict(last)[0])

def linear_predict_for_eval(df, days):

    df = enhance(df)

    df["DayIndex"] = np.arange(len(df))

    X = df[["DayIndex","Volume","rsi","vol_z"]]
    y = df["Close"]

    lr = LinearRegression().fit(X,y)

    last = X.iloc[[-1]].copy()
    last["DayIndex"] += days

    return float(lr.predict(last)[0])


def ewma_predict_for_eval(df, days):

    df = enhance(df)

    momentum = df["return"].tail(5).mean()

    vol_weight = df["Volume"] / df["Volume"].mean()

    weighted = df["Close"] * vol_weight

    base = weighted.ewm(span=20, adjust=False).mean().iloc[-1]

    return float(base * (1 + momentum * 2))

def arima_predict_for_eval(df, days):

    try:
        series = df["Close"]

        # Same logic as main ARIMA
        if series.pct_change().std() > 0.04:
            order = (2,1,2)
        else:
            order = (3,1,2)

        fit = ARIMA(series, order=order).fit()

        forecast = fit.forecast(steps=days)

        return float(forecast.iloc[-1])

    except:
        raise Exception("ARIMA eval failed")



def arma_predict_for_eval(df, days):

    series = df["Close"].pct_change().dropna()

    if len(series) < 60:
        raise Exception("Not enough data")

    vol = series.std()

    order = (2,1) if vol < 0.025 else (1,1)

    # ARMA via ARIMA(p,0,q)
    fit = ARIMA(series, order=(order[0],0,order[1])).fit()

    forecast = fit.forecast(steps=days)

    mom = series.tail(5).mean()

    last_price = df["Close"].iloc[-1]

    return float(last_price * (1 + forecast.iloc[-1] + mom*0.5))


def xgb_predict_for_eval(df, days):
    return xgb_predict(df, days)


def arch_risk_eval(df, days):

    returns = df["Close"].pct_change().dropna()*100

    fit = arch_model(returns, vol="Garch", p=1, q=1).fit(disp="off")

    var = fit.forecast(horizon=days).variance.values[-1]

    vol = float(np.sqrt(var).mean())

    return {
        "volatility": round(vol,2),
        "risk_level":
            "SAFE" if vol < 2 else
            "RISKY" if vol < 4 else
            "DANGEROUS"
    }




# =====================================================
# WALK FORWARD VALIDATION (INFORMATION ONLY)
# =====================================================
def walk_forward_validation(df, days=10):
    try:
        df = enhance(df)

        trades = []
        equity = 100

        for i in range(200, len(df)-days, days):

            train = df.iloc[:i]
            test = df.iloc[i:i+days]

            train["DayIndex"] = np.arange(len(train))
            X = train[["DayIndex","Volume","rsi","vol_z"]]
            y = train["Close"]

            lr = LinearRegression().fit(X,y)

            last = X.iloc[[-1]].copy()
            last["DayIndex"] += days

            pred = float(lr.predict(last)[0])

            entry = float(train["Close"].iloc[-1])

            vol = train["Close"].pct_change().std()

            sl = entry * (1 - vol*2)
            tgt = entry * (1 + vol*3)

            hit = False
            result = 0

            for _, row in test.iterrows():

                high = row["High"]
                low  = row["Low"]

                if low <= sl:
                    result = -1
                    hit = True
                    break

                if high >= tgt:
                    result = 1
                    hit = True
                    break

            if not hit:
                close = float(test["Close"].iloc[-1])
                result = 1 if close > entry else -1

            pnl = (tgt-entry) if result==1 else (entry-sl)

            equity += pnl

            trades.append({
                "result": result,
                "pnl": pnl,
                "equity": equity
            })

        if len(trades) < 20:
            return None

        wins = [t for t in trades if t["result"]==1]
        losses = [t for t in trades if t["result"]==-1]

        win_rate = len(wins)/len(trades)

        total_profit = sum(t["pnl"] for t in wins)
        total_loss = abs(sum(t["pnl"] for t in losses)) + 1e-6

        profit_factor = total_profit/total_loss

        eq = [t["equity"] for t in trades]
        peak = eq[0]
        dd = 0

        for e in eq:
            peak = max(peak,e)
            dd = max(dd, peak-e)

        max_dd = dd

        return {
            "win_rate": round(win_rate,3),
            "profit_factor": round(profit_factor,2),
            "max_drawdown": round(max_dd,2),
            "approved":
                win_rate>0.55 and
                profit_factor>1.3 and
                max_dd < 12
        }

    except:
        return None

# =====================================================
# FORECAST ACCURACY ENGINE
# =====================================================
def evaluate_forecast_accuracy(df, model_fn, horizon=10):

    errors = []

    for i in range(200, len(df)-horizon):

        train = df.iloc[:i]
        test  = df.iloc[i:i+horizon]

        try:
            pred = model_fn(train, horizon)

            actual = float(test["Close"].iloc[-1])

            errors.append({
                "mae": abs(actual - pred),
                "mape": abs(actual - pred) / actual,
                "bias": pred - actual
            })
        except:
            continue

    if not errors:
        return None

    return {
        "MAE": round(np.mean([e["mae"] for e in errors]),2),
        "MAPE": round(np.mean([e["mape"] for e in errors])*100,2),
        "BIAS": round(np.mean([e["bias"] for e in errors]),2)
    }


# =====================================================
# TRADING ACCURACY ENGINE
# =====================================================
def evaluate_trading_accuracy(df, days=10):

    result = walk_forward_validation(df, days)

    if not result:
        return None

    return {
        "win_rate": result["win_rate"],
        "profit_factor": result["profit_factor"],
        "max_drawdown": result["max_drawdown"],
        "approved": result["approved"]
    }



# ==========================================
# ENSEMBLE + RISK
# ==========================================
def ensemble_engine(df, days):

    p_linear = None
    p_arima = None

    try:
        df["DayIndex"] = np.arange(len(df))
        X = df[["DayIndex","Volume","rsi","vol_z"]]
        y = df["Close"]

        lr = LinearRegression().fit(X,y)
        p_linear = float(lr.predict(X.iloc[[-1]])[0])
    except:
        pass

    try:
        ar = ARIMA(df["Close"], order=(2,1,2)).fit()
        p_arima = float(ar.forecast()[0])
    except:
        pass

    p_xgb = xgb_predict(df, days)
    

    prices = [p for p in [p_linear,p_arima,p_xgb] if p is not None]

    if not prices:
        return None, None, None, {}

    final = sum(prices)/len(prices)

    vol = df["Close"].pct_change().std()

    stop_loss = final * (1 - vol*2)
    target = final * (1 + vol*3)

    return final, stop_loss, target, {
        "xgb": p_xgb,
        "linear": p_linear,
        "arima": p_arima
    }


# ==========================================
# SAFE WRAPPER FOR ALL MODELS
# ==========================================
def get_ensemble_risk(df, days):
     try:
        f = enhance(df)
        final, sl, tgt, parts = ensemble_engine(f, days)
        return final, sl, tgt, parts
     except:
        return None, None, None, {}
     

# ==========================================
# MULTI STEP FORECAST PATH GENERATOR
# ==========================================
def generate_forecast_path(df, model, days):

    try:
        df = enhance(df)

        last_price = float(df["Close"].iloc[-1])

        # market characteristics
        vol = df["Close"].pct_change().std()
        drift = df["Close"].pct_change().tail(10).mean()

        path = []

        # ===== LINEAR =====
        if model == "LINEAR":

            df["DayIndex"] = np.arange(len(df))

            X = df[["DayIndex","Volume","rsi","vol_z"]]
            y = df["Close"]

            lr = LinearRegression().fit(X,y)

            avg_volume = float(df["Volume"].tail(20).mean())
            last_rsi = float(df["rsi"].iloc[-1])
            last_volz = float(df["vol_z"].iloc[-1])

            base_preds = []

            for i in range(1, days+1):

                future = pd.DataFrame({
                    "DayIndex": [len(df)+i],
                    "Volume": [avg_volume],
                    "rsi": [last_rsi],
                    "vol_z": [last_volz]
                })

                base_preds.append(float(lr.predict(future)[0]))

            # add realistic dynamics
            for i,p in enumerate(base_preds):

                noise = np.random.normal(0, vol*last_price*0.3)

                momentum = drift * last_price * (i/days)

                price = p + noise + momentum

                path.append({
                    "step": i+1,
                    "price": round(float(price),2)
                })


        # ===== ARIMA =====
        elif model == "ARIMA":

            series = df["Close"]

            order = (2,1,2) if series.pct_change().std()>0.04 else (3,1,2)

            fit = ARIMA(series, order=order).fit()

            fc = fit.forecast(steps=days)

            for i,v in enumerate(fc):

                noise = np.random.normal(0, vol*last_price*0.2)

                path.append({
                    "step": i+1,
                    "price": round(float(v+noise),2)
                })


        # ===== EWMA =====
        elif model == "EWMA":

            momentum = df["return"].tail(5).mean()

            vol_weight = df["Volume"]/df["Volume"].mean()

            weighted = df["Close"]*vol_weight

            base = weighted.ewm(span=20).mean().iloc[-1]

            target = float(base*(1+momentum*2))

            for i in range(1, days+1):

                prog = i/days

                curve = np.tanh(prog*2)

                noise = np.random.normal(0, vol*last_price*0.25)

                p = last_price + (target-last_price)*curve + noise

                path.append({
                    "step": i,
                    "price": round(float(p),2)
                })


        # ===== ARMA =====
        elif model == "ARMA":

            series = df["Close"].pct_change().dropna()

            order = (2,1) if series.std()<0.025 else (1,1)

            fit = ARIMA(series, order=(order[0],0,order[1])).fit()

            fc = fit.forecast(steps=days)

            mom = series.tail(5).mean()

            for i,v in enumerate(fc):

                p = last_price*(1+float(v)+mom*0.5)

                noise = np.random.normal(0, vol*last_price*0.25)

                path.append({
                    "step": i+1,
                    "price": round(float(p+noise),2)
                })


        # ===== ARCH =====
        elif model == "ARCH":

            returns = df["Close"].pct_change().dropna()*100

            fit = arch_model(returns, vol="Garch", p=1, q=1).fit(disp="off")

            var = fit.forecast(horizon=days).variance.values[-1]

            sigma = np.sqrt(var).mean()/100

            for i in range(1, days+1):

                shock = np.random.normal(0, sigma*last_price)

                p = last_price*(1+drift*i/days) + shock

                path.append({
                    "step": i,
                    "price": round(float(p),2)
                })


        return path

    except Exception as e:
        print("PATH ERROR:", e)
        return []




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


# ================= BUY / SELL SIGNAL LOGIC =================

def generate_signal(df, predicted_prices, validation=None):
    try:
        df = df.copy()

        # --- TREND ANALYSIS ---
        df['SMA_200'] = df['Close'].rolling(200).mean()
        df = df.tail(200)

        current = float(df['Close'].iloc[-1])
        sma = float(df['SMA_200'].iloc[-1])

        # Linear slope
        y = df['Close'].values.reshape(-1,1)
        x = np.arange(len(y)).reshape(-1,1)

        lr = LinearRegression()
        lr.fit(x, y)
        slope = lr.coef_[0][0]

        trend = "BULLISH" if slope > 0 and current > sma else "BEARISH"

        # --- PREDICTION ANALYSIS ---
        pred = float(predicted_prices[-1])
        prediction = "BULLISH" if pred > current else "BEARISH"

        # --- VOLUME ANALYSIS ---
        long_vol = df['Volume'].mean()
        recent_vol = df['Volume'].tail(20).mean()

        if recent_vol > 1.05 * long_vol:
            volume = "SUPPORTIVE"
        elif recent_vol < 0.9 * long_vol:
            volume = "WEAK"
        else:
            volume = "NORMAL"

        # --- FINAL DECISION ---
        if trend == "BULLISH" and prediction == "BULLISH" and volume != "WEAK":
            signal = "BUY"
        elif trend == "BEARISH" and prediction == "BEARISH":
            signal = "SELL"
        else:
            signal = "HOLD"

        confidence = round(
            min(abs(pred - current) / current, 0.25), 3
        )

        # ----- VALIDATION OVERLAY -----
        if validation and not validation.get("approved", True):
            signal = "HOLD"

        return {
            "signal": signal,
            "confidence": confidence,
            "reasons": {
                "trend": trend,
                "prediction": prediction,
                "volume": volume,
                "validation":
                    "APPROVED" if not validation
                    else ("APPROVED" if validation.get("approved") else "NOT_PROVEN")
            }
        }

    except Exception as e:
        return {
            "signal": "HOLD",
            "confidence": 0,
            "reasons": {"error": str(e)}
        }

# ============================================================


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

    # ================= SAFE DEFAULT =================
    signal_info = {
        "signal": "HOLD",
        "confidence": 0,
        "reasons": {}
    }

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

     df = enhance(daily_df)

     df["DayIndex"] = np.arange(len(df))

     X = df[["DayIndex", "Volume", "rsi", "vol_z"]].astype(float)
     y = df["Close"].astype(float)

     lr = LinearRegression()
     lr.fit(X, y)

     future_days = np.arange(len(df), len(df) + days)

     avg_volume = float(df["Volume"].tail(30).mean())
     last_rsi = float(df["rsi"].iloc[-1])
     last_volz = float(df["vol_z"].iloc[-1])

     future_X = pd.DataFrame({
        "DayIndex": future_days,
        "Volume": avg_volume,
        "rsi": last_rsi,
        "vol_z": last_volz
     })

     predicted_price = float(lr.predict(future_X)[-1])

     # ---- Walk Forward Validation ----
     validation_info = walk_forward_validation(daily_df, days)


     signal_info = generate_signal(
    daily_df,
    [predicted_price],
    validation_info
)



          # ---- ENSEMBLE + STOP LOSS ----
     ens, sl, tgt, parts = get_ensemble_risk(daily_df, days)


     prediction_result = {
        "model": "Linear Regression (Price + Volume)",
        "expected_price": round(predicted_price, 2),

        "ensemble_price": round(ens,2) if ens else None,
        "stop_loss": round(sl,2) if sl else None,
        "target": round(tgt,2) if tgt else None,

        "explanation": {
            "method": "Trend + RSI + Volume strength",
            "inputs_used": ["Trend","Volume","RSI","Volume Z"],
            "avg_volume_used": round(avg_volume, 2),
            "coefficients": {
                "trend_weight": float(lr.coef_[0]),
                "volume_weight": float(lr.coef_[1])
            },
            "interpretation": "Adds momentum (RSI) to trend model"
        }
     }




    # ==================================================
    # 4️⃣ EWMA
    # ==================================================
    elif model == "EWMA":

     df = enhance(daily_df)

     momentum = df["return"].tail(5).mean()

     vol_weight = df["Volume"] / df["Volume"].mean()

     weighted = df["Close"] * vol_weight

     base = weighted.ewm(span=20, adjust=False).mean().iloc[-1]

    # Momentum adjustment
     ewma_price = base * (1 + momentum * 2)

     # ---- Walk Forward Validation ----
     validation_info = walk_forward_validation(daily_df, days)


     # ✅ SIGNAL
     signal_info = generate_signal(daily_df, [ewma_price],validation_info)


     ens, sl, tgt, parts = get_ensemble_risk(daily_df, days)


     prediction_result = {
        "model": "EWMA (Volume-Weighted)",
        "expected_price": round(float(ewma_price), 2),

        "ensemble_price": round(ens,2) if ens else None,
        "stop_loss": round(sl,2) if sl else None,
        "target": round(tgt,2) if tgt else None,

        "explanation": {
            "method": "EWMA + momentum overlay",
            "concept": "Recent returns tilt EWMA direction",
            "span": 20
        }
     }



    # ==================================================
    # 5️⃣ ARIMA
    # ==================================================
    elif model == "ARIMA":
     try:
      series = daily_df["Close"]

     # Auto difference check
      if series.pct_change().std() > 0.04:
        order = (2,1,2)
      else:
        order = (3,1,2)

      arima = ARIMA(series, order=order)
      fit = arima.fit()
       
      forecast = fit.forecast(steps=days)
      
      pred = float(forecast.iloc[-1])

      # ---- Walk Forward Validation ----
      validation_info = walk_forward_validation(daily_df, days)

      signal_info = generate_signal(daily_df, [pred],validation_info)

      ens, sl, tgt, parts = get_ensemble_risk(daily_df, days)


      prediction_result = {
        "model": "ARIMA",
        "expected_price": round(float(forecast.iloc[-1]), 2),

        "ensemble_price": round(ens,2) if ens else None,
        "stop_loss": round(sl,2) if sl else None,
        "target": round(tgt,2) if tgt else None,

        "explanation": {
            "method": f"Adaptive ARIMA {order}",
            "concept": "Order adapts to volatility"
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
    # ---- Use RETURNS ----
      series = daily_df["Close"].pct_change().dropna()

      if len(series) < 60:
        raise Exception("Not enough data for ARMA")

    # ---- Adaptive order ----
      vol = series.std()

      order = (2, 1) if vol < 0.025 else (1, 1)

      arma = ARIMA_MODEL(series, order=(order[0], 0, order[1]))
      arma_fit = arma.fit()

      forecast = arma_fit.forecast(steps=days)

    # ---- Momentum correction ----
      mom = series.tail(5).mean()

      last_price = daily_df["Close"].iloc[-1]

      predicted_price = last_price * (1 + forecast.iloc[-1] + mom * 0.5)

      # ---- Walk Forward Validation ----
      validation_info = walk_forward_validation(daily_df, days)


      # ✅ SIGNAL
      signal_info = generate_signal(
    daily_df,
    [predicted_price],
    validation_info
)


      ens, sl, tgt, parts = get_ensemble_risk(daily_df, days)


      prediction_result = {
        "model": "ARMA",
        "expected_price": round(float(predicted_price), 2),

        "ensemble_price": round(ens,2) if ens else None,
        "stop_loss": round(sl,2) if sl else None,
        "target": round(tgt,2) if tgt else None,

        "explanation": {
            "method": f"ARMA{order} on RETURNS + momentum",
            "concept": "Return forecast adjusted by recent drift",
            "warning": "Best for 1–10 days horizon"
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

     fit = garch.fit(disp="off")

     var = fit.forecast(horizon=days).variance.values[-1]

     vol = float(np.sqrt(var).mean())

    # ---- Convert to price range ----
     last = daily_df["Close"].iloc[-1]

     upper = last * (1 + vol/100)
     lower = last * (1 - vol/100)

     ens, sl, tgt, parts = get_ensemble_risk(daily_df, days)

     # ✅ ADD THIS — ARCH HAS NO DIRECTION
     signal_info = {
        "signal": "HOLD",
        "confidence": 0,
        "reasons": {
            "trend": "N/A",
            "prediction": "N/A",
            "volume": "N/A"
        }
      }


     prediction_result = {
        "model": "ARCH (Volatility)",

        "volatility_percent": round(vol, 2),

        "ensemble_price": round(ens,2) if ens else None,
        "stop_loss": round(sl,2) if sl else None,
        "target": round(tgt,2) if tgt else None,

        "explanation": {
            "method": "GARCH with price band",
            "meaning": f"Expected range ₹{round(lower,2)} – ₹{round(upper,2)}",
            "interpretation": "Risk not direction"
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

    # ======= NEW ACCURACY MODULE (ALL MODELS) =======

    trading_acc = evaluate_trading_accuracy(daily_df, days)

# -------- MODEL SPECIFIC FORECAST --------

    forecast_acc = None

    # ---- SPEED PROTECTION ----
    if len(daily_df) > 600:
     acc_df = daily_df.tail(600)
    else:
     acc_df = daily_df


    if model == "LINEAR":
     forecast_acc = evaluate_forecast_accuracy(
        acc_df,
        linear_predict_for_eval,
        days
    )

    elif model == "EWMA":
     forecast_acc = evaluate_forecast_accuracy(
        acc_df,
        ewma_predict_for_eval,
        days
    )

    elif model == "ARIMA":
     forecast_acc = evaluate_forecast_accuracy(
        acc_df,
        arima_predict_for_eval,
        days
    )

    elif model == "ARMA":
     forecast_acc = evaluate_forecast_accuracy(
        acc_df,
        arma_predict_for_eval,
        days
    )

    elif model == "ARCH":
     forecast_acc = arch_risk_eval(acc_df, days)


# ===== SMART RELIABILITY (FIXED) =====

    reliability = 0

# ---- CASE 1: BOTH AVAILABLE ----
    if trading_acc and forecast_acc and model != "ARCH":

     reliability = round(
        0.6 * (trading_acc["win_rate"] * 100) +
        0.4 * (100 - forecast_acc["MAPE"]),
    2)


# ---- CASE 2: ONLY FORECAST AVAILABLE ----
    elif forecast_acc and model != "ARCH":

     reliability = round(
        max(0, 100 - forecast_acc["MAPE"]),
    2)


# ---- CASE 3: ONLY TRADING AVAILABLE ----
    elif trading_acc:

     reliability = round(
        trading_acc["win_rate"] * 100,
    2)


# ---- CASE 4: ARCH MODEL ----
    if model == "ARCH" and forecast_acc:

     risk = forecast_acc.get("volatility", 5)

     safety = (
        100 if risk < 2 else
        60 if risk < 4 else
        30
    )

     if trading_acc:
        reliability = round(
            0.5 * (trading_acc["win_rate"] * 100) +
            0.5 * safety,
        2)
     else:
        reliability = safety

     
    forecast_path = generate_forecast_path(
    daily_df,
    model.upper(),
    days
)



    return {
        "company": company,
        "symbol": symbol,
        "prediction_days": days,
        "last_close": last_close,
        "prediction": prediction_result,
        "signal": signal_info,
        "forecast_path": forecast_path,
        "accuracy": {
            "trading": trading_acc,
            "forecast": forecast_acc,
            "reliability_score": reliability
        },


        # ✅ FRONTEND CRITICAL
        "hourly_prices": hourly_prices,
        "last_4_weeks": last_4_weeks
    }


# ==========================================================
#  NEW ENDPOINT – STOCKS BY PRICE (NSE + BSE RELIABLE)
#  ➜ PURE ADDITION – DOES NOT TOUCH EXISTING LOGIC
# ==========================================================

# ==========================================================
#  NEW ENDPOINT – STOCKS BY PRICE USING YOUR NSE CSV + YFINANCE
# ==========================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NSE_CSV_PATH = os.path.join(BASE_DIR, "EQUITY_L.csv")

@app.get("/stocks-by-price")
def stocks_by_price(max: float = Query(100, ge=1)):

    try:
        # ==========================================
        # 1️⃣ LOAD SYMBOLS FROM YOUR CSV
        # ==========================================
        if not os.path.exists(NSE_CSV_PATH):
            return {
                "stocks": [],
                "error": f"CSV NOT FOUND at {NSE_CSV_PATH}"
            }

        df = pd.read_csv(NSE_CSV_PATH)

        if "SYMBOL" not in df.columns:
            return {
                "stocks": [],
                "error": "SYMBOL column not found in CSV"
            }

        symbols = df["SYMBOL"].dropna().unique().tolist()

        # Convert to yfinance format
        tickers = [s + ".NS" for s in symbols]

        result = []

        # ==========================================
        # 2️⃣ ULTRA FAST FETCH – ONLY 1 DAY DATA
        # ==========================================
        CHUNK_SIZE = 100     # Bigger batch = faster
        DAYS = "2d"          # Only last 2 days → super quick

        for i in range(0, len(tickers), CHUNK_SIZE):

            chunk = tickers[i:i + CHUNK_SIZE]

            try:
                data = yf.download(
                    tickers=chunk,
                    period=DAYS,
                    interval="1d",
                    group_by="ticker",
                    progress=False,
                    threads=True
                )

                for s in chunk:
                    try:
                        dfp = data[s]

                        if dfp.empty:
                            continue

                        # Flatten multi index
                        if isinstance(dfp.columns, pd.MultiIndex):
                            dfp.columns = dfp.columns.get_level_values(0)

                        # 👉 LATEST CLOSE
                        price = float(dfp["Close"].iloc[-1])

                        # ==================================
                        # 3️⃣ PRICE SEGREGATION LOGIC
                        # ==================================
                        if price <= max:

                            result.append({
                                "symbol": s.replace(".NS", ""),
                                "price": round(price, 2),
                                "date": str(dfp.index[-1].date())
                            })

                    except:
                        continue

            except:
                continue

        # ==========================================
        # 4️⃣ SORT & RETURN TOP 50
        # ==========================================
        result = sorted(result, key=lambda x: x["price"])[:50]

        return {
            "stocks": result,
            "count": len(result),
            "filter": f"Stocks under ₹{max}",
            "latest_date": result[0]["date"] if result else None
        }

    except Exception as e:
        return {
            "stocks": [],
            "error": str(e)
        }