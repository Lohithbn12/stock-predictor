import { useState, useEffect } from "react";
import HourlyPriceChart from "./HourlyPriceChart";
import Last4WeeksTable from "./Last4WeeksTable";
import StockListPage from "./StockListPage";
import PredictionTablePage from "./PredictionTablePage";

import "./App.css";

const API_URL = "https://stock-predictor-0zst.onrender.com";


function StockPage() {
  const [company, setCompany] = useState("");
  const [days, setDays] = useState(30);
  const [model, setModel] = useState("Linear");
  const [chartRange, setChartRange] = useState("120d"); // ✅ NEW
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [viewMode, setViewMode] = useState("full");
  const [trendMode, setTrendMode] = useState("up");
  const [showPredictionTable, setShowPredictionTable] = useState(false);


  // ===== NEW LOADING FOR LIST PAGE =====
  const [listLoading, setListLoading] = useState(false);
  const [showPredictionPanel, setShowPredictionPanel] = useState(false);


  // ✅ NEW STATE (ONLY ADDITION)
  const [showExplain, setShowExplain] = useState(false);



  // =================== ONLY ADDITIONS ===================
  const [showOverlay, setShowOverlay] = useState(false);
  const [overlayData, setOverlayData] = useState([]);
  // ======================================================

  // ============= SIDEBAR ADDITIONS =================

  const [filteredStocks, setFilteredStocks] = useState([]);

  // ================================================

  // ===== NEW FOR PAGE NAVIGATION =====
  const [showMenu, setShowMenu] = useState(false);
  const [listPage, setListPage] = useState(null);


  // ============= NEW SIDEBAR FUNCTION =============
  const fetchStocksByPrice = async (maxPrice, trend = "up") => {

    if (!maxPrice) return;

    setListLoading(true);

    try {
      const res = await fetch(
        `${API_URL}/stocks-by-price?max=${Number(maxPrice)}&trend=${trend}`
      );

      if (!res.ok) throw new Error("Failed");

      const json = await res.json();

      setFilteredStocks(json.stocks || []);
      setListPage(maxPrice);

    } catch (err) {
      console.log("PRICE FETCH ERROR:", err);
    }

    setListLoading(false);
  };


  const fetchStockData = async () => {
    if (!company || !days) return;

    setLoading(true);
    setError("");
    setData(null);

    try {
      const res = await fetch(
        `${API_URL}/stock?company=${encodeURIComponent(company)}&days=${days}&model=${model}&range=${chartRange}`
      );

      if (!res.ok) throw new Error("Invalid request");

      const json = await res.json();
      setData(json);
    } catch {
      setError("Failed to fetch data");
    }

    setLoading(false);
  };

  // ============= OVERLAY FUNCTION (NEW) =================
  const fetchOverlay = () => {

    if (!data?.hourly_prices) return;

    const lastPoint =
      data.hourly_prices[data.hourly_prices.length - 1];

    const baseDate =
      lastPoint.Datetime || lastPoint.Date;

    let future = [];

    // =================== REAL FIX ===================

    // 1️⃣ IF BACKEND PROVIDED PATH → USE IT
    if (data.forecast_path && data.forecast_path.length > 0) {

      future = data.forecast_path.map((p, i) => ({
        Datetime: new Date(
          new Date(baseDate).getTime() +
          (i + 1) * 24 * 60 * 60 * 1000
        ),
        Close: p.price
      }));

    }

    // 2️⃣ FALLBACK ONLY IF PATH MISSING
    else {

      const startPrice = data.last_close;

      const target =
        data.prediction?.ensemble_price ??
        data.prediction?.expected_price ??
        data.last_close;

      for (let i = 1; i <= days; i++) {

        const progress = i / days;
        const curve = Math.pow(progress, 1.2);

        future.push({
          Datetime: new Date(
            new Date(baseDate).getTime() +
            i * 24 * 60 * 60 * 1000
          ),
          Close: Number(
            (startPrice +
              (target - startPrice) * curve
            ).toFixed(2)
          )
        });
      }
    }

    // =================================================

    const combined = [
      ...data.hourly_prices,
      ...future
    ];

    setOverlayData(combined);
    setShowOverlay(true);
  };




  useEffect(() => {
    console.log("overlayData", overlayData);
  }, [overlayData]);

  useEffect(() => {
    if (company && viewMode !== "full") {
      fetchStockData();
    }
  }, [company]);



  // ======================================================
  // ================== ONLY REAL FIX ==================
  useEffect(() => {
    if (company) {
      fetchStockData();
    }
  }, [chartRange]);
  // ===================================================

  // ========== NEW PAGE SWITCH ==========
  if (listPage) {
    return (
      <StockListPage
        maxPrice={Number(listPage)}
        stocks={filteredStocks}
        loading={listLoading}
        onSelect={(sym, mode, selectedModel) => {
          setCompany(sym);
          setListPage(null);
          setShowMenu(false);


          if (mode === "prediction") {
            setViewMode("prediction");
            setModel(selectedModel || "Linear");
          } else {
            setViewMode("chart");
            setChartRange("120d");
          }
        }}
        onBack={() => setListPage(null)}
      />
    );
  }

  if (showPredictionTable) {
  return (
    <PredictionTablePage
      onBack={() => setShowPredictionTable(false)}
    />
  );
}



  return (
    // ===================== ADDED WRAPPER FOR SIDEBAR =====================
    <div style={{ display: "flex" }}>

      {/* ===== 3 DOT MENU BUTTON ===== */}
      <button
        className="menu-button"
        onClick={() => setShowMenu(!showMenu)}
      >
        ⋮
      </button>

      {/* ===== TOGGLE SIDEBAR ===== */}
      {showMenu && (
        <div style={{
          width: "260px",
          borderRight: "1px solid #ddd",
          padding: "10px",
          background: "#f9fafb"
        }}>

          <h3>Stock Categories</h3>
          {listLoading && (
            <div style={{ padding: "10px", color: "#2563eb" }}>
              Loading stocks...
            </div>
          )}

          <select
            value={trendMode}
            onChange={(e) => setTrendMode(e.target.value)}
            style={{
              marginBottom: "10px",
              padding: "6px",
              width: "100%"
            }}
          >
            <option value="up">🚀 Top 50 Upward</option>
            <option value="down">📉 Top 50 Downward</option>
          </select>


          <button onClick={() => fetchStocksByPrice(50, trendMode)}>

            Stocks under ₹50
          </button>

          <button onClick={() => fetchStocksByPrice(100, trendMode)}>
            Stocks under ₹100
          </button>

          <button onClick={() => fetchStocksByPrice(1000, trendMode)}>
            Stocks under ₹1000
          </button>


        </div>
      )}

      {/* ============== YOUR EXISTING PAGE START =============== */}
      <div className="page" style={{ flex: 1 }}>
        {/* SEARCH CARD */}
        <div className="card" style={{ position: "relative" }}>
          <h1>📈 Stock Predictor</h1>

          {/* ✅ NEW BUTTON */}
          <div style={{
  display: "flex",
  justifyContent: "flex-end",
  gap: "10px",
  marginBottom: "10px"
}}>

  <button
    className="explain-btn"
    onClick={() => setShowExplain(true)}
  >
    Explain
  </button>

  <button
    className="explain-btn"
    onClick={() => setShowPredictionTable(true)}
  >
    Prediction Table
  </button>
</div>


          <p className="subtitle">
            Price prediction using price + volume signals
          </p>

          <div className="search-row">
            <input
              type="text"
              placeholder="Enter stock name (e.g. TCS)"
              value={company}
              onChange={(e) => setCompany(e.target.value)}
            />

            <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
              <option value={1}>1 Days</option>
              <option value={7}>7 Days</option>
              <option value={30}>30 Days</option>
              <option value={90}>90 Days</option>
              <option value={180}>180 Days</option>
              <option value={365}>365 Days</option>
            </select>

            <select value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="Linear">Linear Regression (Price + Volume)</option>
              <option value="EWMA">EWMA (Volume-Weighted)</option>
              <option value="ARIMA">ARIMA</option>
              <option value="ARMA">ARMA</option>
              <option value="ARCH">ARCH / GARCH (Volatility)</option>
            </select>

            <button onClick={fetchStockData}>Search</button>
            {/* ============= NEW BUTTON ============== */}
            <button onClick={() => {
              if (!data) return;
              if (showOverlay) setShowOverlay(false);
              else fetchOverlay();
            }}>
              {showOverlay ? "Hide Overlay" : "Show Prediction Overlay"}
            </button>
            {/* ======================================= */}
          </div>

          {/* ========== MODEL AFTER PREDICTION CLICK ========== */}
          {showPredictionPanel && (
            <div style={{ marginTop: "10px" }}>

              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
              >
                <option value="Linear">Linear Regression</option>
                <option value="EWMA">EWMA</option>
                <option value="ARIMA">ARIMA</option>
                <option value="ARMA">ARMA</option>
                <option value="ARCH">ARCH</option>
              </select>

              <button onClick={fetchStockData}>
                Run Prediction
              </button>

            </div>
          )}
          {/* ================================================== */}


          {loading && <p className="info">Loading...</p>}
          {error && <p className="error">{error}</p>}
        </div>

        {/* RESULT CARD */}
        {data && (
          <div className="card result-card">


            {/* ================= ONLY ADDITION START ================= */}

            <div style={{ display: "flex", gap: "8px", marginBottom: "8px" }}>
              {[
                { label: "1M", value: "30d" },
                { label: "3M", value: "90d" },
                { label: "6M", value: "180d" },
                { label: "1Y", value: "365d" }
              ].map(r => (
                <button
                  key={r.value}
                  onClick={() => setChartRange(r.value)}
                  style={{
                    padding: "4px 10px",
                    borderRadius: "6px",
                    border: "1px solid #2563eb",
                    background: chartRange === r.value ? "#2563eb" : "white",
                    color: chartRange === r.value ? "white" : "#2563eb",
                    cursor: "pointer",
                    fontSize: "12px"
                  }}
                >
                  {r.label}
                </button>
              ))}
            </div>

            <h2>
              {data.company} ({data.symbol})
            </h2>
            {viewMode !== "chart" && (

              <div className="stats">
                <div className="stat-box">
                  <span>Last Close</span>
                  <b>₹{data.last_close}</b>
                </div>

                <div className="stat-box">
                  <span>{data.prediction.model}</span>
                  <b>
                    {data.prediction.expected_price !== undefined
                      ? `₹${data.prediction.expected_price}`
                      : data.prediction.volatility_percent !== undefined
                        ? `${data.prediction.volatility_percent}%`
                        : "N/A"}
                  </b>
                </div>

                {data.prediction.ensemble_price && (
                  <div className="stat-box">
                    <span>Ensemble Price</span>
                    <b>₹{data.prediction.ensemble_price}</b>
                  </div>
                )}

                {data.prediction.stop_loss && (
                  <div className="stat-box">
                    <span>Stop Loss</span>
                    <b>₹{data.prediction.stop_loss}</b>
                  </div>
                )}

                {data.prediction.target && (
                  <div className="stat-box">
                    <span>Target</span>
                    <b>₹{data.prediction.target}</b>
                  </div>
                )}
              </div>

            )}


            <div className="stats">
              <div className="stat-box">
                <span>Last Close</span>
                <b>₹{data.last_close}</b>
              </div>

              <div className="stat-box">
                <span>{data.prediction.model}</span>
                <b>
                  {data.prediction.expected_price !== undefined
                    ? `₹${data.prediction.expected_price}`
                    : data.prediction.volatility_percent !== undefined
                      ? `${data.prediction.volatility_percent}%`
                      : "N/A"}
                </b>
              </div>

              {/* ===== NEW ADDITIONS ===== */}
              {data.prediction.ensemble_price && (
                <div className="stat-box">
                  <span>Ensemble Price</span>
                  <b>₹{data.prediction.ensemble_price}</b>
                </div>
              )}

              {data.prediction.stop_loss && (
                <div className="stat-box">
                  <span>Stop Loss</span>
                  <b>₹{data.prediction.stop_loss}</b>
                </div>
              )}

              {data.prediction.target && (
                <div className="stat-box">
                  <span>Target</span>
                  <b>₹{data.prediction.target}</b>
                </div>
              )}
              {/* ========================= */}
            </div>

            {/* ================= ACCURACY SECTION (COLORED) ================= */}

            {data.accuracy && (
              <div style={{
                marginTop: "12px",
                padding: "12px",
                border: "1px solid #ddd",
                borderRadius: "10px",
                background: "#f9fafb"
              }}>

                <h3>📊 Model Accuracy & Trust</h3>

                {/* ----- COLOR LOGIC ----- */}
                {(() => {
                  const score = data.accuracy.reliability_score || 0;

                  let color =
                    score >= 65 ? "#16a34a" :
                      score >= 45 ? "#ea580c" :
                        "#dc2626";

                  let label =
                    score >= 65 ? "TRUSTABLE" :
                      score >= 45 ? "USE WITH CAUTION" :
                        "NOT RELIABLE";

                  return (
                    <div style={{
                      padding: "8px",
                      marginBottom: "10px",
                      borderRadius: "8px",
                      background: color + "15",
                      border: `1px solid ${color}`
                    }}>

                      <b style={{ color }}>
                        {label}
                      </b>

                      <span style={{ marginLeft: "10px" }}>
                        Score: {score}
                      </span>

                    </div>
                  );
                })()}


                <div style={{
                  display: "flex",
                  gap: "10px",
                  flexWrap: "wrap"
                }}>

                  {/* ---- RELIABILITY ---- */}
                  <div className="stat-box">
                    <span>Reliability Score</span>
                    <b>{data.accuracy.reliability_score ?? "N/A"}</b>
                  </div>


                  {/* ---- TRADING ---- */}
                  {data.accuracy.trading && (
                    <>
                      <div className="stat-box">
                        <span>Win Rate</span>
                        <b>
                          {(data.accuracy.trading.win_rate * 100).toFixed(1)}%
                        </b>
                      </div>

                      <div className="stat-box">
                        <span>Profit Factor</span>
                        <b>{data.accuracy.trading.profit_factor}</b>
                      </div>

                      <div className="stat-box">
                        <span>Max Drawdown</span>
                        <b>{data.accuracy.trading.max_drawdown}</b>
                      </div>
                    </>
                  )}


                  {/* ---- FORECAST ---- */}
                  {data.accuracy.forecast &&
                    data.prediction.model !== "ARCH (Volatility)" && (
                      <>
                        <div className="stat-box">
                          <span>MAPE</span>
                          <b>{data.accuracy.forecast.MAPE}%</b>
                        </div>

                        <div className="stat-box">
                          <span>MAE</span>
                          <b>{data.accuracy.forecast.MAE}</b>
                        </div>
                      </>
                    )}


                  {/* ---- ARCH ---- */}
                  {data.prediction.model === "ARCH (Volatility)" &&
                    data.accuracy.forecast && (
                      <div className="stat-box">
                        <span>Risk Level</span>
                        <b>{data.accuracy.forecast.risk_level}</b>
                      </div>
                    )}

                </div>
              </div>
            )}

            {/* ============================================================= */}


            {/* ✅ NEW COMPARISON SECTION */}
            {data.prediction.comparison && (
              <div style={{ marginTop: "12px" }}>
                <h3>Model Comparison</h3>

                {data.prediction.comparison.map((m, i) => (
                  <div
                    key={i}
                    style={{
                      border: "1px solid #ddd",
                      padding: "8px",
                      marginBottom: "6px",
                      borderRadius: "8px"
                    }}
                  >
                    <b>{m.name}</b>

                    {m.price && (
                      <div>Price: ₹{m.price}</div>
                    )}

                    {m.volatility && (
                      <div>Volatility: {m.volatility}%</div>
                    )}

                    <div>
                      Confidence: <b>{m.confidence}</b>
                    </div>

                    <small>{m.reason}</small>
                  </div>
                ))}
              </div>
            )}

            <h3>Hourly Price Trend</h3>
            <div className="chart-container">

              {/* ================= BUY / SELL SIGNAL ================= */}

              {data && data.signal && (
                <div className="signal-container">

                  <div className="buttons">
                    <button
                      className={
                        data.signal.signal === "BUY"
                          ? "buy active"
                          : "buy"
                      }
                    >
                      BUY
                    </button>

                    <button
                      className={
                        data.signal.signal === "SELL"
                          ? "sell active"
                          : "sell"
                      }
                    >
                      SELL
                    </button>

                    <button
                      className={
                        data.signal.signal === "HOLD"
                          ? "hold active"
                          : "hold"
                      }
                    >
                      HOLD
                    </button>
                  </div>

                  <div className="signal-info">
                    <b>Confidence:</b> {data.signal.confidence}

                    <div className="reasons">
                      Trend: {data.signal.reasons?.trend} |
                      Prediction: {data.signal.reasons?.prediction} |
                      Volume: {data.signal.reasons?.volume}
                    </div>
                  </div>

                </div>
              )}

              <HourlyPriceChart
                prices={data.hourly_prices}
                overlay={showOverlay ? overlayData : null}
              />

            </div>

            <h3>Last 4 Weeks</h3>
            <div className="table-container">
              <Last4WeeksTable data={data.last_4_weeks} />
            </div>
          </div>
        )}

        {/* ✅ EXPLANATION MODAL (NEW) */}
        {showExplain && data && (
          <div className="modal-overlay">
            <div className="modal">
              <h2>How is the predicted value calculated?</h2>

              <p>
                <b>Selected Model:</b> {data.prediction.model}
              </p>

              {/* ===== BACKEND EXPLANATION (PRIMARY) ===== */}
              {data.prediction.explanation && (
                <div style={{ marginBottom: "12px" }}>

                  {data.prediction.explanation.method && (
                    <p>
                      <b>Method:</b> {data.prediction.explanation.method}
                    </p>
                  )}

                  {data.prediction.explanation.concept && (
                    <p>
                      <b>Concept:</b> {data.prediction.explanation.concept}
                    </p>
                  )}

                  {data.prediction.explanation.inputs_used && (
                    <>
                      <b>Inputs Used:</b>
                      <ul>
                        {data.prediction.explanation.inputs_used.map((i, idx) => (
                          <li key={idx}>{i}</li>
                        ))}
                      </ul>
                    </>
                  )}

                  {data.prediction.explanation.coefficients && (
                    <>
                      <b>Model Weights:</b>
                      <ul>
                        <li>
                          Trend impact: {data.prediction.explanation.coefficients.trend_weight}
                        </li>
                        <li>
                          Volume impact: {data.prediction.explanation.coefficients.volume_weight}
                        </li>
                      </ul>
                    </>
                  )}

                  {data.prediction.explanation.interpretation && (
                    <p>
                      <b>Interpretation:</b> {data.prediction.explanation.interpretation}
                    </p>
                  )}

                  {data.prediction.explanation.meaning && (
                    <p>
                      <b>Meaning:</b> {data.prediction.explanation.meaning}
                    </p>
                  )}

                </div>
              )}

              {/* ===== FALLBACK TEXTS (YOUR ORIGINAL LOGIC KEPT) ===== */}

              {data.prediction.model.includes("Linear") && !data.prediction.explanation && (
                <>
                  <p>
                    Linear Regression uses historical prices and trading volume.
                  </p>
                  <ul>
                    <li>Time (day index) captures trend</li>
                    <li>Volume strengthens price signals</li>
                    <li>Model fits a best-fit line to estimate future price</li>
                  </ul>
                </>
              )}

              {data.prediction.model.includes("EWMA") && !data.prediction.explanation && (
                <>
                  <p>
                    EWMA gives more weight to recent prices.
                  </p>
                  <ul>
                    <li>Recent data influences prediction more</li>
                    <li>Volume-weighting improves momentum detection</li>
                  </ul>
                </>
              )}

              {data.prediction.model === "ARIMA" && !data.prediction.explanation && (
                <p>
                  ARIMA predicts future prices using historical price patterns,
                  trends, and differencing.
                </p>
              )}

              {data.prediction.model === "ARMA" && !data.prediction.explanation && (
                <p>
                  ARMA uses past prices and past prediction errors for short-term
                  forecasting.
                </p>
              )}

              {data.prediction.model.includes("ARCH") && !data.prediction.explanation && (
                <p>
                  ARCH/GARCH models market volatility, not price direction.
                  Higher volatility means higher price fluctuation risk.
                </p>
              )}

              <button
                className="close-btn"
                onClick={() => setShowExplain(false)}
              >
                Close
              </button>
            </div>
          </div>

        )}
      </div>
    </div>

  );
}

export default StockPage;
