import { useState, useEffect } from "react";
import HourlyPriceChart from "./HourlyPriceChart";
import Last4WeeksTable from "./Last4WeeksTable";
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

  // ✅ NEW STATE (ONLY ADDITION)
  const [showExplain, setShowExplain] = useState(false);

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


   // ================== ONLY REAL FIX ==================
  useEffect(() => {
    if (company) {
      fetchStockData();
    }
  }, [chartRange]);
  // ===================================================

  return (
    <div className="page">
      {/* SEARCH CARD */}
      <div className="card" style={{ position: "relative" }}>
        <h1>📈 Stock Predictor</h1>

        {/* ✅ NEW BUTTON */}
        <button
          className="explain-btn"
          onClick={() => setShowExplain(true)}
        >
          Explain the predicted value
        </button>

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

          <select value={days} onChange={(e) => setDays(e.target.value)}>
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
        </div>

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
          </div>

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
            <HourlyPriceChart prices={data.hourly_prices} />
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
  );
}

export default StockPage;
