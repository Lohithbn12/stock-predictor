import { useState } from "react";
import HourlyPriceChart from "./HourlyPriceChart";
import Last4WeeksTable from "./Last4WeeksTable";
import "./App.css";

const API_URL = "https://stock-predictor-0zst.onrender.com";

function StockPage() {
  const [company, setCompany] = useState("");
  const [days, setDays] = useState(30);
  const [model, setModel] = useState("Linear");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchStockData = async () => {
    if (!company || !days) return;

    setLoading(true);
    setError("");
    setData(null);

    try {
      const res = await fetch(
        `${API_URL}/stock?company=${encodeURIComponent(company)}&days=${days}&model=${model}`
      );

      if (!res.ok) throw new Error("Invalid request");

      const json = await res.json();
      setData(json);
    } catch {
      setError("Failed to fetch data");
    }

    setLoading(false);
  };

  return (
    <div className="page">
      {/* SEARCH CARD */}
      <div className="card">
        <h1>📈 Stock Predictor</h1>
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
            <option value={7}>7 Days</option>
            <option value={30}>30 Days</option>
            <option value={90}>90 Days</option>
            <option value={180}>180 Days</option>
            <option value={365}>365 Days</option>
          </select>

          {/* MODEL DROPDOWN */}
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
          <h2>
            {data.company} ({data.symbol})
          </h2>

          {/* STATS */}
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

          {/* CHART */}
          <h3>Hourly Price Trend</h3>
          <div className="chart-container">
            <HourlyPriceChart prices={data.hourly_prices} />
          </div>

          {/* TABLE */}
          <h3>Last 4 Weeks</h3>
          <div className="table-container">
            <Last4WeeksTable data={data.last_4_weeks} />
          </div>
        </div>
      )}
    </div>
  );
}

export default StockPage;
