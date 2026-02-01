import { useState } from "react";
import HourlyPriceChart from "./HourlyPriceChart";
import Last4WeeksTable from "./Last4WeeksTable";
import "./App.css";

const API_URL = "https://stock-predictor-0zst.onrender.com";

function StockPage() {
  const [company, setCompany] = useState("");
  const [days, setDays] = useState(30);
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
        `${API_URL}/stock?company=${encodeURIComponent(company)}&days=${days}`
      );

      if (!res.ok) {
        throw new Error("Invalid request");
      }

      const json = await res.json();
      setData(json);
    } catch {
      setError("Failed to fetch data");
    }

    setLoading(false);
  };

  return (
    <div className="page">
      {/* Search Card */}
      <div className="card">
        <h1>📈 Stock Predictor</h1>
        <p className="subtitle">
          Smart stock prediction using Linear Regression & GARCH
        </p>

        <div className="search-row">
          <input
            type="text"
            placeholder="Enter stock name (e.g. Wipro)"
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

          <button onClick={fetchStockData}>Search</button>
        </div>

        {loading && <p className="info">Loading...</p>}
        {error && <p className="error">{error}</p>}
      </div>

      {/* Result Card */}
      {data && (
        <div className="card result-card">
          <h2>
            {data.company} ({data.symbol})
          </h2>

          <div className="stats">
            <div>
              <span>Last Close</span>
              <b>₹{data.last_close ?? "N/A"}</b>
            </div>

            <div>
              <span>Linear Prediction</span>
              <b>
                ₹{data.linear_regression_prediction?.expected_price ?? "N/A"}
              </b>
            </div>

            <div>
              <span>GARCH Volatility</span>
              <b>
                {data.garch_prediction?.volatility_30d_percent
                  ? `${data.garch_prediction.volatility_30d_percent}%`
                  : "N/A"}
              </b>
            </div>
          </div>


          <h3>Hourly Price Trend</h3>
          <HourlyPriceChart prices={data.hourly_prices} />

          <h3>Last 4 Weeks</h3>
          <Last4WeeksTable data={data.last_4_weeks} />
        </div>
      )}
    </div>
  );
}

export default StockPage;
