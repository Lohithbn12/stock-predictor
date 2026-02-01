import { useState } from "react";
import HourlyPriceChart from "./HourlyPriceChart";
import Last4WeeksTable from "./Last4WeeksTable";

function StockPage() {
  const [company, setCompany] = useState("");
  const [predictionDays, setPredictionDays] = useState(30);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchStockData = async () => {
    if (!company || !predictionDays) return;

    setLoading(true);
    setError("");
    setData(null);

    try {
      const res = await fetch(
        `https://stock-predictor-0zst.onrender.com/stock?company=${encodeURIComponent(
          company
        )}&days=${predictionDays}`
      );

      const json = await res.json();

      if (json.error) {
        setError(json.error);
      } else {
        setData(json);
      }
    } catch {
      setError("Failed to fetch data");
    }

    setLoading(false);
  };

  return (
    <div style={{ padding: "30px", fontFamily: "Arial" }}>
      <h1>📈 Stock Predictor</h1>

      {/* Prediction Input */}
      <div style={{ marginBottom: "10px" }}>
        <input
          type="number"
          min="1"
          max="90"
          value={predictionDays}
          onChange={(e) => setPredictionDays(e.target.value)}
          style={{ padding: "8px", width: "120px", marginRight: "10px" }}
        />
        <select disabled style={{ padding: "8px" }}>
          <option>Days</option>
        </select>
      </div>

      {/* Stock Search */}
      <input
        type="text"
        placeholder="Enter company name (e.g. TCS)"
        value={company}
        onChange={(e) => setCompany(e.target.value)}
        style={{ padding: "8px", width: "300px" }}
      />

      <button
        onClick={fetchStockData}
        style={{ marginLeft: "10px", padding: "8px 15px" }}
      >
        Search
      </button>

      {loading && <p>Loading...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {data && (
        <>
          <hr />

          <h2>
            {data.company} ({data.symbol})
          </h2>

          <p><b>Last Close:</b> ₹{data.last_close}</p>
          <p>
            <b>Linear Regression ({data.prediction_days} days):</b> ₹
            {data.linear_regression_prediction}
          </p>
          <p>
            <b>GARCH Volatility ({data.prediction_days} days):</b>{" "}
            {data.garch_volatility_percent}%
          </p>

          <h3>Hourly Price Trend</h3>
          <HourlyPriceChart prices={data.hourly_prices} />

          <h3>Last 4 Weeks</h3>
          <Last4WeeksTable rows={data.last_4_weeks} />
        </>
      )}
    </div>
  );
}

export default StockPage;
