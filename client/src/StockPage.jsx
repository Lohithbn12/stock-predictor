import { useState } from "react";
import HourlyPriceChart from "./HourlyPriceChart";
import Last4WeeksTable from "./Last4WeeksTable";

function StockPage() {
  const [company, setCompany] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [view, setView] = useState("hourly"); // hourly | 4weeks

  const fetchStockData = async () => {
    if (!company) return;

    setLoading(true);
    setError("");
    setData(null);

    try {
      const res = await fetch(
  `https://stock-predictor-0zst.onrender.com/stock?company=${encodeURIComponent(company)}`
);

      const json = await res.json();

      if (json.error) {
        setError(json.error);
      } else {
        setData(json);
        setView("hourly");
      }
    } catch {
      setError("Failed to fetch data");
    }

    setLoading(false);
  };

  return (
    <div style={{ padding: "30px", fontFamily: "Arial" }}>
      <h1>📈 Stock Predictor</h1>

      {/* Search */}
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
          <h3>Predictions</h3>

<p>
  <b>Linear Regression (Trend – 1 Month):</b>{" "}
  ₹{data.linear_regression_prediction.expected_price_1_month}
</p>

<p>
  <b>GARCH Volatility (30 Days):</b>{" "}
  {data.garch_prediction.volatility_30d_percent}%
</p>

<p>
  <b>Expected Price Range (GARCH):</b>{" "}
  ₹{data.garch_prediction.price_range.lower} – ₹{data.garch_prediction.price_range.upper}
</p>

          {/* Toggle buttons */}
          <div style={{ marginTop: "20px" }}>
            <button onClick={() => setView("hourly")}>
              Hourly (30 Days)
            </button>

            <button
              onClick={() => setView("4weeks")}
              style={{ marginLeft: "10px" }}
            >
              Last 4 Weeks
            </button>
          </div>

          {/* Views */}
          {view === "hourly" && (
            <>
              <h3 style={{ marginTop: "20px" }}>
                Hourly Price Trend (Last 30 Days)
              </h3>
              <HourlyPriceChart prices={data.hourly_prices} />
            </>
          )}

          {view === "4weeks" && (
            <>
              <h3 style={{ marginTop: "20px" }}>
                Last 4 Weeks (Daily)
              </h3>
              <Last4WeeksTable data={data.last_4_weeks} />
            </>
          )}
        </>
      )}
    </div>
  );
}

export default StockPage;
