import { useEffect, useState } from "react";

const API_URL = "https://stock-predictor-0zst.onrender.com";

function StockListPage({ maxPrice, onSelect, onBack }) {

  const [stocks, setStocks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {

    const load = async () => {

      if (!maxPrice) return;

      setLoading(true);
      setError("");

      try {

        const res = await fetch(
          `${API_URL}/stocks-by-price?max=${Number(maxPrice)}`
        );

        if (!res.ok) throw new Error("API Failed");

        const json = await res.json();

        setStocks(json.stocks || []);

      } catch (e) {
        console.log(e);
        setError("Failed to load stocks");
      }

      setLoading(false);
    };

    load();

  }, [maxPrice]);


  return (
    <div style={{ padding: "20px" }}>

      <button onClick={onBack}>← Back</button>

      <h2>Stocks under ₹{maxPrice}</h2>

      {loading && (
        <div style={{ marginTop: "20px" }}>
          Loading stocks...
        </div>
      )}

      {error && <div>{error}</div>}

      {!loading && stocks.length === 0 && (
        <div>No stocks found in this range</div>
      )}

      {!loading && (
        <div>
          {stocks.map((s, i) => (
            <div
              key={i}
              onClick={() => onSelect(s.symbol)}
              style={{
                padding: "8px",
                borderBottom: "1px solid #eee",
                cursor: "pointer"
              }}
            >
              {s.symbol} – ₹{s.price}
            </div>
          ))}
        </div>
      )}

    </div>
  );
}

export default StockListPage;
