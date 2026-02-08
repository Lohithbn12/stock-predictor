import { useEffect, useState } from "react";

const API_URL = "https://stock-predictor-0zst.onrender.com";

function StockListPage({ maxPrice, onSelect, onBack }) {

  const [stocks, setStocks] = useState([]);

  useEffect(() => {
    const load = async () => {
      const res = await fetch(`${API_URL}/stocks-by-price?max=${maxPrice}`);
      const json = await res.json();
      setStocks(json.stocks || []);
    };

    load();
  }, [maxPrice]);


  return (
    <div className="page">

      <div className="card">

        <button onClick={onBack}>← Back</button>

        <h2>Stocks under ₹{maxPrice}</h2>

        {stocks.map((s, i) => (
          <div
            key={i}
            style={{
              padding: "8px",
              borderBottom: "1px solid #eee",
              cursor: "pointer"
            }}
            onClick={() => onSelect(s.symbol)}
          >
            {s.symbol} – ₹{s.price}
          </div>
        ))}

      </div>
    </div>
  );
}

export default StockListPage;
