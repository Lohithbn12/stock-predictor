import { useEffect, useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";

const API_URL = "https://stock-predictor-0zst.onrender.com";

function StocksListPage() {

  const [params] = useSearchParams();
  const navigate = useNavigate();

  const max = params.get("max");

  const [stocks, setStocks] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {

    const load = async () => {

      setLoading(true);

      try {
        const res = await fetch(`${API_URL}/stocks-by-price?max=${max}`);
        const json = await res.json();

        setStocks(json.stocks || []);

      } catch (e) {
        console.log(e);
      }

      setLoading(false);
    };

    load();

  }, [max]);


  const openStock = (symbol) => {
    navigate(`/?symbol=${symbol}`);
  };


  return (
    <div style={{ padding: "20px" }}>

      <h2>Stocks under ₹{max}</h2>

      {/* ============ SPINNER ============ */}
      {loading && (
        <div style={{ textAlign: "center", marginTop: "40px" }}>

          <div className="spinner" />

          <p>Loading stocks... please wait</p>

        </div>
      )}
      {/* ================================= */}


      {!loading && (
        <div>
          {stocks.map((s, i) => (
            <div
              key={i}
              onClick={() => openStock(s.symbol)}
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

export default StocksListPage;
