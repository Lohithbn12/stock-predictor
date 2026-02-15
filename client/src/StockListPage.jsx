import { useEffect, useState } from "react";

function StockListPage({ maxPrice, stocks = [], loading = false, onSelect, onBack }) {

  const [visibleStocks, setVisibleStocks] = useState([]);

  // Progressive loading (smooth appearing one by one)
  useEffect(() => {
    setVisibleStocks([]);

    if (!stocks || stocks.length === 0) return;

    let index = 0;

    const interval = setInterval(() => {
      setVisibleStocks(prev => {
        if (index >= stocks.length) {
          clearInterval(interval);
          return prev;
        }
        const next = stocks[index];
        index++;
        return [...prev, next];
      });
    }, 30); // speed of appearance

    return () => clearInterval(interval);
  }, [stocks]);

  return (
    <div style={{ padding: "20px" }}>

      <button
        onClick={onBack}
        style={{
          marginBottom: "15px",
          padding: "6px 12px",
          cursor: "pointer"
        }}
      >
        ← Back
      </button>

      <h2>Stocks under ₹{maxPrice}</h2>

      {loading && (
        <div style={{ marginTop: "20px", color: "#2563eb" }}>
          Loading stocks...
        </div>
      )}

      {!loading && visibleStocks.length === 0 && (
        <div style={{ marginTop: "15px" }}>
          No stocks found in this range
        </div>
      )}

      {!loading && visibleStocks.length > 0 && (
        <div style={{ marginTop: "15px" }}>
          {visibleStocks.map((s, index) => (
            <div
              key={s?.symbol || index}
              onClick={() => s?.symbol && onSelect(s.symbol)}
              style={{
                padding: "10px",
                borderBottom: "1px solid #eee",
                cursor: "pointer",
                transition: "background 0.2s ease"
              }}
              onMouseEnter={(e) => e.currentTarget.style.background = "#f5f5f5"}
              onMouseLeave={(e) => e.currentTarget.style.background = "white"}
            >
              <div>
                <strong>{s?.symbol}</strong> — ₹{s?.price}
              </div>

              {s?.company && (
                <div style={{
                  fontSize: "12px",
                  color: "#666",
                  marginTop: "2px"
                }}>
                  {s.company}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

    </div>
  );
}

export default StockListPage;
