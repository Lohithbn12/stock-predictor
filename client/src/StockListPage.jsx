import { useEffect, useState } from "react";

const API_URL = "https://stock-predictor-0zst.onrender.com";

function StocksListPage({
  maxPrice,
  stocks = [],
  loading = false,
  onSelect,
  onBack
}) {

  return (
    <div style={{ padding: "20px" }}>

      {/* BACK BUTTON */}
      <button
        onClick={onBack}
        style={{
          marginBottom: "10px",
          padding: "6px 12px",
          cursor: "pointer"
        }}
      >
        ← Back
      </button>

      <h2>Stocks under ₹{maxPrice}</h2>

      {/* ============ SPINNER ============ */}
      {loading && (
        <div style={{ textAlign: "center", marginTop: "40px" }}>

          <div className="spinner" />

          <p>Loading stocks... please wait</p>

        </div>
      )}
      {/* ================================= */}


      {!loading && stocks.length === 0 && (
        <p>No stocks found in this range</p>
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

export default StocksListPage;
