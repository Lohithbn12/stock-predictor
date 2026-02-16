import { useEffect, useState } from "react";

function StockListPage({ maxPrice, stocks = [], loading = false, onSelect, onBack }) {

  const [visibleStocks, setVisibleStocks] = useState([]);

  // 🔹 NEW STATES
  const [selectedStock, setSelectedStock] = useState(null);
  const [showDialog, setShowDialog] = useState(false);
  const [step, setStep] = useState("main"); // main | model
  const [selectedModel, setSelectedModel] = useState("Linear");

  // Progressive loading
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
    }, 30);

    return () => clearInterval(interval);
  }, [stocks]);

  // 🔹 HANDLE STOCK CLICK
  const handleStockClick = (symbol) => {
    setSelectedStock(symbol);
    setShowDialog(true);
    setStep("main");
  };

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
              onClick={() => s?.symbol && handleStockClick(s.symbol)}
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
            </div>
          ))}
        </div>
      )}

      {/* ===================== DIALOG ===================== */}
      {showDialog && (
        <div style={{
          position: "fixed",
          inset: 0,
          background: "rgba(0,0,0,0.4)",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          zIndex: 1000
        }}>
          <div style={{
            background: "white",
            padding: "20px",
            borderRadius: "12px",
            width: "300px",
            textAlign: "center"
          }}>

            <h3>{selectedStock}</h3>

            {/* STEP 1 */}
            {step === "main" && (
              <>
                <button
                  style={{ margin: "8px", padding: "8px 12px" }}
                  onClick={() => setStep("model")}
                >
                  Prediction
                </button>

                <button
                  style={{ margin: "8px", padding: "8px 12px" }}
                  onClick={() => {
                    onSelect(selectedStock, "chart", null);
                    setShowDialog(false);
                  }}
                >
                  Chart of Last 4 Months
                </button>
              </>
            )}

            {/* STEP 2 – MODEL SELECTION */}
            {step === "model" && (
              <>
                <p>Select Model</p>

                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  style={{ marginBottom: "10px", padding: "6px" }}
                >
                  <option value="Linear">Linear</option>
                  <option value="EWMA">EWMA</option>
                  <option value="ARIMA">ARIMA</option>
                  <option value="ARMA">ARMA</option>
                  <option value="ARCH">ARCH</option>
                </select>

                <div>
                  <button
                    style={{ margin: "6px", padding: "6px 10px" }}
                    onClick={() => {
                      onSelect(selectedStock, "prediction", selectedModel);
                      setShowDialog(false);
                    }}
                  >
                    Run Prediction
                  </button>

                  <button
                    style={{ margin: "6px", padding: "6px 10px" }}
                    onClick={() => setStep("main")}
                  >
                    Back
                  </button>
                </div>
              </>
            )}

            <div style={{ marginTop: "10px" }}>
              <button
                style={{ fontSize: "12px", color: "#dc2626" }}
                onClick={() => setShowDialog(false)}
              >
                Cancel
              </button>
            </div>

          </div>
        </div>
      )}

    </div>
  );
}

export default StockListPage;
