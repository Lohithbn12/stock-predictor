import { useEffect, useState } from "react";

const API_URL = "https://stock-predictor-0zst.onrender.com";

function PredictionTablePage({ onBack }) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchTable = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/prediction-table`);
      const json = await res.json();
      setData(json.predictions || []);
    } catch (err) {
      console.log("Prediction table error:", err);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchTable();
  }, []);

  return (
    <div style={{ padding: "20px" }}>

      <button
        onClick={onBack}
        style={{ marginBottom: "15px", padding: "6px 12px" }}
      >
        ← Back
      </button>

      <h2>📊 Prediction Tracking Table</h2>

      {loading && <p>Loading...</p>}

      {!loading && data.length === 0 && (
        <p>No predictions logged yet.</p>
      )}

      {!loading && data.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              marginTop: "15px"
            }}
          >
            <thead>
              <tr>
                <th>Date</th>
                <th>Symbol</th>
                <th>Horizon</th>
                <th>Predicted</th>
                <th>Actual</th>
                <th>Error %</th>
                <th>Direction</th>
                <th>Status</th>
              </tr>
            </thead>

            <tbody>
              {data.map((row, i) => (
                <tr key={i}>
                  <td>{row.date}</td>
                  <td>{row.symbol}</td>
                  <td>{row.horizon}D</td>
                  <td>₹{row.predicted}</td>
                  <td>{row.actual ? `₹${row.actual}` : "-"}</td>
                  <td>
                    {row.error_percent
                      ? `${row.error_percent}%`
                      : "-"}
                  </td>

                  <td
                    style={{
                      color:
                        row.direction === "Correct"
                          ? "#16a34a"
                          : row.direction === "Wrong"
                          ? "#dc2626"
                          : "#999",
                      fontWeight: "bold"
                    }}
                  >
                    {row.direction || "-"}
                  </td>

                  <td
                    style={{
                      color:
                        row.status === "Completed"
                          ? "#16a34a"
                          : row.status === "Pending"
                          ? "#ea580c"
                          : "#999"
                    }}
                  >
                    {row.status}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default PredictionTablePage;
