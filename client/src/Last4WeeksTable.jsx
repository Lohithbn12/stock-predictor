function Last4WeeksTable({ data }) {
  if (!data || data.length === 0) return null;

  return (
    <div className="table-container">
      <table
        border="1"
        cellPadding="8"
        style={{
          width: "100%",
          marginTop: "15px",
          borderCollapse: "collapse"
        }}
      >
        <thead>
          <tr>
            <th>Date</th>
            <th>Open</th>
            <th>Close</th>
          </tr>
        </thead>

        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              <td>{row.Date}</td>
              <td>₹{row.Open.toFixed(2)}</td>
              <td
                style={{
                  color: row.Close >= row.Open ? "#16a34a" : "#dc2626",
                  fontWeight: 600
                }}
              >
                ₹{row.Close.toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default Last4WeeksTable;
