function Last4WeeksTable({ data }) {
  if (!data || data.length === 0) return null;

  return (
    <table
      border="1"
      cellPadding="8"
      style={{ marginTop: "15px", borderCollapse: "collapse" }}
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
                color: row.Close >= row.Open ? "green" : "red"
              }}
            >
              ₹{row.Close.toFixed(2)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default Last4WeeksTable;
