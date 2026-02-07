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
          {data.map((row, i) => {

            // ✅ SAFETY HANDLING
            const open = Number(row.Open || 0);
            const close = Number(row.Close || 0);

            // ✅ NICE DATE FORMAT
            const date = row.Date
              ? new Date(row.Date).toLocaleDateString("en-IN")
              : "-";

            return (
              <tr key={i}>
                <td>{date}</td>

                <td>₹{open.toFixed(2)}</td>

                <td
                  style={{
                    color: close >= open ? "#16a34a" : "#dc2626",
                    fontWeight: 600
                  }}
                >
                  ₹{close.toFixed(2)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default Last4WeeksTable;
