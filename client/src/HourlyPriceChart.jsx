import {
  Chart as ChartJS,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  TimeScale
} from "chart.js";
import { Line } from "react-chartjs-2";
import "chartjs-adapter-date-fns";

ChartJS.register(
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
  TimeScale
);

function HourlyPriceChart({ prices, overlay }) {
  if (!prices || prices.length === 0) return null;

  // ================= EXISTING LOGIC (KEPT) =================
  const first = prices[0].Close;
  const last = prices[prices.length - 1].Close;

  const isUp = last >= first;
  const lineColor = isUp ? "#16a34a" : "#dc2626";
  // =========================================================


  // ============== NEW SMART TIME DETECTION ================
  const isHourly =
    prices.length > 1 &&
    Math.abs(
      new Date(prices[1]?.Datetime).getTime() -
      new Date(prices[0]?.Datetime).getTime()
    ) < 3 * 60 * 60 * 1000;
  // ========================================================


  // ================= EXISTING DATASET =====================
  const datasets = [
    {
      label: "Hourly Close Price",
      data: prices.map(p => ({
        x: new Date(p.Datetime),
        y: p.Close
      })),
      borderColor: lineColor,
      borderWidth: 2,
      tension: 0.3,
      pointRadius: 0
    }
  ];
  // ========================================================


  // =============== UPDATED OVERLAY LOGIC ==================
  if (overlay && overlay.length > 0) {

    const actualEnd = prices.length;

    datasets.push({
      label: "Prediction Overlay",

      data: overlay.map((p, i) => ({
        x: new Date(p.Datetime),
        y: p.Close,

        // tag for tooltip
        type: i >= actualEnd ? "forecast" : "actual"
      })),

      borderColor: "#2563eb",
      borderDash: [5, 5],
      borderWidth: 2,
      tension: 0.25,
      pointRadius: 0
    });
  }
  // ========================================================


  const data = { datasets };


  // =============== UPDATED OPTIONS ========================
  const options = {
    responsive: true,
    maintainAspectRatio: false,

    plugins: {
      legend: {
        display: true
      },

      tooltip: {
        mode: "index",
        intersect: false,

        callbacks: {
          label: function (ctx) {

            const prefix =
              ctx.raw?.type === "forecast"
                ? "Forecast"
                : "Actual";

            return `${prefix}: ₹${ctx.parsed.y}`;
          }
        }
      }
    },

    scales: {
      x: {
        type: "time",

        time: {
          unit: isHourly ? "hour" : "day",

          displayFormats: {
            hour: "dd MMM HH:mm",
            day: "dd MMM"
          }
        }
      },

      y: {
        ticks: {
          callback: value => `₹${value}`
        }
      }
    }
  };
  // ========================================================


  return (
    <div className="chart-container" style={{ height: "400px" }}>
      <Line data={data} options={options} />
    </div>
  );
}

export default HourlyPriceChart;
