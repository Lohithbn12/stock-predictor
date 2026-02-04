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

  const first = prices[0].Close;
  const last = prices[prices.length - 1].Close;

  const isUp = last >= first;
  const lineColor = isUp ? "#16a34a" : "#dc2626";

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

  // =============== ONLY ADDITION ===============
  if (overlay && overlay.length > 0) {
    datasets.push({
      label: "Prediction Overlay",
      data: overlay.map(p => ({
        x: new Date(p.Datetime),
        y: p.Close
      })),
      borderColor: "#2563eb",
      borderDash: [5, 5],
      borderWidth: 2,
      tension: 0.3,
      pointRadius: 0
    });
  }
  // ============================================

  const data = { datasets };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true },
      tooltip: { mode: "index", intersect: false }
    },
    scales: {
      x: {
        type: "time",
        time: { unit: "day" }
      },
      y: {
        ticks: {
          callback: value => `₹${value}`
        }
      }
    }
  };

  return (
  <div className="chart-container" style={{ height: "400px" }}>
    <Line data={data} options={options} />
  </div>
);

}

export default HourlyPriceChart;
