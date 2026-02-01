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

function HourlyPriceChart({ prices }) {
  if (!prices || prices.length === 0) return null;

  const first = prices[0].Close;
  const last = prices[prices.length - 1].Close;

  const isUp = last >= first;
  const lineColor = isUp ? "#16a34a" : "#dc2626"; // green / red

  const data = {
    datasets: [
      {
        label: "Hourly Close Price",
        data: prices.map(p => ({
          x: new Date(p.Datetime),
          y: p.Close
        })),
        borderColor: lineColor,
        backgroundColor: lineColor,
        borderWidth: 2,
        tension: 0.3,
        pointRadius: 0
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false, // ✅ critical for mobile
    plugins: {
      legend: { display: false },
      tooltip: {
        mode: "index",
        intersect: false
      }
    },
    scales: {
      x: {
        type: "time",
        time: { unit: "day" },
        ticks: {
          maxTicksLimit: 6,
          autoSkip: true
        },
        grid: {
          display: false
        }
      },
      y: {
        ticks: {
          callback: value => `₹${value}`,
          maxTicksLimit: 5
        },
        grid: {
          color: "rgba(0,0,0,0.05)"
        }
      }
    }
  };

  return (
    <div className="chart-container">
      <Line data={data} options={options} />
    </div>
  );
}

export default HourlyPriceChart;
