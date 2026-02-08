import { BrowserRouter, Routes, Route } from "react-router-dom";

import StockPage from "./StockPage";
import StocksListPage from "./StockListPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>

        {/* MAIN PAGE */}
        <Route path="/" element={<StockPage />} />

        {/* NEW LIST PAGE */}
        <Route path="/stocks" element={<StocksListPage />} />

      </Routes>
    </BrowserRouter>
  );
}

export default App;
