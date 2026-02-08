import { BrowserRouter, Routes, Route } from "react-router-dom";

import StockPage from "./StockPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>

        {/* MAIN PAGE – HANDLES LIST INTERNALLY */}
        <Route path="/" element={<StockPage />} />

      </Routes>
    </BrowserRouter>
  );
}

export default App;
