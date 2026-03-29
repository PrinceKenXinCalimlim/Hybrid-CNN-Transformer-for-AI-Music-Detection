import { BrowserRouter, Routes, Route, Navigate } from "react-router";
import Design2Upload from "./pages/design2/Design2Upload";
import Design2Results from "./pages/design2/Design2Results";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/design2" replace />} />
        <Route path="/design2" element={<Design2Upload />} />
        <Route path="/design2/results" element={<Design2Results />} />
      </Routes>
    </BrowserRouter>
  );
}